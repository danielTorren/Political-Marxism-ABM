"""Sobol' analysis and figures for a sweep written by :mod:`sensitivity_gen`.

    uv run python src/sensitivity/sensitivity_analysis.py                  # newest run
    uv run python src/sensitivity/sensitivity_analysis.py --run-dir Results/sensitivity/2026-08-04_143012

Called automatically at the end of a sweep, and standalone afterwards -- which is the normal case,
because the sweep is the expensive half and the decomposition is seconds. It also reads a
*partial* sweep: if a run was killed, point it at the run directory and it analyses the runs that
completed, reporting how many samples were incomplete.

What it writes, into the run's ``output_data/`` and ``figures/``:

    sobol_indices.csv     S1, ST, their confidence intervals and ST - S1, per output
    sobol_convergence.csv ST recomputed on 1/8, 1/4, 1/2 and all of the design
    sobol_noise.csv       how much of the decomposed variance is stochastic noise, not parameters

Three things have to be read before any index is believed:

* **the confidence intervals**, because a small index and no index look identical at low n_base;
* **the convergence figure**, because a ranking that is still moving has not converged;
* **the noise figure**, because Sobol' decomposes whatever variance it is given, and a stochastic
  model supplies variance that belongs to no parameter. Where the noise share is large the indices
  are measuring the model's own randomness and the answer is more replicates, not more samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pmabm.plots import (  # noqa: E402
    GRID,
    INK,
    INK_SOFT,
    SEQUENTIAL,
    SERIES,
    _finish,
    _style,
)

DEFAULT_ROOT = Path("Results/sensitivity")


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def latest_run(root: Path = DEFAULT_ROOT) -> Path:
    """Newest run directory under ``root`` that actually holds a sweep."""
    if (root / "output_data" / "Y.csv").is_file():
        return root
    candidates = [d for d in root.glob("*") if (d / "output_data").is_dir()]
    if not candidates:
        raise SystemExit(f"No sensitivity runs found under {root}/")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def load_run(run_dir: Path) -> dict:
    """Read a run directory back into ``{problem, runs, samples, y}``.

    ``Y.csv`` is preferred but not required: a sweep killed before it wrote one still has the
    incremental ``runs_stream.csv``, which is averaged here instead.
    """
    problem_path = run_dir / "input_data" / "problem.json"
    if not problem_path.is_file():
        raise SystemExit(f"{problem_path} not found -- is {run_dir} a sensitivity run?")
    record = json.loads(problem_path.read_text(encoding="utf-8"))
    problem = {
        "num_vars": record["num_vars"],
        "names": record["names"],
        "bounds": record["bounds"],
    }

    outdir = run_dir / "output_data"
    runs_path = next(
        (p for p in (outdir / "runs_by_sample.csv", outdir / "runs_stream.csv") if p.is_file()),
        None,
    )
    if runs_path is None:
        raise SystemExit(f"No runs table in {outdir}/")
    runs = pd.read_csv(runs_path).sort_values(["sample", "seed"]).reset_index(drop=True)

    outputs = [o for o in record["outputs"] if o in runs.columns]
    n_samples = record["n_base"] * (
        2 * record["num_vars"] + 2 if record.get("calc_second_order") else record["num_vars"] + 2
    )
    y_path = outdir / "Y.csv"
    if y_path.is_file():
        y = pd.read_csv(y_path)
    else:
        y = (
            runs.groupby("sample")[outputs]
            .mean()
            .reindex(range(n_samples))
            .rename_axis("sample")
            .reset_index()
        )

    samples_path = outdir / "samples.csv"
    samples = pd.read_csv(samples_path) if samples_path.is_file() else None

    return {
        "record": record,
        "problem": problem,
        "runs": runs,
        "samples": samples,
        "y": y,
        "outputs": outputs,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------------------------
def _prepare_column(y: pd.DataFrame, output: str, n_samples: int) -> tuple[np.ndarray, int]:
    """One output as a complete float vector in design order, plus how many gaps were filled.

    Sobol' reads ``Y`` positionally against the sample matrix and cannot take a gap, so samples
    with no usable run are filled with the column mean. That is the least-bad option -- it adds no
    variance and so cannot manufacture an effect -- but it does dilute the indices of whichever
    parameter drove the failure, which is why the count is reported rather than absorbed.
    """
    series = y.set_index("sample")[output].reindex(range(n_samples)).astype(float)
    gaps = int(series.isna().sum())
    if gaps == n_samples:
        raise ValueError(f"every sample is missing {output}")
    return series.fillna(series.mean()).to_numpy(), gaps


def sobol_indices(
    problem: dict,
    y: pd.DataFrame,
    outputs: list[str],
    n_samples: int,
    calc_second_order: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """First-order and total-order indices for every output, as one tidy frame."""
    rows, gaps_by_output = [], {}
    for output in outputs:
        try:
            values, gaps = _prepare_column(y, output, n_samples)
        except ValueError as exc:
            print(f"  ! skipped {output}: {exc}", file=sys.stderr)
            continue
        gaps_by_output[output] = gaps
        if np.allclose(values, values[0]):
            print(f"  ! skipped {output}: constant across every sample", file=sys.stderr)
            continue
        result = sobol_analyze.analyze(
            problem, values, calc_second_order=calc_second_order, print_to_console=False
        )
        for i, name in enumerate(problem["names"]):
            s1, st = float(result["S1"][i]), float(result["ST"][i])
            rows.append(
                {
                    "output": output,
                    "parameter": name,
                    "S1": s1,
                    "S1_conf": float(result["S1_conf"][i]),
                    "ST": st,
                    "ST_conf": float(result["ST_conf"][i]),
                    # What the parameter does only in company. Clipped at zero because the
                    # estimator can put ST marginally below S1 on a purely additive effect.
                    "interaction": max(st - s1, 0.0),
                    # A total-order index below its own confidence interval is indistinguishable
                    # from no effect at this sample size.
                    "ST_significant": st > float(result["ST_conf"][i]),
                    "imputed_samples": gaps,
                }
            )
    return pd.DataFrame(rows), gaps_by_output


def convergence(
    problem: dict,
    y: pd.DataFrame,
    outputs: list[str],
    n_base: int,
    calc_second_order: bool = False,
) -> pd.DataFrame:
    """Recompute ST on nested prefixes of the design, to show whether ``n_base`` was enough.

    The Sobol' matrix is laid out in blocks of ``D + 2`` rows per base sample, so the first
    ``n * (D + 2)`` rows are themselves a valid design of size ``n``. A parameter whose ST is
    still drifting at the largest prefix has not converged, whatever its confidence interval says.
    """
    block = 2 * problem["num_vars"] + 2 if calc_second_order else problem["num_vars"] + 2
    levels = [n for n in (n_base // 8, n_base // 4, n_base // 2, n_base) if n >= 8]
    rows = []
    for output in outputs:
        try:
            values, _ = _prepare_column(y, output, n_base * block)
        except ValueError:
            continue
        if np.allclose(values, values[0]):
            continue
        for n in levels:
            result = sobol_analyze.analyze(
                problem, values[: n * block],
                calc_second_order=calc_second_order, print_to_console=False,
            )
            for i, name in enumerate(problem["names"]):
                rows.append(
                    {
                        "output": output,
                        "parameter": name,
                        "n_base": n,
                        "ST": float(result["ST"][i]),
                        "ST_conf": float(result["ST_conf"][i]),
                        "S1": float(result["S1"][i]),
                    }
                )
    return pd.DataFrame(rows)


def noise_shares(runs: pd.DataFrame, outputs: list[str]) -> pd.DataFrame:
    """How much of the variance the decomposition worked on is stochastic rather than parametric.

    With ``R`` seeds per sample point the averaged output carries a Monte-Carlo variance of
    ``within / R`` on top of the parameter-driven variance, so

        noise_share = (within / R) / Var(seed-averaged Y)

    is the share of the total decomposed variance that no parameter can be responsible for. It is
    an upper bound on how much of ``sum(ST)`` is an artefact. Above roughly 0.2 the indices are
    substantially reporting the model's own randomness, and the fix is more replicates rather than
    a larger ``n_base``: samples sharpen the map of the mean surface, replicates lower the floor.
    """
    rows = []
    for output in outputs:
        if output not in runs.columns:
            continue
        grouped = runs.groupby("sample")[output]
        within = float(grouped.var(ddof=1).mean())  # NaN when there is only one seed
        means = grouped.mean()
        between = float(means.var(ddof=1))
        replicates = float(grouped.count().mean())
        noise = within / replicates if np.isfinite(within) and replicates > 0 else np.nan
        rows.append(
            {
                "output": output,
                "mean_replicates": replicates,
                "var_within_seeds": within,
                "var_between_samples": between,
                "noise_variance": noise,
                "noise_share": noise / between if between > 0 and np.isfinite(noise) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _panels(n: int) -> tuple[plt.Figure, np.ndarray]:
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12.4, 3.4 * rows), squeeze=False)
    return fig, axes.ravel()


def fig_headline_bars(
    indices: pd.DataFrame, headline: list[str], outdir: Path, datadir: Path | None = None
) -> Path:
    """S1 against ST, per parameter, for the headline outputs.

    The gap between the two bars is the parameter's interaction load: a parameter that matters only
    through ST is one whose effect depends on where the others are, which is the ordinary case in a
    model whose central claim is that pressure is necessary but not sufficient.
    """
    _style()
    fig, axes = _panels(len(headline))
    for ax, output in zip(axes, headline):
        block = indices[indices["output"] == output].sort_values("ST")
        if block.empty:
            # Skipped by sobol_indices: constant across the design, or never recorded. Said
            # plainly, because a blank frame reads as a broken figure rather than as a result.
            ax.text(
                0.5, 0.5, f"{output}\nnot decomposed:\nconstant or missing across samples",
                ha="center", va="center", fontsize=9, color=INK_SOFT, transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            continue
        y = np.arange(len(block))
        ax.barh(y + 0.20, block["ST"], height=0.38, color=SERIES[0], label="total (ST)",
                xerr=block["ST_conf"], error_kw={"ecolor": INK_SOFT, "elinewidth": 0.8})
        ax.barh(y - 0.20, block["S1"], height=0.38, color=SERIES[1], label="first (S1)",
                xerr=block["S1_conf"], error_kw={"ecolor": INK_SOFT, "elinewidth": 0.8})
        ax.set_yticks(y)
        ax.set_yticklabels(block["parameter"], fontsize=8)
        ax.set_title(output)
        ax.set_xlabel("share of output variance")
        ax.axvline(0.0, color=GRID, linewidth=0.8)
    for ax in axes[len(headline):]:
        ax.set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Sobol' sensitivity: first-order and total-order indices",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    fig.tight_layout()
    return _finish(fig, axes, outdir, "sobol_headline", indices[indices["output"].isin(headline)],
                   datadir)


def fig_total_heatmap(
    indices: pd.DataFrame, outdir: Path, datadir: Path | None = None
) -> Path:
    """Every parameter against every output, so a parameter that only matters somewhere shows up.

    Read down a column for "what drives this outcome" and across a row for "what does this
    parameter touch". A row that is dark everywhere is a parameter the whole model rests on; a row
    that is pale everywhere is one the conclusions do not depend on.
    """
    _style()
    grid = indices.pivot(index="parameter", columns="output", values="ST")
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.05 * len(grid.columns) + 3.4, 0.42 * len(grid) + 2.4))
    # The colour scale stops at 1.0 even where an estimate exceeds it. A total-order index above
    # 1 is arithmetically impossible and means the output's variance is dominated by Monte-Carlo
    # noise rather than by parameters; letting one such cell set the scale would wash out every
    # index that is real. The printed number is always the estimate itself.
    image = ax.imshow(grid.to_numpy(), cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.grid(False)
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            value = grid.iat[i, j]
            if np.isfinite(value):
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if value > 0.55 else INK,
                )
    fig.colorbar(
        image, ax=ax, label="total-order index (ST), colour clipped at 1.0", shrink=0.85,
        extend="max",
    )
    ax.set_title("Total-order sensitivity across every output", fontsize=11.5, color=INK)
    fig.tight_layout()
    return _finish(fig, ax, outdir, "sobol_total_heatmap", grid.reset_index(), datadir)


def fig_interactions(
    indices: pd.DataFrame, primary: str, outdir: Path, datadir: Path | None = None
) -> Path:
    """Where a parameter's influence lives: on its own, or only in company.

    The right panel is the additivity check. ``sum(S1)`` near 1 means the outcome is close to a sum
    of independent parameter effects and one-at-a-time sensitivity would have been adequate; well
    below 1 means most of the behaviour is interaction, and single-parameter sweeps would have
    misreported it.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), gridspec_kw={"width_ratios": [1.5, 1.0]})

    block = indices[indices["output"] == primary].sort_values("ST")
    y = np.arange(len(block))
    axes[0].barh(y, block["S1"].clip(lower=0), color=SERIES[0], label="first order (own effect)")
    axes[0].barh(y, block["interaction"], left=block["S1"].clip(lower=0), color=SERIES[1],
                 label="interaction (ST - S1)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(block["parameter"], fontsize=8)
    axes[0].set_xlabel("share of output variance")
    axes[0].set_title(f"{primary}: own effect vs interaction")
    axes[0].legend(loc="lower right", fontsize=8)

    additivity = (
        indices.groupby("output")[["S1", "ST"]].sum().sort_values("S1", ascending=False)
    )
    x = np.arange(len(additivity))
    axes[1].bar(x, additivity["S1"], color=SERIES[0], width=0.62)
    axes[1].axhline(1.0, color=SERIES[3], linewidth=1.2, linestyle="--", label="fully additive")
    # A sum of first-order indices lies in [0, 1] when the estimates are sound; an output whose
    # variance is mostly noise can produce a wild one, and left alone a single such bar flattens
    # every other output into invisibility. The axis holds the meaningful range and off-scale
    # bars are labelled with their value, so nothing is hidden.
    low, high = float(additivity["S1"].min()), float(additivity["S1"].max())
    axes[1].set_ylim(min(-0.15, max(low - 0.1, -0.6)), max(1.15, min(high + 0.1, 1.8)))
    bottom, top = axes[1].get_ylim()
    for xi, value in zip(x, additivity["S1"]):
        if value < bottom or value > top:
            axes[1].annotate(
                f"{value:.1f}", (xi, bottom if value < bottom else top),
                xytext=(0, 6 if value < bottom else -10), textcoords="offset points",
                ha="center", fontsize=7, color=INK_SOFT,
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(additivity.index, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("sum of first-order indices")
    axes[1].set_title("Additivity: how much is explained one parameter at a time")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return _finish(fig, axes, outdir, "sobol_interactions", additivity.reset_index(), datadir)


def fig_convergence(
    curve: pd.DataFrame, headline: list[str], outdir: Path, datadir: Path | None = None,
    top: int = 6,
) -> Path:
    """ST against sample size, for the six largest parameters of each headline output.

    Flat lines mean the design is large enough for that ranking. Lines still crossing at the right
    edge mean the ordering is not yet real and ``n_base`` has to double.
    """
    _style()
    fig, axes = _panels(len(headline))
    for ax, output in zip(axes, headline):
        block = curve[curve["output"] == output]
        if block.empty:
            ax.text(
                0.5, 0.5, f"{output}\nnot decomposed", ha="center", va="center",
                fontsize=9, color=INK_SOFT, transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            continue
        largest = (
            block[block["n_base"] == block["n_base"].max()]
            .nlargest(top, "ST")["parameter"].tolist()
        )
        colours = (SERIES * 4)[: len(largest)]
        for colour, name in zip(colours, largest):
            line = block[block["parameter"] == name].sort_values("n_base")
            ax.plot(line["n_base"], line["ST"], marker="o", markersize=3.4,
                    color=colour, linewidth=1.4, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_base (sample points per block)")
        ax.set_ylabel("ST")
        ax.set_title(output)
        ax.legend(fontsize=7, ncol=2, loc="best")
    for ax in axes[len(headline):]:
        ax.set_visible(False)
    fig.suptitle(
        "Convergence of the total-order indices",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    fig.tight_layout()
    return _finish(fig, axes, outdir, "sobol_convergence",
                   curve[curve["output"].isin(headline)], datadir)


def fig_response(
    runs: pd.DataFrame, indices: pd.DataFrame, primary: str, outdir: Path,
    datadir: Path | None = None, top: int = 4, bins: int = 14,
) -> Path:
    """The primary output against the four parameters that matter most for it.

    Sobol' gives magnitude and no direction: it says ``theta`` accounts for a third of the variance
    without saying which way. These panels supply the sign and the shape -- monotone, saturating or
    threshold -- which is what a reader needs to interpret an index at all. Every individual run is
    plotted, so the vertical scatter at a given x is the model's own noise plus the effect of the
    other eleven parameters.
    """
    _style()
    block = indices[indices["output"] == primary]
    chosen = block.nlargest(top, "ST")["parameter"].tolist()
    chosen = [c for c in chosen if c in runs.columns]
    fig, axes = plt.subplots(1, max(len(chosen), 1), figsize=(3.5 * max(len(chosen), 1), 3.5),
                             squeeze=False)
    axes = axes.ravel()
    tables = []
    for ax, name in zip(axes, chosen):
        frame = runs[[name, primary]].dropna()
        ax.scatter(frame[name], frame[primary], s=5, alpha=0.18, color=SERIES[0],
                   edgecolors="none")
        edges = np.linspace(frame[name].min(), frame[name].max(), bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        binned = frame.groupby(pd.cut(frame[name], edges, include_lowest=True),
                              observed=False)[primary].mean()
        ax.plot(centres, binned.to_numpy(), color=SERIES[1], linewidth=2.0, label="binned mean")
        ax.set_xlabel(name)
        ax.set_title(f"ST = {block.loc[block['parameter'] == name, 'ST'].iat[0]:.2f}")
        tables.append(
            pd.DataFrame({"parameter": name, "bin_centre": centres, "mean": binned.to_numpy()})
        )
    axes[0].set_ylabel(primary)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Direction of effect: {primary} against its most influential parameters",
        y=1.04, fontsize=11.5, fontweight="bold", color=INK,
    )
    fig.tight_layout()
    data = pd.concat(tables, ignore_index=True) if tables else None
    return _finish(fig, axes, outdir, "response_curves", data, datadir)


def fig_noise(noise: pd.DataFrame, outdir: Path, datadir: Path | None = None) -> Path:
    """Share of the decomposed variance that is stochastic noise rather than any parameter."""
    _style()
    block = noise.dropna(subset=["noise_share"]).sort_values("noise_share", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6.4, 0.72 * len(block) + 2.6), 4.0))
    x = np.arange(len(block))
    colours = [SERIES[1] if v > 0.2 else SERIES[0] for v in block["noise_share"]]
    ax.bar(x, block["noise_share"], color=colours, width=0.62)
    ax.axhline(0.2, color=SERIES[3], linestyle="--", linewidth=1.2,
               label="0.2: indices substantially noise above here")
    ax.set_xticks(x)
    ax.set_xticklabels(block["output"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("noise share of decomposed variance")
    ax.set_title("How much of the variance no parameter can explain", fontsize=11.5, color=INK)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _finish(fig, ax, outdir, "sobol_noise", noise, datadir)


def plot_all(
    indices: pd.DataFrame,
    curve: pd.DataFrame,
    noise: pd.DataFrame,
    runs: pd.DataFrame,
    headline: list[str],
    figdir: Path,
    datadir: Path | None = None,
) -> list[Path]:
    primary = headline[0]
    written = [
        fig_headline_bars(indices, headline, figdir, datadir),
        fig_total_heatmap(indices, figdir, datadir),
        fig_interactions(indices, primary, figdir, datadir),
        fig_response(runs, indices, primary, figdir, datadir),
    ]
    if not curve.empty:
        written.append(fig_convergence(curve, headline, figdir, datadir))
    if noise["noise_share"].notna().any():
        written.append(fig_noise(noise, figdir, datadir))
    return written


# ---------------------------------------------------------------------------------------------
def analyse_run(
    run_dir: Path, figdir: Path | None = None, outdir: Path | None = None, top: int = 5
) -> int:
    """Analyse one sweep directory end to end: indices, diagnostics, figures, console summary."""
    run = load_run(run_dir)
    record, problem = run["record"], run["problem"]
    outdir = outdir or run_dir / "output_data"
    figdir = figdir or run_dir / "figures"
    outputs = run["outputs"]
    headline = [o for o in (record.get("headline") or outputs[:4]) if o in outputs] or outputs[:1]
    second_order = bool(record.get("calc_second_order"))

    complete = run["runs"].groupby("sample")["failed"].min().eq(0).sum()
    print(
        f"  {run['n_samples']} sample points, {len(run['runs'])} runs, "
        f"{complete} points with at least one good run"
    )

    indices, gaps = sobol_indices(
        problem, run["y"], outputs, run["n_samples"], calc_second_order=second_order
    )
    if indices.empty:
        raise SystemExit("No output could be decomposed -- every one was missing or constant.")
    indices.to_csv(outdir / "sobol_indices.csv", index=False)
    tables = 1

    curve = convergence(
        problem, run["y"], headline, record["n_base"], calc_second_order=second_order
    )
    if not curve.empty:
        curve.to_csv(outdir / "sobol_convergence.csv", index=False)
        tables += 1

    noise = noise_shares(run["runs"], outputs)
    noise.to_csv(outdir / "sobol_noise.csv", index=False)
    tables += 1

    written = plot_all(indices, curve, noise, run["runs"], headline, figdir, outdir)

    # --- console summary ---------------------------------------------------------------------
    imputed = {k: v for k, v in gaps.items() if v}
    if imputed:
        print("  ! samples filled with the column mean (no usable run):")
        for output, count in imputed.items():
            print(f"      {output:30s} {count}")

    for output in headline:
        block = indices[indices["output"] == output]
        if block.empty:
            continue
        print(f"\n  {output}  (sum S1 = {block['S1'].sum():.2f})")
        for _, row in block.nlargest(top, "ST").iterrows():
            flag = "" if row["ST_significant"] else "  (within CI of zero)"
            print(
                f"    {row['parameter']:26s} ST {row['ST']:6.3f} +/- {row['ST_conf']:.3f}"
                f"   S1 {row['S1']:6.3f}{flag}"
            )

    worst = noise.dropna(subset=["noise_share"]).nlargest(3, "noise_share")
    if not worst.empty:
        print("\n  Noisiest outputs (share of decomposed variance that is not parametric)")
        for _, row in worst.iterrows():
            print(f"    {row['output']:30s} {row['noise_share']:.2f}")
        if float(worst["noise_share"].iat[0]) > 0.2:
            print(
                "    ! above 0.2 the indices partly measure the model's own randomness;\n"
                "      raise sample.replicates in bounds.yaml, not n_base."
            )

    print(
        f"\nWrote {len(written)} figures to {figdir}/"
        f"\n      {tables} index tables to {outdir}/"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="a sweep directory; default is the newest under Results/sensitivity/",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--top", type=int, default=5, help="parameters listed per output in the console summary"
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir or latest_run(args.root)
    print(f"Run dir : {run_dir}")
    return analyse_run(run_dir, top=args.top)


if __name__ == "__main__":
    raise SystemExit(main())
