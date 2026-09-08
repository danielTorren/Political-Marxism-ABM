"""Diagnostic figures for a fitted emulator.

    uv run python src/emulator/emulator_plot.py --model-dir <run-dir>/emulator

Reads what :mod:`emulator_train` wrote and draws it. Colour, styling and the write-the-table-
beside-the-image rule come from :mod:`pmabm.plots`, so these figures sit beside the model's own
without restyling.

The figures exist to answer four questions, in this order:

1. *Is the emulator good enough to use?* -- ``accuracy``, ``parity``. Both are drawn against the
   seed-noise ceiling rather than against 1.0. An emulator at R^2 = 0.70 on an output whose
   ceiling is 0.72 is finished; on one whose ceiling is 0.99 it is not, and the same number means
   opposite things in the two cases.
2. *Does it know what it does not know?* -- ``coverage``. A 90% interval that contains the truth
   65% of the time makes every calibration downstream of it wrong, and nothing else in the
   diagnostics would reveal it.
3. *Would more ABM runs help?* -- ``learning_curve``. A curve still rising at the full corpus
   says the eight hours bought an emulator that is sample-limited, not architecture-limited.
4. *Was a network necessary?* -- ``baselines``. If ridge is within noise of the network on every
   output, the response surface is linear and the paper should say so and use the ridge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pmabm.plots import GRID, INK_SOFT, PALETTE_EXTENDED, SERIES, _finish, _style  # noqa: E402


def _grid(n: int, per_row: int = 4, size: tuple[float, float] = (3.1, 2.8)):
    rows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(rows, per_row, figsize=(per_row * size[0], rows * size[1]))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


# ---------------------------------------------------------------------------------------------
def fig_accuracy(metrics: pd.DataFrame, outdir: Path) -> Path:
    """R^2 per output against the ceiling the ABM's own seed noise imposes.

    The grey bar is what a *perfect* function of the parameters could achieve on this corpus; the
    coloured bar is what the emulator achieved. The gap between the coloured bar and the grey one
    is the emulator's remaining error. The gap between the grey bar and 1.0 belongs to the model,
    not to the emulator, and closing it is impossible rather than merely expensive.
    """
    # An output can be unscoreable on the held-out rows: constant there (every run censored at
    # the same period), or undefined (the variogram statistics need enough converted parcels to
    # have a spatial pattern at all). Those carry no information in a bar chart and are listed in
    # a footnote instead of drawn as an empty row.
    scored = metrics[metrics["r2"].notna() & metrics["r2_ceiling"].notna()]
    unscored = sorted(set(metrics["output"]) - set(scored["output"]))
    frame = scored.sort_values("r2_ceiling", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(frame) + 1.8))
    y = np.arange(len(frame))
    ax.barh(y, frame["r2_ceiling"], color=GRID, height=0.72, label="ceiling (1 - seed noise share)")
    ax.barh(y, frame["r2"].clip(lower=0), color=SERIES[0], height=0.44, label="emulator R²")
    ax.set_yticks(y, frame["output"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of held-out variance explained")
    ax.set_title("Emulator accuracy against the ABM's own noise floor")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    if unscored:
        ax.annotate(
            "not scoreable on the held-out rows (constant or undefined there): "
            + ", ".join(unscored),
            (0, -0.5 / max(len(frame), 1) - 0.11), xycoords="axes fraction",
            fontsize=7.5, color=INK_SOFT, va="top", wrap=True,
        )
    return _finish(fig, ax, outdir, "emu_accuracy", metrics)


def fig_parity(payload: dict, metrics: pd.DataFrame, outdir: Path, outputs: list[str]) -> Path:
    """Predicted against actual on held-out design points, with the predictive interval."""
    names = list(payload["scalar_names"])
    chosen = [o for o in outputs if o in names][:8] or names[:8]
    fig, axes = _grid(len(chosen))

    rows = []
    for ax, output in zip(axes, chosen):
        column = names.index(output)
        actual = payload["scalar_actual"][:, column]
        predicted = payload["scalar_mean"][:, column]
        sd = payload["scalar_sd"][:, column]
        ax.errorbar(
            actual, predicted, yerr=sd, fmt="o", ms=2.6, lw=0.5, alpha=0.55,
            color=SERIES[0], ecolor=GRID, mec="none",
        )
        finite = np.isfinite(actual) & np.isfinite(predicted)
        if finite.any():
            lo = float(min(actual[finite].min(), predicted[finite].min()))
            hi = float(max(actual[finite].max(), predicted[finite].max()))
            ax.plot([lo, hi], [lo, hi], color=INK_SOFT, lw=1.0, ls="--", zorder=0)
        row = metrics[metrics["output"] == output]
        note = ""
        if len(row):
            note = f"R² {row['r2'].iloc[0]:.2f} / ceiling {row['r2_ceiling'].iloc[0]:.2f}"
        ax.set_title(output, fontsize=9)
        ax.set_xlabel("ABM")
        ax.set_ylabel("emulator")
        ax.annotate(note, (0.04, 0.93), xycoords="axes fraction", fontsize=8, color=INK_SOFT, va="top")
        rows.append(pd.DataFrame({"output": output, "actual": actual, "predicted": predicted, "sd": sd}))

    fig.suptitle("Held-out predictions, error bars at one predictive sd", y=1.0, fontsize=11)
    fig.tight_layout()
    return _finish(fig, axes, outdir, "emu_parity", pd.concat(rows, ignore_index=True))


def fig_coverage(coverage: pd.DataFrame, outdir: Path) -> Path:
    """Nominal against achieved interval coverage, per output.

    A point below the diagonal is an overconfident interval, which is the failure mode that
    matters: history matching with overconfident intervals rules out parameter regions that
    should have survived, and the resulting "the theory requires X" is an artefact of the
    emulator rather than a finding about the model.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for offset, (output, group) in enumerate(coverage.groupby("output")):
        ax.plot(
            group["level"], group["coverage"], marker="o", ms=3.5, lw=1.0, alpha=0.7,
            color=PALETTE_EXTENDED[offset % len(PALETTE_EXTENDED)],
        )
    mean = coverage.groupby("level")["coverage"].mean()
    ax.plot(mean.index, mean.to_numpy(), color="black", lw=2.4, marker="s", ms=5, label="mean over outputs")
    ax.plot([0, 1], [0, 1], color=INK_SOFT, ls="--", lw=1.0, zorder=0, label="nominal")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("nominal level")
    ax.set_ylabel("achieved coverage on held-out runs")
    ax.set_title("Are the predictive intervals honest?")
    ax.legend(loc="upper left")
    return _finish(fig, ax, outdir, "emu_coverage", coverage)


def fig_trajectories(payload: dict, outdir: Path, series: list[str], n_rows: int = 4) -> Path:
    """A handful of held-out runs: what the ABM did, and what the emulator predicted it would.

    The rows are chosen by spread rather than at random -- the run with the largest final value,
    the smallest, and two between -- because an average-looking sample of a space-filling design
    hides exactly the corners where a surrogate fails.
    """
    available = [s for s in series if f"traj_actual_{s}" in payload]
    if not available:
        raise SystemExit("predictions.npz carries no trajectories")

    fig, axes = _grid(len(available), per_row=3, size=(3.6, 2.9))
    tables = []
    for ax, name in zip(axes, available):
        actual = payload[f"traj_actual_{name}"]
        predicted = payload[f"traj_mean_{name}"]
        band = payload[f"traj_band_{name}"]
        final = np.nan_to_num(actual[:, -1], nan=0.0)
        order = np.argsort(final)
        picks = np.unique(np.linspace(0, len(order) - 1, n_rows).astype(int))
        steps = np.arange(actual.shape[1])
        for slot, position in enumerate(picks):
            row = order[position]
            colour = PALETTE_EXTENDED[slot % len(PALETTE_EXTENDED)]
            ax.fill_between(steps, band[row, :, 0], band[row, :, 1], color=colour, alpha=0.16, lw=0)
            ax.plot(steps, predicted[row], color=colour, lw=1.6, ls="--")
            ax.plot(steps, actual[row], color=colour, lw=1.6)
            tables.append(
                pd.DataFrame(
                    {"series": name, "row": int(row), "t": steps,
                     "actual": actual[row], "predicted": predicted[row],
                     "lo": band[row, :, 0], "hi": band[row, :, 1]}
                )
            )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("period")
    fig.suptitle("Held-out trajectories: ABM (solid) against emulator (dashed, shaded band)", y=1.0)
    fig.tight_layout()
    return _finish(fig, axes, outdir, "emu_trajectories", pd.concat(tables, ignore_index=True))


def fig_learning_curve(curve: pd.DataFrame, outdir: Path) -> Path:
    """Accuracy against corpus size: would more ABM runs still buy anything?"""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(curve["n_points"], curve["mean_r2"], marker="o", color=SERIES[0], label="mean R²")
    if "mean_r2_of_ceiling" in curve.columns:
        ax.plot(
            curve["n_points"], curve["mean_r2_of_ceiling"], marker="s",
            color=SERIES[1], label="mean R² as a share of ceiling",
        )
    ax.set_xscale("log")
    ax.set_xlabel("design points in the training set")
    ax.set_ylabel("held-out accuracy, averaged over scalar outputs")
    ax.set_title("Is the emulator sample-limited?")
    ax.legend(loc="lower right")
    return _finish(fig, ax, outdir, "emu_learning_curve", curve)


def fig_baselines(baselines: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    """The network against ridge and a random forest on the same split."""
    wide = baselines.pivot(index="output", columns="baseline", values="r2")
    wide["emulator"] = metrics.set_index("output")["r2"]
    wide["ceiling"] = metrics.set_index("output")["r2_ceiling"]
    wide = wide.sort_values("emulator")

    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(wide) + 1.4))
    y = np.arange(len(wide))
    columns = [c for c in ("ridge", "random_forest", "emulator") if c in wide.columns]
    height = 0.8 / len(columns)
    for offset, column in enumerate(columns):
        ax.barh(
            y + (offset - (len(columns) - 1) / 2) * height, wide[column].clip(lower=0),
            height=height * 0.9, color=SERIES[offset % len(SERIES)], label=column,
        )
    ax.plot(wide["ceiling"], y, ls="none", marker="|", ms=14, color="black", label="ceiling")
    ax.set_yticks(y, wide.index, fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("held-out R²")
    ax.set_title("Was a network necessary?")
    ax.legend(loc="lower right", ncols=2)
    ax.grid(axis="y", visible=False)
    return _finish(fig, ax, outdir, "emu_baselines", wide.reset_index())


def fig_training(history: pd.DataFrame, outdir: Path) -> Path:
    """Train and validation NLL per ensemble member."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for member, group in history.groupby("member"):
        colour = PALETTE_EXTENDED[int(member) % len(PALETTE_EXTENDED)]
        ax.plot(group["epoch"], group["train_nll"], color=colour, lw=1.2, alpha=0.5)
        ax.plot(group["epoch"], group["val_nll"], color=colour, lw=1.8, label=f"member {member}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Gaussian NLL (standardised units)")
    ax.set_title("Training: faint is train, solid is validation")
    ax.legend(loc="upper right", ncols=2)
    return _finish(fig, ax, outdir, "emu_training", history)


def fig_basis(traj_metrics: pd.DataFrame, outdir: Path) -> Path:
    """How many components each series needed, and how much of it the basis can carry at all."""
    frame = traj_metrics.sort_values("n_components")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.34 * len(frame) + 1.8))
    y = np.arange(len(frame))

    axes[0].barh(y, frame["n_components"], color=SERIES[0], height=0.7)
    axes[0].set_yticks(y, frame["series"], fontsize=8)
    axes[0].set_xlabel("PCA components retained")
    axes[0].set_title("Basis size")
    axes[0].grid(axis="y", visible=False)

    axes[1].barh(y, frame["r2_basis_ceiling"].clip(lower=0), color=GRID, height=0.7, label="basis ceiling")
    axes[1].barh(y, frame["r2_pooled"].clip(lower=0), color=SERIES[1], height=0.42, label="emulator")
    axes[1].set_yticks(y, [""] * len(frame))
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("held-out R², pooled over runs and periods")
    axes[1].set_title("Trajectory accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", visible=False)

    fig.tight_layout()
    return _finish(fig, axes, outdir, "emu_basis", frame)


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="output of emulator_train.py")
    parser.add_argument("--figdir", type=Path, default=None, help="default: <model-dir>/figures")
    parser.add_argument(
        "--outputs", nargs="*", default=None,
        help="scalars to draw parity panels for (default: the first eight)",
    )
    parser.add_argument("--series", nargs="*", default=None, help="trajectory series to draw")
    args = parser.parse_args(argv)

    _style()
    model_dir = args.model_dir
    figdir = Path(args.figdir or (model_dir / "figures"))
    figdir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(model_dir / "metrics_scalars.csv")
    payload = dict(np.load(model_dir / "predictions.npz", allow_pickle=True))
    payload["scalar_names"] = [str(s) for s in payload["scalar_names"]]
    series = [str(s) for s in payload["series"]]

    written = [
        fig_accuracy(metrics, figdir),
        fig_parity(payload, metrics, figdir, args.outputs or list(metrics["output"])),
        fig_coverage(pd.read_csv(model_dir / "coverage.csv"), figdir),
    ]
    if (model_dir / "metrics_trajectories.csv").is_file():
        traj_metrics = pd.read_csv(model_dir / "metrics_trajectories.csv")
        written.append(fig_basis(traj_metrics, figdir))
        written.append(fig_trajectories(payload, figdir, args.series or series))
    if (model_dir / "training_history.csv").is_file():
        written.append(fig_training(pd.read_csv(model_dir / "training_history.csv"), figdir))
    if (model_dir / "learning_curve.csv").is_file():
        written.append(fig_learning_curve(pd.read_csv(model_dir / "learning_curve.csv"), figdir))
    if (model_dir / "baselines.csv").is_file():
        written.append(fig_baselines(pd.read_csv(model_dir / "baselines.csv"), metrics, figdir))

    print(f"Wrote {len(written)} figures to {figdir}/")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
