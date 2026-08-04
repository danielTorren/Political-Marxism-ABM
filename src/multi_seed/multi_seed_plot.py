"""Figures for a multi-seed run: means with 95% confidence intervals.

Uses the same dashboard definitions as the single-run script -- only the renderer differs, so
the two sets of figures are directly comparable panel for panel. On top of those it adds the
figures that only make sense with replication: the regime comparison and the seed spread.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmabm import diagnostics, plots
from pmabm.plots import GRID, INK, INK_SOFT, SERIES, SURFACE, _finish, _style


def _band(ax, frame: pd.DataFrame, column: str, colour: str, label: str) -> None:
    diagnostics.mean_ci_renderer(ax, frame, column, colour, label)


def fig_regime_comparison(
    arms: dict, outdir: Path, name: str = "regimes_mean_ci", datadir: Path | None = None
) -> Path:
    """England against France on the quantities the theory says should diverge."""
    _style()
    panels = [
        ("share_leasehold", "Leasehold share of tenancies"),
        ("share_freehold", "Freehold share"),
        ("share_customary", "Customary share"),
        ("farm_gini", "Farm-size Gini"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6), sharex=True)
    for ax, (column, title) in zip(axes, panels):
        for colour, (label, arm) in zip(SERIES, arms.items()):
            _band(ax, arm["history"], column, colour, label)
        ax.set_title(title)
        ax.set_xlabel("period")
    axes[0].set_ylabel("share")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "England vs France: mean across seeds, 95% CI",
        y=1.04, fontsize=11.5, fontweight="bold", color=INK,
    )
    combined = pd.concat([arm["history"] for arm in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, name, combined, datadir)


def fig_seed_spread(
    history: pd.DataFrame, outdir: Path, name: str = "seed_spread", datadir: Path | None = None
) -> Path:
    """Every seed drawn individually, so replication masking real variability is visible.

    A confidence interval on the mean can look reassuringly tight while individual runs
    diverge wildly; this panel is the check on that.
    """
    _style()
    columns = [
        ("share_leasehold", "Leasehold share"),
        ("farm_gini", "Farm-size Gini"),
        ("wage", "Wage"),
        ("population", "Living population"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6))
    for ax, (column, title) in zip(axes, columns):
        if column not in history:
            continue
        for seed, group in history.groupby("seed"):
            ax.plot(group["t"], group[column], color=SERIES[0], alpha=0.35, linewidth=1.0)
        mean = history.groupby("t")[column].mean()
        ax.plot(mean.index, mean.values, color=SERIES[1], linewidth=2.4, label="mean")
        ax.set_title(title)
        ax.set_xlabel("period")
        if column == "wage":
            ax.set_yscale("symlog")
    axes[0].set_ylabel("value")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Individual seeds behind the mean",
        y=1.04, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, history, datadir)


def fig_fertility_mean_ci(
    fertility: pd.DataFrame,
    outdir: Path,
    name: str = "consolidation_by_fertility_mean_ci",
    datadir: Path | None = None,
) -> Path:
    """Consolidation by land-quality quartile, averaged over seeds."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    labels = sorted(fertility["fertility_group"].unique())
    for ax, (column, title, ylabel) in zip(
        axes,
        [
            ("mean_holding_size", "Consolidation by land quality", "mean holding size of a parcel's farm"),
            ("share_in_large_holdings", "Land in farms of 3+ parcels", "share of occupied parcels"),
        ],
    ):
        for colour, label in zip(SERIES, labels):
            subset = fertility[fertility["fertility_group"] == label]
            fert = subset["mean_fertility"].iloc[0]
            _band(ax, subset, column, colour, f"{label} (φ̄≈{fert:.1f})")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[0].legend(loc="upper left", title="fertility quartile", fontsize=8)
    fig.suptitle(
        "Does better land consolidate faster? Mean across seeds, 95% CI",
        y=1.04, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, fertility, datadir)


def fig_event_study_mean_ci(
    events: pd.DataFrame,
    outdir: Path,
    name: str = "rq1_event_study_mean_ci",
    datadir: Path | None = None,
) -> Path:
    """RQ1 in event time, averaged over seeds."""
    _style()
    metrics = [
        ("iota", "Improving disposition $\\iota^T$"),
        ("capital", "Capital $k_j$"),
        ("rent", "Rent paid $\\rho_j$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        subset = events[events["metric"] == metric]
        for colour, group in zip(SERIES, ["converted", "never converted"]):
            g = subset[subset["group"] == group]
            if g.empty:
                continue
            grouped = g.groupby("event_time")["mean"]
            mean, count = grouped.mean(), grouped.count().clip(lower=1)
            sem = grouped.std(ddof=1) / np.sqrt(count)
            ax.fill_between(
                mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                color=colour, alpha=0.18, linewidth=0,
            )
            ax.plot(mean.index, mean.values, color=colour, label=group)
        ax.axvline(0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_title(title)
        ax.set_xlabel("periods relative to conversion")
    axes[0].set_ylabel("mean across parcels and seeds")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "RQ1  Improvement around conversion: mean across seeds, 95% CI",
        y=1.04, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, events, datadir)


def plot_all(arms: dict, outdir: Path, datadir: Path | None = None) -> list[Path]:
    """Draw every multi-seed figure. ``arms`` maps a regime label to its frames.

    Figures land in ``outdir``; the table view behind each figure lands in ``datadir``, which
    defaults to ``outdir`` so that a caller who does not separate them keeps the flat layout.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = datadir or outdir
    datadir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    england = arms["England"]

    # The shared dashboards, drawn as mean + 95% CI.
    written += diagnostics.render(
        england["history"], diagnostics.mean_ci_renderer, outdir
    )

    written.append(fig_seed_spread(england["history"], outdir, datadir=datadir))
    written.append(plots.fig_population(england["history"], outdir, datadir=datadir))
    written.append(fig_fertility_mean_ci(england["fertility"], outdir, datadir=datadir))
    if not england["events"].empty:
        written.append(fig_event_study_mean_ci(england["events"], outdir, datadir=datadir))
    if len(arms) > 1:
        written.append(fig_regime_comparison(arms, outdir, datadir=datadir))

    diagnostics.figure_index().to_csv(datadir / "figure_index.csv", index=False)
    return written


if __name__ == "__main__":
    import multi_seed_gen

    raise SystemExit(multi_seed_gen.main())
