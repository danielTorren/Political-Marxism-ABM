"""Figures for the research questions.

Colour follows the project's reference categorical palette in its documented slot order,
which is the ordering that clears the colourblind-separation gates on the adjacent pairlist
(stacked areas, lines, bars). Two rules from that palette are load-bearing here:

* the all-pairs forms -- the choropleth maps and the scatter -- are held to a **sequential**
  ramp rather than four categorical hues, because four categorical slots do not clear the
  all-pairs floor;
* aqua and yellow sit below 3:1 on a light surface, so every categorical figure ships direct
  labels *and* writes its underlying data to CSV beside the image (the palette's relief rule).

No figure uses a second y-axis. Where two series of different scale must be compared they are
indexed to a common base instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- reference palette, light surface ----------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedcd6"
#: Blue sequential ramp, light -> dark, for continuous magnitude (conversion time).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pmabm_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

#: Categorical hues beyond the four-slot reference set, for figures that must show more than four
#: series at once -- the sensitivity panels and the larger scenario groups. Appended in a fixed
#: order and never generated, and used through :func:`series_colours` rather than by zipping, so
#: that adding an arm cannot silently drop one or make two share a hue.
#:
#: Measured, not assumed. On the **adjacent-pair** gate every consecutive pair clears the dE 8 CVD
#: target (worst: slots 3/4 at 9.5 protan, slots 2/3 at 9.9 deutan) and the dE 15 normal-vision
#: floor. Slot 7 is slate rather than the green it used to be because pink against green was
#: dE 4.2 under deuteranopia -- a red-green confusion, and an outright failure of that gate.
#:
#: On the stricter **all-pairs** gate the list does *not* pass: slot 1 (blue) against slot 5
#: (violet) is dE 6.9 protan and 4.8 deutan, both cool hues and genuinely confusable. That gate is
#: the relevant one whenever any two marks can end up visually adjacent, which is exactly the case
#: for overlapping lines in one panel -- so **a line panel takes at most four series**, where all
#: pairs clear 8. Beyond four, cut series or facet; do not reach for slot 5. Bars, stacked
#: segments and rows of cells are held to the adjacent-pair gate and may use all seven.
PALETTE_EXTENDED = [*SERIES, "#8b5cf6", "#c2456f", "#334155"]

#: Series per line panel, above which the all-pairs separation above cannot be met.
MAX_LINE_SERIES = 4


def series_colours(n: int) -> list[str]:
    """``n`` distinct hues, falling back to recycling only past the seven available."""
    return [PALETTE_EXTENDED[i % len(PALETTE_EXTENDED)] for i in range(n)]


TENURE_COLOURS = {
    "customary": SERIES[0],
    "leasehold": SERIES[1],
    "freehold": SERIES[2],
    "landless": SERIES[3],
}

#: Where a person is, rather than what tenure a household holds: the same four positions plus
#: the towns. The fifth hue extends the categorical order rather than recycling one, and the
#: sequence below is the stacking order, validated for adjacent-pair CVD separation against the
#: light surface (worst adjacent pair dE 9.1 protan, 22.9 normal vision).
POSITION_COLOURS = {
    "customary": SERIES[0],
    "leasehold": SERIES[1],
    "freehold": SERIES[2],
    "landless": SERIES[3],
    "urban": "#8b5cf6",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 9.5,
            "figure.dpi": 110,
        }
    )


def _finish(
    fig,
    ax_or_axes,
    outdir: Path,
    name: str,
    data: pd.DataFrame | None,
    datadir: Path | None = None,
) -> Path:
    """Save the figure to ``outdir`` and its table view to ``datadir`` (``outdir`` if unset)."""
    for ax in np.atleast_1d(ax_or_axes).ravel():
        if not isinstance(ax, plt.Axes):
            continue
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    if data is not None:
        tables = datadir or outdir
        tables.mkdir(parents=True, exist_ok=True)
        data.to_csv(tables / f"{name}.csv", index=False)  # the table view
    return path


def _median_band(ax, frame: pd.DataFrame, value: str, colour: str, label: str) -> None:
    """Median across seeds with an interquartile ribbon."""
    grouped = frame.groupby("t")[value]
    median, lo, hi = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
    ax.fill_between(median.index, lo, hi, color=colour, alpha=0.16, linewidth=0)
    ax.plot(median.index, median.values, color=colour, label=label)


# ---------------------------------------------------------------------------------------------
def fig_tenure(history: pd.DataFrame, outdir: Path, name: str = "tenure_shares") -> Path:
    """Tenure-state composition over time -- the model's most basic emergent output."""
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    counts = history.groupby("t")[["customary", "leasehold", "freehold", "landless"]].median()
    shares = counts.div(counts.sum(axis=1), axis=0)

    ax.stackplot(
        shares.index,
        *[shares[c].values for c in TENURE_COLOURS],
        colors=list(TENURE_COLOURS.values()),
        labels=[c.capitalize() for c in TENURE_COLOURS],
        edgecolor=SURFACE,
        linewidth=0.8,  # 2px surface gap between stacked segments
    )
    # Direct labels, required because two of these hues are sub-3:1 on a light surface.
    mid = int(len(shares) * 0.62)
    cumulative = 0.0
    for column, colour in TENURE_COLOURS.items():
        height = shares[column].iloc[mid]
        if height > 0.07:
            ax.text(
                shares.index[mid],
                cumulative + height / 2,
                column.capitalize(),
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if column != "freehold" else INK,
                fontweight="bold",
            )
        cumulative += height

    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("period")
    ax.set_ylabel("share of agents")
    ax.set_title("Tenure composition over time (median of seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    return _finish(fig, ax, outdir, name, shares.reset_index())


# ---------------------------------------------------------------------------------------------
def fig_population(
    history: pd.DataFrame,
    outdir: Path,
    name: str = "population_composition",
    datadir: Path | None = None,
) -> Path:
    """Total population, and where those people are, as shifting proportions.

    Two panels rather than one with two scales. The level and the composition answer different
    questions -- whether the countryside is reproducing, and how its people are distributed
    between tenures and the towns -- and a shared axis would force one of them into a secondary
    scale, which is the standard way to make a chart unreadable.
    """
    _style()
    keys = list(POSITION_COLOURS)
    share_cols = [f"share_persons_{k}" for k in keys]
    if not set(share_cols).issubset(history.columns):
        raise KeyError("history lacks the share_persons_* columns; re-run the model")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))

    # --- level: is there a population at all, and where does it live? --------------------
    ax = axes[0]
    totals = history.groupby("t")[
        ["total_population", "population", "population_urban"]
    ].median()
    for colour, (column, label) in zip(
        SERIES, [("total_population", "total (rural + urban)"),
                 ("population", "rural"), ("population_urban", "urban")]
    ):
        ax.plot(totals.index, totals[column].values, color=colour, label=label)
    ax.set_xlim(totals.index.min(), totals.index.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("period")
    ax.set_ylabel("persons")
    ax.set_title("Total population and its rural/urban split")
    ax.legend(loc="best")

    # --- composition: the same series read as proportions ---------------------------------
    ax = axes[1]
    shares = history.groupby("t")[share_cols].median()
    shares.columns = keys
    # Medians across seeds need not sum to one; renormalise so the stack is a composition.
    shares = shares.div(shares.sum(axis=1), axis=0)
    ax.stackplot(
        shares.index,
        *[shares[k].values for k in keys],
        colors=[POSITION_COLOURS[k] for k in keys],
        labels=[k.capitalize() for k in keys],
        edgecolor=SURFACE,
        linewidth=0.8,  # 2px surface gap between stacked segments
    )
    # Direct labels: two of these hues sit below 3:1 against the surface, so identity must not
    # rest on colour alone.
    mid = int(len(shares) * 0.62)
    cumulative = 0.0
    for key in keys:
        height = shares[key].iloc[mid]
        if height > 0.07:
            ax.text(
                shares.index[mid], cumulative + height / 2, key.capitalize(),
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=INK if key in ("freehold", "landless") else "#ffffff",
            )
        cumulative += height
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("period")
    ax.set_ylabel("share of all living persons")
    ax.set_title("Where the population is (median of seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=8)

    table = shares.add_prefix("share_").join(totals)
    return _finish(fig, axes, outdir, name, table.reset_index(), datadir)


# ---------------------------------------------------------------------------------------------
def fig_rq1_event_study(events: pd.DataFrame, outdir: Path, name: str = "rq1_event_study") -> Path:
    """RQ1: improving disposition and capital in event time around a parcel's conversion."""
    _style()
    metrics = [
        ("iota", "Improving disposition  $\\iota^T$"),
        ("capital", "Capital  $k_j$"),
        ("rent", "Rent paid  $\\rho_j$"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.4, 3.6), sharex=True)

    for ax, (metric, title) in zip(axes, metrics):
        subset = events[events["metric"] == metric]
        for colour, group in zip(SERIES, ["converted", "never converted"]):
            g = subset[subset["group"] == group]
            if g.empty:
                continue
            agg = g.groupby("event_time")["mean"].mean()
            ax.plot(agg.index, agg.values, color=colour, label=group)
        ax.axvline(0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_title(title)
        ax.set_xlabel("periods relative to conversion")

    axes[0].set_ylabel("mean across converted parcels")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "RQ1  Does improvement follow market exposure? (parcel-level event study)",
        y=1.04,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    return _finish(fig, axes, outdir, name, events)


# ---------------------------------------------------------------------------------------------
def fig_rq2_maps(
    model, outdir: Path, snapshots: int = 4, name: str = "rq2_conversion_map"
) -> Path:
    """RQ2: where and when conversion happened, on the real England lattice."""
    _style()
    n_rows, n_cols = model.geo.shape
    conv = model.first_conversion
    n_steps = model.p.n_steps
    times = np.linspace(n_steps // snapshots, n_steps, snapshots).astype(int)

    fig, axes = plt.subplots(1, snapshots, figsize=(3.0 * snapshots, 4.4))
    land = np.zeros((n_rows, n_cols), dtype=bool)
    land[model.geo.xy[:, 0], model.geo.xy[:, 1]] = True

    image = None
    for ax, t in zip(np.atleast_1d(axes), times):
        grid = np.full((n_rows, n_cols), np.nan)
        converted = (conv >= 0) & (conv < t)
        grid[model.geo.xy[converted, 0], model.geo.xy[converted, 1]] = conv[converted]
        # Unconverted land drawn as a recessive base so the coastline stays legible.
        ax.imshow(
            np.where(land, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=6,
            interpolation="nearest",
        )
        image = ax.imshow(
            grid, origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=n_steps, interpolation="nearest"
        )
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    cbar = fig.colorbar(
        image, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02, aspect=30
    )
    cbar.set_label("period of first conversion", color=INK_SOFT, fontsize=9)
    cbar.outline.set_visible(False)
    fig.suptitle(
        "RQ2  Spread of leasehold conversion",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    frame = pd.DataFrame(
        {
            "row": model.geo.xy[:, 0],
            "col": model.geo.xy[:, 1],
            "first_conversion": conv,
        }
    )
    return _finish(fig, axes, outdir, name, frame)


def fig_rq2_spread(
    curve: pd.DataFrame, stats: dict, outdir: Path, name: str = "rq2_spread"
) -> Path:
    """RQ2: the semivariogram of conversion time -- contagion or simultaneity, with no origin.

    Reading the panel: the *height* of the curve at short distance is the nugget, and a low
    nugget is local coherence. The *rise* is the distance-dependent lag. The distance at which
    it flattens is the range -- the spatial scale of the process. A flat line at 1.0 throughout
    is simultaneity: parcels a cell apart convert as differently as parcels a country apart.
    """
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if curve.empty or not np.isfinite(curve["gamma_norm"]).any():
        ax.text(0.5, 0.5, "too few conversions", ha="center", transform=ax.transAxes)
        return _finish(fig, ax, outdir, name, curve)

    ax.plot(
        curve["distance"], curve["gamma_norm"],
        marker="o", markersize=4, color=SERIES[0], linewidth=1.8,
    )
    # Total variance is where a spatially structureless field sits at every distance, so it is
    # the reference the curve is read against rather than zero.
    ax.axhline(1.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        "no spatial structure", xy=(0.99, 1.0), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=7.5, color=INK_SOFT,
    )
    if np.isfinite(stats.get("range_cells", np.nan)):
        ax.axvline(stats["range_cells"], color=SERIES[1], linewidth=1.4, linestyle=(0, (2, 2)))
        ax.annotate(
            f"range ≈ {stats['range_cells']:.0f} cells",
            xy=(stats["range_cells"], 0.04), xycoords=("data", "axes fraction"),
            ha="left", va="bottom", fontsize=8, color=SERIES[1],
        )
    ax.annotate(
        f"nugget = {stats['nugget_share']:.2f} of total variance\n"
        f"slope = {stats['slope_norm']:.3g} per cell",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=9.5, color=INK,
    )
    ax.set_xlabel("lattice distance between parcels (cells)")
    ax.set_ylabel("semivariance of conversion time / total variance")
    ax.set_ylim(0.0, None)
    ax.set_title("RQ2  Is conversion a spreading front? (origin-free)")
    return _finish(fig, ax, outdir, name, curve)


def fig_rq2_ablation(ablation: pd.DataFrame, outdir: Path, name: str = "rq2_channel_ablation") -> Path:
    """RQ2: which spread channel is necessary for a spatially spreading pattern?"""
    _style()
    agg = (
        ablation.groupby("channels")
        .agg(
            nugget=("nugget_share", "median"),
            slope=("slope_norm", "median"),
            share=("conversion_share", "median"),
        )
        .sort_values("nugget")
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    # Single-series bars: magnitude, so one hue rather than a categorical set.
    for ax, (column, label, fmt) in zip(
        axes,
        [
            ("nugget", "nugget (share of total variance)", "{:.2f}"),
            ("slope", "variogram slope (per cell)", "{:.3g}"),
            ("share", "share of parcels converted", "{:.0%}"),
        ],
    ):
        ordered = agg.sort_values(column)
        ax.barh(ordered.index, ordered[column], color=SERIES[0], height=0.62)
        ax.axvline(0, color=INK_SOFT, linewidth=1.0)
        ax.set_xlabel(label)
        for y, v in enumerate(ordered[column]):
            if np.isfinite(v):
                ax.text(
                    v, y, "  " + fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK_SOFT,
                )
    # A nugget of 1 means neighbours are as different as opposite corners: no local coherence
    # at all, which is what simultaneity looks like however much land converts.
    axes[0].axvline(1.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[0].set_title("Local coherence (lower = neighbours convert together)")
    axes[1].set_title("Distance-dependent lag (zero = simultaneity)")
    axes[2].set_title("How much converted at all")
    fig.suptitle(
        "RQ2  Ablating the two spread channels",
        y=1.02, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, agg.reset_index())


# ---------------------------------------------------------------------------------------------
def fig_rq3_regimes(regimes: pd.DataFrame, outdir: Path, name: str = "rq3_regimes") -> Path:
    """RQ3: England vs France -- same model and seeds, only theta differs."""
    _style()
    panels = [
        ("share_leasehold", "Leasehold share"),
        ("share_freehold", "Freehold share"),
        ("share_customary", "Customary share"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharex=True)
    for ax, (column, title) in zip(axes, panels):
        for colour, label in zip(SERIES, ["England", "France"]):
            subset = regimes[regimes["label"] == label]
            if subset.empty:
                continue
            _median_band(ax, subset, column, colour, label)
        ax.set_title(title)
        ax.set_xlabel("period")
    axes[0].set_ylabel("share of occupied tenancies")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "RQ3  Does the state-landlord alliance reproduce Brenner's divergence?",
        y=1.04, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, regimes)


# ---------------------------------------------------------------------------------------------
def fig_robustness(robust: pd.DataFrame, outdir: Path, name: str = "robustness") -> Path:
    """The macro-consistency checks: concentration, factor prices, exit, enclosure."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    baseline = robust[robust["label"] == "baseline"]

    # (a) farm-size concentration, baseline vs engrossment off
    ax = axes[0, 0]
    for colour, label in zip(SERIES, ["baseline", "no engrossment"]):
        subset = robust[robust["label"] == label]
        if not subset.empty:
            _median_band(ax, subset, "farm_gini", colour, label)
    ax.set_title("Farm-size concentration")
    ax.set_ylabel("Gini of holding size")
    ax.legend(loc="lower right")

    # (b) rents and wages, indexed to a common base -- never a second y-axis
    ax = axes[0, 1]
    for colour, (column, label) in zip(
        SERIES, [("mean_rhat", "leasehold rent"), ("wage", "wage")]
    ):
        series = baseline.groupby("t")[column].median()
        base = series.iloc[0] if series.iloc[0] not in (0, np.nan) else 1.0
        ax.plot(series.index, series.values / base, color=colour, label=label)
    ax.set_yscale("log")
    ax.set_title("Rents and wages (indexed, t₀ = 1)")
    ax.set_ylabel("index, log scale")
    ax.legend(loc="upper left")

    # (c) the landless pool and cumulative exit to industry
    ax = axes[1, 0]
    for colour, (column, label) in zip(
        SERIES, [("landless", "landless pool"), ("exited", "cumulative exit to industry")]
    ):
        _median_band(ax, baseline, column, colour, label)
    ax.set_title("Wage-labour pool")
    ax.set_ylabel("agents")
    ax.set_xlabel("period")
    ax.legend(loc="upper left")

    # (d) enclosure against dispossession
    ax = axes[1, 1]
    for colour, (column, label) in zip(
        SERIES, [("enclosure", "enclosure share $\\Xi$"), ("share_landless", "landless share")]
    ):
        _median_band(ax, baseline, column, colour, label)
    ax.set_title("Enclosure and landlessness")
    ax.set_ylabel("share")
    ax.set_xlabel("period")
    ax.legend(loc="lower right")

    fig.suptitle(
        "Robustness and macro-consistency checks",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    fig.tight_layout()
    return _finish(fig, axes, outdir, name, robust)


def fig_concentration(models, outdir: Path, name: str = "holding_concentration") -> Path:
    """Are peasant holdings consolidating? Gini, Lorenz, and the top decile's share."""
    from .metrics import concentration_frame, holding_sizes_at, lorenz

    _style()
    frames = []
    for seed, model in enumerate(models):
        frame = concentration_frame(model)
        frame["seed"] = seed
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    reference = models[0]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))

    # (a) Gini over time
    ax = axes[0]
    _median_band(ax, combined, "gini", SERIES[0], "Gini")
    ax.set_title("Concentration of tenant holdings")
    ax.set_ylabel("Gini of holding size")
    ax.set_xlabel("period")
    ax.set_ylim(0, 1)

    # (b) Lorenz curves at three moments -- the distribution behind the Gini
    ax = axes[1]
    last = reference.p.n_steps - 1
    marks = [0, last // 2, last]
    for colour, t in zip(SERIES, marks):
        pop, land = lorenz(holding_sizes_at(reference, t))
        ax.plot(pop, land, color=colour, label=f"t = {t}")
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_title("Lorenz curve of holding size")
    ax.set_xlabel("cumulative share of tenants")
    ax.set_ylabel("cumulative share of land")
    ax.legend(loc="upper left")

    # (c) how much land the largest tenth hold, against how many are still on one parcel
    ax = axes[2]
    for colour, (column, label) in zip(
        SERIES, [("top_decile_land_share", "land held by largest 10%"), ("single_parcel_share", "tenants on a single parcel")]
    ):
        _median_band(ax, combined, column, colour, label)
    ax.set_title("Who holds the land")
    ax.set_xlabel("period")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right")

    fig.suptitle(
        "Consolidation of peasant holdings",
        y=1.03, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, combined)


def fig_consolidation_by_fertility(
    models, outdir: Path, name: str = "consolidation_by_fertility"
) -> Path:
    """Does better land consolidate faster? Trajectories by fertility quartile, plus regions."""
    from .metrics import consolidation_by_fertility, region_consolidation_summary

    _style()
    grouped, regions = [], []
    for seed, model in enumerate(models):
        g = consolidation_by_fertility(model)
        g["seed"] = seed
        grouped.append(g)
        r = region_consolidation_summary(model)
        r["seed"] = seed
        regions.append(r)
    grouped = pd.concat(grouped, ignore_index=True)
    regions = pd.concat(regions, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    # (a) consolidation trajectory by fertility quartile, poorest to best
    ax = axes[0]
    labels = sorted(grouped["fertility_group"].unique())
    for colour, label in zip(SERIES, labels):
        subset = grouped[grouped["fertility_group"] == label]
        fertility = subset["mean_fertility"].iloc[0]
        _median_band(
            ax, subset, "mean_holding_size", colour, f"{label} (φ̄≈{fertility:.1f})"
        )
    ax.set_title("Consolidation by land quality")
    ax.set_xlabel("period")
    ax.set_ylabel("mean holding size of a parcel's farm")
    ax.legend(loc="upper left", title="fertility quartile")

    # (b) share of land sitting in large (3+ parcel) farms
    ax = axes[1]
    for colour, label in zip(SERIES, labels):
        subset = grouped[grouped["fertility_group"] == label]
        _median_band(ax, subset, "share_in_large_holdings", colour, label)
    ax.set_title("Land in farms of 3+ parcels")
    ax.set_xlabel("period")
    ax.set_ylabel("share of occupied parcels")
    ax.set_ylim(0, 1)

    # (c) region-level: land quality against how far it consolidated
    ax = axes[2]
    ax.scatter(
        regions["mean_fertility"], regions["final_mean_holding"],
        s=18, color=SERIES[0], alpha=0.55, linewidth=0,
    )
    clean = regions.dropna(subset=["mean_fertility", "final_mean_holding"])
    if len(clean) > 2 and np.ptp(clean["mean_fertility"]) > 0:
        slope, intercept = np.polyfit(clean["mean_fertility"], clean["final_mean_holding"], 1)
        xs = np.linspace(clean["mean_fertility"].min(), clean["mean_fertility"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=SERIES[1], linewidth=2.2)
        r = float(np.corrcoef(clean["mean_fertility"], clean["final_mean_holding"])[0, 1])
        ax.annotate(
            f"slope = {slope:+.2f}\nr = {r:+.2f}",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=9.5, color=INK,
        )
    ax.set_title("Region: land quality vs final farm size")
    ax.set_xlabel("mean carrying capacity $\\bar\\phi$ of region")
    ax.set_ylabel("final mean holding size")

    fig.suptitle(
        "Does better land consolidate faster?",
        y=1.03, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, grouped)


def fig_geography(model, outdir: Path, name: str = "geography") -> Path:
    """The model's physical inputs: ALC-derived fertility and the estate tessellation."""
    _style()
    n_rows, n_cols = model.geo.shape
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))

    fertility = np.full((n_rows, n_cols), np.nan)
    fertility[model.geo.xy[:, 0], model.geo.xy[:, 1]] = model.geo.phi_bar
    im = axes[0].imshow(fertility, origin="lower", cmap=SEQUENTIAL, interpolation="nearest")
    axes[0].set_title("Carrying capacity $\\bar\\phi_k$ from ALC grades")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)

    estates = np.full((n_rows, n_cols), np.nan)
    estates[model.geo.xy[:, 0], model.geo.xy[:, 1]] = model.geo.landlord
    axes[1].imshow(estates, origin="lower", cmap="tab20", interpolation="nearest")
    axes[1].set_title(f"{model.geo.n_landlords} estates (Voronoi)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(
        "Model geography: real England outline, real land quality",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    frame = pd.DataFrame(
        {
            "row": model.geo.xy[:, 0],
            "col": model.geo.xy[:, 1],
            "phi_bar": model.geo.phi_bar,
            "landlord": model.geo.landlord,
            "region": model.geo.region,
        }
    )
    return _finish(fig, axes, outdir, name, frame)
