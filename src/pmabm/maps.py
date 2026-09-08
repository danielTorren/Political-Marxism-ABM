"""Spatial figures, averaged over seeds.

Everything here reads the two per-parcel frames written by a scenario run --
:func:`pmabm.metrics.parcel_frame` and :func:`pmabm.metrics.geography_frame` -- plus the
per-county series of :func:`pmabm.metrics.county_history_frame`. Nothing here needs a live
``Model``, which is the point: the maps that existed before this module took a model object and
so could only ever draw a single seed, and the scenario suite discards its models.

**Why cross-seed maps rather than one run's map.** Every mechanism in the model is stochastic, so
a single run's conversion map is one draw from a distribution over spatial patterns and says
nothing about which of its features are the model's and which are the draw's. Under
``run.fixed_geography`` the lattice is shared by every seed, so parcel *n* is the same land in
every replicate and the distribution can be summarised at each place:

* :func:`fig_hazard_map` -- the share of seeds in which a parcel had converted by period *t*. A
  probability field, and the correct object under censoring: ``first_conversion`` is ``-1`` for
  land that never converted, so a mean of that column is meaningless while a share is not.
* :func:`fig_conversion_uncertainty` -- the cross-seed standard deviation of conversion time,
  which separates land whose timing is *structurally determined* from land where it is a
  coin-flip. No single-seed map can show this, and it is the figure that says how much of the
  spatial pattern is a finding at all.
* :func:`fig_difference_map` -- two arms differenced parcel-by-parcel on matching seeds, which is
  the spatial form of the paper's stated convention that an ablation is reported against the same
  seeds as its baseline.

**Colour.** Magnitude takes the single-hue sequential ramp of :mod:`pmabm.plots`. Differences take
:data:`DIVERGING`, a two-hue ramp with a neutral grey midpoint and symmetric limits, so zero
change is colourless and sign is read from hue. Its poles separate by dE 29 in normal vision, 18
under protanopia and 27 under deuteranopia, against a target of 8 -- computed, not judged. No
categorical hues are used on any map: four categorical slots do not clear the all-pairs
separation floor, which is the rule :mod:`pmabm.plots` already follows for its choropleths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .plots import GRID, INK, INK_SOFT, SEQUENTIAL, SERIES, SURFACE, _finish, _style

#: Diverging ramp for difference maps: blue and orange poles about a neutral grey. Warm against
#: cool, so the midpoint reads as "nothing" -- a same-family pair (blue/aqua) fails that test even
#: when its endpoints are far apart. Grey rather than a third hue at the centre for the same
#: reason. Equal step count either side, and always drawn with symmetric limits.
DIVERGING = LinearSegmentedColormap.from_list(
    "pmabm_diverging",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f7b48d", "#eb6834", "#a83208"],
)

#: Minimum seeds that must have converted a parcel before its cross-seed timing spread is drawn.
#: Below this the standard deviation is an artefact of two or three draws rather than a measure of
#: how variable the timing is, and the cell is left blank instead of asserting a number.
MIN_SEEDS_FOR_SPREAD = 4


# ---------------------------------------------------------------------------------------------
# lattice helpers
# ---------------------------------------------------------------------------------------------
def _lattice_shape(geography: pd.DataFrame) -> tuple[int, int]:
    return int(geography["row"].max()) + 1, int(geography["col"].max()) + 1


def _grid(geography: pd.DataFrame, values: pd.Series) -> np.ndarray:
    """Scatter a per-parcel series onto the lattice, NaN off-parcel.

    ``values`` is indexed by ``parcel``; it is reindexed onto the geography's own parcel order
    rather than assumed aligned, because a groupby result is sorted by key and a parcel absent
    from one arm would otherwise shift every subsequent cell.
    """
    n_rows, n_cols = _lattice_shape(geography)
    grid = np.full((n_rows, n_cols), np.nan)
    aligned = values.reindex(geography["parcel"].to_numpy()).to_numpy(dtype=float)
    grid[geography["row"].to_numpy(), geography["col"].to_numpy()] = aligned
    return grid


def _land(geography: pd.DataFrame) -> np.ndarray:
    """A faint base layer of every parcel, so the coastline stays legible where data is absent."""
    n_rows, n_cols = _lattice_shape(geography)
    land = np.full((n_rows, n_cols), np.nan)
    land[geography["row"].to_numpy(), geography["col"].to_numpy()] = 1.0
    return land


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _draw(ax, geography: pd.DataFrame, values: pd.Series, **kwargs):
    """One map panel: recessive land base, then the data on top."""
    ax.imshow(
        _land(geography), origin="lower", cmap="Greys", vmin=0, vmax=6, interpolation="nearest"
    )
    image = ax.imshow(
        _grid(geography, values), origin="lower", interpolation="nearest", **kwargs
    )
    _bare(ax)
    return image


def _colorbar(fig, image, axes, label: str) -> None:
    bar = fig.colorbar(
        image, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02, aspect=30
    )
    bar.set_label(label, color=INK_SOFT, fontsize=9)
    bar.outline.set_visible(False)


def _converted_by(parcels: pd.DataFrame, t: int) -> pd.Series:
    """Share of seeds in which each parcel had converted by period ``t``.

    ``first_conversion >= 0`` is the "ever converted" test and the ``<= t`` the censoring one, so
    a parcel that converted at 150 counts as unconverted in the t=100 panel. That is what makes
    the sequence of panels a front rather than a set of thresholded copies of one field.
    """
    flag = (parcels["first_conversion"] >= 0) & (parcels["first_conversion"] <= t)
    return parcels.assign(_c=flag).groupby("parcel")["_c"].mean()


def _snapshots(n_steps: int, count: int = 5) -> list[int]:
    return [int(round(n_steps * f)) - 1 for f in np.linspace(1 / count, 1.0, count)]


def _shared_limit(fields: list[pd.Series], floor: float = 0.1) -> float:
    """One upper limit for a set of small-multiple panels, taken from the data.

    Panels in a series must share a scale -- a per-panel scale would make a growing quantity look
    static -- but fixing that shared scale at 1.0 because the quantity is a share wastes most of
    the ramp whenever the transition does not go to completion, and every panel then reads as the
    same pale wash. This takes the largest value across the whole series instead, so the ramp is
    spent on the range the data actually occupies, and the colourbar carries the number so the
    scale is never implicit. The floor keeps a near-empty series from being scaled up into
    spurious drama.
    """
    highest = max((float(np.nanmax(f)) if len(f) and np.isfinite(f).any() else 0.0) for f in fields)
    return max(floor, min(1.0, np.ceil(highest * 20) / 20))


# ---------------------------------------------------------------------------------------------
# 1. the hazard field
# ---------------------------------------------------------------------------------------------
def fig_hazard_map(
    parcels: pd.DataFrame,
    geography: pd.DataFrame,
    outdir: Path,
    n_steps: int,
    name: str = "map_conversion_hazard",
    datadir: Path | None = None,
    count: int = 5,
) -> Path:
    """Probability of having converted, by place and period, across seeds.

    Read left to right: if the dark area grows *outward from where it already is*, conversion is
    spreading; if it deepens uniformly everywhere, the transition is simultaneous and the theory
    is missing an inter-estate mechanism, which is RQ2's constructive result. Intermediate values
    are the interesting ones -- a parcel at 0.5 converts in half the histories, so the front's
    edge is genuinely indeterminate rather than merely unobserved.
    """
    _style()
    times = _snapshots(n_steps, count)
    fields = [_converted_by(parcels, t) for t in times]
    limit = _shared_limit(fields)

    fig, axes = plt.subplots(1, len(times), figsize=(2.6 * len(times), 3.9))
    axes = np.atleast_1d(axes)

    rows, image = [], None
    for ax, t, share in zip(axes, times, fields):
        image = _draw(ax, geography, share, cmap=SEQUENTIAL, vmin=0.0, vmax=limit)
        ax.set_title(f"t = {t + 1}", fontsize=10)
        table = share.rename("share_of_seeds_converted").reset_index()
        table["t"] = t + 1
        rows.append(table)

    _colorbar(fig, image, axes, "share of seeds in which the parcel had converted")
    fig.suptitle(
        "Where conversion happens, and how reliably: share of seeds converted by each period",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, pd.concat(rows, ignore_index=True), datadir)


# ---------------------------------------------------------------------------------------------
# 2. structure against luck
# ---------------------------------------------------------------------------------------------
def fig_conversion_uncertainty(
    parcels: pd.DataFrame,
    geography: pd.DataFrame,
    outdir: Path,
    name: str = "map_conversion_uncertainty",
    datadir: Path | None = None,
) -> Path:
    """How much of the spatial pattern is the model's, and how much is the draw's.

    Three panels, and the third is the one that cannot be drawn from a single run. Left: how often
    each parcel ever converts. Middle: when, averaged over the seeds in which it did. Right: the
    spread of that timing across seeds -- low where the mechanisms fix the date, high where the
    same land converts early in one history and late in another.

    A pattern in the left panel with a *flat* right panel is a structural result: the model puts
    the transition in the same places every time. The same left panel with a right panel as large
    as the run itself would mean the geography of the transition is a coin-flip that averaging has
    merely smoothed, which is a finding about the model rather than about England, and it is
    exactly what a single-seed map would have concealed.
    """
    _style()
    converted = parcels[parcels["first_conversion"] >= 0]
    grouped = converted.groupby("parcel")["first_conversion"]
    n_seeds = int(parcels["seed"].nunique())

    ever = parcels.assign(_c=parcels["first_conversion"] >= 0).groupby("parcel")["_c"].mean()
    mean_time = grouped.mean()
    spread = grouped.std(ddof=1)
    # Only where enough histories converted for a spread to mean anything.
    enough = grouped.count() >= MIN_SEEDS_FOR_SPREAD
    spread = spread.where(enough)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    panels = [
        (ever, "How often it converts", SEQUENTIAL, (0.0, 1.0), "share of seeds"),
        (mean_time, "When, on average", SEQUENTIAL, (None, None), "period"),
        (
            spread,
            "How variable that timing is",
            SEQUENTIAL,
            (None, None),
            "SD of conversion period across seeds",
        ),
    ]
    for ax, (values, title, cmap, (lo, hi), label) in zip(axes, panels):
        image = _draw(ax, geography, values, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label(label, color=INK_SOFT, fontsize=8)
        bar.outline.set_visible(False)
    axes[2].text(
        0.02, -0.06,
        f"blank where fewer than {MIN_SEEDS_FOR_SPREAD} of {n_seeds} seeds converted",
        transform=axes[2].transAxes, fontsize=7.5, color=INK_SOFT, va="top",
    )
    fig.suptitle(
        "Structure or luck: the same land across every seed",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    table = (
        pd.DataFrame(
            {
                "share_of_seeds_converted": ever,
                "mean_conversion_period": mean_time,
                "sd_conversion_period": spread,
                "n_seeds_converted": grouped.count(),
            }
        )
        .reset_index()
        .merge(geography[["parcel", "row", "col", "county_name", "phi_bar"]], on="parcel")
    )
    return _finish(fig, axes, outdir, name, table, datadir)


# ---------------------------------------------------------------------------------------------
# 3. paired difference between arms
# ---------------------------------------------------------------------------------------------
def fig_difference_map(
    parcels_a: pd.DataFrame,
    parcels_b: pd.DataFrame,
    geography: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
    name: str,
    datadir: Path | None = None,
) -> Path:
    """Where a mechanism matters: two arms differenced parcel-by-parcel on matching seeds.

    The difference is taken *within* seed and then averaged, not between the two arms' averages.
    Those coincide in expectation but not in variance: seeds share an estate layout, an initial
    rent draw and a shock sequence, and differencing inside a seed removes all three, leaving only
    what the mechanism did. This is the spatial form of the convention the paper states for its
    time series.

    Blue is where the first arm converts *more*, orange where it converts less, grey where the
    mechanism changed nothing. Limits are symmetric about zero, so grey means zero rather than
    "middle of the range".
    """
    _style()
    shared = sorted(set(parcels_a["seed"]).intersection(parcels_b["seed"]))
    a = parcels_a[parcels_a["seed"].isin(shared)]
    b = parcels_b[parcels_b["seed"].isin(shared)]

    def per_seed_field(frame: pd.DataFrame) -> pd.DataFrame:
        flag = frame["first_conversion"] >= 0
        return frame.assign(_c=flag).pivot_table(
            index="parcel", columns="seed", values="_c", aggfunc="mean"
        )

    field_a, field_b = per_seed_field(a), per_seed_field(b)
    common = field_a.columns.intersection(field_b.columns)
    delta_per_seed = field_a[common] - field_b[common]
    delta = delta_per_seed.mean(axis=1)

    # Paired interval on the national mean, so the map carries a number as well as a pattern.
    national = delta_per_seed.mean(axis=0)
    n = len(national)
    ci = 1.96 * float(national.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0

    limit = float(np.nanmax(np.abs(delta))) or 1.0
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    image = _draw(ax, geography, delta, cmap=DIVERGING, vmin=-limit, vmax=limit)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(
        f"share of seeds converted: {label_a} minus {label_b}", color=INK_SOFT, fontsize=8
    )
    bar.outline.set_visible(False)
    ax.set_title(
        f"national mean {national.mean():+.3f} ± {ci:.3f}  (paired, n = {n})",
        fontsize=9.5, color=INK_SOFT, fontweight="normal",
    )
    fig.suptitle(
        f"{label_a} against {label_b}, same seeds",
        y=0.98, fontsize=11.5, fontweight="bold", color=INK,
    )
    table = (
        delta.rename("delta_share_converted")
        .reset_index()
        .merge(geography[["parcel", "row", "col", "county_name", "phi_bar"]], on="parcel")
    )
    table["arm_a"], table["arm_b"], table["n_pairs"] = label_a, label_b, n
    return _finish(fig, ax, outdir, name, table, datadir)


# ---------------------------------------------------------------------------------------------
# 4. counties
# ---------------------------------------------------------------------------------------------
def fig_county_choropleth(
    parcels: pd.DataFrame,
    geography: pd.DataFrame,
    outdir: Path,
    name: str = "map_county_outcomes",
    datadir: Path | None = None,
) -> Path:
    """County-level outcomes, drawn on the lattice and tabulated with cross-seed intervals.

    The county is the model's real upper tier -- estates are seeded per county and never straddle
    one -- and is the unit a reader can name. Kept strictly *descriptive*: the ordering of counties
    is not compared against where the historiography locates early agrarian capitalism, which is
    the question the removed RQ9 asked and which needs a named comparison set this figure does not
    have. The CSV beside it carries the per-county mean and 95% interval, so the ranking can be
    read with its uncertainty rather than off the colour alone.
    """
    _style()
    joined = parcels.merge(geography[["parcel", "county", "county_name"]], on="parcel")
    joined["converted"] = joined["first_conversion"] >= 0
    joined["leasehold"] = joined["final_state"] == 1

    # Per seed per county first, so the interval is across seeds rather than across parcels --
    # parcels within a county are not independent draws and would give a spuriously tight one.
    per_seed = joined.groupby(["seed", "county", "county_name"], observed=True).agg(
        share_converted=("converted", "mean"),
        share_leasehold=("leasehold", "mean"),
        mean_holding=("final_holding", "mean"),
        mean_conversion_period=(
            "first_conversion",
            lambda s: float(np.mean(s[s >= 0])) if (s >= 0).any() else np.nan,
        ),
    ).reset_index()

    stats = per_seed.groupby(["county", "county_name"], observed=True).agg(["mean", "std", "count"])
    county_mean = per_seed.groupby("county", observed=True).mean(numeric_only=True)

    parcel_county = geography.set_index("parcel")["county"]
    panels = [
        ("share_converted", "Share of land converted", (0.0, 1.0), "share"),
        ("share_leasehold", "Land under leasehold at the end", (0.0, 1.0), "share"),
        ("mean_conversion_period", "Mean period of conversion", (None, None), "period"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.6))
    for ax, (column, title, (lo, hi), label) in zip(axes, panels):
        # Broadcast the county value back onto every parcel of that county.
        values = parcel_county.map(county_mean[column])
        image = _draw(ax, geography, values, cmap=SEQUENTIAL, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label(label, color=INK_SOFT, fontsize=8)
        bar.outline.set_visible(False)
    fig.suptitle(
        "County outcomes, mean across seeds (descriptive; see the CSV for intervals)",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )

    flat = stats.copy()
    flat.columns = [f"{a}_{b}" for a, b in flat.columns]
    flat = flat.reset_index()
    for column, *_ in panels:
        n = flat[f"{column}_count"].clip(lower=1)
        flat[f"{column}_ci"] = 1.96 * flat[f"{column}_std"] / np.sqrt(n)
    return _finish(fig, axes, outdir, name, flat, datadir)


# ---------------------------------------------------------------------------------------------
# 5. the spread film
# ---------------------------------------------------------------------------------------------
def fig_spread_film(
    county_history: pd.DataFrame,
    geography: pd.DataFrame,
    outdir: Path,
    name: str = "map_spread_film",
    datadir: Path | None = None,
    column: str = "share_leasehold",
    count: int = 6,
    label: str = "share of county land under leasehold",
) -> Path:
    """The transition as a sequence of county maps, averaged over seeds.

    The most direct answer available to RQ2's question, and the one a reader can check by eye: a
    transition that begins somewhere and travels looks different from one that deepens everywhere
    at once, and no summary statistic makes that as plain as six panels in a row. The variogram is
    the measurement; this is the thing the variogram is measuring.

    Counties rather than parcels because the panels are small and a per-parcel field at this size
    reads as noise; the parcel-level version of the same question is
    :func:`fig_hazard_map`.
    """
    _style()
    if column not in county_history.columns:
        raise KeyError(f"county_history has no column {column!r}")
    steps = sorted(county_history["t"].unique())
    times = [steps[int(round(f * (len(steps) - 1)))] for f in np.linspace(1 / count, 1.0, count)]

    parcel_county = geography.set_index("parcel")["county"]
    fields = [
        parcel_county.map(
            county_history[county_history["t"] == t]
            .groupby("county", observed=True)[column]
            .mean()
        )
        for t in times
    ]
    limit = _shared_limit(fields)

    fig, axes = plt.subplots(1, len(times), figsize=(2.5 * len(times), 3.9))
    axes = np.atleast_1d(axes)

    rows, image = [], None
    for ax, t, values in zip(axes, times, fields):
        image = _draw(ax, geography, values, cmap=SEQUENTIAL, vmin=0.0, vmax=limit)
        ax.set_title(f"t = {t}", fontsize=10)
        table = (
            county_history[county_history["t"] == t]
            .groupby("county", observed=True)[column]
            .mean()
            .rename(column)
            .reset_index()
        )
        table["t"] = t
        rows.append(table)

    _colorbar(fig, image, axes, f"{label} (mean of seeds)")
    fig.suptitle(
        "Does the transition travel, or deepen everywhere at once?",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, pd.concat(rows, ignore_index=True), datadir)


# ---------------------------------------------------------------------------------------------
# 6. the ecology confound, as a scatter rather than a map
# ---------------------------------------------------------------------------------------------
def fig_fertility_confound(
    parcels: pd.DataFrame,
    geography: pd.DataFrame,
    outdir: Path,
    name: str = "map_fertility_confound",
    datadir: Path | None = None,
) -> Path:
    """Is early conversion simply where the good land is? RQ6's confound, made visible.

    Left: the model's only spatially patterned input, for reference. Right: conversion timing
    against carrying capacity, binned, with the cross-seed interval. A flat right-hand panel means
    soil does not set the timing and the spatial pattern belongs to the class mechanisms; a sloped
    one means Moore's charge has purchase, established using Brenner's own machinery.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})

    image = _draw(
        axes[0], geography, geography.set_index("parcel")["phi_bar"], cmap=SEQUENTIAL
    )
    axes[0].set_title("Carrying capacity $\\bar\\phi$ (the input)", fontsize=10)
    bar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02)
    bar.set_label("$\\bar\\phi$", color=INK_SOFT, fontsize=8)
    bar.outline.set_visible(False)

    joined = parcels.merge(geography[["parcel", "phi_bar"]], on="parcel")
    converted = joined[joined["first_conversion"] >= 0]
    ax = axes[1]
    table = pd.DataFrame()
    if len(converted) > 20:
        edges = np.quantile(joined["phi_bar"], np.linspace(0, 1, 11))
        edges = np.unique(edges)
        bins = pd.cut(converted["phi_bar"], edges, include_lowest=True)
        # Per seed within bin first: parcels are not independent, seeds are.
        per_seed = converted.assign(bin=bins).groupby(
            ["bin", "seed"], observed=True
        )["first_conversion"].mean().reset_index()
        grouped = per_seed.groupby("bin", observed=True)["first_conversion"]
        mean, count = grouped.mean(), grouped.count().clip(lower=1)
        sem = grouped.std(ddof=1) / np.sqrt(count)
        centres = [interval.mid for interval in mean.index]
        ax.fill_between(
            centres, mean - 1.96 * sem, mean + 1.96 * sem,
            color=SERIES[0], alpha=0.18, linewidth=0,
        )
        ax.plot(centres, mean.values, color=SERIES[0], marker="o", markersize=5)
        clean = per_seed.dropna()
        if len(clean) > 2 and np.ptp(clean["bin"].cat.codes) > 0:
            slope = np.polyfit(
                [i.mid for i in clean["bin"]], clean["first_conversion"], 1
            )[0]
            ax.annotate(
                f"slope = {slope:+.2f} periods per unit $\\bar\\phi$",
                xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=9.5, color=INK,
            )
        table = pd.DataFrame(
            {"phi_bar_bin_mid": centres, "mean_conversion_period": mean.values,
             "ci": (1.96 * sem).values, "n_seeds": count.values}
        )
    ax.set_xlabel("carrying capacity $\\bar\\phi$ of the parcel")
    ax.set_ylabel("mean period of first conversion")
    ax.set_title("Does better land convert earlier?", fontsize=10)

    fig.suptitle(
        "RQ6  Ecology or class structure: the confound stated directly",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    return _finish(fig, axes, outdir, name, table, datadir)


# ---------------------------------------------------------------------------------------------
# 7. two fronts, or one
# ---------------------------------------------------------------------------------------------
def fig_enclosure_fronts(
    parcels: pd.DataFrame,
    geography: pd.DataFrame,
    outdir: Path,
    name: str = "map_enclosure_fronts",
    datadir: Path | None = None,
) -> Path | None:
    """RQ10 asked about geography rather than only timing: do the two fronts travel together?

    Only available under ``enclosure_rule="local"`` -- with the paper's national rule every estate
    encloses at the same moment and the right-hand map is one flat colour, which the panel says
    outright rather than leaving as a puzzle.

    **Read the correlation with the asymmetry in mind.** The two processes share a driver, and the
    model has no field-system layer to give enclosure a geography of its own, so fronts that
    coincide are close to a construction and carry little weight. Fronts that *diverge* despite
    the shared driver are the informative case, and they support Shaw-Taylor's separability
    against Wood and Neeson.
    """
    _style()
    if "enclosure_half_time" not in parcels.columns:
        return None

    def censored_mean(values: pd.Series) -> float:
        """Mean over the seeds that reached the event, NaN where none did.

        Both columns use -1 for "had not happened by the end of the run", which is a censored
        observation and not a period. Averaging it in would place never-enclosed land at the
        *early* end of the ramp -- the exact opposite of what it means -- so it is dropped and
        the cell left blank instead.
        """
        reached = values[values >= 0]
        return float(reached.mean()) if len(reached) else np.nan

    per_parcel = parcels.groupby("parcel").agg(
        enclosure_half_time=("enclosure_half_time", censored_mean),
        conversion=("first_conversion", censored_mean),
    )
    uniform = float(np.nanstd(per_parcel["enclosure_half_time"])) < 1e-9

    fig, axes = plt.subplots(
        1, 3, figsize=(14.6, 4.6),
        gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.45},
    )
    for ax, (column, title, label) in zip(
        axes[:2],
        [
            ("conversion", "Rent conversion", "mean period of first conversion"),
            ("enclosure_half_time", "Enclosure", "period Xi passed one half"),
        ],
    ):
        image = _draw(ax, geography, per_parcel[column], cmap=SEQUENTIAL)
        ax.set_title(title, fontsize=10)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label(label, color=INK_SOFT, fontsize=8)
        bar.outline.set_visible(False)
    if uniform:
        axes[1].set_title("Enclosure (national rule: one date everywhere)", fontsize=9.5)
    never = int(per_parcel["enclosure_half_time"].isna().sum())
    if never:
        axes[1].text(
            0.02, -0.04, f"blank: {never} parcels never passed one half",
            transform=axes[1].transAxes, fontsize=7.5, color=INK_SOFT, va="top",
        )

    ax = axes[2]
    clean = per_parcel.dropna()
    if uniform or len(clean) < 10:
        ax.text(
            0.5, 0.5,
            "no enclosure geography to compare:\nenclosure_rule = \"national\"",
            ha="center", va="center", fontsize=9.5, color=INK_SOFT, transform=ax.transAxes,
        )
        _bare(ax)
    else:
        ax.scatter(
            clean["enclosure_half_time"], clean["conversion"],
            s=6, alpha=0.2, color=SERIES[0], edgecolors="none",
        )
        r = float(np.corrcoef(clean["enclosure_half_time"], clean["conversion"])[0, 1])
        slope, intercept = np.polyfit(clean["enclosure_half_time"], clean["conversion"], 1)
        xs = np.linspace(clean["enclosure_half_time"].min(), clean["enclosure_half_time"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=SERIES[1], linewidth=2.2)
        ax.annotate(
            f"r = {r:+.2f}\nslope = {slope:+.2f}"
            + ("\nfronts travel together" if abs(r) > 0.5 else "\nfronts are separable"),
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=9.5, color=INK,
        )
        ax.set_xlabel("period enclosure passed one half")
        ax.set_ylabel("mean period of first conversion")
    ax.set_title("Do they travel together?", fontsize=10)

    fig.suptitle(
        "RQ10  Enclosure and rent conversion: one process or two?",
        y=1.0, fontsize=11.5, fontweight="bold", color=INK,
    )
    table = per_parcel.reset_index().merge(
        geography[["parcel", "row", "col", "county_name"]], on="parcel"
    )
    return _finish(fig, axes, outdir, name, table, datadir)
