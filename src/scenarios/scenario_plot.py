"""Research-question comparison figures for the scenario suite.

One function per research question of ``paper/main.tex``, plus the complexity ladder and the
model-facing checks. Figure filenames match the ``% figures/...`` comments in the paper's
Results section, so a figure can be traced from the text to the code that drew it and back.

Two rules carried over from :mod:`pmabm.plots`, both load-bearing:

* every series is a mean across seeds with a 95% confidence interval **on that mean**, and where
  the interval is narrow while individual runs diverge the per-seed panel is drawn beside it --
  a tight interval on a bimodal distribution is a fact about arithmetic, not about the model;
* no figure uses a second y-axis, and every categorical figure writes its underlying table to
  CSV beside the image, because two hues of the reference palette sit below 3:1 on a light
  surface and cannot be relied on alone.

Arms that a group did not run are skipped rather than faked, so a partial suite
(``--only rq3``) produces a partial figure set instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmabm import maps
from pmabm.plots import (
    GRID,
    INK,
    INK_SOFT,
    PALETTE_EXTENDED,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    _finish,
    _style,
)
from pmabm.stats import (
    paired_delta,
    paired_delta_table,
    regime_shares,
    transition_probability,
)

#: Extra hues for groups with more arms than the four-slot categorical palette. Defined once in
#: :mod:`pmabm.plots` so the sensitivity figures draw from the same list.
PALETTE = PALETTE_EXTENDED


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _colours(n: int) -> list[str]:
    """``n`` colours, cycling the palette if a group has more arms than it has hues.

    Always use this rather than ``zip(PALETTE, arms)``: zip stops at the shorter argument, so
    an eighth arm added to a group would silently vanish from the figure while still appearing
    in the table beside it. Recycling a hue is a legibility problem the direct labels already
    mitigate; dropping an arm is a correctness problem that nothing catches.
    """
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _band(ax, frame: pd.DataFrame, column: str, colour: str, label: str, by: str = "t") -> None:
    """Mean across seeds with a 95% CI ribbon on the mean."""
    if column not in frame:
        return
    grouped = frame.groupby(by)[column]
    mean, count = grouped.mean(), grouped.count().clip(lower=1)
    sem = grouped.std(ddof=1) / np.sqrt(count)
    ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                    color=colour, alpha=0.18, linewidth=0)
    ax.plot(mean.index, mean.values, color=colour, label=label)


def _final_bar(ax, arms: dict, column: str, title: str, ylabel: str) -> pd.DataFrame:
    """Final-period value of ``column`` per arm, mean across seeds with a 95% CI whisker."""
    rows = []
    for label, frames in arms.items():
        values = frames["summary"][column].dropna()
        if values.empty:
            continue
        ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append({"arm": label, "mean": values.mean(), "ci": ci, "n": len(values)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    x = np.arange(len(frame))
    ax.bar(x, frame["mean"], yerr=frame["ci"], capsize=3, color=_colours(len(frame)),
           edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["arm"], rotation=28, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return frame


def _arm_frames(group_result: dict, arms: list) -> dict:
    """Map display label -> frames, in the order the scenario file declares."""
    return {
        arm.label: group_result[arm.name]
        for arm in arms
        if arm.name in group_result
    }


def _suptitle(fig, text: str) -> None:
    fig.suptitle(text, y=1.05, fontsize=11.5, fontweight="bold", color=INK)


def _sweep_frame(group_result: dict, arms: list) -> pd.DataFrame:
    """Per-seed summaries of a swept group, with the swept value as a numeric column."""
    parts = [
        group_result[arm.name]["summary"]
        for arm in arms
        if arm.name in group_result and arm.sweep_param
    ]
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["sweep_value"] = pd.to_numeric(frame["sweep_value"], errors="coerce")
    return frame.dropna(subset=["sweep_value"])


def _sweep_line(ax, frame: pd.DataFrame, column: str, colour: str, label: str) -> None:
    """Mean and 95% CI of ``column`` against the swept parameter value."""
    grouped = frame.groupby("sweep_value")[column]
    mean, count = grouped.mean(), grouped.count().clip(lower=1)
    sem = grouped.std(ddof=1) / np.sqrt(count)
    ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                    color=colour, alpha=0.18, linewidth=0)
    ax.plot(mean.index, mean.values, color=colour, marker="o", markersize=4, label=label)


# ---------------------------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------------------------
def fig_ladder(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Complexity ladder: what each mechanism adds, tier by tier."""
    _style()
    arms = _arm_frames(result, group.arms)
    panels = [
        ("share_leasehold", "Leasehold share of tenancies", "share"),
        ("farm_gini", "Farm-size Gini", "Gini"),
        ("share_landless", "Landless share", "share"),
        ("output", "Aggregate output $Y(t)$", "output"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), sharex=True)
    for ax, (column, title, ylabel) in zip(axes, panels):
        for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
            _band(ax, frames["history"], column, colour, label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[0].legend(loc="upper left", fontsize=7.5)
    _suptitle(fig, "The complexity ladder: each tier adds one mechanism to the tier before it")
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "ladder_complexity", combined, datadir)


# ---------------------------------------------------------------------------------------------
# RQ1 -- competitive tenancy or secure property
# ---------------------------------------------------------------------------------------------
def fig_rq1_tenure(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Leasehold against Freehold on the same land: Brenner's symbiosis against Allen's yeoman."""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(5.4 * len(arms), 4.0), squeeze=False)
    for ax, (label, frames) in zip(axes[0], arms.items()):
        history = frames["history"]
        for colour, (column, series) in zip(
            SERIES,
            [("leasehold_iota", "Leasehold"), ("freehold_iota", "Freehold"),
             ("customary_iota", "Customary")],
        ):
            _band(ax, history, column, colour, series)
        ax.set_title(label)
        ax.set_xlabel("period")
        ax.set_ylabel("improving disposition $\\iota^T_j$")
    axes[0][0].legend(loc="upper left", title="tenure", fontsize=8)
    _suptitle(
        fig,
        "RQ1  Does improvement need competitive tenancy, or only market exposure? "
        "Brenner predicts Leasehold leads; Allen predicts Freehold does",
    )
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "rq1_freehold_vs_leasehold", combined, datadir)


def fig_rq1_capital(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The same comparison in capital and output per parcel, where the accumulation shows."""
    _style()
    arms = _arm_frames(result, group.arms)
    panels = [
        ("capital", "Capital $k_j$"),
        ("output_per_parcel", "Output per parcel"),
        ("wealth", "Wealth $w_j$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    # Only the reference arm is drawn here; the ablation lives in the figure above.
    label, frames = next(iter(arms.items()))
    for ax, (suffix, title) in zip(axes, panels):
        for colour, tenure in zip(SERIES, ("leasehold", "freehold", "customary")):
            _band(ax, frames["history"], f"{tenure}_{suffix}", colour, tenure.capitalize())
        ax.set_title(title)
        ax.set_xlabel("period")
    axes[0].set_ylabel(f"mean per tenant ({label})")
    axes[0].legend(loc="upper left", fontsize=8)
    _suptitle(fig, "RQ1  Accumulation by tenure state, baseline arm")
    return _finish(fig, axes, outdir, "rq1_accumulation", frames["history"], datadir)


def fig_rq1_event(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The within-tenure half of RQ1: parcels aligned on their own conversion date."""
    _style()
    arms = _arm_frames(result, group.arms)
    label, frames = next(iter(arms.items()))
    events = frames["events"]
    if events.empty:
        return None
    metrics = [
        ("iota", "Improving disposition $\\iota^T$"),
        ("capital", "Capital $k_j$"),
        ("rent", "Rent paid $\\rho_j$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        subset = events[events["metric"] == metric]
        for colour, series in zip(SERIES, ("converted", "never converted")):
            g = subset[subset["group"] == series]
            if g.empty:
                continue
            _band(ax, g.rename(columns={"mean": "value"}), "value", colour, series,
                  by="event_time")
        ax.axvline(0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_title(title)
        ax.set_xlabel("periods relative to conversion")
    axes[0].set_ylabel("mean across parcels and seeds")
    axes[0].legend(loc="upper left", fontsize=8)

    # The diagnostic that decides how to read the panels: a conversion that replaces the sitting
    # tenant records a change of occupant, not a change of disposition.
    share = frames["summary"]["same_occupant_share"].dropna()
    if not share.empty:
        axes[2].text(
            0.02, 0.04,
            f"same occupant either side of conversion: {share.mean():.1%}",
            transform=axes[2].transAxes, fontsize=7.5, color=INK_SOFT,
        )
    _suptitle(fig, "RQ1  Improvement around a parcel's own conversion")
    return _finish(fig, axes, outdir, "rq1_event_study", events, datadir)


# ---------------------------------------------------------------------------------------------
# RQ2 -- spreading or simultaneous
# ---------------------------------------------------------------------------------------------
def fig_rq2_ablation(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Spatial structure of conversion, and conversion reach, under each channel combination."""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.0))

    tables = []
    for ax, (column, title, ylabel) in zip(
        axes,
        [
            ("spread_nugget_share", "Local coherence (nugget)", "share of total variance"),
            ("spread_slope_norm", "Distance-dependent lag", "variance share per cell"),
            ("conversion_share", "Share of parcels ever converted", "share"),
            ("final_share_leasehold", "Final Leasehold share", "share"),
        ],
    ):
        table = _final_bar(ax, arms, column, title, ylabel)
        table["metric"] = column
        tables.append(table)
    axes[0].axhline(1.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[0].text(
        0.02, 0.95, "1.0 = neighbours no more alike than distant parcels",
        transform=axes[0].transAxes, fontsize=7.5, color=INK_SOFT, va="top",
    )
    axes[1].axhline(0.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[1].text(
        0.02, 0.95, "zero = simultaneity", transform=axes[1].transAxes,
        fontsize=7.5, color=INK_SOFT, va="top",
    )
    _suptitle(
        fig,
        "RQ2  Do Brenner's own mechanisms spread, or fire everywhere at once? A nugget near 1 "
        "with a flat slope, while conversion still completes, is simultaneity",
    )
    return _finish(fig, axes, outdir, "rq2_channel_ablation",
                   pd.concat(tables, ignore_index=True), datadir)


def fig_rq2_distance(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The semivariogram of conversion time per channel arm, averaged across seeds.

    One curve per arm on one axis, because the whole comparison is a difference in *shape*: a
    contagious arm rises from a low nugget toward its plateau over some range, while a
    simultaneous arm sits flat at the total variance from the first bin onward.
    """
    _style()
    arms = _arm_frames(result, group.arms)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    rows = []
    for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
        curve = frames.get("variogram")
        if curve is None or curve.empty:
            continue
        grouped = curve.groupby("distance")["gamma_norm"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", markersize=3.5,
                color=colour, linewidth=1.7, alpha=0.9, label=label)
        table = grouped.reset_index()
        table["arm"] = label
        rows.append(table)
    ax.axhline(1.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        "no spatial structure", xy=(0.99, 1.0), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=7.5, color=INK_SOFT,
    )
    ax.set_xlabel("lattice distance between parcels (cells)")
    ax.set_ylabel("semivariance of conversion time / total variance")
    ax.set_ylim(0.0, None)
    ax.legend(loc="lower right", fontsize=8)
    _suptitle(fig, "RQ2  Semivariogram of conversion time by channel set (no origin assumed)")
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return _finish(fig, ax, outdir, "rq2_distance_lag", table, datadir)


# ---------------------------------------------------------------------------------------------
# RQ3 -- the state-landlord alliance
# ---------------------------------------------------------------------------------------------
def fig_rq3_alliance(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The 2x2: theta crossed with the class-conflict mechanism."""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))
    for ax, (column, title, ylabel) in zip(
        axes,
        [
            ("share_leasehold", "Leasehold share", "share"),
            ("share_freehold", "Freehold share", "share"),
            ("share_customary", "Customary share", "share"),
        ],
    ):
        for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
            _band(ax, frames["history"], column, colour, label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[0].legend(loc="upper left", fontsize=7.5)
    _suptitle(
        fig,
        "RQ3  Is the state-landlord alliance necessary, sufficient or permissive? "
        "England surviving with conflict off would make it sufficient -- Bois's charge",
    )
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "rq3_alliance_2x2", combined, datadir)


# ---------------------------------------------------------------------------------------------
# RQ4 -- how secure custom could have been
# ---------------------------------------------------------------------------------------------
def fig_rq4_security(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Marginal frontiers: where does the transition stop completing?"""
    _style()
    swept = [a for a in group.arms if a.sweep_param]
    params = sorted({a.sweep_param for a in swept})
    if not params:
        return None
    fig, axes = plt.subplots(1, len(params), figsize=(5.0 * len(params), 4.0), squeeze=False)
    tables = []
    for ax, param in zip(axes[0], params):
        frame = _sweep_frame(result, [a for a in swept if a.sweep_param == param])
        if frame.empty:
            continue
        for colour, (column, label) in zip(
            SERIES,
            [("final_share_leasehold", "Leasehold"), ("final_share_customary", "Customary")],
        ):
            _sweep_line(ax, frame, column, colour, label)
        # The frontier: the paper's completion criterion, stated on the figure rather than in
        # the caption, so the contour is legible without the text.
        ax.axhline(0.5, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel("final share of tenancies")
        summary = frame.groupby("sweep_value")[
            ["final_share_leasehold", "final_share_customary", "final_farm_gini"]
        ].mean().reset_index()
        summary["sweep_param"] = param
        tables.append(summary)
    axes[0][0].legend(loc="upper right", fontsize=8)
    _suptitle(
        fig,
        "RQ4  How secure could custom have been and England still have got there? "
        "Dashed line: Leasehold majority",
    )
    data = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return _finish(fig, axes, outdir, "rq4_custom_security", data, datadir)


# ---------------------------------------------------------------------------------------------
# RQ5 -- necessity of dispossession
# ---------------------------------------------------------------------------------------------
def fig_rq5_dispossession(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Conversion through succession and fines alone, against the full baseline."""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), sharex=True)
    for ax, (column, title, ylabel) in zip(
        axes,
        [
            ("share_leasehold", "Leasehold share", "share"),
            ("farm_gini", "Farm-size Gini", "Gini"),
            ("share_landless", "Landless share", "share"),
            ("evictions", "Evictions per period", "count"),
        ],
    ):
        for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
            _band(ax, frames["history"], column, colour, label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[0].legend(loc="upper left", fontsize=7.5)
    _suptitle(
        fig,
        "RQ5  Is dispossession necessary, or only sufficient? A transition surviving the "
        "damped arm reconciles Brenner's mechanism with Whittle's Norfolk",
    )
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "rq5_dispossession", combined, datadir)


# ---------------------------------------------------------------------------------------------
# RQ6 -- ecology or class structure
# ---------------------------------------------------------------------------------------------
def fig_rq6_ecology(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Flattening the ALC regional structure: does the spatial pattern survive?"""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))

    for ax, (column, title, ylabel) in zip(
        axes[:2],
        [
            ("share_leasehold", "Leasehold share", "share"),
            ("farm_gini", "Farm-size Gini", "Gini"),
        ],
    ):
        for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
            _band(ax, frames["history"], column, colour, label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[0].legend(loc="upper left", fontsize=8)

    table = _final_bar(
        axes[2], arms, "spread_nugget_share",
        "Local coherence (nugget)", "share of total variance",
    )
    axes[2].axhline(1.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[2].text(
        0.02, 0.95,
        "the only spatially patterned input is\nremoved in the flat arm, so coherence\nsurviving it "
        "is the mechanisms' own",
        transform=axes[2].transAxes, fontsize=7.5, color=INK_SOFT, va="top",
    )
    _suptitle(
        fig,
        "RQ6  Is ecology doing work the theory gives to class structure? Moore's charge, "
        "tested with Brenner's own mechanisms",
    )
    return _finish(fig, axes, outdir, "rq6_ecology_ablation", table, datadir)


# ---------------------------------------------------------------------------------------------
# claims about the model
# ---------------------------------------------------------------------------------------------
def fig_model_checks(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The diagnostics of the paper's "Claims about the model" section."""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))

    # 1. aggregate transition; 2. Gini attributable to engrossment; 3. wage against rent.
    for ax, (column, title, ylabel) in zip(
        axes[:3],
        [
            ("share_leasehold", "Aggregate transition", "Leasehold share"),
            ("farm_gini", "Concentration, attributed", "farm-size Gini"),
            ("share_parcels_vacant", "Parcels no household can take", "share"),
        ],
    ):
        for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
            _band(ax, frames["history"], column, colour, label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[1].text(
        0.02, 0.95,
        "a Gini rising as fast without engrossment\nis turnover, not consolidation",
        transform=axes[1].transAxes, fontsize=7.5, color=INK_SOFT, va="top",
    )
    axes[0].legend(loc="upper left", fontsize=7.5)

    # 4. does the tenure outcome track the population outcome across seeds? If it does, the
    # result is about demography rather than about property relations.
    ax = axes[3]
    label, frames = next(iter(arms.items()))
    final = frames["summary"]
    if {"final_share_leasehold", "total_population_ratio"} <= set(final.columns):
        ax.scatter(final["total_population_ratio"], final["final_share_leasehold"],
                   s=26, color=SERIES[0], alpha=0.8, edgecolor=SURFACE, linewidth=0.6)
        pair = final[["total_population_ratio", "final_share_leasehold"]].dropna()
        if len(pair) > 2 and np.ptp(pair["total_population_ratio"]) > 0:
            r = float(np.corrcoef(pair["total_population_ratio"],
                                  pair["final_share_leasehold"])[0, 1])
            ax.text(0.02, 0.95, f"r = {r:+.2f} across seeds", transform=ax.transAxes,
                    fontsize=8, color=INK_SOFT, va="top")
    ax.set_xlabel("final population / initial population")
    ax.set_ylabel("final Leasehold share")
    ax.set_title("Tenure against population, per seed")
    _suptitle(
        fig,
        "Claims about the model: a negative result in any panel is a defect to repair, "
        "not a finding to report",
    )
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "model_checks", combined, datadir)


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Cross-arm statistics: paired differences and outcome regimes
#
# Both of these replace a mean where a mean is the wrong summary, and both apply to every group
# rather than to one question, so they are driven off `group.arms` generically.
# ---------------------------------------------------------------------------------------------
#: The outcomes worth differencing between arms. Deliberately short: a forest plot of thirty
#: metrics is a table, and the point of the figure is that a reader can see which way each arm
#: moved at a glance.
PAIRED_METRICS = [
    ("final_share_leasehold", "Leasehold share"),
    ("final_farm_gini", "Farm-size Gini"),
    ("final_share_landless_persons", "Landless (persons)"),
    ("conversion_share", "Share ever converted"),
    ("spread_nugget_share", "Nugget (local coherence)"),
    ("total_population_ratio", "Population ratio"),
]

#: Which arm each group is differenced against. The first declared arm is the fallback and is the
#: right answer in most groups, but not all: RQ7's baseline is the full demography rather than the
#: closed population it happens to declare first, and reading the ladder against its bottom tier
#: rather than its top would invert every sign.
BASELINE_ARM = {
    "ladder": "t6_demography",
    "rq1": "shadow_rent_on",
    "rq2": "both_channels",
    "rq3": "england_conflict",
    "rq5": "full_dispossession",
    "rq6": "alc_fertility",
    "checks": "baseline",
}


def _baseline_of(group, result: dict):
    """The arm a group is compared against, and its label."""
    preferred = BASELINE_ARM.get(group.name)
    for arm in group.arms:
        if arm.name == preferred and arm.name in result:
            return arm
    for arm in group.arms:  # fall back to the first arm that actually ran
        if arm.name in result:
            return arm
    return None


def fig_paired_deltas(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Every arm against the group's baseline, differenced *within* seed.

    Arms share a seed list, so seed *s* differs between two arms only in the mechanism under test.
    Differencing inside the seed removes the estate layout, the initial rent draw and the shock
    sequence -- everything the two arms hold in common -- and leaves the mechanism. The interval
    here is therefore typically several times tighter than the one on either arm's own mean, and
    the ``pairing_gain`` column of the CSV records by how much.

    A point whose interval crosses the dashed zero line is an arm this suite cannot distinguish
    from its baseline, which is a result and not a gap: it is how an ablation gets reported as
    having made no difference.
    """
    _style()
    baseline = _baseline_of(group, result)
    if baseline is None:
        return None
    arms = {arm.label: result[arm.name]["summary"] for arm in group.arms if arm.name in result}
    columns = [c for c, _ in PAIRED_METRICS]
    table = paired_delta_table(arms, baseline.label, columns)
    if table.empty:
        return None

    present = [(c, t) for c, t in PAIRED_METRICS if c in set(table["metric"])]
    fig, axes = plt.subplots(1, len(present), figsize=(3.3 * len(present), 4.2), squeeze=False)
    for ax, (column, title) in zip(axes[0], present):
        block = table[(table["metric"] == column) & (table["arm_a"] != baseline.label)]
        block = block.iloc[::-1]  # declaration order top-to-bottom
        y = np.arange(len(block))
        ax.errorbar(
            block["mean_delta"], y, xerr=block["ci_paired"],
            fmt="o", markersize=6, color=SERIES[0], ecolor=INK_SOFT, elinewidth=1.2, capsize=3,
        )
        ax.axvline(0.0, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_yticks(y)
        ax.set_yticklabels(block["arm_a"], fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("difference from baseline", fontsize=8)
    _suptitle(
        fig,
        f"Paired differences against '{baseline.label}', same seeds. An interval crossing zero "
        "is an arm this suite cannot separate from its baseline",
    )
    return _finish(fig, axes, outdir, f"{group.name}_paired_deltas", table, datadir)


def fig_regimes(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """How many seeds transitioned, half-transitioned, and failed -- per arm.

    The check on every mean in this suite. An arm averaging a Leasehold share of 0.5 might have
    every seed half-converted or half its seeds fully converted and half not at all; those are
    different models and only this figure separates them. The ``bimodal`` column of the CSV flags
    arms where the mean should not be quoted without this beside it.

    Classes are cut at the paper's own completion criterion, a Leasehold majority, so
    "transitioned" here means what the paper means by the transition completing.
    """
    _style()
    arms = {arm.label: result[arm.name]["summary"] for arm in group.arms if arm.name in result}
    table = regime_shares(arms)
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(table) + 2.4))
    # An ordered class, so a sequential ramp rather than categorical hues: pale is failure, dark
    # is a completed transition, and the ordering is legible without reading the legend.
    shades = [SEQUENTIAL(0.15), SEQUENTIAL(0.5), SEQUENTIAL(0.9)]
    names = ["failed", "partial", "transitioned"]
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for shade, cls in zip(shades, names):
        width = table[f"share_{cls}"].to_numpy()
        ax.barh(y, width, left=left, color=shade, height=0.66,
                edgecolor=SURFACE, linewidth=1.0, label=cls)
        for yi, (w, l) in enumerate(zip(width, left)):
            if w > 0.08:  # direct labels, so identity never rests on the ramp alone
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if cls == "transitioned" else INK)
        left = left + width
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.arm}   (mean {row.mean:.2f})" + ("  <- bimodal" if row.bimodal else "")
         for row in table.itertuples()],
        fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of seeds")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    _suptitle(
        fig,
        "Outcome regimes per arm: how often the transition completed, rather than its mean share",
    )
    return _finish(fig, ax, outdir, f"{group.name}_regimes", table, datadir)


def fig_ladder_marginal(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """What each tier *adds*, as a paired difference from the tier below it.

    The ladder's whole claim is that a dynamic first appearing at tier *n* is attributable to the
    mechanism tier *n* introduces, and that claim is about consecutive *differences*. Plotting the
    levels leaves the reader to subtract two curves by eye and supplies no interval on the result;
    this plots the difference itself, paired within seed.
    """
    _style()
    ordered = [arm for arm in group.arms if arm.name in result]
    if len(ordered) < 2:
        return None
    columns = [c for c, _ in PAIRED_METRICS]
    rows = []
    for lower, upper in zip(ordered, ordered[1:]):
        step = paired_delta(
            result[upper.name]["summary"], result[lower.name]["summary"],
            columns, label_a=upper.label, label_b=lower.label,
        )
        if step.empty:
            continue
        step["step"] = f"{lower.label} -> {upper.label}"
        rows.append(step)
    if not rows:
        return None
    table = pd.concat(rows, ignore_index=True)

    present = [(c, t) for c, t in PAIRED_METRICS if c in set(table["metric"])]
    fig, axes = plt.subplots(1, len(present), figsize=(3.3 * len(present), 4.4), squeeze=False)
    steps = list(dict.fromkeys(table["step"]))
    for ax, (column, title) in zip(axes[0], present):
        block = (
            table[table["metric"] == column]
            .set_index("step")
            .reindex(steps)
            .dropna(subset=["mean_delta"])
        )
        y = np.arange(len(block))[::-1]
        ax.barh(y, block["mean_delta"], xerr=block["ci_paired"], height=0.62,
                color=SERIES[0], edgecolor=SURFACE, linewidth=0.8,
                error_kw={"ecolor": INK_SOFT, "elinewidth": 1.0})
        ax.axvline(0.0, color=INK_SOFT, linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(block.index, fontsize=7)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("added by this tier", fontsize=8)
    _suptitle(fig, "The complexity ladder as marginal effects: what each tier adds, paired by seed")
    return _finish(fig, axes, outdir, "ladder_marginal_effects", table, datadir)


# ---------------------------------------------------------------------------------------------
# Spatial figures
#
# Thin wrappers: the drawing lives in :mod:`pmabm.maps`, which knows nothing about scenarios, and
# these supply the frames and decide which arm plays which role. Every one is a mean across seeds
# on a shared lattice -- see the module docstring of pmabm.maps for why a single-seed map cannot
# answer any of these questions.
# ---------------------------------------------------------------------------------------------
def _spatial(result: dict, arm) -> tuple | None:
    """``(parcels, geography)`` for an arm, or ``None`` if it has no spatial frames."""
    frames = result.get(arm.name) or {}
    parcels, geography = frames.get("parcels"), frames.get("geography")
    if parcels is None or geography is None or parcels.empty or geography.empty:
        return None
    # Per-seed geography stacks one copy per seed; the maps want one row per parcel.
    if geography["parcel"].duplicated().any():
        geography = geography.drop_duplicates(subset="parcel")
    return parcels, geography


def fig_maps_baseline(group, result: dict, outdir: Path, datadir: Path) -> list[Path]:
    """The hazard field, the structure-or-luck panel and the county view, for the baseline arm."""
    arm = _baseline_of(group, result)
    if arm is None:
        return []
    spatial = _spatial(result, arm)
    if spatial is None:
        return []
    parcels, geography = spatial
    written = [
        maps.fig_hazard_map(
            parcels, geography, outdir, n_steps=arm.params.n_steps,
            name=f"{group.name}_map_hazard", datadir=datadir,
        ),
        maps.fig_conversion_uncertainty(
            parcels, geography, outdir, name=f"{group.name}_map_uncertainty", datadir=datadir
        ),
        maps.fig_county_choropleth(
            parcels, geography, outdir, name=f"{group.name}_map_counties", datadir=datadir
        ),
    ]
    county = (result.get(arm.name) or {}).get("county_history")
    if county is not None and not county.empty:
        written.append(
            maps.fig_spread_film(
                county, geography, outdir, name=f"{group.name}_map_spread_film", datadir=datadir
            )
        )
    return [w for w in written if w is not None]


def fig_maps_differences(group, result: dict, outdir: Path, datadir: Path) -> list[Path]:
    """One paired difference map per arm against the group's baseline."""
    baseline = _baseline_of(group, result)
    if baseline is None:
        return []
    base_spatial = _spatial(result, baseline)
    if base_spatial is None:
        return []
    base_parcels, geography = base_spatial

    written = []
    for arm in group.arms:
        if arm.name == baseline.name:
            continue
        other = _spatial(result, arm)
        if other is None:
            continue
        # A parcel-by-parcel difference is only meaningful on one lattice. An arm that changes L
        # or lords_per_county has its own, and differencing the two would silently compare
        # unrelated land -- so it is skipped rather than drawn wrong.
        if not other[1][["parcel", "row", "col"]].equals(geography[["parcel", "row", "col"]]):
            print(
                f"  ! {group.name}/{arm.name}: different lattice from {baseline.name}, "
                f"no difference map"
            )
            continue
        written.append(
            maps.fig_difference_map(
                other[0], base_parcels, geography, arm.label, baseline.label,
                outdir, name=f"{group.name}_map_diff_{arm.name}", datadir=datadir,
            )
        )
    return [w for w in written if w is not None]


def fig_maps_ecology(group, result: dict, outdir: Path, datadir: Path) -> list[Path]:
    """RQ6's confound: the fertility field beside conversion timing against fertility."""
    arm = _baseline_of(group, result)
    if arm is None:
        return []
    spatial = _spatial(result, arm)
    if spatial is None:
        return []
    written = maps.fig_fertility_confound(
        *spatial, outdir, name=f"{group.name}_map_fertility_confound", datadir=datadir
    )
    return [written] if written is not None else []


# ---------------------------------------------------------------------------------------------
# Two-dimensional sweeps: frontiers and phase diagrams
#
# A one-at-a-time sweep gives a curve and cannot show an interaction. These read the
# ``sweep__<param>`` columns that a multi-parameter sweep writes onto every frame, so the grid is
# recovered from the data rather than parsed out of arm names.
# ---------------------------------------------------------------------------------------------
def _sweep_axes(frames: dict) -> list[str]:
    """The swept parameter names present in a group's summary frames, in a stable order."""
    for arm_frames in frames.values():
        summary = arm_frames.get("summary")
        if summary is None:
            continue
        found = sorted(c[len("sweep__"):] for c in summary.columns if c.startswith("sweep__"))
        if found:
            return found
    return []


def _surface_table(group, result: dict, axes: list[str]) -> pd.DataFrame:
    """One row per grid cell: both axis values, P(transition) and the headline means.

    P(transition) is the share of *seeds* reaching the paper's completion criterion, with a
    binomial interval. It is the right quantity for a frontier and the mean Leasehold share is
    not: the question is whether England arrives, and averaging the share across seeds smears
    precisely the boundary the figure exists to locate.
    """
    rows = []
    for arm in group.arms:
        if arm.name not in result:
            continue
        summary = result[arm.name]["summary"]
        if not all(f"sweep__{a}" in summary.columns for a in axes):
            continue
        p, ci = transition_probability(summary)
        row = {f"{a}": float(summary[f"sweep__{a}"].iloc[0]) for a in axes}
        row.update(
            {
                "arm": arm.name,
                "p_transition": p,
                "p_transition_ci": ci,
                "n_seeds": int(len(summary)),
                "mean_share_leasehold": float(summary["final_share_leasehold"].mean()),
                "mean_farm_gini": float(summary["final_farm_gini"].mean()),
                "mean_conversion_share": float(summary["conversion_share"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _heatmap(
    ax, table: pd.DataFrame, x: str, y: str, value: str, vmin, vmax, contour_at=None,
    show_y: bool = True,
):
    """One grid panel, with an optional contour marking a named level."""
    grid = table.pivot_table(index=y, columns=x, values=value)
    image = ax.imshow(
        grid.to_numpy(), origin="lower", cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto",
        extent=(-0.5, len(grid.columns) - 0.5, -0.5, len(grid.index) - 0.5),
        interpolation="nearest",
    )
    if contour_at is not None and grid.notna().to_numpy().sum() > 3:
        # The frontier itself. Drawn on the grid rather than interpolated onto a finer mesh, so
        # its position is never more precise than the sweep that produced it.
        try:
            cs = ax.contour(
                np.arange(len(grid.columns)), np.arange(len(grid.index)), grid.to_numpy(),
                levels=[contour_at], colors=[SERIES[1]], linewidths=2.2,
            )
            ax.clabel(cs, inline=True, fmt=lambda v: f"{v:.0%}", fontsize=8)
        except (ValueError, TypeError):
            pass
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{v:g}" for v in grid.columns], fontsize=7.5, rotation=45)
    ax.set_yticks(range(len(grid.index)))
    # Tick values on every panel, but the axis *name* only on the leftmost: a repeated y label
    # lands on top of the previous panel's colourbar.
    ax.set_yticklabels([f"{v:g}" for v in grid.index] if show_y else [], fontsize=7.5)
    ax.set_xlabel(x, fontsize=9)
    if show_y:
        ax.set_ylabel(y, fontsize=9)
    ax.grid(False)
    return image


def fig_surface(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """A two-parameter sweep as a surface: where does the transition happen, and how far.

    Three panels on one grid. The first is the frontier proper -- the share of seeds reaching a
    Leasehold majority, with the 50% contour drawn on it. The second is how far the transition got
    on average, which is a different question and is included because a cell can be at P=1 with a
    bare majority or with near-total conversion. The third is concentration, which is the outcome
    the engrossment mechanism is meant to deliver and which need not follow tenure.

    How to read the contour is the whole point of the figure. A contour running parallel to one
    axis means that axis is doing nothing the other cannot do; one running diagonally means the
    two trade off, and the slope is the exchange rate between them.
    """
    _style()
    axes_names = _sweep_axes(result)
    if len(axes_names) != 2:
        return None
    table = _surface_table(group, result, axes_names)
    if table.empty or len(table) < 4:
        return None
    x, y = axes_names

    panels = [
        ("p_transition", "P(transition completes)", (0.0, 1.0), 0.5),
        ("mean_share_leasehold", "Mean final Leasehold share", (0.0, 1.0), None),
        ("mean_farm_gini", "Mean final farm-size Gini", (0.0, 1.0), None),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for i, (ax, (value, title, (lo, hi), contour)) in enumerate(zip(axs, panels)):
        image = _heatmap(ax, table, x, y, value, lo, hi, contour, show_y=i == 0)
        ax.set_title(title, fontsize=10)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.outline.set_visible(False)
    fig.text(
        0.5, -0.02,
        "Orange line: the frontier at P = 50%. Parallel to an axis means that axis is redundant; "
        "diagonal means the two trade off, and its slope is the exchange rate.",
        ha="center", fontsize=8, color=INK_SOFT,
    )
    _suptitle(fig, f"{group.name}: {x} against {y}, {int(table['n_seeds'].max())} seeds per cell")
    return _finish(fig, axs, outdir, f"{group.name}_surface", table, datadir)


def fig_accounting(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The population accounting, under each population rule.

    Not a test of whether demography drives the transition -- that question was cut from the
    paper. This asks whether the bookkeeping holds, which has to be true under every rule
    before any comparison between them is readable.
    Two failures are being looked for and both have bitten before: parcels no household can take
    up, which means an unmodelled reservoir of prospective tenants is missing, and a tenure
    outcome that tracks the population outcome across seeds, which means the result is about
    demography rather than about property relations.
    """
    _style()
    arms = _arm_frames(result, group.arms)
    if not arms:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))

    for ax, (column, title, ylabel) in zip(
        axes[:3],
        [
            ("share_parcels_vacant", "Parcels no household can take", "share"),
            ("total_population", "Total population", "persons"),
            ("share_leasehold", "Leasehold share", "share"),
        ],
    ):
        for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
            _band(ax, frames["history"], column, colour, label)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("period")
    axes[0].axhline(0.02, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
    axes[0].text(
        0.02, 0.95, "near zero is the requirement", transform=axes[0].transAxes,
        fontsize=7.5, color=INK_SOFT, va="top",
    )
    axes[0].legend(loc="upper right", fontsize=7.5)

    # The independence check, per rule rather than only for the baseline: a correlation that
    # appears under one population rule and not another is the confound this group exists to find.
    ax = axes[3]
    rows = []
    for colour, (label, frames) in zip(_colours(len(arms)), arms.items()):
        final = frames["summary"]
        pair = final[["total_population_ratio", "final_share_leasehold"]].dropna()
        if len(pair) < 3:
            continue
        ax.scatter(
            pair["total_population_ratio"], pair["final_share_leasehold"],
            s=22, color=colour, alpha=0.8, edgecolor=SURFACE, linewidth=0.6, label=label,
        )
        r = (
            float(np.corrcoef(pair["total_population_ratio"], pair["final_share_leasehold"])[0, 1])
            if np.ptp(pair["total_population_ratio"]) > 0
            else np.nan
        )
        rows.append({"arm": label, "r_population_vs_tenure": r, "n_seeds": len(pair)})
    if rows:
        text = "\n".join(f"{r['arm']}: r = {r['r_population_vs_tenure']:+.2f}" for r in rows)
        ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=7.5, color=INK_SOFT, va="top")
    ax.set_xlabel("final population / initial")
    ax.set_ylabel("final Leasehold share")
    ax.set_title("Tenure against population, per seed", fontsize=10)
    _suptitle(
        fig,
        "Population accounting under every rule: a defect to repair, not a finding to report",
    )
    table = pd.DataFrame(rows)
    return _finish(fig, axes, outdir, "accounting_checks", table, datadir)


#: Group name -> the figures it produces. A group absent from the run is skipped silently; a
#: figure that raises is reported and skipped, so one bad panel does not lose the whole suite.
FIGURES = {
    "ladder": [fig_ladder, fig_ladder_marginal],
    "rq1": [fig_rq1_tenure, fig_rq1_capital, fig_rq1_event],
    "rq2": [fig_rq2_ablation, fig_rq2_distance, fig_maps_baseline, fig_maps_differences],
    "rq3": [fig_rq3_alliance],
    "rq4": [fig_rq4_security],
    "rq5": [fig_rq5_dispossession],
    "rq6": [fig_rq6_ecology, fig_maps_baseline, fig_maps_differences, fig_maps_ecology],
    "checks": [fig_model_checks, fig_maps_baseline],
    # The two-dimensional sweeps. `structural` is deliberately given no spatial figure: its arms
    # change L, so parcel indices are not comparable between them and a difference map would be
    # differencing different lattices.
    "rq3_surface": [fig_surface],
    "rq4_frontier": [fig_surface],
    "horizon": [fig_rq4_security],
    "structural": [],
    "accounting": [fig_accounting],
}

#: Drawn for *every* group in addition to its own entry above, because both answer a question that
#: applies to any set of arms: which arms differ from the baseline once seeds are paired, and
#: whether each arm's mean is hiding a bimodal outcome.
UNIVERSAL_FIGURES = [fig_paired_deltas, fig_regimes]


def plot_all(groups, results: dict, outdir: Path, datadir: Path | None = None) -> list[Path]:
    """Draw every figure for which the run has data."""
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = datadir or outdir
    datadir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for group in groups:
        for builder in [*FIGURES.get(group.name, []), *UNIVERSAL_FIGURES]:
            try:
                path = builder(group, results[group.name], outdir, datadir)
            except Exception as exc:  # one bad panel must not lose the rest of the suite
                print(f"  ! {group.name}/{builder.__name__} failed: {exc}")
                continue
            # A builder may return one path or several: the spatial ones draw a figure per arm.
            for item in path if isinstance(path, list) else [path]:
                if item is not None:
                    written.append(item)
                    print(f"  {item.name}")
    return written
