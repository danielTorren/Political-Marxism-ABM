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

from pmabm.plots import GRID, INK, INK_SOFT, SERIES, SURFACE, _finish, _style

#: Extra hues for groups with more arms than the four-slot categorical palette. Appended rather
#: than recycled, so adjacent arms never share a colour up to seven.
PALETTE = [*SERIES, "#8b5cf6", "#c2456f", "#4a7c59"]


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
# RQ7 -- demography
# ---------------------------------------------------------------------------------------------
def fig_rq7_demography(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """The transition under each population rule, and with the class mechanism damped."""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), sharex=True)
    for ax, (column, title, ylabel) in zip(
        axes,
        [
            ("share_leasehold", "Leasehold share", "share"),
            ("total_population", "Total population", "persons"),
            ("farm_gini", "Farm-size Gini", "Gini"),
            ("share_parcels_vacant", "Parcels no household can take", "share"),
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
        "RQ7  Does demographic pressure suffice? A transition with population fixed is "
        "Brenner's claim; one requiring endogenous births is Postan's",
    )
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "rq7_demography", combined, datadir)


# ---------------------------------------------------------------------------------------------
# RQ8 -- the edges of symbiosis
# ---------------------------------------------------------------------------------------------
def fig_rq8_symbiosis(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Landlord and tenant accumulation across the extraction sweep."""
    _style()
    frame = _sweep_frame(result, group.arms)
    if frame.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))

    for colour, (column, label) in zip(
        SERIES,
        [("final_share_leasehold", "Leasehold share"), ("final_share_customary", "Customary share")],
    ):
        _sweep_line(axes[0], frame, column, colour, label)
    axes[0].set_title("Tenure outcome")
    axes[0].set_ylabel("final share")
    axes[0].legend(loc="upper right", fontsize=8)

    for colour, (column, label) in zip(
        SERIES, [("final_farm_gini", "Farm-size Gini"), ("conversion_share", "Converted share")]
    ):
        _sweep_line(axes[1], frame, column, colour, label)
    axes[1].set_title("Concentration and reach")
    axes[1].set_ylabel("value")
    axes[1].legend(loc="upper left", fontsize=8)

    # The symbiosis band proper: both sides accumulating at once. Output stands in for tenant
    # prosperity and the wage for the labour market it implies; a band exists where output is
    # still rising as extraction rises, and closes where it turns over.
    for colour, (column, label) in zip(
        SERIES, [("final_population", "Population"), ("final_wage", "Wage")]
    ):
        _sweep_line(axes[2], frame, column, colour, label)
    axes[2].set_title("Whether the goose survives")
    axes[2].set_ylabel("value")
    axes[2].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xlabel("$\\theta_{rent}$, extraction share of output")
    _suptitle(
        fig,
        "RQ8  Does symbiosis have a range? Brenner asserts the England/France distinction "
        "without locating its boundary",
    )
    data = frame.groupby("sweep_value").mean(numeric_only=True).reset_index()
    return _finish(fig, axes, outdir, "rq8_symbiosis_band", data, datadir)


# ---------------------------------------------------------------------------------------------
# RQ10 -- enclosure
# ---------------------------------------------------------------------------------------------
def fig_rq10_enclosure(group, result: dict, outdir: Path, datadir: Path) -> Path:
    """Is the landless pool there without enclosure, and does the timing separate?"""
    _style()
    arms = _arm_frames(result, group.arms)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), sharex=True)
    for ax, (column, title, ylabel) in zip(
        axes,
        [
            ("share_landless", "Landless share", "share"),
            ("enclosure", "Enclosure share $\\Xi(t)$", "share"),
            ("share_leasehold", "Leasehold share", "share"),
            ("share_persons_urban", "Cumulative exit to industry", "share of persons"),
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
        "RQ10  Enclosure as driver or consequence? A comparable landless pool without it "
        "supports Shaw-Taylor against Wood on separability",
    )
    combined = pd.concat([f["history"] for f in arms.values()], ignore_index=True)
    return _finish(fig, axes, outdir, "rq10_enclosure", combined, datadir)


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
#: Group name -> the figures it produces. A group absent from the run is skipped silently; a
#: figure that raises is reported and skipped, so one bad panel does not lose the whole suite.
FIGURES = {
    "ladder": [fig_ladder],
    "rq1": [fig_rq1_tenure, fig_rq1_capital, fig_rq1_event],
    "rq2": [fig_rq2_ablation, fig_rq2_distance],
    "rq3": [fig_rq3_alliance],
    "rq4": [fig_rq4_security],
    "rq5": [fig_rq5_dispossession],
    "rq6": [fig_rq6_ecology],
    "rq7": [fig_rq7_demography],
    "rq8": [fig_rq8_symbiosis],
    "rq10": [fig_rq10_enclosure],
    "checks": [fig_model_checks],
}


def plot_all(groups, results: dict, outdir: Path, datadir: Path | None = None) -> list[Path]:
    """Draw every figure for which the run has data."""
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = datadir or outdir
    datadir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for group in groups:
        for builder in FIGURES.get(group.name, []):
            try:
                path = builder(group, results[group.name], outdir, datadir)
            except Exception as exc:  # one bad panel must not lose the rest of the suite
                print(f"  ! {group.name}/{builder.__name__} failed: {exc}")
                continue
            if path is not None:
                written.append(path)
                print(f"  {path.name}")
    return written
