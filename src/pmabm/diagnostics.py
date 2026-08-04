"""Diagnostic dashboards for understanding the model's internal dynamics.

The point of these figures is not to answer the research questions -- ``plots.py`` does that --
but to make the machinery visible, so that behaviour which is merely a *consequence of a simple
assumption* can be told apart from behaviour that is genuinely *emergent*. Each panel therefore
carries a short note in its caption saying which of the two it is likely to be.

The same figure definitions serve both the single-run and multi-seed scripts. They differ only
in how a series is drawn -- one line, or a mean with a 95% confidence band across seeds -- so
that is the only thing passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import GRID, INK, INK_SOFT, SERIES, SURFACE, _style

#: Draws one series onto an axis. ``frames`` is a long frame with a ``t`` column and one row
#: per (seed, t); the renderer decides whether to show every seed, a mean, or a band.
Renderer = Callable[[plt.Axes, pd.DataFrame, str, str, str], None]


def single_run_renderer(ax, frame: pd.DataFrame, column: str, colour: str, label: str) -> None:
    """One run: plot the series as it happened, with no aggregation."""
    if column not in frame:
        return
    data = frame[["t", column]].dropna()
    ax.plot(data["t"], data[column], color=colour, label=label)


def mean_ci_renderer(ax, frame: pd.DataFrame, column: str, colour: str, label: str) -> None:
    """Many seeds: mean with a 95% confidence interval on the mean.

    The band is ``+/- 1.96 * SE`` rather than a percentile spread, so it answers "where is the
    average behaviour" rather than "how variable is any single run".
    """
    if column not in frame:
        return
    grouped = frame.groupby("t")[column]
    mean = grouped.mean()
    count = grouped.count()
    sem = grouped.std(ddof=1) / np.sqrt(count.clip(lower=1))
    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
    ax.fill_between(mean.index, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.plot(mean.index, mean.values, color=colour, label=label)


@dataclass
class Panel:
    title: str
    ylabel: str
    series: list[tuple[str, str]]
    """(column, legend label) pairs. Colours are assigned in palette slot order."""
    note: str = ""
    """Whether to read the panel as an assumption's shadow or as emergent behaviour."""
    logy: bool = False
    ylim: tuple[float, float] | None = None


@dataclass
class Figure:
    name: str
    title: str
    panels: list[Panel]
    ncols: int = 3
    caption: str = ""
    panel_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# The dashboards
# ---------------------------------------------------------------------------------------------
FIGURES: list[Figure] = [
    Figure(
        name="dyn_tenure",
        title="Tenure composition and the pace of conversion",
        caption="Population-level state of the transition.",
        panels=[
            Panel(
                "Agents by tenure state",
                "agents",
                [("customary", "Customary"), ("leasehold", "Leasehold"),
                 ("freehold", "Freehold"), ("landless", "Landless")],
                note="EMERGENT: the mix is an outcome, only Customary is seeded.",
            ),
            Panel(
                "Parcels whose tenure has converted",
                "parcels",
                [("parcels_leasehold_tenure", "leasehold tenure"),
                 ("converted_parcels", "ever converted")],
                note="ASSUMPTION-LED: tenure ratchets, so these can only rise.",
            ),
            Panel(
                "Conversion events per period",
                "events",
                [("conversions_at_vacancy", "at a vacancy"),
                 ("conversions_inplace", "in place (fine escalation)"),
                 ("freehold_diversions", "diverted to freehold")],
                note="EMERGENT: timing follows fiscal pressure and neighbours.",
            ),
        ],
    ),
    Figure(
        name="dyn_wealth",
        title="Wealth: who accumulates, and how unequally",
        caption="Wealth by class and the spread within the tenant population.",
        panels=[
            Panel(
                "Mean wealth by tenure",
                "wealth",
                [("customary_wealth", "Customary"), ("leasehold_wealth", "Leasehold"),
                 ("freehold_wealth", "Freehold"), ("landless_wealth_median", "Landless (median)")],
                note="EMERGENT: divergence between classes is not imposed.",
            ),
            Panel(
                "Spread of tenant wealth",
                "wealth",
                [("tenant_wealth_p10", "10th pct"), ("tenant_wealth_median", "median"),
                 ("tenant_wealth_p90", "90th pct"), ("tenant_wealth_max", "max")],
                note="EMERGENT: a fan-out here is the accumulation dynamic working.",
            ),
            Panel(
                "Landlord wealth",
                "wealth",
                [("landlord_wealth_p10", "10th pct"), ("landlord_wealth_median", "median"),
                 ("landlord_wealth_p90", "90th pct")],
                note="ASSUMPTION-LED: W(t+1)=W(t)-Delta(t), an inferred equation.",
            ),
        ],
    ),
    Figure(
        name="dyn_improvement",
        title="Improvement: disposition, investment and the capital stock",
        caption="The mechanism at the heart of RQ1.",
        panels=[
            Panel(
                "Improving disposition by tenure",
                "$\\iota^T$",
                [("customary_iota", "Customary"), ("leasehold_iota", "Leasehold"),
                 ("freehold_iota", "Freehold")],
                note="ASSUMPTION-LED: Customary is pinned at zero by construction.",
                ylim=(0, 1),
            ),
            Panel(
                "Capital stock by tenure",
                "$k_j$",
                [("customary_capital", "Customary"), ("leasehold_capital", "Leasehold"),
                 ("freehold_capital", "Freehold")],
                note="EMERGENT: the gap follows from disposition plus depreciation.",
                logy=True,
            ),
            Panel(
                "Investment flow per period",
                "capital added",
                [("investment", "gross investment")],
                note="EMERGENT: rises and falls with surplus, not scheduled.",
            ),
        ],
    ),
    Figure(
        name="dyn_production",
        title="Production and land",
        caption="Output, soil condition, and how hard the land is being worked.",
        panels=[
            Panel(
                "Aggregate output",
                "$Y(t)$",
                [("output", "total output")],
                note="EMERGENT: joint product of fertility, capital and labour.",
            ),
            Panel(
                "Mean output per tenant by tenure",
                "$y_j$",
                [("customary_output", "Customary"), ("leasehold_output", "Leasehold"),
                 ("freehold_output", "Freehold")],
                note="MIXED: leasehold farms are larger, so compare with holding size.",
            ),
            Panel(
                "Land fertility",
                "$\\phi_k$",
                [("phi_p10", "10th pct"), ("mean_phi", "mean"), ("phi_p90", "90th pct")],
                note="ASSUMPTION-LED: logistic regrowth toward a fixed ceiling.",
            ),
        ],
    ),
    Figure(
        name="dyn_productivity",
        title="Productivity of the land and of the farmer",
        caption="The quantity the whole argument turns on.",
        panels=[
            Panel(
                "Land productivity",
                "output per parcel worked",
                [("output_per_parcel", "all tenures"),
                 ("customary_output_per_parcel", "Customary"),
                 ("leasehold_output_per_parcel", "Leasehold"),
                 ("freehold_output_per_parcel", "Freehold")],
                note="EMERGENT: rises only if capital accumulates faster than soil depletes.",
            ),
            Panel(
                "Labour productivity",
                "output per worker",
                [("output_per_worker", "all tenures"),
                 ("customary_output_per_worker", "Customary"),
                 ("leasehold_output_per_worker", "Leasehold"),
                 ("freehold_output_per_worker", "Freehold")],
                note="EMERGENT: this is what Brenner says frees people to leave the land.",
            ),
            Panel(
                "Return on capital, and soil depletion",
                "ratio",
                [("output_per_capital", "output per unit capital"),
                 ("phi_depletion", "fertility depleted vs ceiling")],
                note="MIXED: diminishing returns are assumed; the depletion path is emergent.",
            ),
        ],
    ),
    Figure(
        name="dyn_domestic_market",
        title="The domestic market created by dispossession",
        caption="Wood's loop: productivity throws people off the land, and they become its market.",
        panels=[
            Panel(
                "Who is left on the land",
                "agents",
                [("agricultural_population", "working the land"),
                 ("urban_population", "left for industry"),
                 ("landless", "landless but still rural")],
                note="EMERGENT: the split is an outcome of eviction and exit rates.",
            ),
            Panel(
                "Urban demand against marketed output",
                "units of produce",
                [("urban_demand", "urban demand"), ("marketed_output", "marketed output")],
                note="EMERGENT: only market-exposed holdings sell; custom is eaten at home.",
            ),
            Panel(
                "Price of produce",
                "price",
                [("goods_price", "produce price")],
                note="EMERGENT: rises when the towns outgrow what the farms sell them.",
            ),
        ],
    ),
    Figure(
        name="dyn_rents",
        title="Rent, fines and the landlord's income",
        caption="Where the lord's money comes from, and how the two rent regimes diverge.",
        panels=[
            Panel(
                "Mean rent paid by tenure",
                "rent per tenant",
                [("customary_rent", "Customary (real)"), ("leasehold_rent", "Leasehold")],
                note="MIXED: erosion is assumed, the leasehold level is auctioned.",
            ),
            Panel(
                "Landlord receipts and pressure",
                "per landlord",
                [("landlord_receipts_mean", "total receipts"),
                 ("landlord_fines_mean", "of which arbitrary fines")],
                note="EMERGENT: the fine/rent mix shifts as tenure converts.",
            ),
            Panel(
                "Relative fiscal pressure",
                "$\\Delta_i / C_i$",
                [("mean_fiscal_pressure_rel", "mean across estates")],
                note="ASSUMPTION-LED: driven by the inflation rate.",
            ),
        ],
    ),
    Figure(
        name="dyn_labour",
        title="The labour market",
        caption="Demand, the reserve army, and how many hands a farm commands.",
        panels=[
            Panel(
                "Demand against the landless pool",
                "persons",
                [("labour_demand", "demand"), ("labour_pool", "landless pool"),
                 ("hired_total", "actually hired")],
                note="EMERGENT: demand responds to the wage via marginal product.",
            ),
            Panel(
                "Wage",
                "$\\omega(t)$",
                [("wage", "wage")],
                note="EMERGENT: tatonnement outcome, but sensitive to its cap.",
                logy=True,
            ),
            Panel(
                "Wage labourers per employer",
                "labourers",
                [("labourers_per_leasehold_tenant", "per leasehold tenant"),
                 ("labourers_per_landlord", "per landlord estate")],
                note="EMERGENT: this is the triad forming, if it forms.",
            ),
        ],
    ),
    Figure(
        name="dyn_concentration",
        title="Concentration of holdings",
        caption="Whether farms are merging, and how far.",
        panels=[
            Panel(
                "Gini of holding size",
                "Gini",
                [("farm_gini", "Gini")],
                note="EMERGENT: engrossment plus differential survival.",
                ylim=(0, 1),
            ),
            Panel(
                "Holding size distribution",
                "parcels per farm",
                [("holding_median", "median"), ("holding_p90", "90th pct"),
                 ("holding_max", "largest farm")],
                note="EMERGENT, but check the magnitude against history.",
            ),
            Panel(
                "Mean holding by tenure",
                "parcels",
                [("customary_holding", "Customary"), ("leasehold_holding", "Leasehold"),
                 ("freehold_holding", "Freehold")],
                note="ASSUMPTION-LED for Customary: it cannot exceed one parcel.",
            ),
        ],
    ),
    Figure(
        name="dyn_population",
        title="Population flows",
        caption="Who is born, who leaves, and whether the countryside can still staff itself.",
        panels=[
            Panel(
                "Persons and households",
                "count",
                [("population", "persons alive"), ("households", "households"),
                 ("population_landless", "landless persons")],
                note="EMERGENT: births and deaths are per-household hazards.",
            ),
            Panel(
                "Cumulative departures",
                "households",
                [("exited", "exited to industry"), ("deceased", "superseded or extinct")],
                note="EMERGENT: exit is a hazard on the depth of a deficit.",
            ),
            Panel(
                "Vital rates per period",
                "persons",
                [("births", "births"), ("deaths", "deaths"),
                 ("natural_increase", "natural increase")],
                note="EMERGENT: zero surplus is calibrated to replacement.",
            ),
            Panel(
                "Turnover events per period",
                "events",
                [("evictions", "evictions"), ("successions", "conveyances"),
                 ("engrossments", "engrossments"), ("exits_to_industry", "exits"),
                 ("extinctions", "lines extinct")],
                note="EMERGENT: rates respond to ecological and market stress.",
            ),
        ],
        ncols=2,
    ),
    Figure(
        name="dyn_demography",
        title="Household demography and proletarianisation",
        caption="Whether the household is the unit that grows, and where its surplus members go.",
        panels=[
            Panel(
                "Household size",
                "members",
                [("mean_household_size", "mean"),
                 ("occupied_household_size", "landholding"),
                 ("landless_household_size", "landless"),
                 ("max_household_size", "largest")],
                note="EMERGENT: land quality sets how many members a holding carries.",
            ),
            Panel(
                "Per-head surplus by class",
                "surplus / subsistence",
                [("surplus_per_head_occupied", "landholding"),
                 ("surplus_per_head_landless", "landless")],
                note="EMERGENT: what the vital rates read. Zero is replacement.",
            ),
            Panel(
                "Proletarianisation",
                "share of persons",
                [("share_landless_persons", "landless"),
                 ("non_agricultural_share", "left the land")],
                note="EMERGENT: per person, not per household.",
                ylim=(0, 1),
            ),
            Panel(
                "Flows into wage labour",
                "events",
                [("partitions", "households shedding"),
                 ("partition_persons", "persons shed"),
                 ("disinherited", "disinherited at conveyance")],
                note="EMERGENT: zero disinherited unless impartible_inheritance is on.",
            ),
            Panel(
                "Can the land be staffed?",
                "share of parcels",
                [("share_parcels_vacant", "vacant parcels")],
                note="DIAGNOSTIC: if this climbs, the population has collapsed.",
                ylim=(0, 1),
            ),
        ],
        ncols=2,
    ),
    Figure(
        name="dyn_ideology",
        title="Ideology, resistance and enclosure",
        caption="The institutional layer.",
        panels=[
            Panel(
                "Landlord improving disposition",
                "$\\iota_i$",
                [("mean_iota_landlord", "mean across landlords")],
                note="EMERGENT under ideology_rule='material': the share of a lord's income "
                     "already coming from market rent. Nothing diffuses.",
                ylim=(0, 1),
            ),
            Panel(
                "Effective resistance and observation",
                "index",
                [("landlord_theta_eff", "$\\theta^{eff}$"), ("landlord_observed", "$\\bar I_i$")],
                note="MIXED: theta is a scenario parameter, the signal is emergent.",
            ),
            Panel(
                "Enclosure of the commons",
                "$\\Xi(t)$",
                [("enclosure", "share enclosed")],
                note="ASSUMPTION-LED: a diffusion equation with a scheduled jump at t*.",
                ylim=(0, 1),
            ),
        ],
    ),
]


def render(
    frame: pd.DataFrame, renderer: Renderer, outdir: Path, prefix: str = ""
) -> list[Path]:
    """Draw every dashboard in :data:`FIGURES` from a long ``frame`` with a ``t`` column."""
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    for spec in FIGURES:
        n = len(spec.panels)
        ncols = min(spec.ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.3 * ncols, 3.7 * nrows), squeeze=False
        )
        flat = axes.ravel()

        for ax, panel in zip(flat, spec.panels):
            drawn = 0
            for colour, (column, label) in zip(SERIES, panel.series):
                if column in frame.columns and frame[column].notna().any():
                    renderer(ax, frame, column, colour, label)
                    drawn += 1
            ax.set_title(panel.title)
            ax.set_ylabel(panel.ylabel)
            ax.set_xlabel("period")
            if panel.logy:
                ax.set_yscale("symlog")
            if panel.ylim:
                ax.set_ylim(*panel.ylim)
            if drawn > 1:
                ax.legend(loc="best")
            if panel.note:
                ax.text(
                    0.0, -0.30, panel.note, transform=ax.transAxes,
                    fontsize=7.5, color=INK_SOFT, va="top", wrap=True,
                )
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        for ax in flat[n:]:
            ax.set_visible(False)

        fig.suptitle(spec.title, y=1.02, fontsize=12, fontweight="bold", color=INK)
        fig.tight_layout()
        path = outdir / f"{prefix}{spec.name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=180, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    return written


def figure_index() -> pd.DataFrame:
    """Tabular index of every dashboard panel, for the README and for orientation."""
    rows = []
    for spec in FIGURES:
        for panel in spec.panels:
            rows.append(
                {
                    "figure": spec.name,
                    "panel": panel.title,
                    "series": ", ".join(label for _, label in panel.series),
                    "reading": panel.note,
                }
            )
    return pd.DataFrame(rows)
