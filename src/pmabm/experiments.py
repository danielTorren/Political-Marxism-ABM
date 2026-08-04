"""Experiment definitions for the paper's three research questions plus robustness checks.

Each experiment returns tidy dataframes rather than figures, so that the same results can be
plotted, tabulated or re-analysed without re-running the model.

Runs are repeated across seeds because every mechanism in the model is stochastic; a single
run cannot distinguish a mechanism from a lucky draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params
from .geography import build as build_geography, load_artifact
from .metrics import (
    event_study,
    history_frame,
    spread_variogram,
    spread_variogram_curve,
    summary,
)
from .model import Model


@dataclass
class RunResult:
    label: str
    seed: int
    params: Params
    history: pd.DataFrame
    summary: dict
    model: Model


def run_one(params: Params, label: str = "run", artifact: dict | None = None) -> RunResult:
    """Run a single simulation and package its outputs."""
    model = Model(params, artifact=artifact).run()
    return RunResult(
        label=label,
        seed=params.seed,
        params=params,
        history=history_frame(model),
        summary=summary(model),
        model=model,
    )


def run_replicates(
    params: Params, seeds: list[int], label: str, artifact: dict | None = None
) -> list[RunResult]:
    return [run_one(params.with_(seed=s), label=label, artifact=artifact) for s in seeds]


def stack_histories(results: list[RunResult]) -> pd.DataFrame:
    """Long frame of every replicate's history, tagged by label and seed."""
    frames = []
    for r in results:
        df = r.history.copy()
        df["label"] = r.label
        df["seed"] = r.seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"label": r.label, "seed": r.seed, **r.summary} for r in results]
    )


# ---------------------------------------------------------------------------------------------
# RQ1 -- is improvement a consequence of market exposure, or a precondition of it?
# ---------------------------------------------------------------------------------------------
def rq1_improvement(
    base: Params, seeds: list[int], artifact: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event study of the improving disposition around each parcel's own conversion.

    Returns ``(event_frame, continuity_frame)``. The second is the diagnostic that matters for
    interpreting the first: it reports how often conversion actually retained the sitting
    tenant, since the paper's prediction is phrased about a tenant living through conversion
    while its vacancy rule generally replaces them.
    """
    events, continuity = [], []
    for seed in seeds:
        result = run_one(base.with_(seed=seed), label="england", artifact=artifact)
        frame = event_study(result.model)
        frame["seed"] = seed
        events.append(frame)
        continuity.append({"seed": seed, **result.summary})
    return pd.concat(events, ignore_index=True), pd.DataFrame(continuity)


# ---------------------------------------------------------------------------------------------
# RQ2 -- does conversion spread from a localised seed, and which channel drives it?
# ---------------------------------------------------------------------------------------------
#: The four on/off combinations of the two transmission channels. "none" is the null model in
#: which conversion can only be driven by each landlord's own fiscal pressure.
#:
#: There are two channels rather than three because tenant mobility was withdrawn: the sources
#: place the fight over mobility before this period, and the version in which lords bid against
#: one another to retain mobile tenants is the account Brenner argues against. Competitive
#: allocation replaced it, but governs *who holds land* rather than how conversion travels, so
#: it is not ablated here -- it is tested against concentration instead.
CHANNEL_SETS: dict[str, dict[str, bool]] = {
    "both channels": dict(channel_observation=True, channel_ideology=True),
    "observation only": dict(channel_observation=True, channel_ideology=False),
    "ideology only": dict(channel_observation=False, channel_ideology=True),
    "none": dict(channel_observation=False, channel_ideology=False),
}


def rq2_spread(
    base: Params, seeds: list[int], artifact: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ablate the two spread channels and measure the spatial structure of conversion for each.

    Returns ``(ablation_frame, curve_frame)``; the curve frame holds the binned semivariograms of
    the full-channel model, one set of rows per seed, for the illustrative panel.
    """
    rows, curves = [], []
    for name, switches in CHANNEL_SETS.items():
        for seed in seeds:
            params = base.with_(seed=seed, **switches)
            model = Model(params, artifact=artifact).run()
            spread = spread_variogram(model)
            converted = int((model.first_conversion >= 0).sum())
            rows.append(
                {
                    "channels": name,
                    "seed": seed,
                    "slope_norm": spread["slope_norm"],
                    "nugget_share": spread["nugget_share"],
                    "range_cells": spread["range_cells"],
                    "r": spread["r"],
                    "n_converted": converted,
                    "conversion_share": converted / model.geo.n_parcels,
                    **{k: v for k, v in switches.items()},
                }
            )
            if name == "both channels":
                curve = spread_variogram_curve(model)
                curve["seed"] = seed
                curves.append(curve)
    curve_frame = (
        pd.concat(curves, ignore_index=True)
        if curves
        else pd.DataFrame(columns=["distance", "gamma", "gamma_norm", "n_pairs", "seed"])
    )
    return pd.DataFrame(rows), curve_frame


def conversion_map(params: Params, artifact: dict | None = None) -> tuple[Model, np.ndarray]:
    """A single run plus its per-parcel first-conversion times, for the map panels."""
    model = Model(params, artifact=artifact).run()
    return model, model.first_conversion.copy()


# ---------------------------------------------------------------------------------------------
# RQ3 -- does the state-landlord alliance reproduce the England/France divergence?
# ---------------------------------------------------------------------------------------------
def rq3_regimes(
    base: Params,
    seeds: list[int],
    theta_england: float = 2.0,
    theta_france: float = 6.0,
    artifact: dict | None = None,
) -> pd.DataFrame:
    """Identical model and seeds; only the state-landlord alliance parameter differs."""
    results: list[RunResult] = []
    for label, theta in (("England", theta_england), ("France", theta_france)):
        for seed in seeds:
            results.append(
                run_one(base.with_(seed=seed, theta=theta), label=label, artifact=artifact)
            )
    return stack_histories(results)


# ---------------------------------------------------------------------------------------------
# Robustness and macro-consistency (paper: Robustness and macro-consistency checks)
# ---------------------------------------------------------------------------------------------
def robustness(
    base: Params, seeds: list[int], artifact: dict | None = None
) -> pd.DataFrame:
    """Baseline against targeted mechanism switch-offs, for the sanity checks."""
    variants = {
        "baseline": {},
        "no engrossment": dict(enable_engrossment=False),
        "no enclosure": dict(enable_enclosure=False),
        "no competitive allocation": dict(competitive_allocation=False),
    }
    results = []
    for name, switches in variants.items():
        results.extend(
            run_replicates(base.with_(**switches), seeds, label=name, artifact=artifact)
        )
    return stack_histories(results)


def run_all(
    base: Params, seeds: list[int] | None = None
) -> dict[str, pd.DataFrame]:
    """Execute the whole suite once, sharing the geography artifact across every run."""
    seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
    artifact = load_artifact()
    print(f"Running experiment suite: {len(seeds)} seeds, {base.n_steps} steps")

    print("  RQ1 improvement event study ...")
    rq1_events, rq1_continuity = rq1_improvement(base, seeds, artifact=artifact)

    print(f"  RQ2 spread, {len(CHANNEL_SETS)} channel combinations ...")
    rq2_ablation, rq2_scatter = rq2_spread(base, seeds, artifact=artifact)

    print("  RQ3 England vs France ...")
    rq3 = rq3_regimes(base, seeds, artifact=artifact)

    print("  Robustness checks ...")
    robust = robustness(base, seeds, artifact=artifact)

    return {
        "rq1_events": rq1_events,
        "rq1_continuity": rq1_continuity,
        "rq2_ablation": rq2_ablation,
        "rq2_scatter": rq2_scatter,
        "rq3_regimes": rq3,
        "robustness": robust,
    }
