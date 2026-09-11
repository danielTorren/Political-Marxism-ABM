"""Derived measures for the paper's emergent outputs and validation targets.

The unit of the RQ1 event study is deliberately the **parcel**, not the person. Conversion in
this model happens at a vacancy and replaces the sitting tenant (paper: Vacancy resolution),
so the parcel is the only entity with a continuous history spanning its own conversion date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Model, Tenure


def history_frame(model: Model) -> pd.DataFrame:
    """Per-period aggregate series.

    Tenure shares are shares of *tenancies*, so the denominator is households: one household
    holds one tenancy however many members it has. ``share_landless_persons``, recorded by the
    model itself, is the same quantity per head, and is the one to read as proletarianisation.
    """
    df = pd.DataFrame(model.history)
    occupied = df[["customary", "leasehold", "freehold"]].sum(axis=1)
    for state in ("customary", "leasehold", "freehold"):
        df[f"share_{state}"] = df[state] / occupied.replace(0, np.nan)
    df["share_landless"] = df["landless"] / (occupied + df["landless"]).replace(0, np.nan)
    return df


def event_study(model: Model, window: int = 25) -> pd.DataFrame:
    """Align every converted parcel on its own conversion date and average in event time.

    Returns tidy rows of ``(event_time, metric, mean, sem, n, cohort)`` for the improving
    disposition, capital and rent, plus a never-converted control series held at its own
    calendar mean.

    The cohort column is ``cohort`` rather than ``group`` because the scenario runner stamps its
    own ``group`` (the scenario group) onto every frame it writes, which silently overwrote this
    one and left the RQ1 event-study figure with nothing to draw.
    """
    conv = model.first_conversion
    converted = np.nonzero(conv >= 0)[0]
    if len(converted) == 0:
        return pd.DataFrame(columns=["event_time", "metric", "mean", "sem", "n", "cohort"])

    panels = {
        "iota": model.panel_iota,
        "capital": model.panel_k,
        "rent": model.panel_rho,
    }
    n_steps = model.p.n_steps
    rows = []

    for name, panel in panels.items():
        # (n_converted, 2*window+1) matrix of values in event time, NaN where out of range.
        aligned = np.full((len(converted), 2 * window + 1), np.nan)
        for row, parcel in enumerate(converted):
            t0 = conv[parcel]
            lo, hi = t0 - window, t0 + window
            src_lo, src_hi = max(0, lo), min(n_steps - 1, hi)
            if src_hi < src_lo:
                continue
            dst_lo = src_lo - lo
            aligned[row, dst_lo : dst_lo + (src_hi - src_lo + 1)] = panel[src_lo : src_hi + 1, parcel]

        with np.errstate(invalid="ignore"):
            mean = np.nanmean(aligned, axis=0)
            count = np.sum(~np.isnan(aligned), axis=0)
            sd = np.nanstd(aligned, axis=0)
        sem = np.divide(sd, np.sqrt(np.maximum(count, 1)), out=np.zeros_like(sd), where=count > 0)
        rows.append(
            pd.DataFrame(
                {
                    "event_time": np.arange(-window, window + 1),
                    "metric": name,
                    "mean": mean,
                    "sem": sem,
                    "n": count,
                    "cohort": "converted",
                }
            )
        )

    # Control: parcels that never converted, in calendar time, as a flat reference level.
    never = np.nonzero(conv < 0)[0]
    if len(never):
        for name, panel in panels.items():
            with np.errstate(invalid="ignore"):
                level = float(np.nanmean(panel[:, never]))
            rows.append(
                pd.DataFrame(
                    {
                        "event_time": np.arange(-window, window + 1),
                        "metric": name,
                        "mean": level,
                        "sem": 0.0,
                        "n": len(never),
                        "cohort": "never converted",
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)


def occupant_continuity(model: Model) -> dict:
    """How often does conversion keep the *same* occupant?

    The paper's RQ1 is phrased about a tenant's own conversion, but its vacancy rule replaces
    the sitting tenant at most conversions. This quantifies how far apart those two are.
    """
    conv = model.first_conversion
    converted = np.nonzero(conv >= 0)[0]
    if len(converted) == 0:
        return {
            "n_conversions": 0,
            "same_occupant_share": float("nan"),
            "customary_to_leasehold_inplace": 0,
        }
    same_person = 0
    tenure_transition = 0
    for parcel in converted:
        t0 = int(conv[parcel])
        if t0 == 0:
            continue
        before_state = model.panel_state[t0 - 1, parcel]
        after_state = model.panel_state[t0, parcel]
        before_person = model.panel_occupant[t0 - 1, parcel]
        after_person = model.panel_occupant[t0, parcel]
        if before_state == int(Tenure.CUSTOMARY) and after_state == int(Tenure.LEASEHOLD):
            tenure_transition += 1
            # The question RQ1 actually asks: is it the *same tenant* either side?
            if before_person >= 0 and before_person == after_person:
                same_person += 1
    return {
        "n_conversions": int(len(converted)),
        "customary_to_leasehold_inplace": int(tenure_transition),
        "same_occupant_share": float(same_person / len(converted)),
    }


def holding_sizes_at(model: Model, t: int) -> np.ndarray:
    """Sizes of every distinct tenant holding at period ``t``."""
    occupants = model.panel_occupant[t]
    sizes = model.panel_holding[t]
    held = occupants >= 0
    if not held.any():
        return np.array([], dtype=float)
    # One entry per tenant, not per parcel: take the holding size once per distinct occupant.
    _, first = np.unique(occupants[held], return_index=True)
    return sizes[held][first].astype(float)


def lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative population share against cumulative land share."""
    if len(values) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    v = np.sort(np.asarray(values, dtype=float))
    total = v.sum()
    if total <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.concatenate([[0.0], np.cumsum(v) / total])
    pop = np.linspace(0.0, 1.0, len(cum))
    return pop, cum


def concentration_frame(model: Model) -> pd.DataFrame:
    """Per-period concentration of *tenant holdings* -- are peasant farms consolidating?"""
    from .model import _gini

    rows = []
    for t in range(model.p.n_steps):
        sizes = holding_sizes_at(model, t)
        if len(sizes) == 0:
            continue
        ordered = np.sort(sizes)[::-1]
        top_decile = max(1, int(round(0.1 * len(ordered))))
        rows.append(
            {
                "t": t,
                "gini": _gini(sizes),
                "n_holdings": len(sizes),
                "mean_holding": float(sizes.mean()),
                "max_holding": float(sizes.max()),
                "top_decile_land_share": float(ordered[:top_decile].sum() / ordered.sum()),
                "single_parcel_share": float((sizes == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def consolidation_by_fertility(model: Model, n_groups: int = 4) -> pd.DataFrame:
    """Consolidation over time, grouped by the land quality of the region.

    Consolidation is measured at the *parcel* level -- the size of the holding each parcel
    belongs to -- so that a region's score reflects how much of its land sits in large farms,
    which is the quantity Brenner's engrossment argument is about.
    """
    region = model.geo.region
    phi_bar = model.geo.phi_bar
    regions = np.unique(region)
    fertility = np.array([phi_bar[region == r].mean() for r in regions])

    # Rank regions into equal-count fertility groups, poorest first.
    order = np.argsort(fertility)
    group_of_region = {}
    for rank, idx in enumerate(order):
        group_of_region[regions[idx]] = min(n_groups - 1, rank * n_groups // len(regions))
    labels = [f"Q{g + 1}" for g in range(n_groups)]

    parcel_group = np.array([group_of_region[r] for r in region])
    rows = []
    for t in range(model.p.n_steps):
        holding = model.panel_holding[t]
        occupied = model.panel_occupant[t] >= 0
        for g in range(n_groups):
            sel = occupied & (parcel_group == g)
            if not sel.any():
                continue
            rows.append(
                {
                    "t": t,
                    "fertility_group": labels[g],
                    "group_index": g,
                    "mean_fertility": float(phi_bar[parcel_group == g].mean()),
                    "mean_holding_size": float(holding[sel].mean()),
                    "share_in_large_holdings": float((holding[sel] >= 3).mean()),
                }
            )
    return pd.DataFrame(rows)


def region_consolidation_summary(model: Model) -> pd.DataFrame:
    """One row per region: land quality against how far and how fast it consolidated."""
    region = model.geo.region
    phi_bar = model.geo.phi_bar
    final = model.panel_holding[model.p.n_steps - 1]
    occupied = model.panel_occupant[model.p.n_steps - 1] >= 0

    rows = []
    for r in np.unique(region):
        sel = region == r
        live = sel & occupied
        if live.sum() < 5:  # too little land to characterise
            continue
        # Time to reach a mean holding of two parcels: a simple "speed" measure.
        trajectory = np.array(
            [
                model.panel_holding[t][sel & (model.panel_occupant[t] >= 0)].mean()
                if (sel & (model.panel_occupant[t] >= 0)).any()
                else np.nan
                for t in range(model.p.n_steps)
            ]
        )
        reached = np.nonzero(trajectory >= 2.0)[0]
        rows.append(
            {
                "region": int(r),
                "mean_fertility": float(phi_bar[sel].mean()),
                "n_parcels": int(sel.sum()),
                "final_mean_holding": float(final[live].mean()),
                "periods_to_double": int(reached[0]) if len(reached) else np.nan,
                # A region where nothing converted has no conversion time; NaN, not an error.
                "mean_conversion_time": (
                    float(model.first_conversion[sel][model.first_conversion[sel] >= 0].mean())
                    if (model.first_conversion[sel] >= 0).any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


#: Distance bins for the semivariogram. Twenty across half the lattice extent leaves hundreds of
#: thousands of pairs per bin at the default ``L``, far more than the estimate needs, and keeps
#: the curve readable.
VARIOGRAM_BINS = 20

#: Pair enumeration is O(n^2). Above this many converted parcels the curve is estimated on a
#: random subsample of *parcels* rather than of pairs, which keeps every bin unbiased -- pair
#: subsampling would over-weight whichever distances happened to be drawn.
VARIOGRAM_MAX_PARCELS = 12_000

#: Fraction of the lattice's own extent out to which the variogram is estimated. Beyond about
#: half, the only pairs available are a thin shell of opposite-corner ones, so the estimate
#: describes the shape of England more than it describes the transition.
VARIOGRAM_MAX_LAG_FRAC = 0.5


def _variogram_bins(
    model: Model,
    n_bins: int = VARIOGRAM_BINS,
    max_lag_frac: float = VARIOGRAM_MAX_LAG_FRAC,
    max_parcels: int = VARIOGRAM_MAX_PARCELS,
    subsample_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    """``(centres, gamma, counts, total_variance, n_parcels)`` or ``None`` if too few conversions.

    Pairs are accumulated in row blocks so the full (n, n) distance matrix is never held; only
    ``i < j`` is counted, so each pair enters exactly one bin exactly once.
    """
    conv = model.first_conversion
    idx = np.nonzero(conv >= 0)[0]
    if len(idx) > max_parcels:
        idx = np.sort(
            np.random.default_rng(subsample_seed).choice(idx, size=max_parcels, replace=False)
        )
    if len(idx) < 10:
        return None

    xy = model.geo.xy[idx].astype(float)
    t = conv[idx].astype(float)
    span = float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    if span <= 0:
        return None
    edges = np.linspace(0.0, max_lag_frac * span, n_bins + 1)

    sum_sq = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for start in range(0, len(idx), 512):
        stop = min(start + 512, len(idx))
        d = np.hypot(
            xy[start:stop, 0][:, None] - xy[:, 0][None, :],
            xy[start:stop, 1][:, None] - xy[:, 1][None, :],
        )
        dt2 = (t[start:stop][:, None] - t[None, :]) ** 2
        upper = np.arange(len(idx))[None, :] > np.arange(start, stop)[:, None]
        bins = np.digitize(d[upper], edges) - 1
        keep = (bins >= 0) & (bins < n_bins)
        sum_sq += np.bincount(bins[keep], weights=dt2[upper][keep], minlength=n_bins)
        counts += np.bincount(bins[keep], minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sum_sq / counts
    gamma[counts == 0] = np.nan
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, gamma, counts, float(t.var()), len(idx)


def spread_variogram(model: Model, **kwargs) -> dict:
    """Is conversion a spreading front or simultaneous? Measured **without** an origin.

    The empirical semivariogram of first-conversion time: for every pair of converted parcels at
    lattice distance :math:`d`,

    .. math:: \\gamma(d) = \\tfrac{1}{2}\\,\\mathbb{E}\\big[(t_i - t_j)^2\\big]

    This replaces an earlier regression of conversion time on distance from a designated
    "ecological seed" county. That construction required naming a centre, and named it using the
    same ALC field that drives conversion -- so a positive slope was partly underwritten by the
    measurement rather than found in the run. A variogram has no centre to choose, so the
    question RQ2 asks becomes answerable without first answering a question RQ2 does not ask.

    Contagion and simultaneity separate on two numbers rather than one, which the radial slope
    could not do:

    * ``nugget_share`` -- :math:`\\gamma` in the nearest bin over the total variance of conversion
      times. Near 0: adjacent parcels convert at nearly the same time, so there is local temporal
      coherence to spread. Near 1: neighbours differ as much as opposite corners do, which is
      simultaneity plus noise and no spatial process at all.
    * ``slope_norm`` -- OLS slope of :math:`\\gamma(d)` over total variance against :math:`d`, per
      lattice cell. Positive: parcels further apart convert further apart in time. Near zero with
      a high nugget is the outcome RQ2 is built to be able to falsify.

    ``range_cells`` is the distance at which :math:`\\gamma` first reaches 95% of total variance --
    the spatial scale of the process, and the thing the radial slope was blindest to. A short
    range with a low nugget means *local patches* converting independently; a range comparable to
    the lattice means a single country-scale front. Both give a positive radial slope from a
    well-placed origin, and they are different claims.
    """
    binned = _variogram_bins(model, **kwargs)
    nan = float("nan")
    if binned is None:
        return {
            "n": int((model.first_conversion >= 0).sum()),
            "n_pairs": 0,
            "slope_norm": nan,
            "nugget_share": nan,
            "plateau_share": nan,
            "range_cells": nan,
            "r": nan,
            "total_variance": nan,
        }
    centres, gamma, counts, total_var, n = binned
    ok = np.isfinite(gamma)
    # A run in which every parcel converted in the same period has no temporal variance to
    # apportion across distance; that is simultaneity in its limiting case, not missing data.
    norm = gamma / total_var if total_var > 0 else np.full_like(gamma, nan)
    reached = np.nonzero(ok & (norm >= 0.95))[0]
    return {
        "n": n,
        "n_pairs": int(counts.sum()),
        "slope_norm": (
            float(np.polyfit(centres[ok], norm[ok], 1)[0])
            if ok.sum() > 2 and total_var > 0
            else nan
        ),
        "nugget_share": float(norm[ok][0]) if ok.any() and total_var > 0 else nan,
        "plateau_share": float(norm[ok][-1]) if ok.any() and total_var > 0 else nan,
        "range_cells": float(centres[reached[0]]) if len(reached) else nan,
        "r": (
            float(np.corrcoef(centres[ok], gamma[ok])[0, 1])
            if ok.sum() > 2 and np.ptp(gamma[ok]) > 0
            else nan
        ),
        "total_variance": total_var,
    }


def spread_variogram_curve(model: Model, **kwargs) -> pd.DataFrame:
    """The binned semivariogram itself, for the RQ2 figure and its data table."""
    binned = _variogram_bins(model, **kwargs)
    if binned is None:
        return pd.DataFrame(columns=["distance", "gamma", "gamma_norm", "n_pairs"])
    centres, gamma, counts, total_var, _ = binned
    return pd.DataFrame(
        {
            "distance": centres,
            "gamma": gamma,
            "gamma_norm": gamma / total_var if total_var > 0 else np.nan,
            "n_pairs": counts,
        }
    )


def summary(model: Model) -> dict:
    """One-line description of a completed run."""
    df = history_frame(model)
    last = df.iloc[-1]
    spread = spread_variogram(model)
    return {
        "final_share_customary": float(last["share_customary"]),
        "final_share_leasehold": float(last["share_leasehold"]),
        "final_share_freehold": float(last["share_freehold"]),
        "final_share_landless": float(last["share_landless"]),
        "final_farm_gini": float(last["farm_gini"]),
        "converted_parcels": int(last["converted_parcels"]),
        "parcels": int(model.geo.n_parcels),
        "conversion_share": float(last["converted_parcels"] / model.geo.n_parcels),
        "spread_slope_norm": spread["slope_norm"],
        "spread_nugget_share": spread["nugget_share"],
        "spread_range_cells": spread["range_cells"],
        "spread_r": spread["r"],
        "final_enclosure": float(last["enclosure"]),
        "final_wage": float(last["wage"]),
        # A cumulative flow, not ``last["exited"]``: with return migration the urban state is
        # no longer terminal, so the count of households currently there is a stock. Taken from
        # the model's own running total rather than summed out of ``df``, which would undercount
        # whenever the history is recorded less often than every period.
        "cumulative_exited": int(model.cumulative_exits),
        "cumulative_returned": int(model.cumulative_returns),
        "urban_households": int(last["exited"]),
        # Demography. Reported alongside the tenure outcomes because the two turned out to be
        # tightly coupled: with a closed population the final leasehold share correlated at 0.97
        # with how far the population had fallen, so any regime comparison has to be read
        # against these before it can be read as a result about property relations.
        "final_population": float(last["population"]),
        "final_total_population": float(last["total_population"]),
        "final_households": float(last["households"]),
        "population_ratio": float(last["population"] / max(df.iloc[0]["population"], 1)),
        "total_population_ratio": float(
            last["total_population"] / max(df.iloc[0]["total_population"], 1)
        ),
        "final_share_persons_urban": float(last["share_persons_urban"]),
        "final_mean_household_size": float(last["mean_household_size"]),
        "final_share_landless_persons": float(last["share_landless_persons"]),
        "final_share_parcels_vacant": float(last["share_parcels_vacant"]),
        "cumulative_births": int(model.cumulative_births),
        "cumulative_deaths": int(model.cumulative_deaths),
        "cumulative_partitions": int(model.cumulative_partitions),
        **occupant_continuity(model),
    }


# ---------------------------------------------------------------------------------------------
# Per-parcel frames: the unit of observation is a *place*
#
# Every other frame in this module is indexed by time, by event time, or by lag distance, so the
# spatial state of a run -- which land converted when, on what soil, in whose estate -- lived
# only inside the ``Model`` object and was discarded when the run ended. That is why the scenario
# suite could report a variogram statistic but could not draw a map. These two frames are what a
# runner persists instead.
#
# They are split because the two halves have different cardinality. Under ``fixed_geography`` the
# lattice is built once per arm and shared by every seed, so the static facts about a parcel are
# identical in every replicate; storing them per seed would repeat eight columns across every
# one. :func:`geography_frame` is therefore written once per arm and joined back onto
# :func:`parcel_frame` on ``parcel`` at plot time.
# ---------------------------------------------------------------------------------------------

#: Storage dtype of every column of :func:`parcel_frame`. Explicit rather than inferred, because
#: this is the largest frame a scenario run writes -- one row per parcel per seed, so tens of
#: millions across a full suite -- and pandas would default every count to ``int64`` and every
#: measure to ``float64``, roughly doubling it for no gain in a quantity used to draw maps.
PARCEL_DTYPES = {
    "parcel": "int32",
    "commons": "bool",
    "customary_rent": "float32",
    "first_conversion": "int32",
    "parcel_tenure": "int8",
    "final_state": "int8",
    "final_holding": "int16",
    "final_iota": "float32",
    "final_k": "float32",
    "final_y": "float32",
    "final_rho": "float32",
    "final_phi": "float32",
    "final_enclosure": "float32",
    "enclosure_half_time": "int32",
}


def parcel_frame(model: Model) -> pd.DataFrame:
    """One row per parcel: everything spatial that *varies between seeds*.

    ``commons`` and ``customary_rent`` are here rather than in :func:`geography_frame` despite
    looking like fixed features of the land: both are drawn from the model's own generator at
    construction, so they differ from seed to seed even when the lattice does not.

    Two distinct tenure columns, because the paper keeps them distinct. ``parcel_tenure`` is the
    tenure attached to the *land*, which only ratchets forward; ``final_state`` is the tenure
    state of whoever occupies it at the end, and is ``-1`` where nobody does. A parcel converted
    to leasehold and then left vacant reads as Leasehold in the first and vacant in the second.

    ``first_conversion`` is ``-1`` for land that never converted. That is a censored observation
    rather than a missing one, and the map functions treat it as such: a mean over seeds of the
    raw column is meaningless, which is why the share of seeds converted by a given period is the
    quantity to plot.
    """
    last = model.p.n_steps - 1
    frame = pd.DataFrame(
        {
            "parcel": np.arange(model.geo.n_parcels),
            "commons": model.commons,
            "customary_rent": model.customary_rent,
            "first_conversion": model.first_conversion,
            "parcel_tenure": model.parcel_tenure,
            "final_state": model.panel_state[last],
            "final_holding": model.panel_holding[last],
            "final_iota": model.panel_iota[last],
            "final_k": model.panel_k[last],
            "final_y": model.panel_y[last],
            "final_rho": model.panel_rho[last],
            # Realised fertility, not the carrying capacity: phi_bar is static and lives in
            # geography_frame, while this is what the ecological cycle left of it.
            "final_phi": model.phi,
            # Enclosure as experienced by this parcel, i.e. its own estate's. Constant across
            # every parcel under ``enclosure_rule="national"``, which is the point of carrying it
            # -- the RQ9 comparison of the two fronts is only available under the local rule and
            # the column says so on its face rather than in a caption.
            "final_enclosure": model.enclosure_by_estate[model.geo.landlord],
            "enclosure_half_time": model.enclosure_half_time[model.geo.landlord],
        }
    )
    return frame.astype(PARCEL_DTYPES)


def geography_frame(geo) -> pd.DataFrame:
    """One row per parcel: the static facts a run cannot change.

    Takes a :class:`~pmabm.geography.Geography` rather than a ``Model``, so a runner holding a
    shared lattice can build this without a model in hand.

    ``county_name`` is carried alongside ``county`` even though it is derivable, because the
    index is an artifact ordering with no meaning outside the run that produced it, and a
    choropleth legend needs the name.
    """
    county = geo.county.astype(int)
    return pd.DataFrame(
        {
            "parcel": np.arange(geo.n_parcels, dtype="int32"),
            "row": geo.xy[:, 0].astype("int16"),
            "col": geo.xy[:, 1].astype("int16"),
            "county": county.astype("int16"),
            "county_name": pd.Categorical([geo.county_names[c] for c in county]),
            "county_grade": geo.county_grade[county].astype("float32"),
            "phi_bar": geo.phi_bar.astype("float32"),
            "landlord": geo.landlord.astype("int32"),
        }
    )


def county_history_frame(model: Model) -> pd.DataFrame:
    """One row per county per period: coarse in space, complete in time.

    The complement to :func:`parcel_frame`, which is the other way round. Together they cover
    both axes without storing the full parcel-by-period panel, which at the default lattice
    would be some 136 GB across a scenario suite.

    **The shares here are shares of land, not of tenancies**, unlike :func:`history_frame`. The
    denominator is occupied parcels in the county, so ``share_leasehold`` answers "how much of
    this county is under leasehold" rather than "what fraction of its tenants hold by lease".
    That is the quantity a choropleth should show, and the two diverge exactly where the theory
    says they should -- consolidation means fewer, larger leasehold farms, so the land share runs
    ahead of the tenancy share.

    ``share_converted`` is cumulative and taken from ``first_conversion`` rather than from the
    occupant's current state, so land that converted and later fell vacant still counts. That is
    what makes it the spread measure: it only ever rises, so a map of it over time is a front.

    Consolidation is reported at the parcel level -- the mean size of the holding each parcel
    belongs to -- rather than as a within-county Gini. Same choice as
    :func:`consolidation_by_fertility`, and for the same reason: it is the quantity Brenner's
    engrossment argument is about, being how much of a county's land sits in large farms.
    """
    county = model.geo.county.astype(int)
    n_counties = model.geo.n_counties
    parcels_by_county = np.bincount(county, minlength=n_counties).astype(float)
    conv = model.first_conversion
    live = parcels_by_county > 0

    def by_county(mask: np.ndarray) -> np.ndarray:
        return np.bincount(county[mask], minlength=n_counties).astype(float)

    def mean_by_county(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
        total = np.bincount(county[mask], weights=values[mask], minlength=n_counties)
        count = np.bincount(county[mask], minlength=n_counties)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    states = {
        "customary": int(Tenure.CUSTOMARY),
        "leasehold": int(Tenure.LEASEHOLD),
        "freehold": int(Tenure.FREEHOLD),
    }
    rows = []
    for t in range(model.p.n_steps):
        state = model.panel_state[t]
        holding = model.panel_holding[t]
        occupied = state >= 0
        occupied_by_county = by_county(occupied)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = np.where(occupied_by_county > 0, occupied_by_county, np.nan)
        block = {
            "t": t,
            "county": np.arange(n_counties),
            "n_parcels": parcels_by_county,
            "n_occupied": occupied_by_county,
            "share_vacant": 1.0 - occupied_by_county / np.where(live, parcels_by_county, np.nan),
            "share_converted": by_county((conv >= 0) & (conv <= t))
            / np.where(live, parcels_by_county, np.nan),
            "mean_holding_of_parcel": mean_by_county(occupied, holding.astype(float)),
            "share_in_large_holdings": (
                by_county(occupied & (holding >= 3)) / denom
            ),
            "mean_iota": mean_by_county(occupied, np.nan_to_num(model.panel_iota[t])),
            "mean_k": mean_by_county(occupied, np.nan_to_num(model.panel_k[t])),
            "mean_phi": mean_by_county(np.ones_like(occupied), model.phi),
        }
        for name, code in states.items():
            block[f"share_{name}"] = by_county(occupied & (state == code)) / denom
        rows.append(pd.DataFrame(block))

    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["n_parcels"] > 0].reset_index(drop=True)
    names = np.asarray(model.geo.county_names, dtype=object)
    frame["county_name"] = pd.Categorical(names[frame["county"].to_numpy()])
    return frame.astype(
        {
            "t": "int16",
            "county": "int16",
            "n_parcels": "int32",
            "n_occupied": "int32",
            **{
                c: "float32"
                for c in frame.columns
                if c not in ("t", "county", "n_parcels", "n_occupied", "county_name")
            },
        }
    )
