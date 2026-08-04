"""Tests for the Sobol' sensitivity harness in ``src/sensitivity``.

The expensive half -- the sweep itself -- is not exercised here. What is tested is everything
that can silently give a wrong *answer* rather than an error: the design, the coercion of sample
rows back into ``Params``, the row alignment between the sample matrix and ``Y`` (Sobol' reads
``Y`` positionally, so a misalignment produces plausible indices for the wrong parameters), and
that the wrappers recover the indices of a function whose sensitivities are known.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
for _extra in (SRC, SRC / "multi_seed", SRC / "sensitivity"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import sensitivity_analysis as sa  # noqa: E402
import sensitivity_gen as sg  # noqa: E402

from pmabm.config import ENGLAND  # noqa: E402
from pmabm.geography import GeographyMissing, load_artifact  # noqa: E402
from pmabm.metrics import history_frame  # noqa: E402
from pmabm.model import Model  # noqa: E402


# ---------------------------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------------------------
def test_build_problem_reads_bounds_in_order():
    problem, integers = sg.build_problem(
        {"parameters": {"theta": [1.0, 8.0], "alpha_1": [1.0, 4.0], "tau": [2, 5]}}
    )
    assert problem["names"] == ["theta", "alpha_1", "tau"]
    assert problem["bounds"] == [[1.0, 8.0], [1.0, 4.0], [2.0, 5.0]]
    assert problem["num_vars"] == 3
    assert integers == ["tau"]  # int-typed field of Params, detected from the annotation


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"not_a_parameter": [0.0, 1.0]}, "Unknown parameter"),
        ({"urban_demand": [0.0, 1.0]}, "switch"),        # bool
        ({"exit_rule": [0.0, 1.0]}, "switch"),           # str
        ({"theta": [1.0]}, "must be [lo, hi]"),
        ({"theta": [8.0, 1.0]}, "hi > lo"),
        ({}, "no parameters"),
    ],
)
def test_build_problem_rejects_what_cannot_be_decomposed(parameters, message):
    with pytest.raises(SystemExit) as excinfo:
        sg.build_problem({"parameters": parameters})
    assert message in str(excinfo.value)


def test_design_has_the_documented_size():
    """``n_base * (D + 2)`` rows: the cost estimate in the yaml has to be the real one."""
    problem, _ = sg.build_problem(
        {"parameters": {"theta": [1.0, 8.0], "alpha_1": [1.0, 4.0], "chi": [0.5, 4.0]}}
    )
    design = sg.sample_design(problem, 16, calc_second_order=False)
    assert design.shape == (16 * (3 + 2), 3)
    for column, (lo, hi) in zip(design.T, problem["bounds"]):
        assert lo <= column.min() and column.max() <= hi


def test_overrides_round_integers_and_keep_floats():
    problem = {"names": ["tau", "theta"], "bounds": [[2.0, 5.0], [1.0, 8.0]]}
    overrides = sg.overrides_for(np.array([3.7, 2.5]), problem, {"tau"})
    assert overrides == {"tau": 4, "theta": 2.5}
    assert isinstance(overrides["tau"], int)
    # An override dict must be directly acceptable to Params, or the sweep fails per run.
    assert ENGLAND.with_(**overrides).tau == 4


def test_geography_fields_are_the_ones_geography_actually_uses():
    """If this drifts, a varied layout parameter is silently held constant across the sweep."""
    import inspect

    from pmabm import geography

    source = inspect.getsource(geography.build)
    used = {f for f in sg.GEOGRAPHY_FIELDS if f"params.{f}" in source}
    assert used == set(sg.GEOGRAPHY_FIELDS)


# ---------------------------------------------------------------------------------------------
# Row alignment: the failure mode that produces confident nonsense
# ---------------------------------------------------------------------------------------------
def test_average_over_seeds_keeps_design_order_when_a_sample_fails():
    runs = pd.DataFrame(
        {
            "sample": [0, 0, 2, 2, 3, 3],  # sample 1 failed outright and produced no row
            "seed": [0, 1, 0, 1, 0, 1],
            "conversion_share": [0.1, 0.3, 0.5, 0.5, 0.2, np.nan],
        }
    )
    frame = sg.average_over_seeds(runs, ["conversion_share"], n_samples=4)
    assert frame["sample"].tolist() == [0, 1, 2, 3]
    assert frame["conversion_share"].iloc[0] == pytest.approx(0.2)   # mean of the two seeds
    assert np.isnan(frame["conversion_share"].iloc[1])               # the gap is preserved
    assert frame["conversion_share"].iloc[3] == pytest.approx(0.2)   # NaN seed skipped, not fatal


def test_missing_outputs_are_dropped_rather_than_invented():
    runs = pd.DataFrame({"sample": [0, 1], "seed": [0, 0], "conversion_share": [0.1, 0.2]})
    frame = sg.average_over_seeds(runs, ["conversion_share", "never_recorded"], n_samples=2)
    assert list(frame.columns) == ["sample", "conversion_share"]


def test_imputation_is_counted_not_absorbed():
    n_samples = 8
    y = pd.DataFrame(
        {"sample": range(n_samples), "x": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]}
    )
    values, gaps = sa._prepare_column(y, "x", n_samples)
    assert gaps == 2
    assert not np.isnan(values).any()
    assert values[1] == pytest.approx(np.nanmean(y["x"]))  # mean adds no variance of its own


def test_prepare_column_refuses_an_entirely_missing_output():
    y = pd.DataFrame({"sample": [0, 1], "x": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="every sample is missing"):
        sa._prepare_column(y, "x", 2)


# ---------------------------------------------------------------------------------------------
# The decomposition, against a function whose sensitivities are known
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def analytic():
    """``y = x1 + 4*x2 + 0*x3`` over unit ranges: ST ordering x2 > x1 > x3, and additive.

    Sampled and analysed through the same wrappers the sweep uses, so this checks the plumbing
    (design, column ordering, positional alignment) and not SALib.
    """
    problem, _ = sg.build_problem(
        {"parameters": {"alpha_1": [0.0, 1.0], "theta": [0.0, 1.0], "chi": [0.0, 1.0]}}
    )
    design = sg.sample_design(problem, 64, calc_second_order=False)
    values = design[:, 0] + 4.0 * design[:, 1]
    y = pd.DataFrame({"sample": range(len(design)), "y": values})
    return problem, y, len(design)


def test_indices_recover_a_known_ranking(analytic):
    problem, y, n_samples = analytic
    indices, gaps = sa.sobol_indices(problem, y, ["y"], n_samples)
    ordered = indices.set_index("parameter")
    assert gaps == {"y": 0}
    assert ordered.loc["theta", "ST"] > ordered.loc["alpha_1", "ST"]
    assert ordered.loc["alpha_1", "ST"] > ordered.loc["chi", "ST"]
    # 16:1 variance ratio for a coefficient ratio of 4:1.
    assert ordered.loc["theta", "ST"] == pytest.approx(16 / 17, abs=0.05)
    assert ordered.loc["chi", "ST"] == pytest.approx(0.0, abs=0.02)
    assert not bool(ordered.loc["chi", "ST_significant"])
    # Purely additive, so first order accounts for everything and nothing is left for interaction.
    assert indices["S1"].sum() == pytest.approx(1.0, abs=0.05)
    assert indices["interaction"].max() < 0.05


def test_convergence_uses_nested_prefixes_of_the_design(analytic):
    problem, y, _ = analytic
    curve = sa.convergence(problem, y, ["y"], n_base=64)
    assert sorted(curve["n_base"].unique()) == [8, 16, 32, 64]
    largest = curve[(curve["n_base"] == 64) & (curve["parameter"] == "theta")]["ST"].iat[0]
    assert largest == pytest.approx(16 / 17, abs=0.05)


def test_constant_output_is_skipped_not_decomposed(analytic):
    """A censored measure can come out constant; that must not become a table of zeros."""
    problem, y, n_samples = analytic
    flat = y.assign(y=1.0)
    indices, _ = sa.sobol_indices(problem, flat, ["y"], n_samples)
    assert indices.empty


def test_noise_share_measures_the_stochastic_floor():
    """Two seeds, known within- and between-sample variance."""
    runs = pd.DataFrame(
        {
            "sample": [0, 0, 1, 1],
            "seed": [0, 1, 0, 1],
            "x": [0.0, 2.0, 10.0, 12.0],  # within var 2.0 per sample, means 1.0 and 11.0
        }
    )
    row = sa.noise_shares(runs, ["x"]).iloc[0]
    assert row["mean_replicates"] == 2.0
    assert row["var_within_seeds"] == pytest.approx(2.0)
    assert row["var_between_samples"] == pytest.approx(50.0)
    assert row["noise_variance"] == pytest.approx(1.0)   # within / replicates
    assert row["noise_share"] == pytest.approx(0.02)


def test_noise_share_is_undefined_without_replication():
    runs = pd.DataFrame({"sample": [0, 1], "seed": [0, 0], "x": [1.0, 2.0]})
    assert np.isnan(sa.noise_shares(runs, ["x"]).iloc[0]["noise_share"])


# ---------------------------------------------------------------------------------------------
# The extra outputs
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def finished():
    try:
        artifact = load_artifact()
    except GeographyMissing:
        pytest.skip("geography artifact not built; run `uv run pmabm build-geography`")
    params = ENGLAND.with_(L=30, lords_per_county=2, n_steps=25, seed=0)
    return Model(params, artifact=artifact).run()


def test_timing_outputs_are_censored_not_missing(finished):
    history = history_frame(finished)
    timing = sg.timing_outputs(finished, history)
    steps = finished.p.n_steps
    for value in timing.values():
        assert np.isfinite(value)
        assert 0 <= value <= steps
    assert timing["t_first_conversion"] <= timing["t_half_conversion"]

    share = history["converted_parcels"].to_numpy() / finished.geo.n_parcels
    if share.max() < 0.5:
        assert timing["t_half_conversion"] == steps  # censored at the horizon, never NaN
    else:
        reached = int(np.nonzero(share >= 0.5)[0][0])
        assert timing["t_half_conversion"] == history["t"].iloc[reached]
