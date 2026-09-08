"""Tests for the cross-arm statistics of :mod:`pmabm.stats`.

The paired estimator is the one piece of arithmetic in this project whose correctness is not
visible in a figure: a wrong pairing produces a plausible-looking interval of the wrong width.
So the properties tested here are the ones a reader of the figures is entitled to assume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmabm.stats import (
    REGIME_LABELS,
    classify_regimes,
    paired_delta,
    paired_delta_table,
    regime_shares,
    sign_test,
    transition_probability,
)


def _summary(values, seeds=None, column="final_share_leasehold") -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {"seed": range(len(values)) if seeds is None else seeds, column: values}
    )


# --- pairing ------------------------------------------------------------------------------
def test_paired_delta_matches_seeds_rather_than_positions():
    """The whole point: seed 3 of one arm is compared with seed 3 of the other."""
    a = _summary([0.1, 0.2, 0.3], seeds=[0, 1, 2])
    b = _summary([0.3, 0.2, 0.1], seeds=[2, 1, 0])  # same values, reversed seed order
    out = paired_delta(a, b, ["final_share_leasehold"])
    # Every seed has an identical value in both arms, so every difference is exactly zero. A
    # positional comparison would have produced +/-0.2 differences and a spurious interval.
    assert out["mean_delta"].iat[0] == pytest.approx(0.0)
    assert out["n_pairs"].iat[0] == 3


def test_paired_delta_drops_unshared_seeds():
    a = _summary([0.1, 0.2, 0.3, 0.4], seeds=[0, 1, 2, 3])
    b = _summary([0.0, 0.1, 0.2], seeds=[0, 1, 2])
    out = paired_delta(a, b, ["final_share_leasehold"])
    # Seed 3 ran in only one arm; including it would contribute an unpaired observation.
    assert out["n_pairs"].iat[0] == 3
    assert out["mean_delta"].iat[0] == pytest.approx(0.1)


def test_pairing_beats_the_unpaired_interval_on_correlated_arms():
    """Why the estimator was changed, as a number.

    Seeds differ a great deal and the arms differ from each other by a near-constant offset --
    which is what shared geography, shared rent draws and shared shocks produce. The unpaired
    interval is then dominated by between-seed spread that cancels exactly in the paired one.
    """
    rng = np.random.default_rng(0)
    seed_effect = rng.normal(0.5, 0.2, 32)
    a = _summary(seed_effect + 0.05)
    b = _summary(seed_effect)
    out = paired_delta(a, b, ["final_share_leasehold"]).iloc[0]
    assert out["mean_delta"] == pytest.approx(0.05, abs=1e-9)
    assert out["ci_paired"] < out["ci_unpaired"]
    assert out["pairing_gain"] > 5.0


def test_paired_delta_table_includes_a_zero_row_for_the_baseline():
    arms = {"base": _summary([0.4, 0.5]), "other": _summary([0.6, 0.7])}
    table = paired_delta_table(arms, "base", ["final_share_leasehold"])
    baseline_row = table[table["arm_a"] == "base"].iloc[0]
    assert baseline_row["mean_delta"] == pytest.approx(0.0)
    assert not baseline_row["significant"]


# --- the sign test ------------------------------------------------------------------------
def test_sign_test_is_exact_and_two_sided():
    # All eight of eight in one direction: 2 * (1/2)^8 = 0.0078125.
    assert sign_test(np.ones(8)) == pytest.approx(2 / 2**8)
    # A balanced split cannot reject.
    assert sign_test(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(1.0)


def test_sign_test_drops_ties_conservatively():
    """A seed the ablation did not move is evidence for the null, not against it."""
    with_ties = sign_test(np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    without = sign_test(np.array([1.0, 1.0]))
    assert with_ties == pytest.approx(without)


def test_sign_test_of_nothing_is_undefined_rather_than_significant():
    assert np.isnan(sign_test(np.array([])))
    assert np.isnan(sign_test(np.zeros(5)))


# --- regimes ------------------------------------------------------------------------------
def test_regimes_cut_at_the_papers_completion_criterion():
    frame = _summary([0.0, 0.19, 0.2, 0.49, 0.5, 0.95])
    labels = classify_regimes(frame).tolist()
    assert labels == ["failed", "failed", "partial", "partial", "transitioned", "transitioned"]


def test_regime_shares_separate_two_arms_with_the_same_mean():
    """The failure mode the figure exists to catch."""
    # 0.45 rather than 0.5 for the flat arm: 0.5 *is* a Leasehold majority and so counts as
    # transitioned, which is the criterion the paper uses and the boundary the other test pins.
    everyone_halfway = _summary([0.45] * 8)
    all_or_nothing = _summary([0.0] * 4 + [0.9] * 4)
    table = regime_shares({"halfway": everyone_halfway, "split": all_or_nothing})
    means = table.set_index("arm")["mean"]
    assert means["halfway"] == pytest.approx(means["split"])  # indistinguishable by mean
    rows = table.set_index("arm")
    assert rows.loc["halfway", "share_partial"] == pytest.approx(1.0)
    assert rows.loc["split", "share_failed"] == pytest.approx(0.5)
    assert rows.loc["split", "share_transitioned"] == pytest.approx(0.5)
    assert bool(rows.loc["split", "bimodal"]) and not bool(rows.loc["halfway", "bimodal"])
    assert {f"share_{name}" for name in REGIME_LABELS} <= set(table.columns)


def test_transition_probability_is_a_share_of_seeds():
    p, ci = transition_probability(_summary([0.1, 0.2, 0.6, 0.9]))
    assert p == pytest.approx(0.5)
    assert ci > 0
    # A unanimous outcome has no binomial spread left to report.
    p1, ci1 = transition_probability(_summary([0.9, 0.95]))
    assert p1 == pytest.approx(1.0) and ci1 == pytest.approx(0.0)
