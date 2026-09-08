"""Tests for the neural-network emulator in ``src/emulator``.

The expensive halves -- generating the corpus and fitting the ensemble -- are not exercised here.
What is tested is everything that can silently give a wrong *answer* rather than an error:

* the design space, including that a switch is refused where a continuous range is expected;
* the input encoding, since a feature order that differs between training and prediction produces
  confident nonsense rather than a failure;
* the trajectory transform/PCA round trip, which is what stands between a predicted share and a
  share outside [0, 1];
* that splits keep replicate seeds of a design point together, without which every reported
  accuracy figure is measuring the model's seed noise instead of the emulator's error;
* that the noise floor recovers a known variance ratio, since every accuracy figure is reported
  against it;
* that a saved emulator reloads to the same predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
for _extra in (SRC, SRC / "multi_seed", SRC / "sensitivity", SRC / "emulator"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import emulator_data as ed  # noqa: E402
import emulator_gen as eg  # noqa: E402

torch = pytest.importorskip("torch", reason="the emulator extra is not installed")
import emulator_net as en  # noqa: E402


# ---------------------------------------------------------------------------------------------
# The design space
# ---------------------------------------------------------------------------------------------
def test_build_space_reads_parameters_and_switches():
    space = eg.build_space(
        {
            "parameters": {"theta": [1.0, 8.0], "alpha_1": [1.0, 4.0], "tau": [2, 5]},
            "switches": {"channel_ideology": [True, False], "ideology_rule": ["material", "contagion"]},
        }
    )
    assert space.cont_names == ["theta", "alpha_1", "tau"]
    assert space.integers == frozenset({"tau"})
    assert space.switch_names == ["channel_ideology", "ideology_rule"]
    assert space.n_dims == 5
    assert space.n_arms == 4


def test_build_space_rejects_a_switch_given_a_range():
    with pytest.raises(SystemExit, match="list it under `switches:`"):
        eg.build_space({"parameters": {"channel_ideology": [0.0, 1.0]}})


def test_build_space_rejects_a_level_params_would_refuse():
    with pytest.raises(SystemExit, match="rejects level"):
        eg.build_space(
            {"parameters": {"theta": [1.0, 8.0]}, "switches": {"ideology_rule": ["material", "nonsense"]}}
        )


def test_sample_design_stays_inside_bounds_and_levels():
    space = eg.build_space(
        {
            "parameters": {"theta": [1.0, 8.0], "tau": [2, 5]},
            "switches": {"channel_ideology": [True, False]},
        }
    )
    design = eg.sample_design(space, 64, seed=3)

    assert len(design) == 64
    assert design["theta"].between(1.0, 8.0).all()
    assert design["tau"].between(2, 5).all()
    assert design["tau"].dtype.kind == "i"          # integer fields are rounded, not passed as floats
    assert set(design["channel_ideology"]) <= {True, False}
    # A scrambled Sobol' sequence should balance a binary switch across the design, not pile up
    # on one level; a bug in the level mapping typically shows as an all-one-level column.
    assert 0.3 < design["channel_ideology"].mean() < 0.7


def test_overrides_round_trip_into_params():
    from pmabm.config import Params

    space = eg.build_space(
        {
            "parameters": {"theta": [1.0, 8.0], "tau": [2, 5]},
            "switches": {"ideology_rule": ["material", "contagion"]},
        }
    )
    row = eg.sample_design(space, 8, seed=1).to_dict("records")[0]
    overrides = eg.overrides_for(row, space)

    assert isinstance(overrides["tau"], int)
    assert isinstance(overrides["theta"], float)
    params = Params(**overrides)                     # would raise if a value were the wrong type
    assert params.ideology_rule in ("material", "contagion")


# ---------------------------------------------------------------------------------------------
# Input encoding
# ---------------------------------------------------------------------------------------------
def _encoder() -> ed.Encoder:
    return ed.Encoder(
        cont_names=["theta", "alpha_1"],
        switch_names=["channel_ideology", "ideology_rule"],
        switch_levels={"channel_ideology": [True, False], "ideology_rule": ["material", "contagion"]},
        bounds=np.array([[1.0, 8.0], [1.0, 4.0]]),
    )


def test_encoder_feature_order_and_widths():
    encoder = _encoder()
    assert encoder.feature_names == [
        "theta", "alpha_1", "channel_ideology", "ideology_rule=material", "ideology_rule=contagion",
    ]

    frame = pd.DataFrame(
        {
            "theta": [2.0, 6.0],
            "alpha_1": [1.5, 3.5],
            "channel_ideology": [True, False],
            "ideology_rule": ["material", "contagion"],
        }
    )
    encoded = encoder.encode(frame)
    assert encoded.shape == (2, 5)
    assert encoded[0, 2] == 1.0 and encoded[1, 2] == -1.0       # booleans are +/-1
    assert list(encoded[0, 3:]) == [1.0, 0.0]                   # one-hot, first level
    assert list(encoded[1, 3:]) == [0.0, 1.0]


def test_encoder_rejects_an_unknown_level():
    frame = pd.DataFrame(
        {"theta": [2.0], "alpha_1": [1.5], "channel_ideology": [True], "ideology_rule": ["reproduction"]}
    )
    with pytest.raises(SystemExit, match="unknown level"):
        _encoder().encode(frame)


# ---------------------------------------------------------------------------------------------
# Trajectory compression
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,values", [
    ("none", np.array([[-2.0, 0.0, 3.5]])),
    ("log1p", np.array([[0.0, 1.0, 250.0]])),
    ("logit", np.array([[0.02, 0.5, 0.97]])),
])
def test_transforms_invert(kind, values):
    assert np.allclose(ed.invert_transform(ed.apply_transform(values, kind), kind), values, atol=1e-6)


def test_logit_keeps_a_reconstructed_share_inside_the_unit_interval():
    # Logistic take-off curves with varying midpoints: the shape every conversion series in this
    # model has, and the one a raw-units basis reconstructs to values outside [0, 1].
    steps = np.linspace(0, 1, 60)
    curves = np.stack([0.95 / (1.0 + np.exp(-18.0 * (steps - m))) for m in np.linspace(0.25, 0.75, 40)])

    logit = ed.fit_series_basis("share", curves, "logit", variance_target=0.9999, max_components=8)
    rebuilt = logit.reconstruct(logit.project(curves))
    assert rebuilt.min() >= 0.0 and rebuilt.max() <= 1.0
    assert np.abs(rebuilt - curves).max() < 0.02

    # The reason the transform is there at all: the same basis size in raw units leaves the
    # interval, which is not something clipping afterwards would fix honestly.
    raw = ed.fit_series_basis("share", curves, "none", variance_target=0.999, max_components=8)
    assert raw.reconstruct(raw.project(curves)).min() < 0.0


def test_basis_reproduces_smooth_curves_in_few_components():
    steps = np.linspace(0, 4 * np.pi, 200)
    curves = np.stack([a * np.sin(steps) + b * steps for a, b in zip(np.linspace(1, 3, 50), np.linspace(0, 2, 50))])
    basis = ed.fit_series_basis("smooth", curves, "none", variance_target=0.9999, max_components=12)

    # Two generating factors, so two components should carry essentially all of it.
    assert basis.n_components <= 3
    assert np.allclose(basis.reconstruct(basis.project(curves)), curves, atol=1e-6)


def test_trajectory_basis_layout_and_incomplete_rows():
    curves = np.random.default_rng(0).normal(size=(20, 2, 30))
    curves[3, 0, 15:] = np.nan                      # a run that ended early
    basis = ed.fit_trajectory_basis(
        curves[np.arange(20) != 3], ["a", "b"], {"a": "none", "b": "none"},
        variance_target=0.99, max_components=5, min_components=2,
    )
    coeffs = basis.project(curves, ["a", "b"])

    spans = basis.slices
    assert coeffs.shape == (20, basis.n_coeffs)
    assert np.isnan(coeffs[3, spans["a"]]).all()    # incomplete series -> masked, not imputed
    assert np.isfinite(coeffs[3, spans["b"]]).all() # the other series is unaffected


# ---------------------------------------------------------------------------------------------
# Splits and the noise floor
# ---------------------------------------------------------------------------------------------
def test_split_keeps_every_replicate_of_a_point_in_one_fold():
    points = np.repeat(np.arange(100), 3)
    masks = ed.split_by_point(points, val_fraction=0.1, test_fraction=0.1, seed=7)

    assert sum(int(m.sum()) for m in masks.values()) == len(points)
    folds = {name: set(points[mask].tolist()) for name, mask in masks.items()}
    assert folds["train"].isdisjoint(folds["val"])
    assert folds["train"].isdisjoint(folds["test"])
    assert folds["val"].isdisjoint(folds["test"])
    assert len(folds["test"]) == 10


def test_noise_floor_recovers_a_known_variance_ratio():
    rng = np.random.default_rng(0)
    n_points, n_seeds = 400, 8
    signal = rng.normal(0.0, 1.0, size=n_points)                       # between-point sd 1
    values = np.repeat(signal, n_seeds) + rng.normal(0.0, 1.0, size=n_points * n_seeds)
    points = np.repeat(np.arange(n_points), n_seeds)

    # Equal variances, so half the total is irreducible seed noise.
    assert ed.noise_floor(values, points)[0] == pytest.approx(0.5, abs=0.05)


def test_noise_floor_is_nan_without_replicates():
    values = np.arange(10.0)
    assert np.isnan(ed.noise_floor(values, np.arange(10))).all()


# ---------------------------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------------------------
def _small_emulator() -> en.Emulator:
    encoder = _encoder()
    steps = np.linspace(0, 1, 25)
    basis = ed.fit_trajectory_basis(
        np.stack([np.stack([steps * k, steps**2 * k]) for k in np.linspace(0.2, 0.9, 30)]),
        ["a", "b"], {"a": "none", "b": "none"}, variance_target=0.999, max_components=4,
    )
    torch.manual_seed(0)
    net = en.EmulatorNet(
        n_inputs=len(encoder.feature_names), n_scalars=2, n_coeffs=basis.n_coeffs, hidden=[16, 16]
    )
    return en.Emulator(
        members=[net], encoder=encoder, basis=basis, scalar_names=["x", "y"],
        x_mean=np.zeros(5), x_std=np.ones(5),
        y_mean=np.zeros(2), y_std=np.ones(2),
        c_mean=np.zeros(basis.n_coeffs), c_std=np.ones(basis.n_coeffs),
        y_lo=np.array([-10.0, -10.0]), y_hi=np.array([10.0, 10.0]),
        config={"model": {"hidden": [16, 16]}}, n_steps=25,
    )


def _design() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "theta": [2.0, 6.0, 4.0],
            "alpha_1": [1.5, 3.5, 2.0],
            "channel_ideology": [True, False, True],
            "ideology_rule": ["material", "contagion", "material"],
        }
    )


def test_nll_is_masked_and_penalises_overconfidence():
    mu = torch.zeros(4, 2)
    target = torch.tensor([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0]] * 4)           # the badly-predicted column is masked out

    tight = en.gaussian_nll(mu, torch.full((4, 2), 0.1), target, mask)
    loose = en.gaussian_nll(mu, torch.full((4, 2), 1.0), target, mask)
    # With the mask on, only the exactly-predicted column counts, so a tighter sigma is better.
    assert float(tight) < float(loose)
    # Unmasked, the 5-sigma miss in the second column dominates and confidence is punished.
    full = torch.ones(4, 2)
    assert float(en.gaussian_nll(mu, torch.full((4, 2), 0.1), target, full)) > float(
        en.gaussian_nll(mu, torch.full((4, 2), 1.0), target, full)
    )


def test_predictions_respect_the_recorded_output_range():
    emulator = _small_emulator()
    emulator.y_lo = np.array([0.0, 0.0])
    emulator.y_hi = np.array([1.0, 1.0])
    predicted = emulator.predict(_design(), draws=8).scalars.to_numpy()
    assert predicted.min() >= 0.0 and predicted.max() <= 1.0


def test_cheap_and_sampled_scalar_paths_agree():
    emulator = _small_emulator()
    fast, _ = emulator.predict_scalars(_design())
    assert np.allclose(fast, emulator.predict(_design(), draws=4).scalars.to_numpy())


def test_save_load_round_trip(tmp_path):
    emulator = _small_emulator()
    before = emulator.predict(_design(), draws=16, seed=1)
    emulator.save(tmp_path / "model")
    after = en.Emulator.load(tmp_path / "model").predict(_design(), draws=16, seed=1)

    assert np.allclose(before.scalars.to_numpy(), after.scalars.to_numpy())
    assert np.allclose(before.scalars_sd.to_numpy(), after.scalars_sd.to_numpy())
    for name, curves in before.trajectories.items():
        assert np.allclose(curves, after.trajectories[name])
