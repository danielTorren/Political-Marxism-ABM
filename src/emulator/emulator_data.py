"""Corpus loading, input encoding, trajectory compression and splits.

Shared by :mod:`emulator_train`, :mod:`emulator_apply` and :mod:`emulator_plot`, so that the
encoding used at prediction time is by construction the one used at training time. Everything
needed to reproduce it -- the feature order, the standardisation statistics, the per-series
transform and PCA basis -- is written to the fitted-model directory as arrays, not re-derived.

Three things in here are less obvious than they look.

**Splits are by design point, never by run.** All replicate seeds of a point go into the same
fold. Splitting by run would put seed 0 of a point in train and seed 1 in test, and the reported
test error would then be measuring the model's seed noise instead of the emulator's
generalisation error -- flattering, and wrong in the direction that matters.

**Trajectories are transformed before they are compressed.** A share compressed and reconstructed
in its raw units can come back outside [0, 1], and an output series can come back negative. The
transform (logit for shares, log1p for non-negative quantities) puts every series on an
unbounded scale where a linear basis is well behaved, and the inverse puts the prediction back
inside the feasible set automatically rather than by clipping after the fact.

**The noise floor is computed, not assumed.** With replicate seeds at every point, the variance
of an output splits into a between-point part the emulator can in principle explain and a
within-point part no function of the parameters can. :func:`noise_floor` reports the second as a
share of the total, which is the ceiling on R^2 for that output. An emulator at R^2 = 0.72 on an
output whose ceiling is 0.75 is close to perfect; on one whose ceiling is 0.99 it is poor. The
two cases are indistinguishable without this number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Distance from 0 and 1 that a share is clipped to before the logit. Small enough not to distort
#: the interior, large enough that a series pinned at exactly 0 for its first fifty periods --
#: which ``conversion_share`` always is -- maps to a finite number rather than -inf.
LOGIT_EPS = 1e-4


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
@dataclass
class Corpus:
    """A generated corpus, as written by :mod:`emulator_gen`."""

    run_dir: Path
    design: pd.DataFrame          # one row per design point: parameters and switches
    runs: pd.DataFrame            # one row per (point, seed): the scalar outputs
    traj: np.ndarray              # (n_runs, n_series, n_steps), aligned with ``runs``
    series: list[str]             # names of the trajectory series, in array order
    space: dict                   # the design-space record written by emulator_gen
    partial: bool                 # corpus from a sweep that had not finished

    @property
    def n_steps(self) -> int:
        return int(self.traj.shape[2])

    @property
    def points(self) -> np.ndarray:
        return self.runs["point"].to_numpy()


def load_corpus(run_dir: Path, drop_failed: bool = True) -> Corpus:
    """Read ``output_data`` of a generation run directory.

    ``drop_failed`` removes runs that raised or whose parameters ``Params`` rejected. They are
    counted first: a design region that fails *systematically* is a finding about the model, not
    a data-cleaning step, and the caller prints the count.
    """
    run_dir = Path(run_dir)
    outdir = run_dir / "output_data" if (run_dir / "output_data").is_dir() else run_dir
    indir = run_dir / "input_data"

    design = pd.read_csv(outdir / "X.csv")
    runs = pd.read_csv(outdir / "runs.csv")
    blob = np.load(outdir / "trajectories.npz", allow_pickle=True)
    traj = blob["values"]
    series = [str(s) for s in blob["series"]]
    partial = bool(blob["partial"])
    space = json.loads((indir / "space.json").read_text(encoding="utf-8"))

    # The array is written in the same row order as runs.csv, but check rather than trust: a
    # silent misalignment here would train the emulator on other points' trajectories.
    if len(traj) != len(runs):
        raise SystemExit(
            f"corpus is inconsistent: {len(runs)} scalar rows against {len(traj)} trajectory "
            f"blocks in {outdir}"
        )
    if not np.array_equal(blob["point"], runs["point"].to_numpy()):
        raise SystemExit(f"corpus row order differs between runs.csv and trajectories.npz in {outdir}")

    if drop_failed and "failed" in runs.columns:
        keep = runs["failed"].to_numpy() == 0
        runs, traj = runs.loc[keep].reset_index(drop=True), traj[keep]

    return Corpus(run_dir, design, runs, traj, series, space, partial)


# ---------------------------------------------------------------------------------------------
# Input encoding
# ---------------------------------------------------------------------------------------------
@dataclass
class Encoder:
    """Design rows -> the network's input matrix.

    Continuous parameters pass through as they are and are standardised by the caller; boolean
    switches become +/-1; multi-level switches become one-hot blocks. The feature order is fixed
    at construction and stored with the fitted model, so a prediction made months later encodes
    its inputs the way the network was trained to read them.
    """

    cont_names: list[str]
    switch_names: list[str]
    switch_levels: dict[str, list]
    bounds: np.ndarray

    @classmethod
    def from_space(cls, space: dict) -> "Encoder":
        return cls(
            cont_names=list(space["cont_names"]),
            switch_names=list(space.get("switch_names") or []),
            switch_levels={k: list(v) for k, v in (space.get("switch_levels") or {}).items()},
            bounds=np.asarray(space["bounds"], dtype=float),
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.cont_names)
        for switch in self.switch_names:
            levels = self.switch_levels[switch]
            if len(levels) == 2 and all(isinstance(v, (bool, np.bool_)) for v in levels):
                names.append(switch)
            else:
                names.extend(f"{switch}={level}" for level in levels)
        return names

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """``(n_rows, n_features)``. Accepts any frame carrying the design columns."""
        missing = [c for c in self.cont_names + self.switch_names if c not in frame.columns]
        if missing:
            raise SystemExit(f"design columns missing from input: {', '.join(missing)}")

        blocks = [frame[self.cont_names].to_numpy(dtype=float)]
        for switch in self.switch_names:
            levels = self.switch_levels[switch]
            column = frame[switch].to_numpy()
            if len(levels) == 2 and all(isinstance(v, (bool, np.bool_)) for v in levels):
                truth = np.asarray([_as_bool(v) for v in column], dtype=float)
                blocks.append((2.0 * truth - 1.0).reshape(-1, 1))
            else:
                onehot = np.zeros((len(column), len(levels)))
                index = {str(level): i for i, level in enumerate(levels)}
                for row, value in enumerate(column):
                    key = str(value)
                    if key not in index:
                        raise SystemExit(f"unknown level {value!r} for switch {switch!r}")
                    onehot[row, index[key]] = 1.0
                blocks.append(onehot)
        return np.hstack(blocks)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def expand_design(runs: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """Join each run onto its design point, so every run carries its parameters."""
    return runs[["point", "seed"]].merge(design, on="point", how="left")


# ---------------------------------------------------------------------------------------------
# Trajectory compression
# ---------------------------------------------------------------------------------------------
def apply_transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind in ("none", "identity", None):
        return values
    if kind == "log1p":
        return np.log1p(np.clip(values, 0.0, None))
    if kind == "logit":
        clipped = np.clip(values, LOGIT_EPS, 1.0 - LOGIT_EPS)
        return np.log(clipped / (1.0 - clipped))
    raise SystemExit(f"unknown transform {kind!r}; use none, log1p or logit")


def invert_transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind in ("none", "identity", None):
        return values
    if kind == "log1p":
        return np.expm1(values)
    if kind == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    raise SystemExit(f"unknown transform {kind!r}; use none, log1p or logit")


@dataclass
class SeriesBasis:
    """Transform + mean + truncated PCA basis for one trajectory series."""

    name: str
    transform: str
    mean: np.ndarray              # (n_steps,)
    components: np.ndarray        # (n_components, n_steps)
    explained: np.ndarray         # cumulative variance share at each retained component

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def project(self, curves: np.ndarray) -> np.ndarray:
        """``(n_runs, n_steps)`` in raw units -> ``(n_runs, n_components)`` coefficients."""
        centred = apply_transform(curves, self.transform) - self.mean
        return centred @ self.components.T

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """Coefficients -> ``(n_runs, n_steps)`` back in raw units."""
        return invert_transform(coeffs @ self.components + self.mean, self.transform)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "transform": self.transform,
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "explained": self.explained.tolist(),
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "SeriesBasis":
        return cls(
            name=blob["name"],
            transform=blob["transform"],
            mean=np.asarray(blob["mean"], dtype=float),
            components=np.asarray(blob["components"], dtype=float),
            explained=np.asarray(blob["explained"], dtype=float),
        )


def fit_series_basis(
    name: str,
    curves: np.ndarray,
    transform: str,
    variance_target: float = 0.999,
    max_components: int = 12,
    min_components: int = 3,
) -> SeriesBasis:
    """PCA over the transformed curves of one series, truncated at ``variance_target``.

    Fitted on complete rows only: a run that ended early leaves NaN, and an incomplete curve
    would otherwise have to be imputed, which is exactly the fabrication the masked loss exists
    to avoid.
    """
    transformed = apply_transform(curves, transform)
    complete = np.isfinite(transformed).all(axis=1)
    if complete.sum() < 2:
        raise SystemExit(f"series {name!r} has fewer than two complete runs in the corpus")

    matrix = transformed[complete]
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    # Economy SVD: n_steps is small (200) and n_runs large, so this is cheap and exact.
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / max(variance.sum(), 1e-30)
    keep = int(np.searchsorted(cumulative, variance_target) + 1)
    keep = int(np.clip(keep, min_components, min(max_components, right.shape[0])))
    return SeriesBasis(name, transform, mean, right[:keep], cumulative[:keep])


@dataclass
class TrajectoryBasis:
    """The per-series bases together, with the flat coefficient layout the network predicts."""

    bases: list[SeriesBasis]

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.bases]

    @property
    def n_coeffs(self) -> int:
        return sum(b.n_components for b in self.bases)

    @property
    def slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for basis in self.bases:
            out[basis.name] = slice(start, start + basis.n_components)
            start += basis.n_components
        return out

    @property
    def coeff_names(self) -> list[str]:
        return [f"{b.name}:pc{k}" for b in self.bases for k in range(b.n_components)]

    def project(self, traj: np.ndarray, series: list[str]) -> np.ndarray:
        """``(n_runs, n_series, n_steps)`` -> ``(n_runs, n_coeffs)``, NaN where incomplete."""
        index = {name: i for i, name in enumerate(series)}
        out = np.full((traj.shape[0], self.n_coeffs), np.nan)
        for basis, span in ((b, self.slices[b.name]) for b in self.bases):
            curves = traj[:, index[basis.name], :]
            complete = np.isfinite(curves).all(axis=1)
            out[complete, span] = basis.project(curves[complete])
        return out

    def reconstruct(self, coeffs: np.ndarray) -> dict[str, np.ndarray]:
        """``(n_runs, n_coeffs)`` -> ``{series: (n_runs, n_steps)}`` in raw units."""
        spans = self.slices
        return {b.name: b.reconstruct(coeffs[:, spans[b.name]]) for b in self.bases}

    def to_dict(self) -> dict:
        return {"bases": [b.to_dict() for b in self.bases]}

    @classmethod
    def from_dict(cls, blob: dict) -> "TrajectoryBasis":
        return cls([SeriesBasis.from_dict(b) for b in blob["bases"]])


def fit_trajectory_basis(
    traj: np.ndarray,
    series: list[str],
    transforms: dict[str, str],
    variance_target: float = 0.999,
    max_components: int = 12,
    min_components: int = 3,
) -> TrajectoryBasis:
    """Fit a basis per requested series. Fit on *training rows only*, never the whole corpus."""
    index = {name: i for i, name in enumerate(series)}
    bases = []
    for name, transform in transforms.items():
        if name not in index:
            raise SystemExit(f"series {name!r} is not in the corpus; it has {', '.join(series)}")
        bases.append(
            fit_series_basis(
                name, traj[:, index[name], :], transform,
                variance_target, max_components, min_components,
            )
        )
    return TrajectoryBasis(bases)


# ---------------------------------------------------------------------------------------------
# Splits and the noise floor
# ---------------------------------------------------------------------------------------------
def split_by_point(
    points: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, np.ndarray]:
    """Row masks for train/val/test, split on the *design point* so replicates stay together."""
    unique = np.unique(points)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = int(round(len(unique) * test_fraction))
    n_val = int(round(len(unique) * val_fraction))
    test_points = set(shuffled[:n_test].tolist())
    val_points = set(shuffled[n_test : n_test + n_val].tolist())

    in_test = np.asarray([p in test_points for p in points])
    in_val = np.asarray([p in val_points for p in points])
    return {"train": ~(in_test | in_val), "val": in_val, "test": in_test}


def noise_floor(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Within-point share of total variance, per column: the ceiling on any emulator's R^2.

    ``1 - noise_floor`` is the largest R^2 a perfect function of the parameters could achieve on
    this corpus, because the remainder is variation between replicate seeds at *identical*
    parameters. Columns whose points have only one seed return NaN.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    frame = pd.DataFrame(values)
    frame["point"] = points
    grouped = frame.groupby("point")
    counts = grouped.size()
    if (counts > 1).sum() == 0:
        return np.full(values.shape[1], np.nan)

    within = grouped[list(range(values.shape[1]))].var(ddof=1).mean(axis=0).to_numpy()
    total = np.nanvar(values, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(within / np.where(total > 0, total, np.nan), 0.0, 1.0)


def standardiser(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-column mean and standard deviation, with zero-variance columns left alone."""
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return np.where(np.isfinite(mean), mean, 0.0), std
