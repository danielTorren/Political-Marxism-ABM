"""The network itself, and the fitted object that wraps it.

The emulator is an *ensemble of heteroscedastic multi-output MLPs*. One trunk reads the encoded
parameter vector; two pairs of heads come off it, predicting a mean and a standard deviation for
(a) the scalar outputs and (b) the trajectory basis coefficients. Sharing the trunk is deliberate
and is the main reason to use a network here rather than an independent Gaussian process per
output: the model's outputs are not independent quantities that happen to share inputs, they are
different views of one process. Conversion share, leasehold share, concentration and the wage all
move together, and a shared representation gets to use that.

Two uncertainties are kept apart, because they answer different questions and are used for
different things.

*Aleatoric* -- the sigma each network predicts -- is the ABM's own seed-to-seed spread at those
parameters. It is irreducible: more ABM runs will not shrink it, and it is what turns the
emulator into a likelihood for calibration.

*Epistemic* -- the disagreement between ensemble members' means -- is the emulator's ignorance,
and *is* reducible by running the ABM at more points. It is what an active-learning loop should
sample against, and what says whether a prediction in a thinly covered corner should be believed.

``Emulator.predict`` returns both, and their sum in quadrature as the total predictive spread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from emulator_data import Encoder, TrajectoryBasis

ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class EmulatorNet(nn.Module):
    """Shared trunk, four heads. Everything it sees and emits is in standardised units."""

    def __init__(
        self,
        n_inputs: int,
        n_scalars: int,
        n_coeffs: int,
        hidden: list[int],
        activation: str = "silu",
        layer_norm: bool = True,
        dropout: float = 0.0,
        min_sigma: float = 0.02,
        max_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        act = ACTIVATIONS.get(activation)
        if act is None:
            raise SystemExit(f"unknown activation {activation!r}; use one of {sorted(ACTIVATIONS)}")

        layers: list[nn.Module] = []
        width = n_inputs
        for size in hidden:
            layers.append(nn.Linear(width, size))
            if layer_norm:
                layers.append(nn.LayerNorm(size))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = size
        self.trunk = nn.Sequential(*layers)

        self.scalar_mu = nn.Linear(width, n_scalars) if n_scalars else None
        self.scalar_raw_sigma = nn.Linear(width, n_scalars) if n_scalars else None
        self.coeff_mu = nn.Linear(width, n_coeffs) if n_coeffs else None
        self.coeff_raw_sigma = nn.Linear(width, n_coeffs) if n_coeffs else None
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)

        # Start every sigma near 1 in standardised units, i.e. "I know nothing yet". Starting it
        # near the floor makes the first epochs' NLL enormous wherever the mean is wrong, and the
        # optimiser spends them repairing sigma rather than learning the response surface.
        for head in (self.scalar_raw_sigma, self.coeff_raw_sigma):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, 0.5413)  # softplus(0.5413) ~= 1.0

    def _sigma(self, raw: torch.Tensor) -> torch.Tensor:
        return (self.min_sigma + F.softplus(raw)).clamp(max=self.max_sigma)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        out: dict[str, torch.Tensor] = {}
        if self.scalar_mu is not None:
            out["scalar_mu"] = self.scalar_mu(h)
            out["scalar_sigma"] = self._sigma(self.scalar_raw_sigma(h))
        if self.coeff_mu is not None:
            out["coeff_mu"] = self.coeff_mu(h)
            out["coeff_sigma"] = self._sigma(self.coeff_raw_sigma(h))
        return out


def gaussian_nll(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked Gaussian negative log-likelihood, averaged over the observed entries.

    ``mask`` is False wherever the corpus has no value -- a run that ended early, a metric the
    model could not compute on a degenerate state. Those entries contribute nothing, rather than
    being imputed: an imputed target is a fact the emulator did not learn from the ABM.
    """
    if mask.sum() == 0:
        return mu.sum() * 0.0
    z = (target - mu) / sigma
    per_entry = 0.5 * z**2 + torch.log(sigma)
    return (per_entry * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------------------------
# The fitted object
# ---------------------------------------------------------------------------------------------
@dataclass
class Prediction:
    """What the emulator returns. Every frame is indexed like the input design frame."""

    scalars: pd.DataFrame            # posterior mean
    scalars_sd: pd.DataFrame         # total predictive sd
    scalars_sd_aleatoric: pd.DataFrame
    scalars_sd_epistemic: pd.DataFrame
    trajectories: dict[str, np.ndarray]        # {series: (n_rows, n_steps)} posterior mean
    trajectory_bands: dict[str, np.ndarray]    # {series: (n_rows, n_steps, 2)} lo/hi quantiles
    band_level: float


@dataclass
class Emulator:
    """A trained ensemble plus everything needed to encode inputs and decode outputs."""

    members: list[EmulatorNet]
    encoder: Encoder
    basis: TrajectoryBasis
    scalar_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    c_mean: np.ndarray
    c_std: np.ndarray
    y_lo: np.ndarray                 # per-scalar observed range in the training corpus, used to
    y_hi: np.ndarray                 # keep a predicted share inside [0, 1] and a count positive
    config: dict
    n_steps: int

    # -- use ---------------------------------------------------------------------------------
    def encode(self, design: pd.DataFrame) -> torch.Tensor:
        raw = self.encoder.encode(design)
        return torch.as_tensor((raw - self.x_mean) / self.x_std, dtype=torch.float32)

    @torch.no_grad()
    def raw_outputs(self, design: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-member means and sigmas in standardised units. ``(n_members, n_rows, n_outputs)``."""
        x = self.encode(design)
        keys = ("scalar_mu", "scalar_sigma", "coeff_mu", "coeff_sigma")
        stacks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        for member in self.members:
            member.eval()
            out = member(x)
            for key in keys:
                if key in out:
                    stacks[key].append(out[key].numpy())
        return {k: np.stack(v) for k, v in stacks.items() if v}

    def predict(
        self, design: pd.DataFrame, draws: int = 256, band_level: float = 0.9, seed: int = 0
    ) -> Prediction:
        """Posterior mean and spread for every scalar, and reconstructed trajectories.

        Trajectory bands are Monte-Carlo rather than analytic. The basis is linear but the
        transform is not, so a symmetric interval on the coefficients is not a symmetric interval
        on a share; sampling coefficients from the ensemble mixture and inverting each draw gives
        an interval that respects [0, 1] by construction.
        """
        out = self.raw_outputs(design)
        rng = np.random.default_rng(seed)
        index = design.index

        # --- scalars: a Gaussian mixture over ensemble members ------------------------------
        mu, sigma = out["scalar_mu"], out["scalar_sigma"]
        mean_std = mu.mean(axis=0)
        aleatoric_std = np.sqrt((sigma**2).mean(axis=0))
        epistemic_std = mu.std(axis=0) if len(self.members) > 1 else np.zeros_like(mean_std)

        mean = mean_std * self.y_std + self.y_mean
        mean = np.clip(mean, self.y_lo, self.y_hi)
        aleatoric = aleatoric_std * self.y_std
        epistemic = epistemic_std * self.y_std
        total = np.sqrt(aleatoric**2 + epistemic**2)

        frames = {
            name: pd.DataFrame(values, columns=self.scalar_names, index=index)
            for name, values in (
                ("mean", mean), ("total", total), ("aleatoric", aleatoric), ("epistemic", epistemic)
            )
        }

        # --- trajectories: sample, reconstruct, summarise ------------------------------------
        trajectories: dict[str, np.ndarray] = {}
        bands: dict[str, np.ndarray] = {}
        if "coeff_mu" in out and self.basis.bases:
            c_mu, c_sigma = out["coeff_mu"], out["coeff_sigma"]
            n_members, n_rows, n_coeffs = c_mu.shape
            pick = rng.integers(0, n_members, size=(draws, n_rows))
            noise = rng.standard_normal((draws, n_rows, n_coeffs))
            take_mu = np.take_along_axis(c_mu, pick[:, :, None], axis=0) if n_members > 1 else np.repeat(c_mu, draws, axis=0)
            take_sd = np.take_along_axis(c_sigma, pick[:, :, None], axis=0) if n_members > 1 else np.repeat(c_sigma, draws, axis=0)
            sampled = (take_mu + noise * take_sd) * self.c_std + self.c_mean

            flat = sampled.reshape(draws * n_rows, n_coeffs)
            curves = self.basis.reconstruct(flat)
            tail = (1.0 - band_level) / 2.0
            for name, values in curves.items():
                shaped = values.reshape(draws, n_rows, -1)
                trajectories[name] = shaped.mean(axis=0)
                bands[name] = np.stack(
                    [
                        np.quantile(shaped, tail, axis=0),
                        np.quantile(shaped, 1.0 - tail, axis=0),
                    ],
                    axis=-1,
                )

        return Prediction(
            scalars=frames["mean"],
            scalars_sd=frames["total"],
            scalars_sd_aleatoric=frames["aleatoric"],
            scalars_sd_epistemic=frames["epistemic"],
            trajectories=trajectories,
            trajectory_bands=bands,
            band_level=band_level,
        )

    # -- cheap paths, for the millions of evaluations sensitivity and calibration need ------
    def predict_scalars(self, design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean and total sd for the scalars only, as arrays. No Monte Carlo.

        :meth:`predict` reconstructs trajectories by sampling, which is right for a handful of
        curves and hopeless for the 10^6 evaluations a Sobol' decomposition on the surrogate
        needs. This path costs one forward pass per member.
        """
        out = self.raw_outputs(design)
        mu, sigma = out["scalar_mu"], out["scalar_sigma"]
        mean = mu.mean(axis=0) * self.y_std + self.y_mean
        aleatoric = np.sqrt((sigma**2).mean(axis=0)) * self.y_std
        epistemic = (mu.std(axis=0) if len(self.members) > 1 else np.zeros_like(mu[0])) * self.y_std
        return np.clip(mean, self.y_lo, self.y_hi), np.sqrt(aleatoric**2 + epistemic**2)

    def predict_trajectory_mean(self, design: pd.DataFrame) -> dict[str, np.ndarray]:
        """Trajectories reconstructed from the ensemble-mean coefficients. No Monte Carlo.

        Not identical to the mean of :meth:`predict`'s sampled curves: the basis is linear but
        the transform is not, so reconstructing the mean coefficient is not the mean of the
        reconstructions. The difference is second-order and irrelevant to a variance
        decomposition taken *over parameters*, which is what this path exists for; use
        :meth:`predict` wherever the curve itself is the answer.
        """
        out = self.raw_outputs(design)
        if "coeff_mu" not in out:
            return {}
        coeffs = out["coeff_mu"].mean(axis=0) * self.c_std + self.c_mean
        return self.basis.reconstruct(coeffs)

    # -- persistence -------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(self.members):
            torch.save(member.state_dict(), path / f"member_{index}.pt")
        np.savez(
            path / "stats.npz",
            x_mean=self.x_mean, x_std=self.x_std,
            y_mean=self.y_mean, y_std=self.y_std,
            c_mean=self.c_mean, c_std=self.c_std,
            y_lo=self.y_lo, y_hi=self.y_hi,
        )
        (path / "basis.json").write_text(json.dumps(self.basis.to_dict()), encoding="utf-8")
        (path / "emulator.json").write_text(
            json.dumps(
                {
                    "scalar_names": self.scalar_names,
                    "n_members": len(self.members),
                    "n_steps": self.n_steps,
                    "encoder": {
                        "cont_names": self.encoder.cont_names,
                        "switch_names": self.encoder.switch_names,
                        "switch_levels": self.encoder.switch_levels,
                        "bounds": self.encoder.bounds.tolist(),
                    },
                    "config": self.config,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Emulator":
        path = Path(path)
        meta = json.loads((path / "emulator.json").read_text(encoding="utf-8"))
        stats = np.load(path / "stats.npz")
        basis = TrajectoryBasis.from_dict(json.loads((path / "basis.json").read_text(encoding="utf-8")))
        encoder = Encoder(
            cont_names=meta["encoder"]["cont_names"],
            switch_names=meta["encoder"]["switch_names"],
            switch_levels=meta["encoder"]["switch_levels"],
            bounds=np.asarray(meta["encoder"]["bounds"], dtype=float),
        )
        model_cfg = meta["config"].get("model", {})
        members = []
        for index in range(int(meta["n_members"])):
            net = EmulatorNet(
                n_inputs=len(stats["x_mean"]),
                n_scalars=len(meta["scalar_names"]),
                n_coeffs=basis.n_coeffs,
                hidden=list(model_cfg.get("hidden", [256, 256, 256])),
                activation=model_cfg.get("activation", "silu"),
                layer_norm=bool(model_cfg.get("layer_norm", True)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                min_sigma=float(model_cfg.get("min_sigma", 0.02)),
                max_sigma=float(model_cfg.get("max_sigma", 5.0)),
            )
            net.load_state_dict(torch.load(path / f"member_{index}.pt", map_location="cpu"))
            net.eval()
            members.append(net)

        return cls(
            members=members,
            encoder=encoder,
            basis=basis,
            scalar_names=list(meta["scalar_names"]),
            x_mean=stats["x_mean"], x_std=stats["x_std"],
            y_mean=stats["y_mean"], y_std=stats["y_std"],
            c_mean=stats["c_mean"], c_std=stats["c_std"],
            y_lo=stats["y_lo"], y_hi=stats["y_hi"],
            config=meta["config"],
            n_steps=int(meta["n_steps"]),
        )
