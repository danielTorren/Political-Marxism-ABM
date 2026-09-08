"""Fit the emulator to a generated corpus.

    uv run python src/emulator/emulator_train.py --run-dir Results/emulator/2026-09-08_143012
    uv run python src/emulator/emulator_train.py --run-dir <dir> --epochs 50 --ensemble 2

Reads the corpus written by :mod:`emulator_gen` and the settings in ``constants/training.yaml``,
and writes a fitted ensemble plus its diagnostics into ``<run-dir>/emulator/``:

    emulator.json, basis.json, stats.npz, member_*.pt   the fitted object (Emulator.load)
    metrics_scalars.csv        per-output test accuracy, against the noise floor
    metrics_trajectories.csv   per-series test accuracy and basis dimension
    coverage.csv               are the predictive intervals the width they claim to be
    learning_curve.csv         would more ABM runs still help
    baselines.csv              what ridge and a random forest achieve on the same split
    predictions.npz            held-out predictions, for emulator_plot.py

Every accuracy figure is reported *twice*: raw R^2, and R^2 as a share of the ceiling implied by
the ABM's own seed noise at repeated design points. The second is the one to read. An output
whose replicate spread accounts for a third of its variance cannot be predicted past R^2 = 0.67
by anything, and a raw 0.65 there is a near-perfect emulator rather than a mediocre one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from emulator_data import (  # noqa: E402
    Encoder,
    Corpus,
    expand_design,
    fit_trajectory_basis,
    load_corpus,
    noise_floor,
    split_by_point,
    standardiser,
)
from emulator_net import Emulator, EmulatorNet, gaussian_nll  # noqa: E402

DEFAULT_TRAINING = HERE / "constants" / "training.yaml"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------------------------
# Assembling the training arrays
# ---------------------------------------------------------------------------------------------
def assemble(corpus: Corpus, cfg: dict) -> dict:
    """Encode inputs, split by design point, fit the basis on train rows, standardise.

    The order matters and is the reason this is one function: the PCA basis is fitted on the
    training rows *only*. Fitting it on the whole corpus before splitting would let the test rows
    influence the representation their own predictions are scored in -- a mild leak, but exactly
    the kind that makes a surrogate look better than it is.
    """
    data_cfg = cfg.get("data", {})
    pca_cfg = cfg.get("pca", {})

    encoder = Encoder.from_space(corpus.space)
    design_rows = expand_design(corpus.runs, corpus.design)
    x_raw = encoder.encode(design_rows)

    scalar_names = [s for s in corpus.space["scalars"] if s in corpus.runs.columns]
    dropped = [s for s in corpus.space["scalars"] if s not in corpus.runs.columns]
    if dropped:
        print(f"  ! scalars absent from the corpus, dropped: {', '.join(dropped)}", file=sys.stderr)
    y_raw = corpus.runs[scalar_names].to_numpy(dtype=float)

    masks = split_by_point(
        corpus.points,
        float(data_cfg.get("val_fraction", 0.1)),
        float(data_cfg.get("test_fraction", 0.1)),
        int(data_cfg.get("split_seed", 7)),
    )

    transforms = dict(corpus.space.get("trajectories") or {})
    transforms = {k: v for k, v in transforms.items() if k in corpus.series}
    basis = fit_trajectory_basis(
        corpus.traj[masks["train"]],
        corpus.series,
        transforms,
        variance_target=float(pca_cfg.get("variance_target", 0.999)),
        max_components=int(pca_cfg.get("max_components", 12)),
        min_components=int(pca_cfg.get("min_components", 3)),
    )
    c_raw = basis.project(corpus.traj, corpus.series)

    train = masks["train"]
    x_mean, x_std = standardiser(x_raw[train])
    y_mean, y_std = standardiser(y_raw[train])
    c_mean, c_std = standardiser(c_raw[train])
    if not bool(data_cfg.get("standardise_inputs", True)):
        x_mean, x_std = np.zeros_like(x_mean), np.ones_like(x_std)
    if not bool(data_cfg.get("standardise_targets", True)):
        y_mean, y_std = np.zeros_like(y_mean), np.ones_like(y_std)
        c_mean, c_std = np.zeros_like(c_mean), np.ones_like(c_std)

    return {
        "encoder": encoder,
        "basis": basis,
        "scalar_names": scalar_names,
        "masks": masks,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "c_raw": c_raw,
        "x": (x_raw - x_mean) / x_std,
        "y": (y_raw - y_mean) / y_std,
        "c": (c_raw - c_mean) / c_std,
        "stats": (x_mean, x_std, y_mean, y_std, c_mean, c_std),
        "y_lo": np.nanmin(y_raw[train], axis=0),
        "y_hi": np.nanmax(y_raw[train], axis=0),
        "design_rows": design_rows,
    }


# ---------------------------------------------------------------------------------------------
# Fitting one ensemble member
# ---------------------------------------------------------------------------------------------
def fit_member(
    data: dict,
    cfg: dict,
    seed: int,
    device: torch.device,
    train_mask: np.ndarray,
    verbose: bool = True,
) -> tuple[EmulatorNet, list[dict]]:
    """Fit one network. Returns it with the best validation weights restored, plus its history."""
    model_cfg, train_cfg = cfg.get("model", {}), cfg.get("train", {})
    torch.manual_seed(seed)

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.nan_to_num(array, nan=0.0), dtype=torch.float32, device=device)

    def mask_of(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.isfinite(array), dtype=torch.float32, device=device)

    val_mask = data["masks"]["val"]
    x_tr, x_va = tensor(data["x"][train_mask]), tensor(data["x"][val_mask])
    y_tr, y_va = tensor(data["y"][train_mask]), tensor(data["y"][val_mask])
    c_tr, c_va = tensor(data["c"][train_mask]), tensor(data["c"][val_mask])
    my_tr, my_va = mask_of(data["y"][train_mask]), mask_of(data["y"][val_mask])
    mc_tr, mc_va = mask_of(data["c"][train_mask]), mask_of(data["c"][val_mask])

    net = EmulatorNet(
        n_inputs=x_tr.shape[1],
        n_scalars=y_tr.shape[1],
        n_coeffs=c_tr.shape[1],
        hidden=list(model_cfg.get("hidden", [256, 256, 256])),
        activation=model_cfg.get("activation", "silu"),
        layer_norm=bool(model_cfg.get("layer_norm", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        min_sigma=float(model_cfg.get("min_sigma", 0.02)),
        max_sigma=float(model_cfg.get("max_sigma", 5.0)),
    ).to(device)

    epochs = int(train_cfg.get("epochs", 400))
    batch_size = int(train_cfg.get("batch_size", 256))
    warmup = int(train_cfg.get("warmup_epochs", 10))
    patience = int(train_cfg.get("patience", 60))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))
    weights = train_cfg.get("loss_weights", {}) or {}
    w_scalar = float(weights.get("scalars", 1.0))
    w_traj = float(weights.get("trajectories", 1.0))

    optimiser = torch.optim.AdamW(
        net.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    scheduler_name = str(train_cfg.get("scheduler", "cosine"))

    def loss_of(out: dict, y, my, c, mc) -> torch.Tensor:
        total = torch.zeros((), device=device)
        if "scalar_mu" in out:
            total = total + w_scalar * gaussian_nll(out["scalar_mu"], out["scalar_sigma"], y, my)
        if "coeff_mu" in out:
            total = total + w_traj * gaussian_nll(out["coeff_mu"], out["coeff_sigma"], c, mc)
        return total

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_train = x_tr.shape[0]
    best = {"val": np.inf, "epoch": -1, "state": None}
    history: list[dict] = []

    for epoch in range(epochs):
        # Linear warmup then cosine decay: the heteroscedastic head is unstable in the first few
        # hundred steps, when a large learning rate can drive sigma to its floor before the mean
        # is anywhere near right, and the NLL then has no gradient left to recover with.
        if epoch < warmup:
            factor = (epoch + 1) / max(warmup, 1)
        elif scheduler_name == "cosine":
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            factor = 1.0
        for group in optimiser.param_groups:
            group["lr"] = float(train_cfg.get("lr", 1e-3)) * max(factor, 1e-3)

        net.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        running = 0.0
        for start in range(0, n_train, batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_of(net(x_tr[batch]), y_tr[batch], my_tr[batch], c_tr[batch], mc_tr[batch])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimiser.step()
            running += float(loss) * len(batch)

        net.eval()
        with torch.no_grad():
            val = float(loss_of(net(x_va), y_va, my_va, c_va, mc_va)) if len(x_va) else running / n_train
        history.append({"epoch": epoch, "train_nll": running / n_train, "val_nll": val})

        if val < best["val"] - 1e-5:
            best = {
                "val": val,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in net.state_dict().items()},
            }
        elif patience and epoch - best["epoch"] >= patience:
            if verbose:
                print(f"      early stop at epoch {epoch} (best {best['epoch']}, val {best['val']:.4f})")
            break

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch:4d}  train {running / n_train:8.4f}  val {val:8.4f}")

    if best["state"] is not None:
        net.load_state_dict(best["state"])
    net.eval()
    return net.to("cpu"), history


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def _r2(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Column-wise R^2, NaN-aware, against the variance of the *held-out* rows."""
    residual = np.nansum((actual - predicted) ** 2, axis=0)
    total = np.nansum((actual - np.nanmean(actual, axis=0)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - residual / np.where(total > 0, total, np.nan)


def score_scalars(
    emulator: Emulator, data: dict, corpus: Corpus, mask: np.ndarray, levels: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-output accuracy on held-out points, reported against the seed-noise ceiling."""
    design = data["design_rows"].loc[mask].reset_index(drop=True)
    actual = data["y_raw"][mask]
    prediction = emulator.predict(design, draws=1)   # draws only affect trajectories

    mean = prediction.scalars.to_numpy()
    sd = prediction.scalars_sd.to_numpy()
    floor = noise_floor(actual, corpus.points[mask])
    r2 = _r2(actual, mean)
    rmse = np.sqrt(np.nanmean((actual - mean) ** 2, axis=0))
    ceiling = 1.0 - floor

    metrics = pd.DataFrame(
        {
            "output": emulator.scalar_names,
            "rmse": rmse,
            "mae": np.nanmean(np.abs(actual - mean), axis=0),
            "r2": r2,
            "noise_floor": floor,
            "r2_ceiling": ceiling,
            # The figure to read: how much of the *explainable* variance was explained. Above 1
            # means the emulator beat the ceiling, which happens when the corpus is small enough
            # that the noise floor is itself noisily estimated -- treat it as "at the ceiling".
            "r2_of_ceiling": r2 / np.where(ceiling > 0, ceiling, np.nan),
            "sd_predicted_mean": np.nanmean(sd, axis=0),
            "sd_seed_observed": np.sqrt(floor * np.nanvar(actual, axis=0, ddof=1)),
        }
    )

    rows = []
    for level in levels:
        # Two-sided Gaussian interval of the stated level, scored against the held-out runs.
        z = float(np.sqrt(2.0) * torch.erfinv(torch.tensor(level)).item())
        inside = np.abs(actual - mean) <= z * sd
        rows.append(
            pd.DataFrame(
                {
                    "output": emulator.scalar_names,
                    "level": level,
                    "coverage": np.nanmean(inside.astype(float), axis=0),
                }
            )
        )
    coverage = pd.concat(rows, ignore_index=True)

    payload = {
        "scalar_actual": actual,
        "scalar_mean": mean,
        "scalar_sd": sd,
        "scalar_sd_aleatoric": prediction.scalars_sd_aleatoric.to_numpy(),
        "scalar_sd_epistemic": prediction.scalars_sd_epistemic.to_numpy(),
    }
    return metrics, coverage, payload


def score_trajectories(
    emulator: Emulator, data: dict, corpus: Corpus, mask: np.ndarray, draws: int = 128
) -> tuple[pd.DataFrame, dict]:
    """Per-series accuracy in raw units, plus the basis dimension each series needed."""
    design = data["design_rows"].loc[mask].reset_index(drop=True)
    prediction = emulator.predict(design, draws=draws)
    index = {name: i for i, name in enumerate(corpus.series)}

    rows, payload = [], {}
    for basis in emulator.basis.bases:
        actual = corpus.traj[mask][:, index[basis.name], :].astype(float)
        predicted = prediction.trajectories[basis.name]
        finite = np.isfinite(actual)
        residual = np.where(finite, actual - predicted, np.nan)

        # Pooled over rows and periods: one number for "how well is this curve predicted".
        flat_actual = actual[finite]
        flat_pred = predicted[finite]
        total = np.sum((flat_actual - flat_actual.mean()) ** 2)
        rows.append(
            {
                "series": basis.name,
                "transform": basis.transform,
                "n_components": basis.n_components,
                "variance_explained": float(basis.explained[-1]),
                "rmse": float(np.sqrt(np.nanmean(residual**2))),
                "mae": float(np.nanmean(np.abs(residual))),
                "r2_pooled": float(1.0 - np.sum((flat_actual - flat_pred) ** 2) / total)
                if total > 0 else np.nan,
                # A basis-truncation error floor: how well the *actual* curve is reproduced by
                # its own projection. The emulator cannot beat this, and if the two are equal the
                # limit is the basis, not the network -- raise pca.max_components, not the width.
                "r2_basis_ceiling": _basis_ceiling(basis, actual),
            }
        )
        payload[f"traj_actual_{basis.name}"] = actual.astype(np.float32)
        payload[f"traj_mean_{basis.name}"] = predicted.astype(np.float32)
        payload[f"traj_band_{basis.name}"] = prediction.trajectory_bands[basis.name].astype(np.float32)

    return pd.DataFrame(rows), payload


def _basis_ceiling(basis, actual: np.ndarray) -> float:
    complete = np.isfinite(actual).all(axis=1)
    if complete.sum() < 2:
        return float("nan")
    curves = actual[complete]
    rebuilt = basis.reconstruct(basis.project(curves))
    total = np.sum((curves - curves.mean()) ** 2)
    return float(1.0 - np.sum((curves - rebuilt) ** 2) / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------
def fit_baselines(data: dict, names: list[str], scalar_names: list[str]) -> pd.DataFrame:
    """Ridge and random forest on the same split, scored the same way.

    A surrogate paper that does not say what a linear model achieves has not shown that the
    network was necessary.

    Fitted *per output*, on the rows where that output is finite. The obvious multi-output fit on
    complete rows is not available: several metrics are undefined on some runs by construction --
    the variogram statistics need enough converted parcels to have a spatial pattern at all, and
    ``same_occupant_share`` needs at least one conversion -- so requiring every output at once
    throws away most of the corpus and silently scores the baselines on a biased remainder. The
    network handles the same problem with a masked loss; this is the equivalent.
    """
    if not names:
        return pd.DataFrame()
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import RidgeCV
    except ImportError:
        print("  ! scikit-learn not installed; baselines skipped", file=sys.stderr)
        return pd.DataFrame()

    train, test = data["masks"]["train"], data["masks"]["test"]
    models = {
        "ridge": lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0
        ),
    }
    unknown = [n for n in names if n not in models]
    if unknown:
        print(f"  ! unknown baseline(s) {', '.join(unknown)}, skipped", file=sys.stderr)

    rows = []
    for column, output in enumerate(scalar_names):
        finite = np.isfinite(data["y_raw"][:, column])
        x_tr, y_tr = data["x"][train & finite], data["y_raw"][train & finite, column]
        x_te, y_te = data["x"][test & finite], data["y_raw"][test & finite, column]
        if len(x_tr) < 10 or len(x_te) < 3:
            print(
                f"  ! {output}: {len(x_tr)} train / {len(x_te)} test rows are finite; "
                f"baselines skipped for this output",
                file=sys.stderr,
            )
            continue
        for name in names:
            if name not in models:
                continue
            predicted = models[name]().fit(x_tr, y_tr).predict(x_te)
            rows.append(
                {
                    "baseline": name,
                    "output": output,
                    "n_test": int(len(y_te)),
                    "r2": float(_r2(y_te[:, None], predicted[:, None])[0]),
                    "rmse": float(np.sqrt(np.mean((y_te - predicted) ** 2))),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def build_emulator(members: list[EmulatorNet], data: dict, cfg: dict, n_steps: int) -> Emulator:
    x_mean, x_std, y_mean, y_std, c_mean, c_std = data["stats"]
    return Emulator(
        members=members,
        encoder=data["encoder"],
        basis=data["basis"],
        scalar_names=data["scalar_names"],
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, c_mean=c_mean, c_std=c_std,
        y_lo=data["y_lo"], y_hi=data["y_hi"],
        config=cfg,
        n_steps=n_steps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="a directory from emulator_gen.py")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--outdir", type=Path, default=None, help="default: <run-dir>/emulator")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ensemble", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-learning-curve", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.training)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.ensemble is not None:
        cfg.setdefault("train", {})["ensemble"] = args.ensemble
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device

    corpus = load_corpus(args.run_dir, drop_failed=bool(cfg.get("data", {}).get("drop_failed", True)))
    outdir = Path(args.outdir or (Path(args.run_dir) / "emulator"))
    outdir.mkdir(parents=True, exist_ok=True)

    data = assemble(corpus, cfg)
    masks = data["masks"]
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))

    n_points = corpus.design.shape[0]
    print(f"Corpus     : {args.run_dir}{'  (PARTIAL sweep)' if corpus.partial else ''}")
    print(f"             {len(corpus.runs)} runs over {corpus.runs['point'].nunique()} of {n_points} points")
    print(f"Inputs     : {data['x'].shape[1]} features from {len(data['encoder'].cont_names)} "
          f"parameters + {len(data['encoder'].switch_names)} switches")
    print(f"Targets    : {len(data['scalar_names'])} scalars, "
          f"{data['basis'].n_coeffs} basis coefficients over {len(data['basis'].bases)} series")
    for basis in data["basis"].bases:
        print(f"               {basis.name:26s} {basis.n_components:2d} components, "
              f"{100 * basis.explained[-1]:.3f}% of variance")
    print(f"Split      : {masks['train'].sum()} train / {masks['val'].sum()} val / "
          f"{masks['test'].sum()} test runs, split by design point")
    print(f"Device     : {device}")

    # --- the ensemble ------------------------------------------------------------------------
    n_members = int(cfg.get("train", {}).get("ensemble", 5))
    base_seed = int(cfg.get("train", {}).get("seed", 0))
    members, histories = [], []
    started = time.perf_counter()
    for index in range(n_members):
        print(f"\n  member {index + 1}/{n_members}")
        net, history = fit_member(data, cfg, base_seed + index, device, masks["train"])
        members.append(net)
        histories.append(pd.DataFrame(history).assign(member=index))
    print(f"\nFitted {n_members} members in {(time.perf_counter() - started) / 60:.1f} min")

    emulator = build_emulator(members, data, cfg, corpus.n_steps)
    emulator.save(outdir)
    pd.concat(histories, ignore_index=True).to_csv(outdir / "training_history.csv", index=False)

    # --- scoring -----------------------------------------------------------------------------
    levels = [float(v) for v in cfg.get("report", {}).get("coverage_levels", [0.5, 0.9, 0.95])]
    metrics, coverage, scalar_payload = score_scalars(emulator, data, corpus, masks["test"], levels)
    traj_metrics, traj_payload = score_trajectories(emulator, data, corpus, masks["test"])
    metrics.to_csv(outdir / "metrics_scalars.csv", index=False)
    coverage.to_csv(outdir / "coverage.csv", index=False)
    traj_metrics.to_csv(outdir / "metrics_trajectories.csv", index=False)

    print("\nHeld-out accuracy, scalars (r2_of_ceiling is the figure to read):")
    print(f"  {'output':30s} {'r2':>7s} {'ceiling':>8s} {'of ceiling':>11s} {'rmse':>9s}")
    for row in metrics.itertuples(index=False):
        print(
            f"  {row.output:30s} {row.r2:7.3f} {row.r2_ceiling:8.3f} "
            f"{row.r2_of_ceiling:11.3f} {row.rmse:9.4g}"
        )
    print("\nHeld-out accuracy, trajectories:")
    print(f"  {'series':26s} {'pcs':>4s} {'r2':>7s} {'basis max':>10s} {'rmse':>9s}")
    for row in traj_metrics.itertuples(index=False):
        print(
            f"  {row.series:26s} {row.n_components:4d} {row.r2_pooled:7.3f} "
            f"{row.r2_basis_ceiling:10.3f} {row.rmse:9.4g}"
        )
    print("\nInterval coverage (nominal -> achieved, averaged over outputs):")
    for level, group in coverage.groupby("level"):
        print(f"  {level:.2f} -> {group['coverage'].mean():.3f}")

    np.savez_compressed(
        outdir / "predictions.npz",
        **scalar_payload,
        **traj_payload,
        scalar_names=np.asarray(emulator.scalar_names, dtype=object),
        series=np.asarray([b.name for b in emulator.basis.bases], dtype=object),
        point=corpus.points[masks["test"]],
        seed=corpus.runs["seed"].to_numpy()[masks["test"]],
    )

    # --- learning curve ----------------------------------------------------------------------
    fractions = [] if args.no_learning_curve else list(cfg.get("report", {}).get("learning_curve", []))
    if fractions:
        print("\nLearning curve (one member per fraction, to keep the cost down):")
        train_points = np.unique(corpus.points[masks["train"]])
        rng = np.random.default_rng(int(cfg.get("data", {}).get("split_seed", 7)))
        shuffled = rng.permutation(train_points)
        rows = []
        for fraction in fractions:
            keep = set(shuffled[: max(2, int(len(shuffled) * float(fraction)))].tolist())
            subset = np.asarray([p in keep for p in corpus.points]) & masks["train"]
            net, _ = fit_member(data, cfg, base_seed + 100, device, subset, verbose=False)
            partial = build_emulator([net], data, cfg, corpus.n_steps)
            part_metrics, _, _ = score_scalars(partial, data, corpus, masks["test"], levels)
            rows.append(
                {
                    "fraction": float(fraction),
                    "n_points": len(keep),
                    "n_runs": int(subset.sum()),
                    "mean_r2": float(part_metrics["r2"].mean()),
                    "mean_r2_of_ceiling": float(part_metrics["r2_of_ceiling"].mean()),
                }
            )
            print(f"  {fraction:5.3f}  {len(keep):5d} points  mean r2 {rows[-1]['mean_r2']:.3f}")
        pd.DataFrame(rows).to_csv(outdir / "learning_curve.csv", index=False)

    # --- baselines ---------------------------------------------------------------------------
    if not args.no_baselines:
        baselines = fit_baselines(
            data, list(cfg.get("report", {}).get("baselines", [])), data["scalar_names"]
        )
        if len(baselines):
            baselines.to_csv(outdir / "baselines.csv", index=False)
            summary = baselines.groupby("baseline")["r2"].mean()
            print("\nBaselines, mean r2 over the scalar outputs:")
            for name, value in summary.items():
                print(f"  {name:16s} {value:.3f}")
            print(f"  {'emulator':16s} {metrics['r2'].mean():.3f}")

    (outdir / "train_meta.json").write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "corpus_partial": corpus.partial,
                "n_runs": int(len(corpus.runs)),
                "n_points": int(corpus.runs["point"].nunique()),
                "split": {k: int(v.sum()) for k, v in masks.items()},
                "device": str(device),
                "training_yaml": str(args.training),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote the fitted emulator and its diagnostics to {outdir}/")
    print(
        f"\nNext:\n"
        f"    uv run python src/emulator/emulator_plot.py --model-dir {outdir}\n"
        f"    uv run python src/emulator/emulator_apply.py sobol --model-dir {outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
