"""Use a trained emulator: sensitivity at scale, parameter sweeps, and history matching.

    uv run python src/emulator/emulator_apply.py sobol     --model-dir <dir> --n-base 16384
    uv run python src/emulator/emulator_apply.py sobol     --model-dir <dir> --series conversion_share
    uv run python src/emulator/emulator_apply.py sweep     --model-dir <dir> --over theta
    uv run python src/emulator/emulator_apply.py sweep     --model-dir <dir> --over alpha_1 alpha_2
    uv run python src/emulator/emulator_apply.py predict   --model-dir <dir> --points rows.csv
    uv run python src/emulator/emulator_apply.py calibrate --model-dir <dir> --targets targets.yaml

Each subcommand writes into ``<model-dir>/apply/<name>/`` and prints where.

The point of the whole exercise is here. Every one of these asks the model a question that would
take days to weeks of ABM time and takes seconds on the surrogate:

``sobol``      the full variance decomposition at a sample size the ABM cannot reach. The direct
               sweep in ``src/sensitivity`` runs n_base * (D + 2) * replicates model runs -- at
               n_base=512 and 23 parameters that is ~51,000 runs -- and switches second-order
               indices off on cost grounds. Here n_base=16384 with second order costs ~800,000
               *network* evaluations, which is a minute. With ``--series`` it also produces
               something the direct sweep cannot afford at all: indices computed *at every
               period*, so a parameter's influence can be watched rising and falling over the
               transition rather than read once at the end.

``sweep``      the response along one or two parameters with everything else held at baseline,
               with the emulator's own uncertainty attached, for the paper's response figures.

``predict``    arbitrary parameter rows from a CSV, including trajectories -- the cheap stand-in
               for a scenario run.

``calibrate``  history matching. Sample the space, compare predicted outputs against targets
               (Clark/Kerridge/Allen-derived, or whatever the paper argues for), and return the
               parameter region that is *not implausible*. This is what turns the emulator from a
               fast copy of the model into an argument about the model: the question "which
               parameter settings could have produced England" has no tractable answer without a
               surrogate, and it is a stronger claim than any single calibrated run.

Every prediction carries two uncertainties (see :mod:`emulator_net`): the ABM's own seed spread,
which no amount of computing removes, and the emulator's ignorance, which more ABM runs would.
``calibrate`` uses both, and reports the second so that a "not implausible" verdict resting on a
thinly sampled corner of the corpus can be recognised as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from emulator_net import Emulator  # noqa: E402

#: Rows per forward pass. Large enough to keep the matrix multiplies efficient, small enough that
#: a 10^6-row Sobol' design never materialises its trajectories all at once.
BATCH = 16384


# ---------------------------------------------------------------------------------------------
# Baseline and design frames
# ---------------------------------------------------------------------------------------------
def find_baseline(model_dir: Path, override: Path | None) -> dict:
    """The values held fixed when only some parameters are varied.

    Taken from the corpus's own ``params_baseline.json`` where it can be found -- the model
    directory normally sits inside the generation run directory -- so a sweep is taken around the
    same point the paper's baseline runs use. Falls back to the midpoint of each parameter's
    design range, which is a defensible default and is reported as such.
    """
    candidates = []
    if override is not None:
        candidates.append(Path(override))
    candidates += [
        model_dir.parent / "input_data" / "params_baseline.json",
        model_dir / "input_data" / "params_baseline.json",
    ]
    for path in candidates:
        if path and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def baseline_row(emulator: Emulator, baseline: dict) -> dict:
    """One design row: every parameter and switch at its baseline value."""
    row = {}
    for index, name in enumerate(emulator.encoder.cont_names):
        lo, hi = emulator.encoder.bounds[index]
        value = baseline.get(name)
        row[name] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
            else float((lo + hi) / 2.0)
        row[name] = float(np.clip(row[name], lo, hi))
    for name in emulator.encoder.switch_names:
        levels = emulator.encoder.switch_levels[name]
        value = baseline.get(name, levels[0])
        row[name] = value if str(value) in [str(v) for v in levels] else levels[0]
    return row


def apply_arm(row: dict, arm: list[str] | None, emulator: Emulator) -> dict:
    """Apply ``--arm name=value`` overrides to a design row."""
    for item in arm or []:
        if "=" not in item:
            raise SystemExit(f"--arm expects name=value, got {item!r}")
        name, value = item.split("=", 1)
        if name in emulator.encoder.switch_names:
            levels = emulator.encoder.switch_levels[name]
            match = [lv for lv in levels if str(lv).lower() == value.strip().lower()]
            if not match:
                raise SystemExit(f"{name!r} has levels {levels}, not {value!r}")
            row[name] = match[0]
        elif name in emulator.encoder.cont_names:
            row[name] = float(value)
        else:
            raise SystemExit(f"{name!r} is not an input of this emulator")
    return row


def batched_scalars(emulator: Emulator, design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    means, sds = [], []
    for start in range(0, len(design), BATCH):
        chunk = design.iloc[start : start + BATCH].reset_index(drop=True)
        mean, sd = emulator.predict_scalars(chunk)
        means.append(mean)
        sds.append(sd)
    return np.vstack(means), np.vstack(sds)


def batched_series(emulator: Emulator, design: pd.DataFrame, series: str) -> np.ndarray:
    blocks = []
    for start in range(0, len(design), BATCH):
        chunk = design.iloc[start : start + BATCH].reset_index(drop=True)
        blocks.append(emulator.predict_trajectory_mean(chunk)[series].astype(np.float32))
    return np.vstack(blocks)


def outdir_for(model_dir: Path, name: str) -> Path:
    path = model_dir / "apply" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------------------------
# sobol
# ---------------------------------------------------------------------------------------------
def cmd_sobol(args: argparse.Namespace) -> int:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    emulator = Emulator.load(args.model_dir)
    baseline = baseline_row(emulator, find_baseline(args.model_dir, args.baseline))
    baseline = apply_arm(baseline, args.arm, emulator)

    names = args.parameters or emulator.encoder.cont_names
    unknown = [n for n in names if n not in emulator.encoder.cont_names]
    if unknown:
        raise SystemExit(f"not continuous inputs of this emulator: {', '.join(unknown)}")
    index_of = {n: i for i, n in enumerate(emulator.encoder.cont_names)}
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [emulator.encoder.bounds[index_of[n]].tolist() for n in names],
    }

    design_matrix = sobol_sample.sample(
        problem, args.n_base, calc_second_order=args.second_order
    )
    frame = pd.DataFrame([baseline] * len(design_matrix))
    for column, name in enumerate(names):
        frame[name] = design_matrix[:, column]

    print(f"Emulator   : {args.model_dir}")
    print(f"Design     : {len(design_matrix)} emulator evaluations over {len(names)} parameters")
    print(f"Held at    : " + ", ".join(f"{n}={baseline[n]}" for n in emulator.encoder.switch_names))

    outdir = outdir_for(args.model_dir, "sobol")
    mean, _ = batched_scalars(emulator, frame)

    rows = []
    for column, output in enumerate(emulator.scalar_names):
        values = mean[:, column]
        if not np.isfinite(values).all() or np.ptp(values) == 0:
            continue
        result = sobol_analyze.analyze(
            problem, values, calc_second_order=args.second_order, print_to_console=False
        )
        for position, name in enumerate(names):
            rows.append(
                {
                    "output": output,
                    "parameter": name,
                    "S1": float(result["S1"][position]),
                    "S1_conf": float(result["S1_conf"][position]),
                    "ST": float(result["ST"][position]),
                    "ST_conf": float(result["ST_conf"][position]),
                }
            )
        if args.second_order:
            for i, first in enumerate(names):
                for j, second in enumerate(names[i + 1 :], start=i + 1):
                    rows.append(
                        {
                            "output": output,
                            "parameter": f"{first} x {second}",
                            "S1": np.nan, "S1_conf": np.nan,
                            "ST": float(result["S2"][i, j]),
                            "ST_conf": float(result["S2_conf"][i, j]),
                        }
                    )
    indices = pd.DataFrame(rows)
    indices.to_csv(outdir / "indices_scalars.csv", index=False)
    print(f"\nWrote scalar indices for {indices['output'].nunique()} outputs to {outdir}/")

    headline = args.headline or (emulator.scalar_names[:1])
    for output in headline:
        block = indices[(indices["output"] == output) & (~indices["parameter"].str.contains(" x "))]
        if not len(block):
            continue
        print(f"\n  {output}: total-order indices, largest first")
        for row in block.sort_values("ST", ascending=False).head(10).itertuples(index=False):
            print(f"    {row.parameter:26s} ST {row.ST:6.3f}   S1 {row.S1:6.3f}")

    # --- time-resolved indices ----------------------------------------------------------------
    # The thing the direct sweep cannot afford. A parameter whose ST rises through the run has a
    # cumulative effect; one whose ST is large early and falls is a trigger. The paper's claim
    # that fiscal pressure is necessary but not sufficient, with the neighbour signal doing the
    # tipping, is a statement about exactly this shape.
    for series in args.series or []:
        if series not in emulator.basis.names:
            raise SystemExit(f"{series!r} is not an emulated series; it has {emulator.basis.names}")
        curves = batched_series(emulator, frame, series)
        steps = range(0, curves.shape[1], args.stride)
        rows = []
        for step in steps:
            values = curves[:, step].astype(float)
            if np.ptp(values) == 0:
                continue
            result = sobol_analyze.analyze(problem, values, calc_second_order=False, print_to_console=False)
            for position, name in enumerate(names):
                rows.append(
                    {
                        "series": series, "t": int(step), "parameter": name,
                        "S1": float(result["S1"][position]),
                        "ST": float(result["ST"][position]),
                    }
                )
        pd.DataFrame(rows).to_csv(outdir / f"indices_{series}_by_period.csv", index=False)
        print(f"  time-resolved indices for {series} -> indices_{series}_by_period.csv")

    (outdir / "meta.json").write_text(
        json.dumps(
            {"n_base": args.n_base, "n_evaluations": len(design_matrix),
             "second_order": args.second_order, "parameters": names, "held": baseline},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    return 0


# ---------------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------------
def cmd_sweep(args: argparse.Namespace) -> int:
    emulator = Emulator.load(args.model_dir)
    baseline = baseline_row(emulator, find_baseline(args.model_dir, args.baseline))
    baseline = apply_arm(baseline, args.arm, emulator)

    over = args.over
    if not 1 <= len(over) <= 2:
        raise SystemExit("--over takes one or two parameter names")
    index_of = {n: i for i, n in enumerate(emulator.encoder.cont_names)}
    grids = []
    for name in over:
        if name not in index_of:
            raise SystemExit(f"{name!r} is not a continuous input of this emulator")
        lo, hi = emulator.encoder.bounds[index_of[name]]
        grids.append(np.linspace(lo, hi, args.resolution))

    mesh = np.meshgrid(*grids, indexing="ij")
    frame = pd.DataFrame([baseline] * mesh[0].size)
    for name, values in zip(over, mesh):
        frame[name] = values.ravel()

    prediction = emulator.predict(frame, draws=args.draws)
    table = frame[over].copy()
    for column in emulator.scalar_names:
        table[column] = prediction.scalars[column].to_numpy()
        table[f"{column}_sd"] = prediction.scalars_sd[column].to_numpy()
        table[f"{column}_sd_epistemic"] = prediction.scalars_sd_epistemic[column].to_numpy()

    outdir = outdir_for(args.model_dir, "sweep")
    stem = "_x_".join(over)
    table.to_csv(outdir / f"sweep_{stem}.csv", index=False)

    if args.series:
        for series in args.series:
            if series not in prediction.trajectories:
                raise SystemExit(f"{series!r} is not an emulated series")
            curves = prediction.trajectories[series]
            long = pd.DataFrame(curves, columns=[f"t{t}" for t in range(curves.shape[1])])
            for name in over:
                long.insert(0, name, frame[name].to_numpy())
            long.to_csv(outdir / f"sweep_{stem}_{series}.csv", index=False)

    print(f"Swept {len(frame)} points over {', '.join(over)} -> {outdir}/")
    print(f"  everything else held at baseline: "
          + ", ".join(f"{n}={baseline[n]}" for n in emulator.encoder.switch_names))
    return 0


# ---------------------------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------------------------
def cmd_predict(args: argparse.Namespace) -> int:
    emulator = Emulator.load(args.model_dir)
    baseline = baseline_row(emulator, find_baseline(args.model_dir, args.baseline))

    given = pd.read_csv(args.points)
    frame = pd.DataFrame([baseline] * len(given))
    for column in given.columns:
        if column in frame.columns:
            frame[column] = given[column].to_numpy()
        else:
            print(f"  ! column {column!r} is not an emulator input, ignored", file=sys.stderr)

    prediction = emulator.predict(frame, draws=args.draws, band_level=args.band)
    outdir = outdir_for(args.model_dir, "predict")

    table = frame.copy()
    for column in emulator.scalar_names:
        table[column] = prediction.scalars[column].to_numpy()
        table[f"{column}_sd"] = prediction.scalars_sd[column].to_numpy()
    table.to_csv(outdir / "scalars.csv", index=False)

    for series, curves in prediction.trajectories.items():
        band = prediction.trajectory_bands[series]
        long = pd.DataFrame(
            {
                "row": np.repeat(np.arange(curves.shape[0]), curves.shape[1]),
                "t": np.tile(np.arange(curves.shape[1]), curves.shape[0]),
                "mean": curves.ravel(),
                "lo": band[:, :, 0].ravel(),
                "hi": band[:, :, 1].ravel(),
            }
        )
        long.to_csv(outdir / f"trajectory_{series}.csv", index=False)

    print(f"Predicted {len(frame)} points -> {outdir}/ ({args.band:.0%} bands)")
    return 0


# ---------------------------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------------------------
def cmd_calibrate(args: argparse.Namespace) -> int:
    """History matching: which parameter settings could have produced the observed pattern.

    Implausibility for target ``z`` with tolerance ``sd_obs``:

        I(theta) = |z - E[f(theta)]| / sqrt( Var_emulator + sd_obs^2 + sd_discrepancy^2 )

    A point is ruled *out* when any target's implausibility exceeds the threshold (3 by the usual
    convention -- Pukelsheim's three-sigma rule bounds the tail of any unimodal distribution at
    5%). What survives is the *not-implausible* set, which is the honest object: history matching
    rules regions out, it does not rule any region in.

    ``sd_discrepancy`` is the one the modeller has to argue for, and it should not be small. It
    is the admission that the ABM is a caricature of sixteenth-century England, so that even the
    best parameter setting will not reproduce the series exactly; setting it to zero asserts the
    model is true and will rule out everything.
    """
    from scipy.stats import qmc

    emulator = Emulator.load(args.model_dir)
    baseline = baseline_row(emulator, find_baseline(args.model_dir, args.baseline))
    targets = yaml.safe_load(Path(args.targets).read_text(encoding="utf-8")) or {}

    scalar_targets = targets.get("scalars") or {}
    anchors = targets.get("trajectory_anchors") or {}
    threshold = float(targets.get("threshold", args.threshold))
    discrepancy = float(targets.get("discrepancy", 0.0))

    unknown = [k for k in scalar_targets if k not in emulator.scalar_names]
    if unknown:
        raise SystemExit(f"targets name outputs the emulator does not predict: {', '.join(unknown)}")

    engine = qmc.Sobol(d=len(emulator.encoder.cont_names), scramble=True, seed=args.seed)
    unit = engine.random(args.n_samples)
    lo, hi = emulator.encoder.bounds[:, 0], emulator.encoder.bounds[:, 1]
    sampled = lo + unit * (hi - lo)

    frame = pd.DataFrame([baseline] * args.n_samples)
    for column, name in enumerate(emulator.encoder.cont_names):
        frame[name] = sampled[:, column]
    frame = pd.DataFrame(apply_arm(row, args.arm, emulator) for row in frame.to_dict("records"))

    mean, sd = batched_scalars(emulator, frame)
    column_of = {name: i for i, name in enumerate(emulator.scalar_names)}

    implausibility = np.zeros(len(frame))
    per_target = {}
    for name, spec in scalar_targets.items():
        value = float(spec["value"] if isinstance(spec, dict) else spec)
        tolerance = float(spec.get("sd", args.default_tolerance)) if isinstance(spec, dict) else args.default_tolerance
        column = column_of[name]
        spread = np.sqrt(sd[:, column] ** 2 + tolerance**2 + discrepancy**2)
        score = np.abs(value - mean[:, column]) / np.where(spread > 0, spread, np.nan)
        per_target[name] = score
        implausibility = np.maximum(implausibility, score)

    if anchors:
        curves = {s: batched_series(emulator, frame, s) for s in anchors}
        for name, spec in anchors.items():
            step = int(spec["t"])
            if not 0 <= step < curves[name].shape[1]:
                raise SystemExit(
                    f"trajectory anchor {name}@t{step} is outside the emulated run length "
                    f"(0..{curves[name].shape[1] - 1}). The emulator only knows the periods its "
                    f"corpus was run for."
                )
            value, tolerance = float(spec["value"]), float(spec.get("sd", args.default_tolerance))
            # No per-period sd from the cheap path, so the tolerance carries the whole spread
            # here. Widen it accordingly, or use `predict` on the survivors for a proper band.
            score = np.abs(value - curves[name][:, step]) / max(
                np.sqrt(tolerance**2 + discrepancy**2), 1e-9
            )
            per_target[f"{name}@t{step}"] = score
            implausibility = np.maximum(implausibility, score)

    if not per_target:
        raise SystemExit("targets file lists neither `scalars:` nor `trajectory_anchors:`")

    keep = implausibility <= threshold
    outdir = outdir_for(args.model_dir, "calibrate")
    table = frame[emulator.encoder.cont_names].copy()
    table["implausibility"] = implausibility
    table["not_implausible"] = keep
    for name, score in per_target.items():
        table[f"I_{name}"] = score
    table.to_csv(outdir / "implausibility.csv", index=False)

    print(f"Sampled    : {args.n_samples} points")
    print(f"Threshold  : {threshold} on the maximum implausibility over {len(per_target)} targets")
    print(f"Surviving  : {int(keep.sum())} ({100 * keep.mean():.2f}% of the space)")
    if keep.sum() == 0:
        print(
            "\n  Nothing survives. Either the targets are outside anything the model can produce "
            "-- itself a result about the theory, and the one worth reporting -- or the model "
            "discrepancy term is too small for a model this stylised. Raise `discrepancy:` and "
            "re-run before concluding the first."
        )
        return 0

    marginals = []
    for index, name in enumerate(emulator.encoder.cont_names):
        values = sampled[keep, index]
        marginals.append(
            {
                "parameter": name,
                "prior_lo": float(lo[index]),
                "prior_hi": float(hi[index]),
                "post_min": float(values.min()),
                "post_q05": float(np.quantile(values, 0.05)),
                "post_median": float(np.median(values)),
                "post_q95": float(np.quantile(values, 0.95)),
                "post_max": float(values.max()),
                # How much of its prior range the parameter has given up. Near 0 means the
                # targets say nothing about it; near 1 means they pin it down.
                "range_reduction": float(
                    1.0 - (values.max() - values.min()) / max(hi[index] - lo[index], 1e-12)
                ),
            }
        )
    frame_marginals = pd.DataFrame(marginals).sort_values("range_reduction", ascending=False)
    frame_marginals.to_csv(outdir / "marginals.csv", index=False)

    print("\nParameters the targets actually constrain:")
    print(f"  {'parameter':26s} {'reduction':>10s} {'5%':>9s} {'median':>9s} {'95%':>9s}")
    for row in frame_marginals.head(12).itertuples(index=False):
        print(
            f"  {row.parameter:26s} {row.range_reduction:10.2f} {row.post_q05:9.3g} "
            f"{row.post_median:9.3g} {row.post_q95:9.3g}"
        )
    print(f"\nWrote {outdir}/")
    return 0


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-dir", type=Path, required=True, help="output of emulator_train.py")
        p.add_argument("--baseline", type=Path, default=None, help="params_baseline.json to hold at")
        p.add_argument(
            "--arm", nargs="*", default=None,
            help="switch settings, e.g. --arm channel_ideology=false ideology_rule=material",
        )

    p_sobol = sub.add_parser("sobol", help="variance decomposition on the surrogate")
    common(p_sobol)
    p_sobol.add_argument("--n-base", type=int, default=16384, help="SALib N; powers of two")
    p_sobol.add_argument("--parameters", nargs="*", default=None, help="default: all of them")
    p_sobol.add_argument("--second-order", action="store_true", help="affordable here; off in the ABM sweep")
    p_sobol.add_argument("--headline", nargs="*", default=None, help="outputs to print")
    p_sobol.add_argument("--series", nargs="*", default=None, help="time-resolved indices for these series")
    p_sobol.add_argument("--stride", type=int, default=5, help="periods between time-resolved points")
    p_sobol.set_defaults(func=cmd_sobol)

    p_sweep = sub.add_parser("sweep", help="response along one or two parameters")
    common(p_sweep)
    p_sweep.add_argument("--over", nargs="+", required=True)
    p_sweep.add_argument("--resolution", type=int, default=61)
    p_sweep.add_argument("--series", nargs="*", default=None)
    p_sweep.add_argument("--draws", type=int, default=128)
    p_sweep.set_defaults(func=cmd_sweep)

    p_predict = sub.add_parser("predict", help="predict arbitrary parameter rows from a CSV")
    common(p_predict)
    p_predict.add_argument("--points", type=Path, required=True)
    p_predict.add_argument("--draws", type=int, default=256)
    p_predict.add_argument("--band", type=float, default=0.9)
    p_predict.set_defaults(func=cmd_predict)

    p_cal = sub.add_parser("calibrate", help="history matching against targets")
    common(p_cal)
    p_cal.add_argument("--targets", type=Path, required=True)
    p_cal.add_argument("--n-samples", type=int, default=131072)
    p_cal.add_argument("--threshold", type=float, default=3.0)
    p_cal.add_argument("--default-tolerance", type=float, default=0.05)
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
