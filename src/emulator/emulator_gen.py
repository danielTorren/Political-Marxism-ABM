"""Generate the training corpus for the neural-network emulator.

    uv run python src/emulator/emulator_gen.py --dry-run     # design size and cost, no runs
    uv run python src/emulator/emulator_gen.py               # the shipped 2048 x 3 design
    uv run python src/emulator/emulator_gen.py --n-points 256 --replicates 2 --steps 60

Two files govern the corpus, so nothing that ought to be recorded is passed as a flag:

    constants/constants.yaml   the baseline the design is taken around, and run control
    constants/design.yaml      which parameters and switches vary, and what is recorded

The design is a *scrambled Sobol' sequence*, not the Saltelli design used by src/sensitivity.
Both are Sobol' sequences; they are put to different uses. The Saltelli design spends most of its
rows on the A/B cross-samples its estimator requires, which are wasted on a surrogate; a surrogate
wants plain space-filling coverage, and recovers the sensitivity indices afterwards by evaluating
the trained network millions of times (``emulator_apply.py sobol``).

Each invocation writes a fresh timestamped run directory:

    Results/emulator/emulator_2026-09-08_143012/
        input_data/    both yaml files as given, the resolved baseline, the design record
        output_data/   X.csv          one row per design point: the parameters and switches
                       runs.csv       one row per (point, seed): the scalar outputs
                       trajectories.npz  (n_runs, n_series, n_steps) float32, plus its index
        figures/       written later by emulator_plot.py

``checkpoint_every`` rewrites all three tables periodically, so a sweep killed halfway still
leaves a loadable corpus -- point ``emulator_train.py --run-dir`` at it and it will train on
whatever finished.

What is deliberately *not* recorded: the per-parcel spatial field. ``awareness_radius`` and
``zeta`` are in the design, so the lattice is rebuilt for every point and parcel indices are not
comparable across the corpus. A spatial-field emulator would need those two parameters held
fixed, and is a separate corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

SRC = Path(__file__).resolve().parents[1]
for extra in (SRC, SRC / "multi_seed", SRC / "sensitivity"):
    sys.path.insert(0, str(extra))

from pmabm.config import Params  # noqa: E402
from pmabm.geography import build as build_geography, load_artifact  # noqa: E402
from pmabm.metrics import history_frame, summary  # noqa: E402
from pmabm.model import Model  # noqa: E402

import multi_seed_gen  # noqa: E402  (build_params, make_run_dir, _git_commit)
from sensitivity_gen import (  # noqa: E402
    GEOGRAPHY_FIELDS,
    time_slice_outputs,
    timing_outputs,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONSTANTS = HERE / "constants" / "constants.yaml"
DEFAULT_DESIGN = HERE / "constants" / "design.yaml"

#: Throughput per worker, used only by ``--dry-run`` to print an estimate. Measured rather than
#: derived: ``src/sensitivity/constants/constants.yaml`` records ~13 runs a minute on 13 workers
#: at L=155 and n_steps=200, which is ~1 run per worker per minute against a single run's ~37 s of
#: CPU. The gap is the point -- memory bandwidth binds before cores do, so the speedup is well
#: short of linear, and dividing a per-run time by the worker count understates the wall clock by
#: nearly half. Re-measure it here if the machine changes.
RUNS_PER_MINUTE_PER_WORKER = 1.0
REFERENCE_STEPS = 200


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# ---------------------------------------------------------------------------------------------
# The input space
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Space:
    """The design space: continuous parameters plus categorical switches.

    ``cont_names``/``bounds``/``integers`` are the continuous half, mapped from the unit cube by
    an affine transform. ``switch_names``/``switch_levels`` are the categorical half, mapped by
    slicing [0, 1) into equal parts -- so that a scrambled Sobol' sequence balances the switch
    combinations across the design as well as filling the continuous space.
    """

    cont_names: list[str]
    bounds: np.ndarray                       # (n_cont, 2)
    integers: frozenset[str]
    switch_names: list[str] = field(default_factory=list)
    switch_levels: dict[str, list] = field(default_factory=dict)

    @property
    def n_dims(self) -> int:
        return len(self.cont_names) + len(self.switch_names)

    @property
    def n_arms(self) -> int:
        n = 1
        for name in self.switch_names:
            n *= len(self.switch_levels[name])
        return n


def _field_kind(name: str) -> str:
    """``"int"``, ``"float"``, ``"bool"``, ``"str"`` ... for a field of :class:`Params`."""
    fld = Params.__dataclass_fields__[name]
    return fld.type if isinstance(fld.type, str) else getattr(fld.type, "__name__", "")


def build_space(design: dict) -> Space:
    """Turn the ``parameters:`` and ``switches:`` blocks into a :class:`Space`.

    Unlike :func:`sensitivity_gen.build_problem` this *accepts* the boolean and string switches:
    a variance decomposition over a categorical is meaningless, but a surrogate simply learns a
    different response surface in each arm. Everything else is checked the same way -- an unknown
    name is an error rather than a silent no-op, and a level a field would reject is caught here
    rather than in every one of the runs that would have used it.
    """
    entries = design.get("parameters") or {}
    if not entries:
        raise SystemExit("design.yaml lists no parameters under `parameters:`")

    valid = set(Params.__dataclass_fields__)
    unknown = sorted((set(entries) | set(design.get("switches") or {})) - valid)
    if unknown:
        raise SystemExit(
            f"Unknown name(s) in design.yaml: {', '.join(unknown)}\n"
            f"Valid names are the fields of Params in src/pmabm/config.py"
        )

    names, limits, integers = [], [], []
    for name, span in entries.items():
        kind = _field_kind(name)
        if kind in ("bool", "str"):
            raise SystemExit(
                f"{name!r} is a {kind} switch; list it under `switches:` with its levels, "
                f"not under `parameters:` with a range."
            )
        if not (isinstance(span, (list, tuple)) and len(span) == 2):
            raise SystemExit(f"bounds for {name!r} must be [lo, hi], got {span!r}")
        lo, hi = float(span[0]), float(span[1])
        if not hi > lo:
            raise SystemExit(f"bounds for {name!r} must have hi > lo, got [{lo}, {hi}]")
        names.append(name)
        limits.append([lo, hi])
        if kind == "int":
            integers.append(name)

    switch_names, switch_levels = [], {}
    for name, levels in (design.get("switches") or {}).items():
        if not isinstance(levels, (list, tuple)) or len(levels) < 2:
            raise SystemExit(f"switch {name!r} needs at least two levels, got {levels!r}")
        # Reject a bad level now, once, rather than in every run that would have drawn it.
        for level in levels:
            try:
                Params(**{name: level})
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"switch {name!r} rejects level {level!r}: {exc}") from exc
        switch_names.append(name)
        switch_levels[name] = list(levels)

    return Space(
        cont_names=names,
        bounds=np.asarray(limits, dtype=float),
        integers=frozenset(integers),
        switch_names=switch_names,
        switch_levels=switch_levels,
    )


def sample_design(space: Space, n_points: int, seed: int) -> pd.DataFrame:
    """A scrambled Sobol' design over the whole space, as a table of actual parameter values.

    Returns one row per design point with a column per continuous parameter and per switch,
    holding the value that will be handed to ``Params`` -- not the unit-cube coordinate. The
    encoding into network inputs happens later, in :mod:`emulator_data`, so that the corpus on
    disk stays readable as parameters.
    """
    if n_points < 2 or n_points & (n_points - 1):
        print(
            f"  ! n_points={n_points} is not a power of two. A scrambled Sobol' sequence is "
            f"balanced in blocks of 2^k; truncating mid-block costs the low-discrepancy "
            f"property that makes the design space-filling.",
            file=sys.stderr,
        )

    engine = qmc.Sobol(d=space.n_dims, scramble=True, seed=seed)
    unit = engine.random(n_points)

    frame = pd.DataFrame(index=range(n_points))
    lo, hi = space.bounds[:, 0], space.bounds[:, 1]
    scaled = lo + unit[:, : len(space.cont_names)] * (hi - lo)
    for column, name in enumerate(space.cont_names):
        values = scaled[:, column]
        frame[name] = np.rint(values).astype(int) if name in space.integers else values

    for offset, name in enumerate(space.switch_names):
        levels = space.switch_levels[name]
        column = unit[:, len(space.cont_names) + offset]
        index = np.clip((column * len(levels)).astype(int), 0, len(levels) - 1)
        frame[name] = [levels[i] for i in index]

    frame.insert(0, "point", range(n_points))
    return frame


def overrides_for(row: dict, space: Space) -> dict:
    """One design row as a ``Params`` override dict."""
    out: dict = {}
    for name in space.cont_names:
        out[name] = int(row[name]) if name in space.integers else float(row[name])
    for name in space.switch_names:
        value = row[name]
        out[name] = bool(value) if isinstance(value, (bool, np.bool_)) else value
    return out


# ---------------------------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------------------------
def trajectory_matrix(history: pd.DataFrame, series: list[str], n_steps: int) -> np.ndarray:
    """``(n_series, n_steps)`` of the requested per-period columns, NaN-padded.

    Padding rather than truncation or refusal, because a run that ends early -- the model can
    reach a state with no occupied parcels -- still carries usable information for the periods it
    did run, and the training code masks NaN. A column the model never produced is all-NaN and is
    reported by the caller.
    """
    out = np.full((len(series), n_steps), np.nan, dtype=np.float32)
    if history.empty:
        return out
    # Positioned by the period each row belongs to, not by its position in the frame: the model
    # records aggregates every ``record_every`` periods, so row i is period i only when that is
    # 1. Everything in between stays NaN and is masked by the training code, exactly as the
    # tail of a run that ended early is.
    if "t" in history.columns:
        step = history["t"].to_numpy(dtype=np.int64)
    else:
        step = np.arange(len(history), dtype=np.int64)
    keep = (step >= 0) & (step < n_steps)
    step = step[keep]
    for index, name in enumerate(series):
        if name in history.columns:
            out[index, step] = history[name].to_numpy(dtype=np.float32)[keep]
    return out


def _derived_columns(history: pd.DataFrame, n_parcels: int) -> pd.DataFrame:
    """Per-period quantities the model records only as counts.

    ``conversion_share`` is the primary trajectory and does not exist in ``model.history``: the
    model stores ``converted_parcels``, whose scale depends on the lattice. Dividing here keeps
    the corpus comparable across points even though the lattice is rebuilt for each of them.
    """
    if "converted_parcels" in history.columns:
        history = history.assign(
            conversion_share=history["converted_parcels"] / max(n_parcels, 1)
        )
    return history


# ---------------------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------------------
# Set once per process by the pool initializer, so the baseline, the England outline and the
# shared lattice are pickled once per worker rather than once per run. Through the initializer
# rather than module state because Windows *spawns* workers, re-importing this module, which
# would otherwise leave every worker running the dataclass defaults.
_BASE: Params = Params()
_ARTIFACT: dict | None = None
_GEOGRAPHY = None
_GEO_SEED = 0
_SERIES: list[str] = []


def _init_worker(base: Params, artifact: dict, geography, geo_seed: int, series: list[str]) -> None:
    global _BASE, _ARTIFACT, _GEOGRAPHY, _GEO_SEED, _SERIES
    _BASE, _ARTIFACT, _GEOGRAPHY, _GEO_SEED, _SERIES = (
        base, artifact, geography, geo_seed, series,
    )


def _run_point(payload: tuple) -> tuple[dict, np.ndarray]:
    """Run one (point, seed) pair; return its scalar row and its trajectory block.

    A design point can be invalid rather than merely extreme -- ``Params.__post_init__`` rejects
    combinations the model is not defined for -- and a run can fail on its own terms. Either way
    the row comes back flagged with the reason and an all-NaN trajectory, instead of taking a
    multi-hour sweep down with it.
    """
    index, seed, overrides, needs_geography, n_steps = payload
    row = {"point": index, "seed": seed}
    try:
        params = _BASE.with_(seed=seed, **overrides)
        geography = (
            build_geography(params, np.random.default_rng(_GEO_SEED), artifact=_ARTIFACT)
            if needs_geography
            else _GEOGRAPHY
        )
        model = Model(params, geography=geography, artifact=_ARTIFACT).run()
        history = _derived_columns(history_frame(model), model.geo.n_parcels)
        row.update(summary(model))
        row.update(timing_outputs(model, history))
        row.update(time_slice_outputs(history))
        row["failed"] = 0
        row["error"] = ""
        return row, trajectory_matrix(history, _SERIES, n_steps)
    except Exception as exc:  # noqa: BLE001 - one bad corner must not end an 8-hour sweep
        row["failed"] = 1
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, np.full((len(_SERIES), n_steps), np.nan, dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------
def run_corpus(
    base: Params,
    design: pd.DataFrame,
    space: Space,
    seeds: list[int],
    series: list[str],
    artifact: dict,
    geography,
    needs_geography: bool,
    workers: int,
    geo_seed: int,
    outdir: Path,
    checkpoint_every: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run every (point, seed) pair; return the scalar table and the trajectory array.

    Rows come back in completion order and are sorted at the end, but the trajectory array is
    built by *slot*: run ``k`` of the payload list always writes block ``k``, so the array and
    the sorted table are aligned by construction rather than by a join that a failed run could
    misalign.
    """
    payloads = [
        (int(row["point"]), seed, overrides_for(row, space), needs_geography, base.n_steps)
        for row in design.to_dict("records")
        for seed in seeds
    ]
    total = len(payloads)
    trajectories = np.full((total, len(series), base.n_steps), np.nan, dtype=np.float32)
    rows: list[dict | None] = [None] * total
    slot_of = {(p[0], p[1]): k for k, p in enumerate(payloads)}

    started = time.perf_counter()
    every = max(1, total // 50)
    checkpoint_at = max(1, int(total * checkpoint_every)) if checkpoint_every > 0 else total + 1
    done = 0
    last_checkpoint = 0

    def record(result: tuple[dict, np.ndarray]) -> None:
        nonlocal done, last_checkpoint
        row, block = result
        slot = slot_of[(row["point"], row["seed"])]
        rows[slot] = row
        trajectories[slot] = block
        done += 1
        if done == 1 or done % every == 0 or done == total:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            failed = sum(r["failed"] for r in rows if r is not None)
            print(
                f"    {done:6d}/{total}  {100 * done / total:5.1f}%  "
                f"{elapsed / 60:7.1f} min elapsed, {(total - done) / rate / 60:7.1f} min left"
                f"{f', {failed} failed' if failed else ''}",
                flush=True,
            )
        if done - last_checkpoint >= checkpoint_at or done == total:
            last_checkpoint = done
            write_corpus(outdir, design, rows, trajectories, series, base.n_steps, partial=done < total)

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(base, artifact, geography, geo_seed, series),
        ) as pool:
            futures = [pool.submit(_run_point, p) for p in payloads]
            for future in as_completed(futures):
                record(future.result())
    else:
        _init_worker(base, artifact, geography, geo_seed, series)
        for payload in payloads:
            record(_run_point(payload))

    table = pd.DataFrame([r for r in rows if r is not None])
    return table, trajectories


def write_corpus(
    outdir: Path,
    design: pd.DataFrame,
    rows: list[dict | None],
    trajectories: np.ndarray,
    series: list[str],
    n_steps: int,
    partial: bool = False,
) -> None:
    """Write X, the scalar table and the trajectory array, dropping not-yet-run slots.

    Rewritten whole on every checkpoint rather than appended: a failed run carries no output
    columns, so appending rows of differing width would misalign the file. A full rewrite of a
    few thousand rows and a ~50 MB array costs a second, against a single model run's 37.
    """
    finished = [k for k, row in enumerate(rows) if row is not None]
    table = pd.DataFrame([rows[k] for k in finished])
    design.to_csv(outdir / "X.csv", index=False)
    table.to_csv(outdir / "runs.csv", index=False)
    np.savez_compressed(
        outdir / "trajectories.npz",
        values=trajectories[finished],
        point=table["point"].to_numpy() if len(table) else np.zeros(0, dtype=int),
        seed=table["seed"].to_numpy() if len(table) else np.zeros(0, dtype=int),
        series=np.asarray(series, dtype=object),
        n_steps=np.asarray(n_steps),
        partial=np.asarray(partial),
    )


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--n-points", type=int, default=None, help="override sample.n_points")
    parser.add_argument("--replicates", type=int, default=None, help="override sample.replicates")
    parser.add_argument("--steps", type=int, default=None, help="override model.n_steps")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None, help="0 or 1 runs sequentially")
    parser.add_argument("--tag", default=None, help="suffix on the run directory")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the design size and cost estimate, and exit without running the model",
    )
    args = parser.parse_args(argv)

    constants = load_yaml(args.constants)
    design_cfg = load_yaml(args.design)
    if args.steps is not None:
        constants.setdefault("model", {})["n_steps"] = args.steps
    base = multi_seed_gen.build_params(constants)

    run_cfg = constants.get("run") or {}
    sample_cfg = design_cfg.get("sample") or {}
    outputs_cfg = design_cfg.get("outputs") or {}

    space = build_space(design_cfg)
    n_points = int(args.n_points or sample_cfg.get("n_points", 1024))
    replicates = int(args.replicates or sample_cfg.get("replicates", 3))
    seed_start = int(sample_cfg.get("seed_start", 0))
    seeds = list(range(seed_start, seed_start + max(1, replicates)))
    design_seed = int(sample_cfg.get("design_seed", 12345))

    scalars = list(outputs_cfg.get("scalars") or [])
    trajectories_cfg = dict(outputs_cfg.get("trajectories") or {})
    series = list(trajectories_cfg)
    if not scalars and not series:
        raise SystemExit("design.yaml requests no outputs")

    design = sample_design(space, n_points, design_seed)
    n_runs = n_points * len(seeds)

    varied_geography = sorted(GEOGRAPHY_FIELDS.intersection(space.cont_names + space.switch_names))
    fixed_geography = bool(run_cfg.get("fixed_geography", True)) and not varied_geography
    geo_seed = int(run_cfg.get("geography_seed", 0))

    workers = args.workers if args.workers is not None else run_cfg.get("workers")
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(int(workers), n_runs))

    throughput = RUNS_PER_MINUTE_PER_WORKER * max(workers, 1) * (REFERENCE_STEPS / max(base.n_steps, 1))
    cost = n_runs / throughput / 60.0

    print(f"Constants  : {args.constants}")
    print(f"Design     : {args.design}")
    print(
        f"             scrambled Sobol', {len(space.cont_names)} continuous parameters"
        + (f" + {len(space.switch_names)} switches ({space.n_arms} arms)" if space.switch_names else "")
    )
    print(f"             {n_points} points x {len(seeds)} seeds = {n_runs} model runs")
    print(f"Parameters : {', '.join(space.cont_names)}")
    if space.switch_names:
        print(f"Switches   : {', '.join(space.switch_names)}")
        if space.n_arms > n_points / 8:
            print(
                f"  ! {space.n_arms} switch combinations against {n_points} points is about "
                f"{n_points / space.n_arms:.0f} points per arm. The arms share a trunk, so this "
                f"is not fatal, but drop a switch or raise n_points if the fit is poor.",
                file=sys.stderr,
            )
    print(f"Outputs    : {len(scalars)} scalars, {len(series)} trajectories x {base.n_steps} steps")
    print(
        f"Geography  : {'one shared layout' if fixed_geography else 'rebuilt per point'}"
        + (f" (because {', '.join(varied_geography)} is in the design)" if varied_geography else "")
    )
    print(f"Workers    : {workers}")
    print(
        f"Estimated  : {cost:.1f} h wall clock at {throughput:.1f} runs a minute "
        f"(measured throughput, not cores x per-run time)"
    )

    if args.dry_run:
        print("\nDry run: nothing executed.")
        return 0

    root = Path(args.outdir or run_cfg.get("outdir", "Results/emulator"))
    timestamped = not args.no_timestamp and bool(run_cfg.get("timestamped", True))
    paths = multi_seed_gen.make_run_dir(root, timestamped=timestamped, tag=args.tag)
    outdir, indir = paths["output"], paths["input"]
    print(f"Run dir    : {paths['run']}\n")

    artifact = load_artifact()
    geography = (
        build_geography(base, np.random.default_rng(geo_seed), artifact=artifact)
        if fixed_geography
        else None
    )

    # --- inputs, written before the sweep so a killed run is still documented ---------------
    for path in (args.constants, args.design):
        if path.is_file():
            shutil.copy2(path, indir / path.name)
    (indir / "params_baseline.json").write_text(
        json.dumps(asdict(base), indent=2, default=str), encoding="utf-8"
    )
    (indir / "space.json").write_text(
        json.dumps(
            {
                "cont_names": space.cont_names,
                "bounds": space.bounds.tolist(),
                "integers": sorted(space.integers),
                "switch_names": space.switch_names,
                "switch_levels": space.switch_levels,
                "scalars": scalars,
                "trajectories": trajectories_cfg,
                "n_points": n_points,
                "seeds": seeds,
                "design_seed": design_seed,
                "n_steps": base.n_steps,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (indir / "run_meta.json").write_text(
        json.dumps(
            {
                "started": datetime.now().isoformat(timespec="seconds"),
                "command": " ".join(sys.argv),
                "git_commit": multi_seed_gen._git_commit(),
                "n_points": n_points,
                "n_runs": n_runs,
                "n_steps": base.n_steps,
                "workers": workers,
                "fixed_geography": fixed_geography,
                "geography_seed": geo_seed,
                "varied_geography_fields": varied_geography,
                "run_dir": str(paths["run"]),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # --- the sweep ---------------------------------------------------------------------------
    print(f"Running {n_runs} model runs at n_steps={base.n_steps}:")
    started = time.perf_counter()
    table, trajectories = run_corpus(
        base, design, space, seeds, series, artifact, geography,
        needs_geography=not fixed_geography, workers=workers, geo_seed=geo_seed,
        outdir=outdir, checkpoint_every=float(run_cfg.get("checkpoint_every", 0.05)),
    )
    minutes = (time.perf_counter() - started) / 60

    failed = int(table["failed"].sum())
    print(f"\nRan {len(table)} runs in {minutes:.1f} min ({failed} failed)")
    if failed:
        for reason, count in table.loc[table["failed"] == 1, "error"].value_counts().head(5).items():
            print(f"    {count:5d}  {reason}")
    missing = [s for i, s in enumerate(series) if np.isnan(trajectories[:, i, :]).all()]
    if missing:
        print(
            f"  ! trajectory series never produced by any run, all-NaN in the corpus: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
    print(f"      corpus to {outdir}/\n      inputs to {indir}/")
    print(
        f"\nNext:\n"
        f"    uv run python src/emulator/emulator_train.py --run-dir {paths['run']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
