"""Run a Sobol' variance-based sensitivity analysis of the model.

    uv run python src/sensitivity/sensitivity_gen.py
    uv run python src/sensitivity/sensitivity_gen.py --dry-run          # design size only
    uv run python src/sensitivity/sensitivity_gen.py --n-base 256
    uv run python src/sensitivity/sensitivity_gen.py --n-base 8 --replicates 1 --steps 60

Two files govern the sweep, so nothing is passed as a flag that ought to be recorded:

    constants/constants.yaml   the baseline the decomposition is taken around
    constants/bounds.yaml      which parameters vary, over what range, and how many runs

The design is SALib's Sobol' sequence, ``n_base * (D + 2)`` sample points for ``D`` parameters,
each run for ``replicates`` seeds whose outputs are averaged before the decomposition. First-order
(S1) and total-order (ST) indices only -- second order is off, and ``ST - S1`` already reports how
much of a parameter's influence runs through interactions.

Averaging over seeds matters more here than in the other runners. Sobol' attributes variance in the
output to variance in the inputs, and a stochastic model contributes variance of its own that
belongs to no parameter; with one seed per point that noise is silently divided up among the
indices. The analysis script reports the residual share so it can be judged rather than assumed.

Each invocation writes a fresh timestamped run directory:

    Results/sensitivity/sensitivity_2026-08-04_143012/
        input_data/    both yaml files as given, the resolved baseline, the SALib problem
        output_data/   samples.csv (X), runs_by_sample.csv (every run), Y.csv (seed-averaged)
                       plus the Sobol' indices and diagnostics written by the analysis script
        figures/       the sensitivity figures

``runs_by_sample.csv`` is written incrementally while the sweep runs, so a sweep killed halfway
still leaves usable results: point ``sensitivity_analysis.py`` at the run directory afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from SALib.sample import sobol as sobol_sample

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "multi_seed"))  # multi_seed_gen imports multi_seed_plot by name

from pmabm.config import Params  # noqa: E402
from pmabm.geography import build as build_geography, load_artifact  # noqa: E402
from pmabm.metrics import history_frame, summary  # noqa: E402
from pmabm.model import Model  # noqa: E402

import multi_seed_gen  # noqa: E402  (build_params, make_run_dir, _git_commit)
import sensitivity_analysis  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONSTANTS = HERE / "constants" / "constants.yaml"
DEFAULT_BOUNDS = HERE / "constants" / "bounds.yaml"

#: Parameters the geography is built from. If any of these is varied, one shared layout would
#: make the variation invisible, so the layout is rebuilt per sample instead.
GEOGRAPHY_FIELDS = frozenset(
    {"L", "lords_per_county", "zeta", "awareness_radius", "uniform_fertility"}
)


# ---------------------------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _field_kind(name: str) -> str:
    """``"int"``, ``"float"``, ``"bool"``, ``"str"`` ... for a field of :class:`Params`.

    ``config.py`` uses ``from __future__ import annotations``, so the recorded type is already
    the source string; fall back to the object's name for the case where it is not.
    """
    field = Params.__dataclass_fields__[name]
    return field.type if isinstance(field.type, str) else getattr(field.type, "__name__", "")


def build_problem(bounds: dict) -> tuple[dict, list[str]]:
    """Turn the ``parameters:`` block into a SALib problem, and list which entries are integers.

    Rejects anything that cannot carry a variance decomposition: unknown names, and the boolean
    and string switches. A Sobol' index over a categorical is not a small index, it is a
    meaningless one -- those belong in ``src/scenarios`` as arms.
    """
    entries = bounds.get("parameters") or {}
    if not entries:
        raise SystemExit("bounds.yaml lists no parameters to vary")

    valid = set(Params.__dataclass_fields__)
    unknown = sorted(set(entries) - valid)
    if unknown:
        raise SystemExit(
            f"Unknown parameter(s) in bounds.yaml: {', '.join(unknown)}\n"
            f"Valid names are the fields of Params in src/pmabm/config.py"
        )

    names, limits, integers = [], [], []
    for name, span in entries.items():
        kind = _field_kind(name)
        if kind in ("bool", "str"):
            raise SystemExit(
                f"{name!r} is a {kind} switch, which cannot be varied continuously.\n"
                f"Compare it as a scenario arm instead (src/scenarios/scenario_gen.py)."
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

    return {"num_vars": len(names), "names": names, "bounds": limits}, integers


def sample_design(problem: dict, n_base: int, calc_second_order: bool) -> np.ndarray:
    if n_base < 2 or n_base & (n_base - 1):
        print(
            f"  ! n_base={n_base} is not a power of two; the Sobol' sequence is balanced in "
            f"blocks of 2^k, so the indices will be biased. Use 64, 128, 256 ...",
            file=sys.stderr,
        )
    return sobol_sample.sample(problem, n_base, calc_second_order=calc_second_order)


def overrides_for(row: np.ndarray, problem: dict, integers: set[str]) -> dict:
    """One sample row as a ``Params`` override dict, rounding the integer-typed fields."""
    out: dict[str, float | int] = {}
    for name, value in zip(problem["names"], row):
        out[name] = int(round(value)) if name in integers else float(value)
    return out


# ---------------------------------------------------------------------------------------------
# Extra outputs: when, not only whether
# ---------------------------------------------------------------------------------------------
def timing_outputs(model: Model, history: pd.DataFrame) -> dict:
    """Conversion *timing*, which no field of ``summary`` reports.

    Both measures are right-censored at ``n_steps``: a run in which half the parcels never
    convert is recorded as reaching that point on the last period rather than as missing, so a
    slow run and a stalled run sit at the same end of the scale instead of dropping out of the
    decomposition. Read them alongside ``conversion_share``, which says whether the censoring
    bound was hit.
    """
    share = history["converted_parcels"].to_numpy() / max(model.geo.n_parcels, 1)
    steps = int(model.p.n_steps)

    def first_at(threshold: float) -> float:
        hit = np.nonzero(share >= threshold)[0]
        return float(history["t"].iloc[hit[0]]) if hit.size else float(steps)

    return {
        "t_first_conversion": first_at(1.0 / max(model.geo.n_parcels, 1)),
        "t_half_conversion": first_at(0.5),
    }


#: Points through the run at which the sliced outputs below are read, as a percentage of
#: ``n_steps``. Percentages rather than absolute periods so that the columns mean the same thing
#: if the run length ever changes, and so a design that sweeps ``n_steps`` stays comparable.
TIME_SLICES = (25, 50, 75, 100)


def time_slice_outputs(
    history: pd.DataFrame, slices: tuple[int, ...] = TIME_SLICES
) -> dict:
    """The headline quantities part-way through the run, not only at the end.

    A Sobol' decomposition of the final state answers "what determines where this ends up" and
    cannot answer "what determines how it gets there", which is a different question and in this
    model probably has a different answer: the conversion decision's own coefficients should
    dominate early, while the mechanisms that need a stock of converted land to work on --
    engrossment, the ideology channel, the labour market -- can only matter later. A parameter
    whose index rises through the run is one whose effect is cumulative; one whose index falls is
    a trigger.

    Read the resulting indices as being about a *transient*, and note that the same run supplies
    every slice, so the slices are not independent of one another.
    """
    if history.empty:
        return {}
    last = int(history["t"].max())
    columns = {
        "share_leasehold": "share_leasehold",
        "farm_gini": "farm_gini",
        "share_landless": "share_landless",
    }
    out: dict[str, float] = {}
    for pct in slices:
        # Nearest recorded period at or below the target, so a short run still fills every slice.
        target = last * pct / 100.0
        row = history[history["t"] <= target]
        row = row.iloc[-1] if len(row) else history.iloc[0]
        for name, column in columns.items():
            if column in history.columns:
                out[f"{name}_t{pct}"] = float(row[column])
    return out


# ---------------------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------------------
# Set once per process by the pool initializer, so the baseline, the England outline and the
# shared estate layout are pickled once per worker rather than once per run. They go through the
# initializer rather than through module state because Windows starts workers by *spawning* a
# fresh interpreter, which re-imports this module and would otherwise leave every worker running
# the dataclass defaults instead of the baseline in constants.yaml.
_BASE: Params = Params()
_ARTIFACT: dict | None = None
_GEOGRAPHY = None
_GEO_SEED = 0


def _init_worker(base: Params, artifact: dict, geography, geo_seed: int) -> None:
    global _BASE, _ARTIFACT, _GEOGRAPHY, _GEO_SEED
    _BASE, _ARTIFACT, _GEOGRAPHY, _GEO_SEED = base, artifact, geography, geo_seed


def _run_point(payload: tuple) -> dict:
    """Run one (sample, seed) pair and return its outputs as a flat row.

    A sample can be invalid rather than merely extreme -- ``Params.__post_init__`` rejects
    combinations the model is not defined for -- and a run can fail on its own terms. Either way
    the row comes back flagged with the reason instead of taking the sweep down, and the analysis
    script reports how many there were.
    """
    index, seed, overrides, needs_geography = payload
    row = {"sample": index, "seed": seed, **overrides}
    try:
        params = _BASE.with_(seed=seed, **overrides)
        geography = (
            build_geography(params, np.random.default_rng(_GEO_SEED), artifact=_ARTIFACT)
            if needs_geography
            else _GEOGRAPHY
        )
        model = Model(params, geography=geography, artifact=_ARTIFACT).run()
        history = history_frame(model)
        row.update(summary(model))
        row.update(timing_outputs(model, history))
        row.update(time_slice_outputs(history))
        row["failed"] = 0
        row["error"] = ""
    except Exception as exc:  # noqa: BLE001 - one bad corner must not end a multi-hour sweep
        row["failed"] = 1
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


# ---------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------
def run_sweep(
    base: Params,
    design: np.ndarray,
    problem: dict,
    integers: set[str],
    seeds: list[int],
    artifact: dict,
    geography,
    needs_geography: bool,
    workers: int,
    geo_seed: int,
    stream_path: Path | None = None,
) -> pd.DataFrame:
    """Run every (sample, seed) pair and return one row per run, in design order."""
    payloads = [
        (index, seed, overrides_for(row, problem, integers), needs_geography)
        for index, row in enumerate(design)
        for seed in seeds
    ]
    total = len(payloads)
    started = time.perf_counter()
    rows: list[dict] = []
    every = max(1, total // 50)

    def record(row: dict) -> None:
        rows.append(row)
        done = len(rows)
        if done != 1 and done % every and done != total:
            return
        elapsed = time.perf_counter() - started
        rate = done / max(elapsed, 1e-9)
        failed = sum(r["failed"] for r in rows)
        print(
            f"    {done:5d}/{total}  {100 * done / total:5.1f}%  "
            f"{elapsed / 60:6.1f} min elapsed, {(total - done) / rate / 60:6.1f} min left"
            f"{f', {failed} failed' if failed else ''}",
            flush=True,
        )
        if stream_path is not None:
            # Rewritten whole rather than appended, because a failed run carries no output
            # columns and appending rows of differing width would misalign the file. A full
            # rewrite of a few thousand rows costs nothing next to a single model run.
            pd.DataFrame(rows).to_csv(stream_path, index=False)

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(base, artifact, geography, geo_seed),
        ) as pool:
            futures = [pool.submit(_run_point, p) for p in payloads]
            for future in as_completed(futures):
                record(future.result())
    else:
        _init_worker(base, artifact, geography, geo_seed)
        for payload in payloads:
            record(_run_point(payload))

    return (
        pd.DataFrame(rows)
        .sort_values(["sample", "seed"])
        .reset_index(drop=True)
    )


def average_over_seeds(runs: pd.DataFrame, outputs: list[str], n_samples: int) -> pd.DataFrame:
    """Collapse replicate seeds to one row per sample, in design order.

    Reindexed onto the full design so that row ``i`` of the returned frame is row ``i`` of the
    sample matrix even when every seed of some sample failed -- Sobol' reads ``Y`` positionally,
    so a dropped row would silently misalign every index after it.
    """
    present = [c for c in outputs if c in runs.columns]
    missing = [c for c in outputs if c not in runs.columns]
    if missing:
        print(
            f"  ! outputs not produced by any run, dropped: {', '.join(missing)}",
            file=sys.stderr,
        )
    frame = (
        runs.groupby("sample")[present]
        .mean()  # skips the NaNs left by failed runs
        .reindex(range(n_samples))
    )
    frame.index.name = "sample"
    return frame.reset_index()


# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--bounds", type=Path, default=DEFAULT_BOUNDS)
    parser.add_argument("--n-base", type=int, default=None, help="override sample.n_base")
    parser.add_argument(
        "--replicates", type=int, default=None, help="override sample.replicates"
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="override model.n_steps, for a quick pilot at reduced cost",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="parallel processes; 0 or 1 runs sequentially (default: from yaml, else cpu-1)",
    )
    parser.add_argument("--tag", default=None, help="suffix on the run directory")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the design size and exit without running the model",
    )
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="write the raw sweep only; run sensitivity_analysis.py yourself afterwards",
    )
    args = parser.parse_args(argv)

    constants = load_yaml(args.constants)
    bounds = load_yaml(args.bounds)
    if args.steps is not None:
        constants.setdefault("model", {})["n_steps"] = args.steps
    base = multi_seed_gen.build_params(constants)

    run_cfg = constants.get("run") or {}
    sample_cfg = bounds.get("sample") or {}
    outputs_cfg = bounds.get("outputs") or {}

    problem, integer_list = build_problem(bounds)
    integers = set(integer_list)
    n_base = int(args.n_base or sample_cfg.get("n_base", 64))
    replicates = int(args.replicates or sample_cfg.get("replicates", 2))
    seed_start = int(sample_cfg.get("seed_start", 0))
    seeds = list(range(seed_start, seed_start + max(1, replicates)))
    calc_second_order = bool(sample_cfg.get("calc_second_order", False))

    design = sample_design(problem, n_base, calc_second_order)
    n_samples, n_runs = len(design), len(design) * len(seeds)

    varied_geography = sorted(GEOGRAPHY_FIELDS.intersection(problem["names"]))
    fixed_geography = bool(run_cfg.get("fixed_geography", True)) and not varied_geography
    geo_seed = int(run_cfg.get("geography_seed", 0))

    workers = args.workers if args.workers is not None else run_cfg.get("workers")
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(int(workers), n_runs))

    outputs = list(outputs_cfg.get("analyse") or [])
    if not outputs:
        raise SystemExit("bounds.yaml lists no outputs under outputs.analyse")
    headline = [o for o in (outputs_cfg.get("headline") or outputs[:4]) if o in outputs]

    print(f"Constants  : {args.constants}")
    print(f"Bounds     : {args.bounds}")
    print(
        f"Design     : Sobol', {problem['num_vars']} parameters, n_base={n_base}"
        f"{', second order' if calc_second_order else ', first + total order'}"
    )
    print(f"             {n_samples} sample points x {len(seeds)} seeds = {n_runs} model runs")
    print(f"Parameters : {', '.join(problem['names'])}")
    if integers:
        print(f"             (rounded to integers: {', '.join(sorted(integers))})")
    print(f"Outputs    : {len(outputs)} quantities, headline {', '.join(headline)}")
    print(
        f"Geography  : {'one shared layout' if fixed_geography else 'rebuilt per sample'}"
        + (f" (because {', '.join(varied_geography)} is varied)" if varied_geography else "")
    )
    print(f"Workers    : {workers}")

    if args.dry_run:
        print("\nDry run: nothing executed.")
        return 0

    root = Path(args.outdir or run_cfg.get("outdir", "Results/sensitivity"))
    timestamped = not args.no_timestamp and bool(run_cfg.get("timestamped", True))
    paths = multi_seed_gen.make_run_dir(root, timestamped=timestamped, tag=args.tag)
    outdir, figdir, indir = paths["output"], paths["figures"], paths["input"]
    print(f"Run dir    : {paths['run']}\n")

    artifact = load_artifact()
    geography = (
        build_geography(base, np.random.default_rng(geo_seed), artifact=artifact)
        if fixed_geography
        else None
    )

    # --- inputs, written before the sweep so a killed run is still documented ---------------
    for path in (args.constants, args.bounds):
        if path.is_file():
            shutil.copy2(path, indir / path.name)
    (indir / "params_baseline.json").write_text(
        json.dumps(asdict(base), indent=2, default=str), encoding="utf-8"
    )
    problem_record = {
        **problem,
        "integers": sorted(integers),
        "calc_second_order": calc_second_order,
        "n_base": n_base,
        "replicates": len(seeds),
        "seeds": seeds,
        "outputs": outputs,
        "headline": headline,
    }
    (indir / "problem.json").write_text(
        json.dumps(problem_record, indent=2), encoding="utf-8"
    )
    (indir / "run_meta.json").write_text(
        json.dumps(
            {
                "started": datetime.now().isoformat(timespec="seconds"),
                "command": " ".join(sys.argv),
                "git_commit": multi_seed_gen._git_commit(),
                "constants_path": str(args.constants),
                "bounds_path": str(args.bounds),
                "n_samples": n_samples,
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
    stream_path = outdir / "runs_stream.csv"
    runs = run_sweep(
        base, design, problem, integers, seeds, artifact, geography,
        needs_geography=not fixed_geography, workers=workers, geo_seed=geo_seed,
        stream_path=stream_path,
    )
    minutes = (time.perf_counter() - started) / 60

    # --- tables ------------------------------------------------------------------------------
    samples = pd.DataFrame(design, columns=problem["names"])
    samples.insert(0, "sample", range(len(samples)))
    samples.to_csv(outdir / "samples.csv", index=False)
    runs.to_csv(outdir / "runs_by_sample.csv", index=False)
    stream_path.unlink(missing_ok=True)  # superseded by the sorted table

    y_frame = average_over_seeds(runs, outputs, n_samples)
    y_frame.to_csv(outdir / "Y.csv", index=False)

    failed = int(runs["failed"].sum())
    print(f"\nSwept {len(runs)} runs in {minutes:.1f} min ({failed} failed)")
    if failed:
        reasons = runs.loc[runs["failed"] == 1, "error"].value_counts().head(5)
        for reason, count in reasons.items():
            print(f"    {count:4d}  {reason}")
    print(f"      tables to {outdir}/\n      inputs to {indir}/")

    # --- analysis ----------------------------------------------------------------------------
    if args.no_analysis or not run_cfg.get("analysis", True):
        print(
            f"\nAnalysis skipped. Run it with:\n"
            f"    uv run python src/sensitivity/sensitivity_analysis.py "
            f"--run-dir {paths['run']}"
        )
        return 0

    print("\nSobol' analysis:")
    return sensitivity_analysis.analyse_run(paths["run"], figdir=figdir, outdir=outdir)


if __name__ == "__main__":
    raise SystemExit(main())
