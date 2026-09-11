"""Run the model across many seeds and hand the results to :mod:`multi_seed_plot`.

    uv run python src/multi_seed/multi_seed_gen.py
    uv run python src/multi_seed/multi_seed_gen.py --seeds 20

Every series is reported as a mean across seeds with a 95% confidence interval on that mean,
so the figures answer "what does this model typically do" rather than "what happened in run 0".

Each invocation writes a fresh timestamped run directory, so runs accumulate instead of
overwriting one another:

    Results/multi_seed/multi_seed_2026-08-03_143012/
        input_data/    the constants file as given, plus the fully resolved parameters
        output_data/   every table
        figures/       every figure

The resolved parameters, not just the yaml, are what makes a run reproducible: defaults from
``pmabm/config.py`` that the yaml never mentions are recorded explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmabm.config import Params  # noqa: E402
from pmabm.geography import build as build_geography, load_artifact  # noqa: E402
from pmabm.metrics import (  # noqa: E402
    concentration_frame,
    consolidation_by_fertility,
    event_study,
    history_frame,
    occupant_continuity,
    spread_variogram,
    spread_variogram_curve,
    summary,
)
from pmabm.model import Model  # noqa: E402

import multi_seed_plot  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONSTANTS = HERE / "constants" / "constants.yaml"


def load_constants(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_params(constants: dict) -> Params:
    overrides = dict(constants.get("model") or {})
    valid = set(Params.__dataclass_fields__)
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise SystemExit(
            f"Unknown parameter(s) in constants.yaml: {', '.join(unknown)}\n"
            f"Valid names are in pmabm/config.py"
        )
    for key in ("consumption_ratio", "consumption_range", "customary_rent_range",
                "mobility_range", "shock_magnitude"):
        if key in overrides and isinstance(overrides[key], list):
            overrides[key] = tuple(overrides[key])
    return Params(**overrides)


#: How a run directory is named. Date first, so that listing a suite's directory puts its runs
#: in the order they were made -- a stamp led by the clock time sorts by hour of day and
#: interleaves different dates, which is exactly wrong for a directory of accumulated runs.
RUN_STAMP = "%Y-%m-%d_%H%M%S"


def run_dir_name(root: Path, tag: str | None = None, when: datetime | None = None) -> str:
    """The name of one run directory under ``root``: ``<suite>_<stamp>[-tag]``.

    The suite name is the leaf of ``root`` (``Results/scenarios`` gives ``scenarios``), so a
    directory carries its own provenance once it is copied off the cluster or dropped next to
    output from another suite.
    """
    stamp = (when or datetime.now()).strftime(RUN_STAMP)
    name = f"{root.name}_{stamp}" if root.name else stamp
    return f"{name}-{tag}" if tag else name


def make_run_dir(root: Path, timestamped: bool = True, tag: str | None = None) -> dict[str, Path]:
    """Create ``root/<suite>_<timestamp>[-tag]/{input_data,output_data,figures}``, return paths.

    Timestamped by default so that successive runs accumulate rather than silently overwrite
    each other; ``timestamped=False`` restores the old flat behaviour for anything that expects
    a fixed path, which is what the SLURM array script uses to key its directories by job id
    instead.
    """
    if timestamped:
        run = root / run_dir_name(root, tag)
    else:
        run = root
    paths = {
        "run": run,
        "input": run / "input_data",
        "output": run / "output_data",
        "figures": run / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _git_commit() -> str | None:
    """Short commit of the working tree, so a run directory says which code produced it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=HERE, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def write_inputs(
    indir: Path, constants_path: Path, constants: dict, params: Params, run_meta: dict
) -> None:
    """Record everything needed to repeat the run: the yaml as given and the resolved state."""
    if constants_path.is_file():
        shutil.copy2(constants_path, indir / constants_path.name)
    (indir / "constants_resolved.yaml").write_text(
        yaml.safe_dump(constants, sort_keys=False), encoding="utf-8"
    )
    # asdict() flattens tuple-valued fields to lists, which round-trip back through build_params.
    (indir / "params.json").write_text(
        json.dumps(asdict(params), indent=2, default=str), encoding="utf-8"
    )
    (indir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, default=str), encoding="utf-8"
    )


def mean_ci(frame: pd.DataFrame, by: str = "t") -> pd.DataFrame:
    """Collapse a long multi-seed frame to mean, 95% CI bounds and n, per column."""
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["seed"], errors="ignore")
    numeric[by] = frame[by]
    grouped = numeric.groupby(by)
    mean = grouped.mean()
    count = grouped.count().clip(lower=1)
    sem = grouped.std(ddof=1) / np.sqrt(count)
    out = mean.add_suffix("_mean")
    out = out.join((mean - 1.96 * sem).add_suffix("_lo"))
    out = out.join((mean + 1.96 * sem).add_suffix("_hi"))
    out["n_seeds"] = grouped.size()
    return out.reset_index()


def _run_one_seed(payload: tuple) -> dict:
    """Run a single seed and return only frames, so workers need not ship whole models back.

    Module level and picklable, because process workers cannot take a closure.
    """
    params, seed, label, artifact, geography = payload
    model = Model(params.with_(seed=seed), geography=geography, artifact=artifact).run()
    frames = {}
    for key, frame in (
        ("history", history_frame(model)),
        ("concentration", concentration_frame(model)),
        ("fertility", consolidation_by_fertility(model)),
        ("events", event_study(model)),
        ("variogram", spread_variogram_curve(model)),
    ):
        frame = frame.copy()
        frame["seed"] = seed
        frame["arm"] = label
        frames[key] = frame
    # One spread measure, not two. The old pair existed because the origin varied by seed, so the
    # run's own slope and the paper's ecological-seed slope were different numbers and both had to
    # be carried. The variogram has no origin, so there is one number per run and it means the
    # same thing in every arm.
    frames["summary"] = pd.DataFrame(
        [
            {
                "arm": label,
                "seed": seed,
                **summary(model),
                **occupant_continuity(model),
                **{f"spread_{k}": v for k, v in spread_variogram(model).items()},
            }
        ]
    )
    return frames


def run_arm(
    params: Params,
    seeds: list[int],
    label: str,
    artifact: dict,
    workers: int = 1,
    geography=None,
) -> dict:
    """Run one arm (a regime) across seeds, collecting long frames and per-seed summaries.

    Seeds are independent, so they are run in parallel where ``workers > 1``. Results are
    reassembled in seed order, which keeps the output byte-identical to a sequential run.
    """
    payloads = [(params, seed, label, artifact, geography) for seed in seeds]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collected = list(pool.map(_run_one_seed, payloads))
    else:
        collected = []
        for payload in payloads:
            collected.append(_run_one_seed(payload))
            print(f"    seed {payload[1]:3d}  done")

    return {
        key: pd.concat([c[key] for c in collected], ignore_index=True)
        for key in ("history", "concentration", "fertility", "events", "summary")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--seeds", type=int, default=None, help="override run.n_seeds")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="parallel processes; 0 or 1 runs sequentially (default: from yaml, else cpu-1)",
    )
    parser.add_argument(
        "--fixed-geography", action="store_true",
        help="reuse one estate layout across all seeds, so only event history varies",
    )
    parser.add_argument(
        "--tag", default=None,
        help="suffix appended to the timestamped run directory, e.g. --tag household-demography",
    )
    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="write straight into outdir instead of a timestamped subdirectory",
    )
    args = parser.parse_args(argv)

    constants = load_constants(args.constants)
    params = build_params(constants)
    run_cfg = constants.get("run") or {}
    n_seeds = args.seeds or int(run_cfg.get("n_seeds", 10))
    start = int(run_cfg.get("seed_start", 0))
    seeds = list(range(start, start + n_seeds))
    root = Path(args.outdir or run_cfg.get("outdir", "Results/multi_seed"))
    timestamped = not args.no_timestamp and bool(run_cfg.get("timestamped", True))
    paths = make_run_dir(root, timestamped=timestamped, tag=args.tag)
    outdir, figdir, indir = paths["output"], paths["figures"], paths["input"]

    artifact = load_artifact()  # the England outline and ALC baselines, identical everywhere

    workers = args.workers if args.workers is not None else run_cfg.get("workers")
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(int(workers), n_seeds))

    # By default each seed also redraws the estate layout, so cross-seed spread mixes
    # structural variation with event variation. Holding the layout fixed separates them.
    fixed = args.fixed_geography or bool(run_cfg.get("fixed_geography", False))
    geography = None
    if fixed:
        import numpy as _np

        geography = build_geography(params, _np.random.default_rng(0), artifact=artifact)

    write_inputs(
        indir,
        args.constants,
        constants,
        params,
        {
            "started": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git_commit": _git_commit(),
            "constants_path": str(args.constants),
            "seeds": seeds,
            "n_seeds": n_seeds,
            "workers": workers,
            "fixed_geography": fixed,
            "compare_regimes": bool(run_cfg.get("compare_regimes", False)),
            "france_theta": float(run_cfg.get("france_theta", 6.0)),
            "run_dir": str(paths["run"]),
        },
    )

    print(f"Constants : {args.constants}")
    print(f"Run dir   : {paths['run']}")
    print(
        f"Running   : {n_seeds} seeds x {params.n_steps} periods, "
        f"{workers} worker{'s' if workers > 1 else ''}"
        f"{', fixed geography' if fixed else ''}"
    )

    print("  England arm:")
    arms = {
        "England": run_arm(params, seeds, "England", artifact, workers, geography)
    }

    if run_cfg.get("compare_regimes", False):
        france_theta = float(run_cfg.get("france_theta", 6.0))
        print(f"  France arm (theta={france_theta}):")
        arms["France"] = run_arm(
            params.with_(theta=france_theta), seeds, "France", artifact, workers, geography
        )

    # --- tables ---------------------------------------------------------------------------
    for key in ("history", "concentration", "fertility", "summary"):
        combined = pd.concat([arm[key] for arm in arms.values()], ignore_index=True)
        combined.to_csv(outdir / f"{key}_by_seed.csv", index=False)

    england = arms["England"]
    mean_ci(england["history"]).to_csv(outdir / "history_mean_ci.csv", index=False)
    mean_ci(england["concentration"]).to_csv(outdir / "concentration_mean_ci.csv", index=False)

    # --- figures ---------------------------------------------------------------------------
    written = multi_seed_plot.plot_all(arms, figdir, datadir=outdir)

    # --- headline table ---------------------------------------------------------------------
    print("\nFinal-period means with 95% CI (across seeds)")
    for label, arm in arms.items():
        final = arm["summary"]
        print(f"\n  {label}")
        for column in (
            "final_share_leasehold", "final_share_freehold", "final_share_customary",
            "final_farm_gini", "conversion_share",
            # Nugget and slope together: a low nugget with a positive slope is a spreading
            # front, a nugget near 1 with a flat slope is simultaneity.
            "spread_nugget_share", "spread_slope_norm", "spread_range_cells",
            "same_occupant_share",
        ):
            if column not in final:
                continue
            values = final[column].dropna()
            if values.empty:
                continue
            ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            print(f"    {column:26s} {values.mean():8.3f}  +/- {ci:.3f}")

    (outdir / "summary.json").write_text(
        json.dumps(
            {
                label: arm["summary"].mean(numeric_only=True).to_dict()
                for label, arm in arms.items()
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    tables = len(list(outdir.glob("*.csv"))) + len(list(outdir.glob("*.json")))
    print(
        f"\nWrote {len(written)} figures to {figdir}/"
        f"\n      {tables} tables to {outdir}/"
        f"\n      inputs to {indir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
