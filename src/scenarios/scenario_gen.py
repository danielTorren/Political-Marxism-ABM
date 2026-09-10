"""Run the scenario ladder and the per-research-question arms of ``model_scenarios``.

    uv run python src/scenarios/scenario_gen.py
    uv run python src/scenarios/scenario_gen.py --only ladder rq3
    uv run python src/scenarios/scenario_gen.py --only rq6 --seeds 8

Every arm is a full multi-seed replication against the *same* seed list, so a difference
between arms is never a difference between draws. Output follows the layout the other runners
use, one timestamped directory per invocation:

    Results/scenarios/2026-08-03_143012/
        input_data/    scenarios.yaml as given, plus the resolved parameters of every arm
        output_data/   one parquet file per arm per frame kind, plus a flat summary CSV per group
        figures/       the research-question comparison figures

The whole suite is large -- around sixty arms -- so ``--only`` and ``--seeds`` are the normal
way to use this during development, and the full sweep is an overnight job. The run prints its
own arm and run counts before starting so that is visible in advance rather than in hindsight.

**Output layout.** Tables are written one file per arm, under
``output_data/<group>/<arm>/<kind>.parquet``, rather than one stacked CSV per group. Three
reasons, all of which bite at the scale of the full suite:

* parquet is typed and compressed, so the per-parcel frame -- one row per parcel per seed, tens
  of millions of rows across the suite -- costs a few hundred MB rather than a few GB, and a
  figure can read the two columns it needs instead of parsing every column of every arm;
* one file per arm means a figure comparing two arms reads two small files, and re-running a
  single arm rewrites only that arm;
* the ``summary`` frame is *additionally* written as a flat CSV per group, because it is one row
  per seed and is the table a person actually opens.

``geography.parquet`` is written once per arm rather than once per seed when
``run.fixed_geography`` is set, since the lattice is then shared by every replicate; it carries
no ``seed`` column in that case, and is joined onto ``parcels.parquet`` on ``parcel`` alone.
With per-seed geography it carries one and the join is on both. ``run_meta.json`` records which.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "multi_seed"))  # multi_seed_gen imports multi_seed_plot by name

from pmabm.config import Params  # noqa: E402
from pmabm.geography import build as build_geography, load_artifact  # noqa: E402
from pmabm.metrics import (  # noqa: E402
    concentration_frame,
    county_history_frame,
    event_study,
    geography_frame,
    history_frame,
    occupant_continuity,
    parcel_frame,
    spread_variogram,
    spread_variogram_curve,
    summary,
)
from pmabm.model import Model  # noqa: E402

import multi_seed_gen  # noqa: E402  (make_run_dir, mean_ci, _git_commit)
import scenario_plot  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = SRC.parent / "model_scenarios" / "scenarios.yaml"

#: Fields of Params that the yaml may give as a list and the dataclass wants as a tuple.
TUPLE_FIELDS = (
    "consumption_ratio",
    "consumption_range",
    "customary_rent_range",
    "mobility_range",
    "shock_magnitude",
    "goods_price_cap",
)

#: The frames collected from every run, one row per seed appended to each. ``variogram`` is the
#: binned semivariogram behind RQ2's spread measure; ``parcels`` is the per-parcel frame that
#: makes the spatial figures possible at all; the rest match what the multi-seed runner already
#: collects, so the two sets of tables are directly comparable.
FRAME_KINDS = (
    "history",
    "concentration",
    "events",
    "variogram",
    "parcels",
    "county_history",
    "summary",
)

#: Handled outside :data:`FRAME_KINDS` because its cardinality differs: under
#: ``run.fixed_geography`` it is one table per *arm*, not one per seed. See the module docstring.
GEOGRAPHY_KIND = "geography"

#: Parquet codec. zstd over the default snappy: the per-parcel frame is mostly small integers and
#: repeated categorical metadata, which zstd compresses substantially better at no meaningful
#: cost in read time for frames of this size.
PARQUET_COMPRESSION = "zstd"


@dataclass
class Arm:
    """One parameterisation to be run across every seed."""

    group: str
    name: str
    label: str
    params: Params
    describes: str = ""
    sweep_param: str | None = None
    sweep_value: float | str | None = None
    sweep_values: dict = field(default_factory=dict)
    """Every swept parameter of this arm and its value, including for multi-parameter sweeps.

    ``sweep_param``/``sweep_value`` remain the single-parameter case, because the existing sweep
    figures take a numeric x-axis from them and a cross product has no single axis. This carries
    the full record instead, which is what a two-dimensional frontier or phase diagram needs: each
    entry becomes a ``sweep__<param>`` column on every frame, so a heatmap can pivot on two axes
    without parsing the arm name.
    """

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"


@dataclass
class Group:
    """A set of arms compared against each other, usually one research question."""

    name: str
    question: str = ""
    note: str = ""
    arms: list[Arm] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _coerce(overrides: dict) -> dict:
    out = dict(overrides)
    for key in TUPLE_FIELDS:
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


def make_params(base: dict, overrides: dict) -> Params:
    """Merge an arm's overrides onto the shared base and validate the field names.

    Unknown names are refused rather than ignored: a typo in a scenario file would otherwise
    silently produce a *baseline* run under an ablation's label, which is the single most
    dangerous failure mode this whole script has.
    """
    merged = _coerce({**base, **overrides})
    valid = set(Params.__dataclass_fields__)
    unknown = sorted(set(merged) - valid)
    if unknown:
        raise SystemExit(
            f"Unknown parameter(s) in scenarios.yaml: {', '.join(unknown)}\n"
            f"Valid names are the fields of pmabm/config.py::Params"
        )
    return Params(**merged)


def _expand_sweep(group: str, entry: dict, base: dict) -> list[Arm]:
    """Turn one ``sweep`` entry into one arm per value (cross product if two are swept)."""
    sweep = entry["sweep"]
    label = entry.get("label", entry["name"])
    keys = list(sweep)
    grids = [sweep[k] for k in keys]
    arms: list[Arm] = []
    # itertools.product without importing: the sweeps here are one or two deep.
    combos: list[list] = [[]]
    for grid in grids:
        combos = [combo + [value] for combo in combos for value in grid]
    for combo in combos:
        overrides = {**(entry.get("model") or {}), **dict(zip(keys, combo))}
        tag = "_".join(f"{k}{v}".replace(".", "p") for k, v in zip(keys, combo))
        shown = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        arms.append(
            Arm(
                group=group,
                name=f"{entry['name']}__{tag}",
                label=f"{label}: {shown}",
                params=make_params(base, overrides),
                describes=entry.get("describes", ""),
                # Only single-parameter sweeps get a numeric axis; a cross product has none.
                sweep_param=keys[0] if len(keys) == 1 else None,
                sweep_value=combo[0] if len(keys) == 1 else None,
                sweep_values=dict(zip(keys, combo)),
            )
        )
    return arms


def build_groups(config: dict, only: list[str] | None) -> list[Group]:
    """Resolve the yaml into groups of arms, honouring ``--only``."""
    base = _coerce(config.get("base") or {})
    groups: list[Group] = []

    if config.get("ladder") and (only is None or "ladder" in only):
        ladder = Group(
            name="ladder",
            question="Which mechanism is responsible for which dynamic?",
            note="Tiers are cumulative; the marginal effect of a mechanism is the difference "
            "between consecutive tiers.",
        )
        for entry in config["ladder"]:
            ladder.arms.append(
                Arm(
                    group="ladder",
                    name=entry["name"],
                    label=entry.get("label", entry["name"]),
                    params=make_params(base, entry.get("model") or {}),
                    describes=entry.get("describes", ""),
                )
            )
        groups.append(ladder)

    for name, spec in (config.get("groups") or {}).items():
        if only is not None and name not in only:
            continue
        group = Group(
            name=name,
            question=(spec.get("question") or "").strip(),
            note=(spec.get("note") or "").strip(),
        )
        for entry in spec.get("arms") or []:
            if "sweep" in entry:
                group.arms.extend(_expand_sweep(name, entry, base))
            else:
                group.arms.append(
                    Arm(
                        group=name,
                        name=entry["name"],
                        label=entry.get("label", entry["name"]),
                        params=make_params(base, entry.get("model") or {}),
                        describes=entry.get("describes", ""),
                    )
                )
        if group.arms:
            groups.append(group)

    if only is not None:
        found = {g.name for g in groups}
        missing = [name for name in only if name not in found]
        if missing:
            available = ["ladder", *(config.get("groups") or {})]
            raise SystemExit(
                f"No such group(s): {', '.join(missing)}. Available: {', '.join(available)}"
            )
    return groups


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------
def _tag(frame: pd.DataFrame, arm_meta: dict, seed: int | None = None) -> pd.DataFrame:
    """Label a frame with its arm, and with its seed unless the frame is seed-invariant."""
    if seed is not None:
        frame["seed"] = seed
    for column, value in arm_meta.items():
        frame[column] = value
    return frame


#: The parameters the lattice actually depends on. Everything else an arm varies is
#: behavioural, so arms differing only in behaviour can share one lattice -- and in the shipped
#: suite 209 of the 216 arms do, which is 209 identical rebuilds of a 3-second construction.
_GEOGRAPHY_FIELDS = (
    "L",
    "awareness_radius",
    "lords_per_county",
    "random_awareness_graph",
    "uniform_fertility",
    "zeta",
)

_GEOGRAPHY_CACHE: dict[tuple, object] = {}
_POOLS: dict[tuple, object] = {}

#: Set once per worker process by :func:`_init_worker`.
_WORKER_ARTIFACT: dict | None = None

#: Lattices already built inside this worker, keyed as :func:`_geography_key`. The suite needs
#: only seven distinct lattices across all 216 arms, and 209 of them share one, so this holds a
#: handful of entries and is built at most once each per worker. Rebuilding rather than
#: shipping the lattice in the payload is what lets every arm share one process pool: a pool
#: carrying a pre-built lattice can only serve arms that use that lattice.
_WORKER_GEOGRAPHIES: dict = {}


def _geography_key(params: Params) -> tuple:
    return tuple(getattr(params, field_) for field_ in _GEOGRAPHY_FIELDS)


def _shared_geography(params: Params, artifact: dict, cache: dict | None = None):
    """The lattice for ``params``, built once per distinct geography key.

    A pure function of the six fields in :data:`_GEOGRAPHY_FIELDS`: the generator is seeded at
    zero rather than from the run seed, so every call for a given key returns the same lattice
    and a worker can reproduce the parent's exactly. ``cache`` selects which store to memoise
    into, so the same function serves the parent and each worker.
    """
    store = _GEOGRAPHY_CACHE if cache is None else cache
    key = _geography_key(params)
    geography = store.get(key)
    if geography is None:
        geography = build_geography(params, np.random.default_rng(0), artifact=artifact)
        store[key] = geography
    return geography


def _init_worker(artifact: dict) -> None:
    global _WORKER_ARTIFACT
    _WORKER_ARTIFACT = artifact
    _WORKER_GEOGRAPHIES.clear()


def _get_pool(workers: int, artifact: dict):
    """The single process pool the whole suite runs on.

    One pool rather than one per lattice, which is what allows the work queue to be flat: a
    worker builds whichever lattice a payload asks for and keeps it, so any worker can take any
    arm. With a pool per lattice the suite paid for seven pools of ``workers`` processes and
    could never run two arms at once.
    """
    pool = _POOLS.get("pool")
    if pool is None:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(artifact,)
        )
        _POOLS["pool"] = pool
    return pool


def shutdown_pools() -> None:
    for pool in _POOLS.values():
        pool.shutdown(wait=True)
    _POOLS.clear()


def _run_one_seed(payload: tuple) -> dict[str, pd.DataFrame]:
    """Run one seed of one arm. Module level and picklable, for the process pool."""
    params, seed, arm_meta, want_geography = payload
    # ``want_geography`` doubles as the mode flag: per-seed geography means the lattice is the
    # seed's own and is built inside ``Model``; otherwise the arm shares one, which this worker
    # reproduces from the params rather than receiving.
    geography = (
        None
        if want_geography
        else _shared_geography(params, _WORKER_ARTIFACT, cache=_WORKER_GEOGRAPHIES)
    )
    model = Model(
        params.with_(seed=seed),
        geography=geography,
        artifact=_WORKER_ARTIFACT,
    ).run()

    frames: dict[str, pd.DataFrame] = {
        "history": history_frame(model),
        "concentration": concentration_frame(model),
        "events": event_study(model),
        "variogram": spread_variogram_curve(model),
        "parcels": parcel_frame(model),
        "county_history": county_history_frame(model),
        "summary": pd.DataFrame(
            [
                {
                    **summary(model),
                    **occupant_continuity(model),
                    **{f"spread_{k}": v for k, v in spread_variogram(model).items()},
                }
            ]
        ),
    }
    # Only asked for when the lattice was built inside this worker, i.e. when the arm is running
    # per-seed geography. With a shared lattice the caller already has it and builds it once.
    if want_geography:
        frames[GEOGRAPHY_KIND] = geography_frame(model.geo)
    for frame in frames.values():
        _tag(frame, arm_meta, seed)
    return frames


def _arm_meta(arm: Arm) -> dict:
    """The identifying columns stamped onto every frame this arm produces."""
    return {
        "group": arm.group,
        "arm": arm.name,
        "arm_label": arm.label,
        "sweep_param": arm.sweep_param if arm.sweep_param is not None else "",
        "sweep_value": arm.sweep_value if arm.sweep_value is not None else np.nan,
        # One column per swept parameter, so a two-way sweep is pivotable. Prefixed rather than
        # named bare because a swept parameter can share a name with a recorded output.
        **{f"sweep__{k}": v for k, v in arm.sweep_values.items()},
    }


def _arm_payloads(
    arm: Arm, seeds: list[int], fixed_geography: bool
) -> list[tuple]:
    """The (arm, seed) tasks for one arm, ready for the pool.

    The geography is per **arm**, not one for the whole suite, because several arms change it:
    ``uniform_fertility`` (RQ6) is applied while the lattice is constructed, so a layout shared
    across arms would silently make that ablation a no-op. Within an arm it is still shared
    across seeds when ``fixed_geography`` is set, which is what makes the spatial comparisons
    in RQ2 and RQ6 comparisons of event history rather than of layout -- the worker rebuilds
    that shared lattice from the params, which is exact because it is seeded at zero.
    """
    arm_meta = _arm_meta(arm)
    per_seed_geography = not fixed_geography
    return [(arm.params, seed, arm_meta, per_seed_geography) for seed in seeds]


def _collect_arm(
    arm: Arm, collected: list[dict], artifact: dict, fixed_geography: bool
) -> dict[str, pd.DataFrame]:
    """Concatenate one arm's per-seed frames into the arm's tables."""
    geography = _shared_geography(arm.params, artifact) if fixed_geography else None
    per_seed_geography = geography is None
    frames = {
        kind: pd.concat([c[kind] for c in collected], ignore_index=True)
        for kind in FRAME_KINDS
    }
    # A shared lattice is one table for the whole arm and carries no seed column; a per-seed
    # lattice is stacked like everything else and does. The join key differs accordingly, which
    # is why ``run_meta.json`` records ``fixed_geography``.
    frames[GEOGRAPHY_KIND] = (
        pd.concat([c[GEOGRAPHY_KIND] for c in collected], ignore_index=True)
        if per_seed_geography
        else _tag(geography_frame(geography), _arm_meta(arm))
    )
    return frames


def write_group_tables(
    group: Group, frames: dict[str, dict[str, pd.DataFrame]], outdir: Path
) -> int:
    """One parquet file per arm per frame kind, plus one flat summary CSV for the group.

    Returns the number of files written. Empty frames are skipped rather than written as empty
    files -- ``variogram`` is empty for an arm in which almost nothing converted, and ``events``
    for one in which nothing did -- so a missing file means "this arm had no such observations",
    which is the same convention :mod:`scenario_plot` follows when it skips a panel.
    """
    written = 0
    for arm in group.arms:
        if arm.name not in frames:
            continue
        arm_dir = outdir / group.name / arm.name
        arm_dir.mkdir(parents=True, exist_ok=True)
        for kind, frame in frames[arm.name].items():
            if frame is None or frame.empty:
                continue
            frame.to_parquet(
                arm_dir / f"{kind}.parquet", index=False, compression=PARQUET_COMPRESSION
            )
            written += 1

    # The one table meant to be read by a person rather than by a figure: one row per seed per
    # arm, small enough to open anywhere, and the first thing to look at after a run.
    parts = [frames[arm.name]["summary"] for arm in group.arms if arm.name in frames]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            outdir / f"{group.name}_summary.csv", index=False
        )
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="group names to run, e.g. --only ladder rq3 rq6 (default: all)",
    )
    parser.add_argument("--seeds", type=int, default=None, help="override run.n_seeds")
    parser.add_argument("--steps", type=int, default=None, help="override n_steps on every arm")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tag", default=None, help="suffix on the run directory")
    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="write straight into outdir instead of a timestamped subdirectory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve and list the arms without running anything",
    )
    args = parser.parse_args(argv)

    config = load_config(args.scenarios)
    run_cfg = config.get("run") or {}
    groups = build_groups(config, args.only)

    if args.steps is not None:
        for group in groups:
            for arm in group.arms:
                arm.params = arm.params.with_(n_steps=args.steps)

    n_seeds = args.seeds or int(run_cfg.get("n_seeds", 32))
    start = int(run_cfg.get("seed_start", 0))
    seeds = list(range(start, start + n_seeds))
    fixed_geography = bool(run_cfg.get("fixed_geography", True))

    workers = args.workers if args.workers is not None else run_cfg.get("workers")
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    # Clamped to the total task count, not to the seed count: the queue is flat across every
    # (arm, seed) pair in the suite, so there is no per-arm ceiling on how many run at once.
    n_arms_total = sum(len(g.arms) for g in groups)
    workers = max(1, min(int(workers), max(1, n_arms_total * n_seeds)))

    n_arms = n_arms_total
    print(f"Scenarios : {args.scenarios}")
    print(f"Groups    : {', '.join(g.name for g in groups)}")
    print(f"Arms      : {n_arms}  ({n_arms * n_seeds} runs at {n_seeds} seeds)")
    print(f"Workers   : {workers}{', fixed geography per arm' if fixed_geography else ''}")

    if args.dry_run:
        for group in groups:
            print(f"\n{group.name}: {group.question}")
            for arm in group.arms:
                print(f"  {arm.name:44s} {arm.label}")
        return 0

    root = Path(args.outdir or run_cfg.get("outdir", "Results/scenarios"))
    timestamped = not args.no_timestamp and bool(run_cfg.get("timestamped", True))
    paths = multi_seed_gen.make_run_dir(root, timestamped=timestamped, tag=args.tag)
    outdir, figdir, indir = paths["output"], paths["figures"], paths["input"]
    print(f"Run dir   : {paths['run']}\n")

    # --- record the inputs -------------------------------------------------------------------
    import shutil

    if args.scenarios.is_file():
        shutil.copy2(args.scenarios, indir / args.scenarios.name)
    (indir / "arms_resolved.json").write_text(
        json.dumps(
            {
                group.name: {
                    arm.name: {"label": arm.label, "params": asdict(arm.params)}
                    for arm in group.arms
                }
                for group in groups
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
                "scenarios_path": str(args.scenarios),
                "groups": [g.name for g in groups],
                "n_arms": n_arms,
                "seeds": seeds,
                "workers": workers,
                "fixed_geography": fixed_geography,
                "run_dir": str(paths["run"]),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # --- run ----------------------------------------------------------------------------------
    artifact = load_artifact()
    results: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    done = 0
    tables = 0
    # One flat queue over every (arm, seed) pair in the suite. Arms differ in cost by more
    # than an order of magnitude -- rq4_frontier is 64 arms and rq1 is 3 -- and running them in
    # sequence meant the pool drained to idle at each arm boundary and could never exceed
    # ``n_seeds`` busy workers. Flat, a free worker takes the next run from anywhere.
    all_arms = [(group, arm) for group in groups for arm in group.arms]
    payloads: list[tuple] = []
    spans: list[tuple] = []  # (group, arm, start, stop) into the flat result list
    for group, arm in all_arms:
        arm_payloads = _arm_payloads(arm, seeds, fixed_geography)
        spans.append((group, arm, len(payloads), len(payloads) + len(arm_payloads)))
        payloads.extend(arm_payloads)

    print(f"Running {len(payloads)} model runs on {workers} workers ...")
    try:
        if workers > 1:
            pool = _get_pool(workers, artifact)
            # chunksize stays at 1: a run costs seconds, so dispatch overhead is noise, while
            # any larger chunk hands one worker a fixed block and reintroduces the tail the
            # flat queue exists to remove.
            flat = list(pool.map(_run_one_seed, payloads, chunksize=1))
        else:
            _init_worker(artifact)
            flat = [_run_one_seed(payload) for payload in payloads]

        by_group: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        for group, arm, start, stop in spans:
            frames = _collect_arm(arm, flat[start:stop], artifact, fixed_geography)
            by_group.setdefault(group.name, {})[arm.name] = frames
            done += 1
            final = frames["summary"]
            lease = final["final_share_leasehold"].mean()
            gini = final["final_farm_gini"].mean()
            print(
                f"  [{done:3d}/{n_arms}] {group.name}/{arm.name:36s} "
                f"leasehold={lease:5.3f}  gini={gini:5.3f}"
            )
        for group in groups:
            collected = by_group.get(group.name, {})
            results[group.name] = collected
            tables += write_group_tables(group, collected, outdir)
    finally:
        # The pools outlive individual arms, so they are closed here rather than per arm --
        # including on the way out of a failed or interrupted run.
        shutdown_pools()

    # --- figures ------------------------------------------------------------------------------
    written = scenario_plot.plot_all(groups, results, figdir, datadir=outdir)

    size_mb = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file()) / 1e6
    print(
        f"\nWrote {len(written)} figures to {figdir}/"
        f"\n      {tables} tables to {outdir}/ ({size_mb:.0f} MB)"
        f"\n      inputs to {indir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
