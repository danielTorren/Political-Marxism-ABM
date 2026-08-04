"""Run the scenario ladder and the per-research-question arms of ``model_scenarios``.

    uv run python src/scenarios/scenario_gen.py
    uv run python src/scenarios/scenario_gen.py --only ladder rq3
    uv run python src/scenarios/scenario_gen.py --only rq6 --seeds 8

Every arm is a full multi-seed replication against the *same* seed list, so a difference
between arms is never a difference between draws. Output follows the layout the other runners
use, one timestamped directory per invocation:

    Results/scenarios/2026-08-03_143012/
        input_data/    scenarios.yaml as given, plus the resolved parameters of every arm
        output_data/   one table per group per frame kind
        figures/       the research-question comparison figures

The whole suite is large -- around sixty arms -- so ``--only`` and ``--seeds`` are the normal
way to use this during development, and the full sweep is an overnight job. The run prints its
own arm and run counts before starting so that is visible in advance rather than in hindsight.
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
    event_study,
    history_frame,
    occupant_continuity,
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

#: The frames collected from every run. ``variogram`` is the binned semivariogram behind RQ2's
#: spread measure; the rest match what the multi-seed runner already collects, so the two sets of
#: tables are directly comparable.
FRAME_KINDS = ("history", "concentration", "events", "variogram", "summary")


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
def _run_one_seed(payload: tuple) -> dict[str, pd.DataFrame]:
    """Run one seed of one arm. Module level and picklable, for the process pool."""
    params, seed, arm_meta, artifact, geography = payload
    model = Model(params.with_(seed=seed), geography=geography, artifact=artifact).run()

    frames: dict[str, pd.DataFrame] = {
        "history": history_frame(model),
        "concentration": concentration_frame(model),
        "events": event_study(model),
        "variogram": spread_variogram_curve(model),
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
    for frame in frames.values():
        frame["seed"] = seed
        for column, value in arm_meta.items():
            frame[column] = value
    return frames


def run_arm(
    arm: Arm, seeds: list[int], artifact: dict, workers: int, fixed_geography: bool
) -> dict[str, pd.DataFrame]:
    """Run one arm across every seed and concatenate its frames.

    The geography is built **per arm**, not once for the whole suite, because several arms
    change it: ``uniform_fertility`` (RQ6) is applied while the lattice is constructed, so a
    layout shared across arms would silently make that ablation a no-op. Within an arm it is
    still shared across seeds when ``fixed_geography`` is set, which is what makes the spatial
    comparisons in RQ2 and RQ6 comparisons of event history rather than of layout.
    """
    geography = (
        build_geography(arm.params, np.random.default_rng(0), artifact=artifact)
        if fixed_geography
        else None
    )
    arm_meta = {
        "group": arm.group,
        "arm": arm.name,
        "arm_label": arm.label,
        "sweep_param": arm.sweep_param if arm.sweep_param is not None else "",
        "sweep_value": arm.sweep_value if arm.sweep_value is not None else np.nan,
    }
    payloads = [(arm.params, seed, arm_meta, artifact, geography) for seed in seeds]

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            collected = list(pool.map(_run_one_seed, payloads))
    else:
        collected = [_run_one_seed(p) for p in payloads]

    return {
        kind: pd.concat([c[kind] for c in collected], ignore_index=True)
        for kind in FRAME_KINDS
    }


def write_group_tables(
    group: Group, frames: dict[str, dict[str, pd.DataFrame]], outdir: Path
) -> None:
    """One table per frame kind per group, with every arm stacked and labelled."""
    for kind in FRAME_KINDS:
        parts = [frames[arm.name][kind] for arm in group.arms if arm.name in frames]
        if not parts:
            continue
        pd.concat(parts, ignore_index=True).to_csv(
            outdir / f"{group.name}_{kind}.csv", index=False
        )


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
    workers = max(1, min(int(workers), n_seeds))

    n_arms = sum(len(g.arms) for g in groups)
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
    for group in groups:
        print(f"{group.name} ({len(group.arms)} arms)")
        collected: dict[str, dict[str, pd.DataFrame]] = {}
        for arm in group.arms:
            collected[arm.name] = run_arm(arm, seeds, artifact, workers, fixed_geography)
            done += 1
            final = collected[arm.name]["summary"]
            lease = final["final_share_leasehold"].mean()
            gini = final["final_farm_gini"].mean()
            print(
                f"  [{done:3d}/{n_arms}] {arm.name:42s} "
                f"leasehold={lease:5.3f}  gini={gini:5.3f}"
            )
        results[group.name] = collected
        write_group_tables(group, collected, outdir)

    # --- figures ------------------------------------------------------------------------------
    written = scenario_plot.plot_all(groups, results, figdir, datadir=outdir)

    tables = len(list(outdir.glob("*.csv")))
    print(
        f"\nWrote {len(written)} figures to {figdir}/"
        f"\n      {tables} tables to {outdir}/"
        f"\n      inputs to {indir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
