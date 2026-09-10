"""Command line entry point.

    uv run pmabm build-geography    fetch ONS + Natural England data (once)
    uv run pmabm run                one baseline run, with summary and tenure figure
    uv run pmabm experiments        the full RQ1-RQ3 suite plus robustness checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ENGLAND, Params

DEFAULT_OUTDIR = Path("Results")


def _params_from_args(args: argparse.Namespace) -> Params:
    overrides = {"n_steps": args.steps, "seed": args.seed}
    if args.landlords is not None:
        overrides["n_landlords"] = args.landlords
    if args.grid is not None:
        overrides["L"] = args.grid
    if getattr(args, "literal_wage_bill", False):
        overrides["charge_wage_bill"] = False
    return ENGLAND.with_(**overrides)


def cmd_build_geography(args: argparse.Namespace) -> None:
    from .build_geography import DEFAULT_OUTPUT, build

    build(output=args.output or DEFAULT_OUTPUT, workers=args.workers)


def cmd_run(args: argparse.Namespace) -> None:
    from . import plots
    from .experiments import run_one
    from .metrics import history_frame

    params = _params_from_args(args)
    print(f"Running {params.n_steps} periods, seed {params.seed} ...")
    result = run_one(params, label="england")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(outdir / "history.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")

    print("\nSummary")
    for key, value in result.summary.items():
        print(f"  {key:34s} {value:.4g}" if isinstance(value, float) else f"  {key:34s} {value}")

    history = result.history.assign(label="england", seed=params.seed)
    plots.fig_geography(result.model, outdir)
    plots.fig_tenure(history, outdir)
    print(f"\nWrote history, summary and figures to {outdir}/")


def cmd_experiments(args: argparse.Namespace) -> None:
    from . import plots
    from .experiments import run_all
    from .experiments import run_one

    base = _params_from_args(args)
    seeds = list(range(args.replicates))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = run_all(base, seeds=seeds)
    for name, frame in results.items():
        frame.to_csv(outdir / f"{name}.csv", index=False)

    print("\nBuilding figures ...")
    # A small set of full runs kept in memory, for the figures that need parcel-level panels
    # rather than the aggregate history.
    detailed = [run_one(base.with_(seed=s), label="england").model for s in seeds[:3]]
    reference = detailed[0]
    plots.fig_geography(reference, outdir)
    plots.fig_rq2_maps(reference, outdir)
    plots.fig_concentration(detailed, outdir)
    plots.fig_consolidation_by_fertility(detailed, outdir)
    plots.fig_tenure(
        results["rq3_regimes"][results["rq3_regimes"]["label"] == "England"], outdir
    )
    plots.fig_rq1_event_study(results["rq1_events"], outdir)
    plots.fig_rq2_spread(results["rq2_scatter"], outdir)
    plots.fig_rq2_ablation(results["rq2_ablation"], outdir)
    plots.fig_rq3_regimes(results["rq3_regimes"], outdir)
    plots.fig_robustness(results["robustness"], outdir)

    _print_headlines(results)
    print(f"\nWrote all frames and figures to {outdir}/")


def _print_headlines(results: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 74)
    print("HEADLINE RESULTS")
    print("=" * 74)

    continuity = results["rq1_continuity"]
    if not continuity.empty:
        print("\nRQ1  improvement as consequence, not precondition")
        print(f"  conversions per run          {continuity['n_conversions'].mean():.0f}")
        print(
            "  of which kept the sitting tenant "
            f"{continuity['same_occupant_share'].mean():.1%}"
        )
        print(
            "  -> where this is near zero the event study measures a change of *occupant*, "
            "not\n     a change of disposition in one tenant. See README, spec note 4."
        )

    ablation = results["rq2_ablation"]
    if not ablation.empty:
        print("\nRQ2  distance-lag slope by channel set (median over seeds)")
        table = (
            ablation.groupby("channels")
            .agg(slope=("slope", "median"), converted=("conversion_share", "median"))
            .sort_values("slope", ascending=False)
        )
        for name, row in table.iterrows():
            print(f"  {name:26s} slope {row['slope']:+7.3f}   converted {row['converted']:6.1%}")

    regimes = results["rq3_regimes"]
    if not regimes.empty:
        print("\nRQ3  final shares by regime (median over seeds)")
        final = regimes[regimes["t"] == regimes["t"].max()]
        for label, group in final.groupby("label"):
            print(
                f"  {label:8s} leasehold {group['share_leasehold'].median():6.1%}   "
                f"freehold {group['share_freehold'].median():6.1%}   "
                f"customary {group['share_customary'].median():6.1%}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pmabm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-geography", help="fetch and cache the geography artifact")
    build.add_argument(
        "--output", type=Path, default=None,
        help="where to write the cached artifact (default data/geography/england_alc.json)",
    )
    build.add_argument(
        "--workers", type=int, default=8, help="concurrent ALC sampling requests",
    )
    build.set_defaults(func=cmd_build_geography)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--steps", type=int, default=200)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--landlords", type=int, default=None)
    common.add_argument("--grid", type=int, default=None, help="lattice long-axis length L")
    common.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    common.add_argument(
        "--literal-wage-bill",
        action="store_true",
        help="reproduce the paper literally, with hired labour free (diverges; see README)",
    )

    run = sub.add_parser("run", parents=[common], help="a single baseline run")
    run.set_defaults(func=cmd_run)

    experiments = sub.add_parser("experiments", parents=[common], help="the full RQ suite")
    experiments.add_argument("--replicates", type=int, default=5, help="number of seeds")
    experiments.set_defaults(func=cmd_experiments)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
