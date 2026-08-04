"""Run the model once and hand the results to :mod:`single_run_plot`.

    uv run python src/single_run/single_run_gen.py
    uv run python src/single_run/single_run_gen.py --constants path/to/other.yaml

Parameters come from ``constants/constants.yaml`` beside this file; anything not named there
falls back to the default in ``pmabm.config``, which mirrors Table 1 of the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow running this file directly, without the package having to be on sys.path already.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pmabm.config import Params  # noqa: E402
from pmabm.metrics import (  # noqa: E402
    concentration_frame,
    consolidation_by_fertility,
    event_study,
    history_frame,
    occupant_continuity,
    region_consolidation_summary,
    spread_variogram,
    summary,
)
from pmabm.model import Model  # noqa: E402

import single_run_plot  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONSTANTS = HERE / "constants" / "constants.yaml"


def load_constants(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_params(constants: dict) -> Params:
    """Turn the ``model:`` block into a :class:`Params`, rejecting unknown keys loudly."""
    overrides = dict(constants.get("model") or {})
    seed = (constants.get("run") or {}).get("seed")
    if seed is not None:
        overrides.setdefault("seed", seed)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--outdir", type=Path, default=None, help="override the yaml outdir")
    args = parser.parse_args(argv)

    constants = load_constants(args.constants)
    params = build_params(constants)
    run_cfg = constants.get("run") or {}
    outdir = Path(args.outdir or run_cfg.get("outdir", "Results/single_run"))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Constants : {args.constants}")
    print(f"Running   : {params.n_steps} periods, seed {params.seed}, theta {params.theta}")
    model = Model(params).run()
    print(f"Parcels   : {model.geo.n_parcels} across {model.geo.n_landlords} estates")

    # --- tabular outputs -----------------------------------------------------------------
    history = history_frame(model)
    history.to_csv(outdir / "history.csv", index=False)
    concentration_frame(model).to_csv(outdir / "concentration.csv", index=False)
    consolidation_by_fertility(model).to_csv(outdir / "consolidation_by_fertility.csv", index=False)
    region_consolidation_summary(model).to_csv(outdir / "regions.csv", index=False)
    event_study(model).to_csv(outdir / "event_study.csv", index=False)

    stats = {**summary(model), **occupant_continuity(model), **spread_variogram(model)}
    (outdir / "summary.json").write_text(json.dumps(stats, indent=2, default=float), encoding="utf-8")

    if run_cfg.get("save_panels", False):
        np.savez_compressed(
            outdir / "panels.npz",
            state=model.panel_state,
            occupant=model.panel_occupant,
            holding=model.panel_holding,
            iota=model.panel_iota,
            capital=model.panel_k,
            rent=model.panel_rho,
            output=model.panel_y,
            first_conversion=model.first_conversion,
            phi_bar=model.geo.phi_bar,
            # Lattice coordinates, so any spatial statistic can be recomputed off the saved
            # panels without re-running: the variogram needs only these two arrays.
            xy=model.geo.xy,
        )

    # --- figures --------------------------------------------------------------------------
    written = single_run_plot.plot_all(model, history, outdir)

    print(f"\nWrote {len(written)} figures and 5 tables to {outdir}/")
    print("\nSummary")
    for key, value in stats.items():
        formatted = f"{value:.4g}" if isinstance(value, float) else value
        print(f"  {key:34s} {formatted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
