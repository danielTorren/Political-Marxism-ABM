"""Figures for a single diagnostic run.

Two families are produced:

* the **dynamics dashboards** of :mod:`pmabm.diagnostics`, which are shared with the multi-seed
  script and drawn here as plain single-run lines;
* the **spatial and cross-sectional** figures, which only make sense for one run because they
  show a particular England rather than an average of several.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmabm import diagnostics, plots


def plot_all(model, history: pd.DataFrame, outdir: Path) -> list[Path]:
    """Draw every single-run figure into ``outdir`` and return the paths written."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # The dashboards want a long frame with a `t` column; one run is a single "seed".
    frame = history.copy()
    frame["seed"] = model.p.seed
    written += diagnostics.render(frame, diagnostics.single_run_renderer, outdir)

    # Spatial and distributional figures, specific to this run's geography.
    written.append(plots.fig_geography(model, outdir))
    written.append(plots.fig_rq2_maps(model, outdir))
    written.append(plots.fig_tenure(frame, outdir))
    written.append(plots.fig_population(frame, outdir))
    written.append(plots.fig_concentration([model], outdir))
    written.append(plots.fig_consolidation_by_fertility([model], outdir))

    # local: keeps import cost down
    from pmabm.metrics import event_study, spread_variogram, spread_variogram_curve

    events = event_study(model)
    if not events.empty:
        written.append(plots.fig_rq1_event_study(events, outdir))

    spread = spread_variogram(model)
    if spread["n"] >= 10:
        written.append(
            plots.fig_rq2_spread(spread_variogram_curve(model), spread, outdir)
        )

    diagnostics.figure_index().to_csv(outdir / "figure_index.csv", index=False)
    return written


if __name__ == "__main__":  # convenience: regenerate figures from a fresh run
    import single_run_gen

    raise SystemExit(single_run_gen.main())
