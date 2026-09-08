# See what's there without running anything
uv run python src/scenarios/scenario_gen.py --dry-run

# One group, small, for plumbing checks
uv run python src/scenarios/scenario_gen.py --only rq2 --seeds 4 --steps 50

# The full suite
uv run python src/scenarios/scenario_gen.py

# Sensitivity (sweep, then analyse; the analysis is seconds and re-runnable)
uv run python src/sensitivity/sensitivity_gen.py
uv run python src/sensitivity/sensitivity_analysis.py

