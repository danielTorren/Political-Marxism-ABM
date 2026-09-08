# See what's there without running anything
uv run python src/scenarios/scenario_gen.py --dry-run

# One group, small, for plumbing checks
uv run python src/scenarios/scenario_gen.py --only rq2 --seeds 4 --steps 50

# The full suite
uv run python src/scenarios/scenario_gen.py

# Sensitivity (sweep, then analyse; the analysis is seconds and re-runnable)
uv run python src/sensitivity/sensitivity_gen.py
uv run python src/sensitivity/sensitivity_analysis.py


# Emulator (needs `uv sync --extra emulator`)
uv run python src/emulator/emulator_gen.py --dry-run                       # design size + cost, no runs
uv run python src/emulator/emulator_gen.py --n-points 256 --replicates 2 --steps 60   # pilot corpus
uv run python src/emulator/emulator_gen.py                                 # the real corpus, ~8 h
uv run python src/emulator/emulator_train.py --run-dir Results/emulator/<timestamp>
uv run python src/emulator/emulator_plot.py  --model-dir Results/emulator/<timestamp>/emulator
uv run python src/emulator/emulator_apply.py sobol --model-dir Results/emulator/<timestamp>/emulator
