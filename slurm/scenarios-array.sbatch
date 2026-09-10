#!/bin/bash
# The scenario suite as a job array: one task per group, 64 seeds each.
#
#   mkdir -p slurm/logs && sbatch slurm/scenarios-array.sbatch
#
# An alternative to slurm/scenarios.sbatch, not an upgrade: the single job now uses a flat
# work queue and saturates all the cores it is given, so this exists for two other reasons.
# A group that fails or runs out of time can be resubmitted on its own, where a single job
# loses all 10,624 runs; and 13 modest allocations often schedule sooner than one 128-core
# node.
#
# Nothing is lost by splitting: every figure is built from one group's results
# (scenario_plot.plot_all indexes results by group name), so each task writes a complete,
# self-contained set of tables and figures for its own group.
#
# The groups are very uneven -- rq4_frontier is 64 arms and rq1 is 3 -- so expect task
# runtimes to differ by an order of magnitude. --time below is sized for the largest.
#
#SBATCH --job-name=pmabm-scen
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/scen-%A_%a-%x.out
#SBATCH --error=slurm/logs/scen-%A_%a-%x.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- environment ------------------------------------------------------------------------
# module load python/3.11
# export PATH="$HOME/.local/bin:$PATH"

# `uv run` syncs .venv on entry. A job script is not an interactive shell, so nothing in
# ~/.bashrc reaches here and these have to be set explicitly.
#
# UV_LINK_MODE=copy: the cache and .venv are on different filesystems whenever the cache is
# pointed at scratch, so hardlinking fails and uv falls back to copying with a warning on
# every task. Saying so up front is quieter and no slower.
#
# Run `uv sync` once on the login node before submitting. The compute nodes do not have the
# login node's /opt/conda, so a .venv built there is discarded and rebuilt on first use --
# and with every array task starting at once, that is one rebuild per task against one
# directory.
export UV_LINK_MODE=copy

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- which group is this task? ----------------------------------------------------------
# Order matches `scenario_gen.py --dry-run`. Keep the array range in the header in step with
# the length of this list: 13 groups is --array=0-12.
#
# Not named GROUPS. That is a bash special variable holding the current user's group IDs, and
# an assignment to it is silently discarded -- which left index 0 resolving to a numeric gid
# and every other index unbound, killing the task under `set -u`.
SCENARIO_GROUPS=(
    ladder
    rq1 rq2 rq3 rq4 rq5 rq6
    rq3_surface rq4_frontier
    horizon structural accounting checks
)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#SCENARIO_GROUPS[@]}" ]; then
    echo "task index $IDX is past the end of SCENARIO_GROUPS (${#SCENARIO_GROUPS[@]} groups):"          "the --array range and the list above have drifted apart" >&2
    exit 1
fi
GROUP="${SCENARIO_GROUPS[$IDX]}"

# Cores per task are a free choice now that the queue is flat -- 64 is simply a reasonable
# slice for one group. Parallelism across groups comes from the array.
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# One directory per group under a single run root, so a resubmitted group overwrites only
# itself and the whole suite still lands in one place.
RUN_ROOT="Results/scenarios/array-${SLURM_ARRAY_JOB_ID:-local}"
OUTDIR="${RUN_ROOT}/${GROUP}"

echo "host        : $(hostname)"
echo "array task  : ${SLURM_ARRAY_JOB_ID:-none}_${IDX}"
echo "group       : ${GROUP}"
echo "cores       : ${SLURM_CPUS_PER_TASK:-unset}"
echo "workers     : ${WORKERS}"
echo "outdir      : ${OUTDIR}"
echo "started     : $(date --iso-8601=seconds)"
echo

# --- run --------------------------------------------------------------------------------
uv run python src/scenarios/scenario_gen.py \
    --only "$GROUP" \
    --seeds 64 \
    --workers "$WORKERS" \
    --outdir "$OUTDIR" \
    --no-timestamp

echo
echo "finished    : $(date --iso-8601=seconds)"
