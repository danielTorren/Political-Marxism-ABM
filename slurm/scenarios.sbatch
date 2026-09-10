#!/bin/bash
# The full scenario suite: 166 arms x 64 seeds = 10,624 model runs.
#
#   mkdir -p slurm/logs && sbatch slurm/scenarios.sbatch
#
# One node only. The runner parallelises with Python's multiprocessing, which cannot span
# nodes, so --nodes=1 with all the cores as --cpus-per-task is the right shape; --ntasks
# greater than 1 would start N independent copies of the whole suite instead.
#
# Writing ~10.5 GB of parquet and drawing the figures at the end is single-threaded and does
# not shrink with more cores, so budget for a tail beyond the ~11 CPU-hours of model time.
#
#SBATCH --job-name=pmabm-scenarios
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/scenarios-%j.out
#SBATCH --error=slurm/logs/scenarios-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- environment ------------------------------------------------------------------------
# Adjust these two lines for your cluster; everything below is portable.
# module load python/3.11
# export PATH="$HOME/.local/bin:$PATH"     # where uv lives, if not already on PATH

# Each worker process would otherwise start its own BLAS thread pool sized to the whole
# node, oversubscribing it by a factor of the worker count. The model is millions of small
# array operations rather than large matrix products, so threaded BLAS buys nothing here and
# the contention is pure loss.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- worker count -----------------------------------------------------------------------
# Passed explicitly, never left to the runner's default. Its fallback is os.cpu_count(),
# which under SLURM reports the machine's core count rather than this allocation's, so on a
# shared node the default would oversubscribe badly.
#
# The runner's work queue is flat across every (arm, seed) pair in the suite, so all 128 are
# used regardless of the seed count, and an arm finishing does not drain the pool.
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}"
echo "cores       : ${SLURM_CPUS_PER_TASK:-unset}"
echo "workers     : ${WORKERS}"
echo "started     : $(date --iso-8601=seconds)"
echo

# --- run --------------------------------------------------------------------------------
# --outdir defaults to Results/scenarios. Point it at scratch if your home quota is small:
#   --outdir "$SCRATCH/pmabm/scenarios"
uv run python src/scenarios/scenario_gen.py \
    --seeds 64 \
    --workers "$WORKERS" \
    --tag "slurm${SLURM_JOB_ID:-}"

echo
echo "finished    : $(date --iso-8601=seconds)"
