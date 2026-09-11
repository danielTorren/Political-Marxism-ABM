# SLURM job scripts

```bash
mkdir -p slurm/logs          # once: SBATCH cannot create its own log directory
sbatch slurm/scenarios.sbatch          # the full suite on one 128-core node  (preferred)
sbatch slurm/scenarios-array.sbatch    # the full suite as 13 per-group jobs
```

Both run **166 arms × 64 seeds = 10,624 model runs**, about **11 CPU-hours** of model time at
~3.7 s per 200-period run.

## The work queue is flat

`scenario_gen.py` builds one list of every `(arm, seed)` pair in the suite and hands it to a
single process pool, so a worker that finishes a run takes the next one from anywhere. Two
consequences worth knowing:

- **Cores are not capped by the seed count.** Ask for 128 and you get 128, whatever
  `--seeds` is set to.
- **Uneven arms cost nothing.** `rq4_frontier` is 64 arms and `rq1` is 3; with arms run in
  sequence the pool drained to idle at every arm boundary, and now it does not.

`chunksize` stays at 1 deliberately. A run costs seconds, so per-task dispatch overhead is
noise, while any larger chunk hands one worker a fixed block of work and reintroduces exactly
the straggler tail the flat queue removes. Chunking is for millisecond tasks.

The lattice is a pure function of the six fields in `_GEOGRAPHY_FIELDS` — it is built with a
fixed `default_rng(0)`, not from the run seed — so each worker rebuilds whichever of the
seven distinct lattices a payload needs and keeps it in a local cache. That is what lets one
pool serve every arm: a pool holding one pre-built lattice could only serve arms using it.

## Which script

| | `scenarios.sbatch` | `scenarios-array.sbatch` |
|---|---|---|
| jobs | 1 | 13 (`--array=0-12`) |
| cores | 128 on one node | 64 per task, up to 832 at once |
| if it dies | lose all 10,624 runs | resubmit that one group |
| output | one directory | one per group |

Use the single job unless your queue makes a 128-core node slow to schedule, or you want
per-group restartability. Nothing is lost by splitting: every figure is built from one
group's results (`scenario_plot.plot_all` indexes by group name), so each array task writes a
complete, self-contained set of tables and figures.

## Sizing

**Memory.** The parent holds every arm's frames until the final plotting pass — roughly
0.86 MB per run, so ~9 GB at 10,624 runs, plus concat headroom. Workers add ~60–80 MB each
above the forked baseline (the panel arrays are ~58 MB per model). Hence `--mem=64G` for the
single job and `--mem=32G` per array task, where no task holds more than one group.

**Disk.** About **1 MB of parquet per run**, so the full suite at 64 seeds writes **~10.5 GB**.
That will overrun a modest home quota, so point `--outdir` at scratch if in doubt.

**Time.** Arms per group after sweep expansion, which is what sets the array's `--time`:

| group | arms | runs | | group | arms | runs |
|---|---:|---:|---|---|---:|---:|
| `rq4_frontier` | 64 | 4,096 | | `rq5` | 4 | 256 |
| `rq3_surface` | 42 | 2,688 | | `horizon` | 4 | 256 |
| `rq4` | 18 | 1,152 | | `checks` | 4 | 256 |
| `ladder` | 7 | 448 | | `rq1` | 3 | 192 |
| `rq2` | 5 | 320 | | `rq6` | 3 | 192 |
| `structural` | 5 | 320 | | `accounting` | 3 | 192 |
| `rq3` | 4 | 256 | | | | |

`horizon` also carries 400-period arms, so its runs cost about twice the rest.

Note that writing 10.5 GB of parquet and drawing the figures is single-threaded and does not
shrink with more cores. At production scale the model phase dominates, but on a short test
run the fixed tail is most of the wall time — do not read a small run's speedup as the
suite's.

## The array variable is not called `GROUPS`

`GROUPS` is a bash special variable holding the current user's group IDs, and an assignment to
it is silently discarded. The array script's list is therefore `SCENARIO_GROUPS`. With the old
name, task 0 resolved to a numeric gid and every other index was unbound, so `set -u` killed
each task the moment it started. The script now also fails loudly if the task index runs past
the end of the list, which is what happens when `--array` and the list drift apart.

## Two things the scripts do that matter

**Thread pinning.** `OMP_NUM_THREADS=1` and friends are set because each worker process would
otherwise start a BLAS thread pool sized to the whole node, oversubscribing it by the worker
count. The model is millions of small array operations rather than large matrix products, so
threaded BLAS buys nothing and the contention is pure loss.

**Explicit `--workers`.** Never left to the runner's default, which falls back to
`os.cpu_count()`. Under SLURM that reports the machine's cores, not the cgroup allocation, so
on a shared node the default would oversubscribe badly. Both scripts pass
`$SLURM_CPUS_PER_TASK`.

## Output

The single job writes `Results/scenarios/scenarios_<date>_<time>-slurm<jobid>/` with
`input_data/`, `output_data/` and `figures/`. Date first, so successive runs list in the order
they were made, and the suite name is carried in the directory itself — useful once it has been
copied off the cluster and sits beside output from another suite.

The array writes `Results/scenarios/array-<jobid>/<group>/` instead — one complete directory
per group, not stamped, because the job id already makes a submission unique and a resubmitted
group has to land beside its siblings rather than in a new directory of its own.

## Other suites

Same shape, different runner and cost. `--dry-run` reports the size of each before you commit.

```bash
uv run python src/multi_seed/multi_seed_gen.py                # 128 runs
uv run python src/sensitivity/sensitivity_gen.py              # 51,200 runs, ~15 h
uv run python src/emulator/emulator_gen.py                    # 6,144 runs, then training
```

Two prerequisites for all of them, both on the login node before you submit:

```bash
uv sync                        # build .venv with a uv-managed interpreter
uv run pmabm build-geography   # the runners only read this cache, they never build it
```

`build-geography` fetches from ONS and Natural England, so it needs outbound network — run it
on the login node, not inside a job.

`uv sync` matters more than it looks. `uv run` syncs on entry, so a job would do it anyway,
but the login node's interpreter may not exist on the compute nodes: a `.venv` pointing at
`/opt/conda/bin/python3` is discarded there and rebuilt from scratch, downloading a CPython
and every wheel. With a job array that is one rebuild per task, all against the same
directory. Syncing once up front leaves the tasks with nothing to do. Note also that
invoking `.venv/bin/python` directly skips the sync and can run against a stale environment.

`sensitivity_gen.py` and `emulator_gen.py` still parallelise over their own design rows rather
than a flattened queue, but their designs are thousands of rows deep, so they saturate a
128-core node as they stand. The emulator runner also checkpoints (`checkpoint_every` in its
constants), which matters at that length.
