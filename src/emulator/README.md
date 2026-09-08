# Emulator

A neural-network surrogate of the ABM: a function that takes a parameter vector and returns, in
milliseconds, what the model would have produced in ~37 seconds of CPU.

It exists because several of the questions the paper wants to ask are not affordable against the
ABM itself. A Sobol' decomposition at publication grade is ~51,000 runs. History matching — "which
parameter settings *could* have produced England?" — needs 10⁵–10⁶. An interactive response
surface needs an answer while someone is still looking at the screen. All three become routine
once a surrogate exists, and none of them is a shortcut around the model: the emulator is trained
on the ABM's own output and is only ever as good as the corpus behind it, which is why every
number it reports comes with the two uncertainties described below.

## The pipeline

```bash
# 1. Generate the training corpus. This is the expensive step, ~8 h at the shipped design.
uv run python src/emulator/emulator_gen.py --dry-run     # design size and cost estimate only
uv run python src/emulator/emulator_gen.py

# 2. Fit the ensemble. Minutes on CPU.
uv run python src/emulator/emulator_train.py --run-dir Results/emulator/<timestamp>

# 3. Look at whether it worked before using it.
uv run python src/emulator/emulator_plot.py --model-dir Results/emulator/<timestamp>/emulator

# 4. Use it.
uv run python src/emulator/emulator_apply.py sobol --model-dir <…>/emulator --n-base 16384 --second-order
uv run python src/emulator/emulator_apply.py sobol --model-dir <…>/emulator --series conversion_share
uv run python src/emulator/emulator_apply.py sweep --model-dir <…>/emulator --over alpha_1 alpha_2
uv run python src/emulator/emulator_apply.py calibrate --model-dir <…>/emulator --targets constants/targets_example.yaml
```

Install the extra first — torch is not part of the base environment:

```bash
uv sync --extra emulator
```

A pilot that exercises the whole pipeline in a few minutes:

```bash
uv run python src/emulator/emulator_gen.py --n-points 256 --replicates 2 --steps 60
uv run python src/emulator/emulator_train.py --run-dir Results/emulator/<timestamp> --epochs 100 --ensemble 2
```

Do not read a pilot's accuracy figures as the emulator's accuracy. 256 points in 29 input
dimensions is far below what the response surface needs; the pilot tests the plumbing.

| Path | Purpose |
| --- | --- |
| `constants/constants.yaml` | the baseline the design is taken around, and run control |
| `constants/design.yaml` | which parameters and switches vary, and what is recorded |
| `constants/training.yaml` | architecture, fitting, and what gets reported |
| `constants/targets_example.yaml` | placeholder history-matching targets, to be replaced |
| `emulator_gen.py` | design → ABM runs → corpus (`X.csv`, `runs.csv`, `trajectories.npz`) |
| `emulator_data.py` | loading, encoding, trajectory compression, splits, the noise floor |
| `emulator_net.py` | the network, and the fitted object (`Emulator.load` / `.predict`) |
| `emulator_train.py` | fits the ensemble and scores it |
| `emulator_apply.py` | `sobol`, `sweep`, `predict`, `calibrate` |
| `emulator_plot.py` | the diagnostic figures |

## What it predicts

**Scalars** — the 21 summary outputs listed in `design.yaml`, which are the same ones
`src/sensitivity` decomposes, so the two can be put side by side.

**Trajectories** — 13 per-period series (conversion share, the tenure shares, the Gini, output,
wage, both rents, enclosure, the landlord's disposition, population), each compressed to a PCA
basis of ~10 coefficients and predicted as coefficients. This is the primary target and the
reason the emulator is worth building rather than a scalar regression: Brenner's and Wood's claims
are about *when* and *in what order* things happen. A final-state emulator cannot be asked whether
conversion led or followed concentration, and cannot be matched against Clark's or Kerridge's
series at all.

Each series is transformed before compression (logit for shares, log1p for positive quantities)
so that a reconstructed share is inside [0, 1] by construction rather than by clipping.

**Not predicted: the spatial field.** `awareness_radius` and `zeta` are in the design, so the
lattice is rebuilt for every point and parcel indices are not comparable across the corpus. A
per-parcel conversion-time emulator is possible and would give predicted conversion *maps*, but it
needs its own corpus with those two parameters held fixed.

## The two uncertainties

Every prediction carries both, and they are used differently.

*Aleatoric* — the sd the network predicts directly. This is the ABM's own seed-to-seed spread at
those parameters. It is irreducible: more ABM runs will not shrink it. It is what makes the
emulator usable as a likelihood in `calibrate`.

*Epistemic* — the disagreement between ensemble members. This is the emulator's ignorance, and it
**is** reducible by running the ABM at more points. It is what says whether a prediction in a
thinly covered corner of the design should be believed, and what an active-learning loop would
sample against.

## Reading the accuracy figures

Every accuracy number is reported twice: raw R², and R² as a share of a *ceiling*.

The ceiling comes from the replicate seeds. At a repeated design point the ABM's outputs still
differ, and that within-point variance is a share of the total that no function of the parameters
can explain. `1 − that share` is the largest R² anything could reach on this corpus.

This matters more than it sounds. `final_farm_gini` is nearly deterministic given the parameters,
so its ceiling is close to 1 and an R² of 0.8 there means the emulator has real work left.
`share_leasehold_t25` is mostly seed noise at that point in the run, with a ceiling near 0.16, and
an R² of 0.15 there is essentially perfect. Without the ceiling the two look like the same result.
`emu_accuracy.png` draws the pair; `r2_of_ceiling` in `metrics_scalars.csv` is the ratio.

The trajectory table adds a second ceiling, `r2_basis_ceiling`: how well the PCA basis reproduces
the held-out curves from their own projection. If the emulator has reached it, the limit is the
basis — raise `pca.max_components`, not the network width.

## Things that will bite

- **Splits are by design point, never by run.** All replicates of a point are in the same fold.
  Split by run and the test error measures the ABM's seed noise instead of the emulator's error,
  which flatters it substantially.
- **`design.yaml` must stay in step with `src/sensitivity/constants/bounds.yaml`.** They define the
  same space on purpose. Move a bound in one and the two analyses stop being comparable; move a
  bound at all and the emulator has to be retrained, because it knows nothing outside its corpus
  and will extrapolate confidently rather than refuse.
- **The emulator is not the model.** It reproduces the ABM's behaviour over the sampled space; it
  does not inherit the ABM's mechanisms. Any claim about *why* something happens still has to be
  made against the ABM. What the emulator supports is claims about *where in parameter space* and
  *how sensitively* — which is a different and complementary kind of statement.
- **History matching rules regions out; it never rules one in.** `calibrate` returns the
  not-implausible set. If nothing survives, look at the `discrepancy:` term before concluding
  anything about the theory: a zero there asserts the ABM is a true model of sixteenth-century
  England and will rule out everything.
