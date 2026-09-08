# Political Marxism ABM

An agent-based model of Brenner's and Wood's theory of the transition from feudalism to
agrarian capitalism in early modern England. The model implements the formal specification in
[`paper/main.tex`](paper/main.tex) (§ Model Design); every mechanism below cites the subsection
it comes from.

## Quick start

```bash
uv sync
uv run pmabm build-geography     # one-off: fetch real England outline + ALC land quality
uv run pmabm run                 # single baseline run (England regime)
uv run pmabm experiments         # full RQ1-RQ3 experiment suite + figures
```

Outputs are written to `Results/`.

### Diagnostic runs

Two standalone scripts exist for studying the model's dynamics rather than answering the
research questions. Each reads its own YAML, so parameters are edited in one visible place
rather than passed as flags:

```bash
uv run python src/single_run/single_run_gen.py     # one run  -> Results/single_run/
uv run python src/multi_seed/multi_seed_gen.py     # N seeds  -> Results/multi_seed/
uv run python src/multi_seed/multi_seed_gen.py --seeds 20
```

| Path | Purpose |
| --- | --- |
| `src/single_run/constants/constants.yaml` | parameters for one run |
| `src/single_run/single_run_gen.py` | runs the model, writes tables, calls the plotter |
| `src/single_run/single_run_plot.py` | single-run figures (plain lines) |
| `src/multi_seed/constants/constants.yaml` | parameters, seed count, regime comparison |
| `src/multi_seed/multi_seed_gen.py` | runs every seed and both regimes, aggregates |
| `src/multi_seed/multi_seed_plot.py` | mean + 95% CI figures |

At the defaults — 37 historic counties, 10 lords each, ~20 tenants per lord, 7,400 parcels — a
single 200-period run takes about **14 s**, and the full 12-run multi-seed suite (both regimes,
6 seeds) about **50 s** on 7 workers. The per-agent hot loops are vectorised via `bincount`
over the parcel axis, which brought the scaling from `parcels^1.74` down to `parcels^1.29`; if
you push much past `L=200`, note that the seven `(n_steps × n_parcels)` panel arrays start to
dominate memory.

Anything absent from a YAML falls back to the default in `pmabm/config.py`; an unrecognised
key is a hard error rather than a silent no-op. The nine **dynamics dashboards** are defined
once, in `pmabm/diagnostics.py`, and shared by both scripts — the only difference is whether a
series is drawn as one line or as a mean with a 95% confidence band, so the two sets are
comparable panel for panel. Every panel carries a note marking it `ASSUMPTION-LED`,
`EMERGENT`, or `MIXED`, and `figure_index.csv` in each output directory tabulates them.

### Sensitivity analysis

The parameters in Table 1 are placeholders, not estimates, so the standing question about any
result is which of them it rests on. `src/sensitivity/` answers that with a Sobol' variance
decomposition ([SALib](https://salib.readthedocs.io)): first-order (`S1`) and total-order (`ST`)
indices, no second order — `ST - S1` already reports how much of a parameter's influence runs
through interactions, at half the runs.

```bash
uv run python src/sensitivity/sensitivity_gen.py --dry-run    # design size, no model runs
uv run python src/sensitivity/sensitivity_gen.py              # the sweep, then the analysis
uv run python src/sensitivity/sensitivity_gen.py --n-base 256 # publication grade
uv run python src/sensitivity/sensitivity_analysis.py         # re-analyse the newest sweep
```

| Path | Purpose |
| --- | --- |
| `src/sensitivity/constants/constants.yaml` | the baseline the decomposition is taken *around*, plus sweep controls |
| `src/sensitivity/constants/bounds.yaml` | which parameters vary, over what range, `n_base`, replicates, which outputs |
| `src/sensitivity/sensitivity_gen.py` | draws the design, runs it in parallel, writes `samples.csv` / `Y.csv` |
| `src/sensitivity/sensitivity_analysis.py` | Sobol' indices, convergence and noise diagnostics, figures |

Cost is `n_base × (parameters + 2) × replicates` runs. At the shipped defaults — 12 parameters,
`n_base=64`, 2 replicates — that is **1,792 runs, about 2.5 hours** at a measured ~13 runs/min on
13 workers; `n_base=256` is a 10-hour overnight job, and `--n-base 8 --replicates 1 --steps 60` is
the pilot. Two design choices are load-bearing:

- **Replicates are averaged before the decomposition.** Sobol' attributes output variance to input
  variance, and a stochastic model supplies variance that belongs to no parameter; with one seed
  per point that noise is silently divided among the indices. `sobol_noise.csv` and the noise
  figure report the residual share — above ~0.2 the fix is more replicates, not a larger `n_base`.
- **Only continuous parameters are sampled.** A variance decomposition over `population_rule` or
  `channel_ideology` would be meaningless, so the boolean and string switches are a hard error in
  `bounds.yaml`; they are compared as arms in `src/scenarios/` instead.

Read the convergence figure before believing any ranking, and the response-curve figure before
interpreting a large index: Sobol' gives magnitude without direction, and those panels supply the
sign and shape of the effect.

### Emulator (neural-network surrogate)

`src/emulator/` trains a network on the ABM's own output so that the questions the model is too
slow to be asked directly become routine: a Sobol' decomposition at 10⁶ evaluations instead of
51,000 model runs, history matching against the historical series, and a response surface that
answers while you are still looking at it. Full documentation in
[`src/emulator/README.md`](src/emulator/README.md).

```bash
uv sync --extra emulator                                       # torch is not in the base install
uv run python src/emulator/emulator_gen.py --dry-run           # design size and cost, no runs
uv run python src/emulator/emulator_gen.py                     # the corpus: ~8 h
uv run python src/emulator/emulator_train.py --run-dir Results/emulator/<timestamp>
uv run python src/emulator/emulator_plot.py  --model-dir Results/emulator/<timestamp>/emulator
uv run python src/emulator/emulator_apply.py sobol --model-dir <…>/emulator --second-order
uv run python src/emulator/emulator_apply.py calibrate --model-dir <…>/emulator --targets targets.yaml
```

It predicts the same 21 scalars the sensitivity suite decomposes **and** 13 full per-period
series, the latter compressed to a PCA basis and predicted as coefficients — because the theory's
claims are about when and in what order things happen, and a final-state surrogate cannot be
matched against a rent or output series at all. Three design choices are load-bearing:

- **It predicts a distribution, not a number.** Trained on individual runs with a heteroscedastic
  loss, so each output comes with the ABM's own seed spread (irreducible) and, from the ensemble,
  the emulator's own ignorance (reducible by running the ABM at more points). The first is what
  makes `calibrate` a likelihood; the second is what says whether a thinly-sampled corner should
  be believed.
- **Accuracy is reported against a ceiling, not against 1.0.** Replicate seeds at each design point
  give the share of each output's variance that is seed noise, and `1 − that` is the most any
  emulator could explain. An R² of 0.15 on `share_leasehold_t25`, whose ceiling is 0.16, is a
  near-perfect fit; the same number on `final_farm_gini`, whose ceiling is ~1, is a poor one.
- **The mechanism switches are inputs.** Unlike the Sobol' sweep, which refuses categoricals
  because a variance index over one is meaningless, the surrogate simply learns a response surface
  per arm. So the RQ2 channel ablations can be asked *anywhere* in parameter space rather than only
  at the baseline — which is the one thing the scenario suite cannot afford.

The emulator reproduces the ABM's behaviour; it does not inherit its mechanisms. Claims about
*why* something happens are still made against the model. What the surrogate adds is *where in
parameter space*, and *how sensitively*.

## What the model is for

It tests whether the class-conflict mechanisms Brenner and Wood identify are *jointly
sufficient* to generate the macro outcomes their theory is invoked to explain, starting from a
small, spatially localised seed rather than assuming market competition from tick one. The
three headline research questions (paper § Research Questions) are:

- **RQ1** — is "improvement" a *consequence* of market exposure rather than a precondition?
- **RQ2** — does conversion spread outward from a localised shock, and which channel drives it?
- **RQ3** — does the state–landlord alliance parameter θ reproduce Brenner's England/France divergence?

## Layout

| Path | Purpose |
| --- | --- |
| `src/pmabm/config.py` | All parameters (paper Table 1) and scenario regimes |
| `src/pmabm/geography.py` | England lattice, estates, neighbour relations (§ Spatial structure) |
| `src/pmabm/build_geography.py` | One-off fetch of ONS boundary + Natural England ALC data |
| `src/pmabm/model.py` | Agent state and the 10-step schedule (§ Model schedule) |
| `src/pmabm/metrics.py` | Recording and derived metrics (§ Emergent outputs) |
| `src/pmabm/experiments.py` | RQ1–RQ3 experiment definitions |
| `src/pmabm/plots.py` | Figures |
| `src/pmabm/cli.py` | Command line entry point |
| `src/sensitivity/` | Sobol' variance decomposition of the parameters |
| `src/emulator/` | Neural-network surrogate: fast sensitivity, sweeps, history matching |
| `tests/test_model.py` | Invariants: accounting, state consistency, reproducibility |

The loose modules at the top of `src/` (`Economy.py`, `Landlords.py`, `Tenants.py`, …) are the
earlier prototype. Nothing in `pmabm` imports them; they are left in place rather than deleted
so the history stays visible, and can be removed whenever you like.

## Data sources

- England boundary: ONS Open Geography Portal, Open Government Licence v3.0.
- Agricultural Land Classification (provisional): Natural England, Open Government Licence v3.0.

Both are fetched by `pmabm build-geography` and cached to `data/geography/`. ALC grades are
aggregated *server-side* per sampling cell, so no shapefile download or GIS dependency is
needed. Fertility is mapped Grade 1 → highest carrying capacity, Grade 5 → lowest; the
ecological "seed" is then simply the worst-quality region the data happens to produce, which
lands in the northern uplands.

Note that ALC is modern data. Treating it as a proxy for early-modern land quality is a
stated assumption: it measures a soil-and-climate endowment that changes on a geological
rather than a century timescale, and it is distinct from the fast-moving, endogenous
depletion the model applies on top of it.

## Specification gaps found while implementing

Building the model surfaced places where `paper/main.tex` is silent, or where its equations do
not do what the surrounding prose claims. They are listed here because several are substantive
enough to need a decision before the results mean anything. In the code they are marked
`SPEC NOTE`.

**Fixed in code (the model is otherwise ill-posed):**

1. **Hired labour is free.** The wealth equation is
   `w(t+1) = w + y − ρ − c`, with no wage bill, yet hired labour enters the production
   function. Capital then raises output, output raises wealth, wealth raises capital, and
   nothing ever prices the extra hands — accumulation diverges to overflow within ~50 periods.
   The code charges `ω(t)·ℓ_hired` by default; `--literal-wage-bill` restores the paper's
   literal version so the divergence can be seen.
2. **The wage tâtonnement is unbounded.** `(D − |L|)/|L|` has no bound as the landless pool
   empties, so the wage can move by orders of magnitude in a period. The ratio is clipped;
   see `wage_adjust_cap`.
3. **Landlord wealth has no update equation.** `W_i(t)` is named as state and used in
   `Ī_i(t)`, but never given a law of motion. Implemented as `W(t+1) = W(t) − Δ(t)`.

**Resolved by decision (each is a deviation from the paper, and each is switchable):**

4. **Conversion is now a ratchet.** The paper's re-letting rule gives an unconverted crisis
   vacancy "a new Customary tenant", which meant a *leasehold* parcel whose tenant failed
   reverted to customary tenure: 1080 of 1083 converted parcels reverted at least once, and
   the transition could never accumulate past ~7%. Tenure is now a property of the parcel
   (`Model.parcel_tenure`) and only moves forward — once customary right is extinguished it
   does not return, and "custom attaches to the land" is scoped to parcels that are still
   customary. This is the single change that lets the transition complete.
5. **Sitting tenants can be converted in place.** The paper says a fine escalation is "a
   special case of the same conversion decision" but only ever reaches that decision at a
   vacancy, so almost no tenant was present either side of their own conversion and RQ1 was
   untestable. `inplace_conversion_rate` restores the entry-fine path: the landlord reopens
   terms on a sitting tenant, who pays the fine and becomes a leaseholder, or is dispossessed
   if they cannot. Roughly a fifth of conversions now retain the tenant.
6. **The improvement driver is competitive selection, not the rent ratio.** Under the paper's
   `M = ρ/y`, a customary tenant with a fixed nominal rent keeps every extra bushel and so has
   the *stronger* incentive to improve — inverting Brenner. The driver is now the estate's
   competitive benchmark relative to own output, applied only to tenures whose survival
   depends on meeting it. Alongside it, `customary_fine_rate` implements Brenner's "squeeze"
   directly: the lord takes a share of customary income above subsistence, confiscating the
   funds improvement would need. `iota_decay` lets a disposition fade when nothing sustains
   it, so it tracks current conditions rather than being an acquired trait.
7. **Ideology diffuses from conversions.** `β₂` now multiplies neighbours' realised
   *conversion share* rather than their ideology, and `β₁` is cut to 0.002. Previously neither
   term referenced any actual break from custom, so ideology climbed to ~1 on its own schedule
   and ablating it in RQ2 tested nothing.
8. **The household is the demographic unit** (`population_rule="household_size"`). A household
   carries a size `n_j`: it supplies `n_j · ℓ̄` labour, owes `n_j ·` per-head subsistence, and
   grows or shrinks through birth and death hazards on *its own* per-head surplus. When it has
   more hands than its land can profitably use — the marginal product of family labour net of
   subsistence falling below the wage net of landless subsistence — it sheds a member into the
   labour market. That makes proletarianisation a demographic mechanism (the non-inheriting
   member driven to wage labour) rather than only a consequence of eviction, and it makes the
   land's productivity govern how many people a holding can hold, parcel by parcel, with no
   aggregate carrying capacity anywhere. `impartible_inheritance=True` adds the strong form, in
   which every member but the heir is expelled at conveyance.

   Fertility and mortality are properties of the individual household, so the aggregate rate is
   whatever the class composition makes it: the landholding household's surplus comes from its
   land, its capital and its rent, and the landless household's from the wage. That is the
   opposite of an aggregate food-population law — the same land supports a growing or shrinking
   population depending on how the surplus is distributed — which is Brenner's claim rather than
   Postan's. `eta_0` is correspondingly read as a *conveyance* hazard and is population-neutral;
   the heir inherits the household whole, so it does not double-count with `mortality_0`.

   **This replaced a closed population, and the reason is worth recording.** Under
   `population_rule="fixed"` nobody is created and exit to industry is absorbing, so population
   can only fall: `N(t) = N(0) − exits(t)` exactly. Over 200 periods a 64-seed run lost 73% of
   its people, and by t≈120 an average of 1,347 of 7,400 parcels sat vacant because no living
   agent could take them up. Worse, the departures arrived in cohorts — every landless household
   faced the same global wage and the same integer deficit counter, so 22% of all exits happened
   within five periods, after which the drained labour pool sent the wage to 2.5× subsistence and
   left it there. The second half of every run therefore did its consolidation in a labour-scarce,
   high-wage regime: the inverse of the historical setting. Across seeds, the final leasehold
   share correlated **+0.97** with how far the population had fallen. Under the household rule
   vacant parcels stay at 0.0%, the five largest exit periods account for 5.7% of departures, and
   that correlation falls to −0.46. `fixed` is retained as a robustness arm, because "does the
   transition still occur with population held fixed?" is a real test of Brenner's claim — but it
   cannot be the default. `vital_rates` applies the same hazards to single-body households, which
   isolates what the hazards do from what the household unit does; `household` is the superseded
   threshold rule, whose defect was that only landholders could reproduce, so proletarianisation
   mechanically sterilised the population.
9. **Departure for industry is a hazard on the depth of a deficit** (`exit_rule="hazard"`), and
   each household draws its own `landless_consumption`. With one shared requirement and a global
   wage, every landless household had an identical net income and crossed zero on the same tick,
   which is what produced the exit avalanches above. Set `exit_rule="counter"` and
   `landless_consumption_spread=0` to recover the original deterministic rule.
10. **The urban sector is a place, not a sink** (`urban_return`). An urban household keeps its
    size and wealth, earns `urban_wage` and pays `urban_consumption` per member, reproduces and
    dies on the same hazards as everyone else, and returns to the landless pool at a rate
    proportional to how much better the rural wage now is — the partition comparison read
    backwards. Scaled by the *size* of the advantage, not its sign: returning households enlarge
    the labour pool and so depress the wage that drew them, and on a sign test that feedback is a
    churn loop rather than a response to conditions.

    `urban_wage` defaults to exactly `urban_consumption`, so the towns are demographically
    stationary and every movement in the rural/urban split is migration answering to rural
    conditions. This is not a free parameter to nudge: at 0.9 against a consumption of 1.0 the
    urban sector loses ~1.9% of its people per period, which drained 74% of the **total**
    population over 200 steps and put a third of parcels back in the vacancy queue. Departing
    from equality asserts something about urban demography the model has no evidence for.

    The point of the channel is neutrality. With exit absorbing, the countryside can only lose
    people, whatever its own vital rates: rural births exceeded rural deaths by 11,256 over 200
    periods while 20,369 persons left and none came back, so a countryside reproducing perfectly
    well still ended at 58% of where it started — an artifact of the valve, not a result.
11. **Population is reported as a composition, not a headcount.** `share_persons_{customary,
    leasehold,freehold,landless,urban}` sum to one over all living persons, against a
    `total_population` that is itself an outcome; see the `population_composition` figure. A
    falling rural headcount cannot distinguish a countryside losing people to towns from one
    failing to reproduce, and the theory claims the first and says nothing about the second. Two
    accounting identities are asserted every period: total persons change only by births and
    deaths, and rural persons only by rural births, rural deaths and net migration.
12. **Rent scales with holding size** (`ρ_j = r̂_i · |P_j|`), so engrossing five parcels is no
    longer rent-free.
13. **Labour demand responds to the wage.** `labour_demand_rule="marginal_product"` hires until
    the marginal product of labour equals the wage. The paper's `δk − ℓ̄` never references the
    wage, so nothing responded to the price of labour and the wage ran to its bound. Demand is
    net of the household's *own* labour `n_j · ℓ̄`, so a large family hires less; since only
    hired labour is charged at ω, family labour is the cheaper of the two and the family farm
    has a genuine cost advantage.
14. **Capital depreciates** (`capital_depreciation`). The paper's capital equation only ever
    adds, so improvements were permanent and free to maintain. Depreciation makes improvement
    something that must be *sustained* out of current surplus — a standing compulsion rather
    than a one-off investment.
15. **The consumption requirement scales with the estate.** Drawing `C_i` from a fixed range
    independent of estate size put every landlord in heavy deficit at t=0 — a 28-parcel estate
    yields ~28 against a requirement of ~100 — so conversion was already likely everywhere in
    period 1 and there was no "custom is sustainable, then inflation erodes it" phase for a
    localised seed to break. It is now a multiple of the estate's own initial rent roll, and
    fiscal pressure enters the logit as a *relative* shortfall. This also removes an unnoticed
    artifact: with absolute Δ, estate size was the main determinant of who converted first.

Coefficients `α₀, α₁, α₂, λ₀, ψ₁` were rescaled to match. `α₀` and `α₁` are set so that fully
eroded customary income is **necessary but not sufficient** — a landlord with no converted
neighbours still converts only rarely — which is the paper's own claim about why the pattern
should spread rather than happen at once, and which does not hold under its original
coefficients. `λ₀` is set so conversion decisively beats the Freehold pathway in the England
regime and loses to it in France; at the paper's value England produced freehold smallholding,
the outcome the theory assigns to France.

Smaller choices the paper left open — how the turnover hazard applies to a multi-parcel
holding, whether an heir inherits the improving disposition, which landless agent takes an
open vacancy, how `r̂_i` is bootstrapped at a landlord's first conversion — are documented at
their point of use in `src/pmabm/model.py`.

16. **Tenant mobility withdrawn; competitive allocation put in its place.** The mobility
    channel never fired — across every run the migration count was exactly **0**, and the
    ablation returned bit-identical results with it on and off, because migration was gated on
    a vacancy being *open*, which only happens when no landless agent can afford the fine.
    Checking it against the sources showed the problem was not the gate but the mechanism:

    - Brenner puts the fight over mobility *before* this period — "by the mid-fifteenth
      century, through flight and resistance, [the peasantry was able] to break definitively
      feudal controls over its mobility and to win full freedom" (p.61). It is a precondition,
      not a variable.
    - For the sixteenth century the pressure runs the *other way*: "competition for land
      induces the peasantry to accept a serious degradation of their personal/tenurial status
      in order to hold on to their land" (p.38), and lords "could always get replacements,
      quite often indeed on better terms" (p.44).
    - Lords bidding to retain mobile tenants is *Postan's* mechanism, which Brenner quotes
      only to reject: "the most effective way of retaining tenants was to lower rents and
      release servile obligations" (p.39 n.17).

    What Wood does describe is allocation: "landlords… would increasingly seek tenants who
    could produce competitively… success would breed success, and competitive farmers would
    have increasing access to even more land, while others lost access altogether… In a system
    of 'competitive rents', in which landlords, wherever possible, would effectively **lease
    land to the highest bidder, at whatever rent the market would bear**" (p.101). So a vacated
    market-exposed parcel now goes to the highest bidder — a sitting neighbour bidding on
    revealed productivity, or a landless entrant bidding what an unimproved holding would
    yield — and the winning bid becomes that parcel's rent. Sitting neighbours enter the
    auction only where the lord is willing to consolidate, so engrossment and the auction are
    the same event from the two sides. RQ2 accordingly has **two** transmission channels, with
    allocation reported against concentration rather than ablated as a spread channel.

### What the baseline run now shows

| | result |
| --- | --- |
| RQ1 | 11.6% of conversions retain the sitting tenant — enough for the event study to measure a disposition change. ι jumps ≈0.29 → 0.52 at conversion and keeps rising; customary ι stays ≈0 |
| RQ2 | contagion is now positive under **every** configuration. Observation is the strongest single channel (slope +1.28), both together +0.92, ideology alone +0.60 — but **"none" still gives +0.70** |
| RQ3 | England 62.7% leasehold / 28.2% freehold / 10.7% customary; France 1.8% / 42.7% / 50.0% |

RQ3 is a clean positive: flipping θ alone reproduces Brenner's divergence, with France's
customary sector still half-intact after 200 periods where England's is nearly gone.

**RQ2 is positive once the geography stops moving between runs.** With the county hierarchy
in place the England distance-lag slope is **+0.16 ± 0.10** over 6 seeds — above zero, where
the earlier grid-based geography gave **−0.12 ± 0.84**, a null. (The slope is per lattice cell,
so it shrinks as `L` rises even when the underlying pattern is unchanged; at `L=100` the same
setup gives +0.32 ± 0.07.)

Nothing about the mechanism changed; the earlier null was a measurement problem. Two causes,
both now fixed:

- The ecological seed origin used to be recomputed each run as the lowest-fertility *sampling
  cell*, and per-parcel jitter occasionally flipped the winner, so the distance regressor was
  measured from a different place in different runs. It is now pinned to the lowest-grade
  **county** (Cumberland), which is a property of the ALC data and identical in every seed.
- Estate layout used to be redrawn nationally each seed. Seeding lords within fixed county
  boundaries removes most of that structural variation.

The effect on precision is large across the board: the confidence interval on final leasehold
share tightens from ±0.074 to ±0.011. The lesson generalises — when a model's spatial scaffold
is itself random, cross-seed intervals measure the scaffold rather than the mechanism.

17. **Improvement ideology is now material, not contagious** (`ideology_rule: material`). The
    landlord's improving disposition was a Bass diffusion — an idea arriving from a neighbour.
    It is now simply the share of his income that already comes from market-determined rent,
    `R^L / (R^C + R^L + fines)`. A lord who lives by competitive rents is by that fact
    committed to competitive production, so his resistance to breaking custom falls. Nothing
    is transmitted and nothing is imposed; the disposition is a read-out of how far the estate
    has been pulled into producing for value rather than subsistence. `ideology_rule:
    contagion` restores the old behaviour, so the two can be compared — including across the
    England/France regimes, which is the interesting version of that test.

    Note the consequence for RQ2: with the material rule, ideology is a *within-estate*
    feedback rather than a between-estate channel, so observation becomes the only genuine
    transmission mechanism left.
18. **Dispossession creates its own market** (`urban_demand: true`). Households in the urban
    sector demand food, and the produce price moves with excess demand against marketed output.
    Only market-exposed holdings sell; customary output is eaten at home, so the two tenures
    differ in kind and the gap widens as the urban sector grows. This closes Wood's loop
    (p.103) — agrarian productivity throws people off the land, and that propertyless mass is
    the domestic market the surviving farms sell into. Note that the urban count is now a
    *stock* rather than a cumulative total, since item 10 lets households leave it again.

### Calibration status

The parameters remain placeholders; only the qualitative relationships above have been set
deliberately. Two things were tuned to a *target*, and should be read as design decisions
rather than findings: the England/France ordering (that is what defines the two regimes), and
the relative magnitude of the pressure and neighbour terms (the paper requires three co-equal
spread channels). Nothing was tuned to produce contagion, which is why RQ2's result stands as
a genuine outcome rather than a fitted one.
