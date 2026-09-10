# Model scenarios

`scenarios.yaml` defines every run the paper's Results section reports. Two kinds of thing live
in it, answering different questions.

**The ladder** (`ladder:`) builds the model up one mechanism at a time — tier 0 is inflation
eroding fixed customary rents against no conversion at all, tier 6 is the full model. Tiers are
cumulative, so the marginal effect of a mechanism is the difference between consecutive tiers. A
run with everything switched on cannot say which mechanism produced which dynamic; that is what
the ladder is for, and it is prior to every research question.

**The groups** (`groups:`) hold one entry per research question of `paper/main.tex`, each with
the arms that question needs. Arms *within* a group are compared to each other; groups are not
compared across.

## Running

```bash
uv run python src/scenarios/scenario_gen.py --dry-run      # list the arms, run nothing
uv run python src/scenarios/scenario_gen.py                # the whole suite
uv run python src/scenarios/scenario_gen.py --only ladder rq3 rq6
uv run python src/scenarios/scenario_gen.py --only rq6 --seeds 4 --steps 40
```

The full suite is 63 arms at 32 seeds — about 2,000 runs, an overnight job. `--only` and
`--seeds` are the normal way to use it during development. `--steps` truncates every arm, which
is useful for checking the plumbing but produces a transition that has barely started, so do not
read results off a short run.

Output goes to one timestamped directory per invocation:

```
Results/scenarios/2026-08-03_143012/
    input_data/    scenarios.yaml as given, plus every arm's fully resolved parameters
    output_data/
        <group>/<arm>/history.parquet        one row per period per seed
                      concentration.parquet  one row per period per seed
                      events.parquet         the RQ1 event study, in event time
                      variogram.parquet      one row per lag bin per seed
                      parcels.parquet        one row per parcel per seed
                      geography.parquet      one row per parcel, shared by every seed
                      summary.parquet        one row per seed
        <group>_summary.csv                   the same summaries, flat and readable
    figures/       the research-question figures, named to match paper/main.tex
```

`input_data/arms_resolved.json` is what makes a run reproducible: it records the resolved
`Params` for every arm, including the defaults from `src/pmabm/config.py` that the yaml never
mentions.

### Why parquet, and one file per arm

The full suite is roughly 0.8 GB written this way and about 3 GB written as stacked CSV, and the
difference is mostly `parcels`, which is one row per parcel per seed — tens of millions of rows
across the suite. Parquet also keeps the dtypes (`metrics.PARCEL_DTYPES` fixes them explicitly)
and lets a figure read the two columns it needs rather than parsing every column of every arm.
One file per arm means comparing two arms reads two small files, and re-running one arm rewrites
only that arm. `<group>_summary.csv` is kept flat because it is one row per seed and is the table
a person actually opens after a run.

### The two per-parcel frames

`parcels` and `geography` are split by what varies. Under `run.fixed_geography` the lattice is
built once per arm and shared by every seed, so a parcel's coordinates, county, soil quality and
estate are identical in every replicate; storing them per seed would repeat eight columns across
every one. `geography.parquet` is therefore written once per arm, carries no `seed` column, and
is joined onto `parcels.parquet` on `parcel` alone. With `fixed_geography: false` it is stacked
per seed like everything else and the join is on both keys — `run_meta.json` records which.

`commons` and `customary_rent` sit in `parcels` despite looking like fixed features of the land:
both are drawn from the model's own generator at construction, so they differ between seeds even
on a shared lattice.

That shared lattice is what makes parcel *n* the same land in every run, and so what licenses the
two operations the spatial figures are built on: averaging a parcel's outcome over seeds, and
differencing two arms parcel-by-parcel on matching seeds. `first_conversion` is `-1` for land
that never converted, which is a *censored* observation rather than a missing one — the quantity
to map is the share of seeds converted by a given period, not the mean of that column.

## Adding an arm

Parameter names are the fields of `pmabm.config.Params`. An unknown name is refused rather than
ignored, because the alternative failure mode — a typo silently producing a *baseline* run under
an ablation's label — is the most dangerous thing this pipeline can do.

Anything not named in an arm inherits `base:`; anything not in `base:` inherits the dataclass
default. To sweep a parameter, give `sweep: {name: [v1, v2, ...]}` instead of `model:`; each
value becomes its own arm, and a single-parameter sweep also gets a numeric axis in the figures.

## Notes that matter for correctness

- **Geography is built per arm, not once per suite.** `uniform_fertility` (RQ6) is applied
  while the lattice is constructed, so a layout shared across arms would silently make that
  ablation a no-op.
- **`fixed_geography: true` is the default here**, unlike the single-regime `multi_seed` run.
  RQ2 and RQ6 compare *spatial* patterns across arms, and redrawing the estate layout per
  seed would mix structural variation into that comparison.
- **There is no "conversion off" switch.** Tier 0 suppresses it with `alpha_0: -30`, which puts
  the conversion probability below 1e-13 at any attainable fiscal pressure, and `lambda_0: -30`
  for the Freehold pathway. Likewise there is no "no eviction" switch: RQ5 uses
  `tau: 1000000`, which suspends eviction for any run of realistic length while leaving the
  shortfall counter live.
- **Two operationalisations are choices, not definitions**, and the paper says so. "Class
  conflict off" (RQ3, RQ7) is `alpha_1: 0` plus `customary_fine_rate: 0`. "Custom's security"
  (RQ4) is swept over `inplace_conversion_rate`, `customary_fine_rate` and `theta` one at a
  time rather than jointly, because a three-way cross product at 32 seeds is 30,000 runs and the
  marginal frontiers are what the argument needs.

## The geography artifact

`data/geography/england_alc.json` holds England's **39 historic counties**, all of them with an
ALC grade. Rebuild it with a network fetch from ONS and Natural England:

```bash
uv run python -m pmabm.build_geography
```

Two things worth knowing about it.

**It held only 37 counties until August 2026.** The filter selecting English counties out of the
England-and-Wales layer tested a mean of boundary *vertices* rather than an area-weighted
centroid. A vertex mean weights a boundary by how finely it happens to be sampled, so Kent,
Cheshire and Somersetshire — each with a long, heavily-vertexed estuarine coast — had their
"centre" dragged offshore and were silently dropped. The criterion is now the share of a county's
sampled interior falling inside England, which separates cleanly: every English county scores
above 0.98 and every Welsh one below 0.01, and any county between 0.05 and 0.95 is printed in the
build log rather than deciding itself.

**The County of London is excluded by name**, via `build_geography.EXCLUDED_COUNTIES`. ONS lists
it separately from Middlesex, but it is a city rather than an agrarian county and Natural England
returns no graded agricultural land inside it — which, retained, meant it fell through to the
national mean baseline and was credited with average farmland. Excluded, its cells go to the
nearest-county assignment in `pmabm.geography`, which returns them to the surrounding counties
whose land they were, and the count comes to exactly the 39 the paper claims.

## Measuring spread without an origin

There is no seed origin any more. `seed_origin_rule`, `Geography.seed_origin`,
`Geography.dist_from_seed`, `geography.reseed_origin` and the five-arm `seed_origin` scenario group
have all been removed, and RQ2's spread measure is now
`pmabm.metrics.spread_variogram`.

**Why.** The old measure regressed each parcel's first-conversion time on its distance from a
designated county, by default the one with the worst ALC grade. That required naming a centre, and
named it out of the *same* fertility field that drives conversion — so a positive slope was partly
underwritten by the choice of ruler rather than found in the run. The apparatus that fenced this
off (a random-county null, both slopes carried per run, a scenario group sweeping five origins) was
all in service of a construction that did not need to exist. None of it was causal: nothing in
`model.py` ever read `dist_from_seed`.

**What replaced it.** The empirical semivariogram of conversion time — for every pair of converted
parcels at lattice distance *d*, half the mean squared difference in their conversion times. No
centre to choose, and it separates contagion from simultaneity on two numbers instead of one:

| column | reads as |
|---|---|
| `spread_nugget_share` | semivariance at the shortest distance, over total variance. Near 0: neighbours convert together, so there is local coherence to spread. Near 1: neighbours differ as much as opposite corners — simultaneity plus noise. |
| `spread_slope_norm` | slope of the curve, variance share per lattice cell. Positive: parcels further apart convert further apart in time. Near zero with a high nugget is the RQ2 falsification. |
| `spread_range_cells` | distance at which semivariance reaches 95% of total variance — the spatial scale. Collapsing onto the first bin means no scale at all. |

`range_cells` is what the radial slope was blindest to: a *short* range with a low nugget means
local patches converting independently, while a range comparable to the lattice means one
country-scale front. Both produce a positive radial slope from a well-placed origin, and they are
different claims about the transition.

The measure is validated in `tests/test_model.py` against two fields whose answer is known by
construction — a radial front (nugget 0.006, slope +0.090, r 0.99) and a shuffle of the same
conversion times, which has an identical marginal distribution and differs only in spatial
arrangement (nugget 1.006, slope −0.001). It is also asserted invariant to translating the
lattice, which is the guarantee `seed_origin_rule` could not give: there, moving the vantage point
changed the sign of the reported slope.

The regional-pattern question — RQ9 in the drafts before enclosure was renumbered into that
slot — has been removed in full, both halves. Its second half was
the origin sweep described above. Its first half compared county mean conversion time against ALC
grade and against the counties the historiography names, which needed a named comparison set
(`EARLY_CAPITALIST_COUNTIES`), a per-county reporting path (`county_conversion_summary`,
`regional_prediction_test`, `missing_named_counties`), a `county` frame through the runner, and a
figure — all for one claim. The ecological question it was a county-level restatement of is still
asked at parcel level by RQ6, via `uniform_fertility`, which is where it belongs.

The county geography itself is untouched and still causal: counties supply each parcel's
ALC-derived fertility baseline and are the unit lords are seeded within, so `Geography.county`,
`county_names` and `county_grade` all remain.
