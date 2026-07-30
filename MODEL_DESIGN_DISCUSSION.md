# Brenner/Wood transition ABM — design discussion so far

Paste this into a fresh Claude chat (web/mobile) to keep discussing on the go. It won't have file access, just this context.

## The goal

An ABM of the transition from feudalism to agrarian capitalism in England, c.1400s–1600s, built to **test the internal validity of Brenner and Wood's theory of transition** — not to illustrate it. The test: do the macro outcomes the theory claims (collapse of customary tenure, land concentration, the landlord/capitalist-tenant/wage-labourer triad, spread of "improvement" ideology) actually emerge from the micro-mechanisms the theory identifies, starting from a small, spatially localised seed of competition — rather than being assumed globally from the start?

This matters because Brenner and Wood's whole critique of rival ("commercialization") theories of transition is that those theories smuggle in an already-existing embryonic capitalism. An ABM that hard-codes competition from tick one would make the same mistake.

## Why the old code (src/) doesn't work

Tenants are hard-assigned to one landlord for the whole run, rent is set by simple consumption-splitting, no land market, no tenant mobility. Competition is a parameter, not a behavior — so it can't test whether competition is genuinely emergent.

## The core design question we resolved: how does competition spread from a seed?

Key tension: inflation (which erodes landlords' fixed customary rent income) is a **uniform, system-wide pressure** — so why would conversion to leasehold/market rent spread locally rather than happening everywhere at once?

Resolution: converting is a *costly institutional break* (evicting/renegotiating against custom), so landlords are heterogeneously reluctant to go first, even under uniform pressure. What turns isolated defections into a spreading pattern is that **a first defector's success becomes visible and changes the calculus for others**. Three channels do this (implemented as independent switches so the model can test which matters):

1. **Tenant mobility** — a tenant leaves a customary estate for better terms elsewhere, forcing the losing landlord's hand.
2. **Observed outcomes** — neighbouring landlords, within an awareness radius, see a defector's realised income/wealth rise and recalibrate.
3. **Ideological legitimation** — "improvement" ideology spreads as a legitimating justification for enclosure (slower-moving, lowers the political/legal cost of the *next* conversion).

We also grounded the *initial* localisation of the seed in land fertility/ecological heterogeneity: some estates are ecologically buffered, some aren't, so demographic/subsistence crises (and the resulting tenancy vacancies a landlord can exploit) cluster non-arbitrarily in space.

## The harder question: why could landlords break customary right at all?

Checked directly against Brenner (1976, *Agrarian Class Structure...*) and Wood (2002, *The Origin of Capitalism*). Two concrete historical levers (Brenner pp.61–63):

1. **Escheat/vacancy absorption** — after the Black Death and demographic collapse, vacant customary holdings (no heir, tenant fled) could simply be absorbed into the landlord's demesne/leasehold sector rather than re-granted as customary. No confrontation needed.
2. **Entry fines** — even where customary rent itself was fixed, landlords retained the right to charge a fine at conveyance (death/sale/inheritance). This became the lever to substitute a de facto market rent for the fixed customary one. Peasants resisted hard (Ket's Rebellion, 1549) and lost.

**But the deeper, comparative answer** (Brenner pp.68–75, Wood pp.98–105) is about the **relationship between the peasantry, the landlords, and the centralising state**:

- In **France**, the monarchy had an independent fiscal interest in a free, secure peasantry (direct royal taxation competing with the lords for the same surplus), so the crown systematically intervened *for* peasants against landlords.
- In **England**, Tudor centralisation ran *through* the landlord class (via Parliament), not against it — the gentry backing the crown against "overmighty" magnates were the same people pushing enclosure. The state had no independent reason to protect customary tenure.

This means English peasants didn't lose because they were locally weaker than French peasants (they weren't — comparable village-level solidarity existed across Western Europe); they lost because the state that could have backed their claims had no structural reason to.

**Implication for the model**: this "political resistance" question doesn't need its own emergent submodel of peasant revolt. Brenner himself treats the state–landlord alliance as a largely exogenous, structural condition of the English case. So the model treats it as a **scenario-level parameter θ** (an "England" regime vs. a counterfactual "France" regime), with only idiosyncratic instance-level resistance left stochastic. This also gives a clean validity test almost for free: flip θ to the France regime and check whether the transition to the triad fails to emerge, as Brenner's comparative logic predicts.

## Current state of the formal model (equations + TikZ diagram now in paper/main.tex, Section 4)

- **Agents**: Landlord (wealth, consumption requirement, awareness radius, ideology exposure), Tenant (single agent type with tenure state ∈ {Customary, Leasehold, Landless} that evolves endogenously — not three hard-coded classes), Land parcel (fertility state, ecological process).
- **Exogenous drivers**: inflation eroding real customary rent; logistic land-fertility process with ecological shocks, driving a turnover/vacancy hazard.
- **Landlord's conversion decision**: logistic discrete-choice at each vacancy, driven by fiscal pressure Δᵢ(t), neighbour-observation term, and effective resistance θ_eff(t).
- **Spread mechanisms**: the three channels above (mobility, observed outcomes, ideology diffusion via a Bass-type contagion equation).
- **Tenant response**: wealth accumulation, surplus reinvestment into land ("improvement"), diminishing-returns productivity function, eviction into a wage-labour pool on repeated rent default.
- **Emergent outputs / validation targets**: tenure-state shares over time, land-concentration Gini, spatial spread pattern of conversion (contagion vs. simultaneous — the key internal-validity test), aggregate output vs. historical series (qualitative check only), and the England-vs-France θ comparative run.

Every equation in the paper is paired with a specific page-cited Brenner or Wood quote justifying it.

## Not yet done

- Results/Discussion/Conclusion sections of the paper are still empty stubs.
- No code has been written yet — this is equations + diagram only, by design, until the mechanism was pinned down.
- Haven't yet decided: functional form/parameter values for calibration-free qualitative runs; whether land is on an explicit spatial lattice or an abstract awareness-radius graph (leaning toward the latter, per Wood's "awareness of land market" framing, but not settled).
