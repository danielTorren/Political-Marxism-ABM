"""Model parameters.

Values mirror Table 1 (`\\label{tab:params}`) of ``paper/main.tex``. They are placeholder
starting points for qualitative runs, not fitted estimates, exactly as the paper states.

Two symbols in the paper are overloaded; they are disambiguated here:

* ``mu`` is both the Cobb-Douglas fertility exponent and the tenant mobility propensity.
  These are ``cobb_phi`` and ``mobility_range`` respectively.
* ``rho`` is both the fertility regrowth rate and the commons-dependent share
  (``rho_commons``). These are ``phi_regrowth`` and ``commons_share``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    """Every free quantity in the model. Frozen: use :meth:`with_` to derive variants."""

    # ---- Space (paper: Spatial structure) ----------------------------------------
    L: int = 155
    """Lattice long-axis length. The grid covers England's bounding box at square cells; those
    whose centroid falls outside the England boundary are dropped.

    Parcels come out at roughly ``0.31 * L**2``, so 155 gives ~7,400 -- which with the default
    ten lords per county leaves about twenty tenants per estate."""
    lords_per_county: int = 10
    """Estates seeded within each historic county, so estates nest inside counties.

    The land hierarchy is county -> estate -> parcel -> tenant. The county tier is real (ONS
    Ancient Counties, December 1921); the number of lords within it is an abstraction. England
    had on the order of 25,000-65,000 *manors*, far more than this, but the model's landlord is
    better read as a substantial landowning family holding several manors -- the tier Wood has
    in mind when he says land was "concentrated in far fewer hands" (p.99). Ten per county is a
    round figure at that level, not a manor count.
    """
    n_landlords: int = 40
    """Ignored when the geography is built from counties; retained for the synthetic and
    legacy paths, and reported back as the realised estate count."""
    awareness_radius: float = 5.0
    """Euclidean distance, in lattice cells, within which landlords observe each other.

    In cells, not kilometres, so it must be retuned if ``L`` changes. At the defaults it gives
    a landlord roughly ten neighbours to watch."""
    zeta: float = 2.0
    """Within-region jitter on the ALC-derived fertility baseline."""
    uniform_fertility: bool = False
    """Flatten the ALC-derived *regional* fertility structure to its national mean.

    Parcel-level jitter (:attr:`zeta`) is retained, so this removes the spatial *pattern* of
    land quality without removing local heterogeneity -- which is exactly the ablation RQ6
    needs. Moore's charge is that Brenner brackets nature; the model has an ecological driver
    Brenner does not, so the question is whether the timing and location of the transition are
    being set by soil rather than by class position. The spread measure
    (:func:`pmabm.metrics.spread_variogram`) needs no origin, so it reads the same way in both
    arms: spatial structure surviving the ablation is structure the *mechanisms* produced, since
    the only spatially patterned input has been removed.
    """

    # ---- Initial agent state (paper: Agents) --------------------------------------
    landlord_wealth_0: float = 100.0
    consumption_rule: str = "rent_roll"
    """``"rent_roll"`` or ``"absolute"``.

    The paper draws C_i from a fixed range independent of estate size, which puts every
    landlord in heavy deficit at t=0 -- a 28-parcel estate yields ~28 against a requirement of
    ~100 -- so conversion is already likely everywhere in the first period and there is no
    "custom is sustainable, then inflation erodes it" phase for a localised seed to break.
    ``rent_roll`` instead scales the requirement to the estate's own initial rent roll, so
    fiscal pressure starts near zero and is *created* by inflation, which is what the paper's
    Exogenous drivers section actually describes.
    """
    consumption_ratio: tuple[float, float] = (0.85, 1.05)
    """Requirement as a multiple of the estate's initial nominal rent roll, under
    ``consumption_rule="rent_roll"``. Straddling 1.0 leaves some houses already living
    slightly beyond their means and others with a little slack."""
    consumption_range: tuple[float, float] = (80.0, 120.0)
    """C_i under ``consumption_rule="absolute"`` -- the paper's literal version."""
    init_freehold_share: float = 0.05
    """Remainder start Customary. Leasehold and Landless both start empty, so that the
    triad is an outcome rather than a precondition."""
    tenant_wealth_0: float = 10.0
    k_trad: float = 1.0
    """Traditional (non-market) capital baseline; also the reset value at succession."""
    customary_rent_range: tuple[float, float] = (0.5, 1.5)
    """r^C_j(0), fixed in nominal terms at grant and attached to the parcel."""
    household_labour: float = 1.0
    """ell-bar."""
    mobility_range: tuple[float, float] = (0.0, 0.3)
    """mu_j, the tenant mobility propensity."""

    # ---- Exogenous processes (paper: Exogenous drivers) ---------------------------
    inflation: float = 0.02
    phi_regrowth: float = 0.1
    """rho in the logistic fertility equation."""
    shock_prob: float = 0.05
    shock_magnitude: tuple[float, float] = (0.0, 2.0)
    """epsilon_k(t) = Bernoulli(shock_prob) * U(shock_magnitude)."""
    subsistence_output: float = 2.0
    """y-bar, the output threshold below which the turnover hazard rises."""
    eta_0: float = 0.02
    """Baseline hazard. Interpreted as ordinary generational succession."""
    eta_1: float = 0.05
    """Sensitivity of the crisis hazard to a subsistence shortfall."""
    eta_2: float = 0.03
    """Sensitivity of the crisis hazard to enclosure, for commons-dependent land."""

    # ---- Conversion, resistance, Freehold -----------------------------------------
    alpha_0: float = -4.0
    alpha_1: float = 2.5
    """Weight on fiscal pressure, which now enters as a *relative* shortfall (see
    :attr:`normalise_fiscal_pressure`).

    Together with ``alpha_0`` this is set so that fully eroded customary income is
    **necessary but not sufficient**: a landlord with no converted neighbours still converts
    only rarely (p ~ 0.02), and it takes the neighbour signal to tip the decision. That is the
    paper's own claim -- uniform inflationary pressure cannot by itself produce a *spreading*
    rather than simultaneous pattern -- and with the paper's own coefficients it does not
    hold, because ``alpha_1 * Delta`` alone saturates the logit.
    """
    alpha_2: float = 3.0
    """Weight on observed neighbour outcomes; large enough that the spread channels decide
    timing rather than merely perturbing it."""
    alpha_3: float = 1.0
    normalise_fiscal_pressure: bool = True
    """Enter fiscal pressure as ``Delta_i / C_i`` rather than in absolute units.

    The paper's ``Delta_i = C_i - R^C_i - R^L_i`` mixes a per-landlord consumption
    requirement, drawn independently of estate size, with rent summed *over* the estate. A
    4-parcel estate therefore shows far more pressure than a 66-parcel one for purely
    arithmetic reasons, and since ``alpha_1 * Delta`` dominates the logit, estate size ends up
    the main determinant of who converts first. Normalising makes the term scale-free.
    """
    inplace_conversion_rate: float = 0.05
    """Per-period hazard that a landlord reopens terms on a *sitting* customary tenant.

    The paper says a fine escalation is "a special case of the same conversion decision", but
    only ever reaches that decision at a vacancy, so the in-place path silently disappears and
    almost no tenant is present both before and after their own conversion. This restores it:
    Brenner's entry fines were levied on the sitting tenant, and "in the long run fines could
    be substituted for competitive commercial rents".
    """
    theta: float = 2.0
    """State-landlord alliance. 2.0 = England regime, 6.0 = France regime."""
    xi_0: float = 0.3
    xi_1: float = 0.5
    lambda_0: float = -5.0
    """Set so that in the England regime the Freehold pathway is decisively rarer than
    conversion (~1% against ~5% with no converted neighbours), and in the France regime the
    ordering reverses. With the paper's -4.0 the two are comparable, and England produces
    freehold smallholding -- the outcome the theory assigns to France."""
    lambda_1: float = 0.3
    lambda_2: float = 1.0
    t_star: int = 100
    """Period at which the state stops contesting enclosure, and the entry-fine loophole
    closes the Freehold pathway."""
    chi: float = 2.0
    """Entry cost into a converted tenancy, in periods of the estate's benchmark rent."""
    xi_inherit: float = 0.5

    # ---- Spread channels (paper: Spread of the market imperative) -----------------
    ideology_rule: str = "reproduction"
    """``"reproduction"``, ``"material"`` or ``"contagion"``.

    Under ``reproduction`` the landlord's improving disposition is an accumulating stock,
    exactly as the tenant's is, driven by how far his *wealth* falls short of what it must
    sustain. This is Brenner's "rules for reproduction" as Wood states them: landlords "came
    to depend on the market in historically unprecedented ways just to secure the conditions
    of their own self-reproduction" (p.53), and the transformation "created a need for such
    improvements simply to permit the principal economic actors -- landlords and peasants --
    to reproduce themselves" (p.66). What is at stake is not this year's income but whether
    the house can go on being the house, which is a question about the stock behind it:

        runway_i  = W_i / C_i                       periods of requirement in reserve
        press_i   = 1 - min(1, runway_i / H)        urgency, 0 when comfortably covered
        iota_i    <- iota_i(1 - decay) + beta_L * press_i * (1 - iota_i)

    A lord with deep reserves feels no urgency even while his rents erode; one near the end of
    his cushion must act. That heterogeneity is material and is not merely estate size.

    Under ``material`` the landlord's improving disposition is not an idea that circulates at
    all: it is simply the share of his income that already comes from market-determined rent
    rather than customary due, ``R^L / (R^C + R^L + fines)``. A lord who lives by competitive
    rents is, by that fact, committed to competitive production, and his resistance to
    breaking custom falls accordingly. Nothing is transmitted and nothing is imposed -- the
    disposition is a read-out of how far the estate has already been pulled into producing
    for value rather than for subsistence.

    ``material`` is the simpler income-side version: the share of receipts already coming from
    market rent, ``R^L / (R^C + R^L + fines)``. Instantaneous rather than accumulating.

    ``contagion`` restores the paper's Bass-type diffusion, in which the disposition spreads
    between neighbouring landlords. Keeping all three switchable is the point: the
    England/France comparison can then ask whether an idea needs to spread for the transition
    to happen, or whether material pressure alone accounts for it.
    """
    beta_landlord: float = 0.15
    """Growth rate of the landlord disposition under reproduction pressure."""
    iota_landlord_decay: float = 0.02
    """Decay when the pressure lifts, mirroring the tenant's ``iota_decay``."""
    reproduction_horizon: float = 10.0
    """H: periods of the household requirement a lord expects to hold in reserve. Wealth above
    this registers as no pressure at all."""
    beta_1: float = 0.002
    """Autonomous ("innovation") term in ideology diffusion. Deliberately small: with the
    paper's 0.01 the disposition reaches ~1 on its own schedule whether or not any tenancy
    has converted, which turns a transmission channel into an exogenous clock."""
    beta_2: float = 0.15
    """Contagion term. Multiplies neighbours' realised **conversion share**, not their
    ideology, so that ideology is seeded by observed breaks from custom -- the feedback the
    paper's own causal diagram draws but its equation omits."""
    iota_landlord_0: float = 0.0

    # ---- Tenant economy (paper: Tenant response) ----------------------------------
    theta_rent: float = 0.3
    beta_tenant: float = 0.1
    """beta^T, growth rate of the tenant improving disposition under competitive pressure."""
    iota_decay: float = 0.02
    """Per-period decay of the improving disposition when competitive pressure is absent.

    Without it the disposition is a ratchet: a dispossessed leaseholder who later takes up a
    customary holding carries a high disposition into a tenure that exerts no competitive
    pressure at all, and keeps it forever. Decay makes improvement a response to *current*
    conditions -- which is the paper's claim -- rather than an acquired trait.
    """
    iota_tenant_0: float = 0.0
    s_bar: float = 0.3
    """Reinvestment ceiling; actual rate is s_bar * iota^T_j(t)."""
    capital_depreciation: float = 0.05
    """Per-period decay of the capital stock.

    The paper's capital equation only ever adds, so improvements are permanent and free to
    maintain and the stock grows without limit for any surviving tenant. Depreciation makes
    improvement something that must be *sustained* out of current surplus, which is what makes
    it a standing compulsion rather than a one-off investment.
    """
    w_min: float = 0.0
    tenant_consumption: float = 1.0
    customary_fine_rate: float = 0.5
    """Share of a customary tenant's surplus above subsistence taken by the lord as an
    arbitrary fine, and credited to the lord's receipts.

    This is Brenner's "squeeze", which the paper quotes but never formalises: extraction
    "tended to confiscate not merely the peasant's income above subsistence...but at the same
    time threatened the funds necessary to refurbish...productivity". It is what makes
    improvement unrewarding under custom, and it gives the lord a non-confrontational
    alternative to conversion -- the tension the theory turns on.
    """
    freehold_shadow_rent: bool = True
    """Whether a Freeholder feels the estate's competitive benchmark as a yardstick.

    A Freeholder pays no rent, so the only thing that can make them improve is the goods-market
    compulsion Wood insists on: "even owner-occupiers would be subject to those pressures once
    the competitive productivity of agrarian capitalism set the terms of economic survival"
    (p.54). The model implements that as a *shadow* rent -- the estate benchmark enters M_j even
    though nothing is paid.

    That construction is what RQ1 turns on, and it cuts both ways. With it ``True`` a Freeholder
    carries the full improving pressure with none of the rent outflow, and so should out-
    accumulate a Leaseholder -- which is Allen's yeoman, but arrived at by construction rather
    than by result. With it ``False`` Freeholders face no competitive test at all and improvement
    requires competitive tenancy by fiat, which is Brenner's claim assumed rather than tested.
    Neither arm settles RQ1 alone; the pair brackets it, which is why this is a switch.
    """
    cobb_phi: float = 0.3
    """mu, the fertility exponent in the production function."""
    cobb_capital: float = 0.3
    """gamma, the capital exponent. Labour takes 1 - mu - gamma."""
    tau: int = 3
    """Consecutive periods of shortfall tolerated before eviction."""

    # ---- Engrossment and enclosure -------------------------------------------------
    psi_0: float = -2.0
    psi_1: float = 3.0
    """Rescaled alongside :attr:`alpha_1` for normalised fiscal pressure."""
    psi_2: float = 0.1
    commons_share: float = 0.2
    nu_1: float = 0.01
    nu_2: float = 0.05
    nu_3: float = 0.1
    enclosure_0: float = 0.0
    enclosure_rule: str = "national"
    """``"national"`` or ``"local"``: whether enclosure diffuses everywhere at once.

    ``"national"`` is the paper's stated specification and the baseline. A single aggregate
    :math:`\\Xi(t)` is driven by the *population* average of landlord improving disposition, so
    enclosure has a timing but no location, and RQ10 can only ever be a timing argument.

    ``"local"`` gives every estate its own :math:`\\Xi_i(t)`, driven by the mean disposition of
    that lord and the lords within :attr:`awareness_radius` -- the same neighbourhood the
    ideology channel already uses. The motivation is an asymmetry in the specification rather
    than a defect in it: Wood ties enclosure to the improvement ethic, that ethic diffuses
    locally here, and enclosure is then the only mechanism in the model that is national while
    its own driver is local.

    **This arm is informative in one direction only, and should be reported as such.** England's
    real enclosure geography was largely set by pre-existing field systems -- the open-field
    Midlands were the enclosable land -- and the model has no field-system layer. So a local
    enclosure front inherits the geography of conversion, because the two share a driver. Two
    fronts that *coincide* under this arm therefore say almost nothing: that is close to
    tautological. Two fronts that *diverge* despite the shared driver are a genuine separability
    result, and one that favours Shaw-Taylor over Wood and Neeson.

    Kept off by default because the specification it departs from is page-cited, and a spatial
    result manufactured by a switch is worth less than a null.
    """

    # ---- Labour market --------------------------------------------------------------
    labour_demand_rule: str = "marginal_product"
    """``"marginal_product"`` or ``"capital_linear"``.

    The paper's rule, ``ell_hired = max(0, delta*k - ell_bar)``, is a function of capital
    alone and never references the wage, so nothing responds to the price of labour and the
    tatonnement has no equilibrium to find -- the wage simply runs to whatever bound it is
    given. ``marginal_product`` instead hires until the marginal product of labour equals the
    wage, which closes the market. ``capital_linear`` restores the paper's rule.
    """
    delta: float = 0.5
    """Used only by the ``capital_linear`` labour-demand rule."""
    kappa: float = 0.1
    wage_0: float = 1.0
    landless_consumption: float = 0.8
    """Subsistence cost per head per period for a household with no land. Interpreted as the
    *mean*: each household draws its own requirement around it, see
    :attr:`landless_consumption_spread`."""
    landless_consumption_spread: float = 0.25
    """Half-width of the uniform draw on each household's own ``landless_consumption``.

    With a single scalar requirement, every landless household faced an identical net income
    ``omega(t)*ell_bar - c``, because the wage is global. Their wealth then crossed zero on the
    same tick and, ``tau_prime`` periods later, an entire eviction cohort left for industry as
    a bloc: 22% of all departures in a 64-seed run happened within five periods, and the
    resulting collapse of the labour pool sent the wage to several times subsistence, where it
    stayed. Drawing the requirement per household breaks the lockstep. Set ``0.0`` to recover
    the old homogeneous behaviour exactly.
    """
    tau_prime: int = 3
    """Consecutive periods in deficit before a landless household leaves the land. Applies
    only under ``exit_rule="counter"``."""
    exit_rule: str = "hazard"
    """``"hazard"`` or ``"counter"``.

    ``counter`` is the original deterministic rule: leave the moment ``tau_prime`` consecutive
    deficit periods have accumulated. Because the deficit is driven by a global wage, that rule
    fires simultaneously across a whole cohort. ``hazard`` instead makes departure a draw whose
    probability rises with how *deep* the household's deficit is, so identical households leave
    at different times and the exit flow is smooth rather than an avalanche.
    """
    lambda_exit: float = 0.35
    """Hazard scale for ``exit_rule="hazard"``: ``p = 1 - exp(-lambda_exit * d)``, where ``d``
    is the accumulated deficit measured in periods of the household's own consumption."""
    charge_wage_bill: bool = True
    """Whether hiring tenants pay for the labour they hire.

    The paper's wealth equation omits a wage bill even though hired labour enters the
    production function, which makes labour free and accumulation unbounded: capital raises
    output, output raises wealth, wealth raises capital, and nothing prices the extra hands.
    Setting this ``True`` charges ``omega(t) * hired``, which is required for the model to be
    well posed. Set ``False`` only to reproduce the paper's literal specification.
    """
    wage_adjust_cap: float = 1.0
    """Numerical guard on the tatonnement. The paper's excess-demand ratio
    ``(D - |L|)/|L|`` is unbounded as the landless pool empties, so the wage can move by
    orders of magnitude in one period; the ratio is clipped to +/- this value."""

    # ---- The domestic market (Wood, p.103) ------------------------------------------------
    urban_demand: bool = True
    """Whether agents who leave the land create a market for what the land produces.

    Wood's claim is that agrarian productivity "created a highly productive agriculture
    capable of sustaining a large population not engaged in agricultural production, but also
    an increasing propertyless mass that would constitute both a large wage-labour force and
    a domestic market for cheap consumer goods" (p.103). Without this the exit channel is a
    one-way drain: agents vanish into industry and never affect the countryside again. With
    it, dispossession feeds back as demand, which is the loop the theory actually describes.
    """
    urban_consumption: float = 1.0
    """Food demanded per head of the non-agricultural population, per period. Also the per-head
    cost an urban household must meet out of its wage."""

    # ---- The urban sector as a place, not a sink -----------------------------------------
    urban_return: bool = True
    """Whether a household that left for industry can ever come back.

    With this ``False`` the urban state is absorbing, and the countryside can only lose people to
    it: over a 200-period run the rural population fell by 42% even while births exceeded deaths,
    because the outflow had no counterpart. That is not a neutral assumption -- it builds a
    one-way ratchet into the accounting and then reports the result as a finding about agrarian
    change. With it ``True`` the flow is governed by relative advantage in both directions, so the
    model neither manufactures nor destroys a rural population by construction, and the observed
    drift becomes something the run has to earn.
    """
    urban_wage: float = 1.0
    """What a member of an urban household earns per period.

    Exogenous, and deliberately so: the model has no industrial sector to derive it from, and
    inventing one would extend the paper's scope well past the agrarian transition it is testing.
    Holding it fixed while the rural wage moves endogenously is what makes the direction of
    migration respond to conditions in the countryside -- which is the causal direction the theory
    cares about -- rather than to an industrial cycle the model does not represent.

    Defaulted to exactly :attr:`urban_consumption`, so that an urban household's per-head surplus
    is zero and, by the anchoring of :attr:`birth_slope` at :attr:`mortality_0`, the urban sector
    is demographically stationary. This is the neutrality the return channel is for: the towns
    neither breed nor consume a population of their own, so every change in the rural/urban split
    is migration responding to rural conditions, and nothing else. Setting it below
    ``urban_consumption`` makes the towns a slow sink -- at 0.9 the urban sector loses about 1.9%
    of its people per period, which drained 74% of the *total* population over 200 steps and put a
    third of parcels back into the vacancy queue -- and setting it above makes them a source. Both
    are substantive claims about urban demography that this model has no evidence for; depart from
    equality only deliberately.
    """
    urban_return_rate: float = 0.05
    """Per-period probability that an urban household whose members would be better off on the
    land actually returns, entering the landless pool. Below 1 because returning is a decision
    taken over time and against a cost, not an arbitrage executed instantly."""
    goods_price_0: float = 1.0
    kappa_goods: float = 0.05
    """Adjustment speed of the produce price to excess urban demand."""
    goods_price_cap: tuple[float, float] = (0.2, 20.0)
    """Bounds on the price, for the same numerical reason the wage is bounded."""

    # ---- Population ---------------------------------------------------------------------
    population_rule: str = "household_size"
    """``"household_size"``, ``"vital_rates"``, ``"fixed"`` or ``"household"``.

    **household_size** (default) is the full demographic model. A household is no longer a
    single body: it carries a size ``n_j``, supplies ``n_j * ell_bar`` labour, and consumes
    ``n_j`` times per-head subsistence. Births raise ``n_j``, deaths lower it, a household that
    reaches zero is extinct and its holding falls vacant with no claimant, and a household too
    crowded to feed itself sheds a member as a new landless household -- which is
    proletarianisation as a *demographic* mechanism, the non-inheriting member driven to wage
    labour, rather than only as a consequence of eviction.

    **vital_rates** applies the same birth and death hazards to single-body households, so a
    birth is immediately a new landless agent. Useful as the intermediate arm: it isolates what
    the hazards do from what the household unit does.

    **fixed** keeps the population closed: nobody is created after initialisation and the only
    outflow is exit to industry. It is retained deliberately, because "does the transition still
    occur with population held fixed?" is a direct test of Brenner's anti-demographic claim.
    Note that it is closed only in the sense of having no births -- exit to industry is
    absorbing, so a long run under this rule depopulates the countryside monotonically and ends
    with more parcels than people to work them. It is a robustness arm, not a default.

    **household** is the superseded threshold rule, kept so that earlier results reproduce: a
    tenant whose wealth passes :attr:`birth_threshold` splits off a new landless household at a
    cost of :attr:`birth_cost`. Its defect is that births were conditional on holding land, so
    proletarianisation mechanically sterilised the population.

    Under both new rules the demographic rate is a property of the individual household's own
    circumstances, and the aggregate rate is whatever the class composition makes it. That is
    the opposite of an aggregate food-population law: the same land can support a growing or a
    shrinking population depending on how the surplus is distributed, which is Brenner's claim
    rather than Postan's.
    """
    birth_threshold: float = 60.0
    birth_cost: float = 30.0
    """Wealth transferred from parent to the new household, so a birth is not free. Used by the
    superseded ``household`` rule and by ``vital_rates``."""

    # ---- Vital rates (population_rule in {"vital_rates", "household_size"}) ----------------
    # Fertility and mortality are functions of a household's own per-head surplus, never of an
    # aggregate ratio of population to food. A landholding household's surplus depends on its
    # land, its capital and the rent it owes; a landless household's depends on the wage. So the
    # demographic regime of the whole model is set by the balance of classes within it, and the
    # productivity of the land enters household by household rather than through a global
    # carrying capacity.
    mortality_0: float = 0.02
    """Baseline probability that a given household member dies in a period.

    Distinct from :attr:`eta_0`, which is now read strictly as a *conveyance* hazard -- the
    generational turnover of the head at which the landlord gets to act on the tenancy -- and
    is population-neutral because the heir inherits the household whole. Were ``eta_0`` also to
    remove a member, the two would double-count the same deaths.
    """
    mortality_crisis: float = 0.15
    """Excess mortality per unit of per-head deficit, i.e. the nutritional channel. A household
    one full period of consumption short per head faces ``mortality_0 + mortality_crisis``."""
    birth_slope: float = 0.04
    """Rise in the per-member birth rate per unit of relative per-head surplus.

    The rate is anchored at :attr:`mortality_0` when surplus is zero, so a household exactly
    meeting subsistence is demographically stationary by construction. Prosperity raises
    fertility up to :attr:`birth_rate_max`; a household in deficit falls below replacement, and
    so is losing people to low fertility as well as to excess mortality.
    """
    birth_rate_max: float = 0.08
    """Cap on the per-member birth rate, so a very rich household cannot breed without limit."""
    household_size_0: int = 3
    """Initial household size, in adult labour equivalents.

    Not 1. Once mortality applies per member, a household of one is extinguished by a single
    death -- at ``mortality_0 = 0.02`` that is a 91% chance of the line failing within 120
    periods, and a run started from single bodies loses half its households to extinction before
    the transition begins, leaving parcels no living household can take up. A household line is
    not as fragile as a person, and starting from an established family is what makes the
    distinction real.
    """
    subsistence_per_head: float = 0.5
    """Per-head subsistence for a landholding household under ``household_size``.

    Separate from :attr:`tenant_consumption`, which stays the per-*household* cost used by the
    single-body rules, so that adopting the household model does not silently recalibrate the
    older arms. The value matters because it sets where partition bites: output rises with
    labour as ``n^(1-mu-gamma)`` while subsistence rises as ``n``, so a household on one parcel
    of average quality becomes unviable at about
    ``n = (output_per_parcel / subsistence_per_head)^(1/(mu+gamma))``. A holding of two parcels
    carries proportionately more, which is the channel through which land quality and
    consolidation govern how many people the countryside can hold.

    Calibrated at 0.5 against a mean output of roughly 1.8 per parcel, which gives a mean
    household of two to three members, a natural extinction rate of a fraction of a percent per
    period, and mild population growth. Raising it towards 0.7 shrinks households back towards
    single bodies and multiplies extinctions, which is the failure mode the household rule exists
    to avoid; lowering it towards 0.45 produces markedly faster population growth.
    """
    household_size_max: int = 20
    """Numerical guard on ``n_j``.

    The largest engrossed holdings do sit against it: with many parcels their marginal product of
    labour stays above the wage at any size, so nothing but the cap stops them growing, and the
    "largest household" series in ``dyn_demography`` is pinned to whatever value is set here.
    That series should therefore be read as a guard rather than as a result. The *mean* size is
    the diagnostic one -- if it approaches the cap, partition is miscalibrated.
    """

    # ---- Partition: how a household sheds members into the proletariat --------------------
    partition_min_size: int = 2
    """A household of one has nobody to send away."""
    partition_surplus: float = 0.0
    """Per-head surplus, relative to per-head consumption, below which a household is crowded
    enough to shed a member. At the default ``0.0`` a household partitions once it can no longer
    feed everyone in it from the holding."""
    partition_rate: float = 0.25
    """Per-period probability that a crowded household actually sheds a member. Below 1 because
    leaving is a decision taken over time, not an accounting identity."""
    partition_dowry: float = 5.0
    """Wealth the departing member takes with them, so partition costs the parent household
    something and the new landless household is not destitute on arrival."""
    impartible_inheritance: bool = False
    """If ``True``, conveyance sends every member but the heir into a single new landless
    household, instead of letting the family continue on the holding.

    This is the strong form of the proletarianisation pump, and it is off by default because the
    resource-pressure partition above already produces the same flow without assuming a
    uniform inheritance custom. Turning it on is the way to ask how much of the transition the
    inheritance regime alone can drive.
    """

    # ---- Mechanism switches ---------------------------------------------------------
    # The paper requires the three spread channels to be independently switchable so that
    # RQ2 can ask which, if any, is necessary for contagion rather than simultaneity.
    channel_observation: bool = True
    channel_ideology: bool = True
    random_awareness_graph: bool = False
    """Rewire the landlord awareness graph at random, preserving every landlord's degree.

    RQ2's *measurement* control rather than one of its ablations. The channel ablations ask
    whether contagion exists; without this arm they cannot show that the semivariogram would
    have *detected* a spreading front had there been one, so "the nugget is near 1" is not yet
    evidence of simultaneity. Here contagion is left fully intact and only its geometry is
    destroyed: each lord watches the same number of other lords, drawn from anywhere in England
    rather than from within :attr:`awareness_radius`. A variogram that goes flat under this arm
    while conversion still completes is the instrument working; one that stays structured is the
    instrument reading something other than spatial transmission, and RQ2 would have to be
    re-measured before it could be reported.

    Rewired once when the lattice is built, so under ``run.fixed_geography`` the random graph is
    shared by every seed exactly as the spatial one is.
    """
    enable_engrossment: bool = True
    enable_enclosure: bool = True
    competitive_allocation: bool = True
    """Let a vacated market-exposed parcel to the highest bidder (paper: Competitive
    allocation), rather than to the wealthiest available landless agent at a fixed estate
    benchmark.

    This replaces the tenant-mobility channel of earlier drafts. Mobility was withdrawn on
    the evidence: Brenner places the fight over it before this period ("by the mid-fifteenth
    century...to break definitively feudal controls over its mobility and to win full
    freedom", p.61), and the version in which lords bid against one another to retain mobile
    tenants is Postan's, which Brenner quotes only to reject (p.39 n.17). What Wood does
    describe is lords choosing among rival takers -- "lease land to the highest bidder, at
    whatever rent the market would bear" (p.101) -- which is an allocation rule, and is what
    this implements. Set ``False`` for the earlier wealthiest-affordable rule.
    """

    # ---- Run control ------------------------------------------------------------------
    n_steps: int = 200
    seed: int = 0

    def with_(self, **changes) -> "Params":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    def __post_init__(self) -> None:
        if self.cobb_phi + self.cobb_capital >= 1.0:
            raise ValueError(
                "Production requires mu + gamma < 1 so labour has a positive exponent; "
                f"got mu={self.cobb_phi}, gamma={self.cobb_capital}"
            )
        if not 0.0 <= self.init_freehold_share <= 1.0:
            raise ValueError("init_freehold_share must be a share in [0, 1]")
        if self.labour_demand_rule not in ("marginal_product", "capital_linear"):
            raise ValueError(f"unknown labour_demand_rule: {self.labour_demand_rule!r}")
        if self.population_rule not in ("fixed", "household", "vital_rates", "household_size"):
            raise ValueError(f"unknown population_rule: {self.population_rule!r}")
        if self.exit_rule not in ("counter", "hazard"):
            raise ValueError(f"unknown exit_rule: {self.exit_rule!r}")
        if self.enclosure_rule not in ("national", "local"):
            raise ValueError(f"unknown enclosure_rule: {self.enclosure_rule!r}")
        if not 0.0 <= self.landless_consumption_spread < 1.0:
            raise ValueError("landless_consumption_spread must be in [0, 1)")
        if self.household_size_0 < 1:
            raise ValueError("household_size_0 must be at least 1")
        if self.household_size_max < self.household_size_0:
            raise ValueError("household_size_max must be at least household_size_0")
        if self.birth_rate_max < self.mortality_0:
            raise ValueError(
                "birth_rate_max below mortality_0 makes every household sub-replacement "
                "regardless of its surplus, so the population can only fall"
            )
        if self.consumption_rule not in ("rent_roll", "absolute"):
            raise ValueError(f"unknown consumption_rule: {self.consumption_rule!r}")
        if self.ideology_rule not in ("reproduction", "material", "contagion"):
            raise ValueError(f"unknown ideology_rule: {self.ideology_rule!r}")
        if not 0.0 <= self.customary_fine_rate <= 1.0:
            raise ValueError("customary_fine_rate must be a share in [0, 1]")


#: The two scenario regimes used for RQ3. Only theta differs: the paper treats the
#: state-landlord alliance as the decisive, exogenous condition separating the cases.
ENGLAND = Params(theta=2.0)
FRANCE = Params(theta=6.0)
