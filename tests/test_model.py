"""Invariant and smoke tests.

These check the things that must hold for any parameterisation -- accounting identities,
state consistency, reproducibility -- rather than pinning particular numeric outcomes, which
are placeholder values by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmabm.config import ENGLAND, Params
from pmabm.geography import GeographyMissing, load_artifact
from pmabm.metrics import (
    PARCEL_DTYPES,
    concentration_frame,
    consolidation_by_fertility,
    event_study,
    geography_frame,
    history_frame,
    holding_sizes_at,
    lorenz,
    occupant_continuity,
    parcel_frame,
    region_consolidation_summary,
    spread_variogram,
    spread_variogram_curve,
    summary,
)
from pmabm.model import MARKET_EXPOSED, OCCUPIED, Model, Tenure, _gini


@pytest.fixture(scope="session")
def artifact():
    try:
        return load_artifact()
    except GeographyMissing:
        pytest.skip("geography artifact not built; run `uv run pmabm build-geography`")


@pytest.fixture(scope="session")
def small(artifact):
    return ENGLAND.with_(L=30, lords_per_county=2, n_steps=40, seed=0)


@pytest.fixture(scope="session")
def finished(small, artifact):
    return Model(small, artifact=artifact).run()


# --- construction -----------------------------------------------------------------------
def test_geography_covers_england(finished):
    geo = finished.geo
    assert geo.n_parcels > 50
    assert geo.phi_bar.min() > 0
    assert len(geo.estate_parcels) == geo.n_landlords == finished.n_landlords
    assert sum(len(e) for e in geo.estate_parcels) == geo.n_parcels


def test_estates_nest_inside_counties(finished):
    """The land hierarchy is county -> estate -> parcel; an estate may not straddle counties."""
    geo = finished.geo
    assert geo.n_counties > 20, "expected the historic counties of England"
    for estate in geo.estate_parcels:
        if len(estate):
            assert len(set(geo.county[estate])) == 1


def test_lords_per_county_is_respected(artifact):
    """Each county gets the configured number of lords, or one per parcel if it is tiny."""
    from pmabm.geography import build as build_geo

    params = ENGLAND.with_(L=40, lords_per_county=4)
    geo = build_geo(params, np.random.default_rng(0), artifact=artifact)
    for c in range(geo.n_counties):
        members = np.nonzero(geo.county == c)[0]
        if len(members) == 0:
            continue
        lords = set(geo.landlord[members])
        assert len(lords) == min(params.lords_per_county, len(members))


def test_county_land_quality_is_historically_ordered(finished):
    """A sanity check on the ALC pipeline: East Anglian arable must beat northern uplands."""
    geo = finished.geo
    grade = dict(zip(geo.county_names, geo.county_grade))
    if "Cambridgeshire" in grade and "Cumberland" in grade:
        assert grade["Cambridgeshire"] < grade["Cumberland"]


def test_variogram_separates_a_spreading_front_from_simultaneity(finished):
    """The spread measure must answer RQ2 correctly on cases whose answer is known.

    Two synthetic conversion fields are laid over the real lattice and estates. A radial front
    from an arbitrary corner is spreading by construction; a shuffle of those same times is
    simultaneity by construction, with the *identical* marginal distribution of conversion times.
    Only the spatial arrangement differs, so anything distinguishing them is measuring space and
    not timing -- which is the property the old distance-from-seed regression could not have,
    since its origin was chosen from the same field that drove conversion.
    """
    xy = finished.geo.xy.astype(float)
    corner = xy.min(axis=0)
    radial = np.hypot(xy[:, 0] - corner[0], xy[:, 1] - corner[1])
    # Rank-scaled to the run's own step count, so both fields are valid conversion times.
    front = np.argsort(np.argsort(radial)) * (finished.p.n_steps - 1) // max(len(xy) - 1, 1)

    class _Stub:
        def __init__(self, times):
            self.geo = finished.geo
            self.p = finished.p
            self.first_conversion = times.astype(np.int32)

    spreading = spread_variogram(_Stub(front))
    shuffled = spread_variogram(_Stub(np.random.default_rng(0).permutation(front)))

    # A front: neighbours convert together (low nugget), and the curve climbs almost perfectly
    # monotonically with distance up to its range.
    assert spreading["nugget_share"] < 0.1
    assert spreading["slope_norm"] > 0
    assert spreading["r"] > 0.9
    # Simultaneity: every distance already looks like the global variance, so the curve is flat
    # at ~1 and the "range" collapses onto the first bin -- there is no spatial scale to find.
    assert shuffled["nugget_share"] == pytest.approx(1.0, abs=0.1)
    assert shuffled["range_cells"] < spreading["range_cells"]
    # Asserted as a ratio rather than an absolute tolerance: the slope is in units of variance
    # share per lattice cell, so its scale depends on L and would need retuning per fixture.
    assert abs(shuffled["slope_norm"]) < 0.05 * spreading["slope_norm"]


def test_variogram_needs_no_origin(finished):
    """Relabelling space must not change the answer.

    The measure depends only on *relative* position, so translating the lattice or transposing it
    leaves it invariant. This is the guarantee the removed ``seed_origin_rule`` could not offer:
    there, moving the vantage point changed the sign of the reported slope.
    """
    import dataclasses

    baseline = spread_variogram(finished)

    class _Stub:
        def __init__(self, geo):
            self.geo = geo
            self.p = finished.p
            self.first_conversion = finished.first_conversion

    shifted = dataclasses.replace(finished.geo, xy=finished.geo.xy + 1000)
    assert spread_variogram(_Stub(shifted))["slope_norm"] == pytest.approx(
        baseline["slope_norm"], nan_ok=True
    )


def test_every_parcel_belongs_to_exactly_one_estate(finished):
    geo = finished.geo
    owners = np.concatenate([np.full(len(e), i) for i, e in enumerate(geo.estate_parcels)])
    parcels = np.concatenate(geo.estate_parcels)
    assert len(np.unique(parcels)) == geo.n_parcels
    assert np.array_equal(geo.landlord[parcels], owners)


def test_parcel_neighbours_are_same_estate(finished):
    geo = finished.geo
    for p, nbrs in enumerate(geo.neighbours):
        assert np.all(geo.landlord[nbrs] == geo.landlord[p])


def test_initial_population_has_no_leasehold_or_landless(small, artifact):
    model = Model(small, artifact=artifact)
    states = model.people.state[: model.people.n]
    assert (states == int(Tenure.LEASEHOLD)).sum() == 0
    assert (states == int(Tenure.LANDLESS)).sum() == 0
    # the triad must be an outcome, so every parcel starts occupied
    assert (model.occupant >= 0).all()


# --- invariants that must hold every period -------------------------------------------------
def test_parcel_occupancy_is_consistent(small, artifact):
    """Every parcel is held, queued, or awaiting resolution -- and holdings agree both ways."""
    model = Model(small, artifact=artifact)
    for _ in range(30):
        model.step()
        occ = model.occupant
        for parcel, person in enumerate(occ):
            if person >= 0:
                assert parcel in model.people.holdings[person], "occupant lacks the parcel"
                assert model.people.state[person] in [int(s) for s in OCCUPIED]
        for person in range(model.people.n):
            for parcel in model.people.holdings[person]:
                assert occ[parcel] == person, "holding not reflected in occupancy"


def test_landless_and_exited_hold_no_land(small, artifact):
    model = Model(small, artifact=artifact)
    for _ in range(30):
        model.step()
        for person in range(model.people.n):
            if model.people.state[person] in (int(Tenure.LANDLESS), int(Tenure.EXITED)):
                assert not model.people.holdings[person]


def test_hired_labour_never_exceeds_the_landless_pool(small, artifact):
    """The rationing rule must stop production drawing on more bodies than exist.

    Compared against the pool *as it stood when hiring cleared* (schedule step 2), not the
    end-of-period pool: within a period the landless can also exit to industry or take up a
    vacancy, so the end-of-period count is legitimately smaller.
    """
    model = Model(small, artifact=artifact)
    for _ in range(30):
        model.step()
        pool_at_hiring = model.history[-1]["labour_pool"]
        hired = model.people.hired[: model.people.n].sum()
        assert hired <= pool_at_hiring + 1e-6, f"hired {hired} exceeds pool {pool_at_hiring}"


def test_customary_holdings_never_grow(small, artifact):
    """Engrossment targets only market-exposed neighbours, so custom stays single-parcel."""
    model = Model(small, artifact=artifact)
    for _ in range(30):
        model.step()
        for person in model.people.in_states((Tenure.CUSTOMARY,)):
            assert len(model.people.holdings[person]) <= 1


def test_state_stays_finite(finished):
    ppl = finished.people
    for name in ("w", "k", "iota", "y", "rho"):
        values = getattr(ppl, name)[: ppl.n]
        assert np.isfinite(values).all(), f"{name} went non-finite"
    assert np.isfinite(finished.phi).all()
    assert np.isfinite(finished.W).all()
    assert 0.0 <= finished.enclosure <= 1.0
    assert (finished.people.iota[: ppl.n] >= 0).all()
    assert (finished.people.iota[: ppl.n] <= 1).all()


def test_fertility_stays_within_bounds(finished):
    assert (finished.phi >= 0).all()


# --- economics -------------------------------------------------------------------------------
def test_customary_rent_erodes_with_inflation(small, artifact):
    """The paper's core standing pressure: fixed nominal rent loses real value."""
    model = Model(small, artifact=artifact)
    first, last = None, None
    for _ in range(40):
        model.step()
        customary = model.people.in_states((Tenure.CUSTOMARY,))
        if len(customary):
            mean_rent = float(model.people.rho[customary].mean())
            first = mean_rent if first is None else first
            last = mean_rent
    assert model.price_index > 1.0
    assert last < first, "real customary rent should decline"


def test_wage_bill_is_charged_by_default(small, artifact):
    """Without it, hired labour is free and accumulation runs away."""
    charged = Model(small.with_(n_steps=60), artifact=artifact).run()
    free = Model(
        small.with_(n_steps=60, charge_wage_bill=False), artifact=artifact
    ).run()
    charged_capital = charged.people.k[: charged.people.n].max()
    free_capital = free.people.k[: free.people.n].max()
    assert free_capital > charged_capital


def test_france_regime_suppresses_conversion(small, artifact):
    """RQ3's mechanism must at least point the right way: higher theta, less leasehold."""
    england = Model(small.with_(theta=2.0, n_steps=60), artifact=artifact).run()
    france = Model(small.with_(theta=6.0, n_steps=60), artifact=artifact).run()
    assert (france.first_conversion >= 0).sum() < (england.first_conversion >= 0).sum()


def test_channels_can_be_switched_off(small, artifact):
    off = small.with_(channel_observation=False, channel_ideology=False)
    model = Model(off, artifact=artifact).run()
    assert np.allclose(model.observed, 0.0)
    assert np.allclose(model.iota_landlord, model.p.iota_landlord_0)


def test_engrossment_switch_controls_concentration(small, artifact):
    with_e = Model(small.with_(n_steps=60), artifact=artifact).run()
    without = Model(small.with_(n_steps=60, enable_engrossment=False), artifact=artifact).run()
    sizes_without = [
        len(without.people.holdings[j]) for j in without.people.in_states(OCCUPIED)
    ]
    assert max(sizes_without) == 1, "no holding should exceed one parcel without engrossment"
    sizes_with = [len(with_e.people.holdings[j]) for j in with_e.people.in_states(OCCUPIED)]
    assert max(sizes_with) >= 1


# --- reproducibility and reporting ---------------------------------------------------------------
def test_tenure_ratchets_and_never_reverts(small, artifact):
    """Once customary right is extinguished on a parcel it must not come back."""
    model = Model(small.with_(n_steps=80), artifact=artifact)
    ever_leasehold = np.zeros(model.geo.n_parcels, dtype=bool)
    for _ in range(80):
        model.step()
        now = model.parcel_tenure == int(Tenure.LEASEHOLD)
        reverted = ever_leasehold & ~now
        assert not reverted.any(), f"{reverted.sum()} parcels reverted from leasehold"
        ever_leasehold |= now


def test_inplace_conversion_retains_some_tenants(small, artifact):
    """RQ1 needs conversions where the same person is present either side."""
    model = Model(small.with_(n_steps=80), artifact=artifact).run()
    stats = occupant_continuity(model)
    assert stats["n_conversions"] > 0
    assert stats["customary_to_leasehold_inplace"] > 0, "no in-place conversion happened"


def test_inplace_conversion_can_be_disabled(small, artifact):
    model = Model(small.with_(n_steps=40, inplace_conversion_rate=0.0), artifact=artifact).run()
    assert sum(h["inplace_conversions"] for h in model.history) == 0


def test_customary_tenure_exerts_no_competitive_pressure(small, artifact):
    """Brenner's squeeze: under custom, improvement is neither compelled nor rewarded.

    A tenant may still *arrive* with a disposition earned under leasehold -- it travels with
    the person -- but custom applies no pressure to sustain it, so it decays, and customary
    tenants must end up far below market-exposed ones.
    """
    model = Model(small.with_(n_steps=120), artifact=artifact).run()
    ppl = model.people
    customary = ppl.in_states((Tenure.CUSTOMARY,))
    exposed = ppl.in_states(MARKET_EXPOSED)
    if len(customary) == 0 or len(exposed) == 0:
        pytest.skip("need both tenures present to compare")
    assert ppl.iota[customary].mean() < ppl.iota[exposed].mean()


def test_disposition_decays_without_pressure(small, artifact):
    model = Model(small.with_(n_steps=5), artifact=artifact)
    person = int(model.people.in_states((Tenure.CUSTOMARY,))[0])
    model.people.iota[person] = 0.9
    model.step()
    assert model.people.iota[person] < 0.9


def test_market_exposed_tenants_do_accumulate(small, artifact):
    model = Model(small.with_(n_steps=80), artifact=artifact).run()
    exposed = model.people.in_states(MARKET_EXPOSED)
    if len(exposed) == 0:
        pytest.skip("no market-exposed tenants survived to compare")
    assert model.people.iota[exposed].max() > 0.0


def test_rent_scales_with_holding_size(small, artifact):
    """A bigger holding must cost more; engrossing must not be rent-free."""
    model = Model(small.with_(n_steps=80), artifact=artifact).run()
    ppl = model.people
    sizes, rents = [], []
    for j in ppl.in_states((Tenure.LEASEHOLD,)):
        if ppl.holdings[j]:
            sizes.append(len(ppl.holdings[j]))
            rents.append(model._rent_for(j))
    if len(set(sizes)) < 2:
        pytest.skip("no variation in holding size")
    sizes, rents = np.array(sizes, dtype=float), np.array(rents, dtype=float)
    # Per-parcel bids vary with land quality, so the correlation is not expected to be tight;
    # what must hold is that bigger farms pay more, i.e. engrossing is not rent-free.
    assert float(np.corrcoef(sizes, rents)[0, 1]) > 0.0
    cut = np.median(sizes)
    big, small = rents[sizes > cut], rents[sizes <= cut]
    if len(big) and len(small):
        assert big.mean() > small.mean()


def test_population_is_closed_under_the_fixed_rule(small, artifact):
    """No agent may be created from nothing; the only outflow is exit to industry."""
    model = Model(small.with_(n_steps=60, population_rule="fixed"), artifact=artifact)
    start = len(model.people.live())
    for _ in range(60):
        model.step()
        live = len(model.people.live())
        assert live <= start, "population grew under the fixed rule"
    exited = len(model.people.in_states((Tenure.EXITED,)))
    assert len(model.people.live()) + exited == start


def test_household_rule_allows_growth(small, artifact):
    model = Model(
        small.with_(n_steps=60, population_rule="household", birth_threshold=15.0),
        artifact=artifact,
    ).run()
    assert sum(h["births"] for h in model.history) > 0


# --- household demography ----------------------------------------------------------------
@pytest.fixture(scope="session")
def demographic(small, artifact):
    return Model(small.with_(n_steps=80, population_rule="household_size"), artifact=artifact).run()


def test_total_population_accounting_identity(demographic):
    """Total persons change only by birth and death; migration is an internal transfer.

    This is the invariant that would have caught the original ratchet, in which the only
    possible flow was outward.
    """
    history = demographic.history
    for previous, current in zip(history, history[1:]):
        expected = previous["total_population"] + current["births"] - current["deaths"]
        assert current["total_population"] == expected, f"broken at t={current['t']}"


def test_rural_population_accounting_identity(demographic):
    """The rural head count moves by rural vital events plus net migration, and nothing else.

    Partition moves members between households and conveyance passes a household to an heir,
    so neither may alter the count.
    """
    history = demographic.history
    for previous, current in zip(history, history[1:]):
        expected = (
            previous["population"]
            + current["births_rural"]
            - current["deaths_rural"]
            - current["exit_persons"]
            + current["return_persons"]
        )
        assert current["population"] == expected, f"broken at t={current['t']}"


def test_population_shares_are_a_composition(demographic):
    """Every living person sits in exactly one position, so the shares must sum to one."""
    keys = ("customary", "leasehold", "freehold", "landless", "urban")
    for row in demographic.history:
        total = sum(row[f"share_persons_{k}"] for k in keys)
        assert total == pytest.approx(1.0, abs=1e-9), f"shares sum to {total} at t={row['t']}"
        assert sum(row[f"persons_{k}"] for k in keys) == row["total_population"]


def test_urban_state_is_not_absorbing(small, artifact):
    """Leaving for industry must be reversible, or the countryside can only drain.

    Driven directly rather than waited for: whether a short run at test scale happens to make
    the countryside attractive enough is incidental to whether the channel exists.
    """
    model = Model(
        small.with_(population_rule="household_size", urban_return_rate=1.0), artifact=artifact
    )
    model.step()
    movers = [int(j) for j in model.people.in_states(OCCUPIED)[:5]]
    for j in movers:
        model.people.state[j] = int(Tenure.EXITED)
        model.people.need[j] = 0.5
    model.wage = 5.0  # the land is now plainly worth more than the town
    before = model.people.total_head_count()
    model._step4b_urban()
    assert all(model.people.state[j] == int(Tenure.LANDLESS) for j in movers)
    assert model.n_returns == len(movers)
    assert model.people.total_head_count() == before, "migration must not create people"


def test_return_migration_scales_with_the_advantage(small, artifact):
    """A vanishing advantage must not move as many households as a decisive one."""
    counts = []
    for wage in (0.85, 5.0):
        model = Model(small.with_(population_rule="household_size"), artifact=artifact)
        model.step()
        for j in model.people.in_states(OCCUPIED)[:400]:
            model.people.state[int(j)] = int(Tenure.EXITED)
            model.people.need[int(j)] = 0.8
        model.wage = wage
        model._step4b_urban()
        counts.append(model.n_returns)
    assert counts[1] > counts[0], f"returns did not respond to the pull: {counts}"


def test_exit_to_industry_happens_at_all(demographic):
    assert sum(h["exits_to_industry"] for h in demographic.history) > 0


def test_urban_return_switch_closes_the_valve(small, artifact):
    model = Model(
        small.with_(n_steps=80, population_rule="household_size", urban_return=False),
        artifact=artifact,
    ).run()
    assert sum(h["returns_from_industry"] for h in model.history) == 0


def test_households_are_larger_than_one_person(demographic):
    """The point of the rule: the household is a unit with members, not a single body."""
    sizes = [h["mean_household_size"] for h in demographic.history]
    assert max(sizes) > 1.0, "household size never grew above one member"
    assert all(s >= 1.0 for s in sizes), "a household cannot have fewer than one member"
    final = demographic.history[-1]
    assert final["population"] >= final["households"]


def test_landless_households_reproduce_and_die(small, artifact):
    """Both directions must apply to the landless, or proletarianisation is self-sterilising.

    A population whose landless cannot reproduce shrinks by construction as it proletarianises,
    which was the defect of the superseded ``household`` rule.
    """
    model = Model(
        small.with_(n_steps=80, population_rule="household_size"), artifact=artifact
    )
    landless_sizes = []
    for _ in range(80):
        model.step()
        ppl = model.people
        pool = ppl.in_states((Tenure.LANDLESS,))
        if len(pool):
            landless_sizes.append(float(ppl.size[pool].max()))
    assert landless_sizes, "no landless households ever existed"
    assert max(landless_sizes) > 1, "no landless household ever grew"
    assert sum(h["deaths"] for h in model.history) > 0


def test_partition_sheds_members_into_the_proletariat(demographic):
    partitions = sum(h["partitions"] for h in demographic.history)
    shed = sum(h["partition_persons"] for h in demographic.history)
    assert partitions > 0, "no household ever shed a member"
    assert shed >= partitions


def test_partition_conserves_persons(small, artifact):
    """Shedding a member moves it; it must neither create nor destroy anyone."""
    model = Model(small.with_(population_rule="household_size"), artifact=artifact)
    parent = int(model.people.in_states(OCCUPIED)[0])
    model.people.size[parent] = 4
    model.people.w[parent] = 50.0
    before = model.people.head_count()
    child = model._shed_member(parent)
    assert child >= 0
    assert model.people.size[parent] == 3
    assert model.people.size[child] == 1
    assert model.people.state[child] == int(Tenure.LANDLESS)
    assert model.people.head_count() == before


def test_impartible_inheritance_disinherits(small, artifact):
    """The strong proletarianisation pump: all but the heir leave at conveyance."""
    model = Model(
        small.with_(
            n_steps=80, population_rule="household_size", impartible_inheritance=True
        ),
        artifact=artifact,
    ).run()
    assert sum(h["disinherited"] for h in model.history) > 0
    baseline = Model(
        small.with_(n_steps=80, population_rule="household_size"), artifact=artifact
    ).run()
    assert sum(h["disinherited"] for h in baseline.history) == 0


def test_conveyance_is_population_neutral(small, artifact):
    """``eta_0`` is a conveyance hazard, not a death: the heir inherits the household whole.

    If conveyance also removed a member it would double-count with ``mortality_0``, and the
    population would drain at the sum of the two rates.
    """
    model = Model(small.with_(population_rule="household_size"), artifact=artifact)
    model.step()
    person = int(model.people.in_states((Tenure.CUSTOMARY,))[0])
    model.people.size[person] = 5
    before = model.people.head_count()
    model._resolve_succession(person)
    assert model.people.head_count() == before


def test_extinct_household_frees_its_land(small, artifact):
    """A line that dies out must not hold its parcels for ever."""
    model = Model(small.with_(population_rule="household_size"), artifact=artifact)
    model.step()
    person = int(model.people.in_states(OCCUPIED)[0])
    parcels = sorted(model.people.holdings[person])
    model.people.size[person] = 0
    model._extinct = [person]
    model._step8_resolve(model._step7_vacancies())
    assert model.people.state[person] == int(Tenure.DECEASED)
    for parcel in parcels:
        assert model.occupant[parcel] != person
    assert not model.people.holdings[person]


def test_land_can_still_be_staffed(demographic):
    """The regression test for the ratchet.

    Under the closed population, exit was absorbing and nothing replaced the departed, so a
    long run ended with thousands of parcels that no living agent could take up. Whatever else
    the demography does, the countryside must not run out of people to work it.
    """
    vacant = [h["share_parcels_vacant"] for h in demographic.history]
    assert max(vacant) < 0.5, f"up to {max(vacant):.0%} of parcels had no occupant"
    first, last = demographic.history[0], demographic.history[-1]
    assert last["population"] > 0.25 * first["population"], "population collapsed"


def test_exit_hazard_is_smoother_than_the_counter(small, artifact):
    """Option 1: a global wage plus a shared deterministic threshold exits whole cohorts at once.

    Compared at equal total outflow, the hazard rule must not concentrate departures into a few
    periods the way the counter does.
    """
    base = small.with_(n_steps=120, population_rule="household_size")
    counter = Model(base.with_(exit_rule="counter", landless_consumption_spread=0.0),
                    artifact=artifact).run()
    hazard = Model(base.with_(exit_rule="hazard"), artifact=artifact).run()

    def peak_share(model):
        flow = np.array([h["exit_persons"] for h in model.history], dtype=float)
        total = flow.sum()
        if total <= 0:
            pytest.skip("no exits under this parameterisation")
        return float(np.sort(flow)[-5:].sum() / total)

    assert peak_share(hazard) < peak_share(counter)


def test_landless_requirements_are_heterogeneous(small, artifact):
    """The other half of option 1: identical needs put every household on one knife-edge."""
    model = Model(small.with_(landless_consumption_spread=0.3), artifact=artifact)
    needs = model.people.need[: model.people.n]
    assert needs.std() > 0.0
    assert needs.min() >= 0.7 * small.landless_consumption
    assert needs.max() <= 1.3 * small.landless_consumption
    flat = Model(small.with_(landless_consumption_spread=0.0), artifact=artifact)
    assert np.ptp(flat.people.need[: flat.people.n]) == 0.0


def test_size_is_both_hands_and_mouths(small, artifact):
    """A member raises output and costs subsistence, or household size would be a free lunch."""
    params = small.with_(population_rule="household_size", customary_fine_rate=0.0)

    # Hands: on identical land and capital, the larger family produces more.
    outputs = {}
    for size in (1, 6):
        model = Model(params, artifact=artifact)
        person = int(model.people.in_states((Tenure.CUSTOMARY,))[0])
        model.people.size[person] = size
        model.step()
        outputs[size] = float(model.people.y[person])
    assert outputs[6] > outputs[1], "more hands did not raise output"

    # Mouths: with nothing to harvest, wealth falls by exactly rent plus per-head subsistence.
    model = Model(params, artifact=artifact)
    person = int(model.people.in_states((Tenure.CUSTOMARY,))[0])
    model.people.size[person] = 4
    model.phi[:] = 0.0  # no land quality, so no output whatever the labour
    before = float(model.people.w[person])
    model.step()
    charged = before - float(model.people.w[person])
    expected = 4 * params.subsistence_per_head + float(model.people.rho[person])
    assert charged == pytest.approx(expected, rel=1e-9)


def test_capital_depreciates_without_reinvestment(small, artifact):
    """A tenant who cannot reinvest must not keep improvements for free."""
    model = Model(small.with_(n_steps=5), artifact=artifact)
    person = int(model.people.in_states((Tenure.CUSTOMARY,))[0])
    model.people.k[person] = 100.0
    model.people.iota[person] = 0.0
    before = model.people.k[person]
    model.step()
    assert model.people.k[person] < before


def test_wage_stays_bounded(finished):
    wages = [h["wage"] for h in finished.history]
    assert all(np.isfinite(w) and w > 0 for w in wages)
    assert max(wages) < 1e5, "wage diverged; the labour market is not clearing"


def test_normalised_pressure_is_scale_free(small, artifact):
    """Relative pressure must not simply track estate size."""
    model = Model(small.with_(n_steps=30), artifact=artifact).run()
    sizes = np.array([len(e) for e in model.geo.estate_parcels], dtype=float)
    if np.ptp(sizes) == 0:
        pytest.skip("estates are all the same size")
    correlation = abs(float(np.corrcoef(sizes, model.delta_relative)[0, 1]))
    assert correlation < 0.98, "relative pressure is still essentially estate size"


def test_productive_neighbour_outbids_a_landless_entrant(small, artifact):
    """Wood's "success would breed success": the improver can outbid the entrant next door."""
    model = Model(small.with_(n_steps=60), artifact=artifact).run()
    ppl = model.people
    exposed = [j for j in ppl.in_states(MARKET_EXPOSED) if ppl.holdings[j] and ppl.y[j] > 0]
    landless = ppl.in_states((Tenure.LANDLESS,))
    if not exposed or len(landless) == 0:
        pytest.skip("need both a producing tenant and a landless agent")
    best = max(exposed, key=lambda j: ppl.y[j] / len(ppl.holdings[j]))
    parcel = next(iter(ppl.holdings[best]))
    assert model._bid(best, parcel) > model._bid(int(landless[0]), parcel)


def test_competitive_rents_are_struck_per_parcel(small, artifact):
    model = Model(small.with_(n_steps=80), artifact=artifact).run()
    leasehold_parcels = model.parcel_tenure == int(Tenure.LEASEHOLD)
    if not leasehold_parcels.any():
        pytest.skip("no leasehold parcels arose")
    rents = model.parcel_rent[leasehold_parcels]
    assert (rents > 0).any(), "auctioned parcels should carry a struck rent"
    ppl = model.people
    for j in ppl.in_states((Tenure.LEASEHOLD,)):
        if ppl.holdings[j]:
            expected = sum(model.parcel_rent[k] for k in ppl.holdings[j])
            assert model._rent_for(j) == pytest.approx(expected)


def test_competitive_allocation_can_be_disabled(small, artifact):
    model = Model(
        small.with_(n_steps=40, competitive_allocation=False), artifact=artifact
    ).run()
    assert len(model.history) == 40


def test_concentration_frame_is_wellformed(finished):
    frame = concentration_frame(finished)
    assert not frame.empty
    assert ((frame["gini"] >= 0) & (frame["gini"] <= 1)).all()
    assert ((frame["top_decile_land_share"] >= 0) & (frame["top_decile_land_share"] <= 1)).all()
    assert ((frame["single_parcel_share"] >= 0) & (frame["single_parcel_share"] <= 1)).all()
    assert frame["mean_holding"].min() >= 1.0
    # At t=0 every tenant holds exactly one parcel, so there is no concentration yet.
    assert frame.iloc[0]["gini"] == pytest.approx(0.0, abs=1e-9)
    assert frame.iloc[0]["single_parcel_share"] == pytest.approx(1.0)


def test_holding_sizes_count_tenants_not_parcels(finished):
    """A five-parcel farm must appear once, not five times."""
    t = finished.p.n_steps - 1
    sizes = holding_sizes_at(finished, t)
    occupied = int((finished.panel_occupant[t] >= 0).sum())
    assert sizes.sum() == pytest.approx(occupied)
    distinct = len(set(finished.panel_occupant[t][finished.panel_occupant[t] >= 0]))
    assert len(sizes) == distinct


def test_lorenz_is_monotone_and_bounded():
    pop, land = lorenz(np.array([1.0, 1.0, 5.0, 20.0]))
    assert pop[0] == 0.0 and pop[-1] == pytest.approx(1.0)
    assert land[0] == 0.0 and land[-1] == pytest.approx(1.0)
    assert np.all(np.diff(land) >= -1e-12)
    # Concentrated holdings must sit below the line of equality.
    assert (land <= pop + 1e-12).all()


def test_consolidation_by_fertility_covers_every_group(finished):
    frame = consolidation_by_fertility(finished, n_groups=4)
    assert not frame.empty
    assert set(frame["fertility_group"]) == {"Q1", "Q2", "Q3", "Q4"}
    # Groups must be ordered poorest to best.
    means = frame.groupby("group_index")["mean_fertility"].first()
    assert means.is_monotonic_increasing
    assert frame["mean_holding_size"].min() >= 1.0


def test_region_summary_matches_geography(finished):
    frame = region_consolidation_summary(finished)
    if frame.empty:
        pytest.skip("no region large enough to characterise")
    assert frame["n_parcels"].min() >= 5
    assert frame["final_mean_holding"].min() >= 1.0
    assert frame["mean_fertility"].between(0, 30).all()


def test_rent_and_revenue_share_units(small, artifact):
    """Rent is struck in money, so it must be comparable with priced revenue.

    Regression test: when the produce price was introduced, bids stayed denominated in
    produce while revenue was priced, so a price of 0.2 charged tenants five times too much.
    """
    model = Model(small.with_(n_steps=60), artifact=artifact).run()
    ppl = model.people
    leaseholders = [j for j in ppl.in_states((Tenure.LEASEHOLD,)) if ppl.holdings[j]]
    if not leaseholders:
        pytest.skip("no leasehold tenancies arose")
    price = model._price()
    for j in leaseholders[:20]:
        parcel = next(iter(ppl.holdings[j]))
        # A fresh bid on this parcel must scale with the price, not sit in produce units.
        model.goods_price = price * 2
        doubled = model._bid(j, parcel)
        model.goods_price = price
        base = model._bid(j, parcel)
        if base > 0:
            assert doubled == pytest.approx(2 * base)


def test_price_is_neutral_when_domestic_market_is_off(small, artifact):
    model = Model(small.with_(n_steps=20, urban_demand=False), artifact=artifact).run()
    assert model._price() == 1.0


def test_reproduction_rule_responds_to_wealth_not_income(small, artifact):
    """A lord with reserves feels no urgency; a lord without does, at identical income."""
    model = Model(small.with_(n_steps=10, ideology_rule="reproduction"), artifact=artifact)
    model.run()
    assert hasattr(model, "landlord_pressure")
    press = model.landlord_pressure
    runway = model.W / np.maximum(model.consumption, 1e-9)
    if np.ptp(runway) == 0:
        pytest.skip("no variation in reserves")
    # Pressure must fall as the cushion grows.
    assert float(np.corrcoef(runway, press)[0, 1]) < 0.0


def test_ideology_rules_are_all_runnable(small, artifact):
    for rule in ("reproduction", "material", "contagion"):
        model = Model(
            small.with_(n_steps=25, ideology_rule=rule), artifact=artifact
        ).run()
        iota = model.iota_landlord
        assert np.isfinite(iota).all()
        assert ((iota >= 0) & (iota <= 1)).all()


def test_urban_demand_creates_a_market(small, artifact):
    """Households leaving the land must show up as demand rather than simply vanishing.

    The urban count is a *stock*, not a cumulative total: it rises with departures and falls
    with returns and urban mortality, so it is not monotone. What must hold is that it is
    populated, and that the demand it carries is priced.
    """
    model = Model(small.with_(n_steps=80), artifact=artifact).run()
    urban = [h["urban_population"] for h in model.history]
    assert max(urban) > 0, "nobody left the land, so the channel is untested"
    for row in model.history:
        assert row["urban_demand"] == pytest.approx(
            row["urban_population"] * model.p.urban_consumption
        )
    assert len({h["goods_price"] for h in model.history}) > 1, "the price never responded"


def test_urban_population_is_a_stock_not_a_cumulative_count(small, artifact):
    """Guards the distinction the summary table depends on."""
    model = Model(small.with_(n_steps=80), artifact=artifact).run()
    cumulative = sum(h["exits_to_industry"] for h in model.history)
    assert cumulative > model.history[-1]["exited"], (
        "with two-way migration and urban mortality the standing urban household count must "
        "be smaller than everyone who ever left"
    )


def test_runs_are_reproducible(small, artifact):
    a = Model(small, artifact=artifact).run()
    b = Model(small, artifact=artifact).run()
    assert a.history[-1] == b.history[-1]
    assert np.array_equal(a.first_conversion, b.first_conversion)


def test_different_seeds_diverge(small, artifact):
    a = Model(small.with_(seed=1), artifact=artifact).run()
    b = Model(small.with_(seed=2), artifact=artifact).run()
    assert a.history[-1] != b.history[-1]


def test_metrics_are_wellformed(finished):
    frame = history_frame(finished)
    assert len(frame) == finished.p.n_steps
    assert frame["t"].is_monotonic_increasing

    stats = summary(finished)
    assert 0.0 <= stats["conversion_share"] <= 1.0
    assert 0.0 <= stats["final_farm_gini"] <= 1.0

    spread = spread_variogram(finished)
    assert set(spread) == {
        "n", "n_pairs", "slope_norm", "nugget_share", "plateau_share",
        "range_cells", "r", "total_variance",
    }

    curve = spread_variogram_curve(finished)
    if not curve.empty:
        # Monotonically increasing bin centres, and every pair counted exactly once.
        assert curve["distance"].is_monotonic_increasing
        assert curve["n_pairs"].sum() == spread["n_pairs"]

    events = event_study(finished, window=10)
    if not events.empty:
        assert set(events["metric"]) <= {"iota", "capital", "rent"}
        assert events["event_time"].min() == -10


@pytest.mark.parametrize(
    "values,expected",
    [([1, 1, 1, 1], 0.0), ([], 0.0), ([0, 0], 0.0)],
)
def test_gini_edge_cases(values, expected):
    assert _gini(np.array(values, dtype=float)) == pytest.approx(expected, abs=1e-9)


def test_gini_increases_with_concentration():
    equal = _gini(np.array([2.0, 2.0, 2.0, 2.0]))
    skewed = _gini(np.array([1.0, 1.0, 1.0, 5.0]))
    assert skewed > equal


def test_params_reject_invalid_production_exponents():
    with pytest.raises(ValueError):
        Params(cobb_phi=0.7, cobb_capital=0.5)


# --- the per-parcel frames --------------------------------------------------------------
# These carry the only spatial record a run leaves behind, so what is tested here is not the
# numbers but the two structural properties the map figures rely on: one row per parcel in
# lattice order, and a clean split between what varies across seeds and what cannot.
def test_parcel_frame_is_one_row_per_parcel_in_order(finished):
    frame = parcel_frame(finished)
    assert len(frame) == finished.geo.n_parcels
    # Positional, not just present: every map joins this against geography on `parcel`, and a
    # reordering would silently draw the right values in the wrong places.
    assert frame["parcel"].tolist() == list(range(finished.geo.n_parcels))
    assert dict(frame.dtypes.astype(str)) == PARCEL_DTYPES


def test_parcel_frame_marks_unconverted_land_as_censored(finished):
    frame = parcel_frame(finished)
    # -1 rather than NaN, and it means "had not converted by the end of the run" rather than
    # "unknown": a mean of this column is meaningless, which is why the maps take the share of
    # seeds converted instead.
    never = frame["first_conversion"] < 0
    assert (frame.loc[never, "first_conversion"] == -1).all()
    within = frame.loc[~never, "first_conversion"]
    assert within.between(0, finished.p.n_steps - 1).all()


def test_parcel_frame_tenure_columns_are_distinct_quantities(finished):
    frame = parcel_frame(finished)
    # parcel_tenure attaches to the land and ratchets; final_state is the occupant's, and -1
    # where nobody holds it. Conflating them would make a vacant converted parcel read as
    # unconverted.
    assert (frame["parcel_tenure"] >= 0).all()
    vacant = frame["final_state"] < 0
    if vacant.any():
        assert (frame.loc[vacant, "final_state"] == -1).all()
        assert frame.loc[vacant, "final_iota"].isna().all()


def test_geography_frame_is_invariant_across_seeds(small, artifact):
    """The property that licenses averaging a parcel's outcome over seeds.

    A shared lattice is what makes parcel *n* the same land in every replicate, so the runner
    stores this once per arm. If it ever became seed-dependent, every cross-seed map and every
    paired difference map would be comparing different places.
    """
    from pmabm.geography import build as build_geography

    geo = build_geography(small, np.random.default_rng(0), artifact=artifact)
    a = Model(small.with_(seed=0), geography=geo, artifact=artifact).run()
    b = Model(small.with_(seed=7), geography=geo, artifact=artifact).run()
    left, right = geography_frame(a.geo), geography_frame(b.geo)
    assert left.equals(right)
    assert len(left) == geo.n_parcels
    assert left["parcel"].tolist() == list(range(geo.n_parcels))


def test_commons_is_seed_dependent_and_so_lives_in_the_parcel_frame(small, artifact):
    """Why `commons` is not in the geography frame despite looking like a fact about the land.

    It is drawn from the model's own generator at construction, so it differs between seeds even
    on a shared lattice. Storing it as geography would freeze one seed's draw and quietly apply
    it to every replicate.
    """
    from pmabm.geography import build as build_geography

    geo = build_geography(small, np.random.default_rng(0), artifact=artifact)
    a = Model(small.with_(seed=0), geography=geo, artifact=artifact).run()
    b = Model(small.with_(seed=7), geography=geo, artifact=artifact).run()
    assert not np.array_equal(a.commons, b.commons)
    assert "commons" in parcel_frame(a).columns
    assert "commons" not in geography_frame(geo).columns


def test_random_awareness_graph_keeps_degree_and_discards_geometry(small, artifact):
    """RQ2's measurement control: same amount of contagion, no geography.

    Degree preservation is the load-bearing part. Rewiring to a *fixed* degree would change how
    much contagion there is as well as where it goes, and the arm exists to vary only the second --
    so a flat variogram under it would then have two possible causes instead of one.
    """
    from pmabm.geography import build as build_geography

    spatial = build_geography(small, np.random.default_rng(0), artifact=artifact)
    rewired = build_geography(
        small.with_(random_awareness_graph=True), np.random.default_rng(0), artifact=artifact
    )
    assert [len(n) for n in spatial.landlord_neighbours] == [
        len(n) for n in rewired.landlord_neighbours
    ]
    # No self-loops, and the targets are genuinely different from the spatial ones somewhere.
    assert all(i not in n for i, n in enumerate(rewired.landlord_neighbours))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(spatial.landlord_neighbours, rewired.landlord_neighbours)
    )
    # Everything else about the lattice is untouched, so the arm differs in one thing only.
    assert np.array_equal(spatial.phi_bar, rewired.phi_bar)
    assert np.array_equal(spatial.landlord, rewired.landlord)


def test_county_history_shares_are_shares_of_land(finished):
    """The county frame's denominator is occupied parcels, not tenancies.

    Distinct from :func:`history_frame` on purpose -- a choropleth should show how much of a
    county is under leasehold, not what fraction of its tenants hold by lease -- and the two
    diverge exactly where consolidation puts more land in fewer leasehold farms.
    """
    from pmabm.metrics import county_history_frame

    frame = county_history_frame(finished)
    assert len(frame) == frame["county"].nunique() * finished.p.n_steps
    tenure = frame[["share_customary", "share_leasehold", "share_freehold"]].sum(
        axis=1, skipna=True
    )
    assert float(np.nanmax(tenure)) <= 1.0 + 1e-6
    # share_converted is cumulative and taken from first_conversion, so it can only rise -- which
    # is what makes a map of it over time a front rather than a set of unrelated snapshots.
    for _, block in frame.sort_values("t").groupby("county"):
        assert block["share_converted"].is_monotonic_increasing


# --- enclosure: national by default, local by switch ------------------------------------
def test_national_enclosure_is_uniform_across_estates(small, artifact):
    """The default rule is the paper's: one date everywhere, so no geography to compare.

    Also the regression guard on the per-estate refactor. The hazard now reads a household's own
    lord's value rather than a scalar, and under the national rule every lord must carry the same
    number for that to be the identical calculation.
    """
    model = Model(small, artifact=artifact).run()
    assert model.p.enclosure_rule == "national"
    assert np.allclose(model.enclosure_by_estate, model.enclosure_by_estate[0])
    assert model.enclosure == pytest.approx(float(model.enclosure_by_estate[0]))
    # One half-time, or none if the run was too short to reach a half.
    reached = model.enclosure_half_time[model.enclosure_half_time >= 0]
    assert len(np.unique(reached)) <= 1


def test_local_enclosure_gives_estates_their_own_front(small, artifact):
    """The variant exists to give RQ10 a geography; this is that it actually has one."""
    model = Model(small.with_(enclosure_rule="local"), artifact=artifact).run()
    spread = model.enclosure_by_estate
    assert spread.std() > 0.0
    assert spread.min() >= 0.0 and spread.max() <= 1.0
    # The aggregate stays a share of land, so it lies inside the per-estate range.
    assert spread.min() - 1e-9 <= model.enclosure <= spread.max() + 1e-9


def test_local_enclosure_leaves_the_national_pace_broadly_alone(small, artifact):
    """Redistributing a process is not the same as accelerating it.

    The point of the local rule is to move enclosure around, not to change how much of it there
    is: a variant that also enclosed far more land would confound RQ10's geography question with a
    level effect. Loose bound, because the two rules are genuinely different dynamics -- a convex
    diffusion driven by local means need not aggregate to the one driven by the global mean.
    """
    national = Model(small, artifact=artifact).run().enclosure
    local = Model(small.with_(enclosure_rule="local"), artifact=artifact).run().enclosure
    assert local == pytest.approx(national, abs=0.15)


def test_enclosure_rule_is_validated():
    with pytest.raises(ValueError, match="enclosure_rule"):
        Params(enclosure_rule="regional")


def test_parcel_frame_carries_each_parcels_own_enclosure(small, artifact):
    """What makes the front comparison possible: enclosure exposure recorded per place."""
    from pmabm.metrics import parcel_frame

    model = Model(small.with_(enclosure_rule="local"), artifact=artifact).run()
    frame = parcel_frame(model)
    assert frame["final_enclosure"].std() > 0.0
    # Every parcel of one estate shares that estate's value, which is what "per estate" means.
    for landlord in np.unique(model.geo.landlord)[:5]:
        block = frame.loc[model.geo.landlord == landlord, "final_enclosure"]
        assert block.nunique() == 1
