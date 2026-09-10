"""Agent state and the per-period schedule.

Implements the paper's Model Design section. The ten numbered steps of :meth:`Model.step`
correspond one-to-one with the paper's Model schedule subsection, and every quantity is
computed once at the start of a period from state as of ``t`` and held fixed while
within-period events resolve -- the synchronous-update convention the paper specifies, so that
two vacancies on the same estate see the same fiscal pressure.

Places where the paper is silent and this implementation had to choose are marked
``SPEC NOTE``; they are collected in the project README.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import Params
from .geography import Geography, build as build_geography


class Tenure(IntEnum):
    """Tenure state sigma_j(t). ``EXITED`` is outside the paper's four states: it marks a
    landless agent who has left the rural model entirely for industry."""

    CUSTOMARY = 0
    LEASEHOLD = 1
    FREEHOLD = 2
    LANDLESS = 3
    EXITED = 4
    """Left the rural model for industry -- the outflow Brenner's argument turns on."""
    DECEASED = 5
    """Died and was succeeded by an heir. Kept distinct from EXITED so that "share who left
    the land for industry" is not inflated by ordinary mortality."""


#: States in which a person is no longer part of the rural population.
GONE = (Tenure.EXITED, Tenure.DECEASED)


OCCUPIED = (Tenure.CUSTOMARY, Tenure.LEASEHOLD, Tenure.FREEHOLD)
#: The same states as raw ints, for membership tests in hot loops.
OCCUPIED_INTS = frozenset(int(s) for s in OCCUPIED)
#: States whose occupants reinvest and hire, i.e. are exposed to the market.
MARKET_EXPOSED = (Tenure.LEASEHOLD, Tenure.FREEHOLD)


@dataclass
class People:
    """Structure-of-arrays store for every household ever created, grown by doubling.

    A single store covers all tenure states including Landless, because the paper models the
    tenant as one agent type whose tenure state evolves rather than as separate classes.

    The unit is the *household*, not the person: ``size`` holds how many members it has, and
    labour supply and consumption both scale with it. Under the single-body population rules
    every ``size`` is 1 and every expression involving it collapses to the original one.
    """

    capacity: int = 4096
    n: int = 0
    state: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    w: np.ndarray = field(default_factory=lambda: np.empty(0))
    k: np.ndarray = field(default_factory=lambda: np.empty(0))
    iota: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Tenant improving disposition iota^T_j(t)."""
    shortfall: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    mobility: np.ndarray = field(default_factory=lambda: np.empty(0))
    landlord: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    rho: np.ndarray = field(default_factory=lambda: np.empty(0))
    hired: np.ndarray = field(default_factory=lambda: np.empty(0))
    size: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    """n_j: members in the household. Labour supplied is ``size * household_labour`` and
    subsistence owed is ``size *`` the per-head requirement."""
    need: np.ndarray = field(default_factory=lambda: np.empty(0))
    """This household's own per-head subsistence cost while landless. Drawn once, so that a
    global wage does not put every landless household on an identical knife-edge."""
    uid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """Identity of the household in this slot, never reused.

    A slot is recycled once its household is dead, so the slot index no longer identifies a
    household across time. Anything asking "is this the same tenant as last period" -- which is
    exactly what RQ1's event study asks -- has to compare this instead."""
    freed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    """Whether the slot is already on the free list, so reclaiming is idempotent."""
    holdings: list[set[int]] = field(default_factory=list)
    n_held: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    """``len(holdings[i])``, maintained alongside the set.

    The event-study panel wants every occupant's holding size once a period and the
    engrossment scan wants a candidate's on every neighbour of every released parcel, which
    together came to some three million ``len()`` calls a run -- the single largest block of
    interpreter overhead in the recording step. The three places that mutate ``holdings``
    (:meth:`add`, ``Model._release`` and ``Model._assign``) are the only places that touch
    this, and ``test_n_held_tracks_holdings`` pins the two together.

    Not the same quantity as ``_aggregate_holdings()["size"]``, which counts parcels whose
    ``occupant`` points at the household: the two diverge whenever a parcel changes hands
    without its former holder being released.
    """
    _free: list[int] = field(default_factory=list)
    _next_uid: int = 0
    _landless_hint: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    _landless_count: int = 0
    _best_w: int = -1
    _best_size: int = -1
    _best_valid: bool = False

    _FIELDS = (
        ("state", np.int8, 0),
        ("w", float, 0.0),
        ("k", float, 0.0),
        ("iota", float, 0.0),
        ("shortfall", np.int32, 0),
        ("mobility", float, 0.0),
        ("landlord", np.int32, -1),
        ("y", float, 0.0),
        ("rho", float, 0.0),
        ("hired", float, 0.0),
        ("size", np.int32, 1),
        ("need", float, 0.0),
        ("uid", np.int64, -1),
        ("freed", bool, False),
        ("n_held", np.int32, 0),
    )

    def __post_init__(self) -> None:
        for name, dtype, fill in self._FIELDS:
            setattr(self, name, np.full(self.capacity, fill, dtype=dtype))
        self.holdings = []
        self._landless_hint = np.empty(1024, dtype=np.intp)
        self._landless_count = 0
        self._free = []
        self._next_uid = 0
        self._best_w = -1
        self._best_size = -1
        self._best_valid = False

    def _grow(self) -> None:
        self.capacity *= 2
        for name, dtype, fill in self._FIELDS:
            old = getattr(self, name)
            new = np.full(self.capacity, fill, dtype=dtype)
            new[: len(old)] = old
            setattr(self, name, new)

    def add(
        self,
        state: Tenure,
        w: float,
        k: float,
        iota: float,
        mobility: float,
        landlord: int,
        holding: set[int] | None = None,
        size: int = 1,
        need: float = 0.0,
    ) -> int:
        if self._free:
            i = self._free.pop()
            reused = True
        else:
            if self.n >= self.capacity:
                self._grow()
            i = self.n
            reused = False
        self.uid[i] = self._next_uid
        self._next_uid += 1
        self.freed[i] = False
        self.state[i] = int(state)
        self.w[i] = w
        self.k[i] = k
        self.iota[i] = iota
        self.shortfall[i] = 0
        self.mobility[i] = mobility
        self.landlord[i] = landlord
        self.y[i] = 0.0
        self.rho[i] = 0.0
        self.hired[i] = 0.0
        self.size[i] = int(size)
        self.need[i] = need
        fresh = set(holding) if holding else set()
        if reused:
            self.holdings[i] = fresh
        else:
            self.holdings.append(fresh)
            self.n += 1
        self.n_held[i] = len(fresh)
        if int(state) == int(Tenure.LANDLESS):
            self.mark_landless(i)
        return i

    def reclaim(self) -> None:
        """Return the slots of households that died in earlier periods to the free list.

        Without this the store only grows: a 200-period run ends with some 80,000 slots of
        which about 83% are tombstones, and every vectorised pass over the population pays for
        all of them. Recycling holds it near the size of the live population instead.

        Keyed off ``DECEASED`` state rather than called wherever a death is recorded, because a
        slot must not be reused while its household still holds land -- a line that dies in
        step 4c keeps its parcels until step 8 resolves them, and step 7 still has to read it
        out of ``_extinct``. Running this at the top of the *following* period is what makes
        that safe, and keying it off state rather than off call sites is what keeps it safe
        when a new death path is added.
        """
        n = self.n
        dead = np.nonzero((self.state[:n] == int(Tenure.DECEASED)) & ~self.freed[:n])[0]
        if dead.size:
            self.freed[dead] = True
            self._free.extend(dead.tolist())

    def refresh_landless(self) -> None:
        """Rebuild the landless hint from state, once a period, so it stays tight instead of
        accumulating everyone who has ever been landless."""
        idx = np.nonzero(self.state[: self.n] == int(Tenure.LANDLESS))[0]
        if idx.size > self._landless_hint.size:
            self._landless_hint = np.empty(max(idx.size * 2, 1024), dtype=np.intp)
        self._landless_hint[: idx.size] = idx
        self._landless_count = int(idx.size)
        # Invalidate rather than rebuild: wealth and household size are still going to move in
        # steps 1 to 7, and the extremes are not wanted until step 8 lets the first parcel.
        self._best_valid = False

    def mark_landless(self, i: int) -> None:
        """Note that ``i`` has become landless. The hint is kept sorted so a tie in the
        auction breaks on the lowest index, exactly as a full ascending scan would."""
        i = int(i)
        # Fold into the cached extremes first, and before the insertion below can return
        # early: a recycled slot arrives here holding a *different* household from the one its
        # surviving hint entry was made for, so "already in the hint" says nothing about
        # whether the extremes are still right.
        if self._best_usable():
            if i == self._best_w or i == self._best_size:
                # The slot of a cached extreme is being marked. Under slot recycling the
                # household there may be a new one whose wealth says nothing about the old
                # maximum, and no constant-time repair can tell the two cases apart, so the
                # cache goes.
                self._best_valid = False
            else:
                # A newcomer that beats the standing best beats every earlier member of the
                # pool too, so this keeps :meth:`landless_best` exact without a rescan. Ties
                # go to the lower index, which is how a full ascending argmax breaks them.
                w, size, bw, bs = self.w, self.size, self._best_w, self._best_size
                if bw < 0 or w[i] > w[bw] or (w[i] == w[bw] and i < bw):
                    self._best_w = i
                if bs < 0 or size[i] > size[bs] or (size[i] == size[bs] and i < bs):
                    self._best_size = i

        hint = self._landless_hint
        count = self._landless_count
        pos = int(np.searchsorted(hint[:count], i))
        if pos < count and hint[pos] == i:
            return
        if count == hint.size:
            bigger = np.empty(hint.size * 2, dtype=np.intp)
            bigger[:count] = hint[:count]
            self._landless_hint = hint = bigger
        hint[pos + 1 : count + 1] = hint[pos:count]
        hint[pos] = i
        self._landless_count = count + 1

    def landless_pool(self) -> np.ndarray:
        """Indices currently Landless, ascending.

        The auction asks for this once per vacancy -- some hundreds of times a period -- and
        the landless are a few per cent of a store that is mostly deceased tombstones, so
        rescanning the whole store each time dominated the schedule. The hint is a sorted
        superset maintained across the period and filtered against live state here, which
        returns the same array in the same order for a fraction of the work.
        """
        sub = self._landless_hint[: self._landless_count]
        return sub[self.state[sub] == int(Tenure.LANDLESS)]

    def _best_usable(self) -> bool:
        """Whether the cached extremes can still be trusted, dropping them if they cannot.

        An extreme stops being one the moment it is let a holding, and it has to be caught
        here rather than only on the next read: ``_install`` debits the winner's entry fine on
        the way out, so a cached index that has left the pool no longer even carries the wealth
        it was cached for, and folding a newcomer against it would compare against a number
        that belongs to nobody in the auction.
        """
        if not self._best_valid:
            return False
        landless = int(Tenure.LANDLESS)
        bw, bs = self._best_w, self._best_size
        if (bw < 0 or self.state[bw] == landless) and (
            bs < 0 or self.state[bs] == landless
        ):
            return True
        self._best_valid = False
        return False

    def landless_best(self) -> tuple[int, int]:
        """The wealthiest and the largest Landless household, or ``(-1, -1)`` if there are none.

        These two are all the auction ever wants from the pool -- landless households differ
        only in wealth and in how many hands they bring -- yet finding them by argmax over the
        whole pool at every one of some 28,000 vacancies a run meant seven passes over five
        thousand entries each time, around a fifth of the schedule.

        Held instead as two cached indices, rebuilt only when they go stale. Staleness has
        exactly two causes and both are covered: a new household joining the pool is folded in
        by :meth:`mark_landless` in constant time, and the cached best being let a holding is
        caught by the state check below. Losing any *other* member cannot move a maximum, and a
        member's own wealth and size do not change between the first vacancy of a period and
        the last -- everything that moves them runs in steps 1 to 7, which is why
        :meth:`refresh_landless` invalidates instead of rebuilding.
        """
        if self._best_usable():
            return self._best_w, self._best_size

        pool = self.landless_pool()
        # The hint is deliberately *not* compacted onto ``pool`` here. A slot that has dropped
        # out of the pool can re-enter it without passing through :meth:`mark_landless` -- step
        # 4b sends returning urban households straight back to Landless in one vectorised
        # write -- and it is its surviving hint entry that lets it back into the auction.
        # Dropping stale entries would quietly change who can be let a holding.
        if pool.size == 0:
            self._best_w = self._best_size = -1
        else:
            self._best_w = int(pool[np.argmax(self.w[pool])])
            self._best_size = int(pool[np.argmax(self.size[pool])])
        self._best_valid = True
        return self._best_w, self._best_size

    def live(self) -> np.ndarray:
        """Indices of every household still present in the rural population."""
        state = self.state[: self.n]
        return np.nonzero(
            (state != int(Tenure.EXITED)) & (state != int(Tenure.DECEASED))
        )[0]

    def head_count(self) -> int:
        """Persons in the *rural* model: the sum of ``size`` over everyone still on the land."""
        return int(self.size[self.live()].sum())

    def present(self) -> np.ndarray:
        """Households still alive anywhere, rural or urban.

        Distinct from :meth:`live`, which is the rural population only. An urban household is
        still a household -- it earns, eats, reproduces and may come back -- so it belongs in the
        demographic step even though it is outside the countryside.
        """
        return np.nonzero(self.state[: self.n] != int(Tenure.DECEASED))[0]

    def total_head_count(self) -> int:
        """Persons anywhere, rural plus urban. The conserved quantity of the demography."""
        return int(self.size[self.present()].sum())

    def in_states(self, states) -> np.ndarray:
        s = self.state[: self.n]
        if len(states) == 1:
            return np.nonzero(s == int(states[0]))[0]
        # One table lookup beats one full-array comparison per state: the mask is built in a
        # single pass over ``state`` regardless of how many tenures are being asked for.
        lut = np.zeros(len(Tenure), dtype=bool)
        for st in states:
            lut[int(st)] = True
        return np.nonzero(lut[s])[0]


class _Draws:
    """Buffered scalar uniform draws off a single generator.

    The schedule takes on the order of a million scalar draws a run -- one per engrossment
    attempt, one per conversion decision, two per sitting customary tenant -- and numpy's
    per-call dispatch costs several times the bit generation itself. Drawing in blocks cuts
    that to a few hundred generator calls.

    The block is taken from the model's own generator, so a run is fully determined by its
    seed; but the stream is consumed in a different order than a call-per-draw implementation
    would, so results are not comparable with those produced before this change.
    """

    __slots__ = ("_rng", "_buf", "_i", "_block")

    def __init__(self, rng: np.random.Generator, block: int = 8192) -> None:
        self._rng = rng
        self._block = block
        self._buf = rng.random(block)
        self._i = 0

    def random(self) -> float:
        i = self._i
        if i >= self._block:
            self._buf = self._rng.random(self._block)
            i = 0
        self._i = i + 1
        return self._buf[i]

    def uniform(self, low: float, high: float) -> float:
        """``low + (high - low) * u``, which is what ``Generator.uniform`` computes."""
        return low + (high - low) * self.random()


def _degree_blocks(
    neighbours: list[np.ndarray], with_self: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bundle a fixed neighbour list into one index block per distinct degree.

    ``with_self`` appends each node's own index after its neighbours, matching the
    ``np.append(neighbours, i)`` the local enclosure rule used. Nodes with no neighbours are
    dropped when ``with_self`` is false: their mean was defined to be zero, not computed.
    """
    degree = np.array([len(nb) for nb in neighbours], dtype=np.intp)
    if with_self:
        degree = degree + 1
    blocks = []
    for d in np.unique(degree):
        if d == 0:
            continue
        who = np.nonzero(degree == d)[0]
        rows = [
            np.append(neighbours[i], i) if with_self else neighbours[i] for i in who
        ]
        blocks.append((who, np.stack(rows).astype(np.intp)))
    return blocks


def _block_means(
    values: np.ndarray, blocks: list[tuple[np.ndarray, np.ndarray]], out: np.ndarray
) -> np.ndarray:
    """Write the mean of ``values`` over each node's block into ``out``, and return it."""
    for who, idx in blocks:
        out[who] = values[idx].mean(axis=1)
    return out


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # The scalar path is taken hundreds of thousands of times a run (every conversion and
    # engrossment draw), where numpy's dispatch overhead dwarfs the arithmetic.
    if isinstance(x, float):
        return 1.0 / (1.0 + math.exp(-_clamp(x)))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _clamp(x: float) -> float:
    return -500.0 if x < -500.0 else (500.0 if x > 500.0 else x)


class Model:
    """One simulation run."""

    def __init__(
        self,
        params: Params,
        geography: Geography | None = None,
        artifact: dict | None = None,
    ) -> None:
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.draws = _Draws(self.rng)
        self.geo = geography if geography is not None else build_geography(
            params, self.rng, artifact=artifact
        )
        self.t = 0

        g = self.geo
        n_parcels = g.n_parcels

        # ---- parcel state -------------------------------------------------------------
        self.phi = g.phi_bar.copy()
        """Fertility starts at carrying capacity: the run begins from an unexhausted soil."""
        self.phi_bar = g.phi_bar
        self.parcel_landlord = g.landlord
        self.commons = self.rng.random(n_parcels) < params.commons_share
        self.customary_rent = self.rng.uniform(*params.customary_rent_range, size=n_parcels)
        """r^C_j(0): fixed nominal, and attached to the parcel -- custom attaches to the land."""
        self.occupant = np.full(n_parcels, -1, dtype=np.int32)
        self.parcel_rent = np.zeros(n_parcels)
        """Competitive rent struck at the last auction (paper: Competitive allocation).

        Held in real terms and reset only when the parcel is re-let, which is what makes a
        lease different from custom: both are fixed at grant, but a lease comes up again.
        """
        self.first_conversion = np.full(n_parcels, -1, dtype=np.int32)
        self.parcel_tenure = np.full(n_parcels, int(Tenure.CUSTOMARY), dtype=np.int8)
        """Tenure attaches to the parcel and only ever ratchets forward.

        Once customary right has been extinguished on a holding it does not come back: a
        failed leaseholder is replaced by another leaseholder, not by a customary tenant. The
        paper's "custom attaches to the land, not the person" then means the narrower and
        correct thing -- that a *still-customary* parcel's next tenant inherits its old fixed
        rent -- rather than that leasehold decays back into custom.
        """
        #: parcel -> tenure it has been designated but not yet filled at (the vacancy queue).
        self.queue: dict[int, Tenure] = {}

        # ---- landlord state -----------------------------------------------------------
        # The realised estate count comes from the geography, since lords are seeded per
        # county; params.n_landlords is only a fallback for synthetic geographies.
        self.n_landlords = n_l = self.geo.n_landlords
        #: Parcels per estate, fixed for the run. Used to weight per-estate quantities into a
        #: national one, so an aggregate is a share of land rather than a mean over estates --
        #: which matters here because estates differ several-fold in size. Floored at 1 so an
        #: estate that somehow held no land cannot zero the weights.
        self._estate_sizes = np.maximum(
            np.array([len(parcels) for parcels in self.geo.estate_parcels], dtype=float), 1.0
        )
        #: The awareness graph rebundled by degree, so a mean over each lord's neighbours is
        #: a handful of array reductions instead of one per lord. Each entry is
        #: ``(lords, index_block)`` with ``index_block`` of shape ``(len(lords), degree)``.
        #:
        #: Grouped by degree rather than flattened into one ``bincount`` on purpose: a
        #: bincount accumulates in a different order from ``mean``, and differs from it in the
        #: last bits, which in a chaotic model is enough to move a trajectory. Reducing a
        #: contiguous last axis runs the same inner loop numpy runs on a single row, so this
        #: form is bit-for-bit what the per-lord loop produced.
        self._nb_blocks = _degree_blocks(self.geo.landlord_neighbours, with_self=False)
        #: The same, with each lord appended after its own neighbours -- the neighbourhood the
        #: local enclosure rule averages over. Never empty, so every lord appears here.
        self._nb_self_blocks = _degree_blocks(self.geo.landlord_neighbours, with_self=True)
        self.W = np.full(n_l, params.landlord_wealth_0)
        self.W0 = self.W.copy()
        if params.consumption_rule == "rent_roll":
            rent_roll = np.array(
                [
                    float(self.customary_rent[g.estate_parcels[i]].sum())
                    for i in range(n_l)
                ]
            )
            self.consumption = np.maximum(
                self.rng.uniform(*params.consumption_ratio, size=n_l) * rent_roll, 1e-6
            )
        else:
            self.consumption = self.rng.uniform(*params.consumption_range, size=n_l)
        self.iota_landlord = np.full(n_l, params.iota_landlord_0)
        self.delta_fiscal = np.zeros(n_l)
        self.delta_relative = np.zeros(n_l)
        self.customary_fines = np.zeros(n_l)
        self.estate_receipts = np.zeros(n_l)
        self.n_births = 0
        self.n_inplace = 0
        self.observed = np.zeros(n_l)
        """Ibar_i(t), the neighbour-observation term."""
        self.theta_eff = np.full(n_l, params.theta)
        # SPEC NOTE: the paper bootstraps rhat_i at a landlord's first conversion "from the
        # estate's own customary rent scale". We use the estate's mean *nominal* customary
        # rent, which is the pre-inflation scale; this is what produces the rent jump at
        # conversion the paper describes, since sitting customary tenants pay r^C/Pi.
        self.r_hat = np.array(
            [
                float(self.customary_rent[g.estate_parcels[i]].mean())
                if len(g.estate_parcels[i])
                else 1.0
                for i in range(n_l)
            ]
        )

        # ---- global state ----------------------------------------------------------------
        self.price_index = 1.0
        self.enclosure = params.enclosure_0
        self.enclosure_by_estate = np.full(n_l, params.enclosure_0)
        self.enclosure_half_time = np.full(n_l, -1, dtype=np.int32)
        """Period at which each estate's Xi_i first passed one half, or -1 if it never did.

        The counterpart of :attr:`first_conversion`, and the reason it exists: RQ9 asks whether
        enclosure and rent conversion are separable processes, and with a national Xi that can
        only be asked about their timing. Two comparable per-place event times let it be asked
        about their geography as well -- do the fronts travel together, or apart. Half is an
        arbitrary threshold, but it is the same arbitrary threshold everywhere and Xi is monotone
        in t, so the ordering it induces does not depend on the level chosen."""
        """Xi_i(t) per estate. Under ``enclosure_rule="national"`` every entry is equal and this
        is simply the scalar broadcast, which is what keeps the two rules on one code path: the
        turnover hazard always reads a household's own lord's value, and under the national rule
        that value is the national one. ``self.enclosure`` remains the reported aggregate in both
        cases, parcel-weighted so that it means the same thing under either rule."""
        self.wage = params.wage_0
        self.goods_price = params.goods_price_0
        self.urban_population = 0
        self.urban_households = 0
        self._receipts_customary = np.zeros(n_l)
        self._receipts_leasehold = np.zeros(n_l)

        # ---- people -----------------------------------------------------------------------
        self.people = People()
        # Zeroed here as well as per period, so that the flow counters and the surplus vector
        # exist before the first step for anything that pokes at the model directly.
        self._reset_counters()
        self._surplus = np.zeros(0)
        self._seed_population()
        self._refresh_period_probabilities()

        # ---- recording --------------------------------------------------------------------
        self.history: list[dict] = []
        # Accumulated by the model rather than summed back out of ``history``, which would
        # undercount by a factor of ``record_every`` the moment the history is not per-period.
        self.cumulative_exits = 0
        self.cumulative_returns = 0
        self.cumulative_births = 0
        self.cumulative_deaths = 0
        self.cumulative_partitions = 0
        steps = params.n_steps
        self.panel_state = np.full((steps, n_parcels), -1, dtype=np.int8)
        self.panel_occupant = np.full((steps, n_parcels), -1, dtype=np.int32)
        self.panel_holding = np.zeros((steps, n_parcels), dtype=np.int16)
        """Size of the holding each parcel belongs to -- the parcel-level view of
        consolidation, which is what lets it be broken down by region and fertility."""
        self.panel_iota = np.full((steps, n_parcels), np.nan)
        self.panel_k = np.full((steps, n_parcels), np.nan)
        self.panel_rho = np.full((steps, n_parcels), np.nan)
        self.panel_y = np.full((steps, n_parcels), np.nan)

    # -------------------------------------------------------------------------------------
    # initialisation
    # -------------------------------------------------------------------------------------
    def _draw_need(self) -> float:
        """This household's own per-head subsistence cost while landless.

        Drawn once and kept for the household's lifetime. With a spread of zero every household
        gets the same requirement and the behaviour is the original homogeneous one.
        """
        p = self.p
        if p.landless_consumption_spread <= 0.0:
            return float(p.landless_consumption)
        lo = 1.0 - p.landless_consumption_spread
        hi = 1.0 + p.landless_consumption_spread
        return float(p.landless_consumption * self.draws.uniform(lo, hi))

    def _seed_population(self) -> None:
        """One household per parcel; Customary except a small Freehold minority.

        Leasehold and Landless both start empty so that the agrarian triad is an outcome of
        the run rather than a precondition of it.
        """
        p = self.p
        for parcel in range(self.geo.n_parcels):
            freehold = self.draws.random() < p.init_freehold_share
            person = self.people.add(
                state=Tenure.FREEHOLD if freehold else Tenure.CUSTOMARY,
                w=p.tenant_wealth_0,
                k=p.k_trad,
                iota=p.iota_tenant_0,
                mobility=self.draws.uniform(*p.mobility_range),
                landlord=int(self.parcel_landlord[parcel]),
                holding={parcel},
                size=p.household_size_0 if p.population_rule == "household_size" else 1,
                need=self._draw_need(),
            )
            self.occupant[parcel] = person
            self.parcel_tenure[parcel] = int(
                Tenure.FREEHOLD if freehold else Tenure.CUSTOMARY
            )

    # -------------------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------------------
    def _estate_tenancies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Counts of active tenancies per estate: (total, leasehold, customary).

        A tenancy is a *separately-let slot*, so a parcel absorbed by engrossment has ceased
        to be its own tenancy and drops out of both numerator and denominator.
        """
        ppl = self.people
        n_l = self.n_landlords
        idx = ppl.in_states(OCCUPIED)
        lord = ppl.landlord[idx]
        keep = lord >= 0
        lord = lord[keep].astype(np.intp)
        state = ppl.state[idx][keep]
        total = np.bincount(lord, minlength=n_l)
        leasehold = np.bincount(lord[state == int(Tenure.LEASEHOLD)], minlength=n_l)
        customary = np.bincount(lord[state == int(Tenure.CUSTOMARY)], minlength=n_l)
        return total, leasehold, customary

    def _price(self) -> float:
        """Price of produce. 1.0 when the domestic-market channel is off, so that with it
        disabled every quantity stays in produce units and nothing is rescaled."""
        return self.goods_price if self.p.urban_demand else 1.0

    def _aggregate_holdings(self) -> dict[str, np.ndarray]:
        """Per-person totals over the parcels they hold, in one pass.

        Every hot loop in the schedule used to walk each tenant's holding set individually,
        which is what made run time scale as parcels^1.74 rather than linearly. Because
        ``occupant`` already maps parcel -> person, the same totals fall out of ``bincount``
        over the parcel axis, so the cost is one vectorised pass regardless of holding sizes.
        """
        ppl = self.people
        n = ppl.n
        valid = self.occupant >= 0
        occ = self.occupant[valid].astype(np.intp)
        return {
            "size": np.bincount(occ, minlength=n).astype(float),
            "phi": np.bincount(occ, weights=self.phi[valid], minlength=n),
            "customary_rent": np.bincount(
                occ, weights=self.customary_rent[valid], minlength=n
            ),
            "parcel_rent": np.bincount(occ, weights=self.parcel_rent[valid], minlength=n),
            "commons": np.bincount(
                occ, weights=self.commons[valid].astype(float), minlength=n
            ),
        }

    def _family_labour(self, n: int) -> np.ndarray:
        """Labour each household supplies from its own members: ``n_j * ell_bar``."""
        return self.people.size[:n].astype(float) * self.p.household_labour

    def _subsistence_per_head(self) -> float:
        """What one member of a landholding household costs to keep.

        Only the household-size rule charges per head; under the single-body rules the household
        is one member and pays ``tenant_consumption``, so those arms are untouched by the
        separate per-head calibration.
        """
        p = self.p
        if p.population_rule == "household_size":
            return p.subsistence_per_head
        return p.tenant_consumption

    def _state_masks(self, n: int) -> dict[str, np.ndarray]:
        s = self.people.state[:n]
        customary = s == int(Tenure.CUSTOMARY)
        leasehold = s == int(Tenure.LEASEHOLD)
        freehold = s == int(Tenure.FREEHOLD)
        return {
            "customary": customary,
            "leasehold": leasehold,
            "freehold": freehold,
            "occupied": customary | leasehold | freehold,
            "landless": s == int(Tenure.LANDLESS),
        }

    def _rent_vector(self, n: int, agg: dict, masks: dict) -> np.ndarray:
        """rho_j(t) for everyone at once; see :meth:`_rent_for` for the per-agent version."""
        rent = np.zeros(n)
        rent[masks["customary"]] = (
            agg["customary_rent"][masks["customary"]] / self.price_index
        )
        if self.p.competitive_allocation:
            rent[masks["leasehold"]] = agg["parcel_rent"][masks["leasehold"]]
        else:
            landlord = self.people.landlord[:n]
            safe = np.clip(landlord, 0, max(self.n_landlords - 1, 0))
            rent[masks["leasehold"]] = (
                self.r_hat[safe][masks["leasehold"]] * agg["size"][masks["leasehold"]]
            )
        return rent

    def _rent_for(self, person: int) -> float:
        """rho_j(t): the state-dependent rent actually paid this period.

        Rent scales with the size of the holding. The paper charges one ``rhat_i`` per
        *tenancy*, which would let a leaseholder engross five parcels and pay what they paid
        with one -- making concentration rent-free and strongly over-incentivised.
        """
        ppl = self.people
        state = ppl.state[person]
        holding = ppl.holdings[person]
        if not holding:
            return 0.0
        if state == int(Tenure.CUSTOMARY):
            nominal = sum(self.customary_rent[k] for k in holding)
            return float(nominal / self.price_index)
        if state == int(Tenure.LEASEHOLD):
            if self.p.competitive_allocation:
                # Each parcel carries the rent its own auction struck.
                return float(sum(self.parcel_rent[k] for k in holding))
            return float(self.r_hat[ppl.landlord[person]] * len(holding))
        return 0.0

    def _benchmark_obligation(self, person: int) -> float:
        """What the estate's going competitive rate would cost this holding.

        This is the yardstick a market-exposed tenant must clear. A customary tenant is not
        measured against it -- their rent is fixed by custom and their tenure secure -- which
        is what the competitive-selection reading of Brenner turns on.
        """
        ppl = self.people
        holding = ppl.holdings[person]
        if not holding or ppl.landlord[person] < 0:
            return 0.0
        return float(self.r_hat[ppl.landlord[person]] * len(holding))

    def _release(self, person: int, to_landless: bool) -> list[int]:
        """Detach a person from their holding. Returns the freed parcels."""
        ppl = self.people
        parcels = sorted(ppl.holdings[person])
        for k in parcels:
            if self.occupant[k] == person:
                self.occupant[k] = -1
        ppl.holdings[person] = set()
        ppl.n_held[person] = 0
        if to_landless:
            ppl.state[person] = int(Tenure.LANDLESS)
            ppl.mark_landless(person)
            ppl.landlord[person] = -1
            ppl.k[person] = 0.0  # accumulated improvement is lost with the tenancy
            ppl.shortfall[person] = 0
            ppl.hired[person] = 0.0
        return parcels

    def _assign(self, person: int, parcel: int, state: Tenure) -> None:
        ppl = self.people
        ppl.state[person] = int(state)
        ppl.landlord[person] = int(self.parcel_landlord[parcel])
        ppl.holdings[person].add(parcel)
        ppl.n_held[person] = len(ppl.holdings[person])
        self.occupant[parcel] = person
        # Tenure ratchets: a parcel never falls back to a less market-exposed state.
        if state == Tenure.LEASEHOLD or self.parcel_tenure[parcel] != int(Tenure.LEASEHOLD):
            self.parcel_tenure[parcel] = int(state)
        if state == Tenure.LEASEHOLD and self.first_conversion[parcel] < 0:
            self.first_conversion[parcel] = self.t

    # -------------------------------------------------------------------------------------
    # the schedule
    # -------------------------------------------------------------------------------------
    def _reset_counters(self) -> None:
        """Per-period event and flow counters, zeroed at the start of each period."""
        self.n_inplace = 0
        self.n_inplace_dispossessed = 0
        self.n_evictions = 0
        self.n_successions = 0
        self.n_engrossments = 0
        self.n_conversions_at_vacancy = 0
        self.n_freehold_diversions = 0
        self.n_exits = 0
        self.n_exit_persons = 0
        self.n_returns = 0
        self.n_return_persons = 0
        self.n_births = 0
        self.n_deaths = 0
        self.n_births_urban = 0
        self.n_deaths_urban = 0
        self.n_partitions = 0
        self.n_partition_persons = 0
        self.n_extinctions = 0
        self.n_disinherited = 0
        self.n_new_customary = 0
        self.investment_flow = 0.0
        self.wage_bill_total = 0.0
        self.fines_total = 0.0
        #: Landholding households that lost their last member this period, so their parcels
        #: fall vacant with no claimant at all. Filled in step 4c, consumed by step 7.
        self._extinct: list[int] = []

    def step(self) -> None:
        """Advance one period, following the paper's ten-step schedule."""
        self._reset_counters()
        self.people.reclaim()
        self.people.refresh_landless()
        self._step1_exogenous()
        self._step1b_domestic_market()
        hired_total, landless_count = self._step2_labour()
        self._step3_production()
        self._step4_wealth()
        self._step5_fiscal()
        self._step6_spread()
        self._refresh_period_probabilities()
        self._step6b_inplace_conversion()
        vacancies = self._step7_vacancies()
        self._step8_resolve(vacancies)
        self._step9_rent_reset()
        self.cumulative_exits += self.n_exits
        self.cumulative_returns += self.n_returns
        self.cumulative_births += self.n_births
        self.cumulative_deaths += self.n_deaths
        self.cumulative_partitions += self.n_partitions
        self._record(hired_total, landless_count)
        self.t += 1

    # -- 1. exogenous ---------------------------------------------------------------------
    def _step1b_domestic_market(self) -> None:
        """Price the produce against demand from the population that has left the land.

        This closes Wood's loop (p.103): agrarian productivity throws people off the land,
        and that propertyless mass is itself the domestic market the surviving farms sell
        into. Without it, exit to industry is a pure drain and the countryside never feels
        the consequence of its own dispossession.
        """
        p = self.p
        if not p.urban_demand:
            return
        ppl = self.people
        n = ppl.n
        # Persons, not households: the urban market is mouths to feed, and a household that
        # left for industry took its whole family with it.
        gone = ppl.state[:n] == int(Tenure.EXITED)
        self.urban_population = int(ppl.size[:n][gone].sum())
        self.urban_households = int(gone.sum())
        demand = self.urban_population * p.urban_consumption
        # Only market-exposed holdings sell; customary production is largely eaten at home.
        masks = self._state_masks(n)
        marketed = float(ppl.y[:n][masks["leasehold"] | masks["freehold"]].sum())
        if marketed <= 0:
            return
        excess = np.clip((demand - marketed) / marketed, -1.0, 1.0)
        self.goods_price = float(
            np.clip(
                self.goods_price * (1.0 + p.kappa_goods * excess), *p.goods_price_cap
            )
        )

    def _step1_exogenous(self) -> None:
        p = self.p
        self.price_index *= 1.0 + p.inflation
        shock = (self.rng.random(len(self.phi)) < p.shock_prob) * self.rng.uniform(
            *p.shock_magnitude, size=len(self.phi)
        )
        growth = p.phi_regrowth * self.phi * (1.0 - self.phi / self.phi_bar)
        self.phi = np.clip(self.phi + growth - shock, 0.0, None)

    # -- 2. labour demand and clearing ------------------------------------------------------
    def _step2_labour(self) -> tuple[float, float]:
        p = self.p
        ppl = self.people
        ppl.hired[: ppl.n] = 0.0

        n = ppl.n
        self._agg = agg = self._aggregate_holdings()
        self._masks = masks = self._state_masks(n)
        self._family = family = self._family_labour(n)
        exposed = (masks["leasehold"] | masks["freehold"]) & (agg["size"] > 0)
        hirers = np.nonzero(exposed)[0]

        # A household hires only what its own members cannot cover, so a large family and a
        # large wage bill are substitutes -- and because only hired labour is charged at omega,
        # the family farm's own labour is the cheaper of the two.
        if p.labour_demand_rule == "marginal_product":
            # h* = [(1-mu-gamma) Phi^mu k^gamma / omega]^(1/(mu+gamma)); see _labour_demand.
            labour_exp = 1.0 - p.cobb_phi - p.cobb_capital
            scale = (
                labour_exp
                * np.maximum(agg["phi"][hirers], 0.0) ** p.cobb_phi
                * np.maximum(ppl.k[hirers], 1e-9) ** p.cobb_capital
            )
            optimum = (scale / max(self.wage, 1e-12)) ** (
                1.0 / (p.cobb_phi + p.cobb_capital)
            )
            demand = np.clip(optimum - family[hirers], 0.0, 1e6)
        else:
            demand = np.maximum(0.0, p.delta * ppl.k[hirers] - family[hirers])
        total_demand = float(demand.sum())
        # The pool is bodies available for hire, not households looking for work.
        pool = float(family[masks["landless"]].sum())

        # Demand is a claim on a finite pool, not a guarantee: ration pro rata so that the
        # production function never draws on more bodies than exist.
        if total_demand > 0:
            ration = min(1.0, pool / total_demand) if pool > 0 else 0.0
            ppl.hired[hirers] = demand * ration

        if pool > 0:
            excess = np.clip((total_demand - pool) / pool, -p.wage_adjust_cap, p.wage_adjust_cap)
            self.wage = float(np.clip(self.wage * (1.0 + p.kappa * excess), 1e-6, 1e6))
        return total_demand, pool

    def _labour_demand(self, person: int) -> float:
        """Hire until the marginal product of labour equals the wage.

        With ``y = Phi^mu k^gamma h^(1-mu-gamma)``, setting ``dy/dh = omega`` gives
        ``h* = [(1-mu-gamma) Phi^mu k^gamma / omega]^(1/(mu+gamma))``. Unlike the paper's
        ``delta*k - ell_bar``, this responds to the price of labour, so the market can clear.
        """
        p = self.p
        ppl = self.people
        holding = ppl.holdings[person]
        if not holding or self.wage <= 0:
            return 0.0
        phi_total = float(self.phi[list(holding)].sum())
        if phi_total <= 0.0:
            return 0.0
        labour_exp = 1.0 - p.cobb_phi - p.cobb_capital
        scale = labour_exp * phi_total**p.cobb_phi * max(ppl.k[person], 1e-9) ** p.cobb_capital
        optimum = (scale / self.wage) ** (1.0 / (p.cobb_phi + p.cobb_capital))
        family = float(ppl.size[person]) * p.household_labour
        return float(max(0.0, min(optimum - family, 1e6)))

    # -- 3. production -----------------------------------------------------------------------
    def _step3_production(self) -> None:
        p = self.p
        ppl = self.people
        n = ppl.n
        agg, masks = self._agg, self._masks
        labour_exp = 1.0 - p.cobb_phi - p.cobb_capital

        ppl.y[:n] = 0.0
        active = masks["occupied"] & (agg["phi"] > 0.0)
        if not active.any():
            return
        labour = self._family[active] + ppl.hired[:n][active]
        ppl.y[:n][active] = (
            agg["phi"][active] ** p.cobb_phi
            * np.maximum(ppl.k[:n][active], 1e-9) ** p.cobb_capital
            * labour**labour_exp
        )

    # -- 4. wealth, capital, disposition -------------------------------------------------------
    def _step4_wealth(self) -> None:
        p = self.p
        ppl = self.people

        n = ppl.n
        agg, masks = self._agg, self._masks
        occupied = masks["occupied"]
        self.customary_fines = np.zeros(self.n_landlords)

        # --- rents, wage bill and the resulting wealth ------------------------------------
        rent = self._rent_vector(n, agg, masks)
        rent[~occupied] = 0.0
        ppl.rho[:n] = rent

        wage_bill = np.zeros(n)
        if p.charge_wage_bill:
            wage_bill[occupied] = self.wage * ppl.hired[:n][occupied]
        self.wage_bill_total = float(wage_bill.sum())

        # Market-exposed holdings sell their produce at the going price; customary production
        # is consumed at home, so it is not revalued by the urban market.
        revenue = ppl.y[:n].copy()
        if p.urban_demand:
            sells = masks["leasehold"] | masks["freehold"]
            revenue[sells] = revenue[sells] * self.goods_price
        # Subsistence is owed per head, so a larger family costs more to keep as well as
        # producing more -- which is what makes household size a real constraint rather than a
        # free source of labour. Output rises with labour as n^(1-mu-gamma) while this rises as
        # n, so every holding has a size beyond which it cannot feed its own household, and that
        # threshold is higher on better or larger land.
        subsistence = self._subsistence_per_head() * ppl.size[:n].astype(float)
        income = revenue - rent - subsistence - wage_bill
        new_w = np.where(occupied, ppl.w[:n] + income, ppl.w[:n])

        # --- Brenner's squeeze: arbitrary fines on customary income above subsistence -------
        if p.customary_fine_rate > 0 and masks["customary"].any():
            surplus = np.where(masks["customary"], np.maximum(income, 0.0), 0.0)
            fine = p.customary_fine_rate * surplus
            new_w -= fine
            self.fines_total = float(fine.sum())
            landlord = ppl.landlord[:n]
            charged = fine > 0
            if charged.any() and self.n_landlords > 0:
                self.customary_fines = np.bincount(
                    np.clip(landlord[charged], 0, self.n_landlords - 1),
                    weights=fine[charged],
                    minlength=self.n_landlords,
                )

        # --- competitive necessity, and the improvement it drives ---------------------------
        # Zero under custom: a fixed customary rent is a lump sum, not a yardstick others set,
        # so it exerts no selection pressure. See the Tenant response section of the paper.
        safe_landlord = np.clip(ppl.landlord[:n], 0, max(self.n_landlords - 1, 0))
        benchmark = self.r_hat[safe_landlord] * agg["size"]
        # Both sides in money: the benchmark is a money rent, so it is measured against
        # revenue rather than against physical output.
        # Freehold's exposure is the shadow-rent construction, and RQ1 needs it ablatable: with
        # it on, a Freeholder feels the full yardstick while paying nothing, which hands Allen's
        # yeoman the result by construction rather than by finding.
        market_exposed = (
            masks["leasehold"] | masks["freehold"]
            if p.freehold_shadow_rent
            else masks["leasehold"]
        )
        exposed = market_exposed & (revenue > 0)
        pressure = np.zeros(n)
        pressure[exposed] = benchmark[exposed] / revenue[exposed]
        pressure[~np.isfinite(pressure)] = 0.0

        # SPEC NOTE: k(t+1) uses iota^T(t), the pre-update value, exactly as the paper writes
        # it -- so a rise in disposition feeds reinvestment with a one-period lag.
        investment = np.where(
            occupied, p.s_bar * ppl.iota[:n] * np.maximum(new_w - p.w_min, 0.0), 0.0
        )
        self.investment_flow = float(investment.sum())
        ppl.k[:n] = np.where(
            occupied,
            np.maximum(p.k_trad, ppl.k[:n] * (1.0 - p.capital_depreciation) + investment),
            ppl.k[:n],
        )
        ppl.iota[:n] = np.where(
            occupied,
            np.clip(
                ppl.iota[:n] * (1.0 - p.iota_decay)
                + p.beta_tenant * pressure * (1.0 - ppl.iota[:n]),
                0.0,
                1.0,
            ),
            ppl.iota[:n],
        )
        ppl.w[:n] = new_w
        ppl.shortfall[:n] = np.where(
            occupied,
            np.where(new_w < 0, ppl.shortfall[:n] + 1, 0),
            ppl.shortfall[:n],
        )

        # Per-head surplus relative to per-head subsistence: the dimensionless quantity the
        # vital rates read. Recorded for landholding households here and for landless
        # households in step 4b, so that both classes are scored on the same scale.
        self._surplus = np.zeros(n)
        per_head = np.maximum(subsistence / np.maximum(ppl.size[:n], 1), 1e-9)
        self._surplus[occupied] = (
            income[occupied] / ppl.size[:n][occupied] / per_head[occupied]
        )

        if p.population_rule == "household":
            # Sparse: only the households that actually cleared the threshold.
            for j in np.nonzero(occupied & (ppl.w[:n] >= p.birth_threshold))[0]:
                self._spawn_household(int(j))

        self._step4_landless()
        self._step4b_urban()
        self._step4c_demography()

    def _step4_landless(self) -> None:
        """Landless income and departure for industry.

        SPEC NOTE: the paper's landless income equation is unconditional on actually being
        hired; unemployment acts through the wage level, which is the "reserve army" channel.
        """
        p = self.p
        ppl = self.people
        n = ppl.n
        landless = ppl.state[:n] == int(Tenure.LANDLESS)
        if not landless.any():
            return
        # The legacy household rule appends agents during step 4, so the surplus vector built
        # from the pre-birth population can be short of the current one.
        if len(self._surplus) < n:
            padded = np.zeros(n)
            padded[: len(self._surplus)] = self._surplus
            self._surplus = padded
        size = ppl.size[:n].astype(float)
        need = np.where(ppl.need[:n] > 0, ppl.need[:n], p.landless_consumption)
        # Both sides scale with the household, so a landless household's per-head balance --
        # and therefore its fate -- does not depend on how many members it has.
        income = (self.wage * p.household_labour - need) * size
        ppl.w[:n][landless] += income[landless]
        self._surplus[landless] = (self.wage * p.household_labour - need[landless]) / need[
            landless
        ]
        ppl.shortfall[:n] = np.where(
            landless,
            np.where(ppl.w[:n] < 0, ppl.shortfall[:n] + 1, 0),
            ppl.shortfall[:n],
        )

        if p.exit_rule == "counter":
            leaving = landless & (ppl.shortfall[:n] >= p.tau_prime)
        else:
            # Depth, not duration: the deficit expressed in periods of this household's own
            # consumption. A household barely underwater rarely leaves; one deeply in debt
            # almost surely does. Identical households therefore leave at different times,
            # which is what stops an eviction cohort departing as a single bloc.
            depth = np.maximum(0.0, -ppl.w[:n]) / np.maximum(need * size, 1e-9)
            hazard = 1.0 - np.exp(-p.lambda_exit * depth)
            leaving = (
                landless & (ppl.shortfall[:n] >= 1) & (self.rng.random(n) < hazard)
            )

        self.n_exits = int(leaving.sum())
        self.n_exit_persons = int(ppl.size[:n][leaving].sum())
        ppl.state[:n][leaving] = int(Tenure.EXITED)

    # -- 4b. the urban sector -------------------------------------------------------------------
    def _step4b_urban(self) -> None:
        """Income, and the return journey, for households that left the land.

        Leaving for industry is a move, not a death. An urban household earns ``urban_wage`` per
        member and pays ``urban_consumption`` per member, so it has a per-head surplus on the same
        scale as everyone else and is subject to the same vital rates in step 4c. It comes back
        when its members would be better off on the land -- the mirror image of the partition
        condition, and the reason the countryside's population is neither manufactured nor drained
        by the accounting.

        Without this the urban state is absorbing and the rural population can only fall, however
        healthy its own natural increase: over a 200-period run births exceeded deaths by 11,256
        while 20,369 persons left for industry and none returned, so a countryside that was
        reproducing perfectly well still ended at 58% of where it began. That is an artifact of
        the one-way valve, not a result about agrarian change.
        """
        p = self.p
        ppl = self.people
        n = ppl.n
        urban = ppl.state[:n] == int(Tenure.EXITED)
        if not urban.any():
            return
        if len(self._surplus) < n:
            padded = np.zeros(n)
            padded[: len(self._surplus)] = self._surplus
            self._surplus = padded

        size = ppl.size[:n].astype(float)
        per_head = max(p.urban_consumption, 1e-9)
        ppl.w[:n][urban] += ((p.urban_wage - p.urban_consumption) * size)[urban]
        self._surplus[urban] = (p.urban_wage - p.urban_consumption) / per_head

        if not p.urban_return:
            return
        # Return when the rural wage, net of this household's own subsistence requirement, beats
        # what industry leaves it -- the same net-position comparison that governs partition.
        # Scaled by the *size* of the advantage rather than triggered by its sign: on a bare sign
        # test a vanishing advantage moves as many households as a decisive one, and because
        # returning households enlarge the labour pool and so depress the very wage that drew
        # them, that produces a churn loop rather than a response to conditions.
        need = np.where(ppl.need[:n] > 0, ppl.need[:n], p.landless_consumption)
        advantage = (self.wage * p.household_labour - need) - (
            p.urban_wage - p.urban_consumption
        )
        pull = np.clip(advantage / np.maximum(need, 1e-9), 0.0, 1.0)
        returning = urban & (self.rng.random(n) < p.urban_return_rate * pull)
        if not returning.any():
            return
        ppl.state[:n][returning] = int(Tenure.LANDLESS)
        for j in np.nonzero(returning)[0]:
            ppl.mark_landless(j)
        ppl.landlord[:n][returning] = -1
        ppl.k[:n][returning] = 0.0
        ppl.shortfall[:n][returning] = 0
        self.n_returns = int(returning.sum())
        self.n_return_persons = int(ppl.size[:n][returning].sum())

    # -- 4c. births, deaths and partition -------------------------------------------------------
    def _vital_rates(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-member birth and death probabilities for the households in ``idx``.

        Both are functions of the household's *own* per-head surplus, never of any aggregate
        ratio of people to food. Fertility is anchored at ``mortality_0`` where surplus is zero,
        so a household exactly meeting subsistence is stationary; above it fertility rises to a
        cap, below it fertility falls towards zero *and* mortality rises. A landholding
        household's surplus comes from its land, its capital and its rent; a landless one's from
        the wage. So the aggregate rate is whatever the balance of classes makes it -- which is
        the point, and the reason this is not the Malthusian law Brenner rejects.
        """
        p = self.p
        s = self._surplus[idx]
        birth = np.clip(p.mortality_0 + p.birth_slope * s, 0.0, p.birth_rate_max)
        death = p.mortality_0 + p.mortality_crisis * np.maximum(0.0, -s)
        return birth, np.clip(death, 0.0, 1.0)

    def _step4c_demography(self) -> None:
        """Births, deaths and partition, for the two endogenous population rules.

        Ordered deaths, then births, then partition, and applied only to households that
        existed at the start of the step -- so a household created here cannot also reproduce
        or partition in the same period.
        """
        p = self.p
        if p.population_rule not in ("vital_rates", "household_size"):
            return
        ppl = self.people
        # Bounded by the surplus vector: a household created after step 4 has no income of its
        # own yet, so it neither reproduces nor partitions until the next period.
        n = min(ppl.n, len(self._surplus))
        # Urban households are included: they earn, eat and reproduce like anyone else (step 4b).
        # Excluding them would make emigration a form of sterilisation and quietly reintroduce the
        # one-way drain that the return channel exists to remove.
        alive = np.nonzero(ppl.state[:n] != int(Tenure.DECEASED))[0]
        if not len(alive):
            return

        birth_rate, death_rate = self._vital_rates(alive)
        sizes = ppl.size[alive]

        # --- deaths ----------------------------------------------------------------------
        # One draw per member, so a large household loses members faster in absolute terms but
        # at the same rate per head.
        deaths = np.minimum(self.rng.binomial(sizes, death_rate), sizes)
        births = self.rng.binomial(sizes, birth_rate)
        self.n_deaths = int(deaths.sum())
        # Split by sector, so that both accounting identities can be checked: the *rural*
        # headcount moves with rural vital events plus net migration, while the *total* moves only
        # with births and deaths, migration being an internal transfer between the two.
        urban_now = ppl.state[alive] == int(Tenure.EXITED)
        self.n_deaths_urban = int(deaths[urban_now].sum())

        if p.population_rule == "household_size":
            new_size = np.clip(sizes - deaths + births, 0, p.household_size_max)
            ppl.size[alive] = new_size
            # Count the births that actually landed, not the ones drawn: where the size cap
            # binds, the surplus draws are discarded, and counting them would break the
            # accounting identities.
            landed = new_size - sizes + deaths
            self.n_births = int(landed.sum())
            self.n_births_urban = int(landed[urban_now].sum())
            # A household that has lost its last member is extinct. Its holding falls vacant
            # with no claimant at all, which is escheat rather than eviction: the parcels
            # revert to the lord's disposal, and step 7 picks them up.
            extinct = alive[new_size <= 0]
            self._extinct = [int(j) for j in extinct if ppl.state[j] in OCCUPIED_INTS]
            for j in extinct:
                # A landholding line's parcels are resolved in step 8; a landless or urban
                # household simply ceases.
                if ppl.state[j] not in OCCUPIED_INTS:
                    ppl.state[j] = int(Tenure.DECEASED)
            self._partition(alive)
            return

        # --- vital_rates: single-body households, so a birth is a new landless household ----
        ppl.size[alive] = np.maximum(sizes - deaths, 0)
        died_out = alive[ppl.size[alive] <= 0]
        self._extinct = [int(j) for j in died_out if ppl.state[j] in OCCUPIED_INTS]
        for j in died_out:
            if ppl.state[j] not in OCCUPIED_INTS:
                ppl.state[j] = int(Tenure.DECEASED)
        for j, count in zip(alive, births):
            if ppl.size[j] <= 0:
                continue
            for _ in range(int(count)):
                self._spawn_household(int(j))

    def _partition(self, alive: np.ndarray) -> None:
        """A household with more hands than its land can use sends one out to work for wages.

        This is proletarianisation as a demographic mechanism rather than only as a consequence
        of eviction: the member who leaves is the one the holding has no productive use for --
        historically the non-inheriting son -- and where they go is the labour market.

        Two things can push a member out. The holding may be unable to *feed* everyone, which is
        the per-head deficit; or it may be unable to *employ* everyone, which is the marginal
        product of family labour falling below the going wage. The second is the operative one
        most of the time, and it is what makes the household's decision an economic one rather
        than a subsistence crisis: a member stays while they add more at home than they could
        earn outside. It also runs the classic peasant response in the right direction -- when
        the wage collapses the family retains its members and works the holding harder, and when
        industry pays well it releases them.

        Either way the trigger is local, so the productivity of a household's own land sets how
        many people it can hold. Good or consolidated land absorbs population; poor or small land
        expels it. There is no aggregate ceiling anywhere in this.
        """
        p = self.p
        ppl = self.people
        big_enough = ppl.size[alive] >= p.partition_min_size
        # Partition applies only to households holding land: a landless household that split in
        # two would change nothing, since both halves already live by the wage.
        holds = np.isin(ppl.state[alive], list(OCCUPIED_INTS))
        starving = self._surplus[alive] < p.partition_surplus
        crowded = alive[big_enough & holds & (starving | self._below_wage(alive))]
        if not len(crowded):
            self.n_partitions = 0
            return
        drawn = crowded[self.rng.random(len(crowded)) < p.partition_rate]
        for j in drawn:
            self._shed_member(int(j))
        self.n_partitions = int(len(drawn))

    def _below_wage(self, idx: np.ndarray) -> np.ndarray:
        """Whether the marginal member is better placed outside the household than in it.

        Both sides are net of what it costs to keep that member. Inside, they add the marginal
        product of labour -- with ``y = Phi^mu k^gamma h^(1-mu-gamma)`` that is
        ``(1-mu-gamma) * y / h`` -- and eat per-head subsistence. Outside, they earn the wage and
        must meet their own landless requirement. Comparing the two *net* positions rather than
        the gross wage against the gross product is what makes the family farm's own labour
        genuinely cheaper than hired labour, and it is why a falling wage causes families to
        retain members and work the holding harder instead of sending them away.
        """
        p = self.p
        ppl = self.people
        labour_exp = 1.0 - p.cobb_phi - p.cobb_capital
        labour = np.maximum(
            ppl.size[idx].astype(float) * p.household_labour + ppl.hired[idx], 1e-9
        )
        inside = labour_exp * ppl.y[idx] / labour - self._subsistence_per_head()
        need = np.where(ppl.need[idx] > 0, ppl.need[idx], p.landless_consumption)
        outside = self.wage * p.household_labour - need
        return outside > inside

    def _shed_member(self, parent: int, members: int = 1) -> int:
        """Move ``members`` out of ``parent`` into a new landless household."""
        p = self.p
        ppl = self.people
        members = min(members, int(ppl.size[parent]) - 1)
        if members < 1:
            return -1
        dowry = min(p.partition_dowry * members, max(ppl.w[parent], 0.0))
        ppl.size[parent] -= members
        ppl.w[parent] -= dowry
        child = ppl.add(
            state=Tenure.LANDLESS,
            w=dowry,
            k=0.0,
            iota=ppl.iota[parent],  # the disposition travels with the person
            mobility=self.draws.uniform(*p.mobility_range),
            landlord=-1,
            size=members,
            need=self._draw_need(),
        )
        self.n_partition_persons += members
        return child

    def _spawn_household(self, parent: int) -> None:
        """A prospering household splits off a new, landless one (population_rule='household').

        Population growth is endogenous to surplus, but at the household level, so the rate
        depends on which classes are accumulating rather than on an aggregate food-population
        law -- the aggregate version being exactly the demographic explanation Brenner rejects.
        """
        p = self.p
        ppl = self.people
        ppl.w[parent] -= p.birth_cost
        ppl.add(
            state=Tenure.LANDLESS,
            w=p.birth_cost,
            k=0.0,
            iota=ppl.iota[parent],
            mobility=self.draws.uniform(*p.mobility_range),
            landlord=-1,
            need=self._draw_need(),
        )
        self.n_births += 1

    # -- 5. landlord fiscal pressure ------------------------------------------------------------
    def _step5_fiscal(self) -> None:
        ppl = self.people
        n_l = self.n_landlords
        idx = ppl.in_states(OCCUPIED)
        lord = ppl.landlord[idx]
        keep = lord >= 0
        lord = lord[keep].astype(np.intp)
        state = ppl.state[idx][keep]
        rho = ppl.rho[idx][keep]
        is_cust = state == int(Tenure.CUSTOMARY)
        is_lease = state == int(Tenure.LEASEHOLD)
        rent_customary = np.bincount(lord[is_cust], weights=rho[is_cust], minlength=n_l)
        rent_leasehold = np.bincount(lord[is_lease], weights=rho[is_lease], minlength=n_l)

        # Arbitrary fines are the lord's other lever: they offset the inflationary erosion of
        # customary rent, and so are a genuine alternative to conversion rather than a
        # side-effect of it.
        receipts = rent_customary + rent_leasehold + self.customary_fines
        self._receipts_customary = rent_customary
        self._receipts_leasehold = rent_leasehold
        self.delta_fiscal = self.consumption - receipts
        if self.p.normalise_fiscal_pressure:
            self.delta_relative = self.delta_fiscal / np.maximum(self.consumption, 1e-9)
        else:
            self.delta_relative = self.delta_fiscal
        # SPEC NOTE: the paper names W_i(t) as landlord state and uses it in Ibar_i, but never
        # writes its update. The only consistent reading is that wealth accumulates the same
        # surplus whose shortfall defines fiscal pressure, i.e. W(t+1) = W(t) - Delta(t).
        self.W = self.W - self.delta_fiscal
        self.estate_receipts = receipts

    # -- 6. spread channels -------------------------------------------------------------------
    def _step6_spread(self) -> None:
        p = self.p
        total, leasehold, _ = self._estate_tenancies()
        with np.errstate(divide="ignore", invalid="ignore"):
            leasehold_share = np.where(total > 0, leasehold / np.maximum(total, 1), 0.0)
        # Guard the wealth ratio at zero: a landlord driven into debt should read as "no
        # demonstrated gain", not as a negative signal that discourages conversion.
        gain = np.maximum(self.W / np.maximum(self.W0, 1e-9), 0.0)
        signal = leasehold_share * gain

        observed = np.zeros(self.n_landlords)
        if p.channel_observation:
            _block_means(signal, self._nb_blocks, observed)
        self.observed = observed

        if p.channel_ideology and p.ideology_rule == "reproduction":
            # Brenner's "rules for reproduction": what presses on a lord is not this year's
            # income but whether the stock behind the house can still sustain it. Wealth worth
            # more than `reproduction_horizon` periods of the requirement registers as no
            # pressure; as the cushion erodes, urgency rises. The disposition then accumulates
            # exactly as the tenant's does, so the two classes are the same kind of object.
            runway = self.W / np.maximum(self.consumption, 1e-9)
            press = np.clip(1.0 - runway / max(p.reproduction_horizon, 1e-9), 0.0, 1.0)
            self.iota_landlord = np.clip(
                self.iota_landlord * (1.0 - p.iota_landlord_decay)
                + p.beta_landlord * press * (1.0 - self.iota_landlord),
                0.0,
                1.0,
            )
            self.landlord_pressure = press

        elif p.channel_ideology and p.ideology_rule == "material":
            # No diffusion at all: the disposition is the share of the lord's income that
            # already comes from market-determined rent. A lord who lives by competitive rents
            # is by that fact committed to competitive production, so his resistance to
            # breaking custom falls -- an emergent read-out of how far the estate has been
            # pulled into producing for value, not an idea arriving from a neighbour.
            market_income = self._receipts_leasehold
            total_income = market_income + self._receipts_customary + self.customary_fines
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(total_income > 0, market_income / total_income, 0.0)
            self.iota_landlord = np.clip(np.nan_to_num(share), 0.0, 1.0)

        elif p.channel_ideology:
            # Contagion runs off neighbours' realised *conversions*, not their ideology. With
            # the paper's neighbour-ideology term neither part of the equation references any
            # actual break from custom, so ideology climbs to ~1 on its own schedule and the
            # channel behaves as an exogenous clock rather than a transmission mechanism --
            # while the paper's own causal diagram draws the arrow from the conversion
            # decision back into ideology.
            neighbour_conversion = _block_means(
                leasehold_share, self._nb_blocks, np.zeros(self.n_landlords)
            )
            self.iota_landlord = np.clip(
                self.iota_landlord
                + p.beta_1 * (1.0 - self.iota_landlord)
                + p.beta_2 * neighbour_conversion * (1.0 - self.iota_landlord),
                0.0,
                1.0,
            )

        self.theta_eff = np.clip(
            p.theta * (1.0 - p.xi_0 - p.xi_1 * self.iota_landlord), 0.0, None
        )

        if p.enable_enclosure:
            gate = 1.0 if self.t >= p.t_star else 0.0
            if p.enclosure_rule == "local":
                # Each lord's own disposition together with those he can see -- the same
                # neighbourhood the ideology channel uses, so enclosure and the ethic that
                # legitimates it now diffuse over the same graph rather than one locally and the
                # other everywhere at once. A lord with no neighbours in range is driven by his
                # own disposition alone, which is the isolated-estate case and not a special one.
                driver = _block_means(
                    self.iota_landlord, self._nb_self_blocks, np.zeros(self.n_landlords)
                )
            else:
                driver = np.full(self.n_landlords, float(self.iota_landlord.mean()))
            self.enclosure_by_estate = np.clip(
                self.enclosure_by_estate
                + (p.nu_1 + p.nu_2 * driver + p.nu_3 * gate)
                * (1.0 - self.enclosure_by_estate),
                0.0,
                1.0,
            )
            # Parcel-weighted, so the reported aggregate is the share of *land* enclosed and is
            # comparable across the two rules; under the national rule it equals the scalar the
            # earlier code carried, exactly.
            self.enclosure = float(
                np.clip(
                    np.average(self.enclosure_by_estate, weights=self._estate_sizes),
                    0.0,
                    1.0,
                )
            )
            newly = (self.enclosure_half_time < 0) & (self.enclosure_by_estate >= 0.5)
            self.enclosure_half_time[newly] = self.t

    # -- 6b. conversion of a sitting tenant --------------------------------------------------------
    def _step6b_inplace_conversion(self) -> None:
        """Fine escalation: the landlord reopens terms on a tenant who is still in occupation.

        The paper says a fine escalation is "a special case of the same conversion decision",
        but only ever reaches that decision at a vacancy -- so the sitting tenant is almost
        never converted, and RQ1's event study ends up measuring a change of occupant. This
        restores the path Brenner describes, in which fines levied on the sitting tenant were
        "in the long run...substituted for competitive commercial rents".
        """
        p = self.p
        if p.inplace_conversion_rate <= 0:
            return
        ppl = self.people
        self.n_inplace = 0
        self.n_inplace_dispossessed = 0
        # Both gates reject the great majority of sitting tenants, so they are drawn for the
        # whole customary population at once and only the survivors are walked. The holding
        # test moves after the gates: it costs a Python attribute lookup per tenant and
        # selects the same actors either way.
        customary = ppl.in_states((Tenure.CUSTOMARY,))
        if customary.size == 0:
            return
        lords = ppl.landlord[customary]
        opened = (lords >= 0) & (
            self.rng.random(customary.size) < p.inplace_conversion_rate
        )
        selected = customary[opened]
        if selected.size == 0:
            return
        p_convert = self._p_convert_by_estate[ppl.landlord[selected]]
        selected = selected[self.rng.random(selected.size) < p_convert]

        for j in selected:
            j = int(j)
            if not ppl.holdings[j]:
                continue
            landlord = int(ppl.landlord[j])
            parcels = sorted(ppl.holdings[j])
            cost = p.chi * self.r_hat[landlord] * len(parcels)
            if ppl.w[j] >= cost:
                ppl.w[j] -= cost
                for parcel in parcels:
                    if p.competitive_allocation:
                        self.parcel_rent[parcel] = self._bid(j, parcel)
                    self._assign(j, parcel, Tenure.LEASEHOLD)
                self.n_inplace += 1
            else:
                # Cannot meet the fine: dispossessed, and the holding goes to the market.
                self._release(j, to_landless=True)
                self.n_inplace_dispossessed += 1
                for parcel in parcels:
                    self.parcel_tenure[parcel] = int(Tenure.LEASEHOLD)
                    if not self._try_fill(parcel, Tenure.LEASEHOLD):
                        self.queue[parcel] = Tenure.LEASEHOLD

    # -- 7. which tenancies fall vacant ----------------------------------------------------------
    def _step7_vacancies(self) -> list[tuple[int, str]]:
        """Return ``(person, kind)`` for every tenancy vacating, kind in {succession, crisis}."""
        p = self.p
        ppl = self.people
        n = ppl.n
        # Recomputed rather than reused from step 2: under the household population rule new
        # agents are appended during step 4, so the cached arrays are the wrong length.
        agg = self._aggregate_holdings()
        masks = self._state_masks(n)
        held = masks["occupied"] & (agg["size"] > 0)

        evicted = held & (ppl.shortfall[:n] >= p.tau)

        with np.errstate(invalid="ignore", divide="ignore"):
            commons_fraction = np.where(
                agg["size"] > 0, agg["commons"] / np.maximum(agg["size"], 1), 0.0
            )
        hazard = p.eta_1 * np.maximum(0.0, p.subsistence_output - ppl.y[:n])
        if p.enable_enclosure:
            # A household is exposed to its *own* lord's enclosure, not to the national figure.
            # Under the national rule every estate carries the same value, so this reduces to the
            # scalar term exactly; under the local rule it is what makes enclosure spatial.
            owner = ppl.landlord[:n]
            local_enclosure = np.where(
                owner >= 0, self.enclosure_by_estate[np.maximum(owner, 0)], self.enclosure
            )
            hazard = hazard + p.eta_2 * local_enclosure * commons_fraction

        # One draw per person per event type, applied with the same precedence as before:
        # eviction first, then the ecological/enclosure hazard, then ordinary succession.
        crisis = held & ~evicted & (self.rng.random(n) < hazard)
        succession = held & ~evicted & ~crisis & (self.rng.random(n) < p.eta_0)

        self.n_evictions = int(evicted.sum())
        self.n_successions = int(succession.sum())

        # An extinct household's parcels are already vacant in every sense except the lord's
        # bookkeeping, so they take precedence: there is nobody left to evict or succeed.
        extinct = {j for j in self._extinct if held[j]}
        self.n_extinctions = len(extinct)

        vacancies: list[tuple[int, str]] = [(int(j), "extinct") for j in sorted(extinct)]
        vacancies += [
            (int(j), "crisis") for j in np.nonzero(evicted | crisis)[0] if j not in extinct
        ]
        vacancies += [
            (int(j), "succession") for j in np.nonzero(succession)[0] if j not in extinct
        ]
        return vacancies

    # -- 8. vacancy resolution ---------------------------------------------------------------------
    def _step8_resolve(self, vacancies: list[tuple[int, str]]) -> None:
        # Retry the queue first: a parcel nobody could afford last period gets another chance
        # against this period's fiscal pressure, and meanwhile earns its landlord nothing.
        for parcel, tenure in list(self.queue.items()):
            if self._try_fill(parcel, tenure):
                del self.queue[parcel]

        order = self.rng.permutation(len(vacancies))
        for idx in order:
            person, kind = vacancies[idx]
            if kind == "succession":
                self._resolve_succession(person)
            elif kind == "extinct":
                # Escheat: the line has failed, so the land reverts with no dispossession to
                # account for. Resolved exactly as a crisis vacancy otherwise.
                self._resolve_crisis(person, to_landless=False)
            else:
                self._resolve_crisis(person)

    def _resolve_succession(self, person: int) -> None:
        """An heir succeeds, but conveyance is still a moment at which the landlord may act.

        SPEC NOTE: ``eta_0`` is read strictly as the *conveyance* hazard -- the generational
        turnover at which the tenancy is regranted and the lord can therefore act on its terms.
        It is population-neutral: the household passes to the heir at its current size, and the
        members who actually die do so through :attr:`Params.mortality_0` in step 4c. Were the
        conveyance also to remove a member, the two processes would double-count the same deaths
        and the population would fall at ``eta_0 + mortality_0``.
        """
        p = self.p
        ppl = self.people
        landlord = int(ppl.landlord[person])
        outgoing = Tenure(int(ppl.state[person]))
        parcels = self._release(person, to_landless=False)
        ppl.state[person] = int(Tenure.DECEASED)  # succeeded by an heir, not lost to industry

        # SPEC NOTE: the paper gives the heir a wealth fraction and resets capital to the
        # traditional baseline, but is silent on the improving disposition. We reset it: the
        # heir is a new person, and a disposition formed by one lifetime's experience of
        # competition is not inherited.
        heir = ppl.add(
            state=outgoing,
            w=p.xi_inherit * ppl.w[person],
            k=p.k_trad,
            iota=p.iota_tenant_0,
            mobility=self.draws.uniform(*p.mobility_range),
            landlord=landlord,
            size=int(ppl.size[person]),
            need=self._draw_need(),
        )

        # Under impartible inheritance the holding passes whole to one heir and the rest of the
        # family has to find its living elsewhere. This is the strong proletarianisation pump:
        # it makes the flow into wage labour a function of the inheritance custom rather than of
        # how hard the holding is pressed.
        if p.impartible_inheritance and ppl.size[heir] > 1:
            shed = int(ppl.size[heir]) - 1
            if self._shed_member(heir, members=shed) >= 0:
                self.n_disinherited += shed

        remaining = [k for k in parcels if not self._try_engross(k)]
        if not remaining:
            ppl.state[heir] = int(Tenure.LANDLESS)
            ppl.mark_landless(heir)
            ppl.landlord[heir] = -1
            return

        # Conversion is defined as Customary -> Leasehold, so only a customary conveyance
        # presents the landlord with the decision; an heir to a leasehold or freehold holding
        # simply continues on the outgoing terms.
        if outgoing != Tenure.CUSTOMARY:
            for k in remaining:
                self._assign(heir, k, outgoing)
            return

        converted = self.draws.random() < self._p_convert(landlord)
        if converted:
            cost = p.chi * self.r_hat[landlord]
            if ppl.w[heir] >= cost:
                ppl.w[heir] -= cost  # the entry fine, levied on the heir at conveyance
                for k in remaining:
                    if p.competitive_allocation:
                        self.parcel_rent[k] = self._bid(heir, k)
                    self._assign(heir, k, Tenure.LEASEHOLD)
            else:
                # Cannot meet the fine: dispossessed, and the parcels go to the open market.
                ppl.state[heir] = int(Tenure.LANDLESS)
                ppl.mark_landless(heir)
                ppl.landlord[heir] = -1
                for k in remaining:
                    if not self._try_fill(k, Tenure.LEASEHOLD):
                        self.queue[k] = Tenure.LEASEHOLD
            return

        tenure = (
            Tenure.FREEHOLD
            if self.draws.random() < self._p_freehold()
            else Tenure.CUSTOMARY
        )
        for k in remaining:
            self._assign(heir, k, tenure)

    def _resolve_crisis(self, person: int, to_landless: bool = True) -> None:
        """No customary claimant remains, so each parcel falls to the landlord's disposal.

        ``to_landless=False`` is the extinction case: there is no household left to become
        landless, so it is marked deceased instead of being added to the labour pool.
        """
        ppl = self.people
        p_engross = self.p.enable_engrossment
        landlord = int(ppl.landlord[person])
        parcels = self._release(person, to_landless=to_landless)
        if not to_landless:
            ppl.state[person] = int(Tenure.DECEASED)

        for parcel in parcels:
            # Nothing between the scan and the two uses below touches an occupant or a state:
            # a failed engrossment draw leaves the lattice exactly as the scan found it.
            exposed = self._exposed_neighbours(parcel) if p_engross else None
            if self._try_engross(parcel, exposed):
                continue

            standing = Tenure(int(self.parcel_tenure[parcel]))
            if standing == Tenure.CUSTOMARY:
                if self.draws.random() < self._p_convert(landlord):
                    tenure = Tenure.LEASEHOLD
                    self.n_conversions_at_vacancy += 1
                elif self.draws.random() < self._p_freehold():
                    tenure = Tenure.FREEHOLD
                    self.n_freehold_diversions += 1
                else:
                    # Custom attaches to the land: the parcel keeps its old fixed rent and
                    # its next tenant takes it on customary terms.
                    if not self._try_fill(parcel, Tenure.CUSTOMARY):
                        self.queue[parcel] = Tenure.CUSTOMARY
                    continue
            else:
                # Already market-exposed. Customary right here is extinguished and does not
                # return, so the parcel is simply re-let on its standing tenure.
                tenure = standing

            if not self._try_fill(parcel, tenure, exposed):
                self.queue[parcel] = tenure

    # -- resolution primitives -----------------------------------------------------------------------
    def _refresh_period_probabilities(self) -> None:
        """Evaluate the conversion probabilities once for the period.

        Every landlord term they depend on is fixed by the time any vacancy resolves --
        ``delta_relative`` in step 5, ``observed`` and ``theta_eff`` in step 6 -- and none of
        them is touched by steps 7 and 8. Recomputing a sigmoid per event was therefore
        recomputing the same number some hundreds of thousands of times a run.
        """
        p = self.p
        self._p_convert_by_estate = _sigmoid(
            p.alpha_0
            + p.alpha_1 * self.delta_relative
            + p.alpha_2 * self.observed
            - p.alpha_3 * self.theta_eff
        )
        gate = 1.0 if self.t >= p.t_star else 0.0
        self._p_freehold_now = float(
            _sigmoid(p.lambda_0 + p.lambda_1 * p.theta - p.lambda_2 * gate)
        )
        # The engrossment hazard still varies with the winning neighbour's capital, so only
        # its landlord half can be lifted out of the loop.
        self._engross_base = p.psi_0 + p.psi_1 * self.delta_relative

    def _p_convert(self, landlord: int) -> float:
        return float(self._p_convert_by_estate[landlord])

    def _p_freehold(self) -> float:
        return self._p_freehold_now

    def _exposed_neighbours(self, parcel: int) -> list[int]:
        """Occupants of ``parcel``'s same-estate neighbours that are market-exposed.

        In neighbour order and with repeats, because a household holding two of the neighbours
        appears twice: both the engrossment scan and the auction below walked this same list
        and both break ties on first-seen, so the order and the repeats are part of the answer.

        Shared between the two because engrossment and the auction are the same event seen
        from the two sides -- when the lord declines to consolidate, the parcel goes under the
        hammer to the very neighbours just scored -- and walking eight neighbours twice was
        pure duplication.
        """
        occupant = self.occupant
        state = self.people.state
        lease = int(Tenure.LEASEHOLD)
        free = int(Tenure.FREEHOLD)
        found: list[int] = []
        for q in self.geo.neighbours[parcel]:
            j = int(occupant[q])
            if j < 0:
                continue
            st = state[j]
            if st == lease or st == free:
                found.append(j)
        return found

    def _try_engross(self, parcel: int, exposed: list[int] | None = None) -> bool:
        """Absorb ``parcel`` into the most capital-intensive market-exposed neighbour."""
        p = self.p
        if not p.enable_engrossment:
            return False
        ppl = self.people
        # Only market-exposed neighbours on the same estate can engross: the paper's j* is the
        # most capital-intensive Leasehold or Freehold tenant already adjacent to the parcel.
        # Ranked by productivity -- output per parcel already farmed -- so that the land goes
        # to whoever is demonstrably farming best, which is Wood's "success would breed
        # success...while others lost access altogether". Scored in the same pass that finds
        # the candidates: this runs on every released parcel, so the per-neighbour Python
        # overhead is the whole cost of the step.
        if exposed is None:
            exposed = self._exposed_neighbours(parcel)
        y = ppl.y
        n_held = ppl.n_held
        best = -1
        best_score = 0.0
        for j in exposed:
            held = n_held[j]
            score = y[j] / (held if held > 0 else 1)
            if best < 0 or score > best_score:
                best = j
                best_score = score
        if best < 0:
            return False
        landlord = int(self.parcel_landlord[parcel])
        q = _sigmoid(self._engross_base[landlord] + p.psi_2 * ppl.k[best])
        if self.draws.random() >= q:
            return False
        # Engrossment is the successful farmer winning the parcel; under competitive
        # allocation the rent it carries is what their own productivity would bid.
        if p.competitive_allocation and ppl.state[best] == int(Tenure.LEASEHOLD):
            self.parcel_rent[parcel] = self._bid(best, parcel)
        self._assign(best, parcel, Tenure(int(ppl.state[best])))
        self.n_engrossments += 1
        return True

    def _bid(self, person: int, parcel: int) -> float:
        """What ``person`` would pay for ``parcel``: a share of what they could raise on it.

        A sitting neighbour bids on *revealed* productivity -- what they are already getting
        per parcel -- while a landless entrant can only bid what an unimproved holding worked
        by one household would yield. That asymmetry is the whole of Wood's "success would
        breed success": the improver can always outbid the entrant for the land next door.
        """
        p = self.p
        ppl = self.people
        held = int(ppl.n_held[person])
        if held and ppl.y[person] > 0:
            expected = ppl.y[person] / held
        else:
            # An entrant can only bid what an unimproved holding worked by their own family
            # would yield, so a larger landless household can outbid a smaller one -- the
            # family-labour advantage, priced into the auction.
            labour_exp = 1.0 - p.cobb_phi - p.cobb_capital
            family = max(float(ppl.size[person]) * p.household_labour, 1e-9)
            expected = (
                max(self.phi[parcel], 0.0) ** p.cobb_phi
                * p.k_trad**p.cobb_capital
                * family**labour_exp
            )
        # Struck in money, not in produce: revenue is priced at the going rate, so a rent
        # denominated in bushels would not be comparable with the income it is paid out of.
        return float(p.theta_rent * expected * self._price())

    def _try_fill(
        self, parcel: int, tenure: Tenure, exposed: list[int] | None = None
    ) -> bool:
        """Let the parcel to the highest bidder who can also meet the entry fine.

        Taking up a *customary* holding is a matter of custom rather than of the market, so it
        is neither auctioned nor fined; only market-exposed tenures go under the hammer.
        """
        p = self.p
        ppl = self.people
        landlord = int(self.parcel_landlord[parcel])
        best_w, best_size = ppl.landless_best()

        if tenure == Tenure.CUSTOMARY or not p.competitive_allocation:
            cost = p.chi * self.r_hat[landlord] if tenure != Tenure.CUSTOMARY else 0.0
            # The wealthiest household is the only one that can settle the question: if it
            # cannot meet the fine then nobody can, and if it can then it is also the one an
            # ascending argmax over the affordable would have picked.
            if best_w < 0 or ppl.w[best_w] < cost:
                return False
            self._install(best_w, parcel, tenure, cost, rent=None)
            return True

        # Landless households differ only in wealth and in how many hands they bring, and both
        # raise what they can bid, so it is enough to enter the best of each: the wealthiest, and
        # the largest. Sitting neighbours bid individually on their own productivity.
        candidates: list[tuple[float, int]] = []
        if best_w >= 0:
            entrants = {best_w, best_size}
            candidates.extend((self._bid(j, parcel), j) for j in entrants)
        # Sitting neighbours enter the auction only where the lord is willing to consolidate:
        # engrossment and the auction are the same event seen from the two sides.
        if p.enable_engrossment:
            if exposed is None:
                exposed = self._exposed_neighbours(parcel)
            candidates.extend((self._bid(j, parcel), j) for j in exposed)
        if not candidates:
            return False

        # "Whatever rent the market would bear": the winning bid becomes the parcel's rent.
        viable = [(b, j) for b, j in candidates if ppl.w[j] >= p.chi * b]
        if not viable:
            return False
        bid, winner = max(viable, key=lambda c: c[0])
        self._install(winner, parcel, tenure, cost=p.chi * bid, rent=bid)
        return True

    def _install(
        self, person: int, parcel: int, tenure: Tenure, cost: float, rent: float | None
    ) -> None:
        p = self.p
        ppl = self.people
        ppl.w[person] -= cost
        if not ppl.holdings[person]:
            # A fresh entrant stocks the holding from scratch. Capital is physical and was
            # lost with any previous tenancy; the improving disposition is not -- it travels
            # with the person, so an experienced farmer does not arrive as a blank slate.
            ppl.k[person] = p.k_trad
            ppl.shortfall[person] = 0
        if rent is not None:
            self.parcel_rent[parcel] = rent
        self._assign(person, parcel, tenure)

    # -- 9. rent reset -------------------------------------------------------------------------------
    def _step9_rent_reset(self) -> None:
        """rhat_i(t+1) = theta_rent * mean realised Leasehold output on the estate."""
        ppl = self.people
        idx = ppl.in_states((Tenure.LEASEHOLD,))
        lord = ppl.landlord[idx]
        keep = lord >= 0
        lord = lord[keep].astype(np.intp)
        totals = np.bincount(lord, weights=ppl.y[idx][keep], minlength=self.n_landlords)
        counts = np.bincount(lord, minlength=self.n_landlords)
        active = counts > 0
        # In money, for the same reason as the bid: this benchmark is compared against, and
        # charged out of, revenue that has been priced.
        self.r_hat[active] = (
            self.p.theta_rent * totals[active] / counts[active] * self._price()
        )

    # -------------------------------------------------------------------------------------------------
    # recording
    # -------------------------------------------------------------------------------------------------
    def _record(self, demand: float, pool: int) -> None:
        """Write the period's panels, and its aggregates if this period is a recording one.

        The panels are the parcel-by-period history every spatial and event-study statistic is
        computed from, so they are never skipped. The aggregates are a time series for the
        plots, and are taken every ``record_every`` periods -- always including the first and
        the last, so that opening and closing values are exact.
        """
        p = self.p
        if self.t % p.record_every == 0 or self.t == p.n_steps - 1:
            self._record_aggregates(demand, pool)
        self._record_panels()

    def _record_aggregates(self, demand: float, pool: int) -> None:
        ppl = self.people
        n = ppl.n
        states = ppl.state[:n]
        counts = {s: int((states == int(s)).sum()) for s in Tenure}

        # Holdings changed during step 8, so the aggregates are recomputed rather than reused.
        agg = self._aggregate_holdings()
        masks = self._state_masks(n)
        occupied = np.nonzero(masks["occupied"])[0]
        sizes = agg["size"][occupied]
        extra: dict[str, float] = {}

        # --- per-tenure means, so the classes can be compared rather than pooled ---------
        for tenure in (Tenure.CUSTOMARY, Tenure.LEASEHOLD, Tenure.FREEHOLD):
            tag = tenure.name.lower()
            sel = masks[tag]
            if not sel.any():
                for field in ("wealth", "capital", "iota", "output", "rent", "holding", "hired", "phi"):
                    extra[f"{tag}_{field}"] = np.nan
                continue
            extra[f"{tag}_wealth"] = float(ppl.w[:n][sel].mean())
            extra[f"{tag}_capital"] = float(ppl.k[:n][sel].mean())
            extra[f"{tag}_iota"] = float(ppl.iota[:n][sel].mean())
            extra[f"{tag}_output"] = float(ppl.y[:n][sel].mean())
            extra[f"{tag}_rent"] = float(ppl.rho[:n][sel].mean())
            extra[f"{tag}_holding"] = float(agg["size"][sel].mean())
            extra[f"{tag}_hired"] = float(ppl.hired[:n][sel].mean())
            total_size = agg["size"][sel].sum()
            extra[f"{tag}_phi"] = (
                float(agg["phi"][sel].sum() / total_size) if total_size > 0 else np.nan
            )

        # --- distributions, because means hide the concentration the theory predicts -----
        def quantiles(values: np.ndarray, tag: str) -> None:
            if len(values) == 0:
                for q in ("p10", "median", "p90", "max"):
                    extra[f"{tag}_{q}"] = np.nan
                return
            extra[f"{tag}_p10"] = float(np.percentile(values, 10))
            extra[f"{tag}_median"] = float(np.median(values))
            extra[f"{tag}_p90"] = float(np.percentile(values, 90))
            extra[f"{tag}_max"] = float(values.max())

        quantiles(ppl.w[occupied] if len(occupied) else np.array([]), "tenant_wealth")
        quantiles(ppl.k[occupied] if len(occupied) else np.array([]), "tenant_capital")
        quantiles(sizes, "holding")
        landless_w = ppl.w[:n][masks["landless"]]
        quantiles(landless_w, "landless_wealth")
        quantiles(self.W, "landlord_wealth")

        # --- landlords -------------------------------------------------------------------
        total_t, leasehold_t, customary_t = self._estate_tenancies()
        hired_by_landlord = np.zeros(self.n_landlords)
        if len(occupied) and self.n_landlords > 0:
            owners = ppl.landlord[occupied]
            valid = owners >= 0
            if valid.any():
                hired_by_landlord = np.bincount(
                    owners[valid], weights=ppl.hired[occupied][valid],
                    minlength=self.n_landlords,
                )
        extra.update(
            {
                "landlord_wealth_mean": float(self.W.mean()),
                "landlord_receipts_mean": float(self.estate_receipts.mean()),
                "landlord_customary_rent": float(
                    self.estate_receipts.mean() - float(self.customary_fines.mean())
                ),
                "landlord_fines_mean": float(self.customary_fines.mean()),
                "landlord_theta_eff": float(self.theta_eff.mean()),
                "landlord_observed": float(self.observed.mean()),
                "tenancies_per_estate": float(total_t.mean()),
                "leasehold_share_of_estates": float(
                    np.mean(np.where(total_t > 0, leasehold_t / np.maximum(total_t, 1), 0.0))
                ),
                "labourers_per_landlord": float(hired_by_landlord.mean()),
                "labourers_per_leasehold_tenant": float(
                    ppl.hired[:n][masks["leasehold"]].mean()
                    if masks["leasehold"].any()
                    else 0.0
                ),
            }
        )

        # --- land ----------------------------------------------------------------------
        occupied_parcels = self.occupant >= 0
        extra.update(
            {
                "phi_median": float(np.median(self.phi)),
                "phi_p10": float(np.percentile(self.phi, 10)),
                "phi_p90": float(np.percentile(self.phi, 90)),
                "phi_occupied": float(self.phi[occupied_parcels].mean())
                if occupied_parcels.any()
                else np.nan,
                "phi_depletion": float(1.0 - self.phi.mean() / self.phi_bar.mean()),
            }
        )

        # --- productivity: the quantity the whole theory is about ---------------------------
        # Land productivity is output per parcel worked; labour productivity is output per
        # person working it, household plus hired. The second is what Brenner means when he
        # says improvement freed people to leave the land.
        family = self._family_labour(n)
        total_output = float(ppl.y[occupied].sum()) if len(occupied) else 0.0
        worked = float(agg["size"][occupied].sum()) if len(occupied) else 0.0
        workers = (
            float((family[occupied] + ppl.hired[occupied]).sum()) if len(occupied) else 0.0
        )
        extra.update(
            {
                "output_per_parcel": total_output / worked if worked > 0 else np.nan,
                "output_per_worker": total_output / workers if workers > 0 else np.nan,
                "output_per_capital": (
                    total_output / float(ppl.k[occupied].sum())
                    if len(occupied) and ppl.k[occupied].sum() > 0
                    else np.nan
                ),
                "goods_price": self.goods_price,
                "urban_population": self.urban_population,
                "urban_demand": self.urban_population * self.p.urban_consumption,
                "marketed_output": float(
                    ppl.y[:n][masks["leasehold"] | masks["freehold"]].sum()
                ),
                "agricultural_population": len(occupied),
                "non_agricultural_share": (
                    self.urban_population
                    / max(ppl.head_count() + self.urban_population, 1)
                ),
            }
        )

        # --- demography -------------------------------------------------------------------
        # Households and persons are now different things, and every share has to say which it
        # means. Tenure shares stay per household (a tenancy is held by a household); the
        # proletarian share is reported both ways, because a landless household of four is four
        # people looking for wages but only one bidder at an auction.
        live = ppl.live()
        heads = ppl.size[live].astype(float)
        landless_sel = masks["landless"]
        urban_sel = ppl.state[:n] == int(Tenure.EXITED)
        # Households created during steps 4c and 8 are not in the step-4 surplus vector, so it
        # is padded rather than indexed past its end; their surplus is recorded next period.
        surplus = np.zeros(n)
        known = min(len(self._surplus), n)
        surplus[:known] = self._surplus[:known]
        occupied_persons = float(ppl.size[occupied].sum()) if len(occupied) else 0.0
        landless_persons = float(ppl.size[:n][landless_sel].sum())
        extra.update(
            {
                "households": int(len(live)),
                "mean_household_size": float(heads.mean()) if len(heads) else np.nan,
                "max_household_size": float(heads.max()) if len(heads) else np.nan,
                "occupied_household_size": float(ppl.size[occupied].mean())
                if len(occupied)
                else np.nan,
                "landless_household_size": float(ppl.size[:n][landless_sel].mean())
                if landless_sel.any()
                else np.nan,
                "population_occupied": occupied_persons,
                "population_landless": landless_persons,
                "share_landless_persons": (
                    landless_persons / (occupied_persons + landless_persons)
                    if occupied_persons + landless_persons > 0
                    else np.nan
                ),
                "returns_from_industry": self.n_returns,
                "return_persons": self.n_return_persons,
                "net_migration_persons": self.n_return_persons - self.n_exit_persons,
                "family_labour_supply": float(family[masks["occupied"]].sum()),
                "deaths": self.n_deaths,
                "births_urban": self.n_births_urban,
                "deaths_urban": self.n_deaths_urban,
                "births_rural": self.n_births - self.n_births_urban,
                "deaths_rural": self.n_deaths - self.n_deaths_urban,
                "partitions": self.n_partitions,
                "partition_persons": self.n_partition_persons,
                "disinherited": self.n_disinherited,
                "extinctions": self.n_extinctions,
                "exit_persons": self.n_exit_persons,
                "natural_increase": self.n_births - self.n_deaths,
                "mean_surplus_per_head": float(
                    surplus[live].mean() if len(live) else np.nan
                ),
                "surplus_per_head_occupied": float(
                    surplus[occupied].mean() if len(occupied) else np.nan
                ),
                "surplus_per_head_landless": float(
                    surplus[landless_sel].mean() if landless_sel.any() else np.nan
                ),
                "vacant_parcels": int((self.occupant < 0).sum()),
                "share_parcels_vacant": float(
                    (self.occupant < 0).sum() / max(self.geo.n_parcels, 1)
                ),
            }
        )

        # --- total population as a composition ------------------------------------------------
        # Every living person sits in exactly one of five positions, rural or urban, so these
        # shares sum to one and the series can be read the way the tenure composition is: what
        # changed is where people are, against a total that is itself an outcome. Reporting the
        # rural headcount alone cannot distinguish a countryside losing people to towns from one
        # failing to reproduce, and those are different claims.
        total_persons = float(ppl.size[ppl.present()].sum())
        urban_persons = float(ppl.size[:n][urban_sel].sum())
        position_persons = {
            "customary": float(ppl.size[:n][masks["customary"]].sum()),
            "leasehold": float(ppl.size[:n][masks["leasehold"]].sum()),
            "freehold": float(ppl.size[:n][masks["freehold"]].sum()),
            "landless": landless_persons,
            "urban": urban_persons,
        }
        extra["total_population"] = total_persons
        extra["population_urban"] = urban_persons
        for tag, value in position_persons.items():
            extra[f"persons_{tag}"] = value
            extra[f"share_persons_{tag}"] = (
                value / total_persons if total_persons > 0 else np.nan
            )
        # Land and labour productivity split by tenure, so the classes can be compared.
        for tenure in (Tenure.CUSTOMARY, Tenure.LEASEHOLD, Tenure.FREEHOLD):
            tag = tenure.name.lower()
            sel = masks[tag]
            land = agg["size"][sel].sum()
            labour = (family[sel] + ppl.hired[:n][sel]).sum()
            out = ppl.y[:n][sel].sum()
            extra[f"{tag}_output_per_parcel"] = float(out / land) if land > 0 else np.nan
            extra[f"{tag}_output_per_worker"] = float(out / labour) if labour > 0 else np.nan

        # --- flows this period -----------------------------------------------------------
        extra.update(
            {
                "investment": self.investment_flow,
                "wage_bill": self.wage_bill_total,
                "fines_flow": self.fines_total,
                "evictions": self.n_evictions,
                "successions": self.n_successions,
                "engrossments": self.n_engrossments,
                "conversions_at_vacancy": self.n_conversions_at_vacancy,
                "conversions_inplace": self.n_inplace,
                "freehold_diversions": self.n_freehold_diversions,
                "exits_to_industry": self.n_exits,
                "extinctions_at_vacancy": self.n_extinctions,
                "labour_ration": float(min(1.0, pool / demand)) if demand > 0 else np.nan,
                "hired_total": float(ppl.hired[: ppl.n].sum()),
            }
        )

        self.history.append(
            {
                "t": self.t,
                "customary": counts[Tenure.CUSTOMARY],
                "leasehold": counts[Tenure.LEASEHOLD],
                "freehold": counts[Tenure.FREEHOLD],
                "landless": counts[Tenure.LANDLESS],
                "exited": counts[Tenure.EXITED],
                "deceased": counts[Tenure.DECEASED],
                "farm_gini": _gini(sizes),
                "output": float(ppl.y[occupied].sum()),
                "wage": self.wage,
                "price_index": self.price_index,
                "enclosure": self.enclosure,
                "mean_rhat": float(self.r_hat.mean()),
                "mean_real_customary_rent": float(
                    ppl.rho[:n][masks["customary"]].mean() if masks["customary"].any() else 0.0
                ),
                "mean_iota_tenant": float(ppl.iota[occupied].mean()) if len(occupied) else 0.0,
                "mean_iota_landlord": float(self.iota_landlord.mean()),
                "mean_iota_customary": float(
                    ppl.iota[:n][masks["customary"]].mean() if masks["customary"].any() else 0.0
                ),
                "mean_iota_leasehold": float(
                    ppl.iota[:n][masks["leasehold"]].mean() if masks["leasehold"].any() else 0.0
                ),
                "mean_fiscal_pressure": float(self.delta_fiscal.mean()),
                "mean_fiscal_pressure_rel": float(self.delta_relative.mean()),
                "mean_capital": float(ppl.k[occupied].mean()) if len(occupied) else 0.0,
                "labour_demand": demand,
                "labour_pool": pool,
                "queued_parcels": len(self.queue),
                "mean_parcel_rent": float(
                    self.parcel_rent[self.parcel_tenure == int(Tenure.LEASEHOLD)].mean()
                    if (self.parcel_tenure == int(Tenure.LEASEHOLD)).any()
                    else 0.0
                ),
                "inplace_conversions": self.n_inplace,
                "inplace_dispossessed": getattr(self, "n_inplace_dispossessed", 0),
                "births": self.n_births,
                # Persons, not households: see ``households`` in extra for the count of units.
                "population": ppl.head_count(),
                "customary_fines": float(self.customary_fines.sum()),
                "parcels_leasehold_tenure": int(
                    (self.parcel_tenure == int(Tenure.LEASEHOLD)).sum()
                ),
                "converted_parcels": int((self.first_conversion >= 0).sum()),
                "mean_phi": float(self.phi.mean()),
                **extra,
            }
        )

    def _record_panels(self) -> None:
        """Parcel-level panel for the RQ1 event study. The unit is the *parcel*, not the
        person: conversion happens at a vacancy and replaces the sitting tenant, so the parcel
        is what has a continuous history spanning its own conversion date."""
        ppl = self.people
        row = self.t
        occ = self.occupant
        held = occ >= 0
        idx = occ[held]
        self.panel_state[row, held] = ppl.state[idx]
        # The household's identity, not its slot: slots are recycled, so the index would make
        # two unrelated households look like one continuing occupant.
        self.panel_occupant[row, held] = ppl.uid[idx]
        self.panel_holding[row, held] = ppl.n_held[idx]
        self.panel_iota[row, held] = ppl.iota[idx]
        self.panel_k[row, held] = ppl.k[idx]
        self.panel_rho[row, held] = ppl.rho[idx]
        self.panel_y[row, held] = ppl.y[idx]

    def run(self, progress: bool = False) -> "Model":
        for step in range(self.p.n_steps):
            self.step()
            if progress and (step + 1) % 25 == 0:
                last = self.history[-1]
                print(
                    f"  t={step + 1:4d}  cust={last['customary']:5d} lease={last['leasehold']:5d} "
                    f"free={last['freehold']:4d} landless={last['landless']:5d} "
                    f"gini={last['farm_gini']:.3f}"
                )
        return self


def _gini(values: np.ndarray) -> float:
    """Gini coefficient; 0 for an empty or degenerate population."""
    if len(values) == 0:
        return 0.0
    v = np.sort(np.asarray(values, dtype=float))
    total = v.sum()
    if total <= 0:
        return 0.0
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum()) / (n * total) - (n + 1.0) / n)

