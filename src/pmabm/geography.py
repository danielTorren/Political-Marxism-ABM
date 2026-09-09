"""England-shaped lattice, estates and neighbour relations.

Implements the runtime half of the paper's Spatial structure section. The GIS work itself is
done once, offline, by :mod:`pmabm.build_geography`; everything here reads only the cached
artifact and is pure numpy.

Two neighbour relations are built, because the paper keeps them at different levels:

* **parcel adjacency** -- Moore (8-connected) neighbours *of the same estate*, used by
  engrossment;
* **landlord awareness** -- estates whose seed points lie within ``awareness_radius``, used by
  the observed-outcome and mobility spread channels.

Everything here is *causal*: each field is read by the model's own dynamics. Measurement
constructs derived from the lattice belong in :mod:`pmabm.metrics` instead -- see
:func:`pmabm.metrics.spread_variogram`, which replaced an earlier ``seed_origin`` /
``dist_from_seed`` pair that lived on :class:`Geography` but was never read by any decision rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Params

DEFAULT_ARTIFACT = Path("data/geography/england_alc.json")


class GeographyMissing(FileNotFoundError):
    """Raised when the cached geography artifact has not been built yet."""


@dataclass(frozen=True)
class Geography:
    """Immutable spatial scaffolding for a run."""

    shape: tuple[int, int]
    """(n_rows, n_cols) of the lattice covering England's bounding box."""
    xy: np.ndarray
    """(P, 2) int array of (row, col) lattice coordinates, one row per parcel."""
    phi_bar: np.ndarray
    """(P,) ecological carrying capacity, from ALC grades plus within-region jitter."""
    region: np.ndarray
    """(P,) index of the ALC sampling region a parcel belongs to."""
    landlord: np.ndarray
    """(P,) owning landlord, by Voronoi tessellation around landlord seed points."""
    neighbours: list[np.ndarray]
    """Per parcel: Moore-adjacent parcels *belonging to the same estate*."""
    landlord_neighbours: list[np.ndarray]
    """Per landlord: other landlords within the awareness radius."""
    estate_parcels: list[np.ndarray]
    """Per landlord: the parcels of its estate."""
    county: np.ndarray
    """(P,) index of the historic county a parcel belongs to."""
    county_names: list[str]
    county_grade: np.ndarray
    """(C,) area-weighted mean ALC grade per county; 1 is excellent, 5 very poor."""

    @property
    def n_parcels(self) -> int:
        return len(self.xy)

    @property
    def n_landlords(self) -> int:
        return len(self.estate_parcels)

    @property
    def n_counties(self) -> int:
        return len(self.county_names)


def _ring_crossings(poly: np.ndarray, px: np.ndarray, py: np.ndarray, inside: np.ndarray) -> None:
    """XOR the even-odd crossing parity of one closed ``poly`` into ``inside``, in place.

    Only points inside the ring's bounding box are tested. That is exact rather than an
    approximation, because every ring in the artifact is closed: a point above or below the box
    straddles no edge at all, a point to its right sees every crossing fall behind it, and a
    point to its left sees the rightward ray cut a closed curve an even number of times. All
    three contribute a parity of zero, which is what XOR-ing nothing into ``inside`` means.

    Without the box the county pass tested all 7,400-odd parcels against all 34,500-odd county
    edges and cost some two seconds -- more than a third of a whole run -- to answer a question
    whose answer is fixed by the lattice.
    """
    x1, y1 = poly[:-1, 0], poly[:-1, 1]
    x2, y2 = poly[1:, 0], poly[1:, 1]
    xlo, xhi = poly[:, 0].min(), poly[:, 0].max()
    ylo, yhi = poly[:, 1].min(), poly[:, 1].max()
    cand = np.nonzero((px >= xlo) & (px <= xhi) & (py >= ylo) & (py <= yhi))[0]
    if cand.size == 0:
        return
    cx = px[cand][:, None]
    cy = py[cand][:, None]
    parity = np.zeros(cand.size, dtype=bool)
    # Process edges in chunks to bound peak memory on large boundary files. XOR-ing each
    # chunk's parity is the parity of the whole, so the chunking is invisible to the result.
    for start in range(0, len(x1), 2000):
        sl = slice(start, start + 2000)
        ex1, ey1, ex2, ey2 = x1[sl], y1[sl], x2[sl], y2[sl]
        straddles = (ey1 > cy) != (ey2 > cy)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (ex2 - ex1) * (cy - ey1) / (ey2 - ey1) + ex1
        parity ^= (straddles & (cx < x_cross)).sum(axis=1) % 2 == 1
    inside[cand] ^= parity


def _rings_to_mask(rings: list, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test of a grid of centroids against boundary ``rings``.

    Even-odd (crossing number) accumulated across every ring handles both offshore islands and
    interior holes without needing to know each ring's winding direction.
    """
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x.ravel()
    py = grid_y.ravel()
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        _ring_crossings(np.asarray(ring, dtype=float), px, py, inside)
    return inside.reshape(grid_y.shape)


def _rings_contain(rings: list, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon of scattered points against one area's rings."""
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        poly = np.asarray(ring, dtype=float)
        if len(poly) < 4:
            continue
        _ring_crossings(poly, px, py, inside)
    return inside


def _ring_centroid(rings: list) -> tuple[float, float]:
    points = np.concatenate([np.asarray(r) for r in rings])
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _fill_nearest(values: np.ndarray) -> np.ndarray:
    """Replace NaNs with the value of the nearest finite cell (brute force; grid is tiny)."""
    filled = values.copy()
    missing = np.argwhere(~np.isfinite(values))
    present = np.argwhere(np.isfinite(values))
    if len(present) == 0:
        raise ValueError("ALC grid contains no finite values")
    for row, col in missing:
        distances = np.hypot(present[:, 0] - row, present[:, 1] - col)
        nearest = present[np.argmin(distances)]
        filled[row, col] = values[nearest[0], nearest[1]]
    return filled


def load_artifact(path: Path = DEFAULT_ARTIFACT) -> dict:
    if not path.exists():
        raise GeographyMissing(
            f"No geography artifact at {path}. Build it once with:\n"
            f"    uv run pmabm build-geography"
        )
    return json.loads(path.read_text(encoding="utf-8"))


#: Bounded memo of :func:`_lattice`. Four entries is enough for a sweep that varies ``L``
#: over a handful of values while holding the artifact fixed, which is every use there is.
_LATTICE_CACHE: dict[tuple, "_Lattice"] = {}
_LATTICE_CACHE_MAX = 4


@dataclass(frozen=True)
class _Lattice:
    """The part of a geography that depends on ``L`` and the artifact and on nothing else.

    Separated out and memoised because it is by far the most expensive part of the build and
    the sweeps rebuild the geography for *every* sample: both the sensitivity design and the
    emulator design vary ``zeta`` and ``awareness_radius``, which are geography fields, so
    without this the county pass and the boundary mask are paid thousands of times over to
    produce the same answer. Nothing here reads the rng or any parameter but ``L``.

    The arrays are handed to every :class:`Geography` built from the same lattice rather than
    copied, and are therefore marked read-only: a stray in-place write would otherwise corrupt
    every later run in the process.
    """

    shape: tuple[int, int]
    xy: np.ndarray
    county: np.ndarray
    grades: np.ndarray
    index_of: np.ndarray
    moore_flat: np.ndarray
    """Concatenated Moore-neighbour candidates of every parcel, ignoring estate membership."""
    moore_ptr: np.ndarray
    """Start offset of each parcel's slice of :attr:`moore_flat`, with a trailing total."""


def _freeze(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


def _lattice(params: Params, artifact: dict) -> _Lattice:
    """Build, or return the memoised copy of, the ``L``-only half of the geography."""
    counties = artifact["counties"]
    key = (
        int(params.L),
        tuple(artifact["bbox"]),
        len(artifact["outline_rings"]),
        tuple(c["code"] for c in counties),
    )
    hit = _LATTICE_CACHE.get(key)
    if hit is not None:
        return hit

    xmin, ymin, xmax, ymax = artifact["bbox"]
    # Square cells in real terms: the long axis gets L cells and the short axis fewer, so that
    # a lattice distance is proportional to a real distance in both directions.
    width, height = xmax - xmin, ymax - ymin
    cell = max(width, height) / params.L
    n_cols = max(1, int(np.ceil(width / cell)))
    n_rows = max(1, int(np.ceil(height / cell)))
    xs = xmin + (np.arange(n_cols) + 0.5) * cell
    ys = ymin + (np.arange(n_rows) + 0.5) * cell

    mask = _rings_to_mask(artifact["outline_rings"], xs, ys)
    if not mask.any():
        raise ValueError("No lattice cell fell inside the England boundary")
    rows, cols = np.nonzero(mask)
    xy = np.column_stack([rows, cols])

    # --- counties: the largest unit of land ---------------------------------------------
    # Each parcel is assigned to the historic county its centroid falls in, and inherits that
    # county's ALC-derived fertility baseline -- exactly the construction the paper describes.
    parcel_x, parcel_y = xs[cols], ys[rows]
    county = np.full(len(xy), -1, dtype=int)
    for index, entry in enumerate(counties):
        unassigned = county < 0
        if not unassigned.any():
            break
        hit_county = _rings_contain(entry["rings"], parcel_x[unassigned], parcel_y[unassigned])
        county[np.nonzero(unassigned)[0][hit_county]] = index

    # Coastal cells whose centroid misses every county polygon go to the nearest county.
    if (county < 0).any():
        centroids = np.array([_ring_centroid(c["rings"]) for c in counties])
        for pi in np.nonzero(county < 0)[0]:
            d = np.hypot(centroids[:, 0] - parcel_x[pi], centroids[:, 1] - parcel_y[pi])
            county[pi] = int(np.argmin(d))

    grades = np.array(
        [
            c["mean_alc_grade"] if c["mean_alc_grade"] is not None else np.nan
            for c in counties
        ]
    )
    if np.isnan(grades).any():  # a county with no graded land takes the national mean
        grades = np.where(np.isnan(grades), np.nanmean(grades), grades)

    # --- Moore-neighbour candidates ------------------------------------------------------
    # Which cells touch which is a fact about the lattice; *which of those count as adjacent*
    # depends on the estate layout and so is applied per run in :func:`build`. Held as one flat
    # array plus offsets so the per-run filter is a single vectorised comparison.
    index_of = -np.ones((n_rows, n_cols), dtype=int)
    index_of[rows, cols] = np.arange(len(xy))
    padded = -np.ones((n_rows + 2, n_cols + 2), dtype=int)
    padded[1:-1, 1:-1] = index_of
    r, c = rows + 1, cols + 1
    stacked = np.stack(
        [
            padded[r + dr, c + dc]
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        ],
        axis=1,
    )
    # Ascending within each parcel's slice, matching the order the offset loop produced.
    stacked.sort(axis=1)
    present = stacked >= 0
    moore_flat = stacked[present]
    moore_ptr = np.zeros(len(xy) + 1, dtype=np.intp)
    np.cumsum(present.sum(axis=1), out=moore_ptr[1:])

    _freeze(xy, county, grades, index_of, moore_flat, moore_ptr)
    built = _Lattice(
        shape=(n_rows, n_cols),
        xy=xy,
        county=county,
        grades=grades,
        index_of=index_of,
        moore_flat=moore_flat,
        moore_ptr=moore_ptr,
    )
    if len(_LATTICE_CACHE) >= _LATTICE_CACHE_MAX:
        _LATTICE_CACHE.pop(next(iter(_LATTICE_CACHE)))
    _LATTICE_CACHE[key] = built
    return built


def build(
    params: Params,
    rng: np.random.Generator,
    artifact: dict | None = None,
    fertility_range: tuple[float, float] = (3.0, 14.0),
) -> Geography:
    """Construct the lattice, estates and neighbour relations for one run.

    ``fertility_range`` maps ALC grades onto carrying capacity: the *worst* grade (5) takes the
    minimum and the *best* (1) the maximum, per the paper's "Grade 1 mapped to the highest
    score, Grade 5 to the lowest".
    """
    artifact = artifact if artifact is not None else load_artifact()
    counties = artifact["counties"]
    lat = _lattice(params, artifact)
    n_rows, n_cols = lat.shape
    xy, county, grades = lat.xy, lat.county, lat.grades

    lo, hi = fertility_range
    baseline = hi - (grades[county] - 1.0) / 4.0 * (hi - lo)
    if params.uniform_fertility:
        # RQ6: hold the national mean everywhere, so land quality has no spatial *pattern*.
        # The jitter below still varies parcel to parcel, so engrossment and turnover retain
        # local heterogeneity -- what is removed is the regional structure, which is the only
        # part of the construction that can give the transition an ecological location.
        baseline = np.full(len(xy), float(baseline.mean()))
    phi_bar = np.clip(
        baseline + rng.uniform(-params.zeta, params.zeta, size=len(xy)), 0.5, None
    )
    region = county  # fertility regions and counties are now the same thing

    # --- estates nest inside counties ------------------------------------------------------
    # Lords are seeded per county rather than nationally, so an estate never straddles a
    # county boundary and the county is a genuine upper tier rather than a label.
    landlord = np.full(len(xy), -1, dtype=int)
    seeds_list: list[np.ndarray] = []
    next_id = 0
    for c in range(len(counties)):
        members = np.nonzero(county == c)[0]
        if len(members) == 0:
            continue
        n_lords = max(1, min(params.lords_per_county, len(members)))
        chosen = rng.choice(len(members), size=n_lords, replace=False)
        local_seeds = xy[members[chosen]].astype(float)
        d2 = ((xy[members][:, None, :] - local_seeds[None, :, :]) ** 2).sum(axis=2)
        landlord[members] = next_id + np.argmin(d2, axis=1)
        seeds_list.append(local_seeds)
        next_id += n_lords

    seeds = np.concatenate(seeds_list) if seeds_list else np.zeros((0, 2))
    n_landlords = next_id
    estate_parcels = [np.nonzero(landlord == i)[0] for i in range(n_landlords)]

    # --- parcel adjacency: Moore neighbours within the same estate ----------------------
    # Which cells touch is cached with the lattice; only the same-estate test depends on this
    # run, and it is one comparison over the flattened candidate list.
    flat = lat.moore_flat
    degree = np.diff(lat.moore_ptr)
    owner = np.repeat(np.arange(len(xy)), degree)
    same = landlord[flat] == landlord[owner]
    counts = np.bincount(owner[same], minlength=len(xy))
    neighbours = np.split(flat[same], np.cumsum(counts)[:-1])

    # --- landlord awareness graph --------------------------------------------------------
    seed_d = np.hypot(seeds[:, None, 0] - seeds[None, :, 0], seeds[:, None, 1] - seeds[None, :, 1])
    np.fill_diagonal(seed_d, np.inf)
    landlord_neighbours = [
        np.nonzero(seed_d[i] <= params.awareness_radius)[0] for i in range(n_landlords)
    ]

    if params.random_awareness_graph:
        # RQ2's measurement control: keep each lord watching the same *number* of other lords,
        # but draw them from anywhere in England. Degree is preserved deliberately -- rewiring
        # to a fixed degree instead would change how much contagion there is as well as where
        # it goes, and the arm exists to vary only the second.
        everyone = np.arange(n_landlords)
        rewired = []
        for i in range(n_landlords):
            others = everyone[everyone != i]
            k = min(len(landlord_neighbours[i]), len(others))
            rewired.append(
                np.sort(rng.choice(others, size=k, replace=False))
                if k > 0
                else np.array([], dtype=int)
            )
        landlord_neighbours = rewired

    return Geography(
        shape=(n_rows, n_cols),
        xy=xy,
        phi_bar=phi_bar,
        region=region,
        landlord=landlord,
        neighbours=neighbours,
        landlord_neighbours=landlord_neighbours,
        estate_parcels=estate_parcels,
        county=county,
        county_names=[c["name"] for c in counties],
        county_grade=grades,
    )
