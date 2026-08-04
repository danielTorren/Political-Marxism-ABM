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
        poly = np.asarray(ring, dtype=float)
        x1, y1 = poly[:-1, 0], poly[:-1, 1]
        x2, y2 = poly[1:, 0], poly[1:, 1]
        # Process edges in chunks to bound peak memory on large boundary files.
        for start in range(0, len(x1), 2000):
            sl = slice(start, start + 2000)
            ex1, ey1, ex2, ey2 = x1[sl], y1[sl], x2[sl], y2[sl]
            straddles = (ey1 > py[:, None]) != (ey2 > py[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = (ex2 - ex1) * (py[:, None] - ey1) / (ey2 - ey1) + ex1
            crossings = straddles & (px[:, None] < x_cross)
            inside ^= crossings.sum(axis=1) % 2 == 1

    return inside.reshape(grid_y.shape)


def _rings_contain(rings: list, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon of scattered points against one area's rings."""
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        poly = np.asarray(ring, dtype=float)
        if len(poly) < 4:
            continue
        x1, y1 = poly[:-1, 0], poly[:-1, 1]
        x2, y2 = poly[1:, 0], poly[1:, 1]
        for start in range(0, len(x1), 2000):
            sl = slice(start, start + 2000)
            ex1, ey1, ex2, ey2 = x1[sl], y1[sl], x2[sl], y2[sl]
            straddles = (ey1 > py[:, None]) != (ey2 > py[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = (ex2 - ex1) * (py[:, None] - ey1) / (ey2 - ey1) + ex1
            inside ^= (straddles & (px[:, None] < x_cross)).sum(axis=1) % 2 == 1
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
    counties = artifact["counties"]
    parcel_x, parcel_y = xs[cols], ys[rows]
    county = np.full(len(xy), -1, dtype=int)
    for index, entry in enumerate(counties):
        unassigned = county < 0
        if not unassigned.any():
            break
        hit = _rings_contain(entry["rings"], parcel_x[unassigned], parcel_y[unassigned])
        idx = np.nonzero(unassigned)[0][hit]
        county[idx] = index

    # Coastal cells whose centroid misses every county polygon go to the nearest county.
    if (county < 0).any():
        centroids = np.array([_ring_centroid(c["rings"]) for c in counties])
        for p in np.nonzero(county < 0)[0]:
            d = np.hypot(centroids[:, 0] - parcel_x[p], centroids[:, 1] - parcel_y[p])
            county[p] = int(np.argmin(d))

    grades = np.array(
        [
            c["mean_alc_grade"] if c["mean_alc_grade"] is not None else np.nan
            for c in counties
        ]
    )
    if np.isnan(grades).any():  # a county with no graded land takes the national mean
        grades = np.where(np.isnan(grades), np.nanmean(grades), grades)

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
    index_of = -np.ones((n_rows, n_cols), dtype=int)
    index_of[rows, cols] = np.arange(len(xy))
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    neighbours: list[np.ndarray] = []
    for p in range(len(xy)):
        r, c = xy[p]
        found = []
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                q = index_of[rr, cc]
                if q >= 0 and landlord[q] == landlord[p]:
                    found.append(q)
        neighbours.append(np.array(found, dtype=int))

    # --- landlord awareness graph --------------------------------------------------------
    seed_d = np.hypot(seeds[:, None, 0] - seeds[None, :, 0], seeds[:, None, 1] - seeds[None, :, 1])
    np.fill_diagonal(seed_d, np.inf)
    landlord_neighbours = [
        np.nonzero(seed_d[i] <= params.awareness_radius)[0] for i in range(n_landlords)
    ]

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
