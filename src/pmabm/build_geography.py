"""One-off preprocessing: fetch the real England outline and Agricultural Land Classification.

Implements the offline half of the paper's Spatial structure section, which specifies that the
GIS work "happens once, offline" and that the running simulation needs only the resulting
lookup of regional fertility values.

Two open, Open Government Licence v3.0 sources are used:

* England boundary -- ONS Open Geography Portal.
* Agricultural Land Classification (provisional) -- Natural England.

ALC grades are aggregated *server-side*: for each cell of a coarse sampling grid we ask the
feature service for the total area in each grade intersecting that cell, which is exactly the
area-weighting the paper describes without downloading the full polygon set.

The result is cached to ``data/geography/england_alc.json`` and is the only geographic input
the model itself reads.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ONS_COUNTRIES = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Countries_December_2022_GB_BUC/FeatureServer/0/query"
)
#: ONS "Ancient Counties (December 1921) Boundaries EW BGC" -- the historic counties of
#: England and Wales, which is the tier the model uses as its largest unit of land.
ONS_ANCIENT_COUNTIES = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "ACTY_DEC_1921_EW_BGC/FeatureServer/0/query"
)
ALC_LAYER = (
    "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
    "Provisional%20Agricultural%20Land%20Classification%20(ALC)%20(England)/FeatureServer/0/query"
)
#: British National Grid. Both services are queried and returned in this projection, so
#: lattice distances are in real metres rather than degrees.
BNG = 27700

#: ALC grade label -> numeric grade. 1 is "excellent", 5 "very poor". Non-agricultural and
#: urban land carries no agricultural quality and is excluded from the weighted mean rather
#: than being scored, so that a heavily urbanised cell reads as "little data" not "poor land".
GRADE_VALUES = {
    "Grade 1": 1.0,
    "Grade 2": 2.0,
    "Grade 3": 3.0,
    "Grade 3a": 3.0,
    "Grade 3b": 3.5,
    "Grade 4": 4.0,
    "Grade 5": 5.0,
}

DEFAULT_OUTPUT = Path("data/geography/england_alc.json")
USER_AGENT = "pmabm/0.1 (research model; ONS + Natural England open data)"


def _get(url: str, params: dict, timeout: int = 90, post: bool = False) -> dict:
    """Query an ArcGIS REST endpoint.

    County polygons are far too large for a query string -- the server answers 502 -- so any
    request carrying a geometry is sent as a form POST instead.
    """
    body = urllib.parse.urlencode(params).encode()
    if post:
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    else:
        request = urllib.request.Request(
            f"{url}?{body.decode()}", headers={"User-Agent": USER_AGENT}
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_england_rings() -> list[list[list[float]]]:
    """Return England's boundary as a list of rings of ``[easting, northing]`` points."""
    payload = _get(
        ONS_COUNTRIES,
        {
            "where": "CTRY22NM='England'",
            "outFields": "CTRY22NM",
            "returnGeometry": "true",
            "outSR": str(BNG),
            "f": "json",
        },
    )
    features = payload.get("features") or []
    if not features:
        raise RuntimeError(f"ONS returned no England feature: {str(payload)[:300]}")
    rings = features[0]["geometry"]["rings"]
    return [[[float(x), float(y)] for x, y in ring] for ring in rings]


def fetch_ancient_counties() -> list[dict]:
    """Historic counties of England and Wales, as rings in British National Grid."""
    payload = _get(
        ONS_ANCIENT_COUNTIES,
        {
            "where": "1=1",
            "outFields": "ACTY1921NM,ACTY1921CD",
            "returnGeometry": "true",
            "outSR": str(BNG),
            "f": "json",
        },
        timeout=120,
    )
    counties = []
    for feature in payload.get("features", []):
        attrs = feature.get("attributes", {})
        rings = feature.get("geometry", {}).get("rings")
        if not rings:
            continue
        counties.append(
            {
                "name": attrs.get("ACTY1921NM"),
                "code": attrs.get("ACTY1921CD"),
                "rings": [[[float(x), float(y)] for x, y in ring] for ring in rings],
            }
        )
    if not counties:
        raise RuntimeError(f"ONS returned no ancient counties: {str(payload)[:300]}")
    return counties


def _alc_for_geometry(geometry: dict, geometry_type: str, post: bool = False) -> float:
    """Area-weighted mean ALC grade inside a geometry, or NaN where no graded land."""
    payload = _get(
        ALC_LAYER,
        {
            "where": "1=1",
            "geometry": json.dumps(geometry),
            "geometryType": geometry_type,
            "inSR": str(BNG),
            "spatialRel": "esriSpatialRelIntersects",
            "groupByFieldsForStatistics": "ALC_GRADE",
            "outStatistics": json.dumps(
                [
                    {
                        "statisticType": "sum",
                        "onStatisticField": "Shape__Area",
                        "outStatisticFieldName": "area",
                    }
                ]
            ),
            "returnGeometry": "false",
            "f": "json",
        },
        post=post,
    )
    weighted = 0.0
    total = 0.0
    for feature in payload.get("features", []):
        attributes = feature.get("attributes", {})
        grade = GRADE_VALUES.get(attributes.get("ALC_GRADE"))
        area = attributes.get("area")
        if grade is None or not area:
            continue
        weighted += grade * float(area)
        total += float(area)
    return weighted / total if total > 0 else float("nan")


def _alc_cell(bounds: tuple[float, float, float, float]) -> float:
    """Area-weighted mean ALC grade within a bounding box."""
    xmin, ymin, xmax, ymax = bounds
    return _alc_for_geometry(
        {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {"wkid": BNG},
        },
        "esriGeometryEnvelope",
    )


def _alc_for_county(county: dict) -> float:
    """Area-weighted mean ALC grade within a county's own boundary.

    This is the quantity the paper describes -- ALC grades area-weighted within each historic
    county -- computed server-side, so no polygon intersection is done locally.
    """
    return _alc_for_geometry(
        {"rings": county["rings"], "spatialReference": {"wkid": BNG}},
        "esriGeometryPolygon",
        post=True,
    )


def fetch_alc_grid(
    bbox: tuple[float, float, float, float], nx: int, ny: int, workers: int = 8
) -> np.ndarray:
    """Sample the mean ALC grade over an ``ny`` x ``nx`` grid spanning ``bbox``.

    Row 0 is the *southern* edge, matching the northing axis rather than image convention.
    """
    xmin, ymin, xmax, ymax = bbox
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    cells = [
        (row, col, (xs[col], ys[row], xs[col + 1], ys[row + 1]))
        for row in range(ny)
        for col in range(nx)
    ]

    grid = np.full((ny, nx), np.nan)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_alc_cell, bounds): (row, col) for row, col, bounds in cells}
        for done, future in enumerate(as_completed(futures), start=1):
            row, col = futures[future]
            try:
                grid[row, col] = future.result()
            except Exception as exc:  # a single failed cell must not lose the whole grid
                print(f"  ! cell ({row},{col}) failed: {type(exc).__name__}: {exc}")
            if done % 50 == 0:
                print(f"  ... {done}/{len(cells)} cells")
    return grid


def _simplify(rings: list, tolerance: float = 1000.0, min_points: int = 8) -> list:
    """Drop vertices closer together than ``tolerance`` metres.

    The lattice cells are several kilometres across, so sub-kilometre boundary detail cannot
    change which cell a point falls in -- it only inflates the cached artifact. Rings that
    would collapse below ``min_points`` are kept whole, so small islands survive.
    """
    out = []
    for ring in rings:
        points = np.asarray(ring, dtype=float)
        if len(points) <= min_points:
            out.append([[round(x, 1), round(y, 1)] for x, y in points])
            continue
        kept = [points[0]]
        for point in points[1:-1]:
            if np.hypot(*(point - kept[-1])) >= tolerance:
                kept.append(point)
        kept.append(points[-1])  # close the ring on its original last vertex
        if len(kept) < 4:  # degenerate after thinning; fall back to the original
            kept = list(points)
        out.append([[round(float(x), 1), round(float(y), 1)] for x, y in kept])
    return out


#: Units in the ONS Ancient Counties layer that are not historic counties of England for this
#: model's purposes, excluded by name.
#:
#: The County of London is the only one. ONS lists it separately from Middlesex, Surrey, Kent and
#: Essex, whose 1921 boundaries are drawn around rather than through it, but it is a city and not
#: an agrarian county: Natural England returns no graded agricultural land inside it at all, which
#: is correct and is precisely why it cannot be used. Retaining it meant it fell back to the
#: national mean fertility baseline, crediting the capital with average farmland; excluding it
#: leaves its cells to the nearest-county assignment in :mod:`pmabm.geography`, which hands them
#: to the surrounding counties whose land they actually were. Dropping it also brings the count to
#: exactly the 39 historic counties the paper claims.
EXCLUDED_COUNTIES = ("London",)

#: Share of a county's own area that must fall inside the England outline for it to be kept.
#: A half is the natural cut for "is this an English county", and every county is in practice
#: far from it -- the English ones score above 0.97 and the Welsh below 0.02 -- so the threshold
#: is not doing delicate work. See :func:`england_area_share`.
ENGLAND_SHARE_THRESHOLD = 0.5


def _ring_area(ring: np.ndarray) -> float:
    """Signed shoelace area of a closed ring."""
    x, y = ring[:, 0], ring[:, 1]
    return float((x[:-1] * y[1:] - x[1:] * y[:-1]).sum() / 2.0)


def _centroid(rings: list) -> tuple[float, float]:
    """Area-weighted centroid of the largest ring.

    Not the mean of the vertices. A vertex mean weights the boundary by how finely it happens to
    be sampled, so a county with an intricate coastline on one side and a straight inland border
    on the other has its "centre" dragged out to sea -- which is exactly what silently dropped
    Kent, Cheshire and Somersetshire from the artifact: all three have long, heavily-vertexed
    estuarine coasts (the Thames and Medway, the Dee and Mersey, the Severn), and their vertex
    means landed in open water outside the England outline.

    The largest ring rather than all of them, because the smaller rings are offshore islands
    (Thanet, Sheppey, the Scillies) whose areas would pull the result back toward the coast.
    """
    largest = max((np.asarray(r, dtype=float) for r in rings), key=lambda r: abs(_ring_area(r)))
    area = _ring_area(largest)
    if area == 0.0:  # degenerate ring; the vertex mean is all there is
        return float(largest[:, 0].mean()), float(largest[:, 1].mean())
    x, y = largest[:, 0], largest[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    return (
        float(((x[:-1] + x[1:]) * cross).sum() / (6.0 * area)),
        float(((y[:-1] + y[1:]) * cross).sum() / (6.0 * area)),
    )


def _interior_points(rings: list, n: int = 28) -> np.ndarray:
    """Up to ``n``x``n`` grid points falling inside ``rings``, as an (M, 2) array.

    Used to measure a county's overlap with England by area rather than by a single point. A
    centroid is one sample and can be wrong for any concave or crescent-shaped county even when
    computed correctly; a sample of the interior cannot be wrong in the same way.
    """
    points = np.concatenate([np.asarray(r, dtype=float) for r in rings])
    xs = np.linspace(points[:, 0].min(), points[:, 0].max(), n + 2)[1:-1]
    ys = np.linspace(points[:, 1].min(), points[:, 1].max(), n + 2)[1:-1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    px, py = grid_x.ravel(), grid_y.ravel()
    inside = _points_in_rings(rings, px, py)
    return np.column_stack([px[inside], py[inside]])


def england_area_share(england_rings: list, county: dict) -> float:
    """Share of a county's sampled interior that lies inside the England outline.

    This is the criterion for "is this an English county", replacing the single-centroid test.
    It is reported rather than only thresholded, so a marginal case would be visible in the
    build log instead of silently deciding itself.
    """
    interior = _interior_points(county["rings"])
    if len(interior) == 0:  # a county too small for the sampling grid to catch: use its centroid
        cx, cy = _centroid(county["rings"])
        return float(_points_in_rings(england_rings, np.array([cx]), np.array([cy]))[0])
    return float(_points_in_rings(england_rings, interior[:, 0], interior[:, 1]).mean())


def build(output: Path = DEFAULT_OUTPUT, workers: int = 8) -> Path:
    """Fetch the boundary, the historic counties and their land quality, and cache them."""
    print("Fetching England boundary from ONS ...")
    rings = fetch_england_rings()
    points = np.concatenate([np.asarray(ring) for ring in rings])
    bbox = (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )
    print(f"  {len(rings)} rings, {len(points)} vertices")

    print("Fetching ancient counties (December 1921) from ONS ...")
    counties = fetch_ancient_counties()
    print(f"  {len(counties)} counties for England and Wales")

    excluded = [c["name"] for c in counties if c["name"] in EXCLUDED_COUNTIES]
    counties = [c for c in counties if c["name"] not in EXCLUDED_COUNTIES]
    if excluded:
        print(f"  excluding by name: {', '.join(excluded)}")
    missing = sorted(set(EXCLUDED_COUNTIES) - set(excluded))
    if missing:
        # The layer's naming changed under us; say so rather than quietly excluding nothing.
        print(f"  ! EXCLUDED_COUNTIES names not found in the layer: {', '.join(missing)}")

    # Keep only the English ones. Wales has its own tenurial history and the theta regime of the
    # paper is specifically English, so the Welsh counties in this ONS layer must go. The test is
    # the share of each county's *area* inside the England outline, not whether a single centroid
    # falls inside it: the earlier single-point version silently dropped Kent, Cheshire and
    # Somersetshire (see :func:`_centroid`), and a wrongly-excluded county is invisible in the
    # output while a wrongly-included one is obvious, which is the worse of the two failure modes.
    shares = [england_area_share(rings, c) for c in counties]
    marginal = [
        (c["name"], s) for c, s in zip(counties, shares) if 0.05 < s < 0.95
    ]
    counties = [c for c, s in zip(counties, shares) if s >= ENGLAND_SHARE_THRESHOLD]
    print(f"  {len(counties)} fall within England")
    if marginal:
        # Never silently: a county straddling the border is a judgement call and should be seen.
        print("  marginal (area share strictly between 0.05 and 0.95):")
        for name, share in sorted(marginal, key=lambda t: -t[1]):
            print(f"    {name:24s} {share:.3f} "
                  f"{'kept' if share >= ENGLAND_SHARE_THRESHOLD else 'dropped'}")

    print("Aggregating Natural England ALC within each county ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        grades = list(pool.map(_alc_for_county, counties))
    for county, grade in zip(counties, grades):
        county["mean_alc_grade"] = None if np.isnan(grade) else round(float(grade), 4)
    graded = [c for c in counties if c["mean_alc_grade"] is not None]
    print(f"  {len(graded)}/{len(counties)} counties returned graded agricultural land")
    if not graded:
        raise RuntimeError("ALC returned no graded land anywhere; refusing to write artifact")
    best = min(graded, key=lambda c: c["mean_alc_grade"])
    worst = max(graded, key=lambda c: c["mean_alc_grade"])
    print(f"  best land : {best['name']} (grade {best['mean_alc_grade']:.2f})")
    print(f"  worst land: {worst['name']} (grade {worst['mean_alc_grade']:.2f})")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "crs": f"EPSG:{BNG}",
                "bbox": bbox,
                "outline_rings": _simplify(rings),
                "counties": [
                    {
                        "name": c["name"],
                        "code": c["code"],
                        "mean_alc_grade": c["mean_alc_grade"],
                        "rings": _simplify(c["rings"]),
                    }
                    for c in counties
                ],
                "sources": {
                    "boundary": "ONS Open Geography Portal, Countries (December 2022) GB BUC, OGL v3.0",
                    "counties": "ONS Open Geography Portal, Ancient Counties (December 1921) EW BGC, OGL v3.0",
                    "land_quality": "Natural England, Provisional Agricultural Land Classification (England), OGL v3.0",
                },
            }
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    return output


def _points_in_rings(rings: list, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for a handful of points against many rings."""
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        poly = np.asarray(ring, dtype=float)
        x1, y1 = poly[:-1, 0], poly[:-1, 1]
        x2, y2 = poly[1:, 0], poly[1:, 1]
        straddles = (y1 > py[:, None]) != (y2 > py[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (x2 - x1) * (py[:, None] - y1) / (y2 - y1) + x1
        inside ^= (straddles & (px[:, None] < x_cross)).sum(axis=1) % 2 == 1
    return inside


if __name__ == "__main__":
    build()
