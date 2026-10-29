"""
s1_schedule.py -- find future Sentinel-1 acquisitions from the published ESA acquisition plans.

Source: https://sentinels.copernicus.eu/copernicus/sentinel-1/acquisition-plans
(one KML per satellite, ~20 days each, re-issued every few days)

Typical use
-----------
    from s1_schedule import get_plan, acquisitions_over

    plan   = get_plan()                               # stitched future plan, all satellites
    future = acquisitions_over(plan, (-119.008, 39.335), modes=["IW"])
    print(future[["satellite", "start", "OrbitRelative", "Mode"]])

Or in one call:
    from s1_schedule import next_acquisitions
    future = next_acquisitions((-119.008, 39.335))

Building blocks (use individually if you want):
    list_plan_files()   -> table of every KML linked on the page (satellite, start, end, url)
    select_plan_files() -> which files to use, and which time window each is authoritative for
    read_plan_kml()     -> one KML (path, url or bytes) -> GeoDataFrame, one row per data take
    get_plan()          -> the three above combined
    acquisitions_over() -> filter a plan to an AOI / modes / time
"""
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import affinity, wkt
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

LISTING_URL = "https://sentinels.copernicus.eu/copernicus/sentinel-1/acquisition-plans"
DEFAULT_CACHE = Path.home() / ".cache" / "s1_acquisition_plans"

_NS = {"k": "http://www.opengis.net/kml/2.2"}
_UA = {"User-Agent": "Mozilla/5.0 (s1_schedule.py)"}

# e.g. https://sentinels.copernicus.eu/documents/d/sentinel/s1d_mp_user_20260924t183918_20261014t202300
# (some links carry a suffix such as "-1" or "?download=true", which we keep in the url)
_LINK_RE = re.compile(
    r"https?://[^\s\"'<>()]*?/(s1[a-d])_mp_user_(\d{8}t\d{6})_(\d{8}t\d{6})[^\s\"'<>()]*",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Downloading
# -----------------------------------------------------------------------------
# Below this count of matched links (per satellite) we assume the listing page
# didn't fully load rather than that ESA really only published this many files.
_MIN_EXPECTED_LINKS = 5


def _fetch_rendered_html(url, wait_selector="text=KML files available", timeout=30_000):
    """
    Fetch `url` after letting its JavaScript run, using Playwright.

    The acquisition-plans page lazy-loads most of its document links client-side
    (a Liferay asset-list widget), so a plain urllib/requests GET only ever sees
    the first one or two entries per satellite. Rendering the page first is the
    only reliable way to get the full listing.

    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(
            "Rendering the acquisition-plans page requires Playwright.\n"
            "Install it with:\n"
            "    pip install playwright\n"
            "    playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=_UA["User-Agent"])
        page.goto(url, timeout=timeout, wait_until="networkidle")
        try:
            page.wait_for_selector(wait_selector, timeout=timeout)
        except Exception:
            pass  # fall through and use whatever loaded; the link-count check below will catch a bad fetch
        html = page.content()
        browser.close()
    return html


def _fetch(url, cache_dir=None, refresh=False):
    """Return the bytes at `url`, caching to `cache_dir` if given."""
    cache_file = None
    if cache_dir is not None:
        name = url.rstrip("/").split("/")[-1].split("?")[0]
        cache_file = Path(cache_dir) / f"{name}.kml"
        if cache_file.exists() and not refresh:
            return cache_file.read_bytes()

    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
    return data


# -----------------------------------------------------------------------------
# Finding the right files
# -----------------------------------------------------------------------------
def list_plan_files(listing_url=LISTING_URL, satellites=None, html=None, render=False):
    """
    Scrape the acquisition-plans page for KML links.

    The date range is taken from the *file name*, not the link text
    (the link text on the page is occasionally wrong).

    The page lazy-loads most of its links via JavaScript, so a plain fetch
    (render=False, the default -- no extra dependencies) usually only sees a
    couple of links per satellite. Two ways to get the full list instead:

      1. Open the page in your normal browser, save it (Ctrl/Cmd+S ->
         "Webpage, HTML only" or "Page Source"), and pass the saved file:
             list_plan_files(html="acquisition-plans.html")
         This is the recommended short-term fix -- no new packages needed.

      2. render=True, which uses Playwright (pip install playwright &&
         playwright install chromium) to render the page itself.

    satellites : e.g. ["S1C", "S1D"]; None = everything on the page.
    html       : raw HTML string, OR a path to a saved .html/.htm file.
    render     : render with Playwright instead (see above); ignored if `html` is given.

    Returns DataFrame [satellite, start, end, url], newest first.
    """
    if html is not None:
        p = Path(html)
        if p.suffix.lower() in (".html", ".htm") and p.exists():
            html = p.read_text(encoding="utf-8", errors="replace")
    elif render:
        html = _fetch_rendered_html(listing_url)
    else:
        html = _fetch(listing_url).decode("utf-8", errors="replace")

    rows = {}
    for m in _LINK_RE.finditer(html):
        url, sat, t0, t1 = m.group(0), m.group(1).upper(), m.group(2), m.group(3)
        rows[url] = dict(
            satellite=sat,
            start=datetime.strptime(t0.lower(), "%Y%m%dt%H%M%S"),
            end=datetime.strptime(t1.lower(), "%Y%m%dt%H%M%S"),
            url=url,
        )

    files = pd.DataFrame(rows.values(), columns=["satellite", "start", "end", "url"])
    # the same plan can be linked twice (e.g. with and without "?download=true")
    files = files.drop_duplicates(subset=["satellite", "start", "end"])
    if satellites is not None:
        files = files[files["satellite"].isin([s.upper() for s in satellites])]
    files = files.sort_values(["satellite", "start"], ascending=[True, False]).reset_index(drop=True)

    counts = files["satellite"].value_counts()
    thin = counts[counts < _MIN_EXPECTED_LINKS]
    if len(thin):
        import warnings
        warnings.warn(
            f"Only found {thin.to_dict()} link(s) for these satellites -- the listing "
            "page probably didn't fully load (it lazy-loads most links via JavaScript). "
            "Save the page from your browser and pass it as html=<path>, or set render=True. "
            "Results below are probably incomplete.",
            stacklevel=2,
        )
    return files


def _subtract(interval, others):
    """[a, b] minus a list of [s, e] intervals -> list of remaining [a, b] pieces."""
    pieces = [interval]
    for s, e in others:
        new = []
        for a, b in pieces:
            if e <= a or s >= b:          # no overlap
                new.append((a, b))
                continue
            if a < s:
                new.append((a, s))
            if e < b:
                new.append((e, b))
        pieces = new
    return pieces


def select_plan_files(files, after=None):
    """
    Decide which files to use and which time window each one is trusted for.

    Rule: a newer file (later start time in its name) supersedes older files
    wherever they overlap. Older files are only used for times the newer files
    don't cover (e.g. when ESA issues a short emergency re-plan, or when an
    older file happens to reach further into the future).

    Returns `files` with an extra 'windows' column (list of (start, end) tuples),
    keeping only files that still contribute something after `after`.
    """
    after = _utc_naive(after)
    out = []
    for sat, g in files.groupby("satellite"):
        g = g.sort_values("start", ascending=False)
        newer = []
        for _, f in g.iterrows():
            lo = max(f["start"], after)
            if f["end"] > lo:
                windows = _subtract((lo, f["end"]), newer)
                if windows:
                    out.append({**f.to_dict(), "windows": windows})
            newer.append((f["start"], f["end"]))
    return pd.DataFrame(out, columns=list(files.columns) + ["windows"])


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
def read_plan_kml(src):
    """
    Parse one acquisition-plan KML into a GeoDataFrame (one row per data take).

    src : local path, URL, or raw bytes.

    Columns: satellite, start, end, Mode, Swath, Polarisation, DatatakeId,
             OrbitAbsolute, OrbitRelative, ObservationDuration, geometry
    Times are naive UTC datetimes.
    """
    if isinstance(src, (bytes, bytearray)):
        root = ET.fromstring(src)
    elif str(src).startswith("http"):
        root = ET.fromstring(_fetch(str(src)))
    else:
        root = ET.parse(src).getroot()

    rows = []
    for pm in root.iter(f"{{{_NS['k']}}}Placemark"):
        d = {
            el.get("name"): el.findtext("k:value", namespaces=_NS)
            for el in pm.findall("k:ExtendedData/k:Data", _NS)
        }
        coords = pm.findtext(".//k:coordinates", namespaces=_NS)
        if not coords:
            continue
        pts = [tuple(map(float, c.split(",")[:2])) for c in coords.split()]
        if len(pts) < 4:
            continue
        rows.append(dict(
            satellite=d.get("SatelliteId"),
            start=pd.to_datetime(d.get("ObservationTimeStart")),
            end=pd.to_datetime(d.get("ObservationTimeStop")),
            Mode=d.get("Mode"),
            Swath=d.get("Swath"),
            Polarisation=d.get("Polarisation"),
            DatatakeId=d.get("DatatakeId"),
            OrbitAbsolute=int(d.get("OrbitAbsolute")),
            OrbitRelative=int(d.get("OrbitRelative")),
            ObservationDuration=int(d.get("ObservationDuration")),
            geometry=Polygon(pts),
        ))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# -----------------------------------------------------------------------------
# High-level
# -----------------------------------------------------------------------------
def get_plan(after=None, satellites=None, cache_dir=DEFAULT_CACHE, refresh=False,
             listing_url=LISTING_URL, verbose=True, html=None, render=False):
    """
    Download (with caching) and stitch the current acquisition plans into one
    GeoDataFrame of planned data takes starting after `after` (default: now).

    satellites : e.g. ["S1C", "S1D"]; None = every satellite with a current plan.
    cache_dir  : where downloaded KMLs are kept (None = no caching).
                 Published files don't change, so caching is safe; the listing
                 page itself is always re-fetched (it changes every few days).
    html       : path to a manually-saved copy of the listing page (see
                 list_plan_files) -- the easy way to get the full listing
                 without extra dependencies.
    render     : render the listing page with Playwright instead. See list_plan_files.
    """
    after = _utc_naive(after)
    files = select_plan_files(list_plan_files(listing_url, satellites, html=html, render=render), after)
    if files.empty:
        raise RuntimeError(f"No acquisition plans found covering times after {after}.")

    parts = []
    for _, f in files.iterrows():
        if verbose:
            win = ", ".join(f"{a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M}" for a, b in f["windows"])
            print(f"{f['satellite']}: {f['url'].split('/')[-1]}  [using {win}]")
        gdf = read_plan_kml(_fetch(f["url"], cache_dir, refresh))
        keep = pd.Series(False, index=gdf.index)
        for a, b in f["windows"]:
            keep |= (gdf["start"] >= a) & (gdf["start"] < b)
        parts.append(gdf[keep].assign(plan_file=f["url"]))

    plan = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(plan, geometry="geometry", crs="EPSG:4326") \
              .sort_values("start").reset_index(drop=True)


def acquisitions_over(plan, aoi, modes=("IW",), after=None, tracks=None,
                      n_per_track=None):
    """
    Filter a plan to data takes that intersect `aoi`.

    aoi         : (lon, lat), shapely geometry, WKT string, or GeoDataFrame/GeoSeries.
    modes       : e.g. ["IW"]; None = all modes (IW, EW, SM, WV).
    after       : only data takes starting after this time (default: now).
    tracks      : optional list of relative orbits (ASF "Path Number") to keep.
    n_per_track : keep only the next N passes per track.

    One row per pass: if a pass is split into several data takes that all hit
    the AOI, only the first is kept.
    """
    aoi = _as_geometry(aoi)
    after = _utc_naive(after)

    # footprints crossing the antimeridian use lon < -180 or > 180, so also test shifted AOIs
    hit = plan.intersects(aoi) \
        | plan.intersects(affinity.translate(aoi, 360)) \
        | plan.intersects(affinity.translate(aoi, -360))
    sel = plan[hit & (plan["start"] > after)]
    if modes is not None:
        sel = sel[sel["Mode"].isin(modes)]
    if tracks is not None:
        sel = sel[sel["OrbitRelative"].isin([int(t) for t in tracks])]

    sel = sel.sort_values("start").drop_duplicates(["satellite", "OrbitAbsolute"])
    if n_per_track is not None:
        sel = sel.groupby("OrbitRelative", group_keys=False).head(n_per_track)
    return sel.reset_index(drop=True)


def next_acquisitions(aoi, modes=("IW",), after=None, satellites=None, tracks=None,
                      n_per_track=None, cache_dir=DEFAULT_CACHE, verbose=True, html=None, render=False):
    """get_plan() + acquisitions_over() in one call."""
    plan = get_plan(after=after, satellites=satellites, cache_dir=cache_dir, verbose=verbose,
                    html=html, render=render)
    return acquisitions_over(plan, aoi, modes=modes, after=after, tracks=tracks,
                             n_per_track=n_per_track)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _utc_naive(t):
    """None -> now; any datetime/string -> naive UTC pandas Timestamp."""
    t = pd.Timestamp.now(tz=timezone.utc) if t is None else pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def _as_geometry(aoi):
    if isinstance(aoi, BaseGeometry):
        return aoi
    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return aoi.to_crs("EPSG:4326").union_all()
    if isinstance(aoi, str):
        return wkt.loads(aoi)
    lon, lat = aoi
    return Point(lon, lat)


if __name__ == "__main__":
    import sys
    lon, lat = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) == 3 else (-119.008, 39.335)
    fut = next_acquisitions((lon, lat))
    print(fut[["satellite", "start", "OrbitRelative", "Mode", "Polarisation", "DatatakeId"]]
          .to_string(index=False))