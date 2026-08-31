"""
Self-contained HTML/SVG world map for the GeoMASST page.

Replaces the Plotly figure with an inline-SVG map whose markers are filtered
client-side, so dragging the collection-date or depth/altitude window updates the
map instantly instead of triggering a Streamlit rerun.

Two layers, same as export_hits_map():
  - hits:       the selected chemical's raw-data matches, filled markers
  - background: the rest of the environmental ReDU corpus, hollow markers

Both are aggregated per (lat, lon, environment material); marker radius scales
with sqrt(sample count). Coordinates are projected equirectangularly into the
960x480 viewBox that bin/assets/world_land_110m_equirect.path is drawn in.
"""

from __future__ import annotations

import html
import json
import math
import os
from typing import Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
LAND_PATH_FILE = os.path.join(ASSETS, "world_land_50m_equirect.path")
# optional detail layers, all pre-projected into the same 960x480 viewBox
# (Natural Earth, public domain: 10m borders, lakes and rivers plus the European
# river/lake supplements, 50m coastline, populated places). Everything started at
# the coarsest 110m tier, which gave 462 rivers and 24 lakes worldwide and about
# nine points per national boundary - hence dams on apparently dry land and
# borders that read as crude polylines beside the finer water.
BORDERS_FILE = os.path.join(ASSETS, "world_borders_10m_equirect.path")
LAKES_FILE = os.path.join(ASSETS, "world_lakes_10m_equirect.path")
RIVERS_FILE = os.path.join(ASSETS, "world_rivers_10m_equirect.json")
CITIES_FILE = os.path.join(ASSETS, "world_cities_equirect.json")
COUNTRY_LABELS_FILE = os.path.join(ASSETS, "world_country_labels_equirect.json")
# Dams from Wikidata (CC0), kept to those with a recorded height of 30 m or more.
# Heights are only an inclusion filter - a fair number are feet recorded as metres,
# so they are never shown.
DAMS_FILE = os.path.join(ASSETS, "world_dams_equirect.json")

# viewBox of the bundled land outline
VB_W, VB_H = 960.0, 480.0

MISSING = {"missing value", "", "nan", "none", "null"}

# Qualitative palette; categories beyond this cycle back through it.
PALETTE = [
    "#1d6fa5", "#2a9d8f", "#a8477a", "#8a5a34", "#c98a2b", "#8fb8d4",
    "#c1502e", "#5b8c5a", "#7b6bb0", "#c2557a", "#4f8a8b", "#b07d3a",
    "#6a8caf", "#9c6b4f", "#588157", "#9d4edd", "#e07a5f", "#3d5a80",
]


def _read_asset(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _load_land_path() -> str:
    return _read_asset(LAND_PATH_FILE)


def _load_points(path: str, limit: int) -> list:
    raw = _read_asset(path)
    if not raw:
        return []
    try:
        return json.loads(raw)[:limit]
    except ValueError:
        return []


def _label_group(gid: str, points: list, cls: str, dx: float = 2.2, dy: float = 1.4) -> str:
    if not points:
        return ""
    txt = "".join(
        f'<text class="{cls}" x="{c["x"] + dx}" y="{c["y"] + dy}">{html.escape(str(c["n"]))}</text>'
        for c in points
    )
    return f'<g id="{gid}">{txt}</g>'


def _detail_svg(max_cities: int = 90, max_countries: int = 177) -> str:
    """Borders, rivers, lakes, countries, cities and dams, drawn under the markers."""
    borders, lakes = _read_asset(BORDERS_FILE), _read_asset(LAKES_FILE)
    out = []
    # rivers come in four width classes so a trunk river does not read like a ditch
    raw_rivers = _read_asset(RIVERS_FILE)
    if raw_rivers:
        try:
            rivers = json.loads(raw_rivers)
        except ValueError:
            rivers = {}
        for cls in ("4", "3", "2", "1"):        # thin first, so trunks draw on top
            d = rivers.get(cls)
            if d:
                out.append(f'<path class="ne-river ne-river-{cls}" d="{d}"></path>')
    if lakes:
        out.append(f'<path class="ne-lake" d="{lakes}"></path>')
    if borders:
        out.append(f'<path class="ne-border" d="{borders}"></path>')

    cities = _load_points(CITIES_FILE, max_cities)
    out.append("".join(
        f'<circle class="ne-city" cx="{c["x"]}" cy="{c["y"]}" r="0.9"></circle>' for c in cities))

    countries = _load_points(COUNTRY_LABELS_FILE, max_countries)

    if not out:
        return ""

    # Each layer that toggles independently needs its own referenced group: document
    # CSS can style a <use> element, but elements inside a referenced subtree do not
    # re-render when an ancestor selector starts or stops matching.
    return (
        f'<g id="gm-detail">{"".join(out)}</g>'
        + _label_group("gm-country-labels", countries, "ne-country-label", dx=0, dy=0)
        + _label_group("gm-city-labels", cities, "ne-city-label")
    )


def _dam_payload(hits: pd.DataFrame, radius: float = 1.2,
                 max_dams: Optional[int] = None) -> str:
    """
    Dam coordinates and names for the neighbourhood of the matches, as data.

    Dams are only ever drawn near a match, so there is no reason to ship the other
    hundred thousand: filtering here turns a ~3.7 MB global list into a few hundred
    entries. Radius is in viewBox units - 1.2 is roughly 50 km at the equator, and
    matches the radius the page applies against the markers still visible after the
    date and depth filters.
    """
    raw = _read_asset(DAMS_FILE)
    if not raw or hits is None or hits.empty:
        return ""
    try:
        data = json.loads(raw)
        cs, ns = data["c"], data["n"]
    except (ValueError, KeyError, TypeError):
        return ""

    hx, hy = _project(hits["lat"], hits["lon"])
    hx, hy = hx.to_numpy(), hy.to_numpy()
    r2 = radius * radius

    keep_c, keep_n = [], []
    for i, name in enumerate(ns):
        x, y = cs[2 * i], cs[2 * i + 1]
        dx = hx - x
        dy = hy - y
        if ((dx * dx + dy * dy) <= r2).any():
            keep_c.extend((x, y))
            keep_n.append(name)
            if max_dams is not None and len(keep_n) >= max_dams:
                break
    if not keep_n:
        return ""
    body = json.dumps({"c": keep_c, "n": keep_n}, separators=(",", ":"), ensure_ascii=False)
    return f'<script type="application/json" id="damData">{body}</script>'


def _project(lat: pd.Series, lon: pd.Series):
    """Equirectangular projection into the land outline's 960x480 viewBox."""
    x = (lon + 180.0) / 360.0 * VB_W
    y = (90.0 - lat) / 180.0 * VB_H
    return x, y


def _fmt_date(ts) -> str:
    """Zero-padded ISO date. strftime('%Y') drops the padding for years < 1000 on
    glibc, and JS then reads '1-01-18' as 2018-01-01, so build it by hand."""
    return f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}"


def _plausible_dates(s: pd.Series) -> pd.Series:
    """ReDU carries a few unparseable/sentinel collection dates (year 1, far-future
    typos). One of them would stretch the slider domain over empty centuries and
    push every real marker into a corner, so treat them as undated instead."""
    lo = pd.Timestamp("1900-01-01")
    hi = pd.Timestamp.now().normalize() + pd.DateOffset(years=1)
    return s.where(s.between(lo, hi))


def _clean(s: pd.Series) -> pd.Series:
    """ReDU encodes absent values as the literal string 'missing value'."""
    out = s.astype(str).str.strip()
    return out.mask(out.str.lower().isin(MISSING))


def _prepare(df: pd.DataFrame, env_col: str, facet_col: Optional[str] = None) -> pd.DataFrame:
    """Parse coordinates, material, date and depth into one tidy frame."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    needed = {"LatitudeandLongitude", env_col}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    work = df.copy()

    latlon = _clean(work["LatitudeandLongitude"]).str.extract(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*$"
    )
    lat = pd.to_numeric(latlon[0], errors="coerce")
    lon = pd.to_numeric(latlon[1], errors="coerce")

    env = _clean(work[env_col])
    ok = lat.between(-90, 90) & lon.between(-180, 180) & env.notna()
    if not ok.any():
        return pd.DataFrame()

    out = pd.DataFrame(index=work.index[ok])
    out["lat"] = lat[ok]
    out["lon"] = lon[ok]
    out["cat"] = env[ok]

    # ReDU stores e.g. '4/10/2015 0:00:00'; anything unparseable becomes NaT and
    # is governed by the "keep undated markers" toggle rather than dropped.
    if "SampleCollectionDateandTime" in work.columns:
        raw = _clean(work["SampleCollectionDateandTime"])[ok]
        out["date"] = _plausible_dates(pd.to_datetime(raw, errors="coerce", format="mixed"))
    else:
        out["date"] = pd.NaT

    if "DepthorAltitudeMeters" in work.columns:
        out["depth"] = pd.to_numeric(_clean(work["DepthorAltitudeMeters"])[ok], errors="coerce")
    else:
        out["depth"] = float("nan")

    ds_col = next((c for c in ("ATTRIBUTE_DatasetAccession", "Dataset") if c in work.columns), None)
    out["dataset"] = _clean(work[ds_col])[ok] if ds_col else None

    if facet_col and facet_col in work.columns:
        out["facet"] = pd.to_numeric(work[facet_col], errors="coerce")[ok]
        out = out[out["facet"].notna()]

    return out


def _aggregate(tidy: pd.DataFrame, max_markers: Optional[int], by: tuple = ()) -> pd.DataFrame:
    """One marker per (lat, lon, material), carrying the ranges its samples span."""
    if tidy.empty:
        return tidy
    keys = [k for k in by if k in tidy.columns] + ["lat", "lon", "cat"]

    def _datasets(s: pd.Series) -> str:
        u = [v for v in pd.Series(s).dropna().unique().tolist()][:4]
        return ", ".join(u)

    agg = (
        tidy.groupby(keys, as_index=False)
        .agg(
            n=("lat", "size"),
            dmin=("date", "min"),
            dmax=("date", "max"),
            zmin=("depth", "min"),
            zmax=("depth", "max"),
            datasets=("dataset", _datasets),
        )
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    if max_markers is not None and len(agg) > max_markers:
        agg = agg.head(max_markers).copy()
    return agg


def _radius(n: int, scale: float, floor: float = 0.0) -> float:
    return round(min(max(scale * math.sqrt(max(int(n), 1)), floor), 22.0), 2)


def _circles(agg: pd.DataFrame, colors: dict, layer: str, scale: float,
             color_key: str = "cat", alt_colors: Optional[dict] = None,
             site_analogues: Optional[dict] = None) -> tuple:
    """Emit <circle> markup plus counts of markers missing date / depth."""
    if agg.empty:
        return "", 0, 0

    x, y = _project(agg["lat"], agg["lon"])
    parts = []
    n_no_date = 0
    n_no_depth = 0

    for (_, r), px, py in zip(agg.iterrows(), x, y):
        cat = str(r["cat"])
        key = r[color_key] if color_key in agg.columns else cat
        color = colors.get(key, "#98a2ad")
        alt_color = (alt_colors or {}).get(cat, "#98a2ad")
        # how many distinct modifications were seen at this site, for the size option
        n_alt = (site_analogues or {}).get((r["lat"], r["lon"]), 0)

        has_date = pd.notna(r["dmin"]) and pd.notna(r["dmax"])
        has_depth = pd.notna(r["zmin"]) and pd.notna(r["zmax"])
        if not has_date:
            n_no_date += 1
        if not has_depth:
            n_no_depth += 1

        attrs = [
            f'class="pt {layer}"',
            f'cx="{px:.1f}"', f'cy="{py:.1f}"',
        ]
        floor = 4.0 if layer == "hit" else 0.0
        r_n = _radius(r["n"], scale, floor=floor)
        r_d = _radius(max(n_alt, 1), scale * 1.6, floor=floor)
        attrs.append(f'r="{r_n}"')
        attrs.append(f'data-rn="{r_n}"')
        attrs.append(f'data-rd="{r_d}"')
        attrs.append(f'data-n="{int(r["n"])}"')
        attrs.append(f'data-nalt="{int(n_alt)}"')
        if layer == "hit":
            attrs.append(f'fill="{color}"')
            attrs.append(f'data-cdelta="{color if color_key == "facet" else alt_color}"')
            attrs.append(f'data-ccat="{alt_color if color_key == "facet" else color}"')
        else:
            attrs.append('fill="none"')  # stroke color comes from CSS, deliberately neutral
        attrs.append(f'data-cat="{html.escape(cat, quote=True)}"')
        # true coordinates, so the browser can re-project when the tile base map
        # (Web Mercator) replaces the bundled vectors (equirectangular)
        attrs.append(f'data-lat="{float(r["lat"]):.5f}" data-lon="{float(r["lon"]):.5f}"')
        if "facet" in agg.columns and pd.notna(r["facet"]):
            attrs.append(f'data-delta="{int(round(float(r["facet"])))}"')

        if has_date:
            attrs.append(f'data-dmin="{_fmt_date(r["dmin"])}"')
            attrs.append(f'data-dmax="{_fmt_date(r["dmax"])}"')
        if has_depth:
            attrs.append(f'data-zmin="{float(r["zmin"]):.1f}"')
            attrs.append(f'data-zmax="{float(r["zmax"]):.1f}"')

        tip = [
            cat if "facet" not in agg.columns or pd.isna(r["facet"])
            else f'{cat}  ·  {_facet_label(r["facet"])}',
            f'{int(r["n"])} sample(s) · {r["lat"]:.3f}, {r["lon"]:.3f}',
            "hits for this molecule" if layer == "hit" else "not matched (ReDU context)",
        ]
        if has_date:
            d0, d1 = _fmt_date(r["dmin"]), _fmt_date(r["dmax"])
            tip.append(d0 if d0 == d1 else f"{d0} → {d1}")
        else:
            tip.append("no recorded date")
        if has_depth:
            z0, z1 = float(r["zmin"]), float(r["zmax"])
            tip.append(f"{z0:.0f}m" if z0 == z1 else f"{z0:.0f}m → {z1:.0f}m")
        else:
            tip.append("no recorded depth/altitude")
        if isinstance(r["datasets"], str) and r["datasets"]:
            tip.append(r["datasets"])
            attrs.append(f'data-ds="{html.escape(r["datasets"], quote=True)}"')

        attrs.append(f'data-tip="{html.escape(chr(10).join(tip), quote=True)}"')
        parts.append("<circle " + " ".join(attrs) + "></circle>")

    return "\n".join(parts), n_no_date, n_no_depth


# Delta colors encode the chemistry: losses run cool, additions run warm, and the
# unmodified parent gets a color from neither ramp so it never reads as a modification.
PARENT_COLOR = "#7a5ea7"
LOSS_RAMP = ["#12374b", "#1b566f", "#2a7ea6", "#3ba3c4", "#57c4cf", "#8ad6da"]
GAIN_RAMP = ["#7f2b18", "#a8442a", "#c96a3a", "#dd8b4f", "#eaa96b", "#f2c48c"]


def _facet_label(v) -> str:
    """Human label for one unit delta mass."""
    iv = int(round(float(v)))
    return "unmodified (Δ 0 Da)" if iv == 0 else f"Δ {iv:+d} Da"


def _lerp_ramp(ramp: list, t: float) -> str:
    """Sample a hex ramp continuously, so every delta gets its own distinct shade
    instead of the ramp wrapping and giving two deltas the same color."""
    t = min(max(t, 0.0), 1.0)
    if len(ramp) == 1:
        return ramp[0]
    pos = t * (len(ramp) - 1)
    i = min(int(pos), len(ramp) - 2)
    f = pos - i
    a, b = ramp[i].lstrip("#"), ramp[i + 1].lstrip("#")
    ch = [round(int(a[j:j + 2], 16) + (int(b[j:j + 2], 16) - int(a[j:j + 2], 16)) * f) for j in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(*ch)


def _delta_colors(deltas) -> dict:
    """Cool for losses, warm for additions, shade ordered by distance from the parent."""
    out = {}
    for signed, ramp in ((-1, LOSS_RAMP), (1, GAIN_RAMP)):
        group = sorted([d for d in deltas if (d < 0 if signed < 0 else d > 0)], key=abs)
        for i, d in enumerate(group):
            out[d] = _lerp_ramp(ramp, i / max(len(group) - 1, 1))
    for d in deltas:
        if d == 0:
            out[d] = PARENT_COLOR
    return out


HIST_BINS = 48
HIST_W, HIST_H = 480.0, 100.0


def _to_secs(series) -> pd.Series:
    """
    Datetimes to epoch seconds, NaT preserved as NaN.

    Subtracting the epoch rather than casting to int64: pandas 2 keeps datetime64[us]
    here, so a cast yields microseconds and any fixed divisor is wrong by 1000x, and
    NaT casts to a huge negative sentinel instead of NaN.
    """
    dt = pd.to_datetime(series, errors="coerce")
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds()


def _bin_counts(agg, col_lo: str, col_hi: str, lo: float, hi: float, bins: int = HIST_BINS) -> list:
    """
    Count markers per bin across [lo, hi].

    A marker covers a range (its samples' min..max), and the slider shows it whenever
    that range overlaps the window - so the histogram counts a marker into every bin
    its range touches, and reading a bar means the same thing as dragging to it.
    """
    counts = [0] * bins
    if agg is None or len(agg) == 0 or col_lo not in agg.columns:
        return counts
    span = max(hi - lo, 1e-9)
    a = pd.to_numeric(agg[col_lo], errors="coerce")
    b = pd.to_numeric(agg[col_hi], errors="coerce")
    for va, vb in zip(a, b):
        if pd.isna(va) or pd.isna(vb):
            continue
        i0 = int(min(max((float(va) - lo) / span, 0.0), 0.999999) * bins)
        i1 = int(min(max((float(vb) - lo) / span, 0.0), 0.999999) * bins)
        for i in range(min(i0, i1), max(i0, i1) + 1):
            counts[i] += 1
    return counts


def _histogram_svg(hit_counts: list, ctx_counts: list) -> str:
    """
    A small stacked distribution strip: context behind, hits in front.

    Heights are square-rooted. One bin usually holds most of the markers, and on a
    linear scale at this height every other bin flattens to an invisible sliver -
    which defeats the point of showing the shape.
    """
    totals = [h + c for h, c in zip(hit_counts, ctx_counts)]
    peak = max(totals) or 1
    n = len(totals)
    w = HIST_W / n

    def _h(v):
        return (math.sqrt(v / peak) * HIST_H) if v > 0 else 0.0

    bars = []
    for i, (h, tot) in enumerate(zip(hit_counts, totals)):
        x = i * w
        if tot:
            th = _h(tot)
            bars.append(f'<rect class="hbar hbar-tot" data-i="{i}" x="{x:.2f}" '
                        f'y="{HIST_H - th:.2f}" width="{w:.2f}" height="{th:.2f}"></rect>')
        if h:
            hh = _h(h)
            bars.append(f'<rect class="hbar hbar-hit" data-i="{i}" x="{x:.2f}" '
                        f'y="{HIST_H - hh:.2f}" width="{w:.2f}" height="{hh:.2f}"></rect>')
    return (f'<svg class="hist" viewBox="0 0 {HIST_W:.0f} {HIST_H:.0f}" preserveAspectRatio="none" '
            f'aria-hidden="true">{"".join(bars)}</svg>')


def build_geomasst_map_html(
    df_hits: pd.DataFrame,
    df_background: Optional[pd.DataFrame] = None,
    env_col: str = "ENVOEnvironmentMaterial",
    compound_name: str = "",
    max_markers: int = 6000,
    facet_col: str = "Unit Delta Mass",
    show_context: bool = True,
    show_detail: bool = False,
    max_dams: Optional[int] = None,
) -> str:
    """
    Build a standalone HTML document for embedding with streamlit.components.v1.html.

    Plain search  -> one map, markers colored by environment material.
    Analog search -> one map for the unmodified parent and one for all analogues
                     colored by delta; picking deltas from the side list splits those
                     into a map each. Every map pans and zooms together.

    df_hits       raw-data matches for one molecule (needs LatitudeandLongitude + env_col;
                  SampleCollectionDateandTime and DepthorAltitudeMeters enable the sliders)
    df_background environmental ReDU sites with no hit, drawn as hollow context markers
    show_context  initial state of the "sites without hits" toggle
    show_detail   initial state of the detailed-basemap toggle (borders, rivers,
                  lakes, countries, cities and dams); off by default
    max_dams      cap the dams embedded for this result (they are already limited to
                  the neighbourhood of the matches); None keeps all of them
    """
    has_facet = bool(facet_col) and facet_col in getattr(df_hits, "columns", [])
    tidy_hits = _prepare(df_hits, env_col, facet_col if has_facet else None)

    analog = (
        has_facet and not tidy_hits.empty and "facet" in tidy_hits.columns
        and tidy_hits["facet"].nunique() > 1
    )

    hits = _aggregate(tidy_hits, max_markers, by=("facet",) if analog else ())
    bg = _aggregate(_prepare(df_background, env_col), max_markers)

    # ---- colors ----
    # Only materials actually hit earn a legend entry; the context layer is drawn
    # neutral, so its materials would be legend rows for colors nobody can see.
    hit_cats = []
    if not hits.empty:
        hit_cats = [str(c) for c in hits.groupby("cat")["n"].sum().sort_values(ascending=False).index]
    material_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(hit_cats)}

    deltas = []
    delta_colors = {}
    if analog:
        deltas = sorted(hits["facet"].unique().tolist())
        delta_colors = _delta_colors(deltas)

    colors = delta_colors if analog else material_colors
    color_key = "facet" if analog else "cat"

    # distinct modifications seen at each site, for the "size by analogues" option
    site_analogues = {}
    if analog and not hits.empty:
        site_analogues = (
            hits[hits["facet"] != 0]
            .groupby(["lat", "lon"])["facet"].nunique().to_dict()
        )

    bg_svg, bg_no_date, bg_no_depth = _circles(bg, colors, "bg", scale=2.0)

    detail_svg = _detail_svg()
    detail = bool(detail_svg)

    # ---- the maps ----
    cards, side = [], []
    hit_no_date = hit_no_depth = 0

    def _card(idx, label, part, kind, delta=None):
        nonlocal hit_no_date, hit_no_depth
        svg, nd, nz = _circles(part, colors, "hit", scale=2.6, color_key=color_key,
                               alt_colors=material_colors, site_analogues=site_analogues)
        hit_no_date += nd
        hit_no_depth += nz
        n_mark = 0 if part.empty else len(part)
        n_samp = 0 if part.empty else int(part["n"].sum())
        ctx_layer = ('<use href="#gm-ctx" class="ctx-layer"></use>' if analog
                     else f'<g class="ctx ctx-layer">{bg_svg}</g>')
        detail_layer = (
            '<use href="#gm-detail" class="detail"></use>'
            '<use href="#gm-country-labels" class="detail country-labels"></use>'
            '<use href="#gm-city-labels" class="detail city-labels"></use>'
            '<g class="dams-live"></g>'
        ) if detail else ''
        attr_delta = "" if delta is None else f' data-delta="{int(round(float(delta)))}"'
        cards.append(
            f'<div class="facet" data-facet="{idx}" data-kind="{kind}"{attr_delta}>'
            f'<div class="facet-head"><span class="facet-name">{html.escape(label)}</span>'
            f'<span class="facet-meta">{n_mark:,} sites &middot; {n_samp:,} files</span></div>'
            f'<svg class="fmap" viewBox="0 0 {VB_W:.0f} {VB_H:.0f}" preserveAspectRatio="xMidYMid meet">'
            f'<rect x="0" y="0" width="{VB_W:.0f}" height="{VB_H:.0f}" fill="var(--ocean)"></rect>'
            f'<g class="zoom"><g class="tiles"></g><use href="#gm-land"></use>{detail_layer}{ctx_layer}'
            f'<g class="hits">{svg}</g></g></svg></div>'
        )
        return n_samp

    if not analog:
        _card(0, compound_name or "All matches", hits, "single")
    else:
        idx = 0
        parent = hits[hits["facet"] == 0]
        if not parent.empty:
            _card(idx, _facet_label(0), parent, "parent", delta=0)
            idx += 1
        analogues = hits[hits["facet"] != 0]
        if not analogues.empty:
            _card(idx, "All analogues", analogues, "combined")
            idx += 1
        # one map per delta, hidden until that delta is picked from the side list
        for d in [x for x in deltas if x != 0]:
            n = _card(idx, _facet_label(d), hits[hits["facet"] == d], "delta", delta=d)
            side.append(
                f'<label class="side-item" data-delta="{int(round(float(d)))}">'
                f'<span class="sw" style="background:{delta_colors[d]}"></span>'
                f'<span class="side-name">{html.escape(_facet_label(d))}</span>'
                f'<span class="side-count">{n:,}</span></label>'
            )
            idx += 1

    n_no_date = hit_no_date + bg_no_date
    n_no_depth = hit_no_depth + bg_no_depth

    # ---- slider domains span both layers ----
    frames = [f for f in (hits, bg) if not f.empty]
    dmins = [f["dmin"].min() for f in frames if f["dmin"].notna().any()]
    dmaxs = [f["dmax"].max() for f in frames if f["dmax"].notna().any()]
    zmins = [f["zmin"].min() for f in frames if f["zmin"].notna().any()]
    zmaxs = [f["zmax"].max() for f in frames if f["zmax"].notna().any()]

    has_dates = bool(dmins and dmaxs)
    has_depth = bool(zmins and zmaxs)

    def _robust(frames, cols, q=0.01):
        """
        Slider domain from the 1st-99th percentile rather than the extremes.

        A handful of sentinel-ish records (a 1905 collection date among data that is
        otherwise 2010 onwards) would otherwise stretch the track over a century and
        leave 99% of the markers inside a few percent of its travel - unusable, and a
        histogram with nothing in it. The outliers stay reachable: a handle parked at
        either end means unbounded, so the default full range still includes everything.
        """
        vals = pd.concat([pd.to_numeric(f[c], errors="coerce") for f in frames for c in cols
                          if c in f.columns]).dropna()
        if vals.empty:
            return None, None, None, None
        return vals.quantile(q), vals.quantile(1 - q), vals.min(), vals.max()

    if has_dates:
        _date_secs = _to_secs(pd.concat(
            [f[c] for f in frames for c in ("dmin", "dmax") if c in f.columns]
        )).dropna()
        d_lo_s, d_hi_s = _date_secs.quantile(0.01), _date_secs.quantile(0.99)
        d_true_lo, d_true_hi = _date_secs.min(), _date_secs.max()
        if not (d_hi_s > d_lo_s):
            d_lo_s, d_hi_s = d_true_lo, max(d_true_hi, d_true_lo + 86400)
        date_lo = _fmt_date(pd.Timestamp(d_lo_s, unit="s"))
        date_hi = _fmt_date(pd.Timestamp(d_hi_s, unit="s"))
        date_true_lo = _fmt_date(pd.Timestamp(d_true_lo, unit="s"))
        date_true_hi = _fmt_date(pd.Timestamp(d_true_hi, unit="s"))
    else:
        date_lo, date_hi = "2000-01-01", "2030-01-01"
        date_true_lo, date_true_hi = date_lo, date_hi

    if has_depth:
        zl, zh, z_true_lo, z_true_hi = _robust(frames, ("zmin", "zmax"))
        z_lo, z_hi = float(zl), float(zh)
        z_true_lo, z_true_hi = float(z_true_lo), float(z_true_hi)
    else:
        z_lo, z_hi, z_true_lo, z_true_hi = 0.0, 1.0, 0.0, 1.0
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0

    # date bins work in epoch seconds so the same binning code serves both sliders
    def _epoch(frame, col):
        if frame is None or frame.empty or col not in frame.columns:
            return frame
        return frame.assign(**{col: _to_secs(frame[col])})

    if has_dates:
        d_lo, d_hi = d_lo_s, d_hi_s
        hits_d = _epoch(_epoch(hits, "dmin"), "dmax")
        bg_d = _epoch(_epoch(bg, "dmin"), "dmax")
        date_hist = _histogram_svg(
            _bin_counts(hits_d, "dmin", "dmax", d_lo, d_hi),
            _bin_counts(bg_d, "dmin", "dmax", d_lo, d_hi),
        )
    else:
        date_hist = ""

    depth_hist = _histogram_svg(
        _bin_counts(hits, "zmin", "zmax", z_lo, z_hi),
        _bin_counts(bg, "zmin", "zmax", z_lo, z_hi),
    ) if has_depth else ""

    # The legend is rebuilt in the browser from the markers actually visible in the
    # current viewport, so zooming into a region tells you what was matched there.
    delta_legend = (
        (f'<span class="leg-item"><span class="sw" style="background:{PARENT_COLOR}"></span>'
         f'<span class="leg-name">unmodified</span></span>' if 0 in deltas else "")
        + '<span class="leg-item"><span class="sw ramp-loss"></span>'
          '<span class="leg-name">losses</span></span>'
          '<span class="leg-item"><span class="sw ramp-gain"></span>'
          '<span class="leg-name">additions</span></span>'
    ) if analog else ""

    cfg = json.dumps({
        "dateLo": date_lo, "dateHi": date_hi,
        "zLo": z_lo, "zHi": z_hi,
        "hasDates": has_dates, "hasDepth": has_depth,
        "dateTrueLo": date_true_lo, "dateTrueHi": date_true_hi,
        "zTrueLo": z_true_lo, "zTrueHi": z_true_hi,
        "analog": analog,
    })

    defs = (f'<g id="gm-land"><path d="{_load_land_path()}" fill="var(--land)" '
            f'stroke="var(--land-stroke)" stroke-width="0.8" '
            f'vector-effect="non-scaling-stroke"></path></g>')
    if detail:
        defs += detail_svg
    if analog:
        defs += f'<g id="gm-ctx">{bg_svg}</g>'

    return (
        _TEMPLATE
        .replace("__CFG__", cfg)
        .replace("__DATE_HIST__", date_hist)
        .replace("__DEPTH_HIST__", depth_hist)
        .replace("__NO_DATE__", f"{n_no_date:,}")
        .replace("__NO_DEPTH__", f"{n_no_depth:,}")
        .replace("__ZLO__", f"{z_true_lo:.0f}")
        .replace("__ZHI__", f"{z_true_hi:.0f}")
        .replace("__DELTA_LEGEND__", delta_legend)
        .replace("__COLORBY_STYLE__", "" if analog else "display:none")
        .replace("__SIDE__", "".join(side))
        .replace("__SIDE_STYLE__", "" if analog else "display:none")
        .replace("__LAYOUT_CLASS__", "layout" if analog else "layout no-side")
        .replace("__GRID_CLASS__", "grid" if analog else "single")
        .replace("__CTX_CLASS__", "" if show_context else "no-ctx")
        .replace("__CTX_CHECKED__", "checked" if show_context else "")
        .replace("__DETAIL_ROW_STYLE__", "" if detail else "display:none")
        .replace("__DETAIL_CHECKED__", "checked" if (show_detail and detail) else "")
        .replace("__DETAIL_CLASS__", " detail-on" if (show_detail and detail) else "")
        .replace("__DEFS__", defs)
        .replace("__DAM_DATA__", _dam_payload(hits, max_dams=max_dams) if detail else "")
        .replace("__CARDS__", "".join(cards))
    )


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root {
  --paper:#eef3f8; --surface:#fff; --surface-2:#f4f7fa; --ink:#16232f; --ink-soft:#3d4d5c;
  --muted:#6b7c8c; --rule:#d7e0e8; --accent:#1d6fa5; --land:#c9d4de; --land-stroke:#a9bac8;
  --ocean:#dde8f1; --ctx:#61798c; --river:#8fb4cc; --lake:#cfe0ee; --border:#8d7f72; --city:#5d6c7a; --country:#7d6f62; --dam:#146c74;
  --shadow:0 1px 2px rgba(22,35,47,.06),0 4px 16px rgba(22,35,47,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#0d1620; --surface:#141e29; --surface-2:#182430; --ink:#eaf1f7; --ink-soft:#c3d0dc;
    --muted:#8ea0b0; --rule:#29394a; --accent:#5b9ce8; --land:#2c3b48; --land-stroke:#435466;
    --ocean:#0f1c28; --ctx:#8ba6bd; --river:#3f6c8c; --lake:#1b3346; --border:#8a7a6a; --city:#93a6b6; --country:#a39484; --dam:#3fb6c0;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);
  }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.wrap { padding:4px 2px 10px; }
.filter-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:0 0 12px; }
@media (max-width:860px) { .filter-grid { grid-template-columns:1fr; } }
.time-card { background:var(--surface); border:1px solid var(--rule); border-radius:12px;
  box-shadow:var(--shadow); padding:14px 18px 16px; }
.time-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; flex-wrap:wrap; gap:6px 16px; }
.time-label { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; font-weight:600; }
.time-window { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; color:var(--accent); font-variant-numeric:tabular-nums; }
.hist-wrap { margin:0 8px -2px; height:34px; }
.hist { width:100%; height:100%; display:block; overflow:visible; }
.hbar-tot { fill:var(--rule); }
.hbar-hit { fill:var(--accent); }
.hbar.dim { opacity:.28; }
.slider-track { position:relative; height:32px; }
.slider-track input[type=range] { position:absolute; top:9px; left:0; width:100%; height:14px; margin:0;
  background:transparent; pointer-events:none; -webkit-appearance:none; appearance:none; }
.slider-track input[type=range]::-webkit-slider-runnable-track { height:4px; background:transparent; }
.slider-track input[type=range]::-moz-range-track { height:4px; background:transparent; }
.slider-track input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; pointer-events:auto;
  width:16px; height:16px; border-radius:50%; background:var(--accent); border:2px solid var(--surface);
  box-shadow:var(--shadow); cursor:grab; margin-top:-6px; }
.slider-track input[type=range]::-moz-range-thumb { pointer-events:auto; width:16px; height:16px;
  border-radius:50%; background:var(--accent); border:2px solid var(--surface); box-shadow:var(--shadow); cursor:grab; }
/* 8px = half the range thumb, the distance its centre is inset from the track edge */
.slider-inner { position:absolute; left:8px; right:8px; top:0; bottom:0; }
.slider-rail { position:absolute; top:14px; left:0; right:0; height:4px; background:var(--rule); border-radius:2px; }
.slider-fill { position:absolute; top:14px; height:4px; background:var(--accent); border-radius:2px; }
.undated-toggle { display:flex; align-items:flex-start; gap:8px; margin:8px 0 0; font-size:12px;
  color:var(--ink-soft); cursor:pointer; user-select:none; }
.undated-toggle input { margin:2px 0 0; accent-color:var(--accent); flex:none; width:14px; height:14px; cursor:pointer; }
.time-buttons { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
.time-btn { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px; padding:5px 10px;
  border:1px solid var(--rule); border-radius:6px; background:var(--surface-2); color:var(--ink-soft); cursor:pointer; }
.time-btn:hover { border-color:var(--accent); color:var(--accent); }
.bar { display:flex; align-items:center; flex-wrap:wrap; gap:6px 8px; margin:0 0 12px; }
.bar-label { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; color:var(--muted); margin-right:2px; }
.leg-item { display:flex; align-items:center; gap:6px; font-size:12px; padding:4px 9px;
  border:1px solid var(--rule); border-radius:20px; background:var(--surface); }
.leg-item .sw { width:10px; height:10px; border-radius:50%; flex:none; }
.sw.ramp-loss { background:linear-gradient(90deg,#12374b,#57c4cf); width:22px; border-radius:5px; }
.sw.ramp-gain { background:linear-gradient(90deg,#7f2b18,#eaa96b); width:22px; border-radius:5px; }

.layout { display:grid; grid-template-columns:222px 1fr; gap:14px; align-items:start; }
.layout.no-side { grid-template-columns:1fr; }
@media (max-width:900px) { .layout { grid-template-columns:1fr; } }
.side { background:var(--surface); border:1px solid var(--rule); border-radius:12px;
  box-shadow:var(--shadow); padding:12px; position:sticky; top:6px; max-height:88vh; overflow:auto; }
.side h3 { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; font-weight:600;
  color:var(--muted); margin:0 0 8px; text-transform:uppercase; letter-spacing:.06em; }
.side-hint { font-size:11px; color:var(--muted); margin:0 0 10px; line-height:1.45; }
.side-item { display:flex; align-items:center; gap:8px; padding:5px 7px; border-radius:6px;
  cursor:pointer; user-select:none; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; }
.side-item:hover { background:var(--surface-2); }
.side-item.on { background:var(--accent); color:#fff; }
.side-item.on .side-count { color:#fff; }
.side-item .sw { width:10px; height:10px; border-radius:50%; flex:none; }
.side-item .side-name { flex:1; }
.side-item .side-count { color:var(--muted); font-size:11px; }
.side-item.empty { opacity:.32; }
.side-actions { display:flex; gap:6px; margin-top:10px; }

.info { position:relative; background:var(--surface); border:1px solid var(--accent);
  border-radius:12px; box-shadow:var(--shadow); padding:12px 34px 12px 14px; margin:0 0 12px; }
.info-body { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:6px 18px;
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; }
.info-row { display:flex; gap:8px; }
.info-row.wide { grid-column:1 / -1; }
.mod-list { display:flex; flex-wrap:wrap; gap:4px 6px; }
.mod-chip { border:1px solid var(--rule); border-radius:5px; padding:1px 6px; background:var(--surface-2); }
.info-k { color:var(--muted); min-width:82px; }
.info-v { color:var(--ink); overflow-wrap:anywhere; }
.info-close { position:absolute; top:8px; right:10px; border:none; background:none; cursor:pointer;
  color:var(--muted); font-size:17px; line-height:1; }
.info-close:hover { color:var(--ink); }
.pt.picked { stroke:var(--ink); stroke-width:2; vector-effect:non-scaling-stroke; }
.maps { display:grid; gap:12px; }
.maps.grid { grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); }
.maps.single { grid-template-columns:1fr; }
.facet { background:var(--surface); border:1px solid var(--rule); border-radius:12px;
  box-shadow:var(--shadow); padding:8px 8px 6px; overflow:hidden; }
.facet.off, .facet.empty { display:none; }
.facet-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; padding:2px 4px 6px; }
.facet-name { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; font-weight:600; }
.facet-meta { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px; color:var(--muted); }
.fmap { width:100%; height:auto; display:block; cursor:grab; touch-action:none; }
.fmap.dragging { cursor:grabbing; }
.pt { cursor:pointer; }
/* no outline on hits; the marker radius is counter-scaled on zoom but a stroke width
   is not, so any border grew thicker the further you zoomed in */
.pt.hit { stroke:none; opacity:.9; }
.pt.hit:hover { opacity:1; }
/* context markers are hollow, so their ring IS the mark - keep it, but non-scaling
   so it stays hairline at every zoom level */
.pt.bg { stroke:var(--ctx); stroke-width:1.2; opacity:.8; pointer-events:all;
  vector-effect:non-scaling-stroke; }
.pt.bg:hover { opacity:1; stroke-width:1.8; }
.pt.hidden { display:none; }
body.no-ctx .ctx-layer { display:none; }
/* detailed basemap: off unless the page opts in. Strokes are non-scaling so the
   coastline detail stays hairline instead of thickening as you zoom. */
.detail { display:none; }
body.detail-on .detail { display:inline; }
.ne-river { fill:none; stroke:var(--river); opacity:.9; vector-effect:non-scaling-stroke;
  stroke-linecap:round; stroke-linejoin:round; }
/* Natural Earth scalerank: 1 is the Amazon and the Nile, 2 the Danube, 4 the Rhine */
.ne-river-1 { stroke-width:2.6; }
.ne-river-2 { stroke-width:1.9; }
.ne-river-3 { stroke-width:1.2; }
.ne-river-4 { stroke-width:.8; opacity:.7; }
.ne-lake { fill:var(--lake); stroke:var(--river); stroke-width:.5; opacity:.9; vector-effect:non-scaling-stroke; }
/* Country outlines are solid and warm-neutral; rivers are thin, cool and
   semi-transparent, so the two never read as the same kind of line. */
.ne-border { fill:none; stroke:var(--border); stroke-width:1.1; opacity:.95;
  vector-effect:non-scaling-stroke; }
.ne-city { fill:var(--city); opacity:.5; }
.ne-city-label { fill:var(--city); font-family:ui-monospace,Menlo,Consolas,monospace;
  paint-order:stroke; stroke:var(--ocean); stroke-width:2.5; vector-effect:non-scaling-stroke; }
/* city names would be unreadable mush at world scale, so they fade in on zoom.
   The rule targets the <use> element, not the text inside the referenced group. */
.ne-country-label { fill:var(--country); font-family:"IBM Plex Sans",system-ui,sans-serif;
  font-weight:600; letter-spacing:.09em; text-anchor:middle; text-transform:uppercase;
  paint-order:stroke; stroke:var(--ocean); stroke-width:2.2; vector-effect:non-scaling-stroke; }
/* squares, not dots - a round mark reads as another match */
.ne-dam { fill:var(--dam); stroke:var(--ocean); stroke-width:1; vector-effect:non-scaling-stroke; }
.ne-dam-label { fill:var(--dam); font-family:ui-monospace,Menlo,Consolas,monospace;
  paint-order:stroke; stroke:var(--ocean); stroke-width:2.2; vector-effect:non-scaling-stroke; }
/* Each label set has its own zoom threshold, so the map fills in as you go deeper
   instead of turning to mush at world scale. */
.country-labels, .city-labels { opacity:0; transition:opacity .15s; }
.dams-live { opacity:.95; }
body:not(.detail-on) .dams-live { display:none; }
body.z2.names-country .country-labels { opacity:.85; }
body.zoomed-in .city-labels { opacity:.95; }
.credits { font-size:10.5px; line-height:1.5; color:var(--muted); margin:12px 2px 0;
  font-family:ui-monospace,Menlo,Consolas,monospace; }
.credits a { color:var(--muted); }
.credits a:hover { color:var(--accent); }
.tiles image { image-rendering:auto; }
body.tiles-on .detail, body.tiles-on .ctx-layer .pt.bg { }
body.tiles-on #gm-land, body.tiles-on .detail { display:none; }
#tooltip { position:fixed; pointer-events:none; background:var(--ink); color:var(--paper);
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; line-height:1.5; padding:8px 10px;
  border-radius:6px; white-space:pre-wrap; box-shadow:var(--shadow); opacity:0;
  transform:translate(-9999px,-9999px); transition:opacity .08s; z-index:50; max-width:320px; }
#tooltip.show { opacity:1; }
</style></head><body class="__CTX_CLASS____DETAIL_CLASS__">
<div class="wrap">

  <div class="filter-grid">
    <div class="time-card">
      <div class="time-head">
        <span class="time-label">Collection date window</span>
        <span class="time-window" id="timeWindowLabel"></span>
      </div>
      <div class="hist-wrap" id="dateHist">__DATE_HIST__</div>
      <div class="slider-track">
        <div class="slider-inner"><div class="slider-rail"></div><div class="slider-fill" id="sliderFill"></div></div>
        <input type="range" id="rangeMin" min="0" max="1000" value="0">
        <input type="range" id="rangeMax" min="0" max="1000" value="1000">
      </div>
      <label class="undated-toggle"><input type="checkbox" id="showUndated" checked>
        <span>Keep the __NO_DATE__ markers with no recorded date visible (independent of the window)</span></label>
      <div class="time-buttons">
        <button class="time-btn" id="btnFull">Full range</button>
        <button class="time-btn" id="btnLast5">Last 5 years</button>
        <button class="time-btn" id="btnLast1">Last 12 months</button>
      </div>
    </div>

    <div class="time-card">
      <div class="time-head">
        <span class="time-label">Depth / altitude window</span>
        <span class="time-window" id="depthWindowLabel"></span>
      </div>
      <div class="hist-wrap" id="depthHist">__DEPTH_HIST__</div>
      <div class="slider-track">
        <div class="slider-inner"><div class="slider-rail"></div><div class="slider-fill" id="depthFill"></div></div>
        <input type="range" id="depthMin" min="0" max="1000" value="0">
        <input type="range" id="depthMax" min="0" max="1000" value="1000">
      </div>
      <label class="undated-toggle"><input type="checkbox" id="showNoDepth" checked>
        <span>Keep the __NO_DEPTH__ markers with no recorded depth/altitude visible (independent of the window)</span></label>
      <div class="time-buttons">
        <button class="time-btn" id="btnDepthFull">Full range (__ZLO__m to __ZHI__m)</button>
        <button class="time-btn" id="btnBelowSurface">Below surface only (&lt; 0m)</button>
        <button class="time-btn" id="btnAboveSurface">Above surface only (&ge; 0m)</button>
      </div>
    </div>
  </div>

  <div class="bar">
    <span class="bar-label">size by</span>
    <select id="sizeMode" class="dropdown">
      <option value="n">matched files</option>
      <option value="nalt">distinct analogues</option>
    </select>
    <span class="bar-label" style="__COLORBY_STYLE__">color by</span>
    <select id="colorMode" class="dropdown" style="__COLORBY_STYLE__">
      <option value="delta">modification</option>
      <option value="cat">sample type</option>
    </select>
    <label class="undated-toggle" style="margin:0 0 0 6px"><input type="checkbox" id="showCtx" __CTX_CHECKED__>
      <span>Show sites with no hit</span></label>
    <label class="undated-toggle" style="margin:0 0 0 6px;__DETAIL_ROW_STYLE__"><input type="checkbox" id="showDetail" __DETAIL_CHECKED__>
      <span>Detailed base map</span></label>
    <label class="undated-toggle" style="margin:0 0 0 6px;__DETAIL_ROW_STYLE__"><input type="checkbox" id="showDams" checked>
      <span>Dams near matches</span></label>
    <label class="undated-toggle" style="margin:0 0 0 6px;__DETAIL_ROW_STYLE__"><input type="checkbox" id="showCountryNames">
      <span>Country names</span></label>
    <label class="undated-toggle" style="margin:0 0 0 6px;__DETAIL_ROW_STYLE__"><input type="checkbox" id="showDamNames">
      <span>Dam names</span></label>
    <label class="undated-toggle" style="margin:0 0 0 6px"><input type="checkbox" id="useTiles">
      <span>Street base map (loads external tiles)</span></label>
    <button class="time-btn" id="btnResetZoom">Reset zoom</button>
    <span class="bar-label" id="zoomLabel"></span>
  </div>

  <div class="bar" id="legendBar">
    <span class="bar-label" id="legendLabel">in view</span>
    <span id="deltaLegend">__DELTA_LEGEND__</span>
    <span id="catLegend"></span>
  </div>

  <div class="__LAYOUT_CLASS__">
    <aside class="side" style="__SIDE_STYLE__">
      <h3>Modifications</h3>
      <p class="side-hint">Showing the parent and all analogues together. Pick deltas to give each its own map.</p>
      <div id="sideList">__SIDE__</div>
      <div class="side-actions">
        <button class="time-btn" id="btnSideAll">Each</button>
        <button class="time-btn" id="btnSideNone">Combined</button>
      </div>
    </aside>
    <div>
      <div class="info" id="info" hidden>
        <button class="info-close" id="infoClose" title="Close">&times;</button>
        <div class="info-body" id="infoBody"></div>
      </div>
      <div class="maps __GRID_CLASS__" id="maps">__CARDS__</div>
    </div>
  </div>
</div>
<svg width="0" height="0" style="position:absolute"><defs>__DEFS__</defs></svg>
__DAM_DATA__
  <p class="credits">
    Base map data <a href="https://www.naturalearthdata.com/" target="_blank" rel="noopener">Natural Earth</a>
    (public domain). Dams from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>
    contributors (<a href="https://opendatacommons.org/licenses/odbl/" target="_blank" rel="noopener">ODbL</a>)
    and <a href="https://www.wikidata.org/" target="_blank" rel="noopener">Wikidata</a>
    (<a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank" rel="noopener">CC0</a>).
    Sample metadata from <a href="https://redu.gnps2.org/" target="_blank" rel="noopener">ReDU</a>.
    <span id="tileCredit" hidden>Street base map &copy;
      <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors,
      &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a>
      (<a href="https://carto.com/basemaps/" target="_blank" rel="noopener">basemap terms</a>).</span>
  </p>
<div id="tooltip"></div>
<script>
(function() {
  var CFG = __CFG__;
  var STEPS = 1000, VB_W = 960, VB_H = 480;
  var tip = document.getElementById('tooltip');
  var allPts = Array.from(document.querySelectorAll('.pt'));
  allPts.forEach(function(el) { el.setAttribute('data-r', el.getAttribute('r')); });

  // ---------- tooltip ----------
  var mapsEl = document.getElementById('maps');
  mapsEl.addEventListener('mousemove', function(e) {
    var el = e.target;
    if (!el.classList || !el.classList.contains('pt')) { tip.classList.remove('show'); return; }
    tip.textContent = el.getAttribute('data-tip');
    var x = e.clientX + 14, y = e.clientY + 14;
    if (x > window.innerWidth - 340) x = e.clientX - 330;
    if (y > window.innerHeight - 100) y = e.clientY - 90;
    tip.style.transform = 'translate(' + x + 'px,' + y + 'px)';
    tip.classList.add('show');
  });
  mapsEl.addEventListener('mouseleave', function() { tip.classList.remove('show'); });

  // ---------- date / depth windows ----------
  var DOMAIN_MIN = Date.parse(CFG.dateLo), DOMAIN_MAX = Date.parse(CFG.dateHi);
  var SPAN = Math.max(DOMAIN_MAX - DOMAIN_MIN, 1);
  function stepToDate(s) { return new Date(DOMAIN_MIN + (s / STEPS) * SPAN); }
  function fmtDate(d) { return d.toLocaleDateString('en-US', { year:'numeric', month:'short' }); }
  var Z_MIN = CFG.zLo, Z_MAX = CFG.zHi, ZSPAN = Math.max(Z_MAX - Z_MIN, 1e-6);
  function stepToDepth(s) { return Z_MIN + (s / STEPS) * ZSPAN; }

  var rangeMin = document.getElementById('rangeMin'), rangeMax = document.getElementById('rangeMax');
  var fill = document.getElementById('sliderFill'), label = document.getElementById('timeWindowLabel');
  var showUndated = document.getElementById('showUndated');
  var depthMin = document.getElementById('depthMin'), depthMax = document.getElementById('depthMax');
  var depthFill = document.getElementById('depthFill'), depthLabel = document.getElementById('depthWindowLabel');
  var showNoDepth = document.getElementById('showNoDepth');
  var dateBars = Array.from(document.querySelectorAll('#dateHist .hbar'));
  var depthBars = Array.from(document.querySelectorAll('#depthHist .hbar'));
  var HIST_BINS = 48;
  function dimBars(bars, lo, hi) {
    var b0 = Math.floor(lo / 1000 * HIST_BINS), b1 = Math.ceil(hi / 1000 * HIST_BINS) - 1;
    bars.forEach(function(r) {
      var i = +r.dataset.i;
      r.classList.toggle('dim', i < b0 || i > b1);
    });
  }
  var showCtx = document.getElementById('showCtx');
  var showDetail = document.getElementById('showDetail');
  var showDams = document.getElementById('showDams');
  var showCountryNames = document.getElementById('showCountryNames');
  var showDamNames = document.getElementById('showDamNames');
  if (showCountryNames) showCountryNames.addEventListener('change', function() {
    document.body.classList.toggle('names-country', showCountryNames.checked);
  });
  if (showDamNames) showDamNames.addEventListener('change', function() {
    damKey = ''; renderDams();
  });
  var sizeMode = document.getElementById('sizeMode');
  var colorMode = document.getElementById('colorMode');

  // ---------- bubble size: matched files, or how many modifications hit that site ----------
  function applySize() {
    var key = (sizeMode && sizeMode.value === 'nalt') ? 'data-rd' : 'data-rn';
    allPts.forEach(function(el) {
      var v = el.getAttribute(key);
      if (v !== null) el.setAttribute('data-r', v);
    });
    schedule();
  }
  if (sizeMode) sizeMode.addEventListener('change', applySize);

  // ---------- color: by modification, or by sample type ----------
  function applyColor() {
    var byCat = colorMode && colorMode.value === 'cat';
    document.querySelectorAll('.pt.hit').forEach(function(el) {
      var c = el.getAttribute(byCat ? 'data-ccat' : 'data-cdelta');
      if (c) el.setAttribute('fill', c);
    });
    var dl = document.getElementById('deltaLegend');
    if (dl) dl.style.display = byCat ? 'none' : '';
    schedule();
  }
  if (colorMode) colorMode.addEventListener('change', applyColor);

  // ---------- click a bubble for its details ----------
  var info = document.getElementById('info'), infoBody = document.getElementById('infoBody');
  function row(k, v) {
    return v ? '<div class="info-row"><span class="info-k">' + k +
               '</span><span class="info-v">' + v + '</span></div>' : '';
  }
  function fmtDelta(d) {
    return d === null ? '' : (+d === 0 ? 'unmodified (Δ 0 Da)' : 'Δ ' + (+d > 0 ? '+' : '') + d + ' Da');
  }
  function showInfo(el) {
    document.querySelectorAll('.pt.picked').forEach(function(p) { p.classList.remove('picked'); });

    // One site can carry several modifications, drawn as overlapping circles - a
    // click lands on whichever is on top, so gather every visible marker sharing
    // this exact coordinate and report the site, not just the circle that was hit.
    var cx = el.getAttribute('cx'), cy = el.getAttribute('cy');
    var peers = Array.from(el.parentNode.querySelectorAll('.pt')).filter(function(p) {
      return !p.classList.contains('hidden') &&
             p.getAttribute('cx') === cx && p.getAttribute('cy') === cy;
    });
    if (!peers.length) peers = [el];
    peers.forEach(function(p) { p.classList.add('picked'); });

    var lon = (+cx / 960 * 360 - 180), lat = (90 - +cy / 480 * 180);
    var files = 0, cats = new Map(), ds = new Set(), mods = [];
    var dmin = null, dmax = null, zmin = null, zmax = null;
    peers.forEach(function(p) {
      var n = +(p.getAttribute('data-n') || 0);
      files += n;
      var c = p.getAttribute('data-cat');
      if (c) cats.set(c, (cats.get(c) || 0) + n);
      var dd = p.getAttribute('data-ds');
      if (dd) dd.split(', ').forEach(function(x) { if (x) ds.add(x); });
      var d = p.getAttribute('data-delta');
      if (d !== null) mods.push({ d: +d, n: n });
      var a = p.getAttribute('data-dmin'), b = p.getAttribute('data-dmax');
      if (a && (dmin === null || a < dmin)) dmin = a;
      if (b && (dmax === null || b > dmax)) dmax = b;
      var za = p.getAttribute('data-zmin'), zb = p.getAttribute('data-zmax');
      if (za !== null && (zmin === null || +za < zmin)) zmin = +za;
      if (zb !== null && (zmax === null || +zb > zmax)) zmax = +zb;
    });
    mods.sort(function(a, b) { return a.d - b.d; });

    var catStr = [...cats.entries()].sort(function(a, b) { return b[1] - a[1]; })
      .map(function(e) { return e[0] + ' (' + e[1] + ')'; }).join(', ');
    var modStr = mods.map(function(m) {
      return '<span class="mod-chip">' + fmtDelta(String(m.d)) + ' · ' + m.n + '</span>';
    }).join('');

    infoBody.innerHTML =
      row('coordinates', lat.toFixed(3) + ', ' + lon.toFixed(3)) +
      row('sample type', catStr) +
      row('layer', el.classList.contains('hit') ? 'matched' : 'no hit (ReDU context)') +
      row('matched files', String(files)) +
      row('analogues here', el.getAttribute('data-nalt')) +
      row('collection', (dmin && dmax) ? (dmin === dmax ? dmin : dmin + ' → ' + dmax) : 'not recorded') +
      row('depth / alt', (zmin !== null && zmax !== null)
            ? (zmin === zmax ? zmin + 'm' : zmin + 'm → ' + zmax + 'm') : 'not recorded') +
      row('datasets', [...ds].join(', ')) +
      (mods.length
        ? '<div class="info-row wide"><span class="info-k">modifications</span>' +
          '<span class="info-v mod-list">' + modStr + '</span></div>'
        : '');
    info.hidden = false;
  }
  document.getElementById('infoClose').addEventListener('click', function() {
    info.hidden = true;
    document.querySelectorAll('.pt.picked').forEach(function(p) { p.classList.remove('picked'); });
  });

  function applyFilters() {
    var tLo = Math.min(+rangeMin.value, +rangeMax.value), tHi = Math.max(+rangeMin.value, +rangeMax.value);
    // The track covers the 1st-99th percentile, so a handle at an end means
    // "everything beyond this too" - otherwise the few outliers past the domain
    // could never be selected at all.
    var wStart = tLo === 0 ? -Infinity : stepToDate(tLo).getTime();
    var wEnd = tHi === STEPS ? Infinity : stepToDate(tHi).getTime();
    var keepUndated = showUndated.checked;
    var zLo = Math.min(+depthMin.value, +depthMax.value), zHi = Math.max(+depthMin.value, +depthMax.value);
    var zWLo = zLo === 0 ? -Infinity : stepToDepth(zLo);
    var zWHi = zHi === STEPS ? Infinity : stepToDepth(zHi);
    var keepNoDepth = showNoDepth.checked;

    allPts.forEach(function(el) {
      var dmin = el.getAttribute('data-dmin'), dmax = el.getAttribute('data-dmax');
      var timeOk = (dmin && dmax) ? (Date.parse(dmax) >= wStart && Date.parse(dmin) <= wEnd) : keepUndated;
      var zmin = el.getAttribute('data-zmin'), zmax = el.getAttribute('data-zmax');
      var depthOk = (zmin !== null && zmax !== null)
        ? (parseFloat(zmax) >= zWLo && parseFloat(zmin) <= zWHi) : keepNoDepth;
      el.classList.toggle('hidden', !(timeOk && depthOk));
    });

    fill.style.left = (tLo / STEPS * 100) + '%';
    fill.style.width = ((tHi - tLo) / STEPS * 100) + '%';
    label.textContent = CFG.hasDates
      ? (tLo === 0 ? fmtDate(new Date(Date.parse(CFG.dateTrueLo))) : fmtDate(stepToDate(tLo)))
        + '  →  ' +
        (tHi === STEPS ? fmtDate(new Date(Date.parse(CFG.dateTrueHi))) : fmtDate(stepToDate(tHi)))
      : 'no dates in this result';
    depthFill.style.left = (zLo / STEPS * 100) + '%';
    depthFill.style.width = ((zHi - zLo) / STEPS * 100) + '%';
    depthLabel.textContent = CFG.hasDepth
      ? (zLo === 0 ? CFG.zTrueLo.toFixed(0) : stepToDepth(zLo).toFixed(0)) + 'm  →  ' +
        (zHi === STEPS ? CFG.zTrueHi.toFixed(0) : stepToDepth(zHi).toFixed(0)) + 'm'
      : 'no depth/altitude in this result';

    dimBars(dateBars, tLo, tHi);
    dimBars(depthBars, zLo, zHi);
    visibleHitStamp++;

    refreshLayout();
  }

  [rangeMin, rangeMax, depthMin, depthMax].forEach(function(r) { r.addEventListener('input', applyFilters); });
  showUndated.addEventListener('change', applyFilters);
  showNoDepth.addEventListener('change', applyFilters);
  if (showCtx) showCtx.addEventListener('change', function() {
    document.body.classList.toggle('no-ctx', !showCtx.checked);
  });
  if (showDetail) showDetail.addEventListener('change', function() {
    document.body.classList.toggle('detail-on', showDetail.checked);
    damKey = ''; renderDams();
  });

  function setTimeFromCutoff(years) {
    var cutoff = Date.parse(CFG.dateTrueHi) - years * 365.25 * 24 * 3600 * 1000;
    rangeMin.value = Math.max(0, Math.round((cutoff - DOMAIN_MIN) / SPAN * STEPS));
    rangeMax.value = STEPS; applyFilters();
  }
  document.getElementById('btnFull').addEventListener('click', function() {
    rangeMin.value = 0; rangeMax.value = STEPS; applyFilters();
  });
  document.getElementById('btnLast5').addEventListener('click', function() { setTimeFromCutoff(5); });
  document.getElementById('btnLast1').addEventListener('click', function() { setTimeFromCutoff(1); });
  document.getElementById('btnDepthFull').addEventListener('click', function() {
    depthMin.value = 0; depthMax.value = STEPS; applyFilters();
  });
  document.getElementById('btnBelowSurface').addEventListener('click', function() {
    depthMin.value = 0;
    depthMax.value = Math.max(0, Math.min(STEPS, Math.round((0 - Z_MIN) / ZSPAN * STEPS)));
    applyFilters();
  });
  document.getElementById('btnAboveSurface').addEventListener('click', function() {
    depthMin.value = Math.max(0, Math.min(STEPS, Math.round((0 - Z_MIN) / ZSPAN * STEPS)));
    depthMax.value = STEPS; applyFilters();
  });

  // ---------- which maps are on screen ----------
  // No deltas picked: parent + one combined analogue map. Deltas picked: parent +
  // a map per picked delta, and the combined map steps aside.
  var sideItems = Array.from(document.querySelectorAll('.side-item'));
  var picked = new Set();

  function refreshLayout() {
    document.querySelectorAll('.facet').forEach(function(card) {
      var kind = card.dataset.kind, on = true;
      if (kind === 'combined') on = picked.size === 0;
      else if (kind === 'delta') on = picked.has(card.dataset.delta);
      card.classList.toggle('off', !on);
      // a map with nothing left after the date/depth windows drops out
      var n = card.querySelectorAll('.hits .pt:not(.hidden)').length;
      card.classList.toggle('empty', n === 0);
    });
    damKey = '';
    sideItems.forEach(function(it) {
      var card = document.querySelector('.facet[data-kind="delta"][data-delta="' + it.dataset.delta + '"]');
      var n = card ? card.querySelectorAll('.hits .pt:not(.hidden)').length : 0;
      it.classList.toggle('empty', n === 0);
      var c = it.querySelector('.side-count');
      if (c) c.textContent = n;
    });
  }

  sideItems.forEach(function(it) {
    it.addEventListener('click', function() {
      var d = it.dataset.delta;
      if (picked.has(d)) { picked.delete(d); it.classList.remove('on'); }
      else { picked.add(d); it.classList.add('on'); }
      refreshLayout();
    });
  });
  var bAll = document.getElementById('btnSideAll'), bNone = document.getElementById('btnSideNone');
  if (bAll) bAll.addEventListener('click', function() {
    sideItems.forEach(function(it) { picked.add(it.dataset.delta); it.classList.add('on'); });
    refreshLayout();
  });
  if (bNone) bNone.addEventListener('click', function() {
    picked.clear(); sideItems.forEach(function(it) { it.classList.remove('on'); });
    refreshLayout();
  });

  // ---------- optional Carto street base map ----------
  // The bundled vectors are equirectangular; XYZ tiles are Web Mercator, so they
  // cannot simply be laid underneath. Turning tiles on switches the whole map to
  // Mercator: markers are re-projected from the coordinates they carry, the vector
  // base layers step aside, and the viewBox grows to keep Mercator square.
  var MERC_LAT = 72;   // clipped; full Mercator reaches 85 and is mostly empty there
  var MERC_Y0 = (0.5 - Math.log(Math.tan(Math.PI / 4 + MERC_LAT * Math.PI / 360)) / (2 * Math.PI)) * VB_W;
  var VB_H_MERC = VB_W - 2 * MERC_Y0;
  var useTiles = document.getElementById('useTiles');
  var tileCredit = document.getElementById('tileCredit');
  function tilesOn() { return !!(useTiles && useTiles.checked); }
  function viewH() { return tilesOn() ? VB_H_MERC : VB_H; }
  function mercX(lon) { return (lon + 180) / 360 * VB_W; }
  function mercY(lat) {
    var f = Math.max(-MERC_LAT, Math.min(MERC_LAT, lat)) * Math.PI / 180;
    return (0.5 - Math.log(Math.tan(Math.PI / 4 + f / 2)) / (2 * Math.PI)) * VB_W - MERC_Y0;
  }

  function reproject() {
    var merc = tilesOn();
    allPts.forEach(function(el) {
      var la = parseFloat(el.getAttribute('data-lat')), lo = parseFloat(el.getAttribute('data-lon'));
      if (isNaN(la) || isNaN(lo)) return;
      el.setAttribute('cx', (merc ? mercX(lo) : (lo + 180) / 360 * VB_W).toFixed(3));
      el.setAttribute('cy', (merc ? mercY(la) : (90 - la) / 180 * VB_H).toFixed(3));
    });
    // trim a trailing .0 so switching back leaves the markup as it was served
    var h = viewH().toFixed(1).replace(/\.0$/, '');
    document.querySelectorAll('.fmap').forEach(function(svg) {
      svg.setAttribute('viewBox', '0 0 ' + VB_W + ' ' + h);
      var bg = svg.querySelector('rect');
      if (bg) bg.setAttribute('height', h);
    });
    document.body.classList.toggle('tiles-on', merc);
    if (tileCredit) tileCredit.hidden = !merc;
    damKey = '';
  }

  var TILE_URL = 'https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png';
  function renderTiles() {
    var groups = Array.from(document.querySelectorAll('.tiles'));
    if (!groups.length) return;
    if (!tilesOn()) { groups.forEach(function(g) { if (g.innerHTML) g.innerHTML = ''; }); return; }
    var z = Math.max(0, Math.min(18, Math.round(Math.log2(VB_W * view.k * pxPerUnit / 256))));
    var n = Math.pow(2, z), span = VB_W / n;
    var x0 = -view.x / view.k, x1 = (-view.x + VB_W) / view.k;
    var y0 = -view.y / view.k, y1 = (-view.y + viewH()) / view.k;
    var i0 = Math.max(0, Math.floor(x0 / span)), i1 = Math.min(n - 1, Math.floor(x1 / span));
    var j0 = Math.max(0, Math.floor((y0 + MERC_Y0) / span)), j1 = Math.min(n - 1, Math.floor((y1 + MERC_Y0) / span));
    var parts = [];
    for (var j = j0; j <= j1; j++) {
      for (var i = i0; i <= i1 && parts.length < 400; i++) {
        parts.push('<image href="' + TILE_URL.replace('{z}', z).replace('{x}', i).replace('{y}', j) +
                   '" x="' + (i * span).toFixed(3) + '" y="' + (j * span - MERC_Y0).toFixed(3) +
                   '" width="' + (span * 1.002).toFixed(3) + '" height="' + (span * 1.002).toFixed(3) +
                   '" preserveAspectRatio="none"></image>');
      }
    }
    var html = parts.join('');
    groups.forEach(function(g) { g.innerHTML = html; });
  }
  if (useTiles) useTiles.addEventListener('change', function() {
    reproject(); render(); renderTiles(); renderDams();
  });

  // ---------- synchronized pan / zoom ----------
  var zoomGroups = Array.from(document.querySelectorAll('.zoom'));
  var cityLabelUses = Array.from(document.querySelectorAll('.city-labels'));
  var cityDots = Array.from(document.querySelectorAll('#gm-detail .ne-city'));
  var countryLabelUses = Array.from(document.querySelectorAll('.country-labels'));
  var damLabelUses = Array.from(document.querySelectorAll('.dam-labels'));
  var maps = Array.from(document.querySelectorAll('.fmap'));
  var MAX_ZOOM = 1500;   // ~0.6 viewBox units across, a few km of ground
  var view = { k: 1, x: 0, y: 0 };
  var zoomLabel = document.getElementById('zoomLabel');
  var pending = false;

  function clampView() {
    view.k = Math.max(1, Math.min(MAX_ZOOM, view.k));
    var minX = VB_W - VB_W * view.k, minY = viewH() - viewH() * view.k;
    view.x = Math.max(minX, Math.min(0, view.x));
    view.y = Math.max(minY, Math.min(0, view.y));
  }
  function render() {
    pending = false;
    measure();
    var t = 'translate(' + view.x.toFixed(2) + ' ' + view.y.toFixed(2) + ') scale(' + view.k.toFixed(4) + ')';
    zoomGroups.forEach(function(g) { g.setAttribute('transform', t); });
    allPts.forEach(function(el) {
      el.setAttribute('r', Math.max(unitsForPx(0.4),
        parseFloat(el.getAttribute('data-r')) / view.k).toFixed(5));
    });
    // labels and city dots are in the base map, but they read at screen size too
    // Base-map type is sized in screen pixels. Sizing it in user units made every
    // label render at roughly 0.6 px per unit - a 4.6 unit label came out under 3 px
    // and was effectively invisible.
    cityLabelUses.forEach(function(u) { u.style.fontSize = unitsForPx(10.5).toFixed(3) + 'px'; });
    countryLabelUses.forEach(function(u) { u.style.fontSize = unitsForPx(11.5).toFixed(3) + 'px'; });
    damLabelUses.forEach(function(u) { u.style.fontSize = unitsForPx(9.5).toFixed(3) + 'px'; });
    var cityR = unitsForPx(2.4).toFixed(3);
    cityDots.forEach(function(c) { c.setAttribute('r', cityR); });
    if (zoomLabel) zoomLabel.textContent = view.k > 1.01 ? view.k.toFixed(1) + '×' : '';
    document.body.classList.toggle('z2', view.k >= 2);
    document.body.classList.toggle('zoomed-in', view.k >= 3);
    updateCatLegend();
    scheduleDams();
    renderTiles();
  }
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(render); } }

  // ---------- dams, drawn for the current viewport only ----------
  // Tens of thousands of them exist; putting them all in the DOM would stall the
  // page, and at world scale they would be a grey smear. Below DAM_ZOOM nothing is
  // drawn at all, so panning around a zoomed-out map costs nothing.
  var DAM_ZOOM = 2, DAM_LABEL_ZOOM = 4, DAM_MARK_CAP = 4000;
  var damEl = document.getElementById('damData');
  var damData = { c: [], n: [] };
  if (damEl) { try { damData = JSON.parse(damEl.textContent); } catch (e) {} }
  var damTimer = null, damKey = '';
  function esc(t) {
    return String(t).replace(/[&<>"]/g, function(c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function renderDams() {
    var groups = Array.from(document.querySelectorAll('.facet:not(.off):not(.empty) .dams-live'));
    var all = Array.from(document.querySelectorAll('.dams-live'));
    if (!all.length) return;
    var on = view.k >= DAM_ZOOM && document.body.classList.contains('detail-on') &&
             (!showDams || showDams.checked);
    if (!on) {
      damKey = '';
      all.forEach(function(g) { if (g.innerHTML) g.innerHTML = ''; });
      return;
    }
    var x0 = -view.x / view.k, x1 = (-view.x + VB_W) / view.k;
    var y0 = -view.y / view.k, y1 = (-view.y + viewH()) / view.k;
    var withLabels = view.k >= DAM_LABEL_ZOOM && !!(showDamNames && showDamNames.checked);
    var key = [x0.toFixed(2), y0.toFixed(2), x1.toFixed(2), withLabels,
               showDams && showDams.checked, visibleHitStamp].join('|');
    if (key === damKey) return;
    damKey = key;

    var r = unitsForPx(2.3), fs = unitsForPx(9.5);   // r is the square's half-side

    // Dams are only ever drawn near the matches. Shown everywhere they are noise,
    // and most sit on water too small for a 110m lakes / 50m rivers base map to
    // carry - which is why they looked like they stood in the middle of dry land.
    var hx = [], hy = [];
    document.querySelectorAll('.facet:not(.off):not(.empty) .hits .pt').forEach(function(el) {
      if (el.classList.contains('hidden')) return;
      hx.push(+el.getAttribute('cx')); hy.push(+el.getAttribute('cy'));
    });
    // ~1.2 viewBox units is roughly 50 km at the equator
    var NEAR_R = 1.2, NEAR_R2 = NEAR_R * NEAR_R;
    function nearHit(x, y) {
      for (var j = 0; j < hx.length; j++) {
        var dx = x - hx[j], dy = y - hy[j];
        if (dx * dx + dy * dy <= NEAR_R2) return true;
      }
      return false;
    }
    // Labels are thinned on a grid rather than by a flat cap. A cap simply takes
    // the first N found, which in a dense region silently drops whatever sorts
    // last - the Danube run-of-river dams, for instance, which carry no recorded
    // height. One label per cell keeps the coverage even, keeps names off each
    // other, and lets a dam surface as soon as it is alone in its cell.
    var cellW = unitsForPx(230), cellH = unitsForPx(22);
    var taken = withLabels ? Object.create(null) : null;
    var c = damData.c, n = damData.n, marks = [], labels = [], i, x, y, cell;
    for (i = 0; i < n.length && marks.length < DAM_MARK_CAP; i++) {
      x = c[2 * i]; y = c[2 * i + 1];
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      if (!nearHit(x, y)) continue;
      marks.push('<rect class="ne-dam" x="' + (x - r).toFixed(5) + '" y="' + (y - r).toFixed(5) +
                 '" width="' + (r * 2).toFixed(5) + '" height="' + (r * 2).toFixed(5) +
                 '" data-tip="' + esc(n[i]) + '\ndam"></rect>');
      if (!withLabels) continue;
      cell = Math.round(x / cellW) + ':' + Math.round(y / cellH);
      if (taken[cell]) continue;
      taken[cell] = 1;
      labels.push('<text class="ne-dam-label" x="' + (x + r * 1.8).toFixed(5) +
                  '" y="' + (y + r * 1.2).toFixed(5) + '" style="font-size:' + fs.toFixed(5) +
                  'px">' + esc(n[i]) + '</text>');
    }
    var html = marks.join('') + labels.join('');
    all.forEach(function(g) { g.innerHTML = ''; });
    groups.forEach(function(g) { g.innerHTML = html; });
  }
  function scheduleDams() { clearTimeout(damTimer); damTimer = setTimeout(renderDams, 110); }
  var visibleHitStamp = 0;
  if (showDams) showDams.addEventListener('change', function() { damKey = ''; renderDams(); });

  // Sample types among the markers currently on screen: zoom into a region and the
  // legend narrows to what was actually matched there.
  var catLegend = document.getElementById('catLegend');
  var legendLabel = document.getElementById('legendLabel');
  function updateCatLegend() {
    if (!catLegend) return;
    var x0 = -view.x / view.k, x1 = (-view.x + VB_W) / view.k;
    var y0 = -view.y / view.k, y1 = (-view.y + VB_H) / view.k;
    var seen = new Map();
    document.querySelectorAll('.facet:not(.off):not(.empty) .hits .pt').forEach(function(el) {
      if (el.classList.contains('hidden')) return;
      var cx = +el.getAttribute('cx'), cy = +el.getAttribute('cy');
      if (cx < x0 || cx > x1 || cy < y0 || cy > y1) return;
      var c = el.getAttribute('data-cat');
      if (!seen.has(c)) seen.set(c, { color: el.getAttribute('data-ccat'), n: 0 });
      seen.get(c).n += +(el.getAttribute('data-n') || 1);
    });
    var items = [...seen.entries()].sort(function(a, b) { return b[1].n - a[1].n; });
    catLegend.innerHTML = items.map(function(e) {
      return '<span class="leg-item"><span class="sw" style="background:' + e[1].color +
             '"></span><span class="leg-name">' + e[0] +
             '</span><span class="leg-count">' + e[1].n + '</span></span>';
    }).join('');
    if (legendLabel) legendLabel.textContent = items.length
      ? (view.k > 1.01 ? 'sample types in view' : 'sample types matched') : 'nothing in view';
  }
  // px per viewBox unit at the current layout width, before the zoom transform
  var pxPerUnit = 1;
  function measure() {
    var m = document.querySelector('.facet:not(.off):not(.empty) .fmap') || document.querySelector('.fmap');
    pxPerUnit = m ? (m.getBoundingClientRect().width / VB_W) || 1 : 1;
  }
  function unitsForPx(px) { return px / (view.k * pxPerUnit); }
  measure();
  window.addEventListener('resize', function() { measure(); schedule(); });

  function toUser(svg, cx, cy) {
    var r = svg.getBoundingClientRect();
    return { x: (cx - r.left) / r.width * VB_W, y: (cy - r.top) / r.height * viewH() };
  }

  maps.forEach(function(svg) {
    svg.addEventListener('wheel', function(e) {
      e.preventDefault();
      var p = toUser(svg, e.clientX, e.clientY);
      var k0 = view.k;
      view.k = Math.max(1, Math.min(MAX_ZOOM, k0 * Math.exp(-e.deltaY * 0.0015)));
      view.x = p.x - (p.x - view.x) * (view.k / k0);
      view.y = p.y - (p.y - view.y) * (view.k / k0);
      clampView(); schedule();
    }, { passive: false });

    // Pointer capture is taken only once a real drag starts. Capturing on
    // pointerdown would retarget the following click to the <svg>, so a plain click
    // on a marker could never be attributed to that marker.
    var drag = null;
    svg.addEventListener('pointerdown', function(e) {
      drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y,
               rect: svg.getBoundingClientRect(), el: e.target, moved: false };
    });
    svg.addEventListener('pointermove', function(e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      if (!drag.moved) {
        drag.moved = true;
        svg.classList.add('dragging');
        try { svg.setPointerCapture(e.pointerId); } catch (err) {}
      }
      view.x = drag.vx + dx / drag.rect.width * VB_W;
      view.y = drag.vy + dy / drag.rect.height * VB_H;
      clampView(); schedule();
    });
    ['pointerup', 'pointercancel'].forEach(function(ev) {
      svg.addEventListener(ev, function(e) {
        if (drag && !drag.moved && ev === 'pointerup' &&
            drag.el && drag.el.classList && drag.el.classList.contains('pt')) {
          showInfo(drag.el);
        }
        if (drag && drag.moved) { try { svg.releasePointerCapture(e.pointerId); } catch (err) {} }
        drag = null; svg.classList.remove('dragging');
      });
    });
  });

  document.getElementById('btnResetZoom').addEventListener('click', function() {
    view = { k: 1, x: 0, y: 0 }; schedule();
  });

  applyFilters();
  applySize();
  applyColor();
  render();
})();
</script>
</body></html>"""
