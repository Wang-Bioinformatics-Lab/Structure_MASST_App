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
LAND_PATH_FILE = os.path.join(ASSETS, "world_land_110m_equirect.path")
# optional detail layers, all pre-projected into the same 960x480 viewBox
# (Natural Earth, public domain: 110m borders and lakes, 50m rivers, populated places)
BORDERS_FILE = os.path.join(ASSETS, "world_borders_110m_equirect.path")
LAKES_FILE = os.path.join(ASSETS, "world_lakes_110m_equirect.path")
RIVERS_FILE = os.path.join(ASSETS, "world_rivers_50m_equirect.path")
CITIES_FILE = os.path.join(ASSETS, "world_cities_equirect.json")

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


def _detail_svg(max_cities: int = 90) -> str:
    """Borders, lakes, rivers and major cities, drawn under the markers."""
    borders, lakes, rivers = _read_asset(BORDERS_FILE), _read_asset(LAKES_FILE), _read_asset(RIVERS_FILE)
    out = []
    if rivers:
        out.append(f'<path class="ne-river" d="{rivers}"></path>')
    if lakes:
        out.append(f'<path class="ne-lake" d="{lakes}"></path>')
    if borders:
        out.append(f'<path class="ne-border" d="{borders}"></path>')

    raw = _read_asset(CITIES_FILE)
    if raw:
        try:
            cities = json.loads(raw)[:max_cities]
        except ValueError:
            cities = []
        dots, labels = [], []
        for c in cities:
            dots.append(f'<circle class="ne-city" cx="{c["x"]}" cy="{c["y"]}" r="0.9"></circle>')
            labels.append(
                f'<text class="ne-city-label" x="{c["x"] + 2.2}" y="{c["y"] + 1.4}">'
                f'{html.escape(str(c["n"]))}</text>'
            )
        out.append("".join(dots))
        labels_g = f'<g id="gm-city-labels">{"".join(labels)}</g>'
    else:
        labels_g = ""
    if not out:
        return ""
    # Two separate defs: a <use> element can be styled by document CSS, but elements
    # *inside* a referenced subtree do not re-render when an ancestor selector starts
    # or stops matching - so anything that toggles needs its own referenced group.
    return f'<g id="gm-detail">{"".join(out)}</g>{labels_g}'


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
             color_key: str = "cat") -> tuple:
    """Emit <circle> markup plus counts of markers missing date / depth."""
    if agg.empty:
        return "", 0, 0

    x, y = _project(agg["lat"], agg["lon"])
    parts = []
    n_no_date = 0
    n_no_depth = 0

    for (_, r), px, py in zip(agg.iterrows(), x, y):
        cat = str(r["cat"])
        color = colors.get(r[color_key] if color_key in agg.columns else cat, "#98a2ad")

        has_date = pd.notna(r["dmin"]) and pd.notna(r["dmax"])
        has_depth = pd.notna(r["zmin"]) and pd.notna(r["zmax"])
        if not has_date:
            n_no_date += 1
        if not has_depth:
            n_no_depth += 1

        attrs = [
            f'class="pt {layer}"',
            f'cx="{px:.1f}"', f'cy="{py:.1f}"', f'r="{_radius(r["n"], scale, floor=4.0 if layer == "hit" else 0.0)}"',
        ]
        if layer == "hit":
            attrs.append(f'fill="{color}"')
        else:
            attrs.append('fill="none"')  # stroke color comes from CSS, deliberately neutral
        attrs.append(f'data-cat="{html.escape(cat, quote=True)}"')
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


def build_geomasst_map_html(
    df_hits: pd.DataFrame,
    df_background: Optional[pd.DataFrame] = None,
    env_col: str = "ENVOEnvironmentMaterial",
    compound_name: str = "",
    max_markers: int = 6000,
    facet_col: str = "Unit Delta Mass",
    show_context: bool = True,
    show_detail: bool = False,
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
                  lakes and major cities); off by default so the plain map stays quiet
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

    bg_svg, bg_no_date, bg_no_depth = _circles(bg, colors, "bg", scale=2.0)

    detail_svg = _detail_svg()
    detail = bool(detail_svg)

    # ---- the maps ----
    cards, side = [], []
    hit_no_date = hit_no_depth = 0

    def _card(idx, label, part, kind, delta=None):
        nonlocal hit_no_date, hit_no_depth
        svg, nd, nz = _circles(part, colors, "hit", scale=2.6, color_key=color_key)
        hit_no_date += nd
        hit_no_depth += nz
        n_mark = 0 if part.empty else len(part)
        n_samp = 0 if part.empty else int(part["n"].sum())
        ctx_layer = ('<use href="#gm-ctx" class="ctx-layer"></use>' if analog
                     else f'<g class="ctx ctx-layer">{bg_svg}</g>')
        detail_layer = ('<use href="#gm-detail" class="detail"></use>'
                        '<use href="#gm-city-labels" class="detail city-labels"></use>') if detail else ''
        attr_delta = "" if delta is None else f' data-delta="{int(round(float(delta)))}"'
        cards.append(
            f'<div class="facet" data-facet="{idx}" data-kind="{kind}"{attr_delta}>'
            f'<div class="facet-head"><span class="facet-name">{html.escape(label)}</span>'
            f'<span class="facet-meta">{n_mark:,} sites &middot; {n_samp:,} files</span></div>'
            f'<svg class="fmap" viewBox="0 0 {VB_W:.0f} {VB_H:.0f}" preserveAspectRatio="xMidYMid meet">'
            f'<rect x="0" y="0" width="{VB_W:.0f}" height="{VB_H:.0f}" fill="var(--ocean)"></rect>'
            f'<g class="zoom"><use href="#gm-land"></use>{detail_layer}{ctx_layer}'
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
    date_lo = _fmt_date(min(dmins)) if has_dates else "2000-01-01"
    date_hi = _fmt_date(max(dmaxs)) if has_dates else "2030-01-01"
    z_lo = float(min(zmins)) if has_depth else 0.0
    z_hi = float(max(zmaxs)) if has_depth else 1.0
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0

    if analog:
        legend = "".join(
            f'<span class="leg-item"><span class="sw" style="background:{PARENT_COLOR}"></span>'
            f'<span class="leg-name">unmodified</span></span>'
            if 0 in deltas else ""
        ) + (
            '<span class="leg-item"><span class="sw ramp-loss"></span>'
            '<span class="leg-name">losses</span></span>'
            '<span class="leg-item"><span class="sw ramp-gain"></span>'
            '<span class="leg-name">additions</span></span>'
        )
        legend_label = "deltas"
    else:
        legend = "".join(
            f'<span class="leg-item"><span class="sw" style="background:{material_colors[c]}"></span>'
            f'<span class="leg-name">{html.escape(c)}</span></span>'
            for c in hit_cats
        )
        legend_label = "hit materials" if hit_cats else ""

    cfg = json.dumps({
        "dateLo": date_lo, "dateHi": date_hi,
        "zLo": z_lo, "zHi": z_hi,
        "hasDates": has_dates, "hasDepth": has_depth,
        "analog": analog,
    })

    defs = (f'<g id="gm-land"><path d="{_load_land_path()}" fill="var(--land)" '
            f'stroke="var(--land-stroke)" stroke-width="0.5"></path></g>')
    if detail:
        defs += detail_svg
    if analog:
        defs += f'<g id="gm-ctx">{bg_svg}</g>'

    return (
        _TEMPLATE
        .replace("__CFG__", cfg)
        .replace("__NO_DATE__", f"{n_no_date:,}")
        .replace("__NO_DEPTH__", f"{n_no_depth:,}")
        .replace("__ZLO__", f"{z_lo:.0f}")
        .replace("__ZHI__", f"{z_hi:.0f}")
        .replace("__LEGEND_LABEL__", legend_label)
        .replace("__LEGEND__", legend)
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
        .replace("__CARDS__", "".join(cards))
    )


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root {
  --paper:#eef3f8; --surface:#fff; --surface-2:#f4f7fa; --ink:#16232f; --ink-soft:#3d4d5c;
  --muted:#6b7c8c; --rule:#d7e0e8; --accent:#1d6fa5; --land:#c9d4de; --land-stroke:#a9bac8;
  --ocean:#dde8f1; --ctx:#93a6b6; --river:#8fb4cc; --lake:#cfe0ee; --border:#aab9c6; --city:#5d6c7a;
  --shadow:0 1px 2px rgba(22,35,47,.06),0 4px 16px rgba(22,35,47,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#0d1620; --surface:#141e29; --surface-2:#182430; --ink:#eaf1f7; --ink-soft:#c3d0dc;
    --muted:#8ea0b0; --rule:#29394a; --accent:#5b9ce8; --land:#2c3b48; --land-stroke:#435466;
    --ocean:#0f1c28; --ctx:#5c7386; --river:#3f6c8c; --lake:#1b3346; --border:#4a5b6d; --city:#93a6b6;
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
.pt.bg { stroke:var(--ctx); stroke-width:.9; opacity:.3; pointer-events:all;
  vector-effect:non-scaling-stroke; }
.pt.bg:hover { opacity:.7; }
.pt.hidden { display:none; }
body.no-ctx .ctx-layer { display:none; }
/* detailed basemap: off unless the page opts in. Strokes are non-scaling so the
   coastline detail stays hairline instead of thickening as you zoom. */
.detail { display:none; }
body.detail-on .detail { display:inline; }
.ne-river { fill:none; stroke:var(--river); stroke-width:.7; opacity:.75; vector-effect:non-scaling-stroke; }
.ne-lake { fill:var(--lake); stroke:var(--river); stroke-width:.5; opacity:.9; vector-effect:non-scaling-stroke; }
.ne-border { fill:none; stroke:var(--border); stroke-width:.6; opacity:.8;
  stroke-dasharray:2 1.4; vector-effect:non-scaling-stroke; }
.ne-city { fill:var(--city); opacity:.5; }
.ne-city-label { fill:var(--city); font-family:ui-monospace,Menlo,Consolas,monospace;
  paint-order:stroke; stroke:var(--ocean); stroke-width:2.5; vector-effect:non-scaling-stroke; }
/* city names would be unreadable mush at world scale, so they fade in on zoom.
   The rule targets the <use> element, not the text inside the referenced group. */
.city-labels { opacity:0; transition:opacity .15s; font-size:4.6px; }
body.zoomed-in .city-labels { opacity:.95; }
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
      <div class="slider-track">
        <div class="slider-rail"></div><div class="slider-fill" id="sliderFill"></div>
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
      <div class="slider-track">
        <div class="slider-rail"></div><div class="slider-fill" id="depthFill"></div>
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
    <span class="bar-label">__LEGEND_LABEL__</span>
    __LEGEND__
    <label class="undated-toggle" style="margin:0 0 0 6px"><input type="checkbox" id="showCtx" __CTX_CHECKED__>
      <span>Show sites with no hit</span></label>
    <label class="undated-toggle" style="margin:0 0 0 6px;__DETAIL_ROW_STYLE__"><input type="checkbox" id="showDetail" __DETAIL_CHECKED__>
      <span>Detailed base map</span></label>
    <button class="time-btn" id="btnResetZoom">Reset zoom</button>
    <span class="bar-label" id="zoomLabel"></span>
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
    <div class="maps __GRID_CLASS__" id="maps">__CARDS__</div>
  </div>
</div>
<svg width="0" height="0" style="position:absolute"><defs>__DEFS__</defs></svg>
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
  var showCtx = document.getElementById('showCtx');
  var showDetail = document.getElementById('showDetail');

  function applyFilters() {
    var tLo = Math.min(+rangeMin.value, +rangeMax.value), tHi = Math.max(+rangeMin.value, +rangeMax.value);
    var wStart = stepToDate(tLo).getTime(), wEnd = stepToDate(tHi).getTime();
    var keepUndated = showUndated.checked;
    var zLo = Math.min(+depthMin.value, +depthMax.value), zHi = Math.max(+depthMin.value, +depthMax.value);
    var zWLo = stepToDepth(zLo), zWHi = stepToDepth(zHi);
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
      ? fmtDate(stepToDate(tLo)) + '  →  ' + fmtDate(stepToDate(tHi)) : 'no dates in this result';
    depthFill.style.left = (zLo / STEPS * 100) + '%';
    depthFill.style.width = ((zHi - zLo) / STEPS * 100) + '%';
    depthLabel.textContent = CFG.hasDepth
      ? stepToDepth(zLo).toFixed(0) + 'm  →  ' + stepToDepth(zHi).toFixed(0) + 'm'
      : 'no depth/altitude in this result';

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
  });

  function setTimeFromCutoff(years) {
    var cutoff = DOMAIN_MAX - years * 365.25 * 24 * 3600 * 1000;
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

  // ---------- synchronized pan / zoom ----------
  var zoomGroups = Array.from(document.querySelectorAll('.zoom'));
  var cityLabelUses = Array.from(document.querySelectorAll('.city-labels'));
  var cityDots = Array.from(document.querySelectorAll('#gm-detail .ne-city'));
  cityDots.forEach(function(c) { c.setAttribute('data-r', c.getAttribute('r')); });
  var maps = Array.from(document.querySelectorAll('.fmap'));
  var view = { k: 1, x: 0, y: 0 };
  var zoomLabel = document.getElementById('zoomLabel');
  var pending = false;

  function clampView() {
    view.k = Math.max(1, Math.min(40, view.k));
    var minX = VB_W - VB_W * view.k, minY = VB_H - VB_H * view.k;
    view.x = Math.max(minX, Math.min(0, view.x));
    view.y = Math.max(minY, Math.min(0, view.y));
  }
  function render() {
    pending = false;
    var t = 'translate(' + view.x.toFixed(2) + ' ' + view.y.toFixed(2) + ') scale(' + view.k.toFixed(4) + ')';
    zoomGroups.forEach(function(g) { g.setAttribute('transform', t); });
    allPts.forEach(function(el) {
      el.setAttribute('r', Math.max(0.25, parseFloat(el.getAttribute('data-r')) / view.k).toFixed(3));
    });
    // labels and city dots are in the base map, but they read at screen size too
    cityLabelUses.forEach(function(u) { u.style.fontSize = (4.6 / view.k).toFixed(3) + 'px'; });
    cityDots.forEach(function(c) {
      c.setAttribute('r', Math.max(0.15, parseFloat(c.getAttribute('data-r')) / view.k).toFixed(3));
    });
    if (zoomLabel) zoomLabel.textContent = view.k > 1.01 ? view.k.toFixed(1) + '×' : '';
    document.body.classList.toggle('zoomed-in', view.k >= 3);
  }
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(render); } }
  function toUser(svg, cx, cy) {
    var r = svg.getBoundingClientRect();
    return { x: (cx - r.left) / r.width * VB_W, y: (cy - r.top) / r.height * VB_H };
  }

  maps.forEach(function(svg) {
    svg.addEventListener('wheel', function(e) {
      e.preventDefault();
      var p = toUser(svg, e.clientX, e.clientY);
      var k0 = view.k;
      view.k = Math.max(1, Math.min(40, k0 * Math.exp(-e.deltaY * 0.0015)));
      view.x = p.x - (p.x - view.x) * (view.k / k0);
      view.y = p.y - (p.y - view.y) * (view.k / k0);
      clampView(); schedule();
    }, { passive: false });

    var drag = null;
    svg.addEventListener('pointerdown', function(e) {
      drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, rect: svg.getBoundingClientRect() };
      svg.setPointerCapture(e.pointerId); svg.classList.add('dragging');
    });
    svg.addEventListener('pointermove', function(e) {
      if (!drag) return;
      view.x = drag.vx + (e.clientX - drag.x) / drag.rect.width * VB_W;
      view.y = drag.vy + (e.clientY - drag.y) / drag.rect.height * VB_H;
      clampView(); schedule();
    });
    ['pointerup', 'pointercancel'].forEach(function(ev) {
      svg.addEventListener(ev, function(e) {
        drag = null; svg.classList.remove('dragging');
        try { svg.releasePointerCapture(e.pointerId); } catch (err) {}
      });
    });
  });

  document.getElementById('btnResetZoom').addEventListener('click', function() {
    view = { k: 1, x: 0, y: 0 }; schedule();
  });

  applyFilters();
  render();
})();
</script>
</body></html>"""
