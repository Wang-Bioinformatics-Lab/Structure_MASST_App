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
LAND_PATH_FILE = os.path.join(HERE, "assets", "world_land_110m_equirect.path")

# viewBox of the bundled land outline
VB_W, VB_H = 960.0, 480.0

MISSING = {"missing value", "", "nan", "none", "null"}

# Qualitative palette; categories beyond this cycle back through it.
PALETTE = [
    "#1d6fa5", "#2a9d8f", "#a8477a", "#8a5a34", "#c98a2b", "#8fb8d4",
    "#c1502e", "#5b8c5a", "#7b6bb0", "#c2557a", "#4f8a8b", "#b07d3a",
    "#6a8caf", "#9c6b4f", "#588157", "#9d4edd", "#e07a5f", "#3d5a80",
]


def _load_land_path() -> str:
    try:
        with open(LAND_PATH_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


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


def _circles(agg: pd.DataFrame, colors: dict, layer: str, scale: float) -> tuple:
    """Emit <circle> markup plus counts of markers missing date / depth."""
    if agg.empty:
        return "", 0, 0

    x, y = _project(agg["lat"], agg["lon"])
    parts = []
    n_no_date = 0
    n_no_depth = 0

    for (_, r), px, py in zip(agg.iterrows(), x, y):
        cat = str(r["cat"])
        color = colors.get(cat, "#98a2ad")

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

        if has_date:
            attrs.append(f'data-dmin="{_fmt_date(r["dmin"])}"')
            attrs.append(f'data-dmax="{_fmt_date(r["dmax"])}"')
        if has_depth:
            attrs.append(f'data-zmin="{float(r["zmin"]):.1f}"')
            attrs.append(f'data-zmax="{float(r["zmax"]):.1f}"')

        tip = [
            cat,
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


def _facet_label(v) -> str:
    """Human label for one unit delta mass."""
    iv = int(round(float(v)))
    return "unmodified (Δ 0 Da)" if iv == 0 else f"Δ {iv:+d} Da"


def build_geomasst_map_html(
    df_hits: pd.DataFrame,
    df_background: Optional[pd.DataFrame] = None,
    env_col: str = "ENVOEnvironmentMaterial",
    compound_name: str = "",
    max_markers: int = 6000,
    facet_col: str = "Unit Delta Mass",
    max_facets: int = 12,
) -> str:
    """
    Build a standalone HTML document for embedding with streamlit.components.v1.html.

    df_hits       raw-data matches for one molecule (needs LatitudeandLongitude + env_col;
                  SampleCollectionDateandTime and DepthorAltitudeMeters enable the sliders)
    df_background the rest of the environmental ReDU corpus, drawn as hollow context markers
    facet_col     when present in df_hits (analog search emits 'Unit Delta Mass'), each
                  modification gets its own small map so analogues can be compared
                  side by side; the maps pan and zoom together.
    max_facets    cap on the number of small maps, taking the deltas with the most hits
    """
    faceted = bool(facet_col) and facet_col in getattr(df_hits, "columns", [])

    tidy_hits = _prepare(df_hits, env_col, facet_col if faceted else None)
    faceted = faceted and not tidy_hits.empty and "facet" in tidy_hits.columns \
        and tidy_hits["facet"].nunique() > 1

    hits = _aggregate(tidy_hits, max_markers, by=("facet",) if faceted else ())
    bg = _aggregate(_prepare(df_background, env_col), max_markers)

    # Category order and colors are shared across every map so the same material is
    # the same color in each facet; hits first, so the molecule's own materials get
    # the most distinct colors.
    cats: list = []
    for frame in (hits, bg):
        if not frame.empty:
            for c in frame.groupby("cat")["n"].sum().sort_values(ascending=False).index:
                if c not in cats:
                    cats.append(str(c))
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}

    bg_svg, bg_no_date, bg_no_depth = _circles(bg, colors, "bg", scale=2.0)

    # ---- one map, or one map per modification ----
    if faceted:
        order = hits.groupby("facet")["n"].sum().sort_values(ascending=False)
        chosen = list(order.index[:max_facets])
        # order the maps by delta so they read like a mass axis, not by abundance
        chosen.sort()
    else:
        chosen = [None]

    cards, chips = [], []
    hit_no_date = hit_no_depth = 0
    for idx, fv in enumerate(chosen):
        part = hits if fv is None else hits[hits["facet"] == fv]
        svg, nd, nz = _circles(part, colors, "hit", scale=2.6)
        hit_no_date += nd
        hit_no_depth += nz

        n_mark = 0 if part.empty else len(part)
        n_samp = 0 if part.empty else int(part["n"].sum())
        label = "All matches" if fv is None else _facet_label(fv)
        meta = f"{n_mark:,} sites &middot; {n_samp:,} files"

        # context and land are defined once and referenced, so N maps do not mean N
        # copies of a 60 KB coastline or of every context marker
        ctx_layer = '<use href="#gm-ctx"></use>' if faceted else f'<g class="ctx">{bg_svg}</g>'
        cards.append(
            f'<div class="facet" data-facet="{idx}">'
            f'<div class="facet-head"><span class="facet-name">{html.escape(label)}</span>'
            f'<span class="facet-meta">{meta}</span></div>'
            f'<svg class="fmap" viewBox="0 0 {VB_W:.0f} {VB_H:.0f}" preserveAspectRatio="xMidYMid meet">'
            f'<rect x="0" y="0" width="{VB_W:.0f}" height="{VB_H:.0f}" fill="var(--ocean)"></rect>'
            f'<g class="zoom"><use href="#gm-land"></use>{ctx_layer}'
            f'<g class="hits">{svg}</g></g></svg></div>'
        )
        if faceted:
            chips.append(
                f'<label class="chip on" data-facet="{idx}">'
                f'<span class="chip-name">{html.escape(label)}</span>'
                f'<span class="chip-count">{n_samp:,}</span></label>'
            )

    n_no_date = hit_no_date + bg_no_date
    n_no_depth = hit_no_depth + bg_no_depth

    # Slider domains span both layers so the windows cover everything drawn.
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

    legend = "\n".join(
        f'<span class="leg-item"><span class="sw" style="background:{colors[c]}"></span>'
        f'<span class="leg-name">{html.escape(c)}</span></span>'
        for c in cats[:24]
    )

    n_hit_markers = 0 if hits.empty else len(hits)
    n_hit_samples = 0 if hits.empty else int(hits["n"].sum())
    n_bg_markers = 0 if bg.empty else len(bg)
    n_total_deltas = 0 if not faceted else int(hits["facet"].nunique())

    summary = (
        f"{n_hit_markers:,} hit markers ({n_hit_samples:,} matched files) &middot; "
        f"{n_bg_markers:,} context markers &middot; "
        f"{html.escape(compound_name) if compound_name else 'selected molecule'}"
    )
    if faceted:
        shown = len(chosen)
        more = "" if shown >= n_total_deltas else f" (top {shown} of {n_total_deltas} by file count)"
        summary += f" &middot; {shown} modification{'s' if shown != 1 else ''}{more}"

    defs = f'<g id="gm-land"><path d="{_load_land_path()}" fill="var(--land)" stroke="var(--land-stroke)" stroke-width="0.5"></path></g>'
    if faceted:
        defs += f'<g id="gm-ctx">{bg_svg}</g>'

    cfg = json.dumps({
        "dateLo": date_lo, "dateHi": date_hi,
        "zLo": z_lo, "zHi": z_hi,
        "hasDates": has_dates, "hasDepth": has_depth,
        "faceted": faceted,
    })

    return (
        _TEMPLATE
        .replace("__CFG__", cfg)
        .replace("__SUMMARY__", summary)
        .replace("__NO_DATE__", f"{n_no_date:,}")
        .replace("__NO_DEPTH__", f"{n_no_depth:,}")
        .replace("__ZLO__", f"{z_lo:.0f}")
        .replace("__ZHI__", f"{z_hi:.0f}")
        .replace("__LEGEND__", legend)
        .replace("__CHIPS__", "".join(chips))
        .replace("__CHIPBAR_STYLE__", "" if faceted else "display:none")
        .replace("__GRID_CLASS__", "grid" if faceted else "single")
        .replace("__DEFS__", defs)
        .replace("__CARDS__", "".join(cards))
    )


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root {
  --paper:#eef3f8; --surface:#fff; --surface-2:#f4f7fa; --ink:#16232f; --ink-soft:#3d4d5c;
  --muted:#6b7c8c; --rule:#d7e0e8; --accent:#1d6fa5; --land:#c9d4de; --land-stroke:#a9bac8;
  --ocean:#dde8f1; --ctx:#93a6b6; --shadow:0 1px 2px rgba(22,35,47,.06),0 4px 16px rgba(22,35,47,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#0d1620; --surface:#141e29; --surface-2:#182430; --ink:#eaf1f7; --ink-soft:#c3d0dc;
    --muted:#8ea0b0; --rule:#29394a; --accent:#5b9ce8; --land:#2c3b48; --land-stroke:#435466;
    --ocean:#0f1c28; --ctx:#5c7386; --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);
  }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.wrap { padding:4px 2px 10px; }
.summary { font-size:12.5px; color:var(--muted); margin:0 0 10px; font-family:ui-monospace,Menlo,Consolas,monospace; }
.filter-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:0 0 14px; }
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
.bar { display:flex; align-items:center; flex-wrap:wrap; gap:6px 8px; margin:10px 0 12px; }
.bar-label { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; color:var(--muted); margin-right:2px; }
.leg-item { display:flex; align-items:center; gap:6px; font-size:12px; padding:4px 9px;
  border:1px solid var(--rule); border-radius:20px; background:var(--surface); }
.leg-item .sw { width:10px; height:10px; border-radius:50%; flex:none; }
.chip { display:flex; align-items:center; gap:7px; font-size:12px; padding:5px 11px; cursor:pointer;
  border:1px solid var(--rule); border-radius:20px; background:var(--surface); user-select:none;
  font-family:ui-monospace,Menlo,Consolas,monospace; transition:opacity .15s,border-color .15s; }
.chip:hover { border-color:var(--accent); }
.chip.off { opacity:.38; }
.chip-count { color:var(--muted); font-size:11px; }
.maps.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:12px; }
.maps.single .facet { width:100%; }
.facet { background:var(--surface); border:1px solid var(--rule); border-radius:12px;
  box-shadow:var(--shadow); padding:8px 8px 6px; overflow:hidden; }
.facet.off { display:none; }
.facet-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; padding:2px 4px 6px; }
.facet-name { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; font-weight:600; }
.facet-meta { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px; color:var(--muted); }
.fmap { width:100%; height:auto; display:block; cursor:grab; touch-action:none; }
.fmap.dragging { cursor:grabbing; }
.pt { cursor:pointer; }
.pt.hit { stroke:var(--surface); stroke-width:1.2; opacity:1; paint-order:stroke; }
.pt.bg { stroke:var(--ctx); stroke-width:.6; opacity:.3; pointer-events:all; }
.pt:hover { stroke-width:1.8; opacity:1; }
.pt.hidden { display:none; }
#tooltip { position:fixed; pointer-events:none; background:var(--ink); color:var(--paper);
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; line-height:1.5; padding:8px 10px;
  border-radius:6px; white-space:pre-wrap; box-shadow:var(--shadow); opacity:0;
  transform:translate(-9999px,-9999px); transition:opacity .08s; z-index:50; max-width:320px; }
#tooltip.show { opacity:1; }
</style></head><body>
<div class="wrap">
  <p class="summary">__SUMMARY__</p>

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

  <div class="bar" id="chipBar" style="__CHIPBAR_STYLE__">
    <span class="bar-label">modifications</span>
    __CHIPS__
    <button class="time-btn" id="btnAllDeltas">All</button>
    <button class="time-btn" id="btnNoDeltas">None</button>
  </div>

  <div class="bar">
    <span class="bar-label">hit materials</span>
    __LEGEND__
    <span class="leg-item"><span class="sw" style="background:none;border:1.5px solid var(--ctx)"></span><span class="leg-name">not matched</span></span>
    <button class="time-btn" id="btnResetZoom">Reset zoom</button>
    <span class="bar-label" id="zoomLabel"></span>
  </div>

  <svg width="0" height="0" style="position:absolute"><defs>__DEFS__</defs></svg>
  <div class="maps __GRID_CLASS__" id="maps">__CARDS__</div>
</div>
<div id="tooltip"></div>
<script>
(function() {
  var CFG = __CFG__;
  var STEPS = 1000, VB_W = 960, VB_H = 480;
  var tip = document.getElementById('tooltip');
  var allPts = Array.from(document.querySelectorAll('.pt'));

  allPts.forEach(function(el) { el.setAttribute('data-r', el.getAttribute('r')); });

  // ---------- tooltip ----------
  document.getElementById('maps').addEventListener('mousemove', function(e) {
    var el = e.target;
    if (!el.classList || !el.classList.contains('pt')) { tip.classList.remove('show'); return; }
    tip.textContent = el.getAttribute('data-tip');
    var x = e.clientX + 14, y = e.clientY + 14;
    if (x > window.innerWidth - 340) x = e.clientX - 330;
    if (y > window.innerHeight - 100) y = e.clientY - 90;
    tip.style.transform = 'translate(' + x + 'px,' + y + 'px)';
    tip.classList.add('show');
  });
  document.getElementById('maps').addEventListener('mouseleave', function() {
    tip.classList.remove('show');
  });

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
  }

  [rangeMin, rangeMax, depthMin, depthMax].forEach(function(r) { r.addEventListener('input', applyFilters); });
  showUndated.addEventListener('change', applyFilters);
  showNoDepth.addEventListener('change', applyFilters);

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

  // ---------- which modifications are shown ----------
  var chips = Array.from(document.querySelectorAll('.chip'));
  function setFacet(idx, on) {
    var card = document.querySelector('.facet[data-facet="' + idx + '"]');
    if (card) card.classList.toggle('off', !on);
    var chip = document.querySelector('.chip[data-facet="' + idx + '"]');
    if (chip) chip.classList.toggle('off', !on);
  }
  chips.forEach(function(c) {
    c.addEventListener('click', function() { setFacet(c.dataset.facet, c.classList.contains('off')); });
  });
  var btnAll = document.getElementById('btnAllDeltas'), btnNone = document.getElementById('btnNoDeltas');
  if (btnAll) btnAll.addEventListener('click', function() { chips.forEach(function(c) { setFacet(c.dataset.facet, true); }); });
  if (btnNone) btnNone.addEventListener('click', function() { chips.forEach(function(c) { setFacet(c.dataset.facet, false); }); });

  // ---------- synchronized pan / zoom across every map ----------
  var zoomGroups = Array.from(document.querySelectorAll('.zoom'));
  var maps = Array.from(document.querySelectorAll('.fmap'));
  var view = { k: 1, x: 0, y: 0 };
  var zoomLabel = document.getElementById('zoomLabel');
  var pending = false;

  function clampView() {
    view.k = Math.max(1, Math.min(40, view.k));
    // keep the map covering the frame, so you cannot pan the world off-screen
    var minX = VB_W - VB_W * view.k, minY = VB_H - VB_H * view.k;
    view.x = Math.max(minX, Math.min(0, view.x));
    view.y = Math.max(minY, Math.min(0, view.y));
  }
  function render() {
    pending = false;
    var t = 'translate(' + view.x.toFixed(2) + ' ' + view.y.toFixed(2) + ') scale(' + view.k.toFixed(4) + ')';
    zoomGroups.forEach(function(g) { g.setAttribute('transform', t); });
    // counter-scale marker radii so points stay a readable size as you zoom in
    allPts.forEach(function(el) {
      el.setAttribute('r', Math.max(0.25, parseFloat(el.getAttribute('data-r')) / view.k).toFixed(3));
    });
    if (zoomLabel) zoomLabel.textContent = view.k > 1.01 ? view.k.toFixed(1) + '×' : '';
  }
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(render); } }

  function toUser(svg, clientX, clientY) {
    var r = svg.getBoundingClientRect();
    return { x: (clientX - r.left) / r.width * VB_W, y: (clientY - r.top) / r.height * VB_H };
  }

  maps.forEach(function(svg) {
    svg.addEventListener('wheel', function(e) {
      e.preventDefault();
      var p = toUser(svg, e.clientX, e.clientY);
      var factor = Math.exp(-e.deltaY * 0.0015);
      var k0 = view.k;
      view.k = Math.max(1, Math.min(40, k0 * factor));
      // hold the point under the cursor fixed
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
