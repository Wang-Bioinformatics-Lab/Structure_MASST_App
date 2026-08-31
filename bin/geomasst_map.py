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


def _prepare(df: pd.DataFrame, env_col: str) -> pd.DataFrame:
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

    return out


def _aggregate(tidy: pd.DataFrame, max_markers: Optional[int]) -> pd.DataFrame:
    """One marker per (lat, lon, material), carrying the ranges its samples span."""
    if tidy.empty:
        return tidy

    def _datasets(s: pd.Series) -> str:
        u = [v for v in pd.Series(s).dropna().unique().tolist()][:4]
        return ", ".join(u)

    agg = (
        tidy.groupby(["lat", "lon", "cat"], as_index=False)
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


def _radius(n: int, scale: float) -> float:
    return round(min(scale * math.sqrt(max(int(n), 1)), 22.0), 2)


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
            f'cx="{px:.1f}"', f'cy="{py:.1f}"', f'r="{_radius(r["n"], scale)}"',
        ]
        if layer == "hit":
            attrs.append(f'fill="{color}"')
        else:
            attrs.append('fill="none"')
            attrs.append(f'stroke="{color}"')
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


def build_geomasst_map_html(
    df_hits: pd.DataFrame,
    df_background: Optional[pd.DataFrame] = None,
    env_col: str = "ENVOEnvironmentMaterial",
    compound_name: str = "",
    max_markers: int = 6000,
) -> str:
    """
    Build a standalone HTML document for embedding with streamlit.components.v1.html.

    df_hits       raw-data matches for one molecule (needs LatitudeandLongitude + env_col;
                  SampleCollectionDateandTime and DepthorAltitudeMeters enable the sliders)
    df_background the rest of the environmental ReDU corpus, drawn as hollow context markers
    """
    hits = _aggregate(_prepare(df_hits, env_col), max_markers)
    bg = _aggregate(_prepare(df_background, env_col), max_markers)

    # Category order and colors are shared across both layers, hits first so the
    # molecule's own materials get the most distinct colors.
    cats: list = []
    for frame in (hits, bg):
        if not frame.empty:
            for c in frame.groupby("cat")["n"].sum().sort_values(ascending=False).index:
                if c not in cats:
                    cats.append(str(c))
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}

    hit_svg, hit_no_date, hit_no_depth = _circles(hits, colors, "hit", scale=2.6)
    bg_svg, bg_no_date, bg_no_depth = _circles(bg, colors, "bg", scale=2.0)

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

    subtitle = html.escape(compound_name) if compound_name else "selected molecule"
    land = _load_land_path()
    cfg = json.dumps({
        "dateLo": date_lo, "dateHi": date_hi,
        "zLo": z_lo, "zHi": z_hi,
        "hasDates": has_dates, "hasDepth": has_depth,
    })

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root {{
  --paper:#eef3f8; --surface:#fff; --surface-2:#f4f7fa; --ink:#16232f; --ink-soft:#3d4d5c;
  --muted:#6b7c8c; --rule:#d7e0e8; --accent:#1d6fa5; --land:#c9d4de; --land-stroke:#a9bac8;
  --ocean:#dde8f1; --shadow:0 1px 2px rgba(22,35,47,.06),0 4px 16px rgba(22,35,47,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --paper:#0d1620; --surface:#141e29; --surface-2:#182430; --ink:#eaf1f7; --ink-soft:#c3d0dc;
    --muted:#8ea0b0; --rule:#29394a; --accent:#5b9ce8; --land:#2c3b48; --land-stroke:#435466;
    --ocean:#0f1c28; --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);
  }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.wrap {{ padding:4px 2px 10px; }}
.summary {{ font-size:12.5px; color:var(--muted); margin:0 0 10px; font-family:ui-monospace,Menlo,Consolas,monospace; }}
.filter-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:0 0 14px; }}
@media (max-width:860px) {{ .filter-grid {{ grid-template-columns:1fr; }} }}
.time-card {{ background:var(--surface); border:1px solid var(--rule); border-radius:12px;
  box-shadow:var(--shadow); padding:14px 18px 16px; }}
.time-card.disabled {{ opacity:.5; pointer-events:none; }}
.time-head {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; flex-wrap:wrap; gap:6px 16px; }}
.time-label {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; font-weight:600; }}
.time-window {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; color:var(--accent); font-variant-numeric:tabular-nums; }}
.slider-track {{ position:relative; height:32px; }}
.slider-track input[type=range] {{ position:absolute; top:9px; left:0; width:100%; height:14px; margin:0;
  background:transparent; pointer-events:none; -webkit-appearance:none; appearance:none; }}
.slider-track input[type=range]::-webkit-slider-runnable-track {{ height:4px; background:transparent; }}
.slider-track input[type=range]::-moz-range-track {{ height:4px; background:transparent; }}
.slider-track input[type=range]::-webkit-slider-thumb {{ -webkit-appearance:none; pointer-events:auto;
  width:16px; height:16px; border-radius:50%; background:var(--accent); border:2px solid var(--surface);
  box-shadow:var(--shadow); cursor:grab; margin-top:-6px; }}
.slider-track input[type=range]::-moz-range-thumb {{ pointer-events:auto; width:16px; height:16px;
  border-radius:50%; background:var(--accent); border:2px solid var(--surface); box-shadow:var(--shadow); cursor:grab; }}
.slider-rail {{ position:absolute; top:14px; left:0; right:0; height:4px; background:var(--rule); border-radius:2px; }}
.slider-fill {{ position:absolute; top:14px; height:4px; background:var(--accent); border-radius:2px; }}
.undated-toggle {{ display:flex; align-items:flex-start; gap:8px; margin:8px 0 0; font-size:12px;
  color:var(--ink-soft); cursor:pointer; user-select:none; }}
.undated-toggle input {{ margin:2px 0 0; accent-color:var(--accent); flex:none; width:14px; height:14px; cursor:pointer; }}
.time-buttons {{ display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }}
.time-btn {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px; padding:5px 10px;
  border:1px solid var(--rule); border-radius:6px; background:var(--surface-2); color:var(--ink-soft); cursor:pointer; }}
.time-btn:hover {{ border-color:var(--accent); color:var(--accent); }}
.legend-bar {{ display:flex; flex-wrap:wrap; gap:6px 8px; margin:10px 0 12px; }}
.leg-item {{ display:flex; align-items:center; gap:6px; font-size:12px; padding:4px 9px;
  border:1px solid var(--rule); border-radius:20px; background:var(--surface); }}
.leg-item .sw {{ width:10px; height:10px; border-radius:50%; flex:none; }}
.map-card {{ background:var(--surface); border:1px solid var(--rule); border-radius:12px;
  box-shadow:var(--shadow); padding:8px; overflow-x:auto; }}
.map-card svg {{ width:100%; height:auto; display:block; min-width:640px; }}
.pt {{ stroke-width:.6; opacity:.85; cursor:pointer; transition:opacity .12s; }}
.pt.hit {{ stroke:var(--surface); }}
.pt.bg {{ stroke-width:.9; opacity:.45; }}
.pt:hover {{ stroke-width:1.6; opacity:1; }}
.pt.hidden {{ display:none; }}
#tooltip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--paper);
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; line-height:1.5; padding:8px 10px;
  border-radius:6px; white-space:pre-wrap; box-shadow:var(--shadow); opacity:0;
  transform:translate(-9999px,-9999px); transition:opacity .08s; z-index:50; max-width:320px; }}
#tooltip.show {{ opacity:1; }}
</style></head><body>
<div class="wrap">
  <p class="summary">{n_hit_markers:,} hit markers ({n_hit_samples:,} matched files) &middot; {n_bg_markers:,} context markers &middot; {subtitle}</p>

  <div class="filter-grid">
    <div class="time-card{'' if has_dates else ' disabled'}">
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
        <span>Keep the {n_no_date:,} markers with no recorded date visible (independent of the window)</span></label>
      <div class="time-buttons">
        <button class="time-btn" id="btnFull">Full range</button>
        <button class="time-btn" id="btnLast5">Last 5 years</button>
        <button class="time-btn" id="btnLast1">Last 12 months</button>
      </div>
    </div>

    <div class="time-card{'' if has_depth else ' disabled'}">
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
        <span>Keep the {n_no_depth:,} markers with no recorded depth/altitude visible (independent of the window)</span></label>
      <div class="time-buttons">
        <button class="time-btn" id="btnDepthFull">Full range ({z_lo:.0f}m to {z_hi:.0f}m)</button>
        <button class="time-btn" id="btnBelowSurface">Below surface only (&lt; 0m)</button>
        <button class="time-btn" id="btnAboveSurface">Above surface only (&ge; 0m)</button>
      </div>
    </div>
  </div>

  <div class="legend-bar">{legend}</div>

  <div class="map-card">
    <svg viewBox="0 0 {VB_W:.0f} {VB_H:.0f}" role="img" aria-label="World map of GeoMASST hits">
      <rect x="0" y="0" width="{VB_W:.0f}" height="{VB_H:.0f}" fill="var(--ocean)"></rect>
      <path d="{land}" fill="var(--land)" stroke="var(--land-stroke)" stroke-width="0.5"></path>
      <g id="context">{bg_svg}</g>
      <g id="points">{hit_svg}</g>
    </svg>
  </div>
</div>
<div id="tooltip"></div>
<script>
(function() {{
  var CFG = {cfg};
  var STEPS = 1000;
  var tip = document.getElementById('tooltip');
  var allPts = Array.from(document.querySelectorAll('.pt'));

  allPts.forEach(function(el) {{
    el.addEventListener('mousemove', function(e) {{
      tip.textContent = el.getAttribute('data-tip');
      var x = e.clientX + 14, y = e.clientY + 14;
      if (x > window.innerWidth - 340) x = e.clientX - 330;
      if (y > window.innerHeight - 100) y = e.clientY - 90;
      tip.style.transform = 'translate(' + x + 'px,' + y + 'px)';
      tip.classList.add('show');
    }});
    el.addEventListener('mouseleave', function() {{
      tip.classList.remove('show');
      tip.style.transform = 'translate(-9999px,-9999px)';
    }});
  }});

  var DOMAIN_MIN = Date.parse(CFG.dateLo), DOMAIN_MAX = Date.parse(CFG.dateHi);
  var SPAN = Math.max(DOMAIN_MAX - DOMAIN_MIN, 1);
  function stepToDate(s) {{ return new Date(DOMAIN_MIN + (s / STEPS) * SPAN); }}
  function fmtDate(d) {{ return d.toLocaleDateString('en-US', {{ year:'numeric', month:'short' }}); }}

  var Z_MIN = CFG.zLo, Z_MAX = CFG.zHi, ZSPAN = Math.max(Z_MAX - Z_MIN, 1e-6);
  function stepToDepth(s) {{ return Z_MIN + (s / STEPS) * ZSPAN; }}

  var rangeMin = document.getElementById('rangeMin'), rangeMax = document.getElementById('rangeMax');
  var fill = document.getElementById('sliderFill'), label = document.getElementById('timeWindowLabel');
  var showUndated = document.getElementById('showUndated');
  var depthMin = document.getElementById('depthMin'), depthMax = document.getElementById('depthMax');
  var depthFill = document.getElementById('depthFill'), depthLabel = document.getElementById('depthWindowLabel');
  var showNoDepth = document.getElementById('showNoDepth');

  function applyFilters() {{
    var tLo = Math.min(+rangeMin.value, +rangeMax.value), tHi = Math.max(+rangeMin.value, +rangeMax.value);
    var wStart = stepToDate(tLo).getTime(), wEnd = stepToDate(tHi).getTime();
    var keepUndated = showUndated.checked;

    var zLo = Math.min(+depthMin.value, +depthMax.value), zHi = Math.max(+depthMin.value, +depthMax.value);
    var zWLo = stepToDepth(zLo), zWHi = stepToDepth(zHi);
    var keepNoDepth = showNoDepth.checked;

    allPts.forEach(function(el) {{
      var dmin = el.getAttribute('data-dmin'), dmax = el.getAttribute('data-dmax');
      var timeOk = (dmin && dmax)
        ? (Date.parse(dmax) >= wStart && Date.parse(dmin) <= wEnd)
        : keepUndated;

      var zmin = el.getAttribute('data-zmin'), zmax = el.getAttribute('data-zmax');
      var depthOk = (zmin !== null && zmax !== null)
        ? (parseFloat(zmax) >= zWLo && parseFloat(zmin) <= zWHi)
        : keepNoDepth;

      el.classList.toggle('hidden', !(timeOk && depthOk));
    }});

    fill.style.left = (tLo / STEPS * 100) + '%';
    fill.style.width = ((tHi - tLo) / STEPS * 100) + '%';
    label.textContent = CFG.hasDates
      ? fmtDate(stepToDate(tLo)) + '  \\u2192  ' + fmtDate(stepToDate(tHi))
      : 'no dates in this result';
    depthFill.style.left = (zLo / STEPS * 100) + '%';
    depthFill.style.width = ((zHi - zLo) / STEPS * 100) + '%';
    depthLabel.textContent = CFG.hasDepth
      ? stepToDepth(zLo).toFixed(0) + 'm  \\u2192  ' + stepToDepth(zHi).toFixed(0) + 'm'
      : 'no depth/altitude in this result';
  }}

  [rangeMin, rangeMax, depthMin, depthMax].forEach(function(r) {{ r.addEventListener('input', applyFilters); }});
  showUndated.addEventListener('change', applyFilters);
  showNoDepth.addEventListener('change', applyFilters);

  function setTimeFromCutoff(years) {{
    var cutoff = DOMAIN_MAX - years * 365.25 * 24 * 3600 * 1000;
    rangeMin.value = Math.max(0, Math.round((cutoff - DOMAIN_MIN) / SPAN * STEPS));
    rangeMax.value = STEPS; applyFilters();
  }}
  document.getElementById('btnFull').addEventListener('click', function() {{
    rangeMin.value = 0; rangeMax.value = STEPS; applyFilters();
  }});
  document.getElementById('btnLast5').addEventListener('click', function() {{ setTimeFromCutoff(5); }});
  document.getElementById('btnLast1').addEventListener('click', function() {{ setTimeFromCutoff(1); }});

  document.getElementById('btnDepthFull').addEventListener('click', function() {{
    depthMin.value = 0; depthMax.value = STEPS; applyFilters();
  }});
  document.getElementById('btnBelowSurface').addEventListener('click', function() {{
    depthMin.value = 0;
    depthMax.value = Math.max(0, Math.min(STEPS, Math.round((0 - Z_MIN) / ZSPAN * STEPS)));
    applyFilters();
  }});
  document.getElementById('btnAboveSurface').addEventListener('click', function() {{
    depthMin.value = Math.max(0, Math.min(STEPS, Math.round((0 - Z_MIN) / ZSPAN * STEPS)));
    depthMax.value = STEPS; applyFilters();
  }});

  applyFilters();
}})();
</script>
</body></html>"""
