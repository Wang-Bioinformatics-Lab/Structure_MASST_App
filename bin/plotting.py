import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import re
import numpy as np
import json
import os
import pyarrow.dataset as ds



def raw_data_sankey(df, col1, col2, col3, col4):

    # 1) Collapse to top10 + "others" first (this also converts NaN → "others")
    df = df.copy()
    for col in (col1, col2, col3, col4):
        top10 = df[col].value_counts(dropna=True).nlargest(10).index
        df[col + "_s"] = df[col].where(df[col].isin(top10), "others")

    stages = [col1 + "_s", col2 + "_s", col3 + "_s", col4 + "_s"]

    # 2) Now cast to str (after the collapse)
    df[stages] = df[stages].astype(str)

    # 3) Only now set color_key so its dtype matches the keys you'll build
    df["color_key"] = df[stages[0]]

    # 4) Build labels / idx as you had (these are strings now)
    labels = []
    for i, stg in enumerate(stages, start=1):
        uniques = df[stg].dropna().unique().tolist()
        labels += [f"{i}_{u}" for u in uniques]
    labels = list(dict.fromkeys(labels))

    idx = {}
    for i, stg in enumerate(stages, start=1):
        for u in df[stg].dropna().unique():
            idx[(stg, u)] = labels.index(f"{i}_{u}")

    # 5) Color map: tile palette to exact length (avoids version-dependent length)
    import math, plotly.express as px
    column_1_vals = df[stages[0]].unique().tolist()

    base = px.colors.qualitative.Safe
    palette = (base * math.ceil(len(column_1_vals) / len(base)))[:len(column_1_vals)]
    color_map = dict(zip(column_1_vals, palette))

    # Optional: pin "others" to a neutral gray
    color_map["others"] = "#B0B0B0"

    source, target, value, link_colors = [], [], [], []
    for i in range(len(stages) - 1):
        grp = (
            df
            .dropna(subset=[stages[i], stages[i+1], "color_key"])
            .groupby([stages[i], stages[i+1], "color_key"])
            .size()
            .reset_index(name="count")
        )
        for _, row in grp.iterrows():
            src_val = row[stages[i]]
            tgt_val = row[stages[i+1]]
            color_key = row["color_key"]
            source.append(idx[(stages[i], src_val)])
            target.append(idx[(stages[i+1], tgt_val)])
            value.append(row["count"])
            link_colors.append(color_map.get(color_key, "rgba(0,0,0,0.3)"))

    fig = go.Figure(go.Sankey(
        textfont=dict(family="Arial, sans-serif", size=12, color="black"),
        arrangement="snap",
        node=dict(
            label=labels,
            color=["#F2F2F2"] * len(labels),
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        ),
    ))

    # Add stage annotations
    stage_names = [col1, col2, col3, col4]
    n = len(stage_names) - 1
    for i, name_ in enumerate(stage_names):
        x = i / n
        xanchor = "left" if i == 0 else "right" if i == n else "center"
        fig.add_annotation(
            x=x, y=1.02, xref="paper", yref="paper",
            text=name_, showarrow=False,
            font=dict(size=14, color="black"),
            xanchor=xanchor
        )

    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=60, t=120, b=20),
    )

    return fig


def export_hits_map(
    df,
    interactive=True,
    out_basename="hit_map",
    max_mri_examples=10,
    env_col="ENVOEnvironmentMaterial",
    # Map engine / style
    engine="mapbox",                 # "mapbox" shows country/city names when zooming
    map_style="open-street-map",     # no token needed; try "carto-positron" too
    projection="natural earth",      # used only when engine="geo"
    # Geo engine decorations
    show_borders=True,
    show_coastlines=True,
    show_graticules=True,
    # Optional admin boundaries overlay (Mapbox engine only)
    admin_geojson=None,              # dict or path to GeoJSON with borders
    admin_line_color="rgba(80,80,80,0.7)",
    admin_line_width=1,
    admin_opacity=0.8,
    # Hover content
    hover_mri="count",               # "none" | "count" | "examples"
    redu_feather = "database/redu.feather"
):
    """
    Build a world map of hit locations from a DataFrame with columns:
      - 'mri'
      - 'LatitudeandLongitude' formatted as 'lat|lon', e.g. '32.876878|-117.234459'
      - env_col (default: 'ENVOEnvironmentMaterial') used for color

    Exports:
      - PNG:  <out_basename>.png   (requires 'kaleido')
      - HTML: <out_basename>.html  (if interactive=True; mouse wheel zoom enabled)

    hover_mri:
      - "none"     → no MRI info in hover
      - "count"    → only show hit count
      - "examples" → show up to max_mri_examples examples
    """

    # -------- ReDU: load all possible MRIs (environmental, with coords), drop ones we've actually hit
    df_redu = None
    if os.path.exists(redu_feather):
        dataset = ds.dataset(redu_feather, format="feather")
        table = dataset.to_table(
            filter=(
                (ds.field("SampleType") == "environmental")
                & (ds.field(env_col) != "missing value")
                & (ds.field("LatitudeandLongitude") != "missing value")
            )
        )
        df_redu = table.to_pandas()
        if "USI" in df_redu.columns:
            df_redu = df_redu.rename(columns={"USI": "mri"})
        # Remove any we actually hit (by MRI)
        if "mri" in df.columns:
            hit_mris = set(df["mri"].astype(str))
            df_redu = df_redu[~df_redu["mri"].astype(str).isin(hit_mris)].copy()
    else:
        df_redu = pd.read_csv("https://redu.gnps2.org/dump", sep = '\t')
        if "USI" in df_redu.columns:
            df_redu = df_redu.rename(columns={"USI": "mri"})
        # Remove any we actually hit (by MRI)
        if "mri" in df.columns:
            hit_mris = set(df["mri"].astype(str))
            df_redu = df_redu[~df_redu["mri"].astype(str).isin(hit_mris)].copy()

            # filter for env and latlon and env
            df_redu = df_redu[(df_redu["SampleType"] == "environmental") & 
            (df_redu["LatitudeandLongitude"] != "missing value") & 
            (df_redu[env_col] != "missing value")
            ]

    # -------- Validate input
    required = {"LatitudeandLongitude", "mri", env_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    # -------- Clean + parse coords (hits)
    df = df[df[env_col].notna() & (df[env_col] != "missing value")].copy()
    latlon = df["LatitudeandLongitude"].astype(str).str.extract(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*$"
    )
    lat = pd.to_numeric(latlon[0], errors="coerce")
    lon = pd.to_numeric(latlon[1], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    df2 = df.loc[valid].copy()
    df2["lat"] = lat.loc[valid].values
    df2["lon"] = lon.loc[valid].values

    # Empty: still return a figure
    if df2.empty:
        fig = (px.scatter_geo(projection=projection) if engine=="geo"
               else px.scatter_mapbox(lat=[], lon=[], mapbox_style=map_style, zoom=1))
        fig.update_layout(
            title="Hit Locations (no valid rows)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        if engine == "geo":
            fig.update_geos(
                showland=True, landcolor="#f6f6f6",
                showocean=True, oceancolor="#eef6ff",
                showcountries=show_borders, countrycolor="rgba(80,80,80,0.5)",
                showcoastlines=show_coastlines, coastlinecolor="rgba(80,80,80,0.5)",
                lataxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
                lonaxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
            )
        return fig, df2

    # -------- Aggregate HITS per (lat, lon, env)
    def _join_examples(s):
        if hover_mri != "examples":
            return ""
        u = pd.Series(s.astype(str).unique())
        return ", ".join(u.iloc[:max_mri_examples]) + (f", … (+{len(u)-max_mri_examples} more)" if len(u) > max_mri_examples else "")

    hits = (
        df2.groupby(["lat", "lon", env_col], as_index=False)
           .agg(n=("mri", "count"), mri_examples=("mri", _join_examples))
           .sort_values("n", ascending=False)
           .reset_index(drop=True)
    )
    cat_order = hits.groupby(env_col)["n"].sum().sort_values(ascending=False).index.tolist()

    # -------- Aggregate POSSIBLE (ReDU–hits) per (lat, lon, env)
    poss = None
    if df_redu is not None and not df_redu.empty:
        rl = df_redu["LatitudeandLongitude"].astype(str).str.extract(
            r"^\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*$"
        )
        rlat = pd.to_numeric(rl[0], errors="coerce")
        rlon = pd.to_numeric(rl[1], errors="coerce")
        rvalid = rlat.between(-90, 90) & rlon.between(-180, 180)
        r2 = df_redu.loc[rvalid & df_redu[env_col].notna() & (df_redu[env_col] != "missing value")].copy()
        if not r2.empty:
            r2["lat"] = rlat.loc[r2.index].values
            r2["lon"] = rlon.loc[r2.index].values
            poss = (
                r2.groupby(["lat", "lon", env_col], as_index=False)
                  .agg(n=("mri", "count"))
                  .sort_values("n", ascending=False)
                  .reset_index(drop=True)
            )
            # Ensure same category order includes any extra envs
            extra_envs = [e for e in poss[env_col].unique().tolist() if e not in cat_order]
            cat_order = cat_order + extra_envs

    # -------- Shared color map for env categories
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe + px.colors.qualitative.Set3
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cat_order)}

    # -------- Base fig = HITS
    center = {"lat": float(hits["lat"].mean()), "lon": float(hits["lon"].mean())}
    if engine == "geo":
        fig = px.scatter_geo(
            hits, lat="lat", lon="lon",
            size="n", size_max=18,
            color=env_col, category_orders={env_col: cat_order},
            color_discrete_map=color_map,
            projection=projection,
            hover_name=None,
            custom_data=[env_col, "n", "mri_examples"],
        )
    else:
        fig = px.scatter_mapbox(
            hits, lat="lat", lon="lon",
            size="n", size_max=20,
            color=env_col, category_orders={env_col: cat_order},
            color_discrete_map=color_map,
            mapbox_style=map_style, center=center, zoom=1,
            hover_name=None,
            custom_data=[env_col, "n", "mri_examples"],
        )

    # Hover for hits
    if hover_mri == "none":
        htmpl_hits = "<b>%{customdata[0]}</b><br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
    elif hover_mri == "count":
        htmpl_hits = "<b>%{customdata[0]}</b><br>Hits: %{customdata[1]}<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
    else:  # "examples"
        htmpl_hits = "<b>%{customdata[0]}</b><br>Hits: %{customdata[1]}<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<br>MRIs: %{customdata[2]}<extra></extra>"

    fig.update_traces(marker=dict(opacity=0.9), hovertemplate=htmpl_hits, selector=dict(mode="markers"))

    # -------- Add POSSIBLE points as unfilled circles, colored like hits, size by count
    if poss is not None and not poss.empty:
        if engine == "geo":
            # Scattergeo
            fig.add_trace(go.Scattergeo(
                lat=poss["lat"], lon=poss["lon"],
                mode="markers",
                marker=dict(
                    size=(poss["n"] / poss["n"].max() * 20).clip(lower=6),  
                    symbol="circle-open",
                    line=dict(width=2, color=[color_map[e] for e in poss[env_col]]),
                    opacity=0.85,
                ),
                text=None,
                hovertemplate="<b>%{customdata[0]}</b><br>Possible MRIs: %{customdata[1]}<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
                customdata=poss[[env_col, "n"]].to_numpy(),
                name="Possible (unhit)",
                showlegend=True,
            ))
        else:
            # Mapbox: possible hits = faint grey circles (behind hits)
            max_n = float(poss["n"].max())
            alpha = 0.5  

            # One combined trace for all possible hits (grey)
            sizes = (poss["n"] / max_n * 22).clip(lower=6)

            fig.add_trace(go.Scattermapbox(
                lat=poss["lat"], lon=poss["lon"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color="grey",     
                    opacity=alpha,
                    symbol="circle",
                ),
                customdata=poss[[env_col, "n"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"      
                    "<br>Possible MRIs: %{customdata[1]}"
                    "<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
                ),
                name="Possible (unhit)",   
                legendgroup="possible-",
                showlegend=True,
            ))

    # Ensure hits are on top: put any legendgroup starting with "possible-" FIRST,
    # so hits (added earlier by px) render LAST and sit above.
    if len(fig.data):
        possible_idxs = [i for i, tr in enumerate(fig.data)
                        if getattr(tr, "legendgroup", "") and str(tr.legendgroup).startswith("possible-")]
        if possible_idxs:
            keep_traces = [tr for i, tr in enumerate(fig.data) if i not in possible_idxs]
            possible_traces = [fig.data[i] for i in possible_idxs]
            fig.data = tuple(possible_traces + keep_traces)


    # -------- Layout & decorations
    fig.update_layout(
        title=f"Hit Locations by {env_col} (total hits={int(hits['n'].sum())}, points={len(hits)})",
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="closest",
        legend_title_text=env_col,
    )

    if engine == "geo":
        fig.update_geos(
            showland=True, landcolor="#f6f6f6",
            showocean=True, oceancolor="#eef6ff",
            showcountries=show_borders, countrycolor="rgba(80,80,80,0.5)",
            showcoastlines=show_coastlines, coastlinecolor="rgba(80,80,80,0.5)",
            lataxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
            lonaxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
        )
    else:
        if admin_geojson is not None:
            if isinstance(admin_geojson, str):
                with open(admin_geojson, "r", encoding="utf-8") as f:
                    admin_geojson = json.load(f)
            fig.update_layout(
                mapbox_layers=[
                    dict(
                        sourcetype="geojson",
                        source=admin_geojson,
                        type="line",
                        color=admin_line_color,
                        line=dict(width=admin_line_width),
                        opacity=admin_opacity,
                        below="traces",
                    )
                ]
            )

    return fig, hits

