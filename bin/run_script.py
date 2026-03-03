#!/usr/bin/env python3
"""
structuremasst_headless.py

Headless (non-Streamlit) workflow for StructureMASST:
  1) get_library_spectra_display(): returns library spectra tables in the same shape/order the UI shows
  2) get_raw_data_results(): returns raw-data (ReDU/MASST) matches via FASSTrecords or FASST

Run from repo root so imports like `bin.*` work.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union
from typing import Iterable, List
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd


HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches
from bin.match_smiles import detect_smiles_or_smarts, neutralize_atoms, tautomerize_smiles
from bin.linkouts import build_dashboard_eic_url, build_spectraresolver_link

try:
    from formula_validation.Formula import Formula
except Exception:
    Formula = None  # optional


# -------------------------
# Helpers
# -------------------------

def _load_config(config_path: Union[str, Path]):
    config_path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("config", str(config_path))
    cfg = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(cfg)
    return cfg

def _tag_stage_values(work: pd.DataFrame, cols: list[str], tag_fmt: str = " [{stage}]") -> pd.DataFrame:
    """
    Make node labels unique across stages by appending a stage tag to every value.
    Example: "blood" -> "blood [2]" for stage 2.
    """
    out = work.copy()
    for stage, c in enumerate(cols, start=1):
        out[c] = out[c].astype("string").map(lambda v: f"{v}{tag_fmt.format(stage=stage)}")
    return out

def export_sankey_plot(
    df: pd.DataFrame,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    col3: Optional[str] = None,
    col4: Optional[str] = None,
    outpath: Union[str, Path] = "sankey.html",
    title: Optional[str] = None,
    stage_labels: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> go.Figure:
    """
    Build + export a Sankey diagram from df using 4 categorical columns.

    Defaults match the Streamlit interface presets:
      col1="ATTRIBUTE_DatasetAccession"
      col2="UBERONBodyPartName"
      col3="NCBIDivision"
      col4="NCBITaxonomy"

    Exports to .svg/.png/.html depending on outpath suffix.
    NOTE: SVG/PNG export requires `kaleido` installed.
    """
    if df is None or df.empty:
        raise ValueError("df is empty")

    # ---- defaults (same as interface preset: datasets_bodypart_division_taxa) ----
    defaults = ["ATTRIBUTE_DatasetAccession", "UBERONBodyPartName", "NCBIDivision", "NCBITaxonomy"]
    cols = [col1, col2, col3, col4]
    cols = [c if c else defaults[i] for i, c in enumerate(cols)]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    if len(set(cols)) != 4:
        raise ValueError(f"Columns must be 4 distinct values, got: {cols}")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=int(max_rows), random_state=0).copy()

    # ---- prep like UI: fill NaN and string 'nan' with stage-specific Unknown_* ----
    work = df[cols].copy()

    fill_values = {cols[0]: "Unknown_1", cols[1]: "Unknown_2", cols[2]: "Unknown_3", cols[3]: "Unknown_4"}
    for c in cols:
        work[c] = work[c].astype("string")
    work = work.fillna(fill_values)
    for c, fv in fill_values.items():
        work.loc[work[c].str.lower() == "nan", c] = fv
        work.loc[work[c].str.strip() == "", c] = fv

    # ---- make node labels unique across stages ----
    work = _tag_stage_values(work, cols)

    # ---- labels ----
    labels: List[str] = []
    for c in cols:
        labels += work[c].unique().tolist()
    labels = list(dict.fromkeys(labels))  # preserve order, remove dups

    # ---- match UI color behavior: color links by first-stage category ----
    first_stage_cats = work[cols[0]].unique().tolist()
    palette = px.colors.qualitative.Safe[: max(5, len(first_stage_cats))]  # UI uses Safe[:5]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(first_stage_cats)}

    source, target, value, link_colors = [], [], [], []
    for i in range(len(cols) - 1):
        for cat in first_stage_cats:
            sub = work[work[cols[0]] == cat]
            grp = sub.groupby([cols[i], cols[i + 1]]).size().reset_index(name="count")
            for _, row in grp.iterrows():
                source.append(labels.index(row[cols[i]]))
                target.append(labels.index(row[cols[i + 1]]))
                value.append(int(row["count"]))
                # same rgba conversion trick as UI (alpha 0.3)
                lc = color_map[cat].replace("rgb", "rgba").replace(")", ", 0.3)")
                link_colors.append(lc)

    node_colors = ["#F2F2F2"] * len(labels)

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            textfont=dict(family="Arial, sans-serif", size=12, color="black"),
            node=dict(
                label=labels,
                color=node_colors,
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
            ),
            link=dict(source=source, target=target, value=value, color=link_colors),
        )
    )

    if stage_labels is None:
        stage_labels = ["Column 1", "Column 2", "Column 3", "Column 4"]

    fig.update_layout(
        title=title,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=60, t=120 if stage_labels else 60, b=20),
    )

    # stage labels above diagram (UI-style)
    if stage_labels:
        n = len(stage_labels) - 1
        for i, lab in enumerate(stage_labels):
            x = i / n
            xanchor = "left" if i == 0 else ("right" if i == n else "center")
            fig.add_annotation(
                x=x,
                y=1.02,
                xref="paper",
                yref="paper",
                text=lab,
                showarrow=False,
                font=dict(size=14, color="black"),
                xanchor=xanchor,
            )

    # ---- export ----
    outpath = Path(outpath)
    suffix = outpath.suffix.lower()

    if suffix in {".html"}:
        fig.write_html(str(outpath), include_plotlyjs="cdn")
    elif suffix in {".svg", ".png", ".pdf", ".jpeg", ".jpg", ".webp"}:
        # requires kaleido
        fig.write_image(str(outpath))
    else:
        raise ValueError(f"Unsupported output extension: {suffix} (use .svg/.png/.pdf/.html)")

    return fig


def _tautomerize_neutralize_smiles(smiles: str) -> str:
    smi = smiles
    try:
        smi = tautomerize_smiles(smi)
    except Exception:
        pass
    try:
        smi = neutralize_atoms(smi)
    except Exception:
        pass
    return smi


def _special_count(s: str) -> int:
    return len(re.findall(r"[^A-Za-z0-9]", s))


def _pick_best_compound_name(names: pd.Series) -> str:
    if names is None:
        return ""
    s = names.dropna().astype(str)
    if s.empty:
        return ""
    vc = s.value_counts()
    top = vc.iloc[0]
    cands = vc[vc == top].index.tolist()
    # tie-break like UI: closest to len≈10, then fewest special chars
    return min(cands, key=lambda x: (abs(len(x) - 10), _special_count(x)))


def _usi_viewer_link(query_spectrum_id: str) -> str:
    x = str(query_spectrum_id)
    if x.startswith("CCMSLIB"):
        usi1 = f"mzspec%3AGNPS%3AGNPS-LIBRARY%3Aaccession%3A{x}"
    else:
        usi1 = f"mzspec%3AMASSBANK%3A%3Aaccession%3A{x}"

    return (
        "http://metabolomics-usi.gnps2.org/dashinterface"
        f"?usi1={usi1}"
        "&width=10.0&height=6.0"
        "&mz_min=None&mz_max=None&max_intensity=125"
        "&annotate_precision=4&annotation_rotation=90"
        "&cosine=standard&fragment_mz_tolerance=0.02"
        "&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D"
    )


def _dereplicate_by_falcon_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors UI dereplication:
      - count similar spectra per representative
      - pick the entry closest to representative spectrum
      - set spectrum_id_int = representative_spectrum_int
    """
    required = {"spectrum_id_int", "representative_spectrum_int"}
    if not required.issubset(df.columns) or df.empty:
        return df.copy()

    out = df.copy()
    out["spectrum_id_int"] = out["spectrum_id_int"].astype("int64")
    out["representative_spectrum_int"] = out["representative_spectrum_int"].astype("int64")

    out["similar_library_spectra"] = (
        out.groupby("representative_spectrum_int")["spectrum_id_int"]
           .transform("size")
           .astype("int64")
    )
    out["spectrum_difference"] = out["spectrum_id_int"] - out["representative_spectrum_int"]
    out = out.sort_values(by=["representative_spectrum_int", "spectrum_difference"])
    out = out.groupby("representative_spectrum_int", as_index=False).first()

    out["spectrum_id_int"] = out["representative_spectrum_int"]
    return out


def _compute_modimass(modification_formula: Optional[str], modification_mass: Optional[float]) -> Optional[float]:
    """
    UI logic, but safer:
      - if formula given and formula_validation is available: use monoisotopic mass
      - otherwise fall back to provided modification_mass
      - require mass > 0
    """
    mass = None
    if modification_formula and Formula is not None:
        try:
            f = Formula.formula_from_str(modification_formula)
            mass = float(f.get_monoisotopic_mass())
        except Exception:
            mass = None

    if mass is None and modification_mass is not None:
        try:
            mass = float(modification_mass)
        except Exception:
            mass = None

    if mass is None or mass <= 0:
        return None
    return mass


def _add_match_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds UI-like link columns when possible:
      - lib_usi
      - best_spectral_match (SpectraResolver)
      - Check LC peak (EIC dashboard link)
    Safe no-ops if columns are missing.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if "query_spectrum_id" in out.columns:
        out["lib_usi"] = out["query_spectrum_id"].astype(str).apply(
            lambda x: (
                f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if x.startswith("CCMSLIB")
                else f"mzspec:MASSBANK::accession:{x}"
            )
        )

    if "USI" in out.columns and "lib_usi" in out.columns:
        out["best_spectral_match"] = out.apply(
            lambda row: build_spectraresolver_link(row["USI"], row["lib_usi"]),
            axis=1
        )

    # UI fills "Check LC peak" if missing/empty
    if "USI" in out.columns and "Precursor_MZ" in out.columns:
        if "Check LC peak" not in out.columns:
            out["Check LC peak"] = pd.NA
        out["Check LC peak"] = out["Check LC peak"].astype(object)

        mask = out["Check LC peak"].isna() | (out["Check LC peak"].astype(str).str.strip() == "")
        if mask.any():
            out.loc[mask, "Check LC peak"] = out.loc[mask].apply(
                lambda row: build_dashboard_eic_url(
                    usi=row["USI"],
                    xic_mz=row["Precursor_MZ"],
                    xic_tolerance=0.05
                ),
                axis=1
            )

    return out


# -------------------------
# MAIN FUNCTION 1
# -------------------------

def get_library_spectra_display(
    query: str,
    searchtype: str = "exact",                 # exact | substructure | tanimoto
    tanimoto_threshold: Optional[float] = None,
    config_path: Union[str, Path] = "config.py",
    normalize_smiles: bool = True,
) -> Dict[str, object]:
    """
    Returns the library spectra “as displayed in the interface” for ONE input query.

    Output dict:
      - query (str): effective query used (after normalization)
      - query_type (str): smiles | smarts | invalid
      - overview (pd.DataFrame): one row per inchikey_first_block (+ n_spectra)
      - spectra_by_inchikey (dict[str, pd.DataFrame]): UI-shaped tables per inchikey
      - all_spectra (pd.DataFrame): raw df from get_library_table()
    """
    cfg = _load_config(config_path)

    q = (query or "").strip()
    q_type = detect_smiles_or_smarts(q)

    if q_type == "smarts":
        # UI forces substructure for SMARTS
        searchtype = "substructure"
    elif q_type == "smiles" and normalize_smiles:
        q = _tautomerize_neutralize_smiles(q)
    elif q_type not in {"smiles", "smarts"}:
        return {
            "query": q,
            "query_type": "invalid",
            "overview": pd.DataFrame(),
            "spectra_by_inchikey": {},
            "all_spectra": pd.DataFrame(),
        }

    tan_th = float(tanimoto_threshold) if (searchtype == "tanimoto" and tanimoto_threshold is not None) else None

    df = get_library_table(
        smiles=q,
        searchtype=searchtype,
        tanimoto_threshold=tan_th,
        sqlite_path=cfg.PATH_TO_SQLITE,
        api_endpoint=cfg.MASSTRECORDS_ENDPOINT,
        timeout=cfg.MASSTRECORDS_TIMEOUT,
    )

    if df is None or df.empty:
        return {
            "query": q,
            "query_type": q_type,
            "overview": pd.DataFrame(),
            "spectra_by_inchikey": {},
            "all_spectra": pd.DataFrame(),
        }

    spectra_by_ik: Dict[str, pd.DataFrame] = {}
    overview_rows = []

    if "inchikey_first_block" not in df.columns:
        # nothing sensible to group on; still return the raw
        return {
            "query": q,
            "query_type": q_type,
            "overview": pd.DataFrame(),
            "spectra_by_inchikey": {},
            "all_spectra": df,
        }

    for ik in df["inchikey_first_block"].dropna().astype(str).unique():
        sub = df[df["inchikey_first_block"].astype(str) == ik].copy()

        if "query_spectrum_id" in sub.columns:
            sub["spectrum_link"] = sub["query_spectrum_id"].astype(str).apply(_usi_viewer_link)

        # UI column order inside InChIKey tabs
        col_order = [c for c in ["spectrum_link", "Compound_Name", "Precursor_MZ"] if c in sub.columns]
        sub = sub[col_order + [c for c in sub.columns if c not in col_order]].reset_index(drop=True)
        spectra_by_ik[ik] = sub

        best_name = _pick_best_compound_name(sub["Compound_Name"]) if "Compound_Name" in sub.columns else ""
        first_smi = ""
        if "Smiles" in sub.columns:
            ss = sub["Smiles"].dropna().astype(str)
            first_smi = ss.iloc[0] if not ss.empty else ""

        overview_rows.append(
            {
                "Compound_Name": best_name,
                "inchikey_first_block": ik,
                "Smiles": first_smi,
                "n_spectra": int(len(sub)),
            }
        )

    overview = pd.DataFrame(overview_rows).sort_values("n_spectra", ascending=False).reset_index(drop=True)

    return {
        "query": q,
        "query_type": q_type,
        "overview": overview,
        "spectra_by_inchikey": spectra_by_ik,
        "all_spectra": df,
    }


# -------------------------
# MAIN FUNCTION 2
# -------------------------

def get_raw_data_results(
    library_spectra: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    mode: str = "fasstrecords",                # fasstrecords | fasst
    config_path: Union[str, Path] = "config.py",
    # shared thresholds
    min_cos: float = 0.70,
    min_peaks: int = 5,
    dereplicate: bool = True,
    add_links: bool = True,
    # fasst-only
    database: str = "metabolomicspanrepo_index_nightly",
    precursor_tol: float = 0.02,
    fragment_tol: float = 0.02,
    modification_search: bool = False,
    modification_formula: Optional[str] = None,
    modification_mass: Optional[float] = None,
    elimination: bool = True,
    addition: bool = False,
    modification_condition: Optional[str] = None,
    output_folder: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns raw-data matches for ONE molecule query.

    Input:
      - library_spectra: either the raw df from get_library_spectra_display()["all_spectra"]
                        OR the dict spectra_by_inchikey

    Output dict:
      - "masst": pd.DataFrame   (may be empty depending on backend)
      - "redu":  pd.DataFrame   (raw data matches; typically what you want)
    """
    cfg = _load_config(config_path)

    if isinstance(library_spectra, dict):
        frames = [df for df in library_spectra.values() if isinstance(df, pd.DataFrame)]
        df_in = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df_in = library_spectra.copy() if isinstance(library_spectra, pd.DataFrame) else pd.DataFrame()

    if df_in.empty:
        return {"masst": pd.DataFrame(), "redu": pd.DataFrame()}

    df_q = _dereplicate_by_falcon_cluster(df_in) if dereplicate else df_in.copy()

    mode_norm = mode.strip().lower()
    if mode_norm not in {"fasstrecords", "fasst"}:
        raise ValueError("mode must be 'fasstrecords' or 'fasst'")

    if mode_norm == "fasstrecords":
        masst_df, redu_df = get_masst_and_redu_tables(
            df_q,
            cosine_threshold=float(min_cos),
            matching_peaks=int(min_peaks),
            sqlite_path=cfg.PATH_TO_SQLITE,
            api_endpoint=cfg.MASSTRECORDS_ENDPOINT,
            timeout=cfg.MASSTRECORDS_TIMEOUT,
            chunk_size=200,
        )

        # UI expects these columns to exist; if backend returns something else, just pass through
        if add_links:
            redu_df = _add_match_links(redu_df)

        return {"masst": masst_df, "redu": redu_df}

    # mode == "fasst"
    modimass = _compute_modimass(modification_formula, modification_mass) if modification_search else None

    masst_df, redu_df = retrieve_raw_data_matches(
        df_q,
        database=database,
        precursor_mz_tol=float(precursor_tol),
        fragment_mz_tol=float(fragment_tol),
        min_cos=float(min_cos),
        matching_peaks=int(min_peaks),
        analog=bool(modification_search),
        modimass=modimass,
        elimination=bool(elimination) if modification_search else False,
        addition=bool(addition) if modification_search else False,
        modification_condition=modification_condition if modification_search else None,
        sqlite_path=cfg.PATH_TO_SQLITE,
        api_endpoint=cfg.MASSTRECORDS_ENDPOINT,
        timeout=cfg.MASSTRECORDS_TIMEOUT,
        output_folder=output_folder,
    )

    if add_links:
        redu_df = _add_match_links(redu_df)

    return {"masst": masst_df, "redu": redu_df}


# -------------------------
# Optional CLI (so it "runs" as a workflow)
# -------------------------

def _cli():
    p = argparse.ArgumentParser(description="Headless StructureMASST workflow for ONE molecule query")
    p.add_argument("--config", default="config.py", help="Path to config.py")
    p.add_argument("--query", required=True, help="SMILES or SMARTS")
    p.add_argument("--searchtype", default="exact", choices=["exact", "substructure", "tanimoto"])
    p.add_argument("--tanimoto-threshold", type=float, default=None)
    p.add_argument("--mode", default="fasstrecords", choices=["fasstrecords", "fasst"])
    p.add_argument("--min-cos", type=float, default=0.70)
    p.add_argument("--min-peaks", type=int, default=5)

    # fasst-only knobs
    p.add_argument("--database", default="metabolomicspanrepo_index_nightly")
    p.add_argument("--precursor-tol", type=float, default=0.02)
    p.add_argument("--fragment-tol", type=float, default=0.02)
    p.add_argument("--mod-search", action="store_true")
    p.add_argument("--mod-formula", default=None)
    p.add_argument("--mod-mass", type=float, default=None)
    p.add_argument("--elimination", action="store_true")
    p.add_argument("--addition", action="store_true")
    p.add_argument("--mod-condition", default=None)

    p.add_argument("--outdir", default="headless_output", help="Where to write TSV outputs")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lib = get_library_spectra_display(
        query=args.query,
        searchtype=args.searchtype,
        tanimoto_threshold=args.tanimoto_threshold,
        config_path=args.config,
    )

    overview: pd.DataFrame = lib["overview"]
    all_spectra: pd.DataFrame = lib["all_spectra"]
    spectra_by_ik: Dict[str, pd.DataFrame] = lib["spectra_by_inchikey"]

    overview.to_csv(outdir / "library_overview.tsv", sep="\t", index=False)
    all_spectra.to_csv(outdir / "library_all_spectra.tsv", sep="\t", index=False)

    # also dump per-inchikey tables (like UI tabs)
    for ik, df in spectra_by_ik.items():
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ik)
        df.to_csv(outdir / f"library_{safe}.tsv", sep="\t", index=False)

    if all_spectra.empty:
        print("No library spectra found.")
        return

    raw = get_raw_data_results(
        library_spectra=all_spectra,
        mode=args.mode,
        config_path=args.config,
        min_cos=args.min_cos,
        min_peaks=args.min_peaks,
        database=args.database,
        precursor_tol=args.precursor_tol,
        fragment_tol=args.fragment_tol,
        modification_search=args.mod_search,
        modification_formula=args.mod_formula,
        modification_mass=args.mod_mass,
        elimination=args.elimination,
        addition=args.addition,
        modification_condition=args.mod_condition,
    )

    raw["masst"].to_csv(outdir / "raw_masst.tsv", sep="\t", index=False)
    raw["redu"].to_csv(outdir / "raw_redu.tsv", sep="\t", index=False)

    # after raw["redu"].to_csv(...)
    df_redu = raw["redu"]
    if not df_redu.empty:
        export_sankey_plot(
            df_redu,
            # defaults are the same as the interface, so you can omit the cols
            outpath=outdir / "rawdata_sankey_default.html",
            stage_labels=["Dataset", "Bodypart", "Division", "Taxonomy"],
            title="Raw data distribution (default view)",
        )

    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    _cli()
