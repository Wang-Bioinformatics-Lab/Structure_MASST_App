#!/usr/bin/env python3
"""
structuremasst_batch_headless.py

Batch runner matching the *current* Streamlit app workflow:

Input CSV:
  required:
    - name
    - query   (preferred)  OR  smiles (will be renamed to query)

  optional:
    - type              (usi|smiles|smarts|class_label)  (if missing, inferred)
    - searchtype         (usi|exact|substructure|tanimoto|class_label) (if missing, inferred)
    - tanimoto_threshold (only used if searchtype==tanimoto)
    - formula            (used for substructure; default "any")
    - allowed_elements   (used for substructure; default "any")

Per input row, creates one output folder and writes:
  - input.txt
  - library_all_spectra.tsv (+ library_overview.tsv and per-InChIKey tables if available)
  - raw_redu.tsv (+ raw_masst.tsv if available)
  - rawdata_sankey.<ext> (optional; default html)

Notes:
  - SMILES queries are tautomerized + neutralized (same as the app)
  - USI queries force mode=fasst (FASSTrecords is not possible for USI-only)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np



# --- make repo imports work when running from anywhere ---
HERE = Path(__file__).resolve().parent
PKG_PATH = (HERE / "..").resolve()
if str(PKG_PATH) not in sys.path:
    sys.path.insert(0, str(PKG_PATH))

from bin.match_smiles import detect_smiles_or_smarts, neutralize_atoms, tautomerize_smiles
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches

# plotting (optional sankey export)
try:
    from bin.plotting import raw_data_sankey
except Exception:
    raw_data_sankey = None  # type: ignore

# linkouts (optional columns in FASST mode)
try:
    from bin.linkouts import build_dashboard_eic_url, build_spectraresolver_link
except Exception:
    build_dashboard_eic_url = None  # type: ignore
    build_spectraresolver_link = None  # type: ignore

# formula handling (optional)
try:
    from formula_validation.Formula import Formula
except Exception:
    Formula = None  # type: ignore


def _load_py_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _safe_name(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:maxlen] if len(s) > maxlen else s) or "unnamed"


def _short_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:10]


def _norm_type(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"usi", "smiles", "smarts", "class_label"}:
        return v
    return None


def _norm_searchtype(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"usi", "exact", "substructure", "tanimoto", "class_label"}:
        return v
    return None


def tautomerize_neutralize_smiles(smiles: str) -> str:
    smi = smiles
    try:
        smi = tautomerize_smiles(smi)
    except Exception:
        smi = smiles
    try:
        smi = neutralize_atoms(smi)
    except Exception:
        pass
    return smi


def _infer_type_from_query(q: str) -> str:
    q = (q or "").strip()
    if q.startswith("mzspec:"):
        return "usi"
    return detect_smiles_or_smarts(q)


def _infer_searchtype(typ: str) -> str:
    if typ == "usi":
        return "usi"
    if typ == "smarts":
        return "substructure"
    if typ == "class_label":
        return "class_label"
    return "exact"


def _ensure_any(x: Optional[str]) -> str:
    if x is None:
        return "any"
    x = str(x).strip()
    return x if x else "any"


def _dereplicate_library_spectra(df_struct: pd.DataFrame) -> pd.DataFrame:
    """
    Matches app behavior:
      - only if spectrum_id_int + representative_spectrum_int exist
      - choose one representative per representative_spectrum_int
      - annotate similar_library_spectra, spectrum_difference
      - set spectrum_id_int := representative_spectrum_int
    """
    required = {"spectrum_id_int", "representative_spectrum_int"}
    if not required.issubset(df_struct.columns):
        out = df_struct.copy()
        if "similar_library_spectra" not in out.columns:
            out["similar_library_spectra"] = 1
        if "unique_spectra_in_mri" not in out.columns:
            out["unique_spectra_in_mri"] = 1
        return out

    out = df_struct.copy()
    out["spectrum_id_int"] = out["spectrum_id_int"].astype("int64")
    out["representative_spectrum_int"] = out["representative_spectrum_int"].astype("int64")

    out["similar_library_spectra"] = (
        out.groupby("representative_spectrum_int")["spectrum_id_int"]
           .transform("size")
           .astype("int64")
    )

    out["spectrum_difference"] = out["spectrum_id_int"] - out["representative_spectrum_int"]
    out = out.sort_values(by=["representative_spectrum_int", "spectrum_difference"])
    out = out.groupby("representative_spectrum_int").first().reset_index()
    out["spectrum_id_int"] = out["representative_spectrum_int"]
    return out


def _make_library_overview(df_lib: pd.DataFrame) -> pd.DataFrame:
    if df_lib.empty:
        return pd.DataFrame(
            columns=["inchikey_first_block", "n_spectra", "Compound_Name", "Smiles"]
        )

    cols = df_lib.columns
    ik_col = "inchikey_first_block" if "inchikey_first_block" in cols else None
    if ik_col is None:
        # fallback overview: just one bucket
        return pd.DataFrame([{
            "inchikey_first_block": "",
            "n_spectra": int(len(df_lib)),
            "Compound_Name": (df_lib["Compound_Name"].dropna().astype(str).iloc[0] if "Compound_Name" in cols and df_lib["Compound_Name"].notna().any() else ""),
            "Smiles": (df_lib["Smiles"].dropna().astype(str).iloc[0] if "Smiles" in cols and df_lib["Smiles"].notna().any() else ""),
        }])

    def _first_nonnull(series: pd.Series) -> str:
        s = series.dropna().astype(str)
        return s.iloc[0] if len(s) else ""

    g = df_lib.groupby(ik_col, dropna=False)
    overview = pd.DataFrame({
        "inchikey_first_block": g.size().index.astype(str),
        "n_spectra": g.size().astype(int).to_numpy(),
        "Compound_Name": g["Compound_Name"].apply(_first_nonnull).to_numpy() if "Compound_Name" in cols else "",
        "Smiles": g["Smiles"].apply(_first_nonnull).to_numpy() if "Smiles" in cols else "",
    })
    overview = overview.sort_values(["n_spectra", "inchikey_first_block"], ascending=[False, True])
    return overview.reset_index(drop=True)


def _export_sankey_if_possible(
    df_redu: pd.DataFrame,
    outpath: Path,
    col1: str,
    col2: str,
    col3: str,
    col4: str,
):
    if raw_data_sankey is None:
        return
    if df_redu is None or df_redu.empty:
        return
    for c in (col1, col2, col3, col4):
        if c not in df_redu.columns:
            return

    fig = raw_data_sankey(df_redu, col1, col2, col3, col4)

    ext = outpath.suffix.lower().lstrip(".")
    try:
        if ext == "html":
            fig.write_html(str(outpath), include_plotlyjs="cdn")
        elif ext in {"png", "svg", "pdf"}:
            # requires kaleido installed (you already prefer it)
            fig.write_image(str(outpath))
        else:
            # unknown ext -> html
            fig.write_html(str(outpath.with_suffix(".html")), include_plotlyjs="cdn")
    except Exception:
        # don't fail the whole run if sankey export fails
        pass


def main():
    p = argparse.ArgumentParser(description="Run StructureMASST batch headless (current app workflow).")
    p.add_argument("--input-csv", required=True, help="CSV with name + query (or smiles).")
    p.add_argument("--outdir", default="batch_output", help="Output root directory")
    p.add_argument("--config", default="config.py", help="Path to config.py")

    p.add_argument("--mode", default="fasstrecords", choices=["fasstrecords", "fasst"])
    p.add_argument("--min-cos", type=float, default=0.70)
    p.add_argument("--min-peaks", type=int, default=5)
    p.add_argument("--min-rank", type=int, default=0, help="FASSTrecords: minimum annotation rank (0 disables).")
    p.add_argument("--max-returned-rows", type=int, default=200, help="FASSTrecords: max returned rows (app uses 200).")

    # FASST-only knobs
    p.add_argument("--database", default="metabolomicspanrepo_index_nightly")
    p.add_argument("--precursor-tol", type=float, default=0.02)
    p.add_argument("--fragment-tol", type=float, default=0.02)
    p.add_argument("--mod-search", action="store_true")
    p.add_argument("--mod-formula", default=None)
    p.add_argument("--mod-mass", type=float, default=None)
    p.add_argument("--elimination", action="store_true")
    p.add_argument("--addition", action="store_true")
    p.add_argument("--mod-condition", default=None)

    # Defaults for missing per-row columns
    p.add_argument("--default-tanimoto", type=float, default=0.8, help="Used if searchtype==tanimoto and row has no threshold.")
    p.add_argument("--default-formula", default="any", help='Used if searchtype==substructure and row has no formula (default "any").')
    p.add_argument("--default-allowed-elements", default="any", help='Used if searchtype==substructure and row has no allowed_elements (default "any").')

    # Sankey export
    p.add_argument("--sankey-ext", default="html", choices=["html", "svg", "png", "pdf"])
    p.add_argument("--sankey-col1", default="ATTRIBUTE_DatasetAccession")
    p.add_argument("--sankey-col2", default="UBERONBodyPartName")
    p.add_argument("--sankey-col3", default="NCBIDivision")
    p.add_argument("--sankey-col4", default="NCBITaxonomy")
    args = p.parse_args()

    outroot = Path(args.outdir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    # load config.py (same as app)
    config_py = Path(args.config).resolve()
    if not config_py.exists():
        raise FileNotFoundError(f"Could not find config.py at: {config_py}")

    config = _load_py_module(config_py, "config")
    sqlite_path = getattr(config, "PATH_TO_SQLITE", None)
    api_endpoint = getattr(config, "MASSTRECORDS_ENDPOINT", None)
    timeout = getattr(config, "MASSTRECORDS_TIMEOUT", None)
    if not sqlite_path or api_endpoint is None or timeout is None:
        raise RuntimeError("config.py must define PATH_TO_SQLITE, MASSTRECORDS_ENDPOINT, MASSTRECORDS_TIMEOUT")

    df = pd.read_csv(args.input_csv)

    # Harmonize input schema (app accepts query/name; also smiles/name)
    if "query" not in df.columns and "smiles" in df.columns:
        df = df.rename(columns={"smiles": "query"})

    if not {"query", "name"}.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: name + (query or smiles).")

    df = df.dropna(subset=["query", "name"]).copy()
    df["query"] = df["query"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # optional columns
    for col in ["type", "searchtype", "tanimoto_threshold", "formula", "allowed_elements"]:
        if col not in df.columns:
            df[col] = None

    summary_rows = []

    # Group rows with the same name and treat them as one combined "library query"
    for g_idx, (name, gdf) in enumerate(df.groupby("name", sort=False), start=0):
        raw_queries = gdf["query"].astype(str).str.strip().tolist()
        group_hash = _short_hash("||".join(raw_queries))
        folder = outroot / f"{g_idx:04d}_{_safe_name(name)}_{group_hash}"
        folder.mkdir(parents=True, exist_ok=True)

        try:
            query_meta = []
            lib_parts = []
            has_any_usi = False

            # --- build one combined library table for this name ---
            for _, row in gdf.iterrows():
                raw_query = str(row["query"]).strip()

                # infer / normalize type + searchtype
                typ = _norm_type(row.get("type"))
                if typ is None:
                    typ = _infer_type_from_query(raw_query)

                st = _norm_searchtype(row.get("searchtype"))
                if st is None:
                    st = _infer_searchtype(typ)

                # canonicalize query (app behavior)
                query = raw_query
                if typ == "smiles":
                    query = tautomerize_neutralize_smiles(query)

                # per-row knobs
                tan_th = None
                if st == "tanimoto":
                    v = row.get("tanimoto_threshold")
                    try:
                        tan_th = float(v) if pd.notna(v) else float(args.default_tanimoto)
                    except Exception:
                        tan_th = float(args.default_tanimoto)

                formula = _ensure_any(row.get("formula")) if st == "substructure" else "any"
                allowed_elements = _ensure_any(row.get("allowed_elements")) if st == "substructure" else "any"
                if st == "substructure":
                    if formula == "any":
                        formula = _ensure_any(args.default_formula)
                    if allowed_elements == "any":
                        allowed_elements = _ensure_any(args.default_allowed_elements)

                # mode override for USI (matches app)
                effective_mode = args.mode
                if typ == "usi" and effective_mode == "fasstrecords":
                    effective_mode = "fasst"

                if typ == "usi" or st == "usi":
                    has_any_usi = True

                query_meta.append(
                    {
                        "query_raw": raw_query,
                        "query_used": query,
                        "type": typ,
                        "searchtype": st,
                        "tanimoto_threshold": tan_th,
                        "formula": formula,
                        "allowed_elements": allowed_elements,
                        "mode_used": effective_mode,
                    }
                )

                # LIBRARY STEP (per sub-query), then combine
                if typ == "usi" or st == "usi":
                    ik = hashlib.sha1(query.encode()).hexdigest()[:12]
                    df_lib_part = pd.DataFrame([{
                        "inchikey_first_block": ik,
                        "Compound_Name": name,
                        "Smiles": "",
                        "Precursor_MZ": np.nan,
                        "query_spectrum_id": query,
                        "USI": query,
                    }])
                else:
                    df_lib_part = get_library_table(
                        smiles=query,
                        searchtype=st,
                        tanimoto_threshold=tan_th,
                        allowed_formula=formula,
                        allowed_elements=allowed_elements,
                        sqlite_path=sqlite_path,
                        api_endpoint=api_endpoint,
                        timeout=timeout,
                    )
                    if df_lib_part is None:
                        df_lib_part = pd.DataFrame()
                    if not isinstance(df_lib_part, pd.DataFrame):
                        raise RuntimeError("get_library_table() did not return a pandas DataFrame")

                # keep a temp marker so we can run RAW STEP per mode if needed
                df_lib_part["__effective_mode"] = effective_mode
                lib_parts.append(df_lib_part)

            df_lib = pd.concat(lib_parts, ignore_index=True, sort=False) if lib_parts else pd.DataFrame()
            df_lib_clean = df_lib.drop(columns=["__effective_mode"], errors="ignore")

            # write group input.txt (all sub-queries)
            lines = [
                f"name\t{name}",
                f"n_queries\t{len(query_meta)}",
                f"mode_requested\t{args.mode}",
            ]
            for j, m in enumerate(query_meta):
                lines.extend([
                    f"query{j}_raw\t{m['query_raw']}",
                    f"query{j}_used\t{m['query_used']}",
                    f"query{j}_type\t{m['type']}",
                    f"query{j}_searchtype\t{m['searchtype']}",
                    f"query{j}_tanimoto_threshold\t{m['tanimoto_threshold']}",
                    f"query{j}_formula\t{m['formula']}",
                    f"query{j}_allowed_elements\t{m['allowed_elements']}",
                    f"query{j}_mode_used\t{m['mode_used']}",
                ])
            (folder / "input.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

            # write combined library outputs
            df_lib_clean.to_csv(folder / "library_all_spectra.tsv", sep="\t", index=False)

            if not df_lib_clean.empty and "inchikey_first_block" in df_lib_clean.columns:
                overview = _make_library_overview(df_lib_clean)
                overview.to_csv(folder / "library_overview.tsv", sep="\t", index=False)

                for ik_val in overview["inchikey_first_block"].astype(str).tolist():
                    sub = df_lib_clean[df_lib_clean["inchikey_first_block"].astype(str) == str(ik_val)].copy()
                    safe_ik = re.sub(r"[^A-Za-z0-9._-]+", "_", str(ik_val))
                    sub.to_csv(folder / f"library_{safe_ik}.tsv", sep="\t", index=False)

            n_lib = int(len(df_lib_clean))

            # if no library spectra and no USI fallback, stop here
            if n_lib == 0 and not has_any_usi:
                (folder / "note.txt").write_text("No library spectra found for this name-group.\n", encoding="utf-8")
                summary_rows.append(
                    {
                        "idx": g_idx,
                        "name": name,
                        "n_inputs": int(len(gdf)),
                        "n_library_spectra": 0,
                        "n_raw_matches": 0,
                        "n_unique_mri": 0,
                        "folder": str(folder),
                        "mode_used": "",
                    }
                )
                continue

            # map IK -> best Compound_Name for later merge
            ik_to_name = None
            if not df_lib_clean.empty and "inchikey_first_block" in df_lib_clean.columns:
                if "Compound_Name" in df_lib_clean.columns:
                    ik_to_name = (
                        df_lib_clean[["inchikey_first_block", "Compound_Name"]]
                        .dropna(subset=["inchikey_first_block"])
                        .drop_duplicates()
                    )

            # --- RAW DATA STEP (run once if all same mode; otherwise per-mode then concat) ---
            # preserve order of first appearance of each mode
            modes_in_group = []
            for m in query_meta:
                if m["mode_used"] not in modes_in_group:
                    modes_in_group.append(m["mode_used"])

            masst_parts = []
            redu_parts = []

            for mode_used in modes_in_group:
                df_lib_mode = df_lib[df_lib["__effective_mode"] == mode_used].drop(columns=["__effective_mode"], errors="ignore")
                df_for_name = _dereplicate_library_spectra(df_lib_mode)

                masst_df_part = pd.DataFrame()
                redu_df_part = pd.DataFrame()

                if mode_used == "fasstrecords":
                    masst_df_part, redu_df_part = get_masst_and_redu_tables(
                        df_for_name,
                        cosine_threshold=float(args.min_cos),
                        matching_peaks=int(args.min_peaks),
                        min_annotation_rank=int(args.min_rank),
                        sqlite_path=sqlite_path,
                        api_endpoint=api_endpoint,
                        timeout=timeout,
                        chunk_size=int(args.max_returned_rows),
                    )

                    if (not isinstance(redu_df_part, pd.DataFrame)) or ("Cosine" not in redu_df_part.columns) or ("Matching Peaks" not in redu_df_part.columns):
                        masst_df_part = pd.DataFrame()
                        redu_df_part = pd.DataFrame()

                else:  # FASST
                    modification_mass = args.mod_mass
                    if args.mod_search and args.mod_formula:
                        if Formula is not None:
                            try:
                                modification_mass = Formula.formula_from_str(args.mod_formula).get_monoisotopic_mass()
                            except Exception:
                                pass

                    # if we have multiple mode runs, avoid clobbering any mode-specific side outputs
                    out_folder_for_mode = str(folder) if len(modes_in_group) == 1 else str((folder / f"_mode_{mode_used}").resolve())
                    Path(out_folder_for_mode).mkdir(parents=True, exist_ok=True)

                    _, redu_df_part = retrieve_raw_data_matches(
                        df_for_name,
                        database=args.database,
                        precursor_mz_tol=float(args.precursor_tol),
                        fragment_mz_tol=float(args.fragment_tol),
                        min_cos=float(args.min_cos),
                        matching_peaks=int(args.min_peaks),
                        analog=bool(args.mod_search),
                        modimass=modification_mass,
                        elimination=bool(args.elimination),
                        addition=bool(args.addition),
                        modification_condition=args.mod_condition,
                        sqlite_path=sqlite_path,
                        api_endpoint=api_endpoint,
                        timeout=timeout,
                        output_folder=out_folder_for_mode,
                    )

                    if redu_df_part is None or not isinstance(redu_df_part, pd.DataFrame):
                        redu_df_part = pd.DataFrame()

                    # optional link columns (same as app; best-effort)
                    if not redu_df_part.empty:
                        if "query_spectrum_id" in redu_df_part.columns:
                            redu_df_part["lib_usi"] = redu_df_part["query_spectrum_id"].astype(str).apply(
                                lambda x: (
                                    x if x.startswith("mzspec:")
                                    else f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if x.startswith("CCMSLIB")
                                    else f"mzspec:MASSBANK::accession:{x}"
                                )
                            )

                            if build_spectraresolver_link is not None and "USI" in redu_df_part.columns:
                                redu_df_part["best_spectral_match"] = redu_df_part.apply(
                                    lambda r: build_spectraresolver_link(r["USI"], r["lib_usi"]),
                                    axis=1,
                                )

                        if build_dashboard_eic_url is not None and "USI" in redu_df_part.columns and "Precursor_MZ" in redu_df_part.columns:
                            if "Check LC peak" not in redu_df_part.columns:
                                redu_df_part["Check LC peak"] = np.nan
                            redu_df_part["Check LC peak"] = redu_df_part["Check LC peak"].astype(object)
                            mask = redu_df_part["Check LC peak"].isna() | (redu_df_part["Check LC peak"].astype(str).str.strip() == "")
                            try:
                                redu_df_part.loc[mask, "Check LC peak"] = redu_df_part.loc[mask].apply(
                                    lambda r: build_dashboard_eic_url(
                                        usi=r["USI"],
                                        xic_mz=r["Precursor_MZ"],
                                        xic_tolerance=0.05,
                                    ),
                                    axis=1,
                                )
                            except Exception:
                                pass

                if isinstance(masst_df_part, pd.DataFrame) and not masst_df_part.empty:
                    masst_parts.append(masst_df_part)
                if isinstance(redu_df_part, pd.DataFrame) and not redu_df_part.empty:
                    redu_parts.append(redu_df_part)

            masst_df = pd.concat(masst_parts, ignore_index=True, sort=False) if masst_parts else pd.DataFrame()
            redu_df = pd.concat(redu_parts, ignore_index=True, sort=False) if redu_parts else pd.DataFrame()

            # merge Compound_Name like app (only if useful)
            if isinstance(redu_df, pd.DataFrame) and not redu_df.empty:
                if ik_to_name is not None and "inchikey_first_block" in redu_df.columns:
                    try:
                        redu_df = redu_df.merge(ik_to_name, on="inchikey_first_block", how="left")
                    except Exception:
                        pass
                redu_df["query_name"] = name

            # write outputs
            masst_df.to_csv(folder / "raw_masst.tsv", sep="\t", index=False)
            redu_df.to_csv(folder / "raw_redu.tsv", sep="\t", index=False)

            n_raw = int(len(redu_df)) if isinstance(redu_df, pd.DataFrame) else 0
            if isinstance(redu_df, pd.DataFrame) and not redu_df.empty:
                if "mri_id_int" in redu_df.columns:
                    n_mri = int(redu_df["mri_id_int"].dropna().nunique())
                elif "mri" in redu_df.columns:
                    n_mri = int(redu_df["mri"].dropna().nunique())
                else:
                    n_mri = 0
            else:
                n_mri = 0

            # optional sankey export
            if isinstance(redu_df, pd.DataFrame) and not redu_df.empty:
                sankey_path = folder / f"rawdata_sankey.{args.sankey_ext.lower()}"
                _export_sankey_if_possible(
                    redu_df,
                    sankey_path,
                    args.sankey_col1,
                    args.sankey_col2,
                    args.sankey_col3,
                    args.sankey_col4,
                )

            summary_rows.append(
                {
                    "idx": g_idx,
                    "name": name,
                    "n_inputs": int(len(gdf)),
                    "n_library_spectra": n_lib,
                    "n_raw_matches": n_raw,
                    "n_unique_mri": n_mri,
                    "folder": str(folder),
                    "mode_used": ",".join(modes_in_group),
                }
            )

        except Exception as e:
            (folder / "error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
            summary_rows.append(
                {
                    "idx": g_idx,
                    "name": name,
                    "n_inputs": int(len(gdf)),
                    "n_library_spectra": None,
                    "n_raw_matches": None,
                    "n_unique_mri": None,
                    "folder": str(folder),
                    "mode_used": None,
                }
            )

    pd.DataFrame(summary_rows).to_csv(outroot / "batch_summary.tsv", sep="\t", index=False)
    print(f"Done. Wrote per-molecule folders under: {outroot}")
    print(f"Summary: {outroot / 'batch_summary.tsv'}")


if __name__ == "__main__":
    main()