#!/usr/bin/env python3
"""
structuremasst_batch_headless.py

Batch runner matching the *current* Streamlit app workflow.

Supports two input modes (mutually exclusive):

  --input-csv   CSV with name + query columns (SMILES / SMARTS / USI / class_label)
  --input-mgf   MGF file whose spectra are queried directly against FASST by peaks

CSV columns:
  required:
    - name
    - query   (preferred)  OR  smiles (will be renamed to query)

  optional:
    - type              (usi|smiles|smarts|class_label)  (if missing, inferred)
    - searchtype         (usi|exact|substructure|tanimoto|class_label) (if missing, inferred)
    - tanimoto_threshold (only used if searchtype==tanimoto)
    - formula            (used for substructure; default "any")
    - allowed_elements   (used for substructure; default "any")

MGF notes:
  - Each spectrum is queried against FASST by peaks (no USI needed).
  - NAME= field is used as the group name; SCANS= value is used as fallback.
  - Library lookup is skipped (no SMILES / InChIKey available).
  - SpectraResolver / Dashboard linkouts are omitted (no USI).
  - Mode is always "fasst" for MGF input.

Per input row/spectrum, creates one output folder and writes:
  - input.txt
  - library_all_spectra.tsv  (empty for MGF input)
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
import subprocess
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
from bin.workflow_stepwise import retrieve_raw_data_matches, retrieve_raw_data_matches_from_peaks
from bin.mgf_utils import parse_mgf_file

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


def parse_mgf(mgf_path: Path) -> list:
    """Parse an MGF file from a path. Delegates to bin.mgf_utils.parse_mgf_file."""
    return parse_mgf_file(mgf_path)


# ---------------------------------------------------------------------------
# LifeMASST support
# ---------------------------------------------------------------------------

_LIFEMASST_CODE = PKG_PATH / "external" / "LifeMASST" / "code"
_LIFEMASST_NF   = PKG_PATH / "external" / "LifeMASST" / "nf_workflow.nf"
_LIFEMASST_DATA = PKG_PATH / "external" / "LifeMASST" / "data"

# Tree key → metadata for nextflow call
LIFEMASST_TREES: dict = {
    "OTL": {
        "rel_path": "trees/OTL_prepared/labelled_supertree_subset_prepped",
        "tax_id": "OpenTreeOfLifeTaxonomyID",
        "tax_id_prefix": "ott",
        "label": "Open Tree of Life (subset)",
        "default_match_level": "NCBIFamily",
    },
    "bird_timetree": {
        "rel_path": "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint",
        "tax_id": "Native_tree_label",
        "tax_id_prefix": "",
        "label": "Bird Tree (time tree)",
        "default_match_level": "NCBISpecies",
    },
    "bird_molecular": {
        "rel_path": "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa",
        "tax_id": "Native_tree_label",
        "tax_id_prefix": "",
        "label": "Bird Tree (molecular tree)",
        "default_match_level": "NCBISpecies",
    },
    "mammal": {
        "rel_path": "trees/mammalian_prepared/Foley2022_Concatenation_HRA_neutral_241_10miss_rooted",
        "tax_id": "Native_tree_label",
        "tax_id_prefix": "",
        "label": "Mammal Tree (molecular tree)",
        "default_match_level": "NCBISpecies",
    },
    "fish_timetree": {
        "rel_path": "trees/fish_prepared/fish_treepl_12k",
        "tax_id": "Native_tree_label",
        "tax_id_prefix": "",
        "label": "Fish Tree (time tree)",
        "default_match_level": "NCBISpecies",
    },
    "fish_molecular": {
        "rel_path": "trees/fish_prepared/fish_raxml_12k",
        "tax_id": "Native_tree_label",
        "tax_id_prefix": "",
        "label": "Fish Tree (molecular tree)",
        "default_match_level": "NCBISpecies",
    },
    "plant": {
        "rel_path": "trees/plants_prepared/1kp_astral_alltaxa_FAA",
        "tax_id": "Native_tree_label",
        "tax_id_prefix": "",
        "label": "Plant Tree (molecular tree)",
        "default_match_level": "NCBISpecies",
    },
}


def _setup_lifemasst_files(
    query_table: pd.DataFrame,
    out_dir: Path,
    redu_tables: dict,
) -> tuple:
    """Prepare LifeMASST input files from batch results.

    Creates:
      out_dir/structuremasst_input.tsv   – query table with numeric id column
      out_dir/lifemasst_input_summary.tsv – organism-level summary per molecule

    Returns (in_path_str, out_path_str, mol_status) where mol_status maps
    molecule name → {"status": str, "n_organisms": int, "details": str}.
    out_path_str is None if no summaries were generated.
    """
    # Lazy import – avoids loading streamlit or rdkit if LifeMASST is not used
    if str(_LIFEMASST_CODE) not in sys.path:
        sys.path.insert(0, str(_LIFEMASST_CODE))
    import summarize_by_molecule  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write query table with numeric id column (mirrors app behaviour)
    input_file = query_table.copy()
    name_enum = {
        str(n).strip(): i
        for i, n in enumerate(input_file["name"].astype(str).str.strip().tolist(), start=1)
    }
    input_file["id"] = input_file["name"].astype(str).str.strip().map(name_enum)
    # get_sparql_sub_structures.py requires a 'type' column (filters on type=='smiles');
    # for MGF/peak queries there is no SMILES, so mark them as 'mgf_spectrum' to skip Wikidata lookup.
    if "type" not in input_file.columns:
        input_file["type"] = "mgf_spectrum"
    in_path = str(out_dir / "structuremasst_input.tsv")
    input_file.to_csv(in_path, sep="\t", index=False)

    # Build per-molecule organism summaries
    collect_dfs = []
    mol_status: dict = {}
    for key, df in redu_tables.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            mol_status[key] = {"status": "no_organism_summary", "n_organisms": 0,
                               "details": "redu table was empty"}
            continue
        if "query_name" not in df.columns:
            mol_status[key] = {"status": "no_organism_summary", "n_organisms": 0,
                               "details": "query_name column missing from redu table"}
            continue
        uniq = df["query_name"].dropna().astype(str).str.strip().unique()
        if len(uniq) != 1:
            mol_status[key] = {"status": "no_organism_summary", "n_organisms": 0,
                               "details": f"ambiguous query_name ({len(uniq)} distinct values)"}
            continue
        try:
            summary = summarize_by_molecule.main(df, "ncbi", uniq[0])
            collect_dfs.append(summary)
            cosine_col = f"masstCosine_{key}"
            n_org = int(summary[cosine_col].notna().sum()) if cosine_col in summary.columns else len(summary)
            mol_status[key] = {"status": "summary_generated", "n_organisms": n_org, "details": ""}
        except Exception as exc:
            print(f"  [LifeMASST] Warning: summarize_by_molecule failed for '{key}': {exc}")
            mol_status[key] = {"status": "no_organism_summary", "n_organisms": 0,
                               "details": f"summarize_by_molecule error: {exc}"}

    if not collect_dfs:
        return in_path, None, mol_status

    merged = collect_dfs[0]
    first_col = merged.columns[0]
    for df in collect_dfs[1:]:
        merged = pd.merge(merged, df, on=first_col, how="outer")

    for col in merged.columns:
        if col.startswith("masstCounts_"):
            merged[col] = merged[col].astype("Int64")
    if "OpenTreeOfLifeTaxonomyID" in merged.columns:
        merged["OpenTreeOfLifeTaxonomyID"] = (
            merged["OpenTreeOfLifeTaxonomyID"].astype(str).str.replace("ott", "", regex=False)
        )
        merged["OpenTreeOfLifeTaxonomyID"] = merged["OpenTreeOfLifeTaxonomyID"].astype("Int64")
    if "NCBI_ID" in merged.columns:
        merged["NCBI_ID"] = merged["NCBI_ID"].astype(str).str.replace(".0", "", regex=False)
        merged["NCBI_ID"] = merged["NCBI_ID"].astype("Int64")

    out_path = str(out_dir / "lifemasst_input_summary.tsv")
    merged.to_csv(out_path, sep="\t", index=False)
    return in_path, out_path, mol_status


def _write_lifemasst_report(mol_report: dict, path: Path) -> None:
    """Write per-molecule LifeMASST report TSV."""
    rows = []
    for name, info in mol_report.items():
        rows.append({
            "molecule": name,
            "status": info.get("status", ""),
            "n_masst_hits": info.get("n_masst_hits", 0),
            "n_organisms_in_summary": info.get("n_organisms_in_summary", 0),
            "n_tree_nodes_with_data": info.get("n_tree_nodes_with_data", 0),
            "details": info.get("details", ""),
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"  [LifeMASST] Molecule report written: {path}")


def _run_lifemasst_all(
    lifemasst_items: list,
    tree_key: str,
    match_level: Optional[str],
    outroot: Path,
    work_dir: Path,
    conda_cache_dir: Path,
    all_molecule_names: Optional[list] = None,
) -> None:
    """Group lifemasst_items by lifemasst_group and run nextflow for each group."""
    from collections import defaultdict as _dd2

    if not lifemasst_items:
        print("[LifeMASST] No items collected; skipping.")
        return

    tree_info = LIFEMASST_TREES[tree_key]
    tree_path = str(_LIFEMASST_DATA / (tree_info["rel_path"] + ".nwk"))
    feature_path = str(_LIFEMASST_DATA / (tree_info["rel_path"] + ".tsv"))
    effective_match_level = match_level or tree_info["default_match_level"]

    if not Path(tree_path).exists():
        print(f"[LifeMASST] WARNING: tree file not found: {tree_path}")

    by_group: dict = _dd2(list)
    for item in lifemasst_items:
        by_group[item["lifemasst_group"]].append(item)

    print(f"\n[LifeMASST] Tree: {tree_info['label']}  |  Match level: {effective_match_level}")
    print(f"[LifeMASST] Running for {len(by_group)} group(s)...")

    for group_name, items in by_group.items():
        print(f"\n  Group '{group_name}' ({len(items)} molecule(s))")

        # Build query table (unique names)
        seen: set = set()
        query_rows = []
        for item in items:
            if item["name"] not in seen:
                query_rows.append({"name": item["name"], "query": item["query_original"]})
                seen.add(item["name"])
        query_table = pd.DataFrame(query_rows)

        redu_tables = {
            item["name"]: item["redu_df"]
            for item in items
            if item["redu_df"] is not None and isinstance(item["redu_df"], pd.DataFrame) and not item["redu_df"].empty
        }

        # Initialise per-molecule report: names with no MASST hits
        names_in_group = {item["name"] for item in items}
        mol_report: dict = {}
        if all_molecule_names:
            for n in all_molecule_names:
                if n not in names_in_group:
                    mol_report[n] = {
                        "n_masst_hits": 0, "n_organisms_in_summary": 0,
                        "n_tree_nodes_with_data": 0,
                        "status": "no_masst_results",
                        "details": "No FASST matches found for this spectrum",
                    }
        for item in items:
            n_hits = len(item["redu_df"]) if isinstance(item["redu_df"], pd.DataFrame) else 0
            mol_report[item["name"]] = {
                "n_masst_hits": n_hits, "n_organisms_in_summary": 0,
                "n_tree_nodes_with_data": 0,
                "status": "no_organism_summary",
                "details": "",
            }

        lm_dir = outroot / f"lifemasst_{_safe_name(group_name)}"
        in_path, out_path, mol_status = _setup_lifemasst_files(query_table, lm_dir, redu_tables)

        # Propagate summary-generation status into report
        for name, info in mol_status.items():
            if name in mol_report:
                mol_report[name]["status"] = info["status"]
                mol_report[name]["n_organisms_in_summary"] = info["n_organisms"]
                mol_report[name]["details"] = info["details"]

        if out_path is None:
            print(f"  [LifeMASST] No valid organism summaries; skipping group '{group_name}'.")
            _write_lifemasst_report(mol_report, lm_dir / "lifemasst_molecule_report.tsv")
            continue

        # Write a nextflow override config to fix paths that are hardcoded for Docker (/app/…)
        nf_override = lm_dir / "nf_override.config"
        nf_override.write_text(
            f"workDir = '{work_dir}'\n"
            f"conda {{\n"
            f"    enabled = true\n"
            f"    cacheDir = '{conda_cache_dir}'\n"
            f"}}\n",
            encoding="utf-8",
        )

        cmd = [
            "nextflow", "run", str(_LIFEMASST_NF),
            "-c", str(nf_override),
            "-w", str(work_dir),
            "--input_molecules", in_path,
            "--structureMASST_input_file", out_path,
            "--tree_path", tree_path,
            "--tree_features", feature_path,
            "--tax_id", tree_info["tax_id"],
            "--tax_id_prefix", tree_info["tax_id_prefix"],
            "--masst_matchLevel", effective_match_level,
            "--wikidata_matchLevel", effective_match_level,
            "--output_folder", str(lm_dir),
        ]

        print(f"  [LifeMASST] Calling nextflow...")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"  [LifeMASST] WARNING: nextflow exited with code {proc.returncode} for group '{group_name}'.")
        else:
            print(f"  [LifeMASST] Done. Results in: {lm_dir}")

        # Update report with tree overlap from merged_metadata.tsv
        merged_meta_path = lm_dir / "merged_metadata.tsv"
        if merged_meta_path.exists():
            try:
                merged_meta = pd.read_csv(merged_meta_path, sep="\t")
                for name in list(mol_report.keys()):
                    col = f"masstCosine_{name}"
                    if col in merged_meta.columns:
                        n_tree = int(merged_meta[col].notna().sum())
                        mol_report[name]["n_tree_nodes_with_data"] = n_tree
                        if mol_report[name]["status"] == "summary_generated":
                            if n_tree > 0:
                                mol_report[name]["status"] = "tree_match_found"
                                mol_report[name]["details"] = f"Matched {n_tree} tree node(s)"
                            else:
                                mol_report[name]["status"] = "no_tree_overlap"
                                mol_report[name]["details"] = (
                                    "Organism summary generated but no matching taxa found in the tree"
                                )
                    elif mol_report[name]["status"] == "summary_generated":
                        mol_report[name]["status"] = "no_tree_overlap"
                        mol_report[name]["details"] = (
                            "Organism summary generated but molecule column absent from merged_metadata"
                        )
            except Exception as exc:
                print(f"  [LifeMASST] Warning: could not parse merged_metadata.tsv for report: {exc}")

        _write_lifemasst_report(mol_report, lm_dir / "lifemasst_molecule_report.tsv")


def _run_mgf_group(
    g_idx: int,
    name: str,
    group_spectra: list,
    args,
    outroot: Path,
    sqlite_path,
    api_endpoint,
    timeout,
) -> dict:
    """Process one name-group of MGF spectra; returns a summary_row dict."""
    group_hash = _short_hash("||".join(s["spectrum_id"] for s in group_spectra))
    folder = outroot / f"{g_idx:04d}_{_safe_name(name)}_{group_hash}"
    folder.mkdir(parents=True, exist_ok=True)

    try:
        # input.txt
        lines = [
            f"name\t{name}",
            f"input_type\tmgf",
            f"n_spectra\t{len(group_spectra)}",
            f"mode_used\tfasst",
        ]
        for j, spec in enumerate(group_spectra):
            lines.append(f"spectrum{j}_id\t{spec['spectrum_id']}")
            lines.append(f"spectrum{j}_precursor_mz\t{spec['precursor_mz']}")
            lines.append(f"spectrum{j}_n_peaks\t{len(spec['peaks'])}")
        (folder / "input.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # No library lookup for MGF spectra
        pd.DataFrame().to_csv(folder / "library_all_spectra.tsv", sep="\t", index=False)

        modification_mass = args.mod_mass
        if args.mod_search and args.mod_formula and Formula is not None:
            try:
                modification_mass = Formula.formula_from_str(args.mod_formula).get_monoisotopic_mass()
            except Exception:
                pass

        _, redu_df = retrieve_raw_data_matches_from_peaks(
            group_spectra,
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
            output_folder=str(folder),
        )

        if redu_df is None or not isinstance(redu_df, pd.DataFrame):
            redu_df = pd.DataFrame()

        redu_df.to_csv(folder / "raw_redu.tsv", sep="\t", index=False)

        n_raw = int(len(redu_df))
        if not redu_df.empty and "mri_id_int" in redu_df.columns:
            n_mri = int(redu_df["mri_id_int"].dropna().nunique())
        elif not redu_df.empty and "mri" in redu_df.columns:
            n_mri = int(redu_df["mri"].dropna().nunique())
        else:
            n_mri = 0

        if not redu_df.empty:
            redu_df["query_name"] = name
            sankey_path = folder / f"rawdata_sankey.{args.sankey_ext.lower()}"
            _export_sankey_if_possible(
                redu_df, sankey_path,
                args.sankey_col1, args.sankey_col2, args.sankey_col3, args.sankey_col4,
            )

        if n_raw == 0:
            (folder / "note.txt").write_text("No raw data matches found for this spectrum group.\n", encoding="utf-8")

        return {
            "idx": g_idx, "name": name,
            "n_inputs": len(group_spectra),
            "n_library_spectra": 0,
            "n_raw_matches": n_raw,
            "n_unique_mri": n_mri,
            "folder": str(folder),
            "mode_used": "fasst",
        }

    except Exception as e:
        (folder / "error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
        return {
            "idx": g_idx, "name": name,
            "n_inputs": len(group_spectra),
            "n_library_spectra": None, "n_raw_matches": None, "n_unique_mri": None,
            "folder": str(folder), "mode_used": "fasst",
        }


def main():
    p = argparse.ArgumentParser(description="Run StructureMASST batch headless (current app workflow).")
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input-csv", help="CSV with name + query (or smiles).")
    inp.add_argument("--input-mgf", help="MGF file; each spectrum is searched by peaks via FASST.")
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

    # LifeMASST
    p.add_argument(
        "--lifemasst",
        action="store_true",
        help="Run LifeMASST (Nextflow) after all batch items are processed.",
    )
    p.add_argument(
        "--lifemasst-tree",
        default="OTL",
        choices=list(LIFEMASST_TREES.keys()),
        metavar="TREE",
        help=(
            "Phylogenetic tree to use for LifeMASST.  "
            "Choices: " + ", ".join(f"{k} ({v['label']})" for k, v in LIFEMASST_TREES.items()) + ".  "
            "Default: OTL (Open Tree of Life subset)."
        ),
    )
    p.add_argument(
        "--lifemasst-match-level",
        default=None,
        help=(
            "Taxonomic level for matching StructureMASST results to the tree.  "
            "E.g. NCBIFamily, NCBIGenus, NCBISpecies, NCBIOrder, NCBIClass, NCBIPhylum, "
            "OpenTreeOfLifeTaxonomyID.  "
            "Defaults to the tree's recommended level when not set."
        ),
    )
    p.add_argument(
        "--lifemasst-group",
        default="default",
        help=(
            "LifeMASST group name used for MGF input (all spectra share this group) "
            "or as the fallback group for CSV rows without a 'lifemasst_group' column.  "
            "Default: 'default'."
        ),
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Nextflow work directory (overrides the hardcoded Docker path in nextflow.config).  "
            "Default: <repo_root>/work/work"
        ),
    )
    p.add_argument(
        "--conda-cache-dir",
        default=None,
        help=(
            "Conda environment cache directory for Nextflow (overrides the Docker path).  "
            "Default: <repo_root>/work/work_env"
        ),
    )

    args = p.parse_args()

    outroot = Path(args.outdir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    # LifeMASST path resolution: default to repo-root/work/{work,work_env}
    # which is where Docker maps them (always outside the container).
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (PKG_PATH / "work" / "work")
    conda_cache_dir = Path(args.conda_cache_dir).resolve() if args.conda_cache_dir else (PKG_PATH / "work" / "work_env")

    # Accumulated items for post-batch LifeMASST run
    lifemasst_items: list = []

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

    # ------------------------------------------------------------------ MGF path
    if args.input_mgf:
        mgf_path = Path(args.input_mgf).resolve()
        if not mgf_path.exists():
            raise FileNotFoundError(f"MGF file not found: {mgf_path}")

        spectra = parse_mgf(mgf_path)
        if not spectra:
            raise ValueError(f"No valid spectra found in {mgf_path}")

        print(f"Parsed {len(spectra)} spectra from {mgf_path.name}")

        # Group by name (same semantics as CSV grouping by name)
        from collections import defaultdict as _dd
        mgf_groups: dict = _dd(list)
        for spec in spectra:
            mgf_groups[spec["name"]].append(spec)

        summary_rows = []
        for g_idx, (name, group_spectra) in enumerate(mgf_groups.items()):
            row = _run_mgf_group(g_idx, name, group_spectra, args, outroot, sqlite_path, api_endpoint, timeout)
            summary_rows.append(row)
            print(f"[{g_idx}] {name}: {row['n_raw_matches']} raw matches across {row['n_unique_mri']} MRIs")

            # Collect for LifeMASST
            if args.lifemasst and row.get("n_raw_matches") and row["n_raw_matches"] > 0:
                redu_path = Path(row["folder"]) / "raw_redu.tsv"
                if redu_path.exists():
                    try:
                        redu_df_lm = pd.read_csv(redu_path, sep="\t")
                        # query_name is written after to_csv in _run_mgf_group, so inject it here
                        if "query_name" not in redu_df_lm.columns:
                            redu_df_lm["query_name"] = name
                        lifemasst_items.append({
                            "name": name,
                            "query_original": "",  # no SMILES for peak-based spectra
                            "redu_df": redu_df_lm,
                            "lifemasst_group": args.lifemasst_group,
                        })
                    except Exception:
                        pass

        pd.DataFrame(summary_rows).to_csv(outroot / "batch_summary.tsv", sep="\t", index=False)
        print(f"Batch complete. Wrote per-spectrum folders under: {outroot}")
        print(f"Summary: {outroot / 'batch_summary.tsv'}")

        if args.lifemasst:
            _run_lifemasst_all(
                lifemasst_items=lifemasst_items,
                tree_key=args.lifemasst_tree,
                match_level=args.lifemasst_match_level,
                outroot=outroot,
                work_dir=work_dir,
                conda_cache_dir=conda_cache_dir,
                all_molecule_names=list(mgf_groups.keys()),
            )

        print(f"\nDone.")
        return

    # ------------------------------------------------------------------ CSV path

    df = pd.read_csv(args.input_csv, encoding="cp1252")

    # Harmonize input schema (app accepts query/name; also smiles/name)
    if "query" not in df.columns and "smiles" in df.columns:
        df = df.rename(columns={"smiles": "query"})

    if not {"query", "name"}.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: name + (query or smiles).")

    df = df.dropna(subset=["query", "name"]).copy()
    df["query"] = df["query"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    # replace all spaces and special chars in name with underscores
    df["name"] = df["name"].str.replace(" ", "_").str.replace("[^a-zA-Z0-9_]", "", regex=True)

    # optional columns
    for col in ["type", "searchtype", "tanimoto_threshold", "formula", "allowed_elements", "lifemasst_group"]:
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

            # Collect for LifeMASST
            if args.lifemasst:
                lg_vals = gdf["lifemasst_group"].dropna().astype(str).str.strip()
                lg = lg_vals.iloc[0] if not lg_vals.empty and lg_vals.iloc[0] else args.lifemasst_group
                original_query = query_meta[0]["query_raw"] if query_meta else ""
                lifemasst_items.append({
                    "name": name,
                    "query_original": original_query,
                    "redu_df": redu_df.copy() if isinstance(redu_df, pd.DataFrame) and not redu_df.empty else pd.DataFrame(),
                    "lifemasst_group": lg,
                })

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
    print(f"Batch complete. Wrote folders under: {outroot}")
    print(f"Summary: {outroot / 'batch_summary.tsv'}")

    # ------------------------------------------------------------------ LifeMASST
    if args.lifemasst:
        _run_lifemasst_all(
            lifemasst_items=lifemasst_items,
            tree_key=args.lifemasst_tree,
            match_level=args.lifemasst_match_level,
            outroot=outroot,
            work_dir=work_dir,
            conda_cache_dir=conda_cache_dir,
            all_molecule_names=list(df["name"].unique()),
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()