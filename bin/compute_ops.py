"""
Operations executed by the Python compute worker (see bin/compute_worker.py).

These are the calls the Streamlit pages and tasks.py used to make: the RDKit,
masst_records, FASST, LifeMASST and GeoMASST work stays in the existing bin/
and external/ modules; this file only adapts arguments and results.

Kept in an importable module (not __main__) so that the worker processes,
which are forked from a single-threaded forkserver, can resolve them.
"""

import base64
import importlib.util
import io
import json
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for extra in (
    ROOT,
    os.path.join(ROOT, "external", "GNPSDataPackage"),
    os.path.join(ROOT, "external", "GeoMASST"),
    os.path.join(ROOT, "external", "LifeMASST", "code"),
):
    if extra not in sys.path:
        sys.path.insert(0 if extra == ROOT else 1, extra)


def _load_config():
    spec = importlib.util.spec_from_file_location("config", os.path.join(ROOT, "config.py"))
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


CONFIG = _load_config()


def _fetcher():
    from bin.run_masstRecords_queries import _get_fetcher
    return _get_fetcher(CONFIG.PATH_TO_SQLITE, CONFIG.MASSTRECORDS_ENDPOINT, CONFIG.MASSTRECORDS_TIMEOUT)


# ---------------------------------------------------------------------------
# Table helpers: tables travel as {"columns": [...], "rows": [[...], ...]}
# ---------------------------------------------------------------------------

def _df_to_table(df):
    if df is None or not hasattr(df, "to_json"):
        return {"columns": [], "rows": []}
    split = json.loads(df.to_json(orient="split", index=False, date_format="iso"))
    return {"columns": split.get("columns", []), "rows": split.get("data", [])}


def _table_to_df(table):
    import pandas as pd
    df = pd.DataFrame(table.get("rows", []), columns=table.get("columns", []))
    # Mirror pd.read_json's dtype inference used by the old Celery round trip
    return df.infer_objects()


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

def op_ping(args):
    return "pong"


def _standardize(smiles):
    """Tautomerize and neutralize a SMILES string (same as the Streamlit app)."""
    from bin.match_smiles import neutralize_atoms, tautomerize_smiles
    try:
        smi = tautomerize_smiles(smiles)
    except Exception as e:
        print(f"Tautomerization failed: {e}")
        smi = smiles
    try:
        smi = neutralize_atoms(smi)
    except Exception as e:
        print(f"Neutralization failed: {e}")
    return smi


def op_detect(args):
    from bin.match_smiles import detect_smiles_or_smarts
    return [detect_smiles_or_smarts(s) for s in args["items"]]


def op_prepare_queries(args):
    """
    For each {"query", "type"?}: infer the type when missing (USI, SMILES, SMARTS)
    and harmonize SMILES (tautomerize + neutralize). Returns
    [{"query", "type", "original_smiles"}].
    """
    from bin.match_smiles import detect_smiles_or_smarts
    out = []
    for q in args["queries"]:
        query = str(q["query"]).strip()
        typ = q.get("type")
        if not typ:
            typ = "usi" if query.startswith("mzspec:") else detect_smiles_or_smarts(query)
        original = ""
        if typ == "smiles":
            original = query
            query = _standardize(query)
        out.append({"query": query, "type": typ, "original_smiles": original})
    return out


def op_render(args):
    """Render SMILES to PNG (base64). Returns null for unparsable input."""
    from rdkit import Chem
    from rdkit.Chem import Draw
    size = int(args.get("size", 300))
    mol = Chem.MolFromSmiles(args["smiles"])
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def op_formulas(args):
    """Molecular formula per SMILES ("" when it does not parse)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    out = []
    for smi in args["items"]:
        mol = Chem.MolFromSmiles(smi) if smi else None
        out.append(rdMolDescriptors.CalcMolFormula(mol) if mol else "")
    return out


def op_formula_mass(args):
    from formula_validation.Formula import Formula
    return Formula.formula_from_str(args["formula"]).get_monoisotopic_mass()


def op_parse_mgf(args):
    from bin.mgf_utils import parse_mgf_bytes
    spectra = parse_mgf_bytes(base64.b64decode(args["data"]))
    return [
        {"spectrum_id": s["spectrum_id"], "name": s["name"],
         "precursor_mz": s["precursor_mz"], "peaks": s["peaks"]}
        for s in spectra
    ]


def op_read_tsv(args):
    import pandas as pd
    return _df_to_table(pd.read_csv(args["path"], sep="\t"))


def op_prep_status(args):
    """Reference-data status for LifeMASST / GeoMASST (bin/dependency_prep.py)."""
    from bin import dependency_prep
    tool = args["tool"]
    report = dependency_prep.status(tool)
    return {
        "label": dependency_prep.LABELS[tool],
        "ready": bool(report["ready"]),
        "available": bool(report.get("available", True)),
        "missing_shipped": bool(report.get("missing_shipped")),
        "note": report.get("note") or "",
        "download_mb": dependency_prep.download_estimate(tool),
        "entries": [
            {
                "Data": e.get("label", e["key"]),
                "Status": "present" if e.get("present") else "missing",
                "Delivered by": "git" if e.get("shipped") else e.get("source", ""),
                "Size (MB)": e.get("size_mb", 0.0),
            }
            for e in report["entries"]
        ],
    }


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

def op_molecule_classes(args):
    from bin.shared_data import get_molecule_classes_cached
    df = get_molecule_classes_cached(_fetcher())
    return df["class_label"].astype(str).tolist() if df is not None else []


def op_library_table(args):
    from bin.run_masstRecords_queries import get_library_table
    df = get_library_table(
        smiles=args["smiles"],
        searchtype=args["searchtype"],
        tanimoto_threshold=args.get("tanimoto_threshold"),
        allowed_formula=args.get("allowed_formula"),
        allowed_elements=args.get("allowed_elements"),
        sqlite_path=CONFIG.PATH_TO_SQLITE,
        api_endpoint=CONFIG.MASSTRECORDS_ENDPOINT,
        timeout=CONFIG.MASSTRECORDS_TIMEOUT,
    )
    return _df_to_table(df)


def op_masst_and_redu(args):
    from bin.run_masstRecords_queries import get_masst_and_redu_tables
    _, redu_df = get_masst_and_redu_tables(
        _table_to_df(args["table"]),
        cosine_threshold=float(args["cosine_threshold"]),
        matching_peaks=int(args["matching_peaks"]),
        min_annotation_rank=int(args.get("min_annotation_rank", 0)),
        sqlite_path=CONFIG.PATH_TO_SQLITE,
        api_endpoint=CONFIG.MASSTRECORDS_ENDPOINT,
        timeout=CONFIG.MASSTRECORDS_TIMEOUT,
        chunk_size=int(args.get("chunk_size", 200)),
    )
    return _df_to_table(redu_df)


def op_fasst_health(args):
    from bin.api_health import test_fasst_api_search_nonblocking
    return test_fasst_api_search_nonblocking()


def _fasst_kwargs(args):
    return dict(
        database=args.get("database", "metabolomicspanrepo_index"),
        precursor_mz_tol=float(args["precursor_mz_tol"]),
        fragment_mz_tol=float(args["fragment_mz_tol"]),
        min_cos=float(args["min_cos"]),
        matching_peaks=int(args["matching_peaks"]),
        analog=bool(args["analog"]),
        modimass=args.get("modimass"),
        elimination=bool(args.get("elimination", False)),
        addition=bool(args.get("addition", False)),
        modification_condition=args.get("modification_condition"),
        sqlite_path=CONFIG.PATH_TO_SQLITE,
        api_endpoint=CONFIG.MASSTRECORDS_ENDPOINT,
        timeout=CONFIG.MASSTRECORDS_TIMEOUT,
        output_folder=args.get("output_folder"),
    )


def op_raw_data_matches(args):
    from bin.workflow_stepwise import retrieve_raw_data_matches
    _, redu_df = retrieve_raw_data_matches(_table_to_df(args["table"]), **_fasst_kwargs(args))
    return _df_to_table(redu_df)


def op_raw_data_matches_from_peaks(args):
    from bin.workflow_stepwise import retrieve_raw_data_matches_from_peaks
    _, redu_df = retrieve_raw_data_matches_from_peaks(args["spectra"], **_fasst_kwargs(args))
    return _df_to_table(redu_df)


# ---------------------------------------------------------------------------
# Downstream tools
# ---------------------------------------------------------------------------

def op_lifemasst_setup(args):
    """prepare_lifemasst_input() + setup_lifemasst_files(), as the Setup LifeMASST button did."""
    from bin.lifemasst_input import prepare_lifemasst_input, setup_lifemasst_files
    query_table = prepare_lifemasst_input(_table_to_df(args["query_table"]))
    redu_tables = {k: _table_to_df(t) for k, t in args["redu_tables"].items()}
    in_path, out_path = setup_lifemasst_files(
        input_file=query_table,
        structureMASST_op_folder=args["session_dir"],
        redu_tables=redu_tables,
    )
    return {"in_path": in_path, "out_path": out_path}


def op_dedup_molecules(args):
    """Deduplicate the LifeMASST molecule file by name (before Nextflow)."""
    import pandas as pd
    df = pd.read_csv(args["in_path"], sep="\t")
    df.drop_duplicates(subset=["name"]).to_csv(args["out_path"], sep="\t", index=False)
    return args["out_path"]


def _date_ranges_to_start(df):
    """
    ReDU has some collection dates given as ISO ranges ('2020-09-18/2020-09-22').
    pandas parses those as timezone-aware, and a single one among naive dates makes
    GeoMASST's date handling fail ("can't compare offset-naive and offset-aware
    datetimes"). Keep the start date of a range.
    """
    col = "SampleCollectionDateandTime"
    if df is None or col not in getattr(df, "columns", []):
        return df
    df = df.copy()
    s = df[col]
    fixed = s.astype(str).str.replace(r"^\s*(\d{4}-\d{2}-\d{2})(?:[T ][^/]*)?/\S+\s*$", r"\1", regex=True)
    df[col] = fixed.where(s.notna(), s)
    return df


def op_geomasst_html(args):
    """GeoMASST map document for the given hits (external/GeoMASST)."""
    from bin.plotting import load_environmental_context
    from geomasst import build_geomasst_map_html
    df_hits = _date_ranges_to_start(_table_to_df(args["table"]))
    html = build_geomasst_map_html(
        df_hits=df_hits,
        df_background=_date_ranges_to_start(load_environmental_context(df_hits)),
        compound_name=args.get("compound_name") or "",
        max_molecule_maps=int(args.get("max_molecule_maps", 5)),
        structures=args.get("structures") or {},
    )
    with open(args["out_path"], "w", encoding="utf-8") as fh:
        fh.write(html)
    return args["out_path"]


LIGHT_OPS = {
    "ping": op_ping,
    "detect": op_detect,
    "prepare_queries": op_prepare_queries,
    "render": op_render,
    "formulas": op_formulas,
    "formula_mass": op_formula_mass,
    "parse_mgf": op_parse_mgf,
    "read_tsv": op_read_tsv,
    "prep_status": op_prep_status,
}
HEAVY_OPS = {
    "molecule_classes": op_molecule_classes,
    "library_table": op_library_table,
    "masst_and_redu": op_masst_and_redu,
    "fasst_health": op_fasst_health,
    "raw_data_matches": op_raw_data_matches,
    "raw_data_matches_from_peaks": op_raw_data_matches_from_peaks,
    "lifemasst_setup": op_lifemasst_setup,
    "dedup_molecules": op_dedup_molecules,
    "geomasst_html": op_geomasst_html,
}


def run_op(op, args):
    """Executed inside a pool process. Exceptions become error responses."""
    fn = LIGHT_OPS.get(op) or HEAVY_OPS.get(op)
    if fn is None:
        return False, f"unknown op: {op}"
    try:
        return True, fn(args)
    except Exception as e:
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"
    finally:
        import gc
        gc.collect()


# ---------------------------------------------------------------------------
# Preload (run once in the forkserver, see bin/compute_preload.py)
# ---------------------------------------------------------------------------

def preload_redu_table():
    """Same preload (and column list) as the Celery worker_init in tasks.py."""
    from bin.shared_data import get_redu_table_cached

    fetch = _fetcher()
    redu_columns_list = fetch("SELECT name FROM pragma_table_info('redu_table')")["name"].tolist()
    columns_to_exclude = [
        "filename", "TermsofPosition", "ComorbidityListDOIDIndex",
        "ENVOBroadScale", "ENVOLocalScale", "ENVOMediumScale", "qiita_sample_name",
        "UniqueSubjectID", "UBERONOntologyIndex", "DOIDOntologyIndex", "ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex", "ENVOLocalScaleIndex", "ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex", "classification", "MS2spectra_count",
        "InternalStandardsUsed", "HumanPopulationDensity",
    ]
    redu_columns_list = [c for c in redu_columns_list if c not in columns_to_exclude]
    print("[INIT] Preloading ReDU table...")
    get_redu_table_cached(fetch, redu_columns_list, CONFIG.PATH_TO_SQLITE)
    print("[INIT] Preload complete.")


def preload_molecule_classes():
    from bin.shared_data import get_molecule_classes_cached
    get_molecule_classes_cached(_fetcher())
