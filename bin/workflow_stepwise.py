import pandas as pd 
from bin.match_smiles import fetch_and_match_smiles, detect_smiles_or_smarts
from bin.match_smiles import detect_smiles_or_smarts
from bin.run_fasst import query_fasst_usi
from bin.run_masstRecords_queries import _get_fetcher
import argparse
from collections import defaultdict, deque
import os
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from io import StringIO
import sqlite3
from urllib.parse import quote_plus
import sys
import time
import numpy as np

HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'GNPSDataPackage'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


from gnpsdata import fasst


FASST_API_SERVER_URL = "https://api.fasst.gnps2.org"

def _as_usi(x: str) -> str:
    """Return a full mzspec USI given either a full USI or a library accession."""
    x = str(x).strip()
    if x.startswith("mzspec:"):
        return x
    # fallback to your existing logic for CCMSLIB / MassBank accessions
    return make_library_usi(x)


def _as_lib_usi(x: str) -> str:
    """Return the 'library-side' USI for linkouts (works for full USIs too)."""
    x = str(x).strip()
    if x.startswith("mzspec:"):
        return x
    if x.startswith("CCMSLIB"):
        return f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}"
    return f"mzspec:MASSBANK::accession:{x}"


def make_library_usi(lib_id):
    if lib_id.startswith("CCMSLIB"):
        return "mzspec:GNPS:GNPS-LIBRARY:accession:{}".format(lib_id)
    else:
        return "mzspec:MASSBANK::accession:{}".format(lib_id)
   
def retrieve_raw_data_matches(
    library_subset: pd.DataFrame,
    analog: bool = False,
    elimination: bool = False,
    addition: bool = False,
    modimass: float | None = None,
    modification_condition: str = None,
    database: str = 'metabolomicspanrepo_index_nightly',
    precursor_mz_tol: float = 0.05,
    fragment_mz_tol: float = 0.05,
    min_cos: float = 0.7,
    matching_peaks: int = 6,
    cache: str = "Yes",
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Query FASST for each spectrum in library_subset and optionally merge ReDU metadata.

    Args:
        library_subset: DataFrame with a 'query_spectrum_id' column.
        analog: Whether to run an analog search.
        database: FASST database name.
        precursor_mz_tol: Precursor m/z tolerance.
        fragment_mz_tol: Fragment m/z tolerance.
        min_cos: Minimum cosine score.
        matching_peaks: Minimum number of matching peaks.
        cache: Cache policy.

    Returns:
        raw_matches: concatenated FASST responses with 'spectrum_id' column.
        redu_enriched: raw_matches merged with redu_df (empty if redu_df is None/empty).
    """


    # 0. load redu data
    print("Loading ReDU table...")

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    # get the column names
    redu_columns = fetch("SELECT name FROM pragma_table_info('redu_table')")
    redu_columns_list = redu_columns["name"].tolist()

    # exclude unwanted columns
    columns_to_exclude = [
        "filename","TermsofPosition","ComorbidityListDOIDIndex","SampleCollectionDateandTime",
        "ENVOBroadScale","ENVOLocalScale","ENVOMediumScale","qiita_sample_name","UniqueSubjectID",
        "UBERONOntologyIndex","DOIDOntologyIndex","ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex","ENVOLocalScaleIndex","ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex","classification","MS2spectra_count"
    ]
    cols = [c for c in redu_columns_list if c not in columns_to_exclude]
    col_sql = ", ".join([f'"{c}"' for c in cols])

    # count total rows
    total_rows = int(fetch("SELECT COUNT(*) as n FROM redu_table")["n"].iloc[0])


    if os.path.exists("database/redu.feather"):
        print("Loading ReDU table from feather file...")
        # First, peek at all available columns without loading everything
        all_columns = pd.read_feather("database/redu.feather", columns=[]).columns

        # Keep only the ones not excluded
        columns_to_keep = [c for c in all_columns if c not in columns_to_exclude]

        # Load only the required subset
        redu_df = pd.read_feather("database/redu.feather", columns=columns_to_keep)

    else:
        print("Loading ReDU table from database...")

        # function to fetch one page
        def fetch_page(offset, limit):
            sql = f"SELECT {col_sql} FROM redu_table LIMIT {limit} OFFSET {offset}"
            return fetch(sql)

        # loop in batches
        chunk_size = int(1E5)  # adjust as needed
        dfs = []
        for offset in range(0, total_rows, chunk_size):
            print(f"[PAGE] offset {offset} / {total_rows}")
            df_chunk = fetch_page(offset, chunk_size)
            dfs.append(df_chunk)

        # combine
        redu_df = pd.concat(dfs, ignore_index=True)


    TTL_SEC = 60*5-15                 # token freshness time (seconds)
    MAX_BATCH_REQUESTS = 50           # max new requests per batch

    # 0) build queue of all IDs to request
    all_ids = library_subset["query_spectrum_id"].astype(str).tolist()
    to_request = deque(all_ids)

    in_flight = {}   # sid -> {"token": str, "submitted_at": float}
    done = set()
    responses = []

    while len(done) < len(all_ids):
        # 1) request up to MAX_BATCH_REQUESTS in one block
        n_submit = min(MAX_BATCH_REQUESTS, len(to_request))
        for _ in range(n_submit):
            sid = to_request.popleft()
            if sid in done:
                continue
            usi_full = _as_usi(sid)
            print("submitted", usi_full)

            token = fasst.query_fasst_api_usi(
                usi_full, database,
                host=FASST_API_SERVER_URL,
                analog=analog,
                lower_delta=170, upper_delta=170,
                precursor_mz_tol=precursor_mz_tol,
                fragment_mz_tol=fragment_mz_tol,
                min_cos=min_cos,
                #cache=cache,
                blocking=False
            )

            in_flight[sid] = {"token": token, "submitted_at": time.time()}

        # 2) attempt to collect EACH in-flight EXACTLY ONCE
        for sid, rec in list(in_flight.items()):
            token = rec["token"]
            age = time.time() - rec["submitted_at"]

            # if token expired before collection attempt -> do NOT collect; re-request next batch
            if age >= TTL_SEC:
                to_request.appendleft(sid)   # prioritize next batch
                del in_flight[sid]
                continue

            # single collection attempt
            df = query_fasst_usi(
                token,
                sid,
                precursor_mz_tol=precursor_mz_tol,
                analog=analog,
                matching_peaks=matching_peaks,
                modimass=modimass,
                elimination=elimination,
                addition=addition
            )
            print(f"Returned {len(df)} rows for SID {sid}")
            if not df.empty:
                responses.append(df)
            else:
                # not ready -> re-request next batch with a fresh token
                to_request.append(sid)

            # remove from in_flight in all cases (never collect twice for same token)
            done.add(sid)
            del in_flight[sid]

    # 3) combine and return (keeps your original two-DF shape)
    if not responses:
        print("No raw data matches found.")
        return pd.DataFrame(), pd.DataFrame()

    raw_matches = pd.concat(responses, ignore_index=True)
    raw_matches.rename(columns={'GNPSLibraryAccession': 'spectrum_id'}, inplace=True)

    print(f"Retrieved {len(raw_matches)} raw data matches.")
    # 3. If ReDU data provided, merge and return enriched DataFrame
    print("Checking ReDU DataFrame for enrichment...")
    if redu_df is None or redu_df.empty:
        return raw_matches, pd.DataFrame()

    print(f"Enriching {len(raw_matches)} raw matches with ReDU metadata...")
    redu_enriched = add_redu(raw_matches, redu_df, modification_condition=modification_condition)
    

    library_subset = library_subset.copy()

    fill_empty = {
        "Smiles": "",                 # will become query_smiles
        "Adduct": "Unknown",
        "Precursor_MZ": np.nan,
        "similar_library_spectra": 0,
        "inchikey_first_block": "UNKNOWN",
    }

    for col, default in fill_empty.items():
        if col not in library_subset.columns:
            library_subset[col] = default
            
    # add Smiles column from library_subset to redu_enriched
    if 'Smiles' in library_subset.columns:
        redu_enriched = redu_enriched.merge(
            library_subset[['query_spectrum_id', 'Smiles', 'Adduct', 'Precursor_MZ', 'similar_library_spectra', 'inchikey_first_block']],
            left_on='query_spectrum_id',
            right_on='query_spectrum_id',
            how='left'
        )
        redu_enriched.rename(columns={'Smiles': 'query_smiles'}, inplace=True)


        redu_enriched['similar_library_spectra'] = redu_enriched['similar_library_spectra'] + redu_enriched['unique_spectra_in_mri'] - 2
        
        # make integer
        redu_enriched['similar_library_spectra'] = redu_enriched['similar_library_spectra'].astype('Int64')

        # make character values from 0 to "9+"
        s = redu_enriched["similar_library_spectra"].astype("Int64")
        b = s.clip(upper=9)
        redu_enriched["similar_library_spectra"] = b.astype("string").where(b < 9, "9+")

    # make library usis for the links
    redu_enriched["lib_usi"] = redu_enriched["query_spectrum_id"].apply(_as_lib_usi)

    if 'Modified' in redu_enriched.columns:
        def build_modifinder_link(row):
            usi1 = quote_plus(f"{row['USI']}")
            usi2 = quote_plus(row['lib_usi'])
            return (
                f"https://modifinder.gnps2.org/"
                f"?USI1={usi2}"
                f"&USI2={usi1}"
                f"&SMILES1={quote_plus(row['query_smiles'])}"
                f"&SMILES2&Helpers=&adduct={quote_plus(row['Adduct'])}"
                "&ppm_tolerance=25&filter_peaks_variable=0.01"
            )
               

        redu_enriched["modification_site"] = redu_enriched.apply(
            lambda row: build_modifinder_link(row)
            if (row["Modified"] != "no" and row["Adduct"] in 
                ['[M+H]1+', '[M-H]1', '[M+Na]1+', '[M+NH4]1+', '[M+K]1+', '[M+Cl]1-', '[M+Br]1-'])
            else '',
            axis=1
        )

        def build_dashboard_link_modi(row):
            usi = quote_plus(f"{row['USI']}")
            mz_modi = str(float(row['Precursor_MZ']) + float(row['Delta Mass']))
            mz_unmodi = str(f"{row['Precursor_MZ']}")
            return (
                f"https://dashboard.gnps2.org//"
                f"?xic_mz={mz_modi}"
                f"%3B"
                f"{mz_unmodi}"
                f"&xic_formula=&xic_peptide=&"
                f"xic_tolerance={str(0.01)}"
                f"&xic_ppm_tolerance=10"
                f"&xic_tolerance_unit=Da&xic_rt_window=&xic_norm=False&xic_file_grouping=FILE&xic_integration_type=AUC&show_ms2_markers=True&ms2marker_color=blue&ms2marker_size=5&ms2_identifier=MS2%3A1939&show_lcms_2nd_map=False&map_plot_zoom=%7B%7D&polarity_filtering=None&polarity_filtering2=None&tic_option=TIC&overlay_usi=None&overlay_mz=None&overlay_rt=None&overlay_color=&overlay_size=&overlay_hover=&overlay_filter_column=&overlay_filter_value=&feature_finding_type=Off&feature_finding_ppm=10&feature_finding_noise=10000&feature_finding_min_peak_rt=0.05&feature_finding_max_peak_rt=1.5&feature_finding_rt_tolerance=0.3&massql_statement=QUERY+scaninfo%28MS2DATA%29&sychronization_session_id=4843dbcfa45a4411996350ee24062088&chromatogram_options=%5B%5D&comment=&map_plot_color_scale=Hot_r&map_plot_quantization_level=Medium&plot_theme=plotly_white#%7B%22usi%22%3A%20%22"
                f"{usi}"
                f"%22%2C%20%22usi2%22%3A%20%22%22%7D"
            )
    
    
        redu_enriched["Check LC peak"] = redu_enriched.apply(
            lambda row: build_dashboard_link_modi(row),
            axis=1
        )

    return raw_matches, redu_enriched
    

def add_redu(
    raw_matches: pd.DataFrame,
    redu_df: pd.DataFrame,
    modification_condition: str = None
) -> pd.DataFrame:
    """
    Enrich raw_matches with ReDU metadata from redu_df via the 'mri' key.

    Steps:
    1. Return early if no ReDU data is provided.
    2. Make a local copy of raw_matches and sort by descending Cosine and Matching Peaks.
    3. If 'USI' exists, split it into 'mri' and 'scan_id' on ':scan:'.
    4. Deduplicate on 'mri', keeping the highest-scoring match.
    5. Rename redu_df.USIs to 'mri' if necessary.
    6. Inner-merge on 'mri' to retain only matches with ReDU metadata.
    7. Fill any NaNs in ReDU columns (those starting with 'redu_') with 'unknown'.
    """
    print("Adding ReDU metadata")
    if redu_df.empty:
        print("[add_redu] No ReDU data provided; returning original matches.")
        return raw_matches.copy()

    # 1. Prepare and sort raw matches
    df = raw_matches.copy()
    
    # 2. Extract 'mri' and 'scan_id' from USI if present
    if "USI" in df.columns:
        df[["mri", "scan_id"]] = df["USI"].str.split(":scan:", n=1, expand=True)
        df["scan_id"] = pd.to_numeric(df["scan_id"], errors="raise").astype(int)
        
    df = df.sort_values(
        by=["Cosine", "Matching Peaks"], 
        ascending=[False, False]
    )

    if 'Modified' in df.columns:
        grp_cols = ['mri', 'Delta Mass']
    else:
        grp_cols = ['mri']


    print(f"Current columns in matches: {df.columns.tolist()}")

    # 3. Keep only the top match per 'mri'
    if "mri" in df.columns:
        df["unique_spectra_in_mri"] = (
            df.groupby(grp_cols)["query_spectrum_id"]
            .transform(lambda s: s.dropna().nunique())
            .astype("Int64")
        )
        df = df.drop_duplicates(subset=grp_cols, keep="first")

    else:
        print("[add_redu] Warning: 'mri' column not found in matches; merging may fail.")



    # 4. Prepare redu_df for merging
    df_redu = redu_df.copy()
    if "USI" in df_redu.columns and "mri" not in df_redu.columns:
        print("[add_redu] Renaming 'USI' column to 'mri' in ReDU DataFrame")
        df_redu = df_redu.rename(columns={"USI": "mri"})

    # 5. Merge matches with ReDU metadata
    merged = df.merge(df_redu, on="mri", how="inner")
    print(f"[add_redu] Merged {len(df_redu)} ReDU records; result has {len(merged)} rows.")

    # 6. Fill missing values in any ReDU-specific columns
    redu_cols = [col for col in merged.columns if col.startswith("redu_")]
    if redu_cols:
        merged[redu_cols] = merged[redu_cols].fillna("unknown")


    if 'Modified' in merged.columns and modification_condition:
        if modification_condition == "Raw file":
            modification_condition = 'mri'

        valid_groups = merged[merged['Modified'] == 'no'][modification_condition].unique()
        merged = merged[merged[modification_condition].isin(valid_groups)]

    return merged

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Query FASST USI')
    parser.add_argument('input', help='Library csv path')
    parser.add_argument('structure', help='structure')
    parser.add_argument('--database', help='Database to query', default='metabolomicspanrepo_index_nightly')
    parser.add_argument('--analog', help='Analog search', default=False, type=bool)
    parser.add_argument('--precursor_mz_tol', help='Precursor m/z tolerance', default=0.05, type=float)
    parser.add_argument('--fragment_mz_tol', help='Fragment m/z tolerance', default=0.05, type=float)
    parser.add_argument('--min_cos', help='Minimum cosine score', default=0.7, type=float)
    parser.add_argument('--cache', help='Use cache', default="Yes")
    parser.add_argument('--test', help='test', default=False, type=bool)
    args = parser.parse_args()

    if args.test:
        print("Test mode enabled. Using hardcoded input structure.")
        input_structure = 'C[C@H](CCC(N[C@H](C(O)=O)CC1=CNC2=C1C=CC=C2)=O)[C@H]3CC[C@@]4([H])[C@]5([H])[C@H](O)C[C@]6([H])C[C@H](O)CC[C@]6(C)[C@H]5C[C@H](O)[C@@]43C' 
    else:
        input_structure = args.structure

    print("Starting structureMASST workflow...")
    # df, df_library_conflicts = structureMASST(library=args.input, input_structure=input_structure, analog=args.analog, database=args.database,
    #                     precursor_mz_tol=args.precursor_mz_tol, fragment_mz_tol=args.fragment_mz_tol, min_cos=args.min_cos, 
    #                     cache=args.cache)

    _, df_library_structurematch = retrieveSpectraCandidates(args.input, input_structure)

    matches = retrieve_raw_data_matches(
        df_library_structurematch, analog=args.analog, database=args.database,
        precursor_mz_tol=args.precursor_mz_tol, fragment_mz_tol=args.fragment_mz_tol,
        min_cos=args.min_cos, matching_peaks=6, cache=args.cache
    )

    print("Saving structureMASST results to tsv")
    matches.to_csv('output/df_raw_matches.tsv', sep="\t", index=False)

