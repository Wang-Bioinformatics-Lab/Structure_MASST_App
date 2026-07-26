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
from bin.shared_data import get_redu_table_cached

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
    output_folder: str | None = None,
    require_redu: bool = True,
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
        output_folder: Folder to save output files.
        require_redu: default True matches the app's normal, documented behaviour -- only
            matches with ReDU sample metadata are returned. Pass False to keep every raw
            FASST match (left join), enriched with dataset accession/title from the broader
            dataset registry where full ReDU metadata isn't available. Explicit opt-in.
    Returns:
        raw_matches: concatenated FASST responses with 'spectrum_id' column.
        redu_enriched: raw_matches merged with redu_df (empty if redu_df is None/empty).
    """

    # 0. load redu data
    print("Loading ReDU table...")

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    # # get the column names
    redu_columns = fetch("SELECT name FROM pragma_table_info('redu_table')")
    redu_columns_list = redu_columns["name"].tolist()

    # # exclude unwanted columns
    columns_to_exclude = [
        "filename","TermsofPosition","ComorbidityListDOIDIndex","SampleCollectionDateandTime",
        "ENVOBroadScale","ENVOLocalScale","ENVOMediumScale","qiita_sample_name","UniqueSubjectID",
        "UBERONOntologyIndex","DOIDOntologyIndex","ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex","ENVOLocalScaleIndex","ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex","classification","MS2spectra_count"
    ]
    redu_columns_list = [c for c in redu_columns_list if c not in columns_to_exclude]

    redu_df = get_redu_table_cached(fetch, redu_columns_list, sqlite_path)
   


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
            
            # print(f"Attempting to collect SID {sid} with token {token} (age {age:.1f}s)")
            # single collection attempt
            df = query_fasst_usi(
                token,
                sid,
                precursor_mz_tol=precursor_mz_tol,
                analog=analog,
                matching_peaks=matching_peaks,
                modimass=modimass,
                elimination=elimination,
                addition=addition,
                log_output=output_folder
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
    redu_enriched = add_redu(raw_matches, redu_df, modification_condition=modification_condition,
                              require_redu=require_redu)
    if not require_redu:
        redu_enriched = _fill_non_redu_dataset_info(redu_enriched, sqlite_path)

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


def _fill_non_redu_dataset_info(df: pd.DataFrame, sqlite_path: str | None) -> pd.DataFrame:
    """For rows that survived a require_redu=False (left) join without full ReDU sample
    metadata, fill in at least a dataset accession + title from the broader dataset/mri
    registry (masst_records.sqlite's dataset_table / mri_table), which covers many more
    files (~920k) and datasets (~4990) than ReDU documents (~412k files) -- so a match can
    still be attributed to a named, real public dataset even without species/body-part/etc.
    No-op (returns df unchanged) if sqlite_path isn't available or the tables aren't found.
    """
    if df.empty or "mri" not in df.columns or not sqlite_path:
        return df
    try:
        con = sqlite3.connect(sqlite_path)
        mri_map = pd.read_sql("SELECT mri, mri_id_int FROM mri_table", con)
        dataset_map = pd.read_sql("SELECT Dataset, title AS dataset_title FROM dataset_table", con)
        con.close()
    except Exception as e:
        print(f"[_fill_non_redu_dataset_info] skipped ({e})")
        return df

    df = df.copy()
    df["dataset_accession_guess"] = df["mri"].str.extract(r"(MSV\d+|GNPS\d+|CCMS\d+)", expand=False)
    df = df.merge(mri_map, on="mri", how="left", suffixes=("", "_lookup"))
    if "mri_id_int_lookup" in df.columns:
        df["mri_id_int"] = df["mri_id_int"].combine_first(df["mri_id_int_lookup"]) if "mri_id_int" in df.columns else df["mri_id_int_lookup"]
        df.drop(columns=["mri_id_int_lookup"], inplace=True)
    df = df.merge(dataset_map, left_on="dataset_accession_guess", right_on="Dataset", how="left",
                  suffixes=("", "_registry"))
    n_registry_only = df["dataset_title"].notna().sum()
    print(f"[_fill_non_redu_dataset_info] {n_registry_only}/{len(df)} rows resolved a dataset title "
          f"from the broader registry (beyond ReDU coverage).")
    return df


def add_redu(
    raw_matches: pd.DataFrame,
    redu_df: pd.DataFrame,
    modification_condition: str = None,
    require_redu: bool = True,
) -> pd.DataFrame:
    """
    Enrich raw_matches with ReDU metadata from redu_df via the 'mri' key.

    Steps:
    1. Return early if no ReDU data is provided.
    2. Make a local copy of raw_matches and sort by descending Cosine and Matching Peaks.
    3. If 'USI' exists, split it into 'mri' and 'scan_id' on ':scan:'.
    4. Deduplicate on 'mri', keeping the highest-scoring match.
    5. Rename redu_df.USIs to 'mri' if necessary.
    6. Merge on 'mri' -- inner (default, require_redu=True) to retain only matches that
       have ReDU sample metadata (the app's normal, documented behaviour: a match with no
       sample context isn't actionable for most users). Pass require_redu=False for a left
       merge that keeps every raw FASST match, with ReDU columns blank where no metadata
       exists -- this is an explicit opt-in for investigative use, not the default.
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
    join_how = "inner" if require_redu else "left"
    merged = df.merge(df_redu, on="mri", how=join_how)
    print(f"[add_redu] Merged {len(df_redu)} ReDU records ({join_how} join); result has {len(merged)} rows"
          f" (of {len(df)} raw matches).")

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

def retrieve_raw_data_matches_from_peaks(
    spectra: list,
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
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10,
    output_folder: str | None = None,
    require_redu: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like retrieve_raw_data_matches but submits spectra by peaks instead of USI.

    Args:
        spectra: list of dicts, each with keys:
            - spectrum_id (str)   unique label used as query_spectrum_id in results
            - precursor_mz (float)
            - peaks (list of [mz, intensity])
        require_redu: default True matches the app's normal, documented behaviour --
            only matches with ReDU sample metadata are returned in the enriched result.
            Pass False to keep every raw FASST match (left join), with dataset accession
            and title filled in from the broader dataset/mri registry (which covers many
            more files than ReDU does) wherever full ReDU metadata isn't available. This
            is an explicit opt-in for investigative use, not the default.
    Returns:
        raw_matches, redu_enriched  (same shape as retrieve_raw_data_matches)
    """
    print("Loading ReDU table...")
    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    redu_columns = fetch("SELECT name FROM pragma_table_info('redu_table')")
    redu_columns_list = redu_columns["name"].tolist()
    columns_to_exclude = [
        "filename", "TermsofPosition", "ComorbidityListDOIDIndex", "SampleCollectionDateandTime",
        "ENVOBroadScale", "ENVOLocalScale", "ENVOMediumScale", "qiita_sample_name", "UniqueSubjectID",
        "UBERONOntologyIndex", "DOIDOntologyIndex", "ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex", "ENVOLocalScaleIndex", "ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex", "classification", "MS2spectra_count",
    ]
    redu_columns_list = [c for c in redu_columns_list if c not in columns_to_exclude]
    redu_df = get_redu_table_cached(fetch, redu_columns_list, sqlite_path)

    TTL_SEC = 60 * 5 - 15
    MAX_BATCH_REQUESTS = 50

    spec_lookup = {s["spectrum_id"]: s for s in spectra}
    all_ids = [s["spectrum_id"] for s in spectra]
    to_request = deque(all_ids)
    in_flight = {}
    done = set()
    responses = []

    while len(done) < len(all_ids):
        n_submit = min(MAX_BATCH_REQUESTS, len(to_request))
        for _ in range(n_submit):
            sid = to_request.popleft()
            if sid in done:
                continue
            spec = spec_lookup[sid]
            print(f"Submitting peaks query for spectrum {sid}")
            token = fasst.query_fasst_api_peaks(
                spec["precursor_mz"],
                spec["peaks"],
                database,
                host=FASST_API_SERVER_URL,
                analog=analog,
                lower_delta=170,
                upper_delta=170,
                precursor_mz_tol=precursor_mz_tol,
                fragment_mz_tol=fragment_mz_tol,
                min_cos=min_cos,
                blocking=False,
            )
            in_flight[sid] = {"token": token, "submitted_at": time.time()}

        for sid, rec in list(in_flight.items()):
            token = rec["token"]
            age = time.time() - rec["submitted_at"]

            if age >= TTL_SEC:
                to_request.appendleft(sid)
                del in_flight[sid]
                continue

            df = query_fasst_usi(
                token,
                sid,
                precursor_mz_tol=precursor_mz_tol,
                analog=analog,
                matching_peaks=matching_peaks,
                modimass=modimass,
                elimination=elimination,
                addition=addition,
                log_output=output_folder,
            )
            print(f"Returned {len(df)} rows for spectrum {sid}")
            if not df.empty:
                responses.append(df)
            else:
                to_request.append(sid)

            done.add(sid)
            del in_flight[sid]

    if not responses:
        print("No raw data matches found.")
        return pd.DataFrame(), pd.DataFrame()

    raw_matches = pd.concat(responses, ignore_index=True)
    raw_matches.rename(columns={"GNPSLibraryAccession": "spectrum_id"}, inplace=True)

    print(f"Retrieved {len(raw_matches)} raw data matches.")

    if redu_df is None or redu_df.empty:
        return raw_matches, pd.DataFrame()

    print(f"Enriching {len(raw_matches)} raw matches with ReDU metadata...")
    redu_enriched = add_redu(raw_matches, redu_df, modification_condition=modification_condition,
                              require_redu=require_redu)
    if not require_redu:
        redu_enriched = _fill_non_redu_dataset_info(redu_enriched, sqlite_path)

    # No library_subset merge: MGF spectra have no SMILES / Adduct / InChIKey
    # Build lib_usi from GNPSLibraryAccession (available as spectrum_id after rename)
    if "spectrum_id" in redu_enriched.columns:
        redu_enriched["lib_usi"] = redu_enriched["spectrum_id"].apply(
            lambda x: (
                x if isinstance(x, str) and x.startswith("mzspec:")
                else f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if isinstance(x, str) and x.startswith("CCMSLIB")
                else f"mzspec:MASSBANK::accession:{x}" if isinstance(x, str) and x.strip()
                else ""
            )
        )
    else:
        redu_enriched["lib_usi"] = ""

    return raw_matches, redu_enriched


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

