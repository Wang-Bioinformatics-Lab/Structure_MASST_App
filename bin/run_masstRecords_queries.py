import pandas as pd 
from bin.match_smiles import fetch_and_match_smiles
from bin.match_smiles import detect_smiles_or_smarts
from bin.run_fasst import query_fasst_usi
# from match_smiles import fetch_and_match_smiles
# from run_fasst import query_fasst_usi
# from make_linkouts import create_gnps_link
import argparse
from collections import defaultdict
import os
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from io import StringIO
import sqlite3
import re

# ——— Shared helpers ———
def _append_limit_offset(sql: str, limit: int, offset: int) -> str:
    base = sql.strip().rstrip(';')
    return f"{base} LIMIT {int(limit)} OFFSET {int(offset)}"

def _has_limit_or_offset(sql: str) -> bool:
    return re.search(r'(?i)\b(LIMIT|OFFSET)\b', sql) is not None

def _fetch_csv(sql: str, api_endpoint: str, timeout: int) -> pd.DataFrame:
    print(f"[API ] Querying with SQL: {sql}")
    resp = requests.get(f"{api_endpoint}.csv", params={"sql": sql, "_stream": "on"}, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), dtype=str)
    print(f"[API ] returned {len(df)} rows")
    return df

def _fetch_sqlite(sql: str, sqlite_path: str) -> pd.DataFrame:
    print(f"[SQL ] Querying with SQL: {sql}")
    with sqlite3.connect(sqlite_path) as conn:
        df = pd.read_sql(sql, conn, dtype=str)
    print(f"[SQL ] returned {len(df)} rows")
    return df

def _get_fetcher(sqlite_path: str, api_endpoint: str, timeout: int):
    use_sqlite = bool(sqlite_path and os.path.isfile(sqlite_path))
    if use_sqlite:
        return lambda sql: _fetch_sqlite(sql, sqlite_path)
    else:
        return lambda sql: _fetch_csv(sql, api_endpoint, timeout)

def _batched_fetch(template_sql: str,
                   id_list: list[int],
                   fetch,
                   chunk_size: int,
                   paginate: bool = False,
                   page_size: int = 50000,
                   max_pages: int | None = None) -> pd.DataFrame:
    """
    When paginate=True, each chunk is fetched in pages using LIMIT/OFFSET with `page_size`.
    Pagination stops when a page returns < page_size rows (or max_pages is hit).
    If the SQL already contains LIMIT/OFFSET, pagination is skipped for that chunk.
    """
    if not id_list:
        print("[BATCH] No IDs to fetch.")
        return pd.DataFrame()

    dfs = []
    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i : i + chunk_size]
        batch_idx = i // chunk_size + 1
        print(f"[BATCH] chunk {batch_idx}: {len(chunk)} IDs")

        sql = template_sql.format(ids=",".join(map(str, chunk)))

        # If the user already specified LIMIT/OFFSET, do a single fetch.
        if not paginate or _has_limit_or_offset(sql):
            df = fetch(sql)
            if not df.empty:
                print(f"[BATCH] chunk {batch_idx}: {len(chunk)} IDs returned {len(df)} rows")
                dfs.append(df)
            continue

        # Paginate with LIMIT/OFFSET
        offset = 0
        pages = 0
        total_rows_this_chunk = 0
        while True:
            pages += 1
            if (max_pages is not None) and (pages > max_pages):
                print(f"[BATCH] chunk {batch_idx}: reached max_pages={max_pages}, stopping pagination")
                break

            paged_sql = _append_limit_offset(sql, page_size, offset)
            df_page = fetch(paged_sql)

            n_rows = len(df_page)
            total_rows_this_chunk += n_rows
            print(f"[BATCH] chunk {batch_idx} page {pages}: offset={offset} limit={page_size} -> {n_rows} rows")

            if n_rows == 0:
                break

            dfs.append(df_page)

            if n_rows < page_size:
                # Last page for this chunk
                break

            offset += page_size

        print(f"[BATCH] chunk {batch_idx}: total {total_rows_this_chunk} rows across {pages} page(s)")

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        print(f"[BATCH] total returned {len(result)} rows")
        return result
    else:
        print("[BATCH] returned 0 rows")
        return pd.DataFrame()


# ——— Part 1: library lookup ———

def get_library_table(
    smiles: str,
    searchtype: str = "exact",
    tanimoto_threshold: float = 0.8,
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 100
) -> pd.DataFrame:
    """
    Given a SMILES, returns the library_table for its InChIKey prefix.
    """
    if searchtype not in ["exact", "substructure", "tanimoto"]:
        raise ValueError(f"Invalid search type: {searchtype}. Must be 'exact' or 'substructure'.")
    
    if searchtype == "exact":
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        prefix = inchi.MolToInchiKey(mol).split('-')[0]

        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
        lib_sql = (
            "SELECT * FROM library_table "
            f"WHERE InChIKey_smiles_firstBlock = '{prefix}' "
            "AND ppmBetweenExpAndThMass <= 20 "
            "AND (msMassAnalyzer NOT IN ('quadrupole', 'ion trap') OR msMassAnalyzer IS NULL)"
        )
        library_df = fetch(lib_sql)

        if 'collision_energy' in library_df.columns:
            library_df['collision_energy'] = library_df['collision_energy'].apply(lambda x: str(x) if pd.notna(x) else 'unknown')
        if 'msMassAnalyzer' in library_df.columns:
            library_df['msMassAnalyzer'] = library_df['msMassAnalyzer'].apply(lambda x: str(x) if pd.notna(x) else '_unknown')

        # Only fillna for known string columns
        library_df[['Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = library_df[
            ['Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')

        
        # rename column InChIKey_smiles_firstBlock to inchikey_first_block
        if 'InChIKey_smiles_firstBlock' in library_df.columns:
            library_df.rename(columns={'InChIKey_smiles_firstBlock': 'inchikey_first_block'}, inplace=True)

        # rename spectrum_id to query_spectrum_id
        if 'spectrum_id' in library_df.columns:
            library_df.rename(columns={'spectrum_id': 'query_spectrum_id'}, inplace=True)

        df_final = library_df.copy()
    else:

        structure_type = detect_smiles_or_smarts(smiles)

        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
        lib_sql_minimal = (
            "SELECT spectrum_id_int, Smiles, InChIKey_smiles FROM library_table "
            "WHERE ppmBetweenExpAndThMass <= 20 "
            "AND (msMassAnalyzer NOT IN ('quadrupole', 'ion trap') OR msMassAnalyzer IS NULL) "
            "AND Smiles IS NOT NULL "
            "AND Smiles != '' "
            "AND Smiles != 'NaN'"
        )
        library_df_minimal = fetch(lib_sql_minimal)

        library_df_minimal = fetch_and_match_smiles(library_df_minimal, smiles, match_type=searchtype, smiles_name='only',
                                             smiles_type=structure_type, formula_base='any', element_diff='any',
                                             max_by_grp=None, max_overall=None, tanimoto_threshold=tanimoto_threshold)
        

        # If no matches, return early
        if isinstance(library_df_minimal, list) or library_df_minimal.empty:
            print("No matching structures found in the library.")
            df_final = pd.DataFrame()
            return df_final
        
        matched_ids = library_df_minimal['spectrum_id_int'].dropna().astype(int).unique().tolist()

        lib_sql_template = (
            "SELECT spectrum_id_int, spectrum_id, Compound_Name, Ion_Mode, collision_energy, Precursor_MZ, Adduct, "
            "msManufacturer, msMassAnalyzer, GNPS_library_membership "
            "FROM library_table WHERE spectrum_id_int IN ({ids})"
        )

        df_metadata = _batched_fetch(lib_sql_template, matched_ids, fetch, chunk_size=100)

        # join and return
        df_final = library_df_minimal.merge(df_metadata, on='spectrum_id_int', how='left')

        if 'collision_energy' in df_final.columns:
            df_final['collision_energy'] = df_final['collision_energy'].apply(lambda x: str(x) if pd.notna(x) else 'unknown')
        if 'msMassAnalyzer' in df_final.columns:
            df_final['msMassAnalyzer'] = df_final['msMassAnalyzer'].apply(lambda x: str(x) if pd.notna(x) else '_unknown')

        df_final[['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = df_final[
            ['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')



        df_final.rename(columns={'spectrum_id': 'query_spectrum_id'}, inplace=True)


    for col in ['spectrum_id_int']:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    return df_final


# ——— Part 2: MASST + ReDU lookup ———

def get_masst_and_redu_tables(
    library_df: pd.DataFrame,
    cosine_threshold: float = 0.7,
    matching_peaks: int = 5,
    sqlite_path: str | None = None,
    api_endpoint: str = "http://127.0.0.1:8001/masst_records",
    timeout: int = 10,
    chunk_size: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a non-empty library_df, returns (masst_df, redu_df).
    If library_df is empty, both returned DataFrames will be empty.
    """
    if library_df.empty:
        print("[PART 2] empty library_df → nothing to fetch")
        return pd.DataFrame(), pd.DataFrame()

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    # — Masst table —
    sids = library_df['spectrum_id_int'].dropna().unique().tolist()
    print(f"[STEP 2] masst_table for {len(sids)} spectrum_id_ints")
    if not sids:
        return pd.DataFrame(), pd.DataFrame()
    masst_sql_tmpl = (
        "SELECT * FROM ("
        "  SELECT *, "
        "         ROW_NUMBER() OVER (PARTITION BY mri_id_int ORDER BY cosine DESC) AS rn "
        "  FROM masst_table "
        "  WHERE spectrum_id_int IN ({ids}) "
        f"    AND cosine >= {cosine_threshold} "
        f"    AND matching_peaks >= {matching_peaks}"
        ") t "
        "WHERE rn = 1"
    )

    masst_df = _batched_fetch(
        masst_sql_tmpl,
        sids,
        fetch,
        chunk_size=chunk_size,
        paginate=True,        # <-- enable pagination mode
        page_size=500000,      # <-- set to your Datasette cap
        max_pages=None        # <-- or set an upper bound if desired
)
    if masst_df.empty:
        print("[STEP 2] no masst hits → exiting part 2")
        return pd.DataFrame(), pd.DataFrame()

    # — add MRI strings —
    mids = masst_df['mri_id_int'].dropna().unique().tolist()
    if mids:
        print(f"[STEP 3a] fetching mri strings for {len(mids)} mri_id_ints")
        mri_sql   = "SELECT mri_id_int, mri FROM mri_table WHERE mri_id_int IN ({ids})"
        mri_map   = _batched_fetch(mri_sql, mids, fetch, 500)
        if not mri_map.empty:
            masst_df = masst_df.merge(mri_map, on='mri_id_int', how='left')
        else:
            masst_df['mri'] = None
    else:
        masst_df['mri'] = None

    # — add spectrum_id strings —
    print(f"[STEP 3b] merging spectrum_id for {len(sids)} spectrum_id_ints")
    spec_map = library_df[['spectrum_id_int', 'query_spectrum_id', 'Adduct', 'Compound_Name', 'Precursor_MZ', 'inchikey_first_block']].drop_duplicates()
    # make sure we match on same datatype
    spec_map['spectrum_id_int'] = spec_map['spectrum_id_int'].astype('Int64')
    masst_df['spectrum_id_int'] = masst_df['spectrum_id_int'].astype('Int64')

    masst_df = masst_df.merge(spec_map, on='spectrum_id_int', how='left')

    # — ReDU table —
    if not mids:
        print("[STEP 4] no mri_ids → skipping redu")
        return masst_df, pd.DataFrame()

    print(f"[STEP 4] redu_table for {len(mids)} mri_id_ints")

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
    sql = "SELECT name FROM pragma_table_info('redu_table')"
    redu_columns = fetch(sql)
    
    redu_columns_list = redu_columns['name'].tolist()
    columns_to_exclude = ['filename', 'TermsofPosition', 'ComorbidityListDOIDIndex', 'SampleCollectionDateandTime', 'ENVOBroadScale', 'ENVOLocalScale', 'ENVOMediumScale', 'qiita_sample_name',
                          'UniqueSubjectID', 'UBERONOntologyIndex', 'DOIDOntologyIndex', 'ENVOEnvironmentBiomeIndex', 'ENVOEnvironmentMaterialIndex', 'ENVOLocalScaleIndex', 'ENVOBroadScaleIndex',
                          'ENVOMediumScaleIndex', 'classification', 'MS2spectra_count']
    redu_columns_list = [col for col in redu_columns_list if col not in columns_to_exclude]

    redu_sql_tmpl = f"SELECT {', '.join(redu_columns_list)} FROM redu_table WHERE mri_id_int IN ({{ids}})"

    redu_df = _batched_fetch(redu_sql_tmpl, mids, fetch, 500)

    return masst_df, redu_df