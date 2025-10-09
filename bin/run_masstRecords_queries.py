import pandas as pd 
from bin.match_smiles import fetch_and_match_smiles
from bin.match_smiles import detect_smiles_or_smarts
from bin.run_fasst import query_fasst_usi
# from match_smiles import fetch_and_match_smiles
# from run_fasst import query_fasst_usi
# from make_linkouts import create_gnps_link
from collections import defaultdict
import os
import requests
from io import StringIO
import sqlite3
import re
from typing import Iterable, Optional, Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, inchi
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import base64
import math
import json
import urllib.parse
from pathlib import Path

# ——— Shared helpers ———
def _append_limit_offset(sql: str, limit: int, offset: int) -> str:
    base = sql.strip().rstrip(';')
    return f"{base} LIMIT {int(limit)} OFFSET {int(offset)}"

def _has_limit_or_offset(sql: str) -> bool:
    return re.search(r'(?i)\b(LIMIT|OFFSET)\b', sql) is not None

# --- internal helper ---------------------------------------------------------
def _coerce_types_except_blobs(
    df: pd.DataFrame,
    blob_cols: Tuple[str, ...] = ("fp_pattern", "fp_morgan",),
    normalize_types: bool = True,
) -> pd.DataFrame:
    """
    Make all columns 'str', EXCEPT columns in blob_cols,
    which are left as bytes (if present).
    If a blob col accidentally arrived as str, convert it back to bytes via latin1.
    """
    if df is None or df.empty:
        return df

    blob_cols = tuple(c for c in blob_cols if c in df.columns)

    # Fix blob columns first: ensure bytes dtype
    for c in blob_cols:
        # If values came in as str (e.g., because a client decoded), re-encode losslessly
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            # convert any str -> bytes (latin1 is a reversible 1:1 mapping)
            df[c] = df[c].apply(lambda v: v.encode("latin1") if isinstance(v, str) else v)

    if normalize_types:
        # Cast everything else to str (preserve previous behavior that used dtype=str)
        for c in df.columns:
            if c in blob_cols:
                continue
            df[c] = df[c].astype(str)

    return df

def _sql_mentions_any(sql: str, cols) -> bool:
    if not cols: 
        return False
    pat = r"|".join(rf"\b{re.escape(c)}\b" for c in cols)
    return re.search(pat, sql, flags=re.IGNORECASE) is not None

_BLOB_JSON_KEY = "$base64"
def _decode_datasette_blob_cell(v):
    """
    Datasette JSON returns blobs as {"$base64": true, "encoded": "..."}.
    Locally we may see bytes/memoryview already. Return raw bytes or None.
    """
    if v is None:
        return None
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict) and v.get(_BLOB_JSON_KEY) and "encoded" in v:
        try:
            return base64.b64decode(v["encoded"])
        except Exception:
            return None
    return None  

# --- public helpers (updated) ---------------------------------------
def _fetch_csv(sql: str, api_endpoint: str, timeout: int,
               blob_cols: Optional[Iterable[str]] = ("fp_pattern","fp_morgan",),
               normalize_types: bool = True) -> pd.DataFrame:
    """
    Fetch via HTTP.
    - If SQL mentions any blob column, use Datasette JSON (_shape=array) and base64-decode blobs to bytes.
    - Otherwise, use CSV exactly as before.
    Keeps previous behavior: strings everywhere except for blob_cols, which are bytes.
    """
    want_json = _sql_mentions_any(sql, blob_cols)

    if want_json:
        # JSON path (safe blob handling)
        print(f"[API ] Querying (JSON) with SQL: {sql}")
        resp = requests.get(
            f"{api_endpoint}.json",
            params={"sql": sql, "_shape": "array"},
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        rows = resp.json() 
        df = pd.DataFrame(rows)
        print(f"[API ] returned {len(df)} rows (json)")

        # Decode only the blob columns we actually received
        if blob_cols:
            for c in blob_cols:
                if c in df.columns:
                    df[c] = df[c].map(_decode_datasette_blob_cell)

        # Keep prior typing behavior for non-blob columns
        if blob_cols:
            df = _coerce_types_except_blobs(df, tuple(blob_cols), normalize_types=normalize_types)
        else:
            # no blobs; optional normalization across the board
            if normalize_types:
                df = df.convert_dtypes()

        return df

    # CSV path 
    print(f"[API ] Querying (CSV) with SQL: {sql}")
    resp = requests.get(f"{api_endpoint}.csv", params={"sql": sql, "_stream": "on"}, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), dtype=str)
    print(f"[API ] returned {len(df)} rows (csv)")

    # Try to coerce types but preserve blobs
    if blob_cols:
        df = _coerce_types_except_blobs(df, tuple(blob_cols), normalize_types=normalize_types)
    return df


def _sqlite_ro_connect(sqlite_path: str, immutable: bool = True) -> sqlite3.Connection:
    # Build a proper file: URI for SQLite
    p = Path(sqlite_path).resolve()
    qs = "mode=ro"
    if immutable:
        qs += "&immutable=1"     # treat file as read-only, can speed things up if DB won’t change
    uri = f"file:{urllib.parse.quote(str(p))}?{qs}"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)

def _fetch_sqlite(sql: str, sqlite_path: str,
                  blob_cols: Optional[Iterable[str]] = ("fp_pattern","fp_morgan",),
                  normalize_types: bool = True) -> pd.DataFrame:
    """
    Read-only SQLite fetch. BLOBs stay as bytes, other cols coerced as before.
    """
    print(f"[SQL ] Querying (RO) with SQL: {sql}")
    with _sqlite_ro_connect(sqlite_path) as conn:
        # Enforce query-only mode at the connection level
        conn.execute("PRAGMA query_only=ON;")
        # Keep blobs as bytes while preventing TEXT→str coercion
        conn.text_factory = bytes
        df = pd.read_sql_query(sql, conn)
    print(f"[SQL ] returned {len(df)} rows")

    if blob_cols:
        df = _coerce_types_except_blobs(df, tuple(blob_cols), normalize_types=normalize_types)
    return df


def _get_fetcher(sqlite_path: str, api_endpoint: str, timeout: int,
                 blob_cols: Optional[Iterable[str]] = ("fp_pattern", "fp_morgan", ),
                 normalize_types: bool = True):
    """
    Returns a single-arg callable(sql) that fetches a DataFrame.
    Uses SQLite if sqlite_path exists, otherwise HTTP API.
    """
    use_sqlite = bool(sqlite_path and os.path.isfile(sqlite_path))
    if use_sqlite:
        return lambda sql: _fetch_sqlite(sql, sqlite_path, blob_cols=blob_cols, normalize_types=normalize_types)
    else:
        return lambda sql: _fetch_csv(sql, api_endpoint, timeout, blob_cols=blob_cols, normalize_types=normalize_types)


def _batched_fetch(template_sql: str,
                   id_list: list[int] | None,
                   fetch,
                   chunk_size: int,
                   paginate: bool = False,
                   page_size: int = 50000,
                   max_pages: int | None = None) -> pd.DataFrame:
    """
    Two modes:
      1) ID-batched mode: if template_sql contains '{ids}', chunk id_list and format into the SQL.
      2) Table mode: if no '{ids}' placeholder is present, ignore id_list/chunk_size and (optionally) paginate
         the whole query with LIMIT/OFFSET until fewer than page_size rows are returned.
    """
    uses_ids = "{ids}" in template_sql

    # -------- Table mode (no {ids} in template_sql)
    if not uses_ids:
        if not paginate or _has_limit_or_offset(template_sql):
            df = fetch(template_sql)
            print(f"[BATCH] table-mode: returned {len(df)} rows (single fetch)")
            return df

        dfs = []
        offset = 0
        pages = 0
        total_rows = 0
        print("[BATCH] table-mode: pagination enabled")
        while True:
            pages += 1
            if (max_pages is not None) and (pages > max_pages):
                print(f"[BATCH] table-mode: reached max_pages={max_pages}, stopping")
                break

            paged_sql = _append_limit_offset(template_sql, page_size, offset)
            df_page = fetch(paged_sql)
            n_rows = len(df_page)
            total_rows += n_rows
            print(f"[BATCH] table-mode page {pages}: offset={offset} limit={page_size} -> {n_rows} rows")

            if n_rows == 0:
                break

            dfs.append(df_page)

            if n_rows < page_size:
                break

            offset += page_size

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            print(f"[BATCH] table-mode total returned {len(result)} rows across {pages} page(s)")
            return result
        else:
            print("[BATCH] table-mode returned 0 rows")
            return pd.DataFrame()

    # -------- ID-batched mode (has {ids} in template_sql)
    if not id_list:
        print("[BATCH] No IDs to fetch.")
        return pd.DataFrame()

    dfs = []
    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i : i + chunk_size]
        batch_idx = i // chunk_size + 1
        print(f"[BATCH] chunk {batch_idx}: {len(chunk)} IDs")

        sql = template_sql.format(ids=",".join(map(str, chunk)))

        # If LIMIT/OFFSET already present or pagination disabled, single fetch
        if not paginate or _has_limit_or_offset(sql):
            df = fetch(sql)
            if not df.empty:
                print(f"[BATCH] chunk {batch_idx}: {len(chunk)} IDs returned {len(df)} rows")
                dfs.append(df)
            continue

        # Paginate with LIMIT/OFFSET per chunk
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

def _quote_for_sql_in(values):
    """Quote a list of string values for SQL IN (...). Handles None and escapes single quotes."""
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).replace("'", "''")
        out.append(f"'{s}'")
    return out

# ——— Part 1: library lookup ———
def _decode_datasette_blob_cell(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, memoryview): return v.tobytes()
    if isinstance(v, (bytes, bytearray)): return bytes(v)
    if isinstance(v, dict) and v.get("$base64") and "encoded" in v:
        try: return base64.b64decode(v["encoded"])
        except Exception: return None
    if isinstance(v, str) and v.strip().startswith("{") and '"$base64"' in v:
        try:
            d = json.loads(v)
            if d.get("$base64") and "encoded" in d:
                return base64.b64decode(d["encoded"])
        except Exception:
            return None
    return None

def _normalize_fp_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(_decode_datasette_blob_cell)
    return df

def _fill_bytes_from_hex(df: pd.DataFrame, col: str):
    hex_col = f"{col}_hex"
    if hex_col in df.columns:
        mask = df[col].isna() & df[hex_col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, hex_col].apply(
                lambda s: bytes.fromhex(s) if isinstance(s, str) and len(s) % 2 == 0 else None
            )
        # drop the helper column
        df.drop(columns=[hex_col], inplace=True)

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
    
        # get monoisotopic mass
        precursor_mz = round(Chem.Descriptors.ExactMolWt(mol), 4)
        mass_tolerance = 0.02

        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
        sql = "SELECT name FROM pragma_table_info('library_table')"
        library_columns = fetch(sql)
        library_columns_list = library_columns['name'].tolist()
        columns_to_exclude = ['fp_pattern', 'fp_morgan', 'fp_morgan_popcnt', 'fp_popcnt']

        library_column_list = [c for c in library_columns_list if c not in columns_to_exclude]

        #Exclude blob columns from the select statement
        prefix = inchi.MolToInchiKey(mol).split('-')[0]
        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
        lib_sql = (
            "SELECT " + ", ".join(library_column_list) + " FROM library_table "
            f"WHERE InChIKey_smiles_firstBlock = '{prefix}' "
            f"AND ABS(ExactMass - {precursor_mz}) <= {mass_tolerance} "
            "AND ppmBetweenExpAndThMass <= 20 "
            "AND (msMassAnalyzer NOT IN ('quadrupole', 'ion trap') OR msMassAnalyzer IS NULL)"
        )
        library_df = fetch(lib_sql)

        if 'collision_energy' in library_df.columns:
            library_df['collision_energy'] = library_df['collision_energy'].apply(lambda x: str(x) if pd.notna(x) else 'unknown')
        if 'msMassAnalyzer' in library_df.columns:
            library_df['msMassAnalyzer'] = library_df['msMassAnalyzer'].apply(lambda x: str(x) if pd.notna(x) else '_unknown')

        library_df[['Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = library_df[
            ['Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')

        if 'InChIKey_smiles_firstBlock' in library_df.columns:
            library_df.rename(columns={'InChIKey_smiles_firstBlock': 'inchikey_first_block'}, inplace=True)

        if 'spectrum_id' in library_df.columns:
            library_df.rename(columns={'spectrum_id': 'query_spectrum_id'}, inplace=True)

        df_final = library_df.copy()
    else:
        structure_type = detect_smiles_or_smarts(smiles)
        fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

        if searchtype == "tanimoto":
            fp_blob_cols = ["fp_morgan"]
            fp_predicate = "AND l.fp_morgan IS NOT NULL AND length(l.fp_morgan) > 0"
            fp_select = "l.fp_morgan, l.fp_morgan_popcnt, hex(l.fp_morgan) AS fp_morgan_hex"
        else:
            fp_blob_cols = ["fp_pattern"]
            fp_predicate = "AND l.fp_pattern IS NOT NULL AND length(l.fp_pattern) > 0"
            fp_select = "l.fp_pattern, hex(l.fp_pattern) AS fp_pattern_hex"

        rep_sql = f"""
        WITH reps AS (
        SELECT MIN(l.spectrum_id_int) AS rep_id
        FROM library_table AS l
        WHERE l.ppmBetweenExpAndThMass <= 20
            AND (l.msMassAnalyzer NOT IN ('quadrupole', 'ion trap') OR l.msMassAnalyzer IS NULL)
            AND l.InChIKey_smiles_firstBlock IS NOT NULL
            AND l.InChIKey_smiles_firstBlock != ''
            AND l.InChIKey_smiles_firstBlock != 'NaN'
            {fp_predicate}
        GROUP BY l.InChIKey_smiles_firstBlock
        )
        SELECT l.spectrum_id_int,
            l.InChIKey_smiles,
            l.InChIKey_smiles_firstBlock,
            l.Smiles,
            {fp_select}
        FROM library_table AS l
        JOIN reps r ON l.spectrum_id_int = r.rep_id
        """
        df_reps = fetch(rep_sql)

        # decode base64 blobs (Datasette JSON), no-op for local bytes
        _normalize_fp_columns(df_reps, fp_blob_cols)

        # if still None (as your debug shows), backfill bytes from hex()
        for c in fp_blob_cols:
            _fill_bytes_from_hex(df_reps, c)

        # ensure popcnt numeric for tanimoto
        if "fp_morgan_popcnt" in df_reps.columns:
            df_reps["fp_morgan_popcnt"] = pd.to_numeric(df_reps["fp_morgan_popcnt"], errors="coerce")

        # Run your existing matcher ONLY on the reps
        library_df_minimal = fetch_and_match_smiles(
            df_reps,
            smiles,
            match_type=searchtype,
            smiles_name='only',
            smiles_type=structure_type,
            formula_base='any',
            element_diff='any',
            max_by_grp=None,
            max_overall=None,
            tanimoto_threshold=tanimoto_threshold
        )

        # Early exit if nothing matched
        if isinstance(library_df_minimal, list) or library_df_minimal.empty:
            print("No matching structures found in the library.")
            df_final = pd.DataFrame()
            return df_final

        matched_blocks = (
            library_df_minimal['InChIKey_smiles_firstBlock']
            .dropna().astype(str).unique().tolist()
        )
        if not matched_blocks:
            df_final = pd.DataFrame()
            return df_final

        matched_blocks_quoted = _quote_for_sql_in(matched_blocks)

        block_ids_sql_tmpl = """
        SELECT spectrum_id_int
        FROM library_table
        WHERE InChIKey_smiles_firstBlock IN ({ids})
        AND ppmBetweenExpAndThMass <= 20
        AND (msMassAnalyzer NOT IN ('quadrupole', 'ion trap') OR msMassAnalyzer IS NULL)
        AND Smiles IS NOT NULL
        AND Smiles != ''
        AND Smiles != 'NaN'
        """

        all_ids_df = _batched_fetch(
            block_ids_sql_tmpl,
            matched_blocks_quoted, 
            fetch,
            chunk_size=30,
            paginate=True,
            page_size=500000,
            max_pages=None
        )


        if all_ids_df is None or all_ids_df.empty:
            df_final = pd.DataFrame()
            return df_final

        all_spectrum_ids = all_ids_df['spectrum_id_int'].dropna().astype(int).unique().tolist()
        if not all_spectrum_ids:
            df_final = pd.DataFrame()
            return df_final

        # Pull minimal columns for ALL matched spectra
        all_min_sql_tmpl = """
        SELECT spectrum_id_int, InChIKey_smiles, InChIKey_smiles_firstBlock
        FROM library_table
        WHERE spectrum_id_int IN ({ids})
        """

        df_all_minimal = _batched_fetch(
            all_min_sql_tmpl,
            all_spectrum_ids,
            fetch,
            chunk_size=500,
            paginate=True,
            page_size=500000,
            max_pages=None
        )

        # Pull metadata for ALL matched spectra
        lib_meta_sql_tmpl = """
        SELECT spectrum_id_int, spectrum_id, Compound_Name, Ion_Mode, collision_energy,
            Precursor_MZ, Adduct, msManufacturer, msMassAnalyzer, Smiles, GNPS_library_membership,
            representative_spectrum_int
        FROM library_table
        WHERE spectrum_id_int IN ({ids})
        """

        df_metadata = _batched_fetch(
            lib_meta_sql_tmpl,
            all_spectrum_ids,
            fetch,
            chunk_size=500,
            paginate=True,
            page_size=500000,
            max_pages=None
        )

        # Join minimal + metadata. If you want to carry any scores from Pass A to every spectrum of the same block,
        # first merge df_all_minimal with library_df_minimal on 'InChIKey_smiles_firstBlock'.
        df_final = df_all_minimal.merge(df_metadata, on='spectrum_id_int', how='left')


        if 'collision_energy' in df_final.columns:
            df_final['collision_energy'] = df_final['collision_energy'].apply(lambda x: str(x) if pd.notna(x) else 'unknown')
        if 'msMassAnalyzer' in df_final.columns:
            df_final['msMassAnalyzer'] = df_final['msMassAnalyzer'].apply(lambda x: str(x) if pd.notna(x) else '_unknown')

        df_final[['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = df_final[
            ['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')

        # drop fingerprint columns if they exist
        for col in ['fp_pattern', 'fp_morgan', 'fp_morgan_popcnt']:
            if col in df_final.columns:
                df_final.drop(columns=[col], inplace=True)
        
        df_final.rename(columns={'InChIKey_smiles_firstBlock': 'inchikey_first_block'}, inplace=True)
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
    
    # This query returns a per-MRI summary of the strongest MASST match.
    # Steps:
    #   1. "filtered": select all MASST hits whose spectrum_id_int is in {ids},
    #      keeping only those with cosine ≥ cosine_threshold and matching_peaks ≥ matching_peaks.
    #   2. "ranked": within each mri_id_int, rank the filtered hits by cosine (highest first)
    #      and assign a row number (rn).
    #   3. "uniq_counts": for each mri_id_int, count how many distinct spectrum_id_int
    #      values contributed to that MRI (unique_spectra_in_mri).
    #   4. Final SELECT: keep only the top-ranked hit per mri_id_int (rn = 1)
    #      and join in the unique_spectra_in_mri counts.
    #
    # Resulting table:
    #   → One row per mri_id_int (the best-cosine match for that dataset),
    #     including all MASST fields for that representative hit plus
    #     a column 'unique_spectra_in_mri' giving how many distinct spectra
    #     from the query molecule matched that MRI.
    masst_sql_tmpl = (
        "WITH filtered AS ("
        "  SELECT * "
        "  FROM masst_table "
        "  WHERE spectrum_id_int IN ({ids}) "
        f"    AND cosine >= {cosine_threshold} "
        f"    AND matching_peaks >= {matching_peaks}"
        "), ranked AS ("
        "  SELECT f.*, "
        "         ROW_NUMBER() OVER (PARTITION BY mri_id_int ORDER BY cosine DESC) AS rn "
        "  FROM filtered f"
        "), uniq_counts AS ("
        "  SELECT mri_id_int, "
        "         COUNT(DISTINCT spectrum_id_int) AS unique_spectra_in_mri "
        "  FROM filtered "
        "  GROUP BY mri_id_int"
        ") "
        "SELECT r.*, u.unique_spectra_in_mri "
        "FROM ranked r "
        "JOIN uniq_counts u USING (mri_id_int) "
        "WHERE r.rn = 1"
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

    #adjust datatypes
    masst_df['unique_spectra_in_mri'] = masst_df['unique_spectra_in_mri'].astype('int64')

    # — add spectrum_id strings —
    print(f"[STEP 3b] merging spectrum_id for {len(sids)} spectrum_id_ints")
    spec_map = library_df[['spectrum_id_int', 'query_spectrum_id', 'Adduct', 'Compound_Name', 'Precursor_MZ', 'inchikey_first_block', 'similar_library_spectra']].drop_duplicates()
    # make sure we match on same datatype
    spec_map['spectrum_id_int'] = spec_map['spectrum_id_int'].astype('Int64')
    masst_df['spectrum_id_int'] = masst_df['spectrum_id_int'].astype('Int64')

    masst_df = masst_df.merge(spec_map, on='spectrum_id_int', how='left')

    mids = masst_df['mri_id_int'].dropna().unique().tolist()

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
                          'ENVOMediumScaleIndex', 'classification', 'MS2spectra_count', 'InternalStandardsUsed', 'HumanPopulationDensity']
    redu_columns_list = [col for col in redu_columns_list if col not in columns_to_exclude]

    MAX_IDS_FOR_IN = 5000

    if len(mids) <= MAX_IDS_FOR_IN:
        redu_sql_tmpl = f"SELECT {', '.join(redu_columns_list)} FROM redu_table WHERE mri_id_int IN ({{ids}})"
        redu_df = _batched_fetch(
            redu_sql_tmpl,
            mids,
            fetch,
            chunk_size=500,   # adjust as needed
            paginate=True,
            page_size=500000,
            max_pages=None
        )
    else:
        print(f"[STEP 4] >{MAX_IDS_FOR_IN} mri_id_ints → full-table scan with pagination")
        redu_sql_all = f"SELECT {', '.join(redu_columns_list)} FROM redu_table"
        redu_df = _batched_fetch(
            redu_sql_all,
            None,             # table-mode: ignore IDs
            fetch,
            chunk_size=0,     # ignored in table-mode
            paginate=True,
            page_size=50000,
            max_pages=None
        )
        # Filter locally (ensure type alignment)
        mids_str = list(map(str, mids))
        redu_df = redu_df[redu_df['mri_id_int'].isin(mids_str)]


    return masst_df, redu_df