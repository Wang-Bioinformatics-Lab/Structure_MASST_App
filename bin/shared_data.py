import pandas as pd

# Global in-memory cache
_redu_df_cache = None
_molecule_classes_cache = None

def get_redu_table_cached(fetch_func, redu_columns_list, sqlite_path, page_size=50000):
    """
    Loads the full redu_table once per Celery worker process
    and reuses it for all subsequent tasks.
    """
    global _redu_df_cache

    if _redu_df_cache is None:
        print("[INIT] Loading full ReDU table into memory (shared per worker)...")
        # 🔽 Move import here to avoid circular dependency
        from bin.run_masstRecords_queries import _batched_fetch

        redu_sql_all = f"SELECT {', '.join(redu_columns_list)} FROM redu_table"
        _redu_df_cache = _batched_fetch(
            redu_sql_all,
            None,
            fetch_func,
            chunk_size=0,
            paginate=True,
            page_size=page_size,
            max_pages=50000
        )
        print(f"[INIT] Loaded ReDU table with {_redu_df_cache.shape[0]:,} rows.")
    return _redu_df_cache

def clear_redu_cache():
    """Force the ReDU cache to reload on next access."""
    global _redu_df_cache
    _redu_df_cache = None
    print("[CACHE] Cleared ReDU table cache.")


def get_molecule_classes_cached(fetch_func, page_size=50000):
    """
    Loads molecule classes with unique molecule counts into memory once per Celery worker process.
    Reuses it for all subsequent tasks.
    """

    global _molecule_classes_cache

    if _molecule_classes_cache is None:
        print("[INIT] Loading molecule classes into memory (shared per worker)...")
        from bin.run_masstRecords_queries import _batched_fetch

        sql = """
            SELECT class_label, COUNT(DISTINCT InChIKey_smiles_firstBlock) AS unique_molecule_count
            FROM (
                SELECT classyfire_class AS class_label, InChIKey_smiles_firstBlock FROM library_table
            UNION ALL
            SELECT classyfire_subclass AS class_label, InChIKey_smiles_firstBlock FROM library_table
            UNION ALL
            SELECT classyfire_direct_parent AS class_label, InChIKey_smiles_firstBlock FROM library_table
        )
        WHERE class_label IS NOT NULL AND class_label != ''
        GROUP BY class_label
        ORDER BY unique_molecule_count DESC
        """
    
        _molecule_classes_cache  = _batched_fetch(
            sql,
            None,
            fetch_func,
            chunk_size=0,
            paginate=True,
            page_size=page_size,
            max_pages=50000
            )
        print(f"[INIT] Loaded {_molecule_classes_cache.shape[0]:,} molecule classes.")
    
    return _molecule_classes_cache

def clear_molecule_classes_cache():
    """Force the molecule classes cache to reload on next access."""
    global _molecule_classes_cache
    _molecule_classes_cache = None
    print("[CACHE] Cleared molecule classes cache.")