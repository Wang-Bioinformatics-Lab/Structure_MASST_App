# /app/bin/shared_data.py
import pandas as pd

# Global in-memory cache
_redu_df_cache = None


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
