# tasks.py
import os
import pandas as pd
import gc
from celery import Celery
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches


# ---------------------------------------------------------------------
# 🟢 Detect environment: if Redis not reachable, run tasks inline (eager)
# ---------------------------------------------------------------------
USE_CELERY = os.getenv("USE_CELERY", "0") == "1"

# Choose broker depending on context (Docker vs local)
if USE_CELERY:
    # Default to the Docker Redis hostname, fallback to localhost
    broker_url = os.getenv(
        "CELERY_BROKER_URL",
        "redis://structure-masst-redis:6379/0"
    )
    result_backend = os.getenv(
        "CELERY_RESULT_BACKEND",
        broker_url
    )
else:
    # Local or debug mode → run everything inline
    broker_url = "memory://"
    result_backend = "cache+memory://"

celery_app = Celery("structuremasst_tasks", broker=broker_url, backend=result_backend)

celery_app.conf.update(
    result_expires=900,
    task_always_eager=not USE_CELERY,   # key: local mode executes inline
    task_eager_propagates=True,         # exceptions bubble up to Streamlit
    task_store_eager_result=True        # so .get() still works even without Celery
)

USING_CELERY = USE_CELERY
# ---------------------------------------------------------------------


@celery_app.task()
def heartbeat_task():
    return "Structure MASST worker is alive (mode: {})".format(
        "Celery" if USING_CELERY else "local"
    )



@celery_app.task()
def run_get_library_table(smiles, searchtype, tanimoto_threshold, sqlite_path, api_endpoint, timeout):
    df = get_library_table(
        smiles=smiles,
        searchtype=searchtype,
        tanimoto_threshold=tanimoto_threshold,
        sqlite_path=sqlite_path,
        api_endpoint=api_endpoint,
        timeout=timeout
    )
    return df.to_json(orient="records")


@celery_app.task()
def run_get_masst_and_redu_tables(
    df_for_name_json,
    cosine_threshold,
    matching_peaks,
    sqlite_path,
    api_endpoint,
    timeout,
    chunk_size=200,
):
    df_for_name = pd.read_json(df_for_name_json)
    masst_df, redu_df = get_masst_and_redu_tables(
        df_for_name,
        cosine_threshold=cosine_threshold,
        matching_peaks=matching_peaks,
        sqlite_path=sqlite_path,
        api_endpoint=api_endpoint,
        timeout=timeout,
        chunk_size=chunk_size,
    )
    return {
        "masst": masst_df.to_json(orient="records"),
        "redu": redu_df.to_json(orient="records"),
    }


@celery_app.task()
def run_retrieve_raw_data_matches(
    df_for_name_json,
    database,
    precursor_mz_tol,
    fragment_mz_tol,
    min_cos,
    matching_peaks,
    analog,
    modimass,
    elimination,
    addition,
    modification_condition,
    sqlite_path,
    api_endpoint,
    timeout,
):
    df_for_name = pd.read_json(df_for_name_json)
    masst_df, redu_df = retrieve_raw_data_matches(
        df_for_name,
        database=database,
        precursor_mz_tol=precursor_mz_tol,
        fragment_mz_tol=fragment_mz_tol,
        min_cos=min_cos,
        matching_peaks=matching_peaks,
        analog=analog,
        modimass=modimass,
        elimination=elimination,
        addition=addition,
        modification_condition=modification_condition,
        sqlite_path=sqlite_path,
        api_endpoint=api_endpoint,
        timeout=timeout,
    )
    return {"redu": redu_df.to_json(orient="records")}
