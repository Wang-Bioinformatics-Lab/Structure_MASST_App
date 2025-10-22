# tasks.py
from celery import Celery
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches
import pandas as pd
import gc

# Connect to Redis (broker) and store results back in Redis
celery_app = Celery(
    "structuremasst_tasks",
    broker="redis://structure-masst-redis",
    backend="redis://structure-masst-redis"
)

celery_app.conf.update(
    result_expires=900,              # 0.25 hour expiration for results (prevents Redis bloat)
)


from celery.signals import worker_init
from bin.shared_data import get_redu_table_cached
from bin.run_masstRecords_queries import _get_fetcher
import importlib.util


@worker_init.connect
def preload_redu_table(**kwargs):
    """
    Preload the ReDU table once per Celery *parent process* before workers fork.
    This makes the table available to all worker subprocesses via copy-on-write.
    """

    # 🔹 Load config.py dynamically (Celery runs in its own context)
    config_path = "/app/config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    sqlite_path = config.PATH_TO_SQLITE
    api_endpoint = config.MASSTRECORDS_ENDPOINT
    timeout = config.MASSTRECORDS_TIMEOUT

    # 🔹 Create the fetcher for SQLite or Datasette
    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    # 🔹 Dynamically detect ReDU columns (excluding heavy/unused ones)
    sql = "SELECT name FROM pragma_table_info('redu_table')"
    redu_columns = fetch(sql)
    redu_columns_list = redu_columns["name"].tolist()

    columns_to_exclude = [
        "filename", "TermsofPosition", "ComorbidityListDOIDIndex", "SampleCollectionDateandTime",
        "ENVOBroadScale", "ENVOLocalScale", "ENVOMediumScale", "qiita_sample_name",
        "UniqueSubjectID", "UBERONOntologyIndex", "DOIDOntologyIndex", "ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex", "ENVOLocalScaleIndex", "ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex", "classification", "MS2spectra_count",
        "InternalStandardsUsed", "HumanPopulationDensity"
    ]
    redu_columns_list = [col for col in redu_columns_list if col not in columns_to_exclude]

    # 🔹 Preload ReDU cache once per worker
    print("[INIT] Preloading ReDU table for this worker...")
    get_redu_table_cached(fetch, redu_columns_list, sqlite_path)
    print("[INIT] Worker preload complete.")


@celery_app.task()
def heartbeat_task():
    return "Structure MASST worker is alive."

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
    # Rebuild the DataFrame from JSON
    df_for_name = pd.read_json(df_for_name_json)

    # Run the actual heavy function
    masst_df, redu_df = get_masst_and_redu_tables(
        df_for_name,
        cosine_threshold=cosine_threshold,
        matching_peaks=matching_peaks,
        sqlite_path=sqlite_path,
        api_endpoint=api_endpoint,
        timeout=timeout,
        chunk_size=chunk_size,
    )

    # Return both results as JSON strings
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
    timeout
):


    # Rebuild the input DataFrame
    df_for_name = pd.read_json(df_for_name_json)

    # Call the heavy function
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

    # Return both DataFrames serialized as JSON
    return {
        "redu": redu_df.to_json(orient="records"),
    }

