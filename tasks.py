# tasks.py
from turtle import st

from celery import Celery
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches, retrieve_raw_data_matches_from_peaks
import pandas as pd
import gc
import time
import redis

# Connect to Redis (broker) and store results back in Redis
celery_app = Celery(
    "structuremasst_tasks",
    broker="redis://structure-masst-redis",
    backend="redis://structure-masst-redis"
)

celery_app.conf.update(
    result_expires=900
)


from celery.signals import worker_init
from bin.shared_data import get_redu_table_cached, get_molecule_classes_cached
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
    

@worker_init.connect
def load_molecule_classes(**kwargs):

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

    # 🔹 Preload molecule classes cache once per worker
    print("[INIT] Preloading molecule classes for this worker...")
    get_molecule_classes_cached(fetch)
    print("[INIT] Worker preload complete.")
    


@celery_app.task()
def heartbeat_task():
    return "Structure MASST worker is alive."

def run_get_library_table(
    smiles,
    searchtype,
    tanimoto_threshold,
    allowed_formula,
    allowed_elements,
    sqlite_path,
    api_endpoint,
    timeout
):
    
    try:
        # Call the heavy function
        result = _run_get_library_table.delay(
            smiles,
            searchtype,
            tanimoto_threshold,
            allowed_formula,
            allowed_elements,
            sqlite_path,
            api_endpoint,
            timeout
        )

        
        # Waiting
        while(1):
            if result.ready():
                break
            time.sleep(0.1)

        df_library_structurematch = pd.read_json(result.get())

        return df_library_structurematch

    except:
        # falling back on calling the actual function directly
        result = _run_get_library_table(
            smiles,
            searchtype,
            tanimoto_threshold,
            allowed_formula,
            allowed_elements,
            sqlite_path,
            api_endpoint,
            timeout
        )

        df_library_structurematch = pd.read_json(result)

        return df_library_structurematch


@celery_app.task()
def _run_get_library_table(smiles, searchtype, tanimoto_threshold, allowed_formula, allowed_elements, sqlite_path, api_endpoint, timeout):
    df = get_library_table(
        smiles=smiles,
        searchtype=searchtype,
        tanimoto_threshold=tanimoto_threshold,
        allowed_formula=allowed_formula,
        allowed_elements=allowed_elements,
        sqlite_path=sqlite_path,
        api_endpoint=api_endpoint,
        timeout=timeout
    )
    return df.to_json(orient="records")


def run_get_masst_and_redu_tables(
    df_for_name,
    cosine_threshold,
    matching_peaks,
    min_annotation_rank,
    sqlite_path,
    api_endpoint,
    timeout,
    chunk_size=200,
):
    try:
        # Serialize the DataFrame to JSON
        df_for_name_json = df_for_name.to_json(orient="records")

        # Call the heavy function asynchronously
        result = _run_get_masst_and_redu_tables.delay(
            df_for_name_json,
            cosine_threshold,
            matching_peaks,
            min_annotation_rank,
            sqlite_path,
            api_endpoint,
            timeout,
            chunk_size,
        )

        # Waiting
        while(1):
            if result.ready():
                break
            time.sleep(0.1)

        result_dict = result.get()

        masst_df = pd.read_json(result_dict["masst"])
        redu_df = pd.read_json(result_dict["redu"])

        return masst_df, redu_df

    except:
        # falling back on calling the actual function directly
        result_dict = _run_get_masst_and_redu_tables(
            df_for_name.to_json(orient="records"),
            cosine_threshold,
            matching_peaks,
            min_annotation_rank,
            sqlite_path,
            api_endpoint,
            timeout,
            chunk_size,
        )

        masst_df = pd.read_json(result_dict["masst"])
        redu_df = pd.read_json(result_dict["redu"])

        return masst_df, redu_df

@celery_app.task()
def _run_get_masst_and_redu_tables(
    df_for_name_json,
    cosine_threshold,
    matching_peaks,
    min_annotation_rank,
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
        min_annotation_rank=min_annotation_rank,
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


def run_retrieve_raw_data_matches(
    df_for_name,
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
    output_folder=None
):
    try:
        # Serialize the DataFrame to JSON
        df_for_name_json = df_for_name.to_json(orient="records")

        # Call the heavy function asynchronously
        result = _run_retrieve_raw_data_matches.delay(
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
            output_folder,
        )

        # Waiting
        while(1):
            if result.ready():
                break
            time.sleep(0.1)

        result_dict = result.get()

        redu_df = pd.read_json(result_dict["redu"])

        return redu_df

    except:
        # falling back on calling the actual function directly
        result_dict = _run_retrieve_raw_data_matches(
            df_for_name.to_json(orient="records"),
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
            output_folder,
        )

        redu_df = pd.read_json(result_dict["redu"])

        return redu_df

@celery_app.task()
def _run_retrieve_raw_data_matches(
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
    output_folder=None
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
        output_folder=output_folder
    )

    # Return both DataFrames serialized as JSON
    return {
        "redu": redu_df.to_json(orient="records"),
    }


def run_retrieve_raw_data_matches_from_peaks(
    spectra: list,
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
    output_folder=None,
):
    """Celery-aware wrapper for retrieve_raw_data_matches_from_peaks.

    Falls back to a direct call if Celery is unavailable (same pattern as the USI variant).
    """
    import json

    spectra_json = json.dumps(spectra)

    try:
        result = _run_retrieve_raw_data_matches_from_peaks.delay(
            spectra_json,
            database, precursor_mz_tol, fragment_mz_tol, min_cos, matching_peaks,
            analog, modimass, elimination, addition, modification_condition,
            sqlite_path, api_endpoint, timeout, output_folder,
        )

        while True:
            if result.ready():
                break
            time.sleep(0.1)

        result_dict = result.get()
        return pd.read_json(result_dict["redu"])

    except Exception:
        result_dict = _run_retrieve_raw_data_matches_from_peaks(
            spectra_json,
            database, precursor_mz_tol, fragment_mz_tol, min_cos, matching_peaks,
            analog, modimass, elimination, addition, modification_condition,
            sqlite_path, api_endpoint, timeout, output_folder,
        )
        return pd.read_json(result_dict["redu"])


@celery_app.task()
def _run_retrieve_raw_data_matches_from_peaks(
    spectra_json,
    database, precursor_mz_tol, fragment_mz_tol, min_cos, matching_peaks,
    analog, modimass, elimination, addition, modification_condition,
    sqlite_path, api_endpoint, timeout, output_folder=None,
):
    import json

    spectra = json.loads(spectra_json)

    _, redu_df = retrieve_raw_data_matches_from_peaks(
        spectra,
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
        output_folder=output_folder,
    )

    return {"redu": redu_df.to_json(orient="records")}

