# tasks.py
from celery import Celery
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches
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

@celery_app.task()
def heartbeat_task():
    return "Structure MASST worker is alive."

def run_get_library_table(
    smiles,
    searchtype,
    tanimoto_threshold,
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
            sqlite_path,
            api_endpoint,
            timeout
        )

        df_library_structurematch = pd.read_json(result)

        return df_library_structurematch


@celery_app.task()
def _run_get_library_table(smiles, searchtype, tanimoto_threshold, sqlite_path, api_endpoint, timeout):
    df = get_library_table(
        smiles=smiles,
        searchtype=searchtype,
        tanimoto_threshold=tanimoto_threshold,
        sqlite_path=sqlite_path,
        api_endpoint=api_endpoint,
        timeout=timeout
    )
    return df.to_json(orient="records")


def run_get_masst_and_redu_tables(
    df_for_name,
    cosine_threshold,
    matching_peaks,
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
    timeout
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
            timeout
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
            timeout
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

