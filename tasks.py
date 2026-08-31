import os
import importlib.util
import time

from celery import Celery
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.workflow_stepwise import retrieve_raw_data_matches, retrieve_raw_data_matches_from_peaks
import pandas as pd
import gc

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


@worker_init.connect
def preload_redu_table(**kwargs):
    """
    Preload the ReDU table once per Celery *parent process* before workers fork.
    This makes the table available to all worker subprocesses via copy-on-write.
    """

    config_path = "/app/config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    sqlite_path = config.PATH_TO_SQLITE
    api_endpoint = config.MASSTRECORDS_ENDPOINT
    timeout = config.MASSTRECORDS_TIMEOUT

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
    sql = "SELECT name FROM pragma_table_info('redu_table')"
    redu_columns = fetch(sql)
    redu_columns_list = redu_columns["name"].tolist()

    columns_to_exclude = [
        "filename", "TermsofPosition", "ComorbidityListDOIDIndex",
        "ENVOBroadScale", "ENVOLocalScale", "ENVOMediumScale", "qiita_sample_name",
        "UniqueSubjectID", "UBERONOntologyIndex", "DOIDOntologyIndex", "ENVOEnvironmentBiomeIndex",
        "ENVOEnvironmentMaterialIndex", "ENVOLocalScaleIndex", "ENVOBroadScaleIndex",
        "ENVOMediumScaleIndex", "classification", "MS2spectra_count",
        "InternalStandardsUsed", "HumanPopulationDensity"
    ]
    redu_columns_list = [col for col in redu_columns_list if col not in columns_to_exclude]

    print("[INIT] Preloading ReDU table for this worker...")
    get_redu_table_cached(fetch, redu_columns_list, sqlite_path)
    print("[INIT] Worker preload complete.")


@worker_init.connect
def load_molecule_classes(**kwargs):

    config_path = "/app/config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    sqlite_path = config.PATH_TO_SQLITE
    api_endpoint = config.MASSTRECORDS_ENDPOINT
    timeout = config.MASSTRECORDS_TIMEOUT

    fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)

    print("[INIT] Preloading molecule classes for this worker...")
    get_molecule_classes_cached(fetch)
    print("[INIT] Worker preload complete.")


# ---------------------------------------------------------------------------
# Celery toggle: read USE_CELERY from env var (Docker sets USE_CELERY=true),
# then fall back to config.py, then default to True.
# ---------------------------------------------------------------------------

_USE_CELERY_CACHE = None


def _use_celery() -> bool:
    global _USE_CELERY_CACHE
    if _USE_CELERY_CACHE is not None:
        return _USE_CELERY_CACHE

    env = os.environ.get("USE_CELERY", "").strip().lower()
    if env in ("1", "true", "yes"):
        _USE_CELERY_CACHE = True
        return True
    if env in ("0", "false", "no"):
        _USE_CELERY_CACHE = False
        return False

    for path in ("config.py", "/app/config.py"):
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("_cfg_uc", path)
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)
                result = bool(getattr(cfg, "USE_CELERY", True))
                _USE_CELERY_CACHE = result
                return result
            except Exception:
                pass

    _USE_CELERY_CACHE = True
    return True


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

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
    if not _use_celery():
        return get_library_table(
            smiles=smiles,
            searchtype=searchtype,
            tanimoto_threshold=tanimoto_threshold,
            allowed_formula=allowed_formula,
            allowed_elements=allowed_elements,
            sqlite_path=sqlite_path,
            api_endpoint=api_endpoint,
            timeout=timeout,
        )

    try:
        result = _run_get_library_table.delay(
            smiles, searchtype, tanimoto_threshold, allowed_formula, allowed_elements,
            sqlite_path, api_endpoint, timeout,
        )
        while True:
            if result.ready():
                break
            time.sleep(0.1)
        return pd.read_json(result.get())
    except Exception:
        return get_library_table(
            smiles=smiles,
            searchtype=searchtype,
            tanimoto_threshold=tanimoto_threshold,
            allowed_formula=allowed_formula,
            allowed_elements=allowed_elements,
            sqlite_path=sqlite_path,
            api_endpoint=api_endpoint,
            timeout=timeout,
        )


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
        timeout=timeout,
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
    if not _use_celery():
        return get_masst_and_redu_tables(
            df_for_name,
            cosine_threshold=cosine_threshold,
            matching_peaks=matching_peaks,
            min_annotation_rank=min_annotation_rank,
            sqlite_path=sqlite_path,
            api_endpoint=api_endpoint,
            timeout=timeout,
            chunk_size=chunk_size,
        )

    try:
        df_for_name_json = df_for_name.to_json(orient="records")
        result = _run_get_masst_and_redu_tables.delay(
            df_for_name_json, cosine_threshold, matching_peaks, min_annotation_rank,
            sqlite_path, api_endpoint, timeout, chunk_size,
        )
        while True:
            if result.ready():
                break
            time.sleep(0.1)
        result_dict = result.get()
        return pd.read_json(result_dict["masst"]), pd.read_json(result_dict["redu"])
    except Exception:
        return get_masst_and_redu_tables(
            df_for_name,
            cosine_threshold=cosine_threshold,
            matching_peaks=matching_peaks,
            min_annotation_rank=min_annotation_rank,
            sqlite_path=sqlite_path,
            api_endpoint=api_endpoint,
            timeout=timeout,
            chunk_size=chunk_size,
        )


@celery_app.task()
def _run_get_masst_and_redu_tables(
    df_for_name_json, cosine_threshold, matching_peaks, min_annotation_rank,
    sqlite_path, api_endpoint, timeout, chunk_size=200,
):
    df_for_name = pd.read_json(df_for_name_json)
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
    output_folder=None,
):
    if not _use_celery():
        _, redu_df = retrieve_raw_data_matches(
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
            output_folder=output_folder,
        )
        return redu_df

    try:
        df_for_name_json = df_for_name.to_json(orient="records")
        result = _run_retrieve_raw_data_matches.delay(
            df_for_name_json, database, precursor_mz_tol, fragment_mz_tol, min_cos,
            matching_peaks, analog, modimass, elimination, addition,
            modification_condition, sqlite_path, api_endpoint, timeout, output_folder,
        )
        while True:
            if result.ready():
                break
            time.sleep(0.1)
        result_dict = result.get()
        return pd.read_json(result_dict["redu"])
    except Exception:
        _, redu_df = retrieve_raw_data_matches(
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
            output_folder=output_folder,
        )
        return redu_df


@celery_app.task()
def _run_retrieve_raw_data_matches(
    df_for_name_json, database, precursor_mz_tol, fragment_mz_tol, min_cos,
    matching_peaks, analog, modimass, elimination, addition,
    modification_condition, sqlite_path, api_endpoint, timeout, output_folder=None,
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
        output_folder=output_folder,
    )
    return {"redu": redu_df.to_json(orient="records")}


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
    if not _use_celery():
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
        return redu_df

    import json
    spectra_json = json.dumps(spectra)

    try:
        result = _run_retrieve_raw_data_matches_from_peaks.delay(
            spectra_json, database, precursor_mz_tol, fragment_mz_tol, min_cos,
            matching_peaks, analog, modimass, elimination, addition,
            modification_condition, sqlite_path, api_endpoint, timeout, output_folder,
        )
        while True:
            if result.ready():
                break
            time.sleep(0.1)
        result_dict = result.get()
        return pd.read_json(result_dict["redu"])
    except Exception:
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
        return redu_df


@celery_app.task()
def _run_retrieve_raw_data_matches_from_peaks(
    spectra_json, database, precursor_mz_tol, fragment_mz_tol, min_cos,
    matching_peaks, analog, modimass, elimination, addition,
    modification_condition, sqlite_path, api_endpoint, timeout, output_folder=None,
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
