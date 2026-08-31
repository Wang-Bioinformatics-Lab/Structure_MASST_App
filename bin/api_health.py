#!/usr/bin/env python3
import os
import time
import requests
import sys

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG_PATH = os.path.join(REPO_ROOT, "external", "GNPSDataPackage")
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gnpsdata import fasst

# single source of truth, shared with the query path in bin/run_fasst.py
from bin.run_fasst import FASST_NONTERMINAL_STATUS

FASST_API_SERVER_URL = os.environ.get("FASST_API_SERVER_URL", "https://api.fasst.gnps2.org")


def get_results(task_id, host="https://api.fasst.gnps2.org", retries_max=120, sleep_s=1, not_found_grace_tries=15):
    """
    Minimal poller:
      - returns dict once results are ready
      - treats any non-terminal status (PENDING, RUNNING, ...) as pending
      - treats NOT_FOUND as pending for a few tries (eventual consistency), then errors
      - treats payloads with 'results' as ready even if 'status' is missing
    """
    url = f"{host.rstrip('/')}/search/result/{task_id}"

    for i in range(retries_max):
        print("WAITING FOR RESULTS", i, task_id)

        try:
            r = requests.get(url, timeout=30)
        except KeyboardInterrupt:
            raise
        except Exception:
            time.sleep(sleep_s)
            continue

        if not r.ok:
            time.sleep(sleep_s)
            continue

        try:
            j = r.json()
        except Exception:
            time.sleep(sleep_s)
            continue

        # Some responses omit "status" but contain "results" when ready
        if "results" in j:
            return j

        status = j.get("status")

        # The API reports RUNNING (not PENDING) for an in-flight task. Falling through
        # to the "return as-is" branch below handed the caller a bare
        # {"status", "message"} payload, which counted as zero results and made the
        # health check declare the whole FASST API down.
        if status in FASST_NONTERMINAL_STATUS:
            time.sleep(sleep_s)
            continue

        if status == "NOT_FOUND":
            # Often transient right after submit; don't fail instantly
            if i < not_found_grace_tries:
                time.sleep(sleep_s)
                continue
            raise Exception(f"Invalid Task ID or host mismatch: {j}")

        # Any other status (including errors) -> return as-is so caller can print it
        return j

    raise Exception("Timeout waiting for results from FASST API")


def test_fasst_api_search_nonblocking():
    usi = "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005435899"
    database = "metabolomicspanrepo_index_nightly"

    print("submitted", 0)

    # IMPORTANT: blocking=False so we always get a task id back
    submit = fasst.query_fasst_api_usi(
        usi,
        database,
        host=FASST_API_SERVER_URL,
        analog=False,
        precursor_mz_tol=0.05,
        fragment_mz_tol=0.05,
        min_cos=0.7,
        blocking=False,
    )

    task_id = submit.get("task_id") or submit.get("id")
    if not task_id:
        raise Exception(f"No task_id returned from submission: {submit}")

    results = get_results(task_id, host=FASST_API_SERVER_URL)

    print(results.keys())
    print(results.get("status"))
    print(results.get("error"))

    response_list = results.get("results", []) or []
    return len(response_list)


if __name__ == "__main__":
    n = test_fasst_api_search_nonblocking()
    print(n)