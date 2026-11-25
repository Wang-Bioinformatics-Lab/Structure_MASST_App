"""
Imported once in the multiprocessing forkserver (see bin/compute_worker.py).

Everything loaded here (heavy modules, the molecule-class list and, with
PRELOAD_REDU=1, the full ReDU table) is inherited copy-on-write by every worker
process forked from the forkserver - what the Celery worker_init hooks in
tasks.py did before.
"""

import os

import bin.compute_ops as ops  # noqa: F401  (also imports RDKit/pandas)
import bin.run_masstRecords_queries  # noqa: F401
import bin.workflow_stepwise  # noqa: F401

if os.environ.get("PRELOAD_REDU", "1") == "1":
    try:
        ops.preload_redu_table()
    except Exception as e:  # keep serving; the table loads lazily instead
        print(f"[INIT] ReDU preload failed, continuing without it: {e}", flush=True)

try:
    ops.preload_molecule_classes()
except Exception as e:
    print(f"[INIT] molecule class preload failed: {e}", flush=True)
