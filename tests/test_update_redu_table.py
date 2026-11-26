"""bin/update_redu_table.py on a small FASSTrecords database."""
import csv
import os
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEXED = [  # mri_table: files FASSTrecords indexed
    ("mzspec:MSV1:peak/a.mzML", 1),             # ReDU lists it under the same path
    ("mzspec:MSV1:ccms_peak/sub/b.mzML", 2),     # converted copy of peak/sub/b.mzXML
    ("mzspec:MSV1:ccms_peak/neg/c.mzML", 3),     # ReDU has pos/c and neg/c: the folder decides
    ("mzspec:MSV1:ccms_peak/other/d.mzML", 4),   # ReDU has pos/d and neg/d: undecidable, no link
]
DUMP = [  # USI, dataset, Country, a column the current table does not have
    ("mzspec:MSV1:peak/a.mzML", "MSV1", "Austria", "x"),
    ("mzspec:MSV1:peak/sub/b.mzXML", "MSV1", "Peru", "x"),
    ("mzspec:MSV1:pos/c.mzML", "MSV1", "Chile", "x"),
    ("mzspec:MSV1:neg/c.mzML", "MSV1", "Chile", "x"),
    ("mzspec:MSV1:pos/d.mzML", "MSV1", "Kenya", "x"),
    ("mzspec:MSV1:neg/d.mzML", "MSV1", "Kenya", "x"),
    ("mzspec:MSV1:peak/e.mzML", "MSV1", "Japan", "x"),  # not indexed at all
]


def setup(tmp_path):
    db = tmp_path / "masst_records.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE mri_table (mri TEXT, mri_id_int INTEGER)")
    con.executemany("INSERT INTO mri_table VALUES (?, ?)", INDEXED)
    con.execute('CREATE TABLE redu_table ("USI" TEXT, "ATTRIBUTE_DatasetAccession" TEXT, "Country" TEXT, "mri_id_int" INTEGER)')
    con.execute("INSERT INTO redu_table VALUES ('mzspec:MSV1:peak/a.mzML', 'MSV1', 'Austria', 1)")
    con.commit(); con.close()
    dump = tmp_path / "redu_dump.tsv"
    with open(dump, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["USI", "ATTRIBUTE_DatasetAccession", "Country", "NewColumn"])
        w.writerows(DUMP)
    return db, dump


def run(db, dump, *flags):
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    r = subprocess.run([sys.executable, "bin/update_redu_table.py", "--db", str(db), "--dump", str(dump), *flags],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def links(db):
    con = sqlite3.connect(db)
    out = dict(con.execute("SELECT USI, mri_id_int FROM redu_table"))
    cols = [r[1] for r in con.execute("PRAGMA table_info(redu_table)")]
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    return out, cols, tables


def test_links_exact_paths_and_copies(tmp_path):
    db, dump = setup(tmp_path)
    log = run(db, dump, "--keep-unmatched")
    out, cols, tables = links(db)
    assert out == {
        "mzspec:MSV1:peak/a.mzML": 1,
        "mzspec:MSV1:peak/sub/b.mzXML": 2,
        "mzspec:MSV1:pos/c.mzML": None,
        "mzspec:MSV1:neg/c.mzML": 3,
        "mzspec:MSV1:pos/d.mzML": None,
        "mzspec:MSV1:neg/d.mzML": None,
        "mzspec:MSV1:peak/e.mzML": None,
    }
    assert cols == ["USI", "ATTRIBUTE_DatasetAccession", "Country", "mri_id_int", "NewColumn"]
    assert f"redu_table_bak_{date.today():%Y%m%d}" in tables
    assert "path rule: 2 ReDU files" in log


def test_dry_run_writes_nothing(tmp_path):
    db, dump = setup(tmp_path)
    run(db, dump, "--keep-unmatched", "--dry-run")
    out, _, tables = links(db)
    assert out == {"mzspec:MSV1:peak/a.mzML": 1}
    assert not any(t.startswith("redu_table_bak_") or t.startswith("redu_table_new_") for t in tables)


def test_exact_paths_only(tmp_path):
    db, dump = setup(tmp_path)
    run(db, dump, "--no-path-rule")
    out, _, _ = links(db)
    assert out == {"mzspec:MSV1:peak/a.mzML": 1}  # without --keep-unmatched only linked rows stay
