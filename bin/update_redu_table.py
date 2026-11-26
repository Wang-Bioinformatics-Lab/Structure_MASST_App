#!/usr/bin/env python3
"""Replace redu_table in the FASSTrecords database (masst_records.sqlite) with a fresh ReDU dump.

A manual maintenance tool: neither the app nor any workflow runs it. The database path comes
from config.py (PATH_TO_SQLITE, which the environment variable of that name overrides), or --db.
Get a dump with:  curl -L -o redu_dump.tsv https://redu.gnps2.org/dump

mri_id_int, the link from a ReDU row to the file FASSTrecords indexed, is not in the dump. It is
derived here: an exact match between the dump's USI and mri_table.mri, and for files indexed under
another path than ReDU lists them (FASST indexes converted copies - ccms_peak/....mzML,
gnps2_converted/... - of files ReDU lists as peak/....mzXML, raw/..., updates/...) the rule FASST
mode uses (align_files_to_redu in bin/workflow_stepwise.py): same dataset and file name, copies
count as one file, and where a name occurs in several folders (pos/neg, rp/hilic) the folders decide
or nothing is linked. Only ReDU rows without an exact link get one this way; --no-path-rule turns
it off.

Safe by construction:
  * the existing table is renamed to redu_table_bak_<date> rather than dropped, so the change is reversible;
  * the new table is built and filled first, and the rename plus swap happen in one transaction;
  * the column order of the old table is preserved, columns the dump no longer has are kept as NULL, and columns the
    dump has gained are appended at the end, so existing queries keep working;
  * it refuses to swap when the new table would link less than half as many rows as the current one;
  * --dry-run reports everything (column differences, match rate, row counts) and writes nothing.

Usage (from the repository root, in the StructureMASST environment):
  python bin/update_redu_table.py --dump redu_dump.tsv --keep-unmatched --drop-empty [--dry-run] [--db ...]
--keep-unmatched keeps the ReDU rows of files FASSTrecords has not indexed (FASST mode needs them).
"""
import argparse, csv, os, sqlite3, sys, time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (PATH_TO_SQLITE)

ap = argparse.ArgumentParser(description="Replace redu_table in the FASSTrecords database with a fresh ReDU dump.")
ap.add_argument("--dump", required=True, type=Path, help="ReDU dump (TSV), e.g. from https://redu.gnps2.org/dump")
ap.add_argument("--db", type=Path, default=Path(config.PATH_TO_SQLITE),
                help=f"FASSTrecords database (default from config.py: {config.PATH_TO_SQLITE})")
ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
ap.add_argument("--keep-unmatched", action="store_true",
                help="also keep dump rows whose USI is not indexed in mri_table (mri_id_int NULL)")
ap.add_argument("--batch", type=int, default=20000)
ap.add_argument("--drop-empty", action="store_true",
                help="drop columns that are empty or \"missing value\" in every kept row; columns of the current "
                     "table are kept regardless, so existing queries cannot break")
ap.add_argument("--no-path-rule", action="store_true",
                help="link exact paths only (no linking of copies indexed under another path)")
a = ap.parse_args()
if not a.db.is_file():
    sys.exit(f"no database at {a.db} (set PATH_TO_SQLITE or pass --db)")
if not a.dump.is_file():
    sys.exit(f"no dump at {a.dump}")
T0 = time.time(); log = lambda m: print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)
csv.field_size_limit(10_000_000)

con = sqlite3.connect(a.db, timeout=120)
old_cols = [r[1] for r in con.execute("PRAGMA table_info(redu_table)")]
if not old_cols:
    sys.exit("redu_table not found in the database")
old_n, old_linked = con.execute("SELECT count(*), count(mri_id_int) FROM redu_table").fetchone()
log(f"current redu_table: {old_n:,} rows ({old_linked:,} linked to an indexed file), {len(old_cols)} columns")

with open(a.dump, newline="", encoding="utf-8", errors="replace") as f:
    head = next(csv.reader(f, delimiter="\t"))
dump_cols = [c.strip() for c in head]
if "USI" not in dump_cols:
    sys.exit("the dump has no USI column - cannot derive mri_id_int")
old_data = [c for c in old_cols if c != "mri_id_int"]
missing = [c for c in old_data if c not in dump_cols]          # in the old table, gone from the dump
added = [c for c in dump_cols if c not in old_data]            # new in the dump
log(f"dump: {len(dump_cols)} columns" + (f"; no longer present: {missing}" if missing else "")
    + (f"; new: {added}" if added else ""))

new_cols = old_cols + [c for c in added]                        # old order first, then anything new
usi_i = dump_cols.index("USI")
EMPTY = {"", "missing value", "N/A", "na", "NA", "nan", "None", "null"}
src_idx = {c: (dump_cols.index(c) if c in dump_cols else None) for c in new_cols}

def kept_rows():
    """Dump rows that belong in the new table, as lists aligned to dump_cols."""
    with open(a.dump, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f, delimiter="\t"); next(rd)
        for rec in rd:
            if len(rec) < len(dump_cols): rec = rec + [""] * (len(dump_cols) - len(rec))
            mid = file_id(rec[usi_i])
            if mid is None and not a.keep_unmatched: continue
            yield rec, mid

log("reading mri_table ...")
mri = dict(con.execute("SELECT mri, mri_id_int FROM mri_table"))
log(f"{len(mri):,} indexed files")

# ReDU files without an exact link -> an indexed copy under another path
by_rule = {}
if not a.no_path_rule:
    import pandas as pd
    from bin.workflow_stepwise import align_files_to_redu
    with open(a.dump, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f, delimiter="\t"); next(rd)
        dump_usis = pd.Series([rec[usi_i] for rec in rd if len(rec) > usi_i])
    files = pd.Series(sorted(mri))
    tied = align_files_to_redu(files, dump_usis)  # sees every file and every ReDU row, so rivals count
    unlinked = set(dump_usis[~dump_usis.isin(files)])
    for f, r in zip(files, tied):
        if r != f and r in unlinked and r not in by_rule:  # first copy in path order, deterministic
            by_rule[r] = mri[f]
    log(f"path rule: {len(by_rule):,} ReDU files without an exact link linked to a copy indexed under another path")

def file_id(usi):
    mid = mri.get(usi)
    return mid if mid is not None else by_rule.get(usi)

if a.drop_empty:                                                # first pass: which columns carry any real value
    log("scanning for empty columns ...")
    filled = [False] * len(dump_cols)
    n = 0
    for rec, _ in kept_rows():
        n += 1
        for j, v in enumerate(rec):
            if not filled[j] and v.strip() not in EMPTY: filled[j] = True
    empty_new = [c for c in added if not filled[dump_cols.index(c)]]
    empty_old = [c for c in old_data if c in dump_cols and not filled[dump_cols.index(c)]]
    new_cols = [c for c in new_cols if c not in empty_new]
    log(f"{n:,} kept rows scanned; dropping {len(empty_new)} empty new columns: {empty_new}")
    if empty_old:
        log(f"empty but kept because the current table has them ({len(empty_old)}): {empty_old}")
    src_idx = {c: (dump_cols.index(c) if c in dump_cols else None) for c in new_cols}

rows, total, matched = [], 0, 0
tmp = "redu_table_new_" + date.today().strftime("%Y%m%d")
if not a.dry_run:
    con.execute(f'DROP TABLE IF EXISTS "{tmp}"')
    con.execute(f'CREATE TABLE "{tmp}" (' + ", ".join(f'"{c}" {"INTEGER" if c == "mri_id_int" else "TEXT"}'
                                                      for c in new_cols) + ")")
ins = f'INSERT INTO "{tmp}" VALUES (' + ",".join("?" * len(new_cols)) + ")"

with open(a.dump, newline="", encoding="utf-8", errors="replace") as f:
    rd = csv.reader(f, delimiter="\t"); next(rd)
    for rec in rd:
        total += 1
        if len(rec) < len(dump_cols): rec = rec + [""] * (len(dump_cols) - len(rec))
        mid = file_id(rec[usi_i])
        if mid is None and not a.keep_unmatched: continue
        matched += mid is not None
        rows.append([mid if c == "mri_id_int" else (rec[src_idx[c]] if src_idx[c] is not None else None)
                     for c in new_cols])
        if len(rows) >= a.batch and not a.dry_run:
            con.executemany(ins, rows); rows.clear()
        if total % 200000 == 0: log(f"  {total:,} dump rows, {matched:,} matched")
if rows and not a.dry_run:
    con.executemany(ins, rows)
kept = matched if not a.keep_unmatched else total
log(f"dump rows {total:,}; matched to an indexed file {matched:,} ({100*matched/max(total,1):.1f}%); "
    f"rows for the new table {kept:,} (old table had {old_n:,})")

if a.dry_run:
    log("dry run - nothing written"); con.close(); sys.exit()
# compare linked with linked: with --keep-unmatched the current table also holds rows without a link
if matched < 0.5 * old_linked:
    con.execute(f'DROP TABLE "{tmp}"'); con.commit()
    sys.exit(f"refusing to swap: only {matched:,} linked rows against {old_linked:,} in the current table")

bak = "redu_table_bak_" + date.today().strftime("%Y%m%d")
con.commit()                      # close the implicit transaction the inserts opened
con.execute("BEGIN IMMEDIATE")
con.execute(f'DROP TABLE IF EXISTS "{bak}"')
con.execute(f'ALTER TABLE redu_table RENAME TO "{bak}"')
con.execute(f'ALTER TABLE "{tmp}" RENAME TO redu_table')
con.commit()
# a date-stamped name: the old index name travels with the renamed backup table, so IF NOT EXISTS would skip it
con.execute(f'CREATE INDEX IF NOT EXISTS "idx_redu_mri_id_int_{date.today():%Y%m%d}" ON redu_table(mri_id_int)')
con.commit()
log(f"swapped: redu_table is the new dump, previous table kept as {bak}")
n, nid = con.execute("SELECT count(*), count(mri_id_int) FROM redu_table").fetchone()
log(f"redu_table now: {n:,} rows, {nid:,} with mri_id_int")
con.close()
