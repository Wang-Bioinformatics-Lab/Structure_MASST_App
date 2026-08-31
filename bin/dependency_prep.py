#!/usr/bin/env python3
"""
First-run data preparation for the tools StructureMASST hands off to.

Both LifeMASST and GeoMASST depend on reference data that is not part of a MASST
search. Some of it is committed to their repositories and only needs verifying;
the rest has to be fetched or derived once. This module knows which is which,
reports what a checkout is still missing, and runs the preparation - so the work
happens once, visibly, on the first LifeMASST or GeoMASST run, rather than
silently inside somebody's first query.

FASSTrecords is deliberately not covered here: that database is installed by
hand.

    python bin/dependency_prep.py --status
    python bin/dependency_prep.py lifemasst
    python bin/dependency_prep.py geomasst
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIFEMASST = PROJECT_ROOT / "external" / "LifeMASST"
GEOMASST = PROJECT_ROOT / "external" / "GeoMASST"

# Shared with the ordinary LifeMASST runs, so the conda environment built during
# preparation is the one the searches then reuse. Both are overridable: the conda
# cache alone runs to several GB, which does not belong on a small root volume.
NF_WORK_DIR = Path(os.environ.get("NF_WORK_DIR") or PROJECT_ROOT / "work" / "work")
NF_ENV_DIR = Path(os.environ.get("NF_CONDA_CACHE") or PROJECT_ROOT / "work" / "work_env")

TOOLS = ("lifemasst", "geomasst")

Log = Callable[[str], None]


class PrepareError(RuntimeError):
    """Preparation could not complete; the message says what to do about it."""


def _load_module(name: str, path: Path):
    """Import a file by path - these live in submodules, not on sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PrepareError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# LifeMASST
# ---------------------------------------------------------------------------

def _lifemasst_status() -> dict:
    manifest_path = LIFEMASST / "data" / "data_manifest.py"
    if not manifest_path.exists():
        return {
            "tool": "lifemasst",
            "available": False,
            "ready": False,
            "entries": [],
            "missing": [],
            "note": (
                f"{manifest_path} is not there. The LifeMASST submodule is not "
                "checked out - run `git submodule update --init --recursive`."
            ),
        }

    manifest = _load_module("lifemasst_data_manifest", manifest_path)
    report = manifest.status()
    entries = report["fetched"] + report["shipped"]
    return {
        "tool": "lifemasst",
        "available": True,
        "ready": report["ready"],
        "entries": entries,
        "missing": report["missing"],
        "missing_shipped": report["missing_shipped"],
        "note": (
            "Some prepared trees are missing from the checkout: "
            + ", ".join(report["missing_shipped"])
            + ". Those ship with the repository - reinitialise the submodule."
            if report["missing_shipped"] else ""
        ),
    }


def _prepare_lifemasst(log: Log, force: bool = False) -> dict:
    if not (LIFEMASST / "nf_workflow.nf").exists():
        raise PrepareError(
            "The LifeMASST submodule is not checked out. Run "
            "`git submodule update --init --recursive` first."
        )

    NF_WORK_DIR.mkdir(parents=True, exist_ok=True)
    NF_ENV_DIR.mkdir(parents=True, exist_ok=True)
    cfg = NF_WORK_DIR.parent / "nf_prepare.config"
    cfg.write_text(
        f'workDir = "{NF_WORK_DIR}"\n'
        f'conda {{\n'
        f'    enabled = true\n'
        f'    cacheDir = "{NF_ENV_DIR}"\n'
        f'}}\n'
    )

    # -resume so a retry after a network hiccup or a half-built conda env picks
    # up where it stopped instead of refetching everything
    cmd = ["nextflow", "run", str(LIFEMASST / "nf_workflow.nf"),
           "-entry", "prepare", "-c", str(cfg), "-resume"]

    env = os.environ.copy()
    # get_data.py reads ReDU straight out of the sqlite. Pass on whatever the app
    # resolved, so a local checkout does not go looking for the in-container path.
    if "PATH_TO_SQLITE" not in env:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from config import PATH_TO_SQLITE
            env["PATH_TO_SQLITE"] = str(PATH_TO_SQLITE)
        except Exception:
            pass

    if shutil.which("nextflow") is None:
        raise PrepareError(
            "nextflow is not on PATH, and LifeMASST's preparation runs through it. "
            "Install it, or run the steps by hand - see "
            "external/LifeMASST/data/README.md."
        )

    log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(LIFEMASST), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:                                   # type: ignore[union-attr]
        log(line.rstrip("\n"))
    if proc.wait() != 0:
        raise PrepareError(
            f"LifeMASST preparation failed (nextflow exit {proc.returncode}). "
            "The log above says which step."
        )
    return _lifemasst_status()


# ---------------------------------------------------------------------------
# GeoMASST
# ---------------------------------------------------------------------------

def _geomasst_module():
    prepare_path = GEOMASST / "geomasst" / "prepare.py"
    if not prepare_path.exists():
        return None
    # loaded by path so a broken/minimal environment cannot drag pandas in
    return _load_module("geomasst_prepare", prepare_path)


def _geomasst_status() -> dict:
    module = _geomasst_module()
    if module is None:
        return {
            "tool": "geomasst",
            "available": False,
            "ready": False,
            "entries": [],
            "missing": [],
            "note": (
                "The GeoMASST submodule is not checked out - run "
                "`git submodule update --init --recursive`. It is a private "
                "repository, so this needs credentials."
            ),
        }
    report = module.status()
    return {
        "tool": "geomasst",
        "available": True,
        "ready": report["ready"],
        "entries": report["entries"],
        "missing": report["broken"],
        "missing_shipped": report["broken"],
        "note": ("GeoMASST downloads nothing on first run - every asset is "
                 "committed to the repository."),
    }


def _prepare_geomasst(log: Log, force: bool = False) -> dict:
    module = _geomasst_module()
    if module is None:
        raise PrepareError(
            "The GeoMASST submodule is not checked out. Run "
            "`git submodule update --init --recursive` first."
        )
    module.prepare(rebuild=force, log=log)
    return _geomasst_status()


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------

_STATUS = {"lifemasst": _lifemasst_status, "geomasst": _geomasst_status}
_PREPARE = {"lifemasst": _prepare_lifemasst, "geomasst": _prepare_geomasst}

LABELS = {"lifemasst": "LifeMASST", "geomasst": "GeoMASST"}


def status(tool: str) -> dict:
    """What this checkout has, and what it still needs."""
    if tool not in _STATUS:
        raise KeyError(f"Unknown tool {tool!r}; expected one of {TOOLS}")
    return _STATUS[tool]()


def is_ready(tool: str) -> bool:
    return bool(status(tool)["ready"])


def prepare(tool: str, log: Log = print, force: bool = False) -> dict:
    """Fetch and derive whatever is missing. Safe to call when nothing is."""
    if tool not in _PREPARE:
        raise KeyError(f"Unknown tool {tool!r}; expected one of {TOOLS}")
    return _PREPARE[tool](log, force)


def ensure_ready(tool: str, log: Log = print, force: bool = False) -> dict:
    """Prepare only if needed - the call to put in front of a Run button."""
    current = status(tool)
    if current["ready"] and not force:
        return current
    return prepare(tool, log=log, force=force)


def download_estimate(tool: str) -> float:
    """Approximate MB still to fetch, for telling the user what they are in for."""
    report = status(tool)
    return float(sum(
        e.get("approx_mb", 0) for e in report["entries"]
        if not e.get("present") and not e.get("shipped")
    ))


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", nargs="?", choices=TOOLS, help="what to prepare")
    ap.add_argument("--status", action="store_true", help="report without preparing")
    ap.add_argument("--force", action="store_true",
                    help="prepare even when it already looks complete "
                         "(GeoMASST: refetch the assets from upstream)")
    args = ap.parse_args()

    targets = [args.tool] if args.tool else list(TOOLS)

    if args.status or not args.tool:
        for tool in targets:
            report = status(tool)
            print(f"\n{LABELS[tool]}: {'ready' if report['ready'] else 'NOT ready'}")
            if report["note"]:
                print(f"  {report['note']}")
            for entry in report["entries"]:
                mark = "ok " if entry.get("present") else "-- "
                origin = "git" if entry.get("shipped") else entry.get("source", "")
                print(f"  {mark} {entry.get('label', entry['key']):<36} {origin}")
            pending = download_estimate(tool)
            if pending:
                print(f"  first run will fetch about {pending:.0f} MB")
        return 0

    try:
        prepare(args.tool, force=args.force)
    except PrepareError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
