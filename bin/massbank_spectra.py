"""MassBank library spectra by accession, read from the MassBank data repository.

FASST fetches a query spectrum through the GNPS USI resolver, which answers
404 ("Unknown MONA USI") for mzspec:MASSBANK USIs. Every MassBank library
spectrum then fails on both FASST services, so these spectra are sent to FASST
by their peaks instead. Records are cached on disk (MASSBANK_CACHE_DIR, default
a folder in the system temp directory).
"""
import os
import re
import tempfile

import requests

RECORDS_URL = "https://raw.githubusercontent.com/MassBank/MassBank-data/main/{folder}/{accession}.txt"
CACHE_DIR = os.environ.get("MASSBANK_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "structuremasst_massbank")

# MSBNK-<contributor folder>-<record id>; nothing else may reach a URL or a file name
_ACCESSION = re.compile(r"^MSBNK-([A-Za-z0-9_]+)-([A-Za-z0-9_]+)$")
_PRECURSOR = re.compile(r"^MS\$FOCUSED_ION: PRECURSOR_M/Z\s+([0-9]+(?:\.[0-9]+)?)")


def accession_of(x):
    """The MassBank accession behind a library id or a mzspec:MASSBANK USI, else None."""
    x = str(x).strip()
    if x.startswith("mzspec:MASSBANK::accession:"):
        x = x.split("accession:", 1)[1]
    return x if _ACCESSION.match(x) else None


def parse_record(text):
    """{'precursor_mz', 'peaks'} from a MassBank record, or None without either."""
    precursor, peaks, in_peaks = None, [], False
    for line in text.splitlines():
        if in_peaks:
            parts = line.split()
            if line.startswith("//") or not line.startswith(" ") or len(parts) < 2:
                break
            try:
                peaks.append([float(parts[0]), float(parts[1])])
            except ValueError:
                break
        elif line.startswith("PK$PEAK:"):
            in_peaks = True
        elif precursor is None:
            m = _PRECURSOR.match(line)
            if m:
                precursor = float(m.group(1))
    if precursor is None or not peaks:
        return None
    return {"precursor_mz": precursor, "peaks": peaks}


def massbank_spectrum(x, timeout=30):
    """Precursor m/z and peaks of a MassBank library spectrum, or None if unavailable."""
    acc = accession_of(x)
    if acc is None:
        return None
    path = os.path.join(CACHE_DIR, f"{acc}.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return parse_record(f.read())
    except OSError:
        pass
    url = RECORDS_URL.format(folder=_ACCESSION.match(acc).group(1), accession=acc)
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(r.text)
        os.replace(tmp, path)
    except OSError:
        pass
    return parse_record(r.text)
