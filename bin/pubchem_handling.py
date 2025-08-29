# pubchem_utils.py
"""
Minimal helpers for PubChem fuzzy name lookup → Canonical SMILES.
"""

import time
from urllib.parse import quote
from typing import List, Optional

import requests

DEFAULT_TIMEOUT = 15
DEFAULT_TRIES = 3
DEFAULT_BACKOFF = 0.7


def _get(
    url: str,
    params: dict | None = None,
    *,
    tries: int = DEFAULT_TRIES,
    backoff: float = DEFAULT_BACKOFF,
    timeout: int = DEFAULT_TIMEOUT,
    accept_json: bool = False,
) -> requests.Response:
    """GET with retries/backoff and sane headers."""
    headers = {
        "User-Agent": "pubchem-utils/1.0",
        "Accept": "application/json" if accept_json else "*/*",
    }
    resp = None
    for i in range(tries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        time.sleep(backoff * (2**i))
    return resp


def pubchem_autocomplete(
    query: str, *, limit: int = 25, timeout: int = DEFAULT_TIMEOUT
) -> List[str]:
    """
    PubChem Autocomplete (typo-tolerant) → list of suggested names.
    """
    if len(query) < 2:
        return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{quote(query)}/JSON"
    r = _get(url, params={"limit": limit}, timeout=timeout, accept_json=True)
    if not (r and r.ok):
        return []
    return r.json().get("dictionary_terms", {}).get("compound", [])[:limit]


def name_to_cid(name: str, *, timeout: int = DEFAULT_TIMEOUT) -> Optional[int]:
    """
    Resolve a chosen name to a PubChem CID.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/cids/JSON"
    r = _get(url, timeout=timeout, accept_json=True)
    if r and r.ok:
        cids = r.json().get("IdentifierList", {}).get("CID", [])
        if cids:
            return int(cids[0])
    return None


def cid_to_canonical_smiles(cid: int, *, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """
    CID → Canonical SMILES (stereochemistry-agnostic).
    Tries TXT fast path, then JSON.
    """
    # TXT fast path
    url_txt = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
    r1 = _get(url_txt, timeout=timeout)
    if r1 and r1.ok:
        t = (r1.text or "").strip()
        if t and not t.lower().startswith(("status", "error", "<!doctype", "<html")):
            return t

    # JSON fallback
    url_json = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    r2 = _get(url_json, timeout=timeout, accept_json=True)
    if r2 and r2.ok:
        props = r2.json().get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("CanonicalSMILES")

    return None
