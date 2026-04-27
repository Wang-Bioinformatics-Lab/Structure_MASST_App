"""Shared MGF parsing utilities used by the Streamlit app and the batch runner."""
from __future__ import annotations


def parse_mgf_lines(lines) -> list:
    """Parse an iterable of MGF text lines into a list of spectrum dicts.

    Each dict contains:
        spectrum_id  (str)   – "scan:<SCANS>" or "spectrum:<idx>"
        name         (str)   – NAME field (grouping key); falls back to spectrum_id
        precursor_mz (float)
        peaks        (list of [mz, intensity])
    Additional MGF header fields are stored as "meta_<key_lower>" entries.
    """
    spectra: list = []
    current: dict = {}
    peaks: list = []
    in_block = False

    for raw_line in lines:
        line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper == "BEGIN IONS":
            in_block = True
            current = {}
            peaks = []
        elif upper == "END IONS":
            in_block = False
            if peaks and "precursor_mz" in current:
                idx = len(spectra)
                scan_val = current.get("scan_id")
                current["spectrum_id"] = f"scan:{scan_val}" if scan_val is not None else f"spectrum:{idx}"
                # NAME= is the grouping key; if absent each spectrum is its own group
                if "name" not in current:
                    current["name"] = current["spectrum_id"]
                current["peaks"] = peaks
                spectra.append(current)
            current = {}
            peaks = []
        elif in_block:
            if "=" in line:
                key, _, val = line.partition("=")
                key_up = key.strip().upper()
                val = val.strip()
                if key_up == "PEPMASS":
                    try:
                        current["precursor_mz"] = float(val.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif key_up == "NAME":
                    current["name"] = val
                elif key_up == "SCANS":
                    current["scan_id"] = val
                elif key_up == "CHARGE":
                    current["charge"] = val
                current[f"meta_{key_up.lower()}"] = val
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass

    return spectra


def parse_mgf_file(path) -> list:
    """Convenience wrapper: parse an MGF file given a Path or path string."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_mgf_lines(fh)


def parse_mgf_bytes(data: bytes) -> list:
    """Convenience wrapper: parse MGF content given raw bytes (e.g. from st.file_uploader)."""
    return parse_mgf_lines(data.decode("utf-8", errors="replace").splitlines())
