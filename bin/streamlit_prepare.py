#!/usr/bin/env python3
"""
Streamlit front for bin/dependency_prep.py.

ensure_ready_ui() is what goes in front of a Run button: on the first run it
fetches and derives whatever the tool is missing, showing the log while it
happens; on every run after that it costs one filesystem check and renders
nothing.
"""
from __future__ import annotations

import time

import streamlit as st

from bin import dependency_prep

_MAX_LOG_LINES = 400          # keep the widget from growing without bound
_REDRAW_SECONDS = 0.4         # log lines arrive faster than a browser can care


def _run_with_live_log(tool: str, force: bool = False):
    """Run the preparation, streaming its output into a scrolling code block."""
    placeholder = st.empty()
    lines: list[str] = []
    last_draw = 0.0

    def log(line: str) -> None:
        nonlocal last_draw
        lines.append(line)
        del lines[:-_MAX_LOG_LINES]
        now = time.monotonic()
        if now - last_draw >= _REDRAW_SECONDS:
            last_draw = now
            placeholder.code("\n".join(lines[-40:]), language="text")

    try:
        report = dependency_prep.prepare(tool, log=log, force=force)
    finally:
        placeholder.code("\n".join(lines[-40:]), language="text")
    return report


def render_status(tool: str, *, expanded: bool = False) -> dict:
    """A read-only view of what this checkout has, plus a re-check / rebuild."""
    report = dependency_prep.status(tool)
    label = dependency_prep.LABELS[tool]
    icon = "✅" if report["ready"] else "⚠️"

    with st.expander(f"{icon} {label} reference data", expanded=expanded):
        if report["note"]:
            st.caption(report["note"])
        rows = []
        for entry in report["entries"]:
            rows.append({
                "Data": entry.get("label", entry["key"]),
                "Status": "present" if entry.get("present") else "missing",
                "Delivered by": "git" if entry.get("shipped") else entry.get("source", ""),
                "Size (MB)": entry.get("size_mb", 0.0),
            })
        st.dataframe(rows, hide_index=True, width="stretch")

        pending = dependency_prep.download_estimate(tool)
        if pending:
            st.caption(f"About {pending:.0f} MB will be fetched the first time you run {label}.")

        if st.button(f"Prepare {label} data now", key=f"prep_now_{tool}"):
            try:
                _run_with_live_log(tool)
                st.success(f"{label} data is ready.")
            except dependency_prep.PrepareError as exc:
                st.error(str(exc))
    return report


def ensure_ready_ui(tool: str) -> bool:
    """
    Prepare the tool's data if this is the first run. Returns True when it is
    safe to continue, False when preparation failed and the caller should stop.
    """
    report = dependency_prep.status(tool)
    if report["ready"]:
        return True

    label = dependency_prep.LABELS[tool]

    if not report["available"]:
        st.error(report["note"])
        return False

    if report.get("missing_shipped"):
        # not a download problem - the checkout itself is incomplete
        st.error(report["note"] or f"{label} is missing files that ship with its repository.")
        return False

    pending = dependency_prep.download_estimate(tool)
    size_note = f" (about {pending:.0f} MB)" if pending else ""

    with st.status(
        f"Preparing {label} reference data{size_note} — this happens once, "
        f"then every later run reuses it.",
        expanded=True,
    ) as box:
        try:
            _run_with_live_log(tool)
        except dependency_prep.PrepareError as exc:
            box.update(label=f"{label} preparation failed", state="error")
            st.error(str(exc))
            return False
        except Exception as exc:                       # noqa: BLE001 - surfaced to the user
            box.update(label=f"{label} preparation failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return False
        box.update(label=f"{label} reference data is ready", state="complete", expanded=False)
    return True
