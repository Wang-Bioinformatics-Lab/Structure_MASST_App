# utils/session_utils.py
import os
import time
import uuid
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Union, List, Dict

import pandas as pd

import streamlit as st
import subprocess


# ---------- Session identity ----------


def get_session_hash() -> str:
    if "_session_hash" not in st.session_state:
        raw = f"{uuid.uuid4()}-{time.time()}-{os.urandom(16).hex()}"
        st.session_state["_session_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return st.session_state["_session_hash"]


def session_output_dir(root: Union[str, os.PathLike] = "user_runs") -> Path:
    sid = get_session_hash()
    out = Path(root) / sid
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------- Save TSV subset & run CLI to produce HTML ----------

CmdTemplate = Union[str, Iterable[str]]

def _format_cmd(cmd: CmdTemplate, **paths: str) -> List[str]:
    """Fill {tsv}, {html}, {extra} placeholders in a command template."""
    if isinstance(cmd, str):
        return [part.format(**paths) for part in cmd.split()]
    return [part.format(**paths) for part in cmd]


def save_subset_and_build_html(
    subset_df: pd.DataFrame,
    *,
    tag: str,
    out_root: Union[str, os.PathLike] = "user_runs",
    cli: CmdTemplate = ("my_tool", "--input", "{tsv}", "--output", "{html}"),
    extra_files: Mapping[str, str] | None = None,
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Union[str, int]]:
    """
    Saves `subset_df` to a per-session TSV and runs a CLI that outputs HTML.

    Files are saved inside a per-session folder named after the session hash. Filenames are just "{tag}.tsv" / "{tag}.html".

    Parameters
    ----------
    subset_df : DataFrame
        The (already selected) subset to persist.
    tag : str
        A short identifier for which button/action produced this output,
        e.g. "top_hits", "selected_rows", "qc_report". Used in filename.
    out_root : path
        Root folder for all user outputs. A subfolder per session will be created.
    cli : command template
        Either a list or a string. May contain placeholders {tsv}, {html}, and keys
        from `extra_files`.
    extra_files : mapping
        Optional mapping of placeholder -> path to include in the command, e.g.,
        {"db": "/path/to/db.sqlite"} and then use "--db", "{db}" in `cli`.
    overwrite : bool
        Whether to overwrite existing files with the same tag.
    encoding : str
        Encoding for writing the TSV.

    Returns
    -------
    dict with keys: tsv, html, returncode, stdout, stderr
    """
    out_dir = session_output_dir(out_root)

    # sanitize tag a bit
    safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag.strip()) or "report"
    tsv_path = out_dir / f"{safe_tag}.tsv"
    html_path = out_dir / f"{safe_tag}.html"

    if overwrite or not tsv_path.exists():
        subset_df.to_csv(tsv_path, sep="\t", index=False, encoding=encoding)

    # Build command
    placeholders = {"tsv": str(tsv_path), "html": str(html_path)}
    if extra_files:
        placeholders.update(extra_files)

    cmd = _format_cmd(cli, **placeholders)

    # Run
    proc = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "tsv": str(tsv_path),
        "html": str(html_path),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


# ---------- Example usage in your main page ----------
# Put this in your main app (e.g., Home) wherever you have your table and buttons.
#
# import streamlit as st
# import pandas as pd
# from utils.session_utils import get_session_hash, save_subset_and_build_html
# from streamlit_extras.switch_page_button import switch_page  # optional helper
#
# st.title("Demo: Per-session TSV → CLI → HTML")
# df = pd.DataFrame({"a": [1,2,3,4], "b": ["x","y","x","y"]})
# st.dataframe(df)
# sid = get_session_hash()
# st.caption(f"Session: {sid}")
#
# if st.button("Make HTML for b == 'x'"):
#     subset = df[df["b"] == "x"]
#     info = save_subset_and_build_html(
#         subset,
#         tag="b_eq_x",
#         cli=("my_cli_tool", "--in", "{tsv}", "--out", "{html}"),
#     )
#     if info["returncode"] == 0:
#         st.success(f"Saved HTML → {info['html']}")
#         # Jump to the viewer page (defined below) to display it
#         try:
#             from streamlit_extras.switch_page_button import switch_page
#             switch_page("Report Viewer")  # matches the page title below
#         except Exception:
#             st.page_link("pages/2_Report_Viewer.py", label="Open report page →")
#     else:
#         st.error("CLI failed. See logs below.")
#         with st.expander("stdout/stderr"):
#             st.code(info["stdout"])
#             st.code(info["stderr"])


# ---------- A simple page to display per-session HTML outputs ----------
# Save this file as pages/2_Report_Viewer.py to create a second tab.
# It lists all HTML files generated in the current session and previews one.

# --- BEGIN: pages/2_Report_Viewer.py ---
# import streamlit as st
# from pathlib import Path
# from utils.session_utils import get_session_hash, session_output_dir
# from streamlit.components.v1 import html as st_html
#
# st.set_page_config(page_title="Report Viewer")
# st.title("Report Viewer")
# sid = get_session_hash()
# out_dir = session_output_dir("user_runs")
# files = sorted(out_dir.glob("*.html"))
# if not files:
#     st.info("No HTML reports for this session yet. Go back and generate one.")
# else:
#     pretty = {f.name: f for f in files}
#     choice = st.selectbox("Select a report", list(pretty.keys()))
#     if choice:
#         html_path = pretty[choice]
#         st.caption(f"Showing: {html_path}")
#         try:
#             html_text = html_path.read_text(encoding="utf-8", errors="ignore")
#             st_html(html_text, height=900, scrolling=True)
#         except Exception as e:
#             st.error(f"Failed to render HTML: {e}")
# --- END: pages/2_Report_Viewer.py ---
