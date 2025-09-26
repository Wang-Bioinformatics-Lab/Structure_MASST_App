import time
import pandas as pd
import streamlit as st
import subprocess
from pathlib import Path
from typing import Dict


def run_topic_MASSTs(input_tsv, output_folder, query_indicator):
    workdir = Path("external/microbe_masst/code")
    cmd = ["python", "masst_client.py", 
    "--mode", "draw", 
    "--out_file", f"../../../{output_folder}/topic_masst_",
    "--input_usi_results_file", f"../../../{input_tsv}",
    "--usi_or_lib_id", " ",
    "--compound_name", f"{query_indicator}"]

    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)

    return proc.returncode, proc.stdout, proc.stderr

def _mri_key(df: pd.DataFrame) -> pd.Series:
    """Return a unified MRI key as string; empty series if no suitable column."""
    if "mri" in df.columns:
        return df["mri"].astype(str)
    if "mri_id_int" in df.columns:
        return df["mri_id_int"].astype(str)
    return pd.Series([], dtype=str)

@st.fragment
def domainmasst_fragment(name: str, output_folder: str, df_redu: pd.DataFrame):
    """Self-contained UI + action to run DomainMASST for a single query `name`."""

    if st.button(
        f"Populate DomainMASST for {name}",
        key=f"{name}_topic_masst_btn",
        disabled=df_redu is None or df_redu.empty,
    ):
        sid = st.session_state.get("_session_hash", "unknown")

        # Prepare input
        topic_masst_df = df_redu.copy()
        keep_cols = [c for c in ["USI", "Cosine", "Matching Peaks", "Delta Mass"] if c in topic_masst_df.columns]
        if not keep_cols:
            st.error("Required columns not present in results (need at least USI).")
            return

        topic_masst_df = topic_masst_df[keep_cols]
        topic_masst_df["Status"] = 1

        input_path = f"{output_folder}/topicMasst_input_{name}.tsv"
        topic_masst_df.to_csv(input_path, sep="\t", index=False, header=True)

        with st.spinner("Running DomainMASSTs…"):
            time.sleep(2)
            returncode, stdout, stderr = run_topic_MASSTs(input_path, output_folder, name)

        if returncode == 0:
            st.session_state["last_topic_masst_name"] = name
            st.session_state["last_topic_masst_output_dir"] = output_folder

            st.success(f"DomainMASSTs for **{name}** completed.")
            st.page_link(
                "pages/domainMASST (under construction).py",
                label="➡️ Click for DomainMASST Results",
            )
        else:
            st.error("DomainMASSTs failed. See logs below.")
            with st.expander("Show logs"):
                st.code(stdout or "", language="text")
                st.code(stderr or "", language="text")


@st.fragment
def domainmasst_intersection_fragment(
    redu_tables: Dict[str, pd.DataFrame],
    output_folder: str,
    button_label: str = "Populate DomainMASST with Molecule Co-occurrence",
    job_name: str = "moleculeIntersection",
):
    """
    Build the intersection of MRIs across all ReDU tables, select 1 row/MRI
    with max Cosine, save TSV, and trigger DomainMASSTs.

    Parameters
    ----------
    redu_tables : dict[str, pd.DataFrame]
        Mapping from query-name -> ReDU DataFrame.
    output_folder : str
        Session output folder (writable).
    button_label : str
        UI label for the action button.
    job_name : str
        Label passed to DomainMASST (also used in file naming).
    """
    # Quick enable/disable guard
    enough = (
        isinstance(redu_tables, dict)
        and len(redu_tables) > 1
        and all(isinstance(df, pd.DataFrame) and len(df) > 0 for df in redu_tables.values())
    )

    if st.button(button_label, key="intersection_topic_masst", disabled=not enough):
        # 1) Compute intersection of MRIs
        mri_sets = [set(_mri_key(df).dropna().unique()) for df in redu_tables.values()]
        common_mris = set.intersection(*mri_sets) if mri_sets else set()

        if not common_mris:
            st.warning("No shared samples across selected molecules.")
            return

        # 2) Keep only rows whose MRI is in ALL tables; normalize Cosine
        filtered_frames = []
        for df in redu_tables.values():
            dfx = df.copy()

            if "Cosine" not in dfx.columns and "cosine" in dfx.columns:
                dfx = dfx.rename(columns={"cosine": "Cosine"})

            key_series = _mri_key(dfx)
            if key_series.empty:
                # skip tables without MRI column
                continue

            dfx["mri_key"] = key_series
            filtered_frames.append(dfx[dfx["mri_key"].isin(common_mris)])

        if not filtered_frames:
            st.warning("No rows remained after intersecting by MRI.")
            return

        cooccurrence_df = pd.concat(filtered_frames, ignore_index=True)

        # 3) Pick one row per MRI (highest Cosine if available)
        cos_series = pd.to_numeric(
            cooccurrence_df.get("Cosine", pd.Series([-1] * len(cooccurrence_df))),
            errors="coerce",
        ).fillna(-1)
        cooccurrence_df["__cos_num"] = cos_series
        idxmax = cooccurrence_df.groupby("mri_key")["__cos_num"].idxmax()
        topic_masst_df = cooccurrence_df.loc[idxmax].drop(columns="__cos_num").reset_index(drop=True)

        if "Cosine" in topic_masst_df.columns:
            topic_masst_df = topic_masst_df.sort_values("Cosine", ascending=False, na_position="last")

        # 4) Prepare input TSV (require at least USI)
        keep_cols = [c for c in ["USI", "Cosine", "Matching Peaks", "Delta Mass"] if c in topic_masst_df.columns]
        if "USI" not in keep_cols:
            st.error("Cannot run DomainMASSTs: required column 'USI' not found.")
            return

        out_tsv = f"{output_folder}/domainMasst_input_{job_name}.tsv"
        topic_masst_df = topic_masst_df[keep_cols].copy()
        topic_masst_df["Status"] = 1
        topic_masst_df.to_csv(out_tsv, sep="\t", index=False, header=True)

        # 5) Run DomainMASSTs
        with st.spinner("Running DomainMASSTs…"):
            time.sleep(2)
            returncode, stdout, stderr = run_topic_MASSTs(out_tsv, output_folder, job_name)

        if returncode == 0:
            st.session_state["last_topic_masst_name"] = job_name
            st.session_state["last_topic_masst_output_dir"] = output_folder
            st.success("DomainMASSTs for Molecule Intersection completed.")
            st.page_link("pages/domainMASST (under construction).py", label="➡️ Click for DomainMASST Results")
        else:
            st.error("DomainMASSTs failed. See logs below.")
            with st.expander("Show logs"):
                st.code(stdout or "", language="text")
                st.code(stderr or "", language="text")