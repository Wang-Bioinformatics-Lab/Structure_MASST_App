import time
import pandas as pd
import streamlit as st
import subprocess
from pathlib import Path
from typing import Dict
import os

def run_LifeMASST(input_tsv, output_folder, query_indicator):
    #TODO: change to life_masst when available
    workdir = Path("external/microbe_masst/code")
    cmd = ["python", "masst_client.py", 
    "--mode", "draw", 
    "--out_file", f"../../../{output_folder}/topic_masst_",
    "--input_usi_results_file", f"../../../{input_tsv}",
    "--usi_or_lib_id", " ",
    "--compound_name", f"{query_indicator}"]

    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)

    return proc.returncode, proc.stdout, proc.stderr

@st.fragment
def lifemasst_fragment(input_file: pd.DataFrame, structureMASST_op_folder: str, redu_tables: Dict[str, pd.DataFrame]):
    """Self-contained UI + action to run LifeMASST for a single query `name`."""

    if st.button(
        f"Populate LifeMASST",
        key="lifemasst_btn",
    ):
        out_dir = os.path.join(structureMASST_op_folder, "lifemasst")
        os.makedirs(out_dir, exist_ok=True)

        # Save the input table
        in_path = os.path.join(out_dir, "structuremasst_input.tsv")
        name_enum = {str(n).strip(): i for i, n in enumerate(input_file["name"].astype(str).str.strip().tolist(), start=1)}
        input_file["id"] = input_file["name"].astype(str).str.strip().map(name_enum)
        input_file.to_csv(in_path, sep="\t", index=False)

        saved, skipped = 0, 0
        for key, df in redu_tables.items():
            if "query_name" not in df.columns or df.empty:
                skipped += 1
                st.warning(f"[{key}] skipped: missing 'query_name' or empty table.")
                continue

            uniq = df["query_name"].dropna().astype(str).str.strip().unique()
            if len(uniq) != 1:
                skipped += 1
                st.warning(f"[{key}] skipped: expected exactly one unique query_name, found {len(uniq)}.")
                continue

            cname = uniq[0]
            idx = name_enum.get(cname)
            if idx is None:
                skipped += 1
                st.warning(f"[{key}] skipped: '{cname}' not found in input_file['name'].")
                continue

            out_path = os.path.join(out_dir, f"{idx}.tsv")
            df.to_csv(out_path, sep="\t", index=False)
            saved += 1
        
        # # 1) Compute intersection of MRIs
        # mri_sets = [set(_mri_key(df).dropna().unique()) for df in redu_tables.values()]
        # common_mris = set.intersection(*mri_sets) if mri_sets else set()

        # if not common_mris:
        #     st.warning("No shared samples across selected molecules.")
        #     return

        # # 2) Keep only rows whose MRI is in ALL tables; normalize Cosine
        # filtered_frames = []
        # for df in redu_tables.values():
        #     dfx = df.copy()

        #     if "Cosine" not in dfx.columns and "cosine" in dfx.columns:
        #         dfx = dfx.rename(columns={"cosine": "Cosine"})

        #     key_series = _mri_key(dfx)
        #     if key_series.empty:
        #         # skip tables without MRI column
        #         continue

        #     dfx["mri_key"] = key_series
        #     filtered_frames.append(dfx[dfx["mri_key"].isin(common_mris)])

        # if not filtered_frames:
        #     st.warning("No rows remained after intersecting by MRI.")
        #     return

        # cooccurrence_df = pd.concat(filtered_frames, ignore_index=True)

        # # 3) Pick one row per MRI (highest Cosine if available)
        # cos_series = pd.to_numeric(
        #     cooccurrence_df.get("Cosine", pd.Series([-1] * len(cooccurrence_df))),
        #     errors="coerce",
        # ).fillna(-1)
        # cooccurrence_df["__cos_num"] = cos_series
        # idxmax = cooccurrence_df.groupby("mri_key")["__cos_num"].idxmax()
        # topic_masst_df = cooccurrence_df.loc[idxmax].drop(columns="__cos_num").reset_index(drop=True)

        # if "Cosine" in topic_masst_df.columns:
        #     topic_masst_df = topic_masst_df.sort_values("Cosine", ascending=False, na_position="last")

        # # 4) Prepare input TSV (require at least USI)
        # keep_cols = [c for c in ["USI", "Cosine", "Matching Peaks", "Delta Mass"] if c in topic_masst_df.columns]
        # if "USI" not in keep_cols:
        #     st.error("Cannot run DomainMASSTs: required column 'USI' not found.")
        #     return

        # out_tsv = f"{output_folder}/domainMasst_input_{job_name}.tsv"
        # topic_masst_df = topic_masst_df[keep_cols].copy()
        # topic_masst_df["Status"] = 1
        # topic_masst_df.to_csv(out_tsv, sep="\t", index=False, header=True)

        # # 5) Run DomainMASSTs
        # with st.spinner("Running DomainMASSTs…"):
        #     time.sleep(2)
        #     returncode, stdout, stderr = run_topic_MASSTs(out_tsv, output_folder, job_name)

        # if returncode == 0:
        #     st.session_state["last_topic_masst_name"] = job_name
        #     st.session_state["last_topic_masst_output_dir"] = output_folder
        #     st.success("DomainMASSTs for Molecule Intersection completed.")
        #     st.page_link("pages/domainMASST (under construction).py", label="➡️ Click for DomainMASST Results")
        # else:
        #     st.error("DomainMASSTs failed. See logs below.")
        #     with st.expander("Show logs"):
        #         st.code(stdout or "", language="text")
        #         st.code(stderr or "", language="text")