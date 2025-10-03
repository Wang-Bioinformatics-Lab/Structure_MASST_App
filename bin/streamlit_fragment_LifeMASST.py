import time
import pandas as pd
import streamlit as st
import subprocess
from pathlib import Path
from typing import Dict
import os
import sys 

HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'code'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


import summarize_by_molecule 

NF_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'nf_workflow.nf'))


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
def lifemasst_fragment(input_file: pd.DataFrame, structureMASST_op_folder: str, redu_tables: Dict[str, pd.DataFrame], append: str = ""):
    """Self-contained UI + action to run LifeMASST for a single query `name`."""

    if st.button(
        f"Setup LifeMASST",
        key="lifemasst_btn" + append,
    ):
        try:
            with st.spinner("Setting up LifeMASSTs…"):
                out_dir = os.path.join(structureMASST_op_folder, "lifemasst")
                os.makedirs(out_dir, exist_ok=True)

                # Save the input table
                in_path = os.path.join(out_dir, "structuremasst_input.tsv")
                name_enum = {str(n).strip(): i for i, n in enumerate(input_file["name"].astype(str).str.strip().tolist(), start=1)}
                input_file["id"] = input_file["name"].astype(str).str.strip().map(name_enum)
                input_file.to_csv(in_path, sep="\t", index=False)

                saved, skipped = 0, 0
                collect_dfs = []

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

                    df_organism_summary_by_id = summarize_by_molecule.main(df, 'ott', uniq[0])

                    collect_dfs.append(df_organism_summary_by_id)   

                    cname = uniq[0]
                    idx = name_enum.get(cname)
                    if idx is None:
                        skipped += 1
                        st.warning(f"[{key}] skipped: '{cname}' not found in input_file['name'].")
                        continue

                    saved += 1
                
                if not collect_dfs:
                    st.warning("No valid tables to process for LifeMASST.")
                    return
                
                # merge all summaries by id
                merged_summary = collect_dfs[0]
                first_col = merged_summary.columns[0]
                for df in collect_dfs[1:]:
                    merged_summary = pd.merge(merged_summary, df, on=first_col, how="outer")

                # make sure all columns starting with masstCounts are integer
                for col in merged_summary.columns:
                    if col.startswith("masstCounts_"):
                        merged_summary[col] = merged_summary[col].astype("Int64")

                # IMPORTANT UNTIL FIXED IN REDU: remove OpenTreeOfLifeTaxonomyID ott prefix
                merged_summary['OpenTreeOfLifeTaxonomyID'] = merged_summary['OpenTreeOfLifeTaxonomyID'].astype(str).str.replace('ott', '', regex=False)
                merged_summary['OpenTreeOfLifeTaxonomyID'] = merged_summary['OpenTreeOfLifeTaxonomyID'].astype("Int64")
                out_path = os.path.join(out_dir, f"lifemasst_input_summary.tsv")
                merged_summary.to_csv(out_path, sep="\t", index=False)

                st.success(f"LifeMASST setup completed.")
                st.page_link(
                    "pages/lifeMASST (under construction).py",
                    label="➡️ Click for LifeMASST Results",
                )
        except subprocess.CalledProcessError as e:
            st.error("LifeMASST setup failed.")
            with st.expander("Show logs"):
                st.code(e.stdout or "", language="text")

