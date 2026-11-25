"""
LifeMASST input preparation (no UI code), used by the compute worker.

setup_lifemasst_files() and prepare_lifemasst_input() moved here unchanged from
the former Streamlit fragment bin/streamlit_fragment_LifeMASST.py.
"""
import os
import sys
from typing import Dict

import pandas as pd

HERE = os.path.dirname(__file__)
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'code'))
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

import summarize_by_molecule  # noqa: E402


def setup_lifemasst_files(
    input_file: pd.DataFrame,
    structureMASST_op_folder: str,
    redu_tables: Dict[str, pd.DataFrame],
):
    out_dir = os.path.join(structureMASST_op_folder, "lifemasst")
    os.makedirs(out_dir, exist_ok=True)

    input_file = input_file.copy()

    in_path = os.path.join(out_dir, "structuremasst_input.tsv")
    name_enum = {
        str(n).strip(): i
        for i, n in enumerate(input_file["name"].astype(str).str.strip().tolist(), start=1)
    }
    input_file["id"] = input_file["name"].astype(str).str.strip().map(name_enum)
    input_file.to_csv(in_path, sep="\t", index=False)

    collect_dfs = []

    for key, df in redu_tables.items():
        if "query_name" not in df.columns or df.empty:
            continue

        uniq = df["query_name"].dropna().astype(str).str.strip().unique()
        if len(uniq) != 1:
            continue

        df_organism_summary_by_id = summarize_by_molecule.main(df, "ncbi", uniq[0])
        collect_dfs.append(df_organism_summary_by_id)

    if not collect_dfs:
        return in_path, None

    merged_summary = collect_dfs[0]
    first_col = merged_summary.columns[0]

    for df in collect_dfs[1:]:
        # NCBITaxonomy_file_count is background data identical across molecules; drop
        # it from incoming dfs to prevent suffix conflicts in iterative merges.
        dup_cols = [c for c in df.columns if c in merged_summary.columns and c != first_col]
        df = df.drop(columns=dup_cols)
        merged_summary = pd.merge(merged_summary, df, on=first_col, how="outer")

    for col in merged_summary.columns:
        if col.startswith("masstCounts_"):
            merged_summary[col] = merged_summary[col].astype("Int64")

    if "OpenTreeOfLifeTaxonomyID" in merged_summary.columns:
        merged_summary["OpenTreeOfLifeTaxonomyID"] = (
            merged_summary["OpenTreeOfLifeTaxonomyID"]
            .astype(str)
            .str.replace("ott", "", regex=False)
        )
        merged_summary["OpenTreeOfLifeTaxonomyID"] = merged_summary["OpenTreeOfLifeTaxonomyID"].astype("Int64")

    if "NCBI_ID" in merged_summary.columns:
        merged_summary["NCBI_ID"] = (
            merged_summary["NCBI_ID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
        )
        merged_summary["NCBI_ID"] = merged_summary["NCBI_ID"].astype("Int64")

    out_path = os.path.join(out_dir, "lifemasst_input_summary.tsv")
    merged_summary.to_csv(out_path, sep="\t", index=False)

    return in_path, out_path

def prepare_lifemasst_input(query_table: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of query_table where SMILES queries are restored to the
    original user-facing SMILES before harmonization.
    """
    if query_table is None or not isinstance(query_table, pd.DataFrame) or query_table.empty:
        return query_table

    out = query_table.copy()

    if "original_smiles" in out.columns and "type" in out.columns:
        mask = (
            out["type"].astype(str).str.strip().eq("smiles")
            & out["original_smiles"].fillna("").astype(str).str.strip().ne("")
        )

        if "query" in out.columns:
            out.loc[mask, "query"] = out.loc[mask, "original_smiles"]

        if "smiles" not in out.columns:
            out["smiles"] = ""

        out.loc[mask, "smiles"] = out.loc[mask, "original_smiles"]

    return out
