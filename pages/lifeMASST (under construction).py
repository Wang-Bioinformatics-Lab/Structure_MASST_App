import streamlit as st
import os
import streamlit.components.v1 as components
import shutil
import time
import subprocess
import json

HERE = os.path.dirname(__file__)  

NF_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'nf_workflow.nf'))


# Write the page label
st.set_page_config(
    page_title="LifeMASST",
    page_icon="🧬",
)

left, right = st.columns([6,1])

with left:
    st.title("LifeMASST (preview)")
    st.write("""
            This page lets you further explore your StructureMASST results in the context of different phylogenetic or taxonomic tree analyses. 
            You can investigate how molecules or substructures are distributed across life and align your results with records recorded in Wikidata.
    """)


output_folder = st.session_state["_session_output_folder"]
lifemasst_folder = os.path.join(output_folder, "lifemasst")
molecule_path = os.path.join(lifemasst_folder, "structuremasst_input.tsv")
structuremasst_path = os.path.join(lifemasst_folder, "lifemasst_input_summary.tsv")

tree_directory = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'data'))


tree_nwk_files = [f for f in os.listdir(tree_directory) if f.endswith(".nwk")]
# create named list of nwk files in tree_directory
# labelled_supertree_redu_limited_upfront.nwk = Open Tree of Life
tree_labels = {
    "labelled_supertree_redu_limited_upfront.nwk": "Open Tree of Life",
    "labelled_supertree.nwk": "Open Tree of Life (full)",
}

molecule_path_abs = os.path.abspath(molecule_path)
out_dir_abs = os.path.abspath(lifemasst_folder)
structuremasst_path_abs = os.path.abspath(structuremasst_path)

os.makedirs(lifemasst_folder, exist_ok=True)


match_id_options = {
    "ott_w_prefix": "OpenTreeOfLifeTaxonomyID ('ott<ID>')",
    "ncbi_w_prefix": "NCBI Taxonomic ID ('ncbi<ID>')",
    "ncbi": "NCBI Taxonomic ID ('<ID>')",
}

match_level_options = {
    "exact": "Same as matching ID",
    "NCBIGenus": "NCBIGenus",
    "NCBIClass": "NCBIClass",
    "NCBIOrder": "NCBIOrder",
    "NCBIFamily": "NCBIFamily",
    "NCBIPhylum": "NCBIPhylum",
    "NCBIKingdom": "NCBIKingdom",
}


tree_selection, match_id_selection, match_level_selection, _ = st.columns([3,3,3,3])


with tree_selection:
    tree_choice = st.selectbox(
        "Select a tree for LifeMASST",
        options=tree_nwk_files,
        format_func=lambda x: tree_labels.get(x, x),
        index=0,
        help="Select the phylogenetic or taxonomic tree to use for LifeMASST analysis.",
    )
    tree_path = os.path.join(tree_directory, tree_choice)

with match_id_selection:
    match_id = st.selectbox(
        "Select a matching ID",
        options=match_id_options.keys(),
        format_func=lambda x: match_id_options[x],
        index=2,
        help="Select the taxonomic identifier used in your tree (can be OpenTreeOfLifeTaxonomyID with prefix or NCBI Taxonomic ID, with or without 'ncbi' prefix).",
    )

with match_level_selection:
    match_level = st.selectbox(
        "Select a matching level",
        options=match_level_options.keys(),
        format_func=lambda x: match_level_options[x],
        index=0,
        help="Select the taxonomic level to match your IDs to in the tree (exact ID, genus, class, order, family, phylum, kingdom).",
    )


lifemasst_button, _ = st.columns([9,3])

with lifemasst_button:
    if st.button(
        f"Run LifeMASST",
        key="lifemasst_btn",
    ):
        try:
            with st.spinner("Running LifeMASST…"):

                # Run nexflow LifeMASST
                subprocess.run(
                    ["nextflow", "run", NF_PATH, 
                    "--input_molecules", molecule_path_abs, 
                    "--structureMASST_input_file", structuremasst_path_abs,
                    "--tree_path", tree_path,
                    "--tax_id", match_id,
                    "--output_folder", out_dir_abs],
                    check=True
                )
        
                empress_zip = os.path.join(output_folder, "lifemasst", "empress_results.zip")

                st.success("Download the LifeMASST results below.")
                
                if os.path.exists(empress_zip):
                    with open(empress_zip, "rb") as f:
                        st.download_button(
                            label="📥 Download Empress Results",
                            data=f,
                            file_name="empress_results.zip"
                        )

        except subprocess.CalledProcessError as e:
            st.error("LifeMASST failed. See logs below.")
            with st.expander("Show logs"):
                st.code(e.stdout or "", language="text")
                st.code(e.stderr or "", language="text")
            