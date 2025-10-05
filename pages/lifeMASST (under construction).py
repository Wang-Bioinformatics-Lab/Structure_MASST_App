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
    "labelled_supertree_subset_prepped": "Open Tree of Life",
    "labelled_supertree_full_prepped": "Open Tree of Life (full) - not recommended",
    "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint": "Bird Tree of Life (time tree)",
    "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa": "Bird Tree of Life (molecular tree)",
    "upload": "Custom Uploaded Tree",
}

molecule_path_abs = os.path.abspath(molecule_path)
out_dir_abs = os.path.abspath(lifemasst_folder)
structuremasst_path_abs = os.path.abspath(structuremasst_path)

os.makedirs(lifemasst_folder, exist_ok=True)


match_id_options = {
    "ott": "OpenTreeOfLifeTaxonomyID",
    "ncbi": "NCBI Taxonomic ID",
}

match_level_options = {
    "exact": "NCBI ID",
    "NCBIGenus": "NCBI Genus",
    "NCBIClass": "NCBI Class",
    "NCBIOrder": "NCBI Order",
    "NCBIFamily": "NCBI Family",
    "NCBIPhylum": "NCBI Phylum",
    "NCBIKingdom": "NCBI Kingdom",
}


tree_selection, _, _, _ = st.columns([3,3,3,3])


with tree_selection:
    tree_choice = st.selectbox(
        "Select a tree for LifeMASST",
        options=list(tree_labels.keys()),   # keys are the real values
        format_func=lambda x: tree_labels.get(x, x),  # labels shown
        index=0,
        help="Select the phylogenetic or taxonomic tree to use for LifeMASST analysis.",
    )

if tree_choice in ["labelled_supertree_subset_prepped", "labelled_supertree_full_prepped"]:
    match_id = "ott"
    id_prefix = "ott"
if tree_choice in ["trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint", "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa"]:
    match_id = "Native_tree_label"
    id_prefix = ""
    if tree_choice == "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint":
        message = """
        We use the rooted timetree with all taxa from Kimball et al. (2019)
        (timetree_all_taxa_OW_2019.nextree.tre, rooted version).
        → Branch lengths represent absolute divergence times estimated with fossil calibrations.
        """
    if tree_choice == "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa":
        message = """
        We use the CYB+ND2 supertree with estimated branch lengths
        (OW_2019_CYB_ND2_estBL.tre).
        → Branch lengths represent molecular substitution distances based on mitochondrial gene data.
        """

    st.info(message)


match_level_selection_masst, match_level_selection_wd, _, _ = st.columns([3,3,3,3])

with match_level_selection_masst:
    match_level_masst = st.selectbox(
        "Matching level for metabolomics raw data",
        options=match_level_options.keys(),
        format_func=lambda x: match_level_options[x],
        index=0,
        help="Select the taxonomic level to match your IDs to in the tree (exact ID, genus, class, order, family, phylum, kingdom).",
    )

with match_level_selection_wd:
    match_level_wd = st.selectbox(
        "Matching level for wikidata molecule-organism records",
        options=match_level_options.keys(),
        format_func=lambda x: match_level_options[x],
        index=0,
        help="Select the taxonomic level to match your IDs to in the tree (exact ID, genus, class, order, family, phylum, kingdom).",
    )
if tree_choice != "upload":
    tree_path = os.path.join(tree_directory, tree_choice) + ".nwk"
    feature_path = os.path.join(tree_directory, tree_choice) + ".tsv"

else:

    tree_upload, match_id_selection, match_id_prefix, _ = st.columns([3,3,3,3])


    with tree_upload:
        uploaded_tree = st.file_uploader(
            "Upload your own tree in Newick format (.nwk)",
            type=["nwk"],
            accept_multiple_files=False,
            help="Upload a Newick formatted tree file. Ensure that the tip labels match the taxonomic IDs you will select.",
        )
        if uploaded_tree is not None:
            tree_path = os.path.join(lifemasst_folder, "uploaded_tree.nwk")
            with open(tree_path, "wb") as f:
                f.write(uploaded_tree.getbuffer())
        else:
            st.warning("Please upload a tree file to proceed.")
            st.stop()


    with match_id_selection:
        match_id = st.selectbox(
            "Select a matching ID between your tree and metabolomics raw data",
            options=match_id_options.keys(),
            format_func=lambda x: match_id_options[x],
            index=2,
            help="Select the taxonomic identifier used in your tree (can be OpenTreeOfLifeTaxonomyID with prefix or NCBI Taxonomic ID, with or without 'ncbi' prefix).",
        )

    # make text input for prefix to id
    with match_id_prefix:
        id_prefix = st.text_input(
            "If your tree uses prefixed IDs (e.g. 'ott12345' or 'ncbi12345'), enter the prefix here (e.g. 'ott' or 'ncbi'). Otherwise, leave blank.",
            value="",
            help="Enter the prefix used in your tree IDs, if any.",
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
                    "--tree_features", feature_path,
                    "--tax_id", match_id,
                    "--tax_id_prefix", id_prefix,
                    "--masst_matchLevel", match_level_masst,
                    "--wikidata_matchLevel", match_level_wd,
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
            