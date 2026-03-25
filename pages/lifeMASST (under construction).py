import streamlit as st
import os
import streamlit.components.v1 as components
import shutil
import time
import subprocess
import json
import pandas as pd
from pathlib import Path

# Tracking
import umami
umami.set_url_base("https://analytics-api.gnps2.org/")
umami.set_website_id('032bfca4-a353-4586-b637-8908d8b71c85')
umami.set_hostname('analytics-api.gnps2.org')

# Add a tracking token
from streamlit.components.v1 import html
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0) # GNPS2 Global
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="74665d88-3b9d-4812-b8fc-7f55ceb08f11"></script>',  width=0, height=0) # Streamlit Apps
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="032bfca4-a353-4586-b637-8908d8b71c85"></script>',  width=0, height=0) # Structure MASST



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



if not st.session_state.get("raw_results"):
    st.warning("This is a downstream tool. Run StructureMASST first to generate LifeMASST results.")
    st.stop()



output_folder = st.session_state["_session_output_folder"]
lifemasst_folder = os.path.join(output_folder, "lifemasst")
molecule_path = os.path.join(lifemasst_folder, "structuremasst_input.tsv")
structuremasst_path = os.path.join(lifemasst_folder, "lifemasst_input_summary.tsv")
tree_directory = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'data'))


# check if structuremasst_path exists
if not os.path.exists(structuremasst_path):
    st.warning("Click the Setup LifeMASST button at the bottom of your StructureMASST results to proceed with LifeMASST.")
    st.stop()



def format_message(msg: str) -> str:
    """Wrap message in a styled HTML info box for Streamlit display."""
    return f"""
    <div style="
        background-color: #f9f6f1;
        border-left: 5px solid #b58900;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        color: #222;
        font-size: 1.05rem;
        line-height: 1.5;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    ">
        {msg}
    </div>
    """


# --- RESULT PATHS (DEFINED OUTSIDE THE BUTTON) ---
empress_zip_path = os.path.join(output_folder, "lifemasst", "empress_results.zip")
tree_png_path = os.path.join(output_folder, "lifemasst", "tree_plot.png")
metadata_tsv_path = os.path.join(output_folder, "lifemasst", "merged_metadata.tsv")

tree_nwk_files = [f for f in os.listdir(tree_directory) if f.endswith(".nwk")]
tree_labels = {
    "trees/OTL_prepared/labelled_supertree_subset_prepped": "Open Tree of Life",
    "trees/timetree_prepared/TimeTree_subset_prepped": "TimeTree",
    #"labelled_supertree_full_prepped": "Open Tree of Life (full) - not recommended",
    "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint": "Bird Tree (time tree)",
    "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa": "Bird Tree (molecular tree)",
    "trees/mammalian_prepared/Foley2022_Concatenation_HRA_neutral_241_10miss_rooted": "Mammal Tree (molecular tree)",
    "trees/fish_prepared/fish_raxml_12k": "Fish Tree (molecular tree)",
    "trees/fish_prepared/fish_treepl_12k": "Fish Tree (time tree)",
    "trees/plants_prepared/1kp_astral_alltaxa_FAA": "Plant Tree (molecular tree)",
    "upload": "Custom Uploaded Tree",
}

# read molecule_path file make unique by 'name' column and save new file

df_molecule = pd.read_csv(molecule_path, sep="\t")
df_molecule_unique = df_molecule.drop_duplicates(subset=["name"])
molecule_path = os.path.join(lifemasst_folder, "structuremasst_input_unique.tsv")
df_molecule_unique.to_csv(molecule_path, sep="\t", index=False)

molecule_path_abs = os.path.abspath(molecule_path)
out_dir_abs = os.path.abspath(lifemasst_folder)
structuremasst_path_abs = os.path.abspath(structuremasst_path)

os.makedirs(lifemasst_folder, exist_ok=True)

match_id_options = {
    "ott": "OpenTreeOfLifeTaxonomyID",
    "ncbi": "NCBI Taxonomic ID",
}

tree_selection, _, _, _ = st.columns([3,3,3,3])

with tree_selection:
    tree_choice = st.selectbox(
        "Select a tree for LifeMASST",
        options=list(tree_labels.keys()),
        format_func=lambda x: tree_labels.get(x, x),
        index=0,
        help="Select the phylogenetic or taxonomic tree to use for LifeMASST analysis.",
    )

# if tree choice does not start with Open Tree of Life
if tree_choice not in ["trees/OTL_prepared/labelled_supertree_subset_prepped", "trees/OTL_prepared/labelled_supertree_full_prepped"]:
    match_level_options = {
        "NCBISpecies": "NCBI Species",
        "NCBIFamily": "NCBI Family",
        "NCBI_ID": "NCBI ID",
        "NCBIGenus": "NCBI Genus",
        "NCBIClass": "NCBI Class",
        "NCBIOrder": "NCBI Order",
        "NCBIPhylum": "NCBI Phylum",
    }
else:
    match_level_options = {
        "NCBIFamily": "NCBI Family",
        "NCBIGenus": "NCBI Genus",
        "NCBISpecies": "NCBI Species",
        "NCBIClass": "NCBI Class",
        "NCBIOrder": "NCBI Order",
        "NCBIPhylum": "NCBI Phylum",
        "OpenTreeOfLifeTaxonomyID": "OpenTreeOfLifeTaxonomyID",
    }

message = ""

if tree_choice in [
    "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint",
    "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa"
]:
    match_id = "Native_tree_label"
    id_prefix = ""
    if tree_choice == "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint":
        message = (
            "Rooted timetree from Kimball et al. (2019)<br/>"
            "→ Branch lengths represent absolute divergence times estimated with fossil calibrations.<br/>"
            "Source: Kimball et al. (2019), <i>Diversity</i> 11, 109. https://doi.org/10.3390/d11070109"
        )
    elif tree_choice == "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa":
        message = (
            "CYB+ND2 supertree from Kimball et al. (2019)<br/>"
            "→ Branch lengths represent molecular substitution distances based on mitochondrial gene data.<br/>"
            "Source: Kimball et al. (2019), <i>Diversity</i> 11, 109. https://doi.org/10.3390/d11070109"
        )

elif tree_choice == "trees/mammalian_prepared/Foley2022_Concatenation_HRA_neutral_241_10miss_rooted":
    match_id = "Native_tree_label"
    id_prefix = ""
    message = (
        "Concatenated phylogeny from Foley et al. (2023)<br/>"
        "→ Branch lengths represent molecular substitution distances based on genome-wide nearly neutral sites.<br/>"
        "Source: Foley et al. (2023), <i>Science</i> 380, eabl8189. https://doi.org/10.1126/science.abl8189"
    )

elif tree_choice == "trees/fish_prepared/fish_treepl_12k":
    match_id = "Native_tree_label"
    id_prefix = ""
    message = (
        "Time-calibrated tree (Actinopterygii 12k; treePL)<br/>"
        "→ Branch lengths represent absolute divergence times (millions of years) estimated via fossil-calibrated treePL.<br/>"
        "Source: Rabosky et al. (2018), <i>Nature</i> 559, 392–395. https://doi.org/10.1038/s41586-018-0273-1"
    )

elif tree_choice == "trees/fish_prepared/fish_raxml_12k":
    match_id = "Native_tree_label"
    id_prefix = ""
    message = (
        "Molecular phylogram (Actinopterygii 12k; RAxML)<br/>"
        "→ Branch lengths represent molecular substitution distances (substitutions per site) from maximum-likelihood inference.<br/>"
        "Source: Rabosky et al. (2018), <i>Nature</i> 559, 392–395. https://doi.org/10.1038/s41586-018-0273-1"
    )

elif tree_choice == "trees/plants_prepared/1kp_astral_alltaxa_FAA":
    match_id = "Native_tree_label"
    id_prefix = ""
    message = (
        "Molecular phylogeny from One Thousand Plant Transcriptomes Initiative (1KP, 2019)<br/>"
        "→ Branch lengths represent molecular substitution distances inferred from large-scale transcriptome data.<br/>"
        "Source: One Thousand Plant Transcriptomes Initiative (2019), <i>Nature</i> 574, 679–685. https://doi.org/10.1038/s41586-019-1693-2"
    )

elif tree_choice in ["trees/OTL_prepared/labelled_supertree_subset_prepped", "trees/OTL_prepared/labelled_supertree_full_prepped"]:
    match_id = "OpenTreeOfLifeTaxonomyID"
    id_prefix = "ott"
    if tree_choice == "trees/OTL_prepared/labelled_supertree_full_prepped":
        message = (
            "Open Tree of Life (full)<br/>"
            "→ Comprehensive tree of life from Open Tree of Life, based on a synthesis of published phylogenies and taxonomies.<br/>"
            "Note: This full tree is very large and may lead to long processing times. The subsetted version is recommended for most analyses.<br/>"
            "Source: OpenTree et al. Open Tree of Life Synthetic Tree https://doi.org/10.5281/zenodo.3937741"
        )
    else:
        message = (
            "Open Tree of Life (subset with available metabolomics data)<br/>"
            "→ Subsetted comprehensive tree of life from Open Tree of Life, based on a synthesis of published phylogenies and taxonomies.<br/>"
            "Source: OpenTree et al. Open Tree of Life Synthetic Tree https://doi.org/10.5281/zenodo.3937741"
        )
elif tree_choice == "trees/timetree_prepared/TimeTree_subset_prepped":
    match_id = "Native_tree_label"
    id_prefix = ""
    message = (
        "TimeTree<br/>"
        "→ Time-calibrated tree of life from TimeTree database, based on published divergence time estimates.<br/>"
        "Source: Kumar et al. (2017), <i>Nucleic Acids Research</i> 45(D1), D109–D114. https://doi.org/10.1093/nar/gkw938"
    )

if message != "":
    message_formatted = format_message(message)
else:
    message_formatted = ""
st.markdown(message_formatted, unsafe_allow_html=True)

match_level_selection_masst, match_level_selection_wd, _, _ = st.columns([3,3,3,3])

with match_level_selection_masst:
    match_level_masst = st.selectbox(
        "Matching level between tree and StructureMASST results",
        options=match_level_options.keys(),
        format_func=lambda x: match_level_options[x],
        index=0,
        help="Select the taxonomic level to match your IDs to in the tree.",
    )

# Built-in tree assets
if tree_choice != "upload":
    tree_path = os.path.join(tree_directory, tree_choice) + ".nwk"
    feature_path = os.path.join(tree_directory, tree_choice) + ".tsv"
else:
    # Upload flow
    tree_upload, match_id_selection, match_id_prefix, _ = st.columns([3,3,3,3])

    # Warning that this was not implemented yet
    st.warning("This feature is not implemented yet.")

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

    with match_id_prefix:
        id_prefix = st.text_input(
            "If your tree uses prefixed IDs (e.g. 'ott12345' or 'ncbi12345'), enter the prefix here (e.g. 'ott' or 'ncbi'). Otherwise, leave blank.",
            value="",
            help="Enter the prefix used in your tree IDs, if any.",
        )
    # NOTE: leaving feature_path unchanged here per your current behavior

# ----------------- HELPERS: mtime-aware cache loaders -----------------
def _maybe_load_bytes(path_str: str, state_key: str, mtime_key: str):
    p = Path(path_str)
    if not p.exists():
        return
    m = p.stat().st_mtime
    if st.session_state.get(mtime_key) != m:
        st.session_state[state_key] = p.read_bytes()
        st.session_state[mtime_key] = m

def _maybe_load_tsv(path_str: str, state_key: str, mtime_key: str):
    p = Path(path_str)
    if not p.exists():
        return
    m = p.stat().st_mtime
    if st.session_state.get(mtime_key) != m:
        st.session_state[state_key] = pd.read_csv(p, sep="\t")
        st.session_state[mtime_key] = m

# ----------- PRELOAD any existing results so they show even before running -----------
_maybe_load_bytes(tree_png_path,     "tree_png_bytes",    "tree_png_mtime")
_maybe_load_tsv(metadata_tsv_path,   "metadata_df",       "metadata_tsv_mtime")
_maybe_load_bytes(empress_zip_path,  "empress_zip_bytes", "empress_zip_mtime")

lifemasst_button, _ = st.columns([9,3])

with lifemasst_button:
    if st.button(
        f"Run LifeMASST",
        key="lifemasst_btn",
    ):
        # Tracking this action
        try:
            umami.new_event(event_name="LifeMASST Button Clicked")
        except Exception as e:
            print(f"Error tracking event: {e}")

        try:
            with st.spinner("Running LifeMASST…"):
                subprocess.run(
                    ["nextflow", "run", NF_PATH,
                     "--input_molecules", molecule_path_abs,
                     "--structureMASST_input_file", structuremasst_path_abs,
                     "--tree_path", tree_path,
                     "--tree_features", feature_path,
                     "--tax_id", match_id,
                     "--tax_id_prefix", id_prefix,
                     "--masst_matchLevel", match_level_masst,
                     "--wikidata_matchLevel", match_level_masst,
                     "--output_folder", out_dir_abs],
                    check=True
                )

                # After the workflow finishes, refresh caches from disk (mtime-based)
                _maybe_load_bytes(tree_png_path,     "tree_png_bytes",    "tree_png_mtime")
                _maybe_load_tsv(metadata_tsv_path,   "metadata_df",       "metadata_tsv_mtime")
                _maybe_load_bytes(empress_zip_path,  "empress_zip_bytes", "empress_zip_mtime")

        except subprocess.CalledProcessError as e:
            st.error("LifeMASST failed. See logs below.")
            with st.expander("Show logs"):
                st.code(e.stdout or "", language="text")
                st.code(e.stderr or "", language="text")

# ----------------- ALWAYS RENDER FROM SESSION STATE (persists across reruns) -----------------
img_bytes = st.session_state.get("tree_png_bytes")
if img_bytes is not None:
    #st.image(img_bytes, caption="LifeMASST Tree Plot", use_container_width=True)
    col_img, col_empty = st.columns([3, 1])

    with col_img:
        st.image(
            img_bytes,
            caption="LifeMASST Tree Plot",
            use_container_width=True,
        )

df = st.session_state.get("metadata_df")
if df is not None:
    st.markdown("### Merged Metadata")
    st.dataframe(df)

    st.success("Download an interactive LifeMASST plot below.")

    zip_bytes = st.session_state.get("empress_zip_bytes")
    st.download_button(
        label="📥 Download Empress Results",
        data=zip_bytes if zip_bytes is not None else b"",
        file_name="empress_results.zip",
        mime="application/zip",
        disabled=zip_bytes is None,
        key="download_empress_results"
    )



