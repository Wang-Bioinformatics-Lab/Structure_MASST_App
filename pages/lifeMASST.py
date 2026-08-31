import os
import time
import uuid
import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html


from dotenv import load_dotenv

from bin.run_masstRecords_queries import _get_fetcher
from bin.shared_data import get_molecule_classes_cached
from bin.streamlit_fragment_LifeMASST import prepare_lifemasst_input, setup_lifemasst_files
from bin.streamlit_prepare import ensure_ready_ui, render_status
from bin.streamlit_search_ui import (
    build_query_table,
    has_search_input,
    render_search_inputs,
    run_structuremasst_search,
)

import importlib.util

# Tracking
import umami
umami.set_url_base("https://analytics-api.gnps2.org/")
umami.set_website_id('032bfca4-a353-4586-b637-8908d8b71c85')
umami.set_hostname('analytics-api.gnps2.org')

html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="74665d88-3b9d-4812-b8fc-7f55ceb08f11"></script>', width=0, height=0)
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="032bfca4-a353-4586-b637-8908d8b71c85"></script>', width=0, height=0)

HERE = os.path.dirname(__file__)
NF_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'nf_workflow.nf'))

# Write the page label
st.set_page_config(
    page_title="LifeMASST",
    page_icon="🧬",
    layout="wide",
)

# st.logo("logo_LifeMASST.png", icon_image="logo_LifeMASST.png")


st.markdown("""
<style>
/* Make the sidebar header area taller so the logo has room */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  height: 140px;              /* adjust as needed */
  padding-top: 8px;
  padding-bottom: 8px;
}

/* Enlarge the logo image in the sidebar header */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] img {
  height: 120px !important;   /* main control: set the target height */
  width: auto !important;     /* keep aspect ratio */
  display: block;
  margin: 0 auto;             /* center horizontally */
}

/* (Optional) control the tiny icon when the sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"] img {
  height: 28px !important;
  width: auto !important;
}
</style>
""", unsafe_allow_html=True)


# This will have to be added to every page, or imported from a common module
st.sidebar.markdown(
    """
    <span style="font-size:0.85em;">
    <strong>Contributors</strong><br>
    Yasin El Abiead (BOKU University)<br>
    Mingxun Wang (UCR)<br>
    </span>
    """,
    unsafe_allow_html=True
)


# ------------------------------
# Load config
# ------------------------------
config_path = "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

load_dotenv("keys.env")
SMARTS_API_KEY = os.getenv("SMARTS_API_KEY", "")

sqlite_path = config.PATH_TO_SQLITE
api_endpoint = config.MASSTRECORDS_ENDPOINT
timeout = config.MASSTRECORDS_TIMEOUT

fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
_molecule_classes_cache = get_molecule_classes_cached(fetch)

# ------------------------------
# Session / folders
# ------------------------------
def get_session_hash() -> str:
    if "_session_hash" not in st.session_state:
        raw = f"{uuid.uuid4()}-{time.time()}-{os.urandom(16).hex()}"
        st.session_state["_session_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return st.session_state["_session_hash"]

sid = get_session_hash()

if "_session_output_folder" not in st.session_state:
    st.session_state["_session_output_folder"] = f"sessionoutput/{sid}"

output_folder = st.session_state["_session_output_folder"]
os.makedirs(output_folder, exist_ok=True)

lifemasst_folder = os.path.join(output_folder, "lifemasst")
os.makedirs(lifemasst_folder, exist_ok=True)

molecule_path = os.path.join(lifemasst_folder, "structuremasst_input.tsv")
structuremasst_path = os.path.join(lifemasst_folder, "lifemasst_input_summary.tsv")
tree_directory = os.path.abspath(os.path.join(HERE, '..', 'external', 'LifeMASST', 'data'))


# ------------------------------
# Detect whether LifeMASST was already set up through StructureMASST
# ------------------------------
lifemasst_files_exist = (
    os.path.exists(os.path.join(lifemasst_folder, "structuremasst_input.tsv"))
    and os.path.exists(os.path.join(lifemasst_folder, "lifemasst_input_summary.tsv"))
)

lifemasst_already_prepared_from_structuremasst = (
    lifemasst_files_exist
    and st.session_state.get("lifemasst_prepared_source") == "structuremasst"
)

# ------------------------------
# State init for shortcut input
# ------------------------------
for k, v in [
    ("lm_name_query", ""),
    ("lm_last_fetched_query", None),
    ("lm_name_suggestions", []),
    ("lm_name_choice", None),
    ("lm_usi_input", ""),
    ("lm_smiles_input", ""),
    ("lm_name_warning", None),
    ("lm_structure_editor_open", False),
    ("lm_new_smiles", ""),
    ("lm_class_label", ""),
]:
    st.session_state.setdefault(k, v)

# keep separate query/raw results for the shortcut mode
st.session_state.setdefault("lifemasst_shortcut_query_table", None)
st.session_state.setdefault("lifemasst_shortcut_raw_results", None)

# ------------------------------
# Helpers
# ------------------------------

# The search itself lives in bin/streamlit_search_ui.py so the GeoMASST page can
# run exactly the same one.
run_shortcut_structuremasst = run_structuremasst_search
# ------------------------------
# Compact shortcut input UI
# ------------------------------
uploaded_file = None
shortcut_smiles_type = None
effective_smiles = ""

show_shortcut_inputs = not lifemasst_already_prepared_from_structuremasst

def render_shortcut_inputs_and_advanced():
    """The shared StructureMASST search controls, keyed for this page."""
    return render_search_inputs(prefix="lm_", molecule_classes=_molecule_classes_cache)
if lifemasst_already_prepared_from_structuremasst:
    st.info("LifeMASST input has already been prepared from StructureMASST for this session. Simply select a tree from the dropdown and hit Run LifeMASST!")
    shortcut_ui_values = {
        "uploaded_file": None,
        "shortcut_smiles_type": None,
        "effective_smiles": "",
        "shortcut_mode": "FASSTrecords",
        "min_cosine": 0.70,
        "min_peaks": 5,
        "min_rank": 0,
        "shortcut_searchtype_ui": "Exact structure match",
        "allowed_formula": "",
        "allowed_elements": "",
        "tanimoto_cutoff": "0.8",
        "prec_tol": 0.02,
        "frag_tol": 0.02,
        "do_modification_search": False,
        "modification_formula": "",
        "modification_mass_text": "",
        "do_elimination": True,
        "do_addition": True,
        "modification_condition": None,
    }

    with st.expander("Discard StructureMASST results and start fresh.", expanded=False):
        shortcut_ui_values = render_shortcut_inputs_and_advanced()
else:
    shortcut_ui_values = render_shortcut_inputs_and_advanced()

uploaded_file = shortcut_ui_values["uploaded_file"]
shortcut_smiles_type = shortcut_ui_values["shortcut_smiles_type"]
effective_smiles = shortcut_ui_values["effective_smiles"]
shortcut_mode = shortcut_ui_values["shortcut_mode"]
min_cosine = shortcut_ui_values["min_cosine"]
min_peaks = shortcut_ui_values["min_peaks"]
min_rank = shortcut_ui_values["min_rank"]
shortcut_searchtype_ui = shortcut_ui_values["shortcut_searchtype_ui"]
allowed_formula = shortcut_ui_values["allowed_formula"]
allowed_elements = shortcut_ui_values["allowed_elements"]
tanimoto_cutoff = shortcut_ui_values["tanimoto_cutoff"]
prec_tol = shortcut_ui_values["prec_tol"]
frag_tol = shortcut_ui_values["frag_tol"]
do_modification_search = shortcut_ui_values["do_modification_search"]
modification_formula = shortcut_ui_values["modification_formula"]
modification_mass_text = shortcut_ui_values["modification_mass_text"]
do_elimination = shortcut_ui_values["do_elimination"]
do_addition = shortcut_ui_values["do_addition"]
modification_condition = shortcut_ui_values["modification_condition"]

# ------------------------------
# Existing / output paths
# ------------------------------
empress_zip_path = os.path.join(output_folder, "lifemasst", "empress_results.zip")
tree_heatmap_path = os.path.join(output_folder, "lifemasst", "tree_heatmap.html")
metadata_tsv_path = os.path.join(output_folder, "lifemasst", "merged_metadata.tsv")

tree_labels = {
    "trees/OTL_prepared/labelled_supertree_subset_prepped": "Open Tree of Life",
    "trees/timetree_prepared/TimeTree_subset_prepped": "TimeTree",
    "trees/avian_prepared/OW2019_timetree_alltaxa_with_root_constraint": "Bird Tree (time tree)",
    "trees/avian_prepared/OW2019_CYB_ND2_estBL_alltaxa": "Bird Tree (molecular tree)",
    "trees/mammalian_prepared/Foley2022_Concatenation_HRA_neutral_241_10miss_rooted": "Mammal Tree (molecular tree)",
    "trees/fish_prepared/fish_raxml_12k": "Fish Tree (molecular tree)",
    "trees/fish_prepared/fish_treepl_12k": "Fish Tree (time tree)",
    "trees/plants_prepared/1kp_astral_alltaxa_FAA": "Plant Tree (molecular tree)",
    "upload": "Custom Uploaded Tree",
}

# read molecule file and deduplicate by name if it exists already
if os.path.exists(molecule_path):
    df_molecule = pd.read_csv(molecule_path, sep="\t")
    df_molecule_unique = df_molecule.drop_duplicates(subset=["name"])
    molecule_path = os.path.join(lifemasst_folder, "structuremasst_input_unique.tsv")
    df_molecule_unique.to_csv(molecule_path, sep="\t", index=False)

molecule_path_abs = os.path.abspath(molecule_path)
out_dir_abs = os.path.abspath(lifemasst_folder)
structuremasst_path_abs = os.path.abspath(structuremasst_path)

match_id_options = {
    "ott": "OpenTreeOfLifeTaxonomyID",
    "ncbi": "NCBI Taxonomic ID",
}

tree_selection, _, _, _ = st.columns([3, 3, 3, 3])
with tree_selection:
    tree_choice = st.selectbox(
        "Select a tree for LifeMASST",
        options=list(tree_labels.keys()),
        format_func=lambda x: tree_labels.get(x, x),
        index=0,
        help="Select the phylogenetic or taxonomic tree to use for LifeMASST analysis.",
    )

if tree_choice not in ["trees/OTL_prepared/labelled_supertree_subset_prepped", "trees/OTL_prepared/labelled_supertree_full_prepped"]:
    match_level_masst_hardcoded = "NCBISpecies"
    match_level_wikidata_hardcoded = "NCBISpecies"
else:
    match_level_masst_hardcoded = "NCBIFamily"
    match_level_wikidata_hardcoded = "NCBIFamily"

min_specificity_options = {
    "":        "No minimum (all flexible matches)",
    "species": "Species or finer",
    "genus":   "Genus or finer",
    "family":  "Family or finer",
    "order":   "Order or finer",
    "class":   "Class or finer",
    "phylum":  "Phylum or finer",
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
        "Source: Kumar et al. (2022), <i>Molecular Biology and Evolution</i> 39(8). https://doi.org/10.1093/molbev/msac174"
    )

def format_message(msg: str) -> str:
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

if message:
    st.markdown(format_message(message), unsafe_allow_html=True)

min_spec_col, _, _, _ = st.columns([3, 3, 3, 3])
with min_spec_col:
    min_specificity = st.selectbox(
        "Minimum StructureMASST match specificity",
        options=list(min_specificity_options.keys()),
        format_func=lambda x: min_specificity_options[x],
        index=0,
        help=(
            "Only show tree nodes where the StructureMASST flexible match is at least this specific. "
            "'No minimum' includes all matches, even kingdom-level propagated hits."
        ),
    )

if tree_choice != "upload":
    tree_path = os.path.join(tree_directory, tree_choice) + ".nwk"
    feature_path = os.path.join(tree_directory, tree_choice) + ".tsv"
else:
    st.warning("Custom uploaded tree is still not implemented.")
    st.stop()

# ------------------------------
# mtime-aware loaders
# ------------------------------
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

def _maybe_load_text(path_str: str, state_key: str, mtime_key: str):
    p = Path(path_str)
    if not p.exists():
        return
    m = p.stat().st_mtime
    if st.session_state.get(mtime_key) != m:
        st.session_state[state_key] = p.read_text(encoding="utf-8")
        st.session_state[mtime_key] = m

_maybe_load_text(tree_heatmap_path, "tree_heatmap_html", "tree_heatmap_mtime")
_maybe_load_tsv(metadata_tsv_path, "metadata_df", "metadata_tsv_mtime")
_maybe_load_bytes(empress_zip_path, "empress_zip_bytes", "empress_zip_mtime")

# ------------------------------
# Reference data: shipped vs fetched, and a way to build it up front
# ------------------------------
render_status("lifemasst")

# ------------------------------
# Single button: Run LifeMASST
# ------------------------------
lifemasst_button, _ = st.columns([9, 3])

with lifemasst_button:
    if st.button("Run LifeMASST", key="lifemasst_btn"):
        try:
            umami.new_event(event_name="LifeMASST Button Clicked")
        except Exception:
            pass

        # On a fresh checkout the trees are already here (they are committed),
        # but the ReDU, Wikidata and plant-distribution tables are not. Build
        # them once, in the open, before the first search rather than during it.
        if not ensure_ready_ui("lifemasst"):
            st.stop()

        try:
            with st.spinner("Running LifeMASST…"):
                # -----------------------------------
                # Decide input source:
                # 1) shortcut input on this page, if provided
                # 2) existing files from StructureMASST session
                # -----------------------------------
                if st.session_state.get("lifemasst_prepared_source") == "structuremasst":
                    has_shortcut_input = False
                else:
                    has_shortcut_input = has_search_input("lm_", shortcut_ui_values)

                if has_shortcut_input:
                    df_input, problem = build_query_table("lm_", shortcut_ui_values)
                    if problem:
                        st.error(problem)
                        st.stop()

                    shortcut_result = run_shortcut_structuremasst(
                        df_input=df_input,
                        mode=shortcut_mode,
                        min_cosine=float(min_cosine),
                        min_peaks=int(min_peaks),
                        min_rank=int(min_rank),
                        tanimoto_cutoff=tanimoto_cutoff,
                        prec_tol=float(prec_tol),
                        frag_tol=float(frag_tol),
                        do_modification_search=bool(do_modification_search),
                        modification_formula=modification_formula,
                        modification_mass_text=modification_mass_text,
                        do_elimination=bool(do_elimination),
                        do_addition=bool(do_addition),
                        modification_condition=modification_condition,
                    )

                    st.session_state["lifemasst_shortcut_query_table"] = shortcut_result["query_table"]
                    st.session_state["lifemasst_shortcut_raw_results"] = shortcut_result["raw_results"]
                    st.session_state.setdefault("lifemasst_prepared_source", None)

                    prepared_input = prepare_lifemasst_input(df_input)

                    redu_tables = {
                        qname: pair["redu"]
                        for qname, pair in shortcut_result["raw_results"].items()
                        if isinstance(pair.get("redu"), pd.DataFrame)
                    }

                    in_path, out_path = setup_lifemasst_files(
                        input_file=prepared_input,
                        structureMASST_op_folder=output_folder,
                        redu_tables=redu_tables,
                    )

                    st.session_state["lifemasst_prepared_source"] = "lifemasst_shortcut"

                    if out_path is None:
                        st.error("LifeMASST setup failed because no valid raw-data summaries could be created.")
                        st.stop()

                else:
                    # fallback: use already prepared files from StructureMASST page
                    in_path = os.path.join(lifemasst_folder, "structuremasst_input.tsv")
                    out_path = os.path.join(lifemasst_folder, "lifemasst_input_summary.tsv")

                    if not os.path.exists(in_path) or not os.path.exists(out_path):
                        st.error(
                            "No shortcut input was provided, and no prepared StructureMASST LifeMASST input files were found. "
                            "Either enter input above or run StructureMASST first."
                        )
                        st.stop()

                # Deduplicate molecule file by name before running Nextflow
                df_molecule = pd.read_csv(in_path, sep="\t")
                df_molecule_unique = df_molecule.drop_duplicates(subset=["name"]).copy()
                unique_path = os.path.join(lifemasst_folder, "structuremasst_input_unique.tsv")
                df_molecule_unique.to_csv(unique_path, sep="\t", index=False)

                molecule_path_abs = os.path.abspath(unique_path)
                structuremasst_path_abs = os.path.abspath(out_path)

                project_root = os.path.abspath(os.path.join(HERE, ".."))
                nf_work_dir  = os.path.join(project_root, "work", "work")
                nf_env_dir   = os.path.join(project_root, "work", "work_env")
                os.makedirs(nf_work_dir, exist_ok=True)
                os.makedirs(nf_env_dir,  exist_ok=True)
                override_cfg_path = os.path.join(out_dir_abs, "nf_override.config")
                with open(override_cfg_path, "w") as _cfg:
                    _cfg.write(
                        f'workDir = "{nf_work_dir}"\n'
                        f'conda {{\n'
                        f'    enabled = true\n'
                        f'    cacheDir = "{nf_env_dir}"\n'
                        f'}}\n'
                    )

                nf_cmd = [
                    "nextflow", "run", NF_PATH,
                    "-c", override_cfg_path,
                    "--input_molecules", molecule_path_abs,
                    "--structureMASST_input_file", structuremasst_path_abs,
                    "--tree_path", tree_path,
                    "--tree_features", feature_path,
                    "--tax_id", match_id,
                    "--tax_id_prefix", id_prefix,
                    "--masst_matchLevel", match_level_masst_hardcoded,
                    "--wikidata_matchLevel", match_level_wikidata_hardcoded,
                    "--output_folder", out_dir_abs,
                ]
                if min_specificity:
                    nf_cmd += ["--min_specificity", min_specificity]
                subprocess.run(nf_cmd, check=True)

                _maybe_load_text(tree_heatmap_path, "tree_heatmap_html", "tree_heatmap_mtime")
                _maybe_load_tsv(metadata_tsv_path, "metadata_df", "metadata_tsv_mtime")
                _maybe_load_bytes(empress_zip_path, "empress_zip_bytes", "empress_zip_mtime")

        except subprocess.CalledProcessError as e:
            st.error("LifeMASST failed. See logs below.")
            with st.expander("Show logs"):
                st.code(e.stdout or "", language="text")
                st.code(e.stderr or "", language="text")
        except Exception as e:
            st.error(str(e))

# ------------------------------
# Render results
# ------------------------------
tree_html = st.session_state.get("tree_heatmap_html")
if tree_html is not None:
    components.html(tree_html, height=900, scrolling=False)

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