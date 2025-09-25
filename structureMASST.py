import os
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import importlib.util
from rdkit import Chem
from PIL import Image
import base64
import io
from bin.workflow_stepwise import retrieve_raw_data_matches 
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables
from bin.match_smiles import detect_smiles_or_smarts, neutralize_atoms, tautomerize_smiles
from bin.pubchem_handling  import pubchem_autocomplete, name_to_cid, cid_to_canonical_smiles
from bin.plotting import raw_data_sankey, export_hits_map
from bin.linkouts import build_dashboard_eic_url, build_spectraresolver_link
from bin.smarts_api import query_smarts
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from formula_validation.Formula import Formula
import requests
import re
import urllib.parse as _url
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Union, List, Dict
import subprocess
import uuid
import time
import numpy as np
from dotenv import load_dotenv
from streamlit_ketcher import st_ketcher

# Load .env file
load_dotenv("keys.env")
SMARTS_API_KEY = os.getenv("SMARTS_API_KEY", "")

st.markdown("""
<style>
/* Make the main content truly wide and add fluid side padding */
.block-container {
  max-width: 100%;
  padding-left: 2vw;
  padding-right: 2vw;
  padding-top: 1rem;
}

/* Responsive typography scale for buttons */
:root{
  /* font-size automatically clamps between 12px and 20px, grows with viewport */
  --btn-font-size: clamp(12px, 1.2vw, 20px);
  /* padding grows a bit with viewport */
  --btn-pad-y: clamp(6px, 0.7vw, 14px);
  --btn-pad-x: clamp(10px, 1.4vw, 24px);
  --btn-radius: 14px;
}

/* Style all Streamlit buttons (works for st.button and st.link_button) */
.stButton > button, .stDownloadButton > button {
  width: 100%;               /* fill the column — column defines relative width */
  font-size: var(--btn-font-size);
  padding: var(--btn-pad-y) var(--btn-pad-x);
  border-radius: var(--btn-radius);
  line-height: 1.2;
}

/* Optional: make selectboxes/inputs feel consistent */
.stTextInput input, .stSelectbox > div > div, .stNumberInput input {
  font-size: clamp(12px, 1.1vw, 18px);
}

/* Small-screen tweaks */
@media (max-width: 768px) {
  .block-container { padding-left: 3vw; padding-right: 3vw; }
  .stButton > button, .stDownloadButton > button { line-height: 1.25; }
}
</style>
""", unsafe_allow_html=True)

# Add a tracking token
html('<script async defer data-website-id="<your_website_id>" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)

# Write the page label
st.set_page_config(
    page_title="StructureMASST", 
    layout="wide",
    page_icon="🔎",
)

st.logo("logo.png", icon_image="logo.png")

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
    Yasin El Abiead (UCSD)<br>
    Wilhan Nunes (UCSD)<br>
    Mingxun Wang (UCR)<br>
    </span>
    """,
    unsafe_allow_html=True
)

try:
    from rdkit.Chem import Draw
    _RD_DRAW_AVAILABLE = True
except ImportError:
    _RD_DRAW_AVAILABLE = False

# datasette masst_records.sqlite --setting max_returned_rows 1000000 --setting sql_time_limit_ms 60000

# — load config —
config_path = "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)



def get_session_hash() -> str:
    if "_session_hash" not in st.session_state:
        raw = f"{uuid.uuid4()}-{time.time()}-{os.urandom(16).hex()}"
        st.session_state["_session_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return st.session_state["_session_hash"]

# create unique session id
sid = get_session_hash()
st.write("Session:", sid)
output_folder = f"sessionoutput/{sid}"
os.makedirs(output_folder, exist_ok=True)

st.session_state["_session_output_folder"] = output_folder

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


# — SMILES or CSV input —
col_name, col_or1, col_smiles, col_or2, col_csv = st.columns([4, 1, 4, 1, 4])

# ---------- state init ----------
for k, v in [
    ("name_query", ""),
    ("last_fetched_query", None),
    ("name_suggestions", []),
    ("name_choice", None),
    ("smiles_input", ""),
    ("name_warning", None),
    ("structure_editor_open", False),
    ("new_smiles", ""),
]:
    st.session_state.setdefault(k, v)

def tautomerize_neutralize_smiles(smiles: str) -> str:
    """Tautomerize and neutralize a SMILES string."""
    try:
        smi = tautomerize_smiles(smiles)
    except Exception as e:
        print(f"Tautomerization failed: {e}")
        smi = smiles
    try:
        smi = neutralize_atoms(smi)
    except Exception as e:
        print(f"Neutralization failed: {e}")
    return smi

def _resolve_name_to_smiles(selected_name: str):
    """Resolve name → CID → Canonical SMILES, enforce single-component rule."""
    st.session_state["name_warning"] = None  # reset
    if not selected_name:
        return
    cid = name_to_cid(selected_name)
    smiles = cid_to_canonical_smiles(cid) if cid else None

    # Rule: must exist and must NOT contain '.'
    if smiles and "." not in smiles:
        st.session_state["smiles_input"] = smiles
    else:
        st.session_state["smiles_input"] = ""
        st.session_state["name_warning"] = (
            "PubChem entry does not represent a singular molecule "
            "(no SMILES available or multi-component SMILES containing '.')."
        )

with col_name:
    # Plain text_input (same style as SMILES). Pressing Enter triggers a rerun.
    name_query = st.text_input(
        "Type a chemical name to search PubChem",
        key="name_query",
        placeholder="e.g., diazepam, caffeine, etc.",
    )

    # If the query changed (e.g., after Enter), fetch suggestions once
    if name_query and name_query != st.session_state["last_fetched_query"]:
        suggestions = pubchem_autocomplete(name_query) or []
        st.session_state["name_suggestions"] = suggestions
        st.session_state["last_fetched_query"] = name_query

        # Preselect top suggestion and immediately try to populate SMILES
        if suggestions:
            st.session_state["name_choice"] = suggestions[0]
            _resolve_name_to_smiles(suggestions[0])

    # Show dropdown only when we have suggestions
    suggestions = st.session_state["name_suggestions"]
    if suggestions:
        def _on_choice_change():
            _resolve_name_to_smiles(st.session_state["name_choice"])
            st.session_state["structure_editor_open"] = False
            st.session_state["new_smiles"] = smiles_input

        st.selectbox(
            "Suggestions",
            options=suggestions,
            key="name_choice",
            index=(
                suggestions.index(st.session_state["name_choice"])
                if st.session_state["name_choice"] in suggestions else 0
            ),
            on_change=_on_choice_change,
        )

    # Show warning (if any) right under the name controls
    if st.session_state["name_warning"]:
        st.warning(st.session_state["name_warning"])

with col_or1:
    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

with col_smiles:
    smiles_input = st.text_input(
        "SMILES/SMARTS",
        key="smiles_input",  # gets auto-populated only if single-component
        placeholder="Enter SMILES or SMARTS",
        help="Enter a valid SMILES or SMARTS string. For SMARTS string creation, you can use third-party tools like SMARTSPlus https://smarts.plus/create"
    )

with col_or2:
    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

with col_csv:
    #add spacer to align widgets
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    # if st.checkbox("Upload Batch file", key="batch_search"):
    with st.popover("Add batch file", icon=":material/file_upload:"):
        uploaded_file = st.file_uploader("Drop CSV file for batch search (smiles and name columns)", type=["csv"])

# — Display molecule if valid SMILES/SMARTS —
#@st.dialog("Visualize Structure")
def show_structure_dialog(mol):
    # If input is a BytesIO (image), show it directly; else, use RDKit rendering
    if isinstance(mol, io.BytesIO):
        #st.info("This is a SMARTS pattern, which may represent multiple structures.")
        st.image(mol, width=500)
    elif mol_to_base64_img:
        st.markdown(mol_to_base64_img(mol), unsafe_allow_html=True)
    else:
        st.info("RDKit not available, cannot render structure.")

try:
    from rdkit.Chem import Draw
    def mol_to_base64_img(mol, size=(300, 300)):
        try:
            img = Draw.MolToImage(mol, size=size)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"<img src='data:image/png;base64,{img_str}' style='margin-top:1em;'/>"
        except Exception:
            return f"<p style='color:red;'>Failed to draw molecule image.</p>"
except ImportError:
    mol_to_base64_img = None


# --- render logic ---
smiles_type = None  # Ensure smiles_type is always defined
default_search_index = 0 # default to exact match search



if smiles_input:
    smiles_input = smiles_input.strip()
    # Use effective SMILES (from editor if available) for type detection
    effective_smiles = st.session_state.get('new_smiles', '') or smiles_input
    smiles_type = detect_smiles_or_smarts(effective_smiles)
    
    if smiles_type == "smiles":
        edit_button = st.button("Edit Structure", icon=":material/edit:")
        if edit_button:
            st.session_state['structure_editor_open'] = True

        if edit_button or st.session_state['structure_editor_open']:
            with st.expander("Structure Editor", expanded=True):
                st.info(f"You can edit the structure below and click Apply to update it.")
                
                # Add a close button for persistent editor
                if st.session_state['structure_editor_open']:
                    if st.button("Close Structure Editor", icon=":material/close:", help="Warning: You will :red-badge[lose your changes]"):
                        st.session_state['structure_editor_open'] = False
                        st.rerun()
                
                # Use existing SMILES if available, otherwise use input
                new_smiles = st_ketcher(effective_smiles)

        else:
            new_smiles = smiles_input

        st.session_state['new_smiles'] = new_smiles


        # Structure rendering with RDKit
        if _RD_DRAW_AVAILABLE and new_smiles:
            mol = Chem.MolFromSmiles(new_smiles)
            if mol:
                st.markdown(mol_to_base64_img(mol), unsafe_allow_html=True)
            else:
                st.warning("Could not parse SMILES for rendering.")

    elif smiles_type == "smarts":
        job_id = str(uuid.uuid4())
        # For SMARTS, use the effective SMILES
        response = query_smarts(effective_smiles, api_key=SMARTS_API_KEY, job_id=job_id, file_format="png")
        if response and 'result' in response and 'image' in response['result']:
            image_data = base64.b64decode(response['result']['image'])
            bytes_io = io.BytesIO(image_data)
            #if st.button("View SMARTS", icon=":material/visibility:"):
            show_structure_dialog(bytes_io)
        else:
            st.warning("Failed to retrieve SMARTS image from the API. Please try again.")
    else:
        st.warning("Not a valid SMILES/SMARTS")

# — mode selection UI —
search_options = ["Exact structure match", "Substructure match", "Tanimoto similarity"]
default_search_index = 0 
col_a1, col_b2, _ = st.columns([2,1,2])
if smiles_type and smiles_type == 'smarts':
    #st.info("SMARTS input detected. Use 'Substructure match' for search.", icon=':material/info:')
    search_options = ["Substructure match"]
with col_a1:
    searchtype_option = st.radio(
        "Find available MS/MS spectra", 
        search_options, 
        horizontal=True, index=default_search_index
    )

if searchtype_option == "Tanimoto similarity":
    with col_b2:
        tanimoto_cutoff = st.text_input("Tanimoto threshold", value="0.8", key="tanimoto_threshold")

# Map UI option to backend value
if searchtype_option == "Exact structure match":
    searchtype_option = "exact"
elif searchtype_option == "Substructure match":
    searchtype_option = "substructure"
elif searchtype_option == "Tanimoto similarity":
    searchtype_option = "tanimoto"


# — run the search —
if st.button("Get Available Spectra", icon=':material/search:'):
    # Use new_smiles from structure editor if available, otherwise use original input
    effective_smiles = st.session_state.get('new_smiles', '') or smiles_input
    if smiles_type == "smiles":
        effective_smiles = tautomerize_neutralize_smiles(effective_smiles)
    
    st.write(f"Using SMILES: {effective_smiles}")

    with st.spinner("Finding spectra..."): 
        # Reset upstream & downstream state
        for key in [
            "selected_queries",
            "df_library_conflicts",
            "grouped_results",
            "raw_results",
            "molecule_overview"
        ]:
            st.session_state.pop(key, None)

        # initialize what's needed 
        st.session_state.selected_queries = {}
        st.session_state.df_library_conflicts = None
        st.session_state.grouped_results = {}
        st.session_state.molecule_overview = {}

        # organize input structure queries
        smiles_list = []
        if smiles_input:
            # Use new_smiles from structure editor if available, otherwise use original input
            effective_smiles = st.session_state.get('new_smiles', '') or smiles_input
            
            smiles_list = [effective_smiles]
            name_list = ['Input_query']
        elif uploaded_file is not None:
            df_in = pd.read_csv(uploaded_file)
            if "smiles" in df_in.columns and "name" in df_in.columns:
                smiles_list = df_in["smiles"].dropna().tolist()
                name_list = df_in["name"].dropna().tolist()
            else:
                st.warning("CSV must contain a 'smiles' and 'name' column.")
                st.stop()
        else:
            st.warning("Please enter a SMILES or upload a CSV.")
            st.stop()

        # process each input structure query separately to retrieve spectra
        grouped_results = defaultdict(dict)
        molecule_overview = defaultdict(dict)
        for smi, name in zip(smiles_list, name_list):
            try:
                df_library_structurematch = get_library_table(
                    smiles=smi,
                    searchtype=searchtype_option,
                    tanimoto_threshold=tanimoto_cutoff if searchtype_option == "tanimoto" else None,
                    sqlite_path=config.PATH_TO_SQLITE,
                    api_endpoint=config.MASSTRECORDS_ENDPOINT,
                    timeout=config.MASSTRECORDS_TIMEOUT
                )
                print(f"Retrieved {len(df_library_structurematch)} records from MASSTrecords")
                if df_library_structurematch.empty:
                    st.info(f"No library entries found for: {smi}")
                    continue
                # Setup for library conflicts. this is currently not doing anything
                df_library_conflicts = pd.DataFrame()
                df_library_conflicts["inchikey_first_block"] = pd.Series(dtype=str)
            except Exception as e:
                st.error(f"Error for {smi}: {e}")

            # Setup for library conflicts. this is currently not doing anything
            print(f"Processing {name} with {len(df_library_structurematch)} matches")
            overview = []
            for ik in df_library_structurematch["inchikey_first_block"].unique():
                sub_struct = df_library_structurematch[df_library_structurematch["inchikey_first_block"] == ik].copy()
                sub_conf   = df_library_conflicts[df_library_conflicts["inchikey_first_block"] == ik].copy()
                grouped_results[name][ik] = {"structure": sub_struct, "conflicts": sub_conf}
                st.session_state.selected_queries[ik] = list(sub_struct["query_spectrum_id"].unique())


                # pick most common Compound_Name (tie-break by len≈20 & fewest special chars)
                names = sub_struct["Compound_Name"].dropna().astype(str)
                if not names.empty:
                    vc = names.value_counts()
                    top = vc.iloc[0]
                    cands = vc[vc == top].index.tolist()
                    def special_count(s): 
                        return len(re.findall(r"[^A-Za-z0-9]", s))
                    best_name = min(cands, key=lambda s: (abs(len(s) - 20), special_count(s)))
                else:
                    best_name = ""

                # grab first SMILES
                smiles = sub_struct["Smiles"].dropna().astype(str)
                inchikey_first_block = sub_struct["inchikey_first_block"].dropna().astype(str)
                first_smi = smiles.iloc[0] if not smiles.empty else ""
                ikb = inchikey_first_block.iloc[0] if not inchikey_first_block.empty else ""

                overview.append({
                    "Compound_Name": best_name,
                    "inchikey_first_block": ikb,
                    "Smiles": first_smi
                })

            st.session_state.molecule_overview[name] = pd.DataFrame(overview)

        # get results into session state
        st.session_state.grouped_results = grouped_results


# render outer tabs for each structure query
if "grouped_results" in st.session_state and st.session_state["grouped_results"]:
    
    with st.expander("Available Library Entries", expanded=True):
        st.markdown("### Available Library Entries")
        name_tabs = st.tabs(list(st.session_state.grouped_results.keys()))

        for fig_id, (name, name_tab) in enumerate(zip(st.session_state.grouped_results.keys(), name_tabs)):
            with name_tab:

                tables = st.session_state.grouped_results[name]

                # number of molecules = number of 2d inchikey keys
                num_molecules = len(tables)

                # total number of matches across all "structure" tables
                total_matches = sum(
                    tbl["structure"].shape[0]
                    for tbl in tables.values()
                )

                # messages on retrieved spectra
                st.markdown(
                    f"##### Retrieved **{total_matches}** spectra for **{num_molecules}** molecule(s) ({name}).",
                )

                st.markdown(
                    "<div style='font-size:0.9em;'>"
                    "Below you see the public MS/MS spectra available for your search. The higher the spectral diversity, the less blind-spots you will have in public metabolomics raw data."
                    "</div>",
                    unsafe_allow_html=True
                )

                if total_matches == 0:
                    st.info("No spectra available for this structure.")
                    continue
                # Sankey per query structure
                ##########

                # concatenate all the 'structure' dfs under this name
                df_all = pd.concat(
                    [v["structure"] for v in st.session_state.grouped_results[name].values()],
                    ignore_index=True
                )

                # Check if all required columns exist before proceeding
                required_cols = ["msMassAnalyzer", "Ion_Mode", "Adduct", "collision_energy"]
                missing_cols = [col for col in required_cols if col not in df_all.columns]
                if not missing_cols:

                    # Fixing levels
                    stages = ["msMassAnalyzer", "Ion_Mode", "Adduct", "collision_energy"]
                    df_sankey = df_all[stages].dropna()

                    # Create labels for Sankey diagram
                    labels = []
                    for col in stages:
                        labels += df_sankey[col].unique().tolist()
                    labels = list(dict.fromkeys(labels))

                    # get up to 5 colors (should not be more different instrument types than that)
                    ms_cats = df_sankey["msMassAnalyzer"].unique().tolist()
                    palette = px.colors.qualitative.Safe[:5]  
                    color_map = {cat: palette[i] for i, cat in enumerate(ms_cats)}

                    # build links
                    source, target, value, link_colors = [], [], [], []
                    for i in range(len(stages) - 1):
                        for cat in ms_cats:
                            df_cat = df_sankey[df_sankey["msMassAnalyzer"] == cat]
                            grp = (
                                df_cat
                                .groupby([stages[i], stages[i + 1]])
                                .size()
                                .reset_index(name="count")
                            )
                            for _, row in grp.iterrows():
                                source.append(labels.index(row[stages[i]]))
                                target.append(labels.index(row[stages[i + 1]]))
                                value.append(row["count"])
                                link_colors.append(color_map[cat].replace("rgb", "rgba").replace(")", f", {0.3})"))


                    # all nodes light grey with black border
                    node_colors = ["#F2F2F2"] * len(labels)

                    fig = go.Figure(
                        go.Sankey(
                            arrangement="snap",
                            # ← trace‑level label styling
                            textfont=dict(family="Arial, sans-serif", size=12, color="black"),

                            node=dict(
                                label=labels,
                                color=node_colors,
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                            ),
                            link=dict(
                                source=source,
                                target=target,
                                value=value,
                                color=link_colors,
                            ),
                        )
                    )

                    # add stage annotations
                    fig.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        margin=dict(l=60, r=60, t=120, b=20),
                    )

                    # add stage labels above the Sankey diagram
                    stage_labels = ["Mass Analyzer", "Ion Mode", "Adduct", "Collision Energy"]

                    # calculate x positions for each stage label
                    n = len(stage_labels) - 1
                    for i, label in enumerate(stage_labels):
                        x = i / n
                        # xanchor depends on position
                        if i == 0:
                            xanchor = "left"
                        elif i == n:
                            xanchor = "right"
                        else:
                            xanchor = "center"

                        # add annotation for each stage label
                        fig.add_annotation(
                            x=x,
                            y=1.02,
                            xref="paper",
                            yref="paper",
                            text=label,
                            showarrow=False,
                            font=dict(size=14, color="black"),
                            xanchor=xanchor
                        )
                    # update layout
                    config_sankey_download = {
                        "toImageButtonOptions": {"format": "svg", "filename": f"plot_{fig_id}"},
                        "displaylogo": False,
                    }
                    st.plotly_chart(fig, width='stretch', key=f"plot_{fig_id}", config=config_sankey_download)

                    try:
                        fig.write_image(
                            f"{output_folder}/library_sankey_{name}.pdf", 
                            format="pdf", 
                            width=1240, 
                            height=400, 
                            scale=2
                        )
                    except RuntimeError as e:
                        if "Kaleido requires Google Chrome to be installed" in str(e):
                            print("⚠️ Skipping PDF export: Chrome not available for Kaleido.")
                        else:
                            raise

                molecule_overview_df = st.session_state.molecule_overview[name]

                if _RD_DRAW_AVAILABLE:
                    @st.cache_data
                    def smiles_to_datauri(smi: str, size=(500, 500)) -> str:
                        """Render a SMILES to PNG data URI."""
                        mol = Chem.MolFromSmiles(smi)
                        if not mol:
                            return ""
                        img = Draw.MolToImage(mol, size=size)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        return f"data:image/png;base64,{b64}"
                    
                    molecule_overview_df = molecule_overview_df.copy()
                    molecule_overview_df["structure"] = molecule_overview_df["Smiles"].apply(smiles_to_datauri)

                    # reorder columns to Compound_Name, Structure, others
                    cols = ["Compound_Name", "structure"] + [col for col in molecule_overview_df.columns if col not in ["Compound_Name", "structure"]]
                    molecule_overview_df = molecule_overview_df[cols]

                    # display it with clickable links   
                    table_mol = st.dataframe(
                        molecule_overview_df,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="multi-row",
                        width='stretch',                
                        column_config={
                            "structure": st.column_config.ImageColumn(
                                "Structure",                     
                                width=300                  
                            )
                        },
                        key=f"{name}_molecule_table",
                        row_height=250
                    )
                else:
                    table_mol = st.dataframe(
                        molecule_overview_df,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="multi-row",
                        width='stretch',
                        key=f"{name}_molecule_table"
                    )

                # grab list of selected row indices
                selected = table_mol.selection.rows

                # show buttons with actions for selected molecules
                col1_mol_level, col2_mol_level, _ = st.columns([2, 2, 6])
                with col1_mol_level:
                    if st.button("Remove selected molecule(s)", key=f"{name}_mol_remove"):
                        if selected:
                            molecule_overview_df = molecule_overview_df.drop(molecule_overview_df.index[selected])
                            st.session_state.molecule_overview[name] = molecule_overview_df
                            unique_inchikeys = molecule_overview_df["inchikey_first_block"].unique().tolist()
                            
                            st.session_state.grouped_results[name] = {
                                ik: data
                                for ik, data in st.session_state.grouped_results[name].items()
                                if ik in unique_inchikeys
                            }

                        else:
                            st.warning("No rows selected!")
                        st.rerun()

                with col2_mol_level:
                    if st.button("Keep only selected molecule(s)", key=f"{name}_mol_keep"):
                        if selected:
                            molecule_overview_df = molecule_overview_df.iloc[selected].reset_index(drop=True)
                            st.session_state.molecule_overview[name] = molecule_overview_df
                            unique_inchikeys = molecule_overview_df["inchikey_first_block"].unique().tolist()
                            # also remove from grouped_results from same name
                            st.session_state.grouped_results[name] = {
                                ik: data
                                for ik, data in st.session_state.grouped_results[name].items()
                                if ik in unique_inchikeys
                            }
                        else:
                            st.warning("No rows selected!")
                        st.rerun()

                # update the session state with the filtered dataframe
                st.session_state.molecule_overview[name] = molecule_overview_df

                # display each query structures available spectra as table
                with st.expander("Molecules by InChIKey", expanded=True):
                    # grab the keys
                    keys = list(st.session_state.grouped_results[name].keys())

                    max_tabs = 100
                    if len(keys) > max_tabs:
                        st.warning(f"⚠️ Too many InChIKeys ({len(keys)}). Showing only the first {max_tabs}.")
                        keys = keys[:max_tabs]

                    # create a tab for each 2D InChIKey (max 100)
                    ik_tabs = st.tabs(keys)
                    for ik, ik_tab in zip(keys, ik_tabs):
                        with ik_tab:

                            data = st.session_state.grouped_results[name][ik]
                            df0 = data["structure"].copy()


                            df0["spectrum_link"] = df0["query_spectrum_id"].apply(
                                lambda x: (
                                    f"http://metabolomics-usi.gnps2.org/dashinterface?usi1=mzspec%3AGNPS%3AGNPS-LIBRARY%3Aaccession%3A{x}&width=10.0&height=6.0&mz_min=None&mz_max=None&max_intensity=125&annotate_precision=4&annotation_rotation=90&cosine=standard&fragment_mz_tolerance=0.02&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D" if x.startswith("CCMSLIB")
                                    else f"http://metabolomics-usi.gnps2.org/dashinterface?usi1=mzspec%3AMASSBANK%3A%3Aaccession%3A{x}&width=10.0&height=6.0&mz_min=None&mz_max=None&max_intensity=125&annotate_precision=4&annotation_rotation=90&cosine=standard&fragment_mz_tolerance=0.02&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D"
                                )
                            )  

                            # Make the spectrum column the first column
                            col_order = ['spectrum_link', 'Compound_Name', 'Precursor_MZ']
                            df0 = df0[col_order + [col for col in df0.columns if col not in col_order]]

                            # display it with clickable links 
                            table_evt = st.dataframe(
                                df0,
                                column_config={
                                    "spectrum_link": st.column_config.LinkColumn(
                                        label="USI Viewer",
                                        display_text="Open Spectrum"
                                    )
                                },
                                hide_index=True,
                                on_select="rerun",
                                selection_mode="multi-row",
                                width='stretch',
                                key=f"{name}_{ik}_table"
                            )

                            # grab list of selected row indices
                            selected = table_evt.selection.rows

                            # show buttons with actions for selected spectra
                            col1, col2, col3, _ = st.columns([2, 2, 2, 4])
                            with col1:
                                if st.button("Remove selected spectra", key=f"{name}_{ik}_remove"):
                                    if selected:
                                        df_filtered = df0.drop(df0.index[selected])
                                        st.session_state.grouped_results[name][ik]["structure"] = df_filtered
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()

                            with col2:
                                if st.button("Keep only selected spectra", key=f"{name}_{ik}_keep"):
                                    if selected:
                                        df_filtered = df0.iloc[selected].reset_index(drop=True)
                                        st.session_state.grouped_results[name][ik]["structure"] = df_filtered
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()

                            # update the session state with the filtered dataframe
                            st.session_state.grouped_results[name][ik]["structure"] = df0
    
    # selection menu for raw data search
    col_a, col_b = st.columns(2)
    with col_a:
        option = st.radio(
            "Mode",
            ["FASSTrecords", "FASST"],
            horizontal=True,
            key="mode"
        )
    with col_b:
        st.empty()

    last_iteration = "08/2025"  # need to get version into sql table

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="
            border-left: 4px solid #2c7be5;
            padding: 1em;
            margin: 0.5em 0;
            background-color: #f0f8ff;
            border-radius: 4px;
        ">
        <h4 style="margin:0 0 0.5em;">
            <strong>FASSTrecords</strong>
        </h4>
        <p style="margin:0; line-height:1.5; font-size:0.95em;">
            This is <strong>very fast</strong> as it relies on precomputed annotations, and therefore especially <br/>
            recommended for substructure‑enabled search and searches of large<br/>
            numbers of spectra or molecules expected in large numbers of samples.<br/>
            Last iteration: <strong>{last_iteration}</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            border-left: 4px solid #e76f51;
            padding: 1em;
            margin: 0.5em 0;
            background-color: #fff5f0;
            border-radius: 4px;
        ">
        <h4 style="margin:0 0 0.5em;">
            <strong>FASST</strong>
        </h4>
        <p style="margin:0; line-height:1.5; font-size:0.95em;">
            Can be rather slow especially for molecules expected to be present in many datasets<br/>
            or large numbers of spectra. Searches can even take a few minutes, depending on traffic.<br/>
            Allows the discovery of unknown chemical analogues through modification search.<br/>
            Always up to date with the latest raw data indexed at <a href="https://fasst.gnps2.org/" target="_blank" style="color:#e76f51;">fasst.gnps2.org</a>.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cosine and Matching Peaks input 
    col3, col4, _, _ = st.columns(4)
    with col3:
        min_cosine = st.text_input("Minimum Cosine", value="0.7")
    with col4:
        min_peaks = st.text_input("Minimum Matching Peaks", value="5")

    # Conditional input for FASST 
    if option == "FASST":
        col5, col6, _, _ = st.columns(4)
        with col6:
            prec_tol = st.text_input("Precursor Tolerance (Da)", value="0.02")
            do_modification_search = st.checkbox("Modification search", value=False)


        with col5:
            frag_tol = st.text_input("Fragment Tolerance (Da)", value="0.02")
            if do_modification_search:
                col_elim, col_add = st.columns(2)
                with col_elim:
                    do_elimination = st.checkbox("Elimination search", value=True)
                with col_add:
                    do_addition = st.checkbox("Addition search", value=True)
                sub_col1, col_or, sub_col2 = st.columns([2,0.5,2])
                with sub_col1:
                    modification_formula = st.text_input("Modification formula", placeholder="O for hydroxylation")
                with col_or:
                    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)
                with sub_col2:
                    modification_mass = st.text_input("Modification mass (Da)", placeholder="15.9949 for O")
                do_subsetModificationSearch = st.checkbox("Only report modified molecules if <condition>", value=False)

                if do_subsetModificationSearch:
                    list_of_values = ["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"]
                    modification_condition = st.selectbox("Unmodified found in same", options=list_of_values)

    ctrl1, ctrl2, _ = st.columns([1,1,7])
    with ctrl1:
        do_search = st.button("Search Raw Data")


    # perform the raw data search
    if do_search:
        time.sleep(2)
        with st.spinner("Searching raw data…"): 

            new_results = {}

            if option == "FASSTrecords":
                # build each queries aggregated table
                for name, ik_dict in st.session_state.grouped_results.items():
                    sel_frames = []
                    for ik, data in ik_dict.items():
                        
                        df_struct = data["structure"]

                        # dereplicate spectra
                        print(f"the df_struct has {len(df_struct)} rows and columns: {df_struct.columns.tolist()}")

                        df_struct["spectrum_id_int"] = df_struct["spectrum_id_int"].astype("int64")
                        df_struct["representative_spectrum_int"] = df_struct["representative_spectrum_int"].astype("int64")
                        df_struct["similar_library_spectra"] = (
                            df_struct.groupby("representative_spectrum_int")["spectrum_id_int"]
                                    .transform("size")
                                    .astype("int64")
                        )

                        # add column with difference between spectrum_id_int and representative_spectrum_int
                        df_struct["spectrum_difference"] = df_struct["spectrum_id_int"] - df_struct["representative_spectrum_int"]

                        #sort so that smallest difference is kept by falcon_cluster (closest to representative spectrum)
                        df_struct = df_struct.sort_values(by=["representative_spectrum_int", "spectrum_difference"])

                        #keep first per falcon cluster
                        df_struct = df_struct.groupby("representative_spectrum_int").first().reset_index()

                        # set spectrum_id_int value to representative_spectrum_int value
                        df_struct["spectrum_id_int"] = df_struct["representative_spectrum_int"]

                        if not df_struct.empty:
                            sel_frames.append(df_struct)

                    if not sel_frames:
                        st.warning(f"No entries left for **{name}**, skipping.")
                        continue

                    df_for_name = pd.concat(sel_frames, ignore_index=True)

                    # get raw data from masstrecords
                    masst_df, redu_df = get_masst_and_redu_tables(df_for_name,
                                                                cosine_threshold=float(min_cosine),
                                                                matching_peaks=int(min_peaks),
                                                                sqlite_path=config.PATH_TO_SQLITE,
                                                                api_endpoint=config.MASSTRECORDS_ENDPOINT,
                                                                timeout=config.MASSTRECORDS_TIMEOUT,
                                                                chunk_size=200)

                    # if cosine not in masst_df.columns return empty dataframes
                    if "cosine" not in masst_df.columns or "matching_peaks" not in masst_df.columns:
                        new_results[name] = {"masst": pd.DataFrame(), "redu": pd.DataFrame()}
                        continue

                    # subset results for sample matches table to best match by sample
                    df_masst_sorted = masst_df.sort_values(by=["cosine", "matching_peaks"], ascending=[False, False])
                    df_masst_unique = df_masst_sorted.drop_duplicates(subset="mri_id_int", keep="first")
                    
                    if 'mri_id_int' in redu_df.columns:
                        # add query spectrum ID and scan ID to redu_df //could potentially move this into get_masst_and_redu_tables
                        redu_df = redu_df.merge(
                            df_masst_unique[["mri_id_int", "scan_id", "query_spectrum_id", 'matching_peaks', 'cosine', 'Adduct', 'Compound_Name', 'Precursor_MZ', 'inchikey_first_block', 'similar_library_spectra', 'unique_spectra_in_mri']],
                            on="mri_id_int",
                            how="left"
                            )

                        redu_df['similar_library_spectra'] = redu_df['similar_library_spectra'] + redu_df['unique_spectra_in_mri'] - 2

                        # make integer
                        redu_df['similar_library_spectra'] = redu_df['similar_library_spectra'].astype('Int64')

                        # make character values from 0 to "9+"
                        s = redu_df["similar_library_spectra"].astype("Int64")       
                        b = s.clip(upper=9)                                          
                        redu_df["similar_library_spectra"] = b.astype("string").where(b < 9, "9+")


                        # rename cosine to Cosine and matching_peaks to Matching Peaks
                        redu_df = redu_df.rename(columns={
                            "cosine": "Cosine",
                            "matching_peaks": "Matching Peaks",
                            "USI": "mri"
                        })

                        redu_df['Delta Mass'] = 0

                        # make library usis for the links
                        redu_df["lib_usi"] = redu_df["query_spectrum_id"].apply(
                            lambda x: (
                                f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if x.startswith("CCMSLIB")
                                else f"mzspec:MASSBANK::accession:{x}" 
                            )
                        )                    
                        # in every row add USI + :scan: + scan_id (as str)
                        redu_df["scan_id"] = pd.to_numeric(redu_df["scan_id"], errors="raise").astype(int)
                        redu_df["USI"] = redu_df["mri"] + ":scan:" + redu_df["scan_id"].astype(str)
                        redu_df["best_spectral_match"] = redu_df.apply(
                            lambda row: build_spectraresolver_link(row["USI"], row["lib_usi"]),
                            axis=1
                            )

                        if "Check LC peak" not in redu_df.columns:
                            redu_df["Check LC peak"] = np.nan

                        # Make sure the destination column can hold strings
                        redu_df["Check LC peak"] = redu_df["Check LC peak"].astype(object)  

                        mask = redu_df["Check LC peak"].isna() | (redu_df["Check LC peak"].astype(str).str.strip() == "")
                        redu_df.loc[mask, "Check LC peak"] = redu_df.loc[mask].apply(
                            lambda row: build_dashboard_eic_url(
                                usi=row['USI'],
                                xic_mz=row['Precursor_MZ'],
                                xic_tolerance=0.05
                            ),
                            axis=1
                        )

                    new_results[name] = {"masst": masst_df, "redu": redu_df}
                
            elif option == "FASST":

                # build each queries aggregated table
                for name, ik_dict in st.session_state.grouped_results.items():
                    sel_frames = []
                    for ik, data in ik_dict.items():
                        df_struct = data["structure"]

                        # dereplicate spectra
                        df_struct["spectrum_id_int"] = df_struct["spectrum_id_int"].astype("int64")
                        df_struct["representative_spectrum_int"] = df_struct["representative_spectrum_int"].astype("int64")
                        df_struct["similar_library_spectra"] = (
                            df_struct.groupby("representative_spectrum_int")["spectrum_id_int"]
                                    .transform("size")
                                    .astype("int64")
                        )
                        # add column with difference between spectrum_id_int and representative_spectrum_int
                        df_struct["spectrum_difference"] = df_struct["spectrum_id_int"] - df_struct["representative_spectrum_int"]

                        #sort so that smallest difference is kept by falcon_cluster (closest to representative spectrum)
                        df_struct = df_struct.sort_values(by=["representative_spectrum_int", "spectrum_difference"])

                        #keep first per falcon cluster
                        df_struct = df_struct.groupby("representative_spectrum_int").first().reset_index()

                        # set spectrum_id_int value to representative_spectrum_int value
                        df_struct["spectrum_id_int"] = df_struct["representative_spectrum_int"]

                        if not df_struct.empty:
                            sel_frames.append(df_struct)

                    if not sel_frames:
                        st.warning(f"No entries left for **{name}**, skipping.")
                        continue

                    df_for_name = pd.concat(sel_frames, ignore_index=True)

                    # if we do modification and got a molecular formula calculate the monoisotopic mass of the expected mass difference
                    formulaModi_object = Formula.formula_from_str(modification_formula) if do_modification_search and modification_formula else None
                    try:
                        modification_mass = formulaModi_object.get_monoisotopic_mass()
                    except AttributeError:
                        modification_mass = modification_mass if 'modification_mass' in locals() else None

                    # retrieve raw data matches through MASST
                    masst_df, redu_df = retrieve_raw_data_matches(
                        df_for_name,
                        database='metabolomicspanrepo_index_nightly',
                        precursor_mz_tol=float(prec_tol),
                        fragment_mz_tol=float(frag_tol),
                        min_cos=float(min_cosine),
                        matching_peaks=int(min_peaks),
                        analog=do_modification_search,
                        modimass=modification_mass,
                        elimination=do_elimination if 'do_elimination' in locals() else False,
                        addition=do_addition if 'do_addition' in locals() else False,
                        modification_condition=modification_condition if 'modification_condition' in locals() else None,
                        sqlite_path=config.PATH_TO_SQLITE,
                        api_endpoint=config.MASSTRECORDS_ENDPOINT,
                        timeout=config.MASSTRECORDS_TIMEOUT
                    )

                    if len(redu_df) > 0:
                        # make library usis for the links
                        redu_df["lib_usi"] = redu_df["query_spectrum_id"].apply(
                            lambda x: (
                                f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if x.startswith("CCMSLIB")
                                else f"mzspec:MASSBANK::accession:{x}" 
                            )
                        )

                        redu_df["best_spectral_match"] = redu_df.apply(
                            lambda row: build_spectraresolver_link(row["USI"], row["lib_usi"]), 
                            axis=1
                            )


                        if "Check LC peak" not in redu_df.columns:
                            redu_df["Check LC peak"] = np.nan

                        # Make sure the destination column can hold strings
                        redu_df["Check LC peak"] = redu_df["Check LC peak"].astype(object)  

                        mask = redu_df["Check LC peak"].isna() | (redu_df["Check LC peak"].astype(str).str.strip() == "")
                        redu_df.loc[mask, "Check LC peak"] = redu_df.loc[mask].apply(
                            lambda row: build_dashboard_eic_url(
                                usi=row['USI'],
                                xic_mz=row['Precursor_MZ'],
                                xic_tolerance=0.05
                            ),
                            axis=1
                        )
         
                    new_results[name] = {"masst": masst_df, "redu": redu_df}

            # store the results in session state
            st.session_state.raw_results = new_results

    # display results in tabs
    if st.session_state.get("raw_results"):
        raw_results = st.session_state.get("raw_results", {})

        # print lens of what has been found
        for name, df_pair in raw_results.items():
            # count unique MRI IDs in the 'redu' dataframe

            if 'mri_id_int' in df_pair['redu'].columns:
                num_unique_mri = df_pair['redu']['mri_id_int'].dropna().nunique()
            elif 'mri' in df_pair['redu'].columns:
                num_unique_mri = df_pair['redu']['mri'].dropna().nunique()
            else:
                num_unique_mri = 0

        has_valid_results = any(
            not df_pair["masst"].empty or not df_pair["redu"].empty
            for df_pair in raw_results.values()
        )

        if has_valid_results:
            with st.expander("Metabolomics Raw Data Matches", expanded=True):
                st.markdown("### Metabolomics Raw Data Matches")

                # create tabs for each query structure
                result_tabs = st.tabs(list(st.session_state.raw_results.keys()))
                for result_fig_id, (name, tab) in enumerate(zip(st.session_state.raw_results.keys(), result_tabs)):
                    with tab:

                        # get the results for this query structure
                        st.markdown(f"Retrieved **{len(st.session_state.raw_results[name]['redu'])}** matching samples with ReDU metadata for **{name}**.")


                        df_redu = st.session_state.raw_results[name]["redu"]

                        if ('mri_id_int' in df_redu.columns or 'mri' in df_redu.columns) and len(df_redu) > 0:

                            
                            opts_key = f"{name}_sankey_options"
                            new_cols = df_redu.columns.tolist()
                            if opts_key not in st.session_state:
                                st.session_state[opts_key] = new_cols[:]
                            else:
                                
                                st.session_state[opts_key] = list(dict.fromkeys(st.session_state[opts_key] + new_cols))

                            column_options = st.session_state[opts_key]
                            # Make sankey diagram
                            ##########

                            def _def_val(v: str) -> str:
                                return v if v in column_options else column_options[0]

                            def _apply_vals(vals: list[str]):
                                for i, v in enumerate(vals, start=1):
                                    st.session_state[f"{name}_col{i}"] = _def_val(v)
                                st.rerun()

                            def sticky_selectbox(label: str, options: list[str], key: str):
                                """Render a selectbox that keeps the user's selection across reruns."""
                                cur = st.session_state.get(key, None)
                                if cur in options:
                                    # Use existing session value; DON'T pass index (prevents Streamlit reset)
                                    return st.selectbox(label, options, key=key)
                                else:
                                    # First time / invalid value -> choose the first option
                                    return st.selectbox(label, options, index=0, key=key)


                            PRESET_MAP = {
                                            "datasets_bodypart_division_taxa": [
                                                "ATTRIBUTE_DatasetAccession", "UBERONBodyPartName", "NCBIDivision", "NCBITaxonomy"
                                            ],
                                            "query_similar_division_taxa": [
                                                "similar_library_spectra", "query_spectrum_id", "NCBIDivision", "NCBITaxonomy"
                                            ],
                                            "compound_bodypart_division_taxa": [
                                                "inchikey_first_block", "Compound_Name", "UBERONBodyPartName", "NCBITaxonomy"
                                            ],
                                            "delta_modified_bodypart_doid": [
                                                "Delta Mass", "Modified", "UBERONBodyPartName", "DOIDCommonName"
                                            ],
                                        }

                            if "default_sankey_vals" not in st.session_state:
                                st.session_state["default_sankey_vals"] = PRESET_MAP["datasets_bodypart_division_taxa"]



                            with st.container():
                                st.markdown(
                                    """
                                    <style>
                                    /* Shared card look */
                                    .tipbox, .presetbox {
                                        font-size: 0.95rem;
                                        line-height: 1.5;
                                        background: #f6f8fa;
                                        border: 1px solid #e5e7eb;
                                        padding: 0.9rem 1rem;
                                        border-radius: 12px;
                                        margin-bottom: 0.9rem;
                                    }
                                    .tipbox h4, .presetbox h4 {
                                        margin: 0 0 0.6rem 0;
                                        font-size: 1rem;
                                        font-weight: 700;
                                    }
                                    .tipbox ul {
                                        margin: 0.25rem 0 0 1.25rem;
                                        padding: 0;
                                        list-style-type: disc;
                                    }
                                    .tipbox li { margin: 0.35rem 0; }

                                    /* Rows inside preset panel */
                                    .preset-row {
                                        display: flex;
                                        align-items: flex-start;
                                        gap: 0.75rem;
                                        padding: 0.6rem 0.5rem;
                                        border-radius: 10px;
                                    }
                                    .preset-row:not(:last-child) {
                                        border-bottom: 1px dashed #e5e7eb;
                                    }
                                    .preset-text {
                                        flex: 1;
                                    }
                                    .preset-title {
                                        font-weight: 700;
                                    }
                                    .preset-sub {
                                        display: block;
                                        margin-top: 0.15rem;
                                        opacity: 0.9;
                                    }

                                    /* Make the Streamlit button align nicely & fill the small right column */
                                    .preset-button .stButton>button {
                                        width: 100%;
                                        height: 32px;
                                        border-radius: 10px;
                                        border: 1px solid #d1d5db;
                                        background: white;
                                    }
                                    .preset-button .stButton>button:hover {
                                        background: #f3f4f6;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                # Preset panel (everything in one grey card; each preset is a clean row)
                                st.markdown("##### Suggested views")


                                # We render the rows right after the header "card" to look continuous.
                                preset_panel = st.container()
                                with preset_panel:
                                    # Row 1
                                    r1c1, r1c2 = st.columns([0.86, 0.14])
                                    with r1c1:
                                        st.markdown(
                                            """
                                            <div class="preset-row">
                                            <div class="preset-text">
                                                <span class="preset-title">ATTRIBUTE_DatasetAccession → UBERONBodyPartName → NCBIDivision → NCBITaxonomy</span>
                                                <span class="preset-sub">Understand biological distributions at a glance while not losing sight of the number of datasets confirming an observed pattern.</span>
                                            </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with r1c2:
                                        st.markdown('<div class="preset-button">', unsafe_allow_html=True)
                                        if st.button("Apply", key=f"{name}_preset_1"):
                                            _apply_vals(PRESET_MAP["datasets_bodypart_division_taxa"])
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    # Divider look is handled by CSS (dashed bottom border on each row block).
                                    r2c1, r2c2 = st.columns([0.86, 0.14])
                                    with r2c1:
                                        st.markdown(
                                            """
                                            <div class="preset-row">
                                            <div class="preset-text">
                                                <span class="preset-title">similar_library_spectra → query_spectrum_id → NCBIDivision → NCBITaxonomy</span>
                                                <span class="preset-sub">Explore if matches to suspicious sample types are driven by different <em>query_spectrum_ids</em> than those for expected sample types.</span>
                                            </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with r2c2:
                                        st.markdown('<div class="preset-button">', unsafe_allow_html=True)
                                        if st.button("Apply", key=f"{name}_preset_2"):
                                            _apply_vals(PRESET_MAP["query_similar_division_taxa"])
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    # Row 3
                                    r3c1, r3c2 = st.columns([0.86, 0.14])
                                    with r3c1:
                                        st.markdown(
                                            """
                                            <div class="preset-row">
                                            <div class="preset-text">
                                                <span class="preset-title">inchikey_first_block → Compound_Name → UBERONBodyPartName → NCBITaxonomy</span>
                                                <span class="preset-sub">After substructure or Tanimoto similarity searches, investigate if molecules differ by sample types.</span>
                                            </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with r3c2:
                                        st.markdown('<div class="preset-button">', unsafe_allow_html=True)
                                        if st.button("Apply", key=f"{name}_preset_3"):
                                            _apply_vals(PRESET_MAP["compound_bodypart_division_taxa"])
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    # Row 4
                                    r4c1, r4c2 = st.columns([0.86, 0.14])
                                    with r4c1:
                                        st.markdown(
                                            """
                                            <div class="preset-row">
                                            <div class="preset-text">
                                                <span class="preset-title">Delta Mass → Modified → UBERONBodyPartName → DOIDCommonName</span>
                                                <span class="preset-sub">For modification searches, explore if specific modifications differ by body part and disease (especially interesting for drugs).</span>
                                            </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with r4c2:
                                        st.markdown('<div class="preset-button">', unsafe_allow_html=True)
                                        if st.button("Apply", key=f"{name}_preset_4"):
                                            _apply_vals(PRESET_MAP["delta_modified_bodypart_doid"])
                                        st.markdown('</div>', unsafe_allow_html=True)

                            def_val = lambda v: v if v in column_options else column_options[0]

                            # initialize session_state defaults 
                            for i, default in enumerate(st.session_state["default_sankey_vals"], start=1):
                                key = f"{name}_col{i}"
                                if key not in st.session_state:
                                    st.session_state[key] = def_val(default)

                            # make four sticky selectboxes (won't reset after buttons/reruns)
                            col1_c, col2_c, col3_c, col4_c = st.columns(4)
                            with col1_c:
                                sticky_selectbox("Column 1", column_options, key=f"{name}_col1")
                            with col2_c:
                                sticky_selectbox("Column 2", column_options, key=f"{name}_col2")
                            with col3_c:
                                sticky_selectbox("Column 3", column_options, key=f"{name}_col3")
                            with col4_c:
                                sticky_selectbox("Column 4", column_options, key=f"{name}_col4")


                            # pull the latest values and build your Sankey immediately
                            col1 = st.session_state[f"{name}_col1"]
                            col2 = st.session_state[f"{name}_col2"]
                            col3 = st.session_state[f"{name}_col3"]
                            col4 = st.session_state[f"{name}_col4"]

                            # if any two cols have the same value give warning
                            if len({col1, col2, col3, col4}) == 4:
                                fig = raw_data_sankey(df_redu, col1, col2, col3, col4)

                                config_sankey_download = {
                                    "toImageButtonOptions": {"format": "svg", "filename": f"plot_{fig_id}"},
                                    "displaylogo": False,
                                }

                                st.plotly_chart(fig, width='stretch', key=f"sankey_{result_fig_id}", config=config_sankey_download)
                                try:
                                    fig.write_image(
                                        f"{output_folder}/rawData_sankey_{name}.pdf", 
                                        format="pdf", 
                                        width=1240, 
                                        height=400, 
                                        scale=2
                                    )
                                except RuntimeError as e:
                                    if "Kaleido requires Google Chrome to be installed" in str(e):
                                        print("⚠️ Skipping PDF export: Chrome not available for Kaleido.")
                                    else:
                                        raise
                            else:
                                st.warning("Warning: Some selected columns have the same value.")


                            # sample matches tab
                            #########
                            # Reorder columns: best_spectral_match first, then modification_site if it exists, then the rest
                            cols = ["best_spectral_match"]
                            if "modification_site" in df_redu.columns:
                                cols.append("modification_site")
                            if "Check LC peak" in df_redu.columns:
                                cols.append("Check LC peak")
                            
                            cols += [col for col in df_redu.columns if col not in cols]
                            df_redu = df_redu[cols]

                            # If modification column exists sort so that modified matches come first
                            if 'Modified' in df_redu.columns:
                                df_redu['Modified'] = df_redu['Modified'].astype(str).str.lower()
                                df_redu.loc[~df_redu['Modified'].isin(['addition', 'elimination', 'no']), 'Modified'] = pd.NA
                                df_redu['Modified'] = pd.Categorical(
                                    df_redu['Modified'],
                                    categories=['addition', 'elimination', 'no'],
                                    ordered=True
                                )
                                df_redu = df_redu.sort_values(by='Modified', ascending=True)


                            column_config = {
                                        "best_spectral_match": st.column_config.LinkColumn(
                                            label="best_spectral_match",
                                            display_text="View MS/MS match"
                                        )
                                    }
                        
                            if 'modification_site' in df_redu.columns:
                                column_config["modification_site"] = st.column_config.LinkColumn(
                                    label="Modification Site",
                                    display_text="View Modification Site"
                                )

                            if 'Check LC peak' in df_redu.columns:
                                column_config["Check LC peak"] = st.column_config.LinkColumn(
                                    label="Check LC peak",
                                    display_text="View LC peak"
                                )
                            # show dataframe
                            table_evt = st.dataframe(
                                df_redu,
                                column_config=column_config,
                                hide_index=True,
                                width='stretch',
                                on_select="rerun",
                                selection_mode="multi-row",
                                key=f"{name}_redu_table",
                            )
                            # grab the selected row positions
                            ui_selected = st.session_state.get(f"{name}_redu_table", {}).get("selection", {}).get("rows", [])

                            # grab the selected row positions from UI
                            ui_selected = st.session_state.get(f"{name}_redu_table", {}).get("selection", {}).get("rows", [])

                            # dropdowns
                            btn_col3, btn_col4, _ = st.columns([2,3,5])

                            with btn_col3:
                                column_options = ["None"] + list(df_redu.columns)
                                sel_col = st.selectbox("Column", column_options, index=0, key=f"{name}_redu_selcol")

                            dropdown_selected = []

                            sel_vals = []

                            with btn_col4:
                                if sel_col != "None":
                                    disp_col = df_redu[sel_col].astype("string").fillna("<NA>")
                                    value_options = sorted(disp_col.unique().tolist())
                                    sel_vals = st.multiselect(
                                        "Value(s)",
                                        value_options,
                                        key=f"{name}_redu_selvals",
                                    )
                                    disp = df_redu[sel_col].astype("string").fillna("<NA>")
                                    dropdown_selected = [i for i, ok in enumerate(disp.isin(sel_vals).to_numpy()) if ok]
                                else:
                                    st.multiselect("Value(s)", [], key=f"{name}_redu_selvals_disabled", disabled=True)

                            # --- FINAL SELECTION ---
                            use_dropdown = (sel_col != "None" and bool(sel_vals))  
                            selected = dropdown_selected if use_dropdown else ui_selected

                            btn_col1, btn_col2, _ = st.columns([2,2, 6])

                            # buttons for selected rows  
                            with btn_col1:
                                if st.button("Remove selected rows", key=f"{name}_redu_remove"):
                                    if selected:
                                        st.session_state.raw_results[name]["redu"] = (
                                            df_redu.drop(df_redu.index[selected]).reset_index(drop=True)
                                        )
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()

                            with btn_col2:
                                if st.button("Keep only selected rows", key=f"{name}_redu_keep"):
                                    if selected:
                                        st.session_state.raw_results[name]["redu"] = (
                                            df_redu.iloc[selected].reset_index(drop=True)
                                        )
                                    else:
                                        st.warning("No rows selected!")
                                    st.rerun()


                            # Make topic MASSTs per query


                            st.markdown(f"##### Downstream tooling")


                            masst_by_query_button, _ = st.columns([4, 6])

                            with masst_by_query_button:
                                if st.button(f"Populate DomainMASST for {name}", key=f"{name}_topic_masst"):
                                    sid = st.session_state["_session_hash"]

                                    # Prepare input
                                    topic_masst_df = st.session_state.raw_results[name]["redu"].copy()
                                    topic_masst_df = topic_masst_df[["USI", "Cosine", "Matching Peaks", "Delta Mass"]]
                                    topic_masst_df["Status"] = 1

                                    input_path = f"{output_folder}/topicMasst_input_{name}.tsv"
                                    topic_masst_df.to_csv(input_path, sep="\t", index=False, header=True)

                                    # Run and report
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
                        else:
                            st.warning("No ReDU metadata matches found.")

            redu_tables = {
                qname: pair["redu"]
                for qname, pair in st.session_state.raw_results.items()
                if isinstance(pair.get("redu"), pd.DataFrame)
            }

            
            if len(redu_tables) > 1 and all(len(df) > 1 for df in redu_tables.values()):

                # helper: build a unified MRI key
                def _mri_key(df: pd.DataFrame) -> pd.Series:
                    if "mri" in df.columns:
                        return df["mri"].astype(str)
                    elif "mri_id_int" in df.columns:
                        return df["mri_id_int"].astype(str)
                    else:
                        # no usable key → empty
                        return pd.Series([], dtype=str)

                masst_by_query_button, message_space_masst_by_query = st.columns([4,4]) 

                with masst_by_query_button:
                    if st.button(f"Populate DomainMASST with Molecule Co-occurrence", key=f"intersection_topic_masst"):

                        # intersection of MRIs across all ReDU tables
                        mri_sets = [set(_mri_key(df).dropna().unique()) for df in redu_tables.values()]
                        common_mris = set.intersection(*mri_sets) if mri_sets else set()

                        if common_mris:
                            # keep only rows whose MRI is present in ALL tables
                            filtered_frames = []
                            for df in redu_tables.values():
                                dfx = df.copy()

                                # unify cosine column name
                                if "Cosine" not in dfx.columns and "cosine" in dfx.columns:
                                    dfx = dfx.rename(columns={"cosine": "Cosine"})

                                # create standardized key column
                                if "mri" in dfx.columns:
                                    dfx["mri_key"] = dfx["mri"].astype(str)
                                elif "mri_id_int" in dfx.columns:
                                    dfx["mri_key"] = dfx["mri_id_int"].astype(str)
                                else:
                                    continue

                                filtered_frames.append(dfx[dfx["mri_key"].isin(common_mris)])

                            if filtered_frames:
                                cooccurrence_df = pd.concat(filtered_frames, ignore_index=True)

                                # pick one row per MRI with highest Cosine
                                cos_series = pd.to_numeric(cooccurrence_df.get("Cosine", pd.Series([-1] * len(cooccurrence_df))), errors="coerce").fillna(-1)
                                cooccurrence_df["__cos_num"] = cos_series

                                idxmax = cooccurrence_df.groupby("mri_key")["__cos_num"].idxmax()
                                cooccurrence_df = cooccurrence_df.loc[idxmax].drop(columns="__cos_num").reset_index(drop=True)

                                # optional: sort by Cosine desc if present
                                if "Cosine" in cooccurrence_df.columns:
                                    topic_masst_df = cooccurrence_df.sort_values("Cosine", ascending=False, na_position="last")
                                    sid = st.session_state["_session_hash"]

                                    # save only USI column without header
                                    topic_masst_df = topic_masst_df[["USI", "Cosine", "Matching Peaks", "Delta Mass"]].copy()
                                    topic_masst_df['Status'] = 1
                                    
                                    # write the file
                                    topic_masst_df.to_csv(f"{output_folder}/domainMasst_input_moleculeIntersection.tsv", sep="\t", index=False, header=True)

                                    with st.spinner("Running DomainMASSTs…"): 
                                        returncode, stdout, stderr = run_topic_MASSTs(f"{output_folder}/domainMasst_input_moleculeIntersection.tsv", output_folder, "moleculeIntersection")

                                    if returncode == 0:
                                        st.session_state["last_topic_masst_name"] = name
                                        st.session_state["last_topic_masst_output_dir"] = output_folder

                                        st.success(f"DomainMASSTs for Molecule Intersection completed.")
                                        st.page_link(
                                            "pages/domainMASST.py", 
                                            label="➡️ Click for DomainMASST Results",
                                        )
                                    else:
                                        st.error("DomainMASSTs failed. See logs below.")
                                        with st.expander("Show logs"):
                                            st.code(stdout or "", language="text")
                                            st.code(stderr or "", language="text")
                    
        else:
            st.warning("No raw data matches found. Please try a different query structure or adjust your search parameters.")
    else:
        st.markdown("")
