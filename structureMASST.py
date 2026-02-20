import os
import streamlit as st

# Write the page label
st.set_page_config(
    page_title="StructureMASST", 
    layout="wide",
    page_icon="🔎",
)
st.logo("logo.png", icon_image="logo.png")

from streamlit.components.v1 import html
import pandas as pd
import importlib.util
from rdkit import Chem
from PIL import Image
import base64
import io
from bin.workflow_stepwise import retrieve_raw_data_matches 
from bin.run_masstRecords_queries import get_library_table, get_masst_and_redu_tables, _get_fetcher
from bin.match_smiles import detect_smiles_or_smarts, neutralize_atoms, tautomerize_smiles
from bin.pubchem_handling  import pubchem_autocomplete, name_to_cid, cid_to_canonical_smiles
from bin.plotting import raw_data_sankey, export_hits_map
from bin.linkouts import build_dashboard_eic_url, build_spectraresolver_link
from bin.smarts_api import query_smarts
from bin.streamlit_fragment_domainMASST import domainmasst_fragment, domainmasst_intersection_fragment
from bin.streamlit_fragment_LifeMASST import lifemasst_fragment
from bin.api_health import test_fasst_api_search_nonblocking
from bin.shared_data import get_molecule_classes_cached
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
import gc


import tasks



# Tracking
import umami
umami.set_url_base("https://analytics-api.gnps2.org/")
umami.set_website_id('032bfca4-a353-4586-b637-8908d8b71c85')
umami.set_hostname('analytics-api.gnps2.org')

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
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0) # GNPS2 Global
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="74665d88-3b9d-4812-b8fc-7f55ceb08f11"></script>',  width=0, height=0) # Streamlit Apps
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="032bfca4-a353-4586-b637-8908d8b71c85"></script>',  width=0, height=0) # Structure MASST



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
    Wilhan Nunes (UCSD)<br>
    Mingxun Wang (UCR)<br>
    </span>
    """,
    unsafe_allow_html=True
)

try:
    from rdkit.Chem import Draw
    from rdkit.Chem import rdMolDescriptors
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

# — preload molecule classes cache —
sqlite_path = config.PATH_TO_SQLITE
api_endpoint = config.MASSTRECORDS_ENDPOINT
timeout = config.MASSTRECORDS_TIMEOUT

# 🔹 Create the fetcher for SQLite or Datasette
fetch = _get_fetcher(sqlite_path, api_endpoint, timeout)
_molecule_classes_cache = get_molecule_classes_cached(fetch)

# — SMILES or CSV input —
col_name, col_or1, col_smiles, col_or2, col_class, col_or3, col_csv = st.columns([4, 1, 4, 1, 4, 1, 4])

# ---------- state init ----------
for k, v in [
    ("name_query", ""),
    ("last_fetched_query", None),
    ("name_suggestions", []),
    ("name_choice", None),
    ("usi_input", ""), 
    ("smiles_input", ""),
    ("name_warning", None),
    ("structure_editor_open", False),
    ("new_smiles", ""),
    ("class_label", ""),
]:
    st.session_state.setdefault(k, v)

# ---------- state init (add) ----------
st.session_state.setdefault("append_library_results", False)

def _uniquify_names(incoming: list[str], used: set[str]) -> list[str]:
    """Make top-level query names unique against existing tabs + within incoming batch."""
    out = []
    for base in incoming:
        base = str(base).strip() or "Input_query"
        cand = base
        i = 2
        while cand in used:
            cand = f"{base}_{i}"
            i += 1
        used.add(cand)
        out.append(cand)
    return out

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
        on_change=lambda: st.session_state.update({'structure_editor_open': False, 'new_smiles': '', 'smiles_input': '', 'class_label': ""}),
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
            _resolve_name_to_smiles(st.session_state.get("name_choice"))
            st.session_state["structure_editor_open"] = False
            # mirror whatever resolve put into the text box; fall back to empty
            st.session_state["new_smiles"] = st.session_state.get("smiles_input", "")


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

    #add spacer to align widgets
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    # if st.checkbox("Upload Batch file", key="batch_search"):
    with st.popover("Add batch file", icon=":material/file_upload:"):
        uploaded_file = st.file_uploader(
            """Drop CSV file for batch search (smiles and name columns). We recommend to stay below 100 molecules or reach out to us to run locally.""",
            type=["csv"]
        )

with col_or1:
    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

with col_smiles:
    smiles_input = st.text_input(
        "SMILES/SMARTS",
        key="smiles_input",  # gets auto-populated only if single-component
        placeholder="Enter SMILES or SMARTS",
        help="Enter a valid SMILES or SMARTS string. For SMARTS string creation, you can use third-party tools like SMARTSPlus https://smarts.plus/create",
        #cleans session state on change and cleans name input and suggestions
        on_change=lambda: st.session_state.update({'structure_editor_open': False, 'new_smiles': '', 'name_query': '', 'name_suggestions': [], 'class_label': ""}),
    )
    smiles_input = smiles_input.strip()
    # Use effective SMILES (from editor if available) for type detection
    effective_smiles = st.session_state.get('new_smiles', '') or smiles_input
    smiles_type = detect_smiles_or_smarts(effective_smiles)



with col_or2:
    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

with col_class:
    # take class_label column from _molecule_classes_cache
    if _molecule_classes_cache is not None:
        class_labels = _molecule_classes_cache["class_label"].tolist()
        class_labels.insert(0, "")
        st.selectbox("Class Label", options=class_labels, key="class_label",
                     help="Select a molecule class to find spectra for all molecules in this class.",
                     on_change=lambda: st.session_state.update({'structure_editor_open': False, 'new_smiles': '', 'smiles_input': ''}))
        
        if st.session_state["class_label"] != "":
            smiles_input = None  # clear smiles input if class label is selected
            smiles_type = "class_label"
            effective_smiles = None

with col_or3:
    st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

with col_csv:

    usi_input = st.text_input(
        "USI (mzspec:...)",
        key="usi_input",
        placeholder="mzspec:...",
        help="Provide a USI to directly inspect this spectrum. No structure is required.",
        on_change=lambda: st.session_state.update({
            "smiles_input": "",
            "new_smiles": "",
            "structure_editor_open": False,
            "class_label": ""
        }),
    ).strip()


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

if usi_input:
    smiles_type = "usi"
elif smiles_input:
    smiles_input = smiles_input.strip()
    # Use effective SMILES (from editor if available) for type detection
    effective_smiles = st.session_state.get('new_smiles', '') or smiles_input
    smiles_type = detect_smiles_or_smarts(effective_smiles)

    df_input = pd.DataFrame(
        {
            "smiles": [effective_smiles],
            "name": ["Input_query"],
            "type": [smiles_type]
        }
    )
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
            st.warning("Failed to retrieve SMARTS image from the API. You can still go ahead.")
    else:
        st.warning("Not a valid SMILES/SMARTS")
elif st.session_state.get("class_label", "") != "":
    smiles_type = "class_label"
    st.info(f"Class Label selected: {st.session_state['class_label']}. Proceed to find spectra for molecules in this class.")



# — mode selection UI —
search_options = ["Exact structure match", "Substructure match", "Tanimoto similarity"]
default_search_index = 0 
col_a1, col_allowedformula, col_tani, col_allowedelements, _ = st.columns([2,1,1,1,2])
if smiles_type and smiles_type == 'smarts':
    #st.info("SMARTS input detected. Use 'Substructure match' for search.", icon=':material/info:')
    search_options = ["Substructure match"]
if st.session_state.get("class_label", "") != "":
    search_options = ["Exact structure match"]
with col_a1:
    searchtype_option = st.radio(
        "Find available MS/MS spectra", 
        search_options, 
        horizontal=True, index=default_search_index
    )

if searchtype_option != "Exact structure match":
    with col_allowedformula:
        allowed_formula = st.text_input(
            "Fixed molecular formula (optional)", 
            value="", 
            key="allowed_formula"
        )
    with col_allowedelements:
        allowed_elements = st.text_input(
            "Allowed element difference (optional)", 
            value="", 
            key="allowed_elements"
        )

if searchtype_option == "Tanimoto similarity":
    with col_tani:
        tanimoto_cutoff = st.text_input("Tanimoto threshold", value="0.8", key="tanimoto_threshold")

# Map UI option to backend value
if searchtype_option == "Exact structure match":
    searchtype_option = "exact"
elif searchtype_option == "Substructure match":
    searchtype_option = "substructure"
elif searchtype_option == "Tanimoto similarity":
    searchtype_option = "tanimoto"


# — run the search — (replace whole block) —

has_prev_library = bool(st.session_state.get("grouped_results"))

btn_col, chk_col, _ = st.columns([2, 2, 4])

with chk_col:
    # Only show the checkbox if there are already results
    if has_prev_library:
        append_mode = st.checkbox(
            "Add to existing",
            key="append_library_results",
            help="Molecule will be added to previous ones as additional query, rather than replacing previously requested molecules."
        )
    else:
        append_mode = False
        st.empty()

with btn_col:
    do_get_spectra = st.button("Get Available Spectra", icon=":material/search:")

if do_get_spectra:
    # Tracking this action
    try:
        umami.new_event(event_name="Get Available Spectra Clicked")
    except Exception as e:
        print(f"Error tracking event: {e}")

    # Use new_smiles from structure editor if available, otherwise use original input
    effective_smiles = st.session_state.get("new_smiles", "") or smiles_input
    if smiles_type == "smiles":
        effective_smiles = tautomerize_neutralize_smiles(effective_smiles)

    with st.spinner("Finding spectra..."):

        # Always clear downstream raw results (they're stale after changing library entries)
        st.session_state.pop("raw_results", None)

        if not append_mode:
            # Reset upstream state (original behavior)
            for key in [
                "selected_queries",
                "df_library_conflicts",
                "grouped_results",
                "raw_results",
                "molecule_overview",
                "query_table",
            ]:
                st.session_state.pop(key, None)

            st.session_state.selected_queries = {}
            st.session_state.df_library_conflicts = None
            st.session_state.grouped_results = {}
            st.session_state.molecule_overview = {}
            st.session_state.query_table = pd.DataFrame()

        else:
            # Append mode: keep existing, but ensure required containers exist
            st.session_state.setdefault("selected_queries", {})
            st.session_state.setdefault("grouped_results", {})
            st.session_state.setdefault("molecule_overview", {})
            if "query_table" not in st.session_state or not isinstance(st.session_state["query_table"], pd.DataFrame):
                st.session_state["query_table"] = pd.DataFrame()

        # -------------------------
        # CANONICAL INPUT TABLE (ALWAYS)
        # -------------------------
        df_input = None

        # Priority: USI > SMILES/SMARTS > Batch CSV > Class label
        if usi_input:
            df_input = pd.DataFrame([{
                "query": usi_input.strip(),
                "name": "USI_Query",
                "type": "usi",
                "searchtype": "usi",
                "formula": "any",
                "allowed_elements": "any",
            }])

        elif smiles_input:
            effective = (st.session_state.get("new_smiles", "") or smiles_input).strip()
            t = detect_smiles_or_smarts(effective)

            if t == "smiles":
                effective = tautomerize_neutralize_smiles(effective)

            df_input = pd.DataFrame([{
                "query": effective,
                "name": "Input_query",
                "type": t,
                "searchtype": ("substructure" if t == "smarts" else searchtype_option),
                "formula": (allowed_formula if searchtype_option == "substructure" else "any"),
                "allowed_elements": (allowed_elements if searchtype_option == "substructure" else "any"),
            }])

        elif uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)

            if "query" not in df_input.columns:
                if "smiles" in df_input.columns:
                    df_input = df_input.rename(columns={"smiles": "query"})
                else:
                    st.warning("CSV must contain either ('query','name') or ('smiles','name').")
                    st.stop()

            if "name" not in df_input.columns:
                df_input["name"] = [f"Input_{i+1}" for i in range(len(df_input))]

            df_input = df_input.dropna(subset=["query", "name"]).copy()
            df_input["query"] = df_input["query"].astype(str).str.strip()
            df_input["name"] = df_input["name"].astype(str).str.strip()

            if "type" not in df_input.columns:
                def _infer_type(q):
                    q = str(q).strip()
                    if q.startswith("mzspec:"):
                        return "usi"
                    return detect_smiles_or_smarts(q)
                df_input["type"] = df_input["query"].apply(_infer_type)

            def _harmonize_query(row):
                if row["type"] == "smiles":
                    return tautomerize_neutralize_smiles(row["query"])
                return row["query"]

            df_input["query"] = df_input.apply(_harmonize_query, axis=1)

            if "searchtype" not in df_input.columns:
                def _infer_searchtype(t):
                    if t == "usi": return "usi"
                    if t == "smarts": return "substructure"
                    if t == "class_label": return "class_label"
                    return "exact"
                df_input["searchtype"] = df_input["type"].apply(_infer_searchtype)

            if "formula" not in df_input.columns:
                df_input["formula"] = "any"
            df_input["formula"] = df_input["formula"].fillna("any")

            if "allowed_elements" not in df_input.columns:
                df_input["allowed_elements"] = "any"
            df_input["allowed_elements"] = df_input["allowed_elements"].fillna("any")

        elif st.session_state.get("class_label", "") != "":
            class_label = st.session_state["class_label"].strip()
            df_input = pd.DataFrame([{
                "query": class_label,
                "name": class_label,
                "type": "class_label",
                "searchtype": "class_label",
                "formula": "any",
                "allowed_elements": "any",
            }])

        else:
            st.warning("Please enter a USI, SMILES/SMARTS, upload a CSV, or select a class label.")
            st.stop()

        # --- IMPORTANT: in append mode, avoid tab name collisions ---
        if append_mode:
            used_names = set(st.session_state.get("grouped_results", {}).keys())
        else:
            used_names = set()

        df_input = df_input.copy()
        df_input["name"] = _uniquify_names(df_input["name"].tolist(), used_names)

        # store / append canonical input table
        if append_mode and isinstance(st.session_state.get("query_table"), pd.DataFrame) and not st.session_state["query_table"].empty:
            st.session_state["query_table"] = (
                pd.concat([st.session_state["query_table"], df_input], ignore_index=True)
                  .drop_duplicates()
            )
        else:
            st.session_state["query_table"] = df_input

        # lists ALWAYS defined
        query_list = df_input["query"].tolist()
        name_list = df_input["name"].tolist()
        type_list = df_input["type"].tolist()
        searchtype_list = df_input["searchtype"].tolist()
        formula_list = df_input["formula"].tolist()
        allowed_elements_list = df_input["allowed_elements"].tolist()

        # Build ONLY the new results, then merge if append_mode
        new_grouped_results = defaultdict(dict)
        new_molecule_overview = defaultdict(list)

        for q, name, typ, searchtype, formula, allowed_elements in zip(
            query_list, name_list, type_list, searchtype_list, formula_list, allowed_elements_list
        ):
            if typ == "usi" or searchtype == "usi":
                ik = hashlib.sha1(q.encode()).hexdigest()[:12]  # fake IK bucket

                df_struct = pd.DataFrame([{
                    "inchikey_first_block": ik,
                    "Compound_Name": name,
                    "Smiles": "",
                    "Precursor_MZ": np.nan,
                    "query_spectrum_id": q,
                    "USI": q,
                }])

                new_grouped_results[name][ik] = {"structure": df_struct, "conflicts": pd.DataFrame()}

                prev = st.session_state["selected_queries"].get(ik, [])
                st.session_state["selected_queries"][ik] = list(dict.fromkeys(prev + [q]))

                new_molecule_overview[name].append({
                    "Compound_Name": name,
                    "inchikey_first_block": ik,
                    "Smiles": "",
                    "USI": q,
                })
                continue

            try:
                tanimoto_threshold = tanimoto_cutoff if searchtype == "tanimoto" else None

                df_library_structurematch = tasks.run_get_library_table(
                    q,
                    searchtype,
                    tanimoto_threshold,
                    formula,
                    allowed_elements,
                    config.PATH_TO_SQLITE,
                    config.MASSTRECORDS_ENDPOINT,
                    config.MASSTRECORDS_TIMEOUT,
                )

                if df_library_structurematch.empty:
                    continue

                df_library_conflicts = pd.DataFrame()
                df_library_conflicts["inchikey_first_block"] = pd.Series(dtype=str)

            except Exception as e:
                st.error(f"Error for {name}: {e}")
                continue

            for ik in df_library_structurematch["inchikey_first_block"].unique():
                sub_struct = df_library_structurematch[df_library_structurematch["inchikey_first_block"] == ik].copy()
                sub_conf = df_library_conflicts[df_library_conflicts["inchikey_first_block"] == ik].copy()

                new_grouped_results[name][ik] = {"structure": sub_struct, "conflicts": sub_conf}

                new_qs = list(sub_struct["query_spectrum_id"].unique())
                prev = st.session_state["selected_queries"].get(ik, [])
                st.session_state["selected_queries"][ik] = list(dict.fromkeys(prev + new_qs))

                names = sub_struct["Compound_Name"].dropna().astype(str) if "Compound_Name" in sub_struct.columns else pd.Series([], dtype=str)
                best_name = names.iloc[0] if len(names) else ""

                smiles = sub_struct["Smiles"].dropna().astype(str) if "Smiles" in sub_struct.columns else pd.Series([], dtype=str)
                first_smi = smiles.iloc[0] if len(smiles) else ""

                new_molecule_overview[name].append({
                    "Compound_Name": best_name,
                    "inchikey_first_block": ik,
                    "Smiles": first_smi
                })

        # finalize: if append_mode and nothing new, keep old results visible
        if not new_grouped_results:
            st.warning("No spectra available for this input. Existing results were kept." if append_mode else "No spectra available for this input.")
        else:
            if append_mode:
                merged = dict(st.session_state.grouped_results)
                merged.update(new_grouped_results)
                st.session_state.grouped_results = merged

                mo = dict(st.session_state.molecule_overview)
                for k, v in new_molecule_overview.items():
                    mo[k] = pd.DataFrame(v)
                st.session_state.molecule_overview = mo
            else:
                st.session_state.grouped_results = new_grouped_results
                st.session_state.molecule_overview = {k: pd.DataFrame(v) for k, v in new_molecule_overview.items()}

        for var in ["df_library_structurematch", "result"]:
            try:
                del locals()[var]
            except KeyError:
                pass
        gc.collect()


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
                    f"##### Retrieved **{total_matches}** spectra for molecule ({name}).",
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

                    # make a copy to avoid SettingWithCopyWarning
                    df_sankey = df_all[stages].copy()

                    # Define default fill values with suffixes
                    fill_values = {
                        "msMassAnalyzer": "Unknown_1",
                        "Ion_Mode": "Unknown_2",
                        "Adduct": "Unknown_3",
                        "collision_energy": "Unknown_4",
                    }

                    # Convert collision_energy to string *before* filling to avoid dtype conflicts
                    if pd.api.types.is_numeric_dtype(df_sankey["collision_energy"]):
                        df_sankey["collision_energy"] = df_sankey["collision_energy"].astype(str)

                    # Fill NaNs with the custom Unknown labels
                    df_sankey = df_sankey.fillna(fill_values)

                    # Fill "nan" strings if any
                    for col, fill_val in fill_values.items():
                        df_sankey.loc[df_sankey[col] == "nan", col] = fill_val

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
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{fig_id}", config=config_sankey_download)

                molecule_overview_df = st.session_state.molecule_overview[name]

                if _RD_DRAW_AVAILABLE:
                    @st.cache_data
                    def smiles_to_formula(smi: str) -> str:
                        mol = Chem.MolFromSmiles(smi)
                        if not mol:
                            return ""
                        return rdMolDescriptors.CalcMolFormula(mol)
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
                    molecule_overview_df["Molecular_Formula"] = (molecule_overview_df["Smiles"].apply(smiles_to_formula))

                    # reorder columns to Compound_Name, Structure, others
                    cols = (
                                ["Compound_Name", "Molecular_Formula", "structure"]
                                + [col for col in molecule_overview_df.columns
                                if col not in ["Compound_Name", "Molecular_Formula", "structure"]]
                            )
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
                col1_mol_level, col2_mol_level, _, col_split_mols_into_queries = st.columns([2, 2, 4, 2])
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
                #if more than 1 molecule, allow splitting
                if num_molecules > 1:
                    with col_split_mols_into_queries:
                        if st.button("Split each molecule into separate query", key=f"{name}_mol_split"):
                            # If nothing selected, split ALL rows
                            df_to_split = (
                                molecule_overview_df.iloc[selected].reset_index(drop=True)
                                if selected
                                else molecule_overview_df.reset_index(drop=True)
                            )

                            if df_to_split.empty:
                                st.warning("No rows to split!")
                            else:
                                src_results = st.session_state.grouped_results.get(name, {})
                                for idx, row in df_to_split.iterrows():
                                    new_name = f"{row.get('Compound_Name')}_{idx+1}"
                                    st.session_state.molecule_overview[new_name] = pd.DataFrame([row])

                                    ik = row.get("inchikey_first_block")
                                    if isinstance(src_results, dict) and ik in src_results:
                                        st.session_state.grouped_results[new_name] = {ik: src_results[ik]}
                                    else:
                                        st.session_state.grouped_results[new_name] = {
                                            "structure": pd.DataFrame(),
                                            "conflicts": pd.DataFrame(),
                                        }

                                    new_query_row = {
                                        "smiles": row["Smiles"],
                                        "name": new_name,
                                        "type": "smiles",
                                        "searchtype": "exact"
                                    }

                                    if "query_table" not in st.session_state or st.session_state.query_table is None:
                                        st.session_state.query_table = pd.DataFrame([new_query_row])
                                    else:
                                        st.session_state.query_table = pd.concat(
                                            [st.session_state.query_table, pd.DataFrame([new_query_row])],
                                            ignore_index=True
                                        )

                                # remove the original group
                                st.session_state.molecule_overview.pop(name, None)
                                st.session_state.grouped_results.pop(name, None)

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


                            def make_dashinterface_link(x: str) -> str:
                                x = str(x)

                                # If already a USI, use it directly
                                if x.startswith("mzspec:"):
                                    usi = x
                                elif x.startswith("CCMSLIB"):
                                    usi = f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}"
                                else:
                                    usi = f"mzspec:MASSBANK::accession:{x}"

                                return (
                                    "http://metabolomics-usi.gnps2.org/dashinterface?"
                                    f"usi1={_url.quote(usi, safe='')}"
                                    "&width=10.0&height=6.0&mz_min=None&mz_max=None&max_intensity=125"
                                    "&annotate_precision=4&annotation_rotation=90&cosine=standard"
                                    "&fragment_mz_tolerance=0.02&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D"
                                )

                            df0["spectrum_link"] = df0["query_spectrum_id"].apply(make_dashinterface_link)


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
    @st.fragment
    def raw_data_search_panel():

        # Message that we can now search all of the above xx spectra across raw data to retrieve matching samples. Include exact number of spectra
        spectra_across_all_molecules = sum(
            len(ik_dict[ik]['structure'])
            for ik_dict in st.session_state.grouped_results.values()
            for ik in ik_dict
        )

        number_of_molecules = len(st.session_state.grouped_results)

        st.markdown(f"##### Next, search all {spectra_across_all_molecules} spectra of {number_of_molecules} molecule(s) across public metabolomics raw data to retrieve matching samples:")


        # --- detect direct USI queries (no library int IDs available) ---
        has_direct_usi = any(
            (not data["structure"].empty)
            and ("query_spectrum_id" in data["structure"].columns)
            and (data["structure"]["query_spectrum_id"].astype(str).str.startswith("mzspec:").any())
            for ik_dict in st.session_state.grouped_results.values()
            for data in ik_dict.values()
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if has_direct_usi:
                # USI-only queries cannot use FASSTrecords (needs spectrum_id_int / representative_spectrum_int)
                option = st.radio(
                    "Mode",
                    ["FASSTrecords", "FASST"],
                    horizontal=True,
                    key="mode",
                    index=1,          # force FASST
                    disabled=True
                )
            else:
                option = st.radio(
                    "Mode",
                    ["FASSTrecords", "FASST"],
                    horizontal=True,
                    key="mode"
                )
        with col_b:
            st.empty()

        last_iteration = "09/2025"  # need to get version into sql table
        if option == "FASSTrecords":
            min_peaks_allowed = 3
            min_cos_allowed = 0.7
        elif option == "FASST":
            min_peaks_allowed = 1
            min_cos_allowed = 0.3

        info_row = st.empty()  # single, stable placeholder

        def render_info_panels(last_iteration: str):
            with info_row.container():  # render both columns atomically
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
                        This generally takes less than 1 minute even for hundreds of spectra as it relies on<br/>
                        precomputed annotations, and is therefore especially recommended for large numbers of<br/> 
                        spectra or molecules expected in large numbers of samples.<br/>
                        Last iteration: <strong>{last_iteration}</strong>.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
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
                        Can be slow especially for molecules expected to be present in many datasets or large<br/>
                        numbers of spectra. Searches can take several minutes and even fail depending on current traffic.<br/>
                        Allows the discovery of unknown chemical analogues through modification search.<br/>
                        Always up to date with the latest raw data indexed at <a href="https://fasst.gnps2.org/" target="_blank" style="color:#e76f51;">fasst.gnps2.org</a>.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

        # ---- Call exactly once per run, before any button/long work branches ----
        render_info_panels(last_iteration)

        # Cosine and Matching Peaks input 
        col3, col4, _, _ = st.columns(4)
        with col3:
            min_cosine = st.number_input(
                "Minimum Cosine",
                min_value=min_cos_allowed, max_value=1.0, value=0.70, step=0.01,
                key="raw_min_cosine_ui"   # <- new, UI-only key
            )

        with col4:
            min_peaks = st.number_input(
                "Minimum Matching Peaks",
                min_value=min_peaks_allowed, value=5, step=1,
                key="raw_min_peaks_ui"   
            )

        col_par_fr_2, col_par_fr_3, _, _ = st.columns(4)

        with col_par_fr_2:
            min_rank = st.number_input(
                "Minimum Annotation Rank (0 disables)",
                min_value=0, max_value=10, value=0, step=1,
                key="raw_min_rank_ui"
            )
        

        # Conditional input for FASST 
        if option == "FASST":
            col5, col6, _, _ = st.columns(4)
            with col6:
                prec_tol = st.text_input("Precursor Tolerance (Da)", value="0.02", key="prec_tol")
                do_modification_search = st.checkbox("Modification search", value=False, key="do_modification_search")


            with col5:
                frag_tol = st.text_input("Fragment Tolerance (Da)", value="0.02", key="frag_tol")
                if do_modification_search:
                    col_elim, col_add = st.columns(2)
                    with col_elim:
                        do_elimination = st.checkbox("Elimination search", value=True, key="do_elimination")
                    with col_add:
                        do_addition = st.checkbox("Addition search", value=True, key="do_addition")
                    sub_col1, col_or, sub_col2 = st.columns([2,0.5,2])
                    with sub_col1:
                        modification_formula = st.text_input("Modification formula", placeholder="O for hydroxylation", key="modification_formula")
                    with col_or:
                        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)
                    with sub_col2:
                        modification_mass = st.text_input("Modification mass (Da)", placeholder="15.9949 for O", key="modification_mass")
                    do_subsetModificationSearch = st.checkbox("Only report modified molecules if <condition>", value=False, key="do_subsetModificationSearch")

                    if do_subsetModificationSearch:
                        list_of_values = ["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"]
                        modification_condition = st.selectbox("Unmodified found in same", options=list_of_values)

        do_search = st.button("Search Raw Data", key="run_rawdata_search_button", icon=':material/search:')

        # perform the raw data search
        if do_search:
            # Tracking this action
            try:
                umami.new_event(event_name="Do Search Button Clicked")
            except Exception as e:
                print(f"Error tracking event: {e}")

            time.sleep(2)
            with st.spinner("Searching raw data…"): 

                if option == "FASSTrecords" and has_direct_usi:
                    st.error("FASSTrecords cannot run for USI input (no library spectrum_id_int available). Use FASST.")
                    st.stop()

                new_results = {}

                if option == "FASSTrecords":
                    # build each queries aggregated table
                    for name, ik_dict in st.session_state.grouped_results.items():
                        sel_frames = []
                        for ik, data in ik_dict.items():
                            
                            df_struct = data["structure"].copy()

                            # --- dereplicate spectra only if library integer IDs exist ---
                            required = {"spectrum_id_int", "representative_spectrum_int"}
                            if required.issubset(df_struct.columns):
                                df_struct["spectrum_id_int"] = df_struct["spectrum_id_int"].astype("int64")
                                df_struct["representative_spectrum_int"] = df_struct["representative_spectrum_int"].astype("int64")
                                df_struct["similar_library_spectra"] = (
                                    df_struct.groupby("representative_spectrum_int")["spectrum_id_int"]
                                            .transform("size")
                                            .astype("int64")
                                )

                                df_struct["spectrum_difference"] = (
                                    df_struct["spectrum_id_int"] - df_struct["representative_spectrum_int"]
                                )

                                df_struct = df_struct.sort_values(by=["representative_spectrum_int", "spectrum_difference"])
                                df_struct = df_struct.groupby("representative_spectrum_int").first().reset_index()
                                df_struct["spectrum_id_int"] = df_struct["representative_spectrum_int"]
                            else:
                                # USI-only / non-library rows: keep as-is (no derep possible)
                                df_struct = df_struct.copy()
                                if "similar_library_spectra" not in df_struct.columns:
                                    df_struct["similar_library_spectra"] = 1
                                if "unique_spectra_in_mri" not in df_struct.columns:
                                    df_struct["unique_spectra_in_mri"] = 1

                            if not df_struct.empty:
                                sel_frames.append(df_struct)

                        if not sel_frames:
                            st.warning(f"No entries left for **{name}**, skipping.")
                            continue

                        df_for_name = pd.concat(sel_frames, ignore_index=True)

                        (masst_df, redu_df) = tasks.run_get_masst_and_redu_tables(
                            df_for_name,
                            float(min_cosine),
                            int(min_peaks),
                            int(min_rank),
                            config.PATH_TO_SQLITE,
                            config.MASSTRECORDS_ENDPOINT,
                            config.MASSTRECORDS_TIMEOUT,
                            200,
                        )

                        # if Cosine not in redu_df.columns return empty dataframes
                        if "Cosine" not in redu_df.columns or "Matching Peaks" not in redu_df.columns:
                            new_results[name] = {"masst": pd.DataFrame(), "redu": pd.DataFrame()}
                            continue

                        if 'mri_id_int' in redu_df.columns:
                            # add Compound_Name from molecule_overview[name]
                            redu_df = redu_df.merge(
                                st.session_state.molecule_overview[name][["inchikey_first_block", "Compound_Name"]],
                                on="inchikey_first_block",
                                how="left"
                            )
                            
                            redu_df["query_name"] = name

                        new_results[name] = {"masst": pd.DataFrame(), "redu": redu_df}
                    
                elif option == "FASST":
                    
                    fasst_works = True
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            result_ok = test_fasst_api_search_nonblocking() > 5
                            if result_ok:
                                break
                        except Exception:
                            result_ok = False

                        if attempt < max_retries:
                            time.sleep(0.5)  
                        else:
                            st.error(
                                "The FASST API is not responding after multiple attempts. "
                                "Please try again later or contact support. "
                                "FASSTrecords mode remains available — refresh the page to continue."
                            )
                            fasst_works = False


                    # build each queries aggregated table
                    if fasst_works:
                        for name, ik_dict in st.session_state.grouped_results.items():
                            sel_frames = []
                            for ik, data in ik_dict.items():
                                df_struct = data["structure"].copy()

                                # --- dereplicate spectra only if library integer IDs exist ---
                                required = {"spectrum_id_int", "representative_spectrum_int"}
                                if required.issubset(df_struct.columns):
                                    df_struct["spectrum_id_int"] = df_struct["spectrum_id_int"].astype("int64")
                                    df_struct["representative_spectrum_int"] = df_struct["representative_spectrum_int"].astype("int64")
                                    df_struct["similar_library_spectra"] = (
                                        df_struct.groupby("representative_spectrum_int")["spectrum_id_int"]
                                                .transform("size")
                                                .astype("int64")
                                    )

                                    df_struct["spectrum_difference"] = (
                                        df_struct["spectrum_id_int"] - df_struct["representative_spectrum_int"]
                                    )

                                    df_struct = df_struct.sort_values(by=["representative_spectrum_int", "spectrum_difference"])
                                    df_struct = df_struct.groupby("representative_spectrum_int").first().reset_index()
                                    df_struct["spectrum_id_int"] = df_struct["representative_spectrum_int"]
                                else:
                                    # USI-only / non-library rows: keep as-is (no derep possible)
                                    df_struct = df_struct.copy()
                                    if "similar_library_spectra" not in df_struct.columns:
                                        df_struct["similar_library_spectra"] = 1
                                    if "unique_spectra_in_mri" not in df_struct.columns:
                                        df_struct["unique_spectra_in_mri"] = 1

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
                            redu_df = tasks.run_retrieve_raw_data_matches(
                                df_for_name,
                                "metabolomicspanrepo_index_nightly",
                                float(prec_tol),
                                float(frag_tol),
                                float(min_cosine),
                                int(min_peaks),
                                do_modification_search,
                                modification_mass,
                                do_elimination if "do_elimination" in locals() else False,
                                do_addition if "do_addition" in locals() else False,
                                modification_condition if "modification_condition" in locals() else None,
                                config.PATH_TO_SQLITE,
                                config.MASSTRECORDS_ENDPOINT,
                                config.MASSTRECORDS_TIMEOUT,
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

                                redu_df = redu_df.merge(
                                    st.session_state.molecule_overview[name][["inchikey_first_block", "Compound_Name"]],
                                    on="inchikey_first_block",
                                    how="left"
                                )

                                redu_df["query_name"] = name


                            new_results[name] = {"masst": pd.DataFrame(), "redu": redu_df}

                    # store the results in session state
                st.session_state.raw_results = new_results

                for var in ["masst_df", "redu_df", "result_data", "result"]:
                    try:
                        del locals()[var]
                    except KeyError:
                        pass
                gc.collect()

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

                                remove_cols = [
                                    "best_spectral_match", "lib_usi", "USI", "scan_id",
                                    "spectrum_id_int", "representative_spectrum_int",
                                    "spectrum_difference", "unique_spectra_in_mri",
                                    "Check LC peak", "mri", "mri_id_int"
                                ]
                                new_cols = [col for col in new_cols if col not in remove_cols]

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
                                                    <span class="preset-sub">If modifications were enabled, explore if specific modifications differ by body part and disease (especially interesting for drugs).</span>
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

                                    st.plotly_chart(fig, use_container_width=True, key=f"sankey_{result_fig_id}", config=config_sankey_download)

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
                                st.markdown(f"##### Downstream tooling for **{name}**")


                                masst_by_query_button, _ = st.columns([4, 6])

                                with masst_by_query_button:
                                    df_redu_current = st.session_state.raw_results[name]["redu"]
                                    domainmasst_fragment(name=name, output_folder=output_folder, df_redu=df_redu_current)


                                life_button, message_space_life = st.columns([4,4])

                                pair = st.session_state.raw_results.get(name)
                                if pair and isinstance(pair.get("redu"), pd.DataFrame):
                                    redu_tables = {name: pair["redu"]}
                                else:
                                    redu_tables = {}


                                with life_button:

                                    ip_table = st.session_state.query_table
                                    ip_table = ip_table[ip_table["name"].astype(str).str.strip() == name]

                                    lifemasst_fragment(
                                        input_file=ip_table,
                                        structureMASST_op_folder=output_folder,
                                        redu_tables=redu_tables,
                                        append=f" for {name}"
                                    )
                            else:
                                st.warning("No ReDU metadata matches found.")

                redu_tables = {
                    qname: pair["redu"]
                    for qname, pair in st.session_state.raw_results.items()
                    if isinstance(pair.get("redu"), pd.DataFrame)
                }

                
                if len(redu_tables) > 1 and any(len(df) > 1 for df in redu_tables.values()):

                    st.markdown(f"##### Downstream tooling for all molecules")
                    masst_by_query_button, message_space_masst_by_query = st.columns([4,4]) 

                    if all(len(df) > 1 for df in redu_tables.values()):
                        with masst_by_query_button:
                            domainmasst_intersection_fragment(
                                redu_tables=redu_tables,
                                output_folder=output_folder,
                                # Optional overrides:
                                button_label="Populate DomainMASST with Molecule Co-occurrence",
                                job_name="moleculeIntersection",
                            )

                    life_button_all, message_space_life = st.columns([4,4])

                    with life_button_all:
                        lifemasst_fragment(
                            input_file=st.session_state.query_table,
                            structureMASST_op_folder=output_folder,
                            redu_tables=redu_tables
                        )

            else:
                st.warning("No raw data matches found. Please try a different query structure or adjust your search parameters.")
        else:
            st.markdown("")
    raw_data_search_panel()