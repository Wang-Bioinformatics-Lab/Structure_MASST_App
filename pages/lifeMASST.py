import os
import io
import time
import uuid
import base64
import shutil
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
from streamlit_ketcher import st_ketcher

from rdkit import Chem
from rdkit.Chem import Draw

from dotenv import load_dotenv
from formula_validation.Formula import Formula

from bin.match_smiles import detect_smiles_or_smarts, neutralize_atoms, tautomerize_smiles
from bin.pubchem_handling import pubchem_autocomplete, name_to_cid, cid_to_canonical_smiles
from bin.run_masstRecords_queries import _get_fetcher
from bin.shared_data import get_molecule_classes_cached
from bin.api_health import test_fasst_api_search_nonblocking
from bin.linkouts import build_dashboard_eic_url, build_spectraresolver_link
from bin.smarts_api import query_smarts
from bin.streamlit_fragment_LifeMASST import prepare_lifemasst_input, setup_lifemasst_files

import tasks
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

st.logo("logo_LifeMASST.png", icon_image="logo_LifeMASST.png")


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
def tautomerize_neutralize_smiles(smiles: str) -> str:
    try:
        smi = tautomerize_smiles(smiles)
    except Exception:
        smi = smiles
    try:
        smi = neutralize_atoms(smi)
    except Exception:
        pass
    return smi

def _resolve_name_to_smiles(selected_name: str):
    st.session_state["lm_name_warning"] = None
    if not selected_name:
        return
    cid = name_to_cid(selected_name)
    smiles = cid_to_canonical_smiles(cid) if cid else None

    if smiles and "." not in smiles:
        st.session_state["lm_smiles_input"] = smiles
    else:
        st.session_state["lm_smiles_input"] = ""
        st.session_state["lm_name_warning"] = (
            "PubChem entry does not represent a singular molecule "
            "(no SMILES available or multi-component SMILES containing '.')."
        )

def mol_to_base64_img(mol, size=(300, 300)):
    try:
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return (
            f"<img src='data:image/png;base64,{img_str}' "
            f"style='margin-top:1em; display:block; margin-left:0; margin-right:auto; text-align:left;'/>"
        )
    except Exception:
        return "<p style='color:red;'>Failed to draw molecule image.</p>"

def run_shortcut_structuremasst(
    df_input: pd.DataFrame,
    mode: str,
    min_cosine: float,
    min_peaks: int,
    min_rank: int,
    tanimoto_cutoff: str | None,
    prec_tol: float,
    frag_tol: float,
    do_modification_search: bool,
    modification_formula: str,
    modification_mass_text: str,
    do_elimination: bool,
    do_addition: bool,
    modification_condition: str | None,
):
    selected_queries = {}
    grouped_results = {}
    molecule_overview = {}
    raw_results = {}

    def _concat_dedup(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        if a is None or not isinstance(a, pd.DataFrame) or a.empty:
            return b.copy() if isinstance(b, pd.DataFrame) else pd.DataFrame()
        if b is None or not isinstance(b, pd.DataFrame) or b.empty:
            return a
        out = pd.concat([a, b], ignore_index=True, sort=False)
        if "query_spectrum_id" in out.columns:
            out = out.drop_duplicates(subset=["query_spectrum_id"])
        else:
            out = out.drop_duplicates()
        return out

    new_grouped_results = {}
    new_molecule_overview = {}

    for _, row in df_input.iterrows():
        q = row["query"]
        name = row["name"]
        typ = row["type"]
        searchtype = row["searchtype"]
        formula = row["formula"]
        allowed_elements = row["allowed_elements"]

        if typ == "usi" or searchtype == "usi":
            ik = hashlib.sha1(str(q).encode()).hexdigest()[:12]
            df_struct = pd.DataFrame([{
                "inchikey_first_block": ik,
                "Compound_Name": name,
                "Smiles": "",
                "Precursor_MZ": np.nan,
                "query_spectrum_id": q,
                "USI": q,
            }])
            new_grouped_results.setdefault(name, {})
            new_grouped_results[name][ik] = {"structure": df_struct, "conflicts": pd.DataFrame()}
            selected_queries[ik] = [q]
            new_molecule_overview[name] = pd.DataFrame([{
                "Compound_Name": name,
                "inchikey_first_block": ik,
                "Smiles": "",
                "USI": q,
            }])
            continue

        df_library_structurematch = tasks.run_get_library_table(
            q,
            searchtype,
            tanimoto_cutoff if searchtype == "tanimoto" else None,
            formula,
            allowed_elements,
            config.PATH_TO_SQLITE,
            config.MASSTRECORDS_ENDPOINT,
            config.MASSTRECORDS_TIMEOUT,
        )

        if df_library_structurematch.empty:
            continue

        grouped_for_name = {}
        overview_rows = []

        for ik in df_library_structurematch["inchikey_first_block"].astype(str).unique():
            sub_struct = df_library_structurematch[df_library_structurematch["inchikey_first_block"].astype(str) == ik].copy()
            grouped_for_name[ik] = {"structure": sub_struct, "conflicts": pd.DataFrame()}

            qs = list(sub_struct["query_spectrum_id"].dropna().astype(str).unique())
            selected_queries[ik] = qs

            names = sub_struct["Compound_Name"].dropna().astype(str)
            best_name = names.iloc[0] if len(names) else ""
            smiles = sub_struct["Smiles"].dropna().astype(str)
            first_smi = smiles.iloc[0] if len(smiles) else ""

            overview_rows.append({
                "Compound_Name": best_name,
                "inchikey_first_block": ik,
                "Smiles": first_smi
            })

        if grouped_for_name:
            new_grouped_results[name] = grouped_for_name
            new_molecule_overview[name] = pd.DataFrame(overview_rows)

    grouped_results = new_grouped_results
    molecule_overview = new_molecule_overview

    has_direct_usi = any(
        (not data["structure"].empty)
        and ("query_spectrum_id" in data["structure"].columns)
        and (data["structure"]["query_spectrum_id"].astype(str).str.startswith("mzspec:").any())
        for ik_dict in grouped_results.values()
        for data in ik_dict.values()
    )

    if mode == "FASSTrecords" and has_direct_usi:
        raise ValueError("FASSTrecords cannot run for direct USI input. Use FASST.")

    if mode == "FASST":
        result_ok = False
        for _ in range(3):
            try:
                if test_fasst_api_search_nonblocking() > 5:
                    result_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not result_ok:
            raise RuntimeError("FASST API is not responding.")

    for name, ik_dict in grouped_results.items():
        sel_frames = []

        for ik, data in ik_dict.items():
            df_struct = data["structure"].copy()

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
                if "similar_library_spectra" not in df_struct.columns:
                    df_struct["similar_library_spectra"] = 1
                if "unique_spectra_in_mri" not in df_struct.columns:
                    df_struct["unique_spectra_in_mri"] = 1

            if not df_struct.empty:
                sel_frames.append(df_struct)

        if not sel_frames:
            raw_results[name] = {"masst": pd.DataFrame(), "redu": pd.DataFrame()}
            continue

        df_for_name = pd.concat(sel_frames, ignore_index=True)

        if mode == "FASSTrecords":
            masst_df, redu_df = tasks.run_get_masst_and_redu_tables(
                df_for_name,
                float(min_cosine),
                int(min_peaks),
                int(min_rank),
                config.PATH_TO_SQLITE,
                config.MASSTRECORDS_ENDPOINT,
                config.MASSTRECORDS_TIMEOUT,
                200,
            )

            if "Cosine" not in redu_df.columns or "Matching Peaks" not in redu_df.columns:
                raw_results[name] = {"masst": pd.DataFrame(), "redu": pd.DataFrame()}
                continue

            if "inchikey_first_block" in redu_df.columns and not molecule_overview[name].empty:
                redu_df["inchikey_first_block"] = redu_df["inchikey_first_block"].astype(str)
                ov = molecule_overview[name][["inchikey_first_block", "Compound_Name"]].copy()
                ov["inchikey_first_block"] = ov["inchikey_first_block"].astype(str)
                redu_df = redu_df.merge(ov, on="inchikey_first_block", how="left")

            redu_df["query_name"] = name
            raw_results[name] = {"masst": pd.DataFrame(), "redu": redu_df}

        else:
            formulaModi_object = Formula.formula_from_str(modification_formula) if do_modification_search and modification_formula else None
            try:
                modification_mass = formulaModi_object.get_monoisotopic_mass()
            except AttributeError:
                modification_mass = float(modification_mass_text) if modification_mass_text else None

            redu_df = tasks.run_retrieve_raw_data_matches(
                df_for_name,
                "metabolomicspanrepo_index_nightly",
                float(prec_tol),
                float(frag_tol),
                float(min_cosine),
                int(min_peaks),
                do_modification_search,
                modification_mass,
                do_elimination,
                do_addition,
                modification_condition,
                config.PATH_TO_SQLITE,
                config.MASSTRECORDS_ENDPOINT,
                config.MASSTRECORDS_TIMEOUT,
                st.session_state["_session_output_folder"],
            )

            if len(redu_df) > 0:
                redu_df["lib_usi"] = redu_df["query_spectrum_id"].apply(
                    lambda x: (
                        x if str(x).startswith("mzspec:")
                        else f"mzspec:GNPS:GNPS-LIBRARY:accession:{x}" if str(x).startswith("CCMSLIB")
                        else f"mzspec:MASSBANK::accession:{x}"
                    )
                )

                redu_df["best_spectral_match"] = redu_df.apply(
                    lambda row: build_spectraresolver_link(row["USI"], row["lib_usi"]),
                    axis=1
                )

                if "Check LC peak" not in redu_df.columns:
                    redu_df["Check LC peak"] = np.nan
                redu_df["Check LC peak"] = redu_df["Check LC peak"].astype(object)

                mask = redu_df["Check LC peak"].isna() | (redu_df["Check LC peak"].astype(str).str.strip() == "")
                redu_df.loc[mask, "Check LC peak"] = redu_df.loc[mask].apply(
                    lambda row: build_dashboard_eic_url(
                        usi=row["USI"],
                        xic_mz=row["Precursor_MZ"],
                        xic_tolerance=0.05
                    ),
                    axis=1
                )

                if "inchikey_first_block" in redu_df.columns and not molecule_overview[name].empty:
                    redu_df["inchikey_first_block"] = redu_df["inchikey_first_block"].astype(str)
                    ov = molecule_overview[name][["inchikey_first_block", "Compound_Name"]].copy()
                    ov["inchikey_first_block"] = ov["inchikey_first_block"].astype(str)
                    redu_df = redu_df.merge(ov, on="inchikey_first_block", how="left")

                redu_df["query_name"] = name

            raw_results[name] = {"masst": pd.DataFrame(), "redu": redu_df}

    return {
        "query_table": df_input,
        "grouped_results": grouped_results,
        "molecule_overview": molecule_overview,
        "raw_results": raw_results,
    }
# ------------------------------
# Compact shortcut input UI
# ------------------------------
uploaded_file = None
shortcut_smiles_type = None
effective_smiles = ""

show_shortcut_inputs = not lifemasst_already_prepared_from_structuremasst

def render_shortcut_inputs_and_advanced():
    global uploaded_file, shortcut_smiles_type, effective_smiles

    # Row 1: name | or | smiles | or | class | or | usi
    col_name, col_or1, col_smiles, col_or2, col_class, col_or3, col_usi = st.columns([4, 1, 4, 1, 4, 1, 4])

    with col_name:
        name_query = st.text_input(
            "Chemical name",
            key="lm_name_query",
            placeholder="e.g. diazepam, caffeine, surfactin C",
            on_change=lambda: st.session_state.update({
                "lm_structure_editor_open": False,
                "lm_new_smiles": "",
                "lm_smiles_input": "",
                "lm_class_label": "",
                "lm_usi_input": "",
            }),
        )

        if name_query and name_query != st.session_state["lm_last_fetched_query"]:
            suggestions = pubchem_autocomplete(name_query) or []
            st.session_state["lm_name_suggestions"] = suggestions
            st.session_state["lm_last_fetched_query"] = name_query

            if suggestions:
                st.session_state["lm_name_choice"] = suggestions[0]
                _resolve_name_to_smiles(suggestions[0])

        suggestions = st.session_state["lm_name_suggestions"]
        if suggestions:
            def _on_choice_change():
                _resolve_name_to_smiles(st.session_state.get("lm_name_choice"))
                st.session_state["lm_structure_editor_open"] = False
                st.session_state["lm_new_smiles"] = st.session_state.get("lm_smiles_input", "")

            st.selectbox(
                "Suggestions",
                options=suggestions,
                key="lm_name_choice",
                index=(
                    suggestions.index(st.session_state["lm_name_choice"])
                    if st.session_state["lm_name_choice"] in suggestions else 0
                ),
                on_change=_on_choice_change,
            )

        if st.session_state["lm_name_warning"]:
            st.warning(st.session_state["lm_name_warning"])

    with col_or1:
        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

    with col_smiles:
        smiles_input = st.text_input(
            "SMILES/SMARTS",
            key="lm_smiles_input",
            placeholder="Enter SMILES or SMARTS",
            on_change=lambda: st.session_state.update({
                "lm_structure_editor_open": False,
                "lm_new_smiles": "",
                "lm_name_query": "",
                "lm_name_suggestions": [],
                "lm_class_label": "",
                "lm_usi_input": "",
            }),
        )
        smiles_input = smiles_input.strip()
        effective_smiles = st.session_state.get("lm_new_smiles", "") or smiles_input
        shortcut_smiles_type = detect_smiles_or_smarts(effective_smiles) if effective_smiles else None

        if shortcut_smiles_type == "smiles" and effective_smiles:
            edit_button = st.button("Edit structure", key="lm_edit_structure")
            if edit_button:
                st.session_state["lm_structure_editor_open"] = True

            if edit_button or st.session_state["lm_structure_editor_open"]:
                with st.expander("Structure editor", expanded=True):
                    if st.button("Close structure editor", key="lm_close_editor"):
                        st.session_state["lm_structure_editor_open"] = False
                        st.rerun()
                    new_smiles = st_ketcher(effective_smiles)
            else:
                new_smiles = effective_smiles

            st.session_state["lm_new_smiles"] = new_smiles
            effective_smiles = new_smiles

    with col_or2:
        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

    with col_class:
        if _molecule_classes_cache is not None:
            class_labels = _molecule_classes_cache["class_label"].tolist()
            class_labels.insert(0, "")
            st.selectbox(
                "Class label",
                options=class_labels,
                key="lm_class_label",
                help="Select a ClassyFire molecule class.",
                on_change=lambda: st.session_state.update({
                    "lm_structure_editor_open": False,
                    "lm_new_smiles": "",
                    "lm_smiles_input": "",
                    "lm_usi_input": "",
                }),
            )

    with col_or3:
        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

    with col_usi:
        usi_input = st.text_input(
            "USI",
            key="lm_usi_input",
            placeholder="mzspec:...",
            on_change=lambda: st.session_state.update({
                "lm_smiles_input": "",
                "lm_new_smiles": "",
                "lm_structure_editor_open": False,
                "lm_class_label": "",
                "lm_name_query": "",
                "lm_name_suggestions": [],
            }),
        ).strip()

    # Row 2: narrow batch uploader at far left, structure preview immediately to its right
    row2_col_left, row2_col_preview, row2_col_rest = st.columns([3, 4, 14])

    with row2_col_left:
        with st.popover("Add batch file", icon=":material/file_upload:"):
            uploaded_file = st.file_uploader(
                "Drop CSV file for batch search (query/name or smiles/name).",
                type=["csv"],
                key="lm_uploaded_csv",
            )

    with row2_col_preview:
        if shortcut_smiles_type == "smiles" and effective_smiles:
            mol = Chem.MolFromSmiles(effective_smiles)
            if mol:
                st.markdown(
                    f"""
                    <div style="text-align:left; width:100%;">
                        {mol_to_base64_img(mol)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Advanced options stay inside the same shortcut block
    with st.expander("Advanced search options", expanded=False):
        search_options = ["Exact structure match", "Substructure match", "Tanimoto similarity"]
        if st.session_state.get("lm_class_label", "") != "":
            search_options = ["Exact structure match"]
        elif shortcut_smiles_type == "smarts":
            search_options = ["Substructure match"]

        shortcut_searchtype_ui = st.radio(
            "Library search type",
            search_options,
            horizontal=True,
            index=0,
            key="lm_searchtype_ui",
        )

        allowed_formula = ""
        allowed_elements = ""
        tanimoto_cutoff = "0.8"

        if shortcut_searchtype_ui != "Exact structure match":
            sc1, sc2 = st.columns(2)
            with sc1:
                allowed_formula = st.text_input(
                    "Fixed molecular formula (optional)",
                    value="",
                    key="lm_allowed_formula"
                )
            with sc2:
                allowed_elements = st.text_input(
                    "Allowed element difference (optional)",
                    value="",
                    key="lm_allowed_elements"
                )

        if shortcut_searchtype_ui == "Tanimoto similarity":
            tanimoto_cutoff = st.text_input(
                "Tanimoto threshold",
                value="0.8",
                key="lm_tanimoto_threshold"
            )

        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            shortcut_mode = st.radio(
                "Mode",
                ["FASSTrecords", "FASST"],
                horizontal=True,
                index=0,
                key="lm_shortcut_mode",
            )

        if shortcut_mode == "FASSTrecords":
            min_cos_allowed = 0.7
            min_peaks_allowed = 3
        else:
            min_cos_allowed = 0.3
            min_peaks_allowed = 1

        c1, c2, c3 = st.columns(3)
        with c1:
            min_cosine = st.number_input(
                "Minimum cosine",
                min_value=min_cos_allowed,
                max_value=1.0,
                value=0.70,
                step=0.01,
                key="lm_min_cosine",
            )
        with c2:
            min_peaks = st.number_input(
                "Minimum matching peaks",
                min_value=min_peaks_allowed,
                value=5,
                step=1,
                key="lm_min_peaks",
            )
        with c3:
            min_rank = st.number_input(
                "Minimum annotation rank (0 disables)",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="lm_min_rank",
            )

        prec_tol = 0.02
        frag_tol = 0.02
        do_modification_search = False
        do_elimination = True
        do_addition = True
        modification_formula = ""
        modification_mass_text = ""
        modification_condition = None

        if shortcut_mode == "FASST":
            f1, f2 = st.columns(2)
            with f1:
                prec_tol = float(st.text_input("Precursor tolerance (Da)", value="0.02", key="lm_prec_tol"))
            with f2:
                frag_tol = float(st.text_input("Fragment tolerance (Da)", value="0.02", key="lm_frag_tol"))

            do_modification_search = st.checkbox(
                "Modification search",
                value=False,
                key="lm_do_modification_search",
            )

            if do_modification_search:
                m1, m2 = st.columns(2)
                with m1:
                    do_elimination = st.checkbox("Elimination search", value=True, key="lm_do_elimination")
                with m2:
                    do_addition = st.checkbox("Addition search", value=True, key="lm_do_addition")

                m3, m4 = st.columns(2)
                with m3:
                    modification_formula = st.text_input("Modification formula", value="", key="lm_modification_formula")
                with m4:
                    modification_mass_text = st.text_input("Modification mass (Da)", value="", key="lm_modification_mass")

                if st.checkbox("Only report modified molecules if condition is met", value=False, key="lm_do_subset_mod"):
                    modification_condition = st.selectbox(
                        "Condition",
                        options=["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"],
                        key="lm_modification_condition",
                    )

    return {
        "uploaded_file": uploaded_file,
        "shortcut_smiles_type": shortcut_smiles_type,
        "effective_smiles": effective_smiles,
        "shortcut_mode": shortcut_mode,
        "min_cosine": min_cosine,
        "min_peaks": min_peaks,
        "min_rank": min_rank,
        "shortcut_searchtype_ui": shortcut_searchtype_ui,
        "allowed_formula": allowed_formula,
        "allowed_elements": allowed_elements,
        "tanimoto_cutoff": tanimoto_cutoff,
        "prec_tol": prec_tol,
        "frag_tol": frag_tol,
        "do_modification_search": do_modification_search,
        "modification_formula": modification_formula,
        "modification_mass_text": modification_mass_text,
        "do_elimination": do_elimination,
        "do_addition": do_addition,
        "modification_condition": modification_condition,
    }

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
tree_png_path = os.path.join(output_folder, "lifemasst", "tree_plot.png")
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

match_level_selection_masst, _, _, _ = st.columns([3, 3, 3, 3])
with match_level_selection_masst:
    match_level_masst = st.selectbox(
        "Matching level between tree and StructureMASST results",
        options=match_level_options.keys(),
        format_func=lambda x: match_level_options[x],
        index=0,
        help="Select the taxonomic level to match your IDs to in the tree.",
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

_maybe_load_bytes(tree_png_path, "tree_png_bytes", "tree_png_mtime")
_maybe_load_tsv(metadata_tsv_path, "metadata_df", "metadata_tsv_mtime")
_maybe_load_bytes(empress_zip_path, "empress_zip_bytes", "empress_zip_mtime")

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
                    has_shortcut_input = any([
                        bool(st.session_state.get("lm_usi_input", "").strip()),
                        bool((st.session_state.get("lm_new_smiles", "") or st.session_state.get("lm_smiles_input", "")).strip()),
                        bool(st.session_state.get("lm_class_label", "").strip()),
                        uploaded_file is not None,
                        bool(st.session_state.get("lm_name_query", "").strip()),
                    ])

                if has_shortcut_input:
                    shortcut_searchtype = {
                        "Exact structure match": "exact",
                        "Substructure match": "substructure",
                        "Tanimoto similarity": "tanimoto",
                    }[shortcut_searchtype_ui]

                    df_input = None

                    if st.session_state.get("lm_usi_input", "").strip():
                        df_input = pd.DataFrame([{
                            "query": st.session_state["lm_usi_input"].strip(),
                            "name": "USI_Query",
                            "type": "usi",
                            "searchtype": "usi",
                            "formula": "any",
                            "allowed_elements": "any",
                            "original_smiles": "",
                        }])

                    elif (st.session_state.get("lm_new_smiles", "") or st.session_state.get("lm_smiles_input", "")).strip():
                        effective = (st.session_state.get("lm_new_smiles", "") or st.session_state.get("lm_smiles_input", "")).strip()
                        t = detect_smiles_or_smarts(effective)
                        original_smiles = effective if t == "smiles" else ""

                        if t == "smiles":
                            effective = tautomerize_neutralize_smiles(effective)

                        df_input = pd.DataFrame([{
                            "query": effective,
                            "name": "Input_query",
                            "type": t,
                            "searchtype": ("substructure" if t == "smarts" else shortcut_searchtype),
                            "formula": (allowed_formula if shortcut_searchtype == "substructure" else "any"),
                            "allowed_elements": (allowed_elements if shortcut_searchtype == "substructure" else "any"),
                            "original_smiles": original_smiles,
                        }])

                    elif uploaded_file is not None:
                        df_input = pd.read_csv(uploaded_file)

                        if "query" not in df_input.columns:
                            if "smiles" in df_input.columns:
                                df_input = df_input.rename(columns={"smiles": "query"})
                            else:
                                st.error("CSV must contain either ('query','name') or ('smiles','name').")
                                st.stop()

                        if "name" not in df_input.columns:
                            df_input["name"] = [f"Input_{i+1}" for i in range(len(df_input))]

                        df_input = df_input.dropna(subset=["query", "name"]).copy()
                        df_input["query"] = df_input["query"].astype(str).str.strip()
                        df_input["name"] = df_input["name"].astype(str).str.strip()
                        df_input["name"] = df_input["name"].str.replace(r"[^\w]", "_", regex=True)

                        # replace all spaces and special characters in names with underscores and quotes with nothing
                        df_input["name"] = df_input["name"].str.replace(r'[\s\W]+', '_', regex=True)
                        df_input["name"] = df_input["name"].str.replace(r'^_+|_+$', '', regex=True)


                        if "type" not in df_input.columns:
                            def _infer_type(q):
                                q = str(q).strip()
                                if q.startswith("mzspec:"):
                                    return "usi"
                                return detect_smiles_or_smarts(q)
                            df_input["type"] = df_input["query"].apply(_infer_type)

                        df_input["original_smiles"] = ""
                        mask_smiles = df_input["type"].astype(str).str.strip().eq("smiles")
                        df_input.loc[mask_smiles, "original_smiles"] = df_input.loc[mask_smiles, "query"]

                        def _harmonize_query(row):
                            if row["type"] == "smiles":
                                return tautomerize_neutralize_smiles(row["query"])
                            return row["query"]

                        df_input["query"] = df_input.apply(_harmonize_query, axis=1)

                        if "searchtype" not in df_input.columns:
                            def _infer_searchtype(t):
                                if t == "usi":
                                    return "usi"
                                if t == "smarts":
                                    return "substructure"
                                if t == "class_label":
                                    return "class_label"
                                return shortcut_searchtype
                            df_input["searchtype"] = df_input["type"].apply(_infer_searchtype)

                        if "formula" not in df_input.columns:
                            df_input["formula"] = "any"
                        df_input["formula"] = df_input["formula"].fillna("any")

                        if "allowed_elements" not in df_input.columns:
                            df_input["allowed_elements"] = "any"
                        df_input["allowed_elements"] = df_input["allowed_elements"].fillna("any")

                    elif st.session_state.get("lm_class_label", "").strip():
                        class_label = st.session_state["lm_class_label"].strip()
                        df_input = pd.DataFrame([{
                            "query": class_label,
                            "name": class_label,
                            "type": "class_label",
                            "searchtype": "class_label",
                            "formula": "any",
                            "allowed_elements": "any",
                            "original_smiles": "",
                        }])

                    elif st.session_state.get("lm_name_query", "").strip():
                        smi = st.session_state.get("lm_smiles_input", "").strip()
                        if not smi:
                            st.error("Name search did not resolve to a usable single-component SMILES.")
                            st.stop()

                        effective = tautomerize_neutralize_smiles(smi)
                        df_input = pd.DataFrame([{
                            "query": effective,
                            "name": st.session_state.get("lm_name_choice") or st.session_state.get("lm_name_query"),
                            "type": "smiles",
                            "searchtype": shortcut_searchtype,
                            "formula": (allowed_formula if shortcut_searchtype == "substructure" else "any"),
                            "allowed_elements": (allowed_elements if shortcut_searchtype == "substructure" else "any"),
                            "original_smiles": smi,
                        }])

                    if df_input is None or df_input.empty:
                        st.error("No valid shortcut input could be prepared.")
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

                subprocess.run(
                    [
                        "nextflow", "run", NF_PATH,
                        "--input_molecules", molecule_path_abs,
                        "--structureMASST_input_file", structuremasst_path_abs,
                        "--tree_path", tree_path,
                        "--tree_features", feature_path,
                        "--tax_id", match_id,
                        "--tax_id_prefix", id_prefix,
                        "--masst_matchLevel", match_level_masst,
                        "--wikidata_matchLevel", match_level_masst,
                        "--output_folder", out_dir_abs
                    ],
                    check=True
                )

                _maybe_load_bytes(tree_png_path, "tree_png_bytes", "tree_png_mtime")
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
img_bytes = st.session_state.get("tree_png_bytes")
if img_bytes is not None:
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