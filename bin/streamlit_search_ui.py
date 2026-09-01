#!/usr/bin/env python3
"""
The StructureMASST search controls, and the search behind them.

Both the LifeMASST and GeoMASST pages let you start a search without going
through StructureMASST first. They should offer the same options, in the same
style, and behave identically - so they share these widgets rather than each
growing their own subset. Every widget key is built from a caller-supplied
prefix, which is what keeps two pages from colliding in session state.

    ui = render_search_inputs(prefix="geo_", molecule_classes=classes)
    if has_search_input("geo_", ui):
        table, err = build_query_table("geo_", ui)
        results = run_structuremasst_search(df_input=table, **search_kwargs(ui))
"""
from __future__ import annotations

import base64
import hashlib
import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher

from rdkit import Chem
from rdkit.Chem import Draw
from formula_validation.Formula import Formula

import config
import tasks
from bin.api_health import test_fasst_api_search_nonblocking
from bin.linkouts import build_dashboard_eic_url, build_spectraresolver_link
from bin.match_smiles import detect_smiles_or_smarts, neutralize_atoms, tautomerize_smiles
from bin.pubchem_handling import pubchem_autocomplete, name_to_cid, cid_to_canonical_smiles

SEARCHTYPE_UI_TO_ARG = {
    "Exact structure match": "exact",
    "Substructure match": "substructure",
    "Tanimoto similarity": "tanimoto",
}


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

def _resolve_name_to_smiles(selected_name: str, prefix: str = "lm_"):
    st.session_state[f"{prefix}name_warning"] = None
    if not selected_name:
        return
    cid = name_to_cid(selected_name)
    smiles = cid_to_canonical_smiles(cid) if cid else None

    if smiles and "." not in smiles:
        st.session_state[f"{prefix}smiles_input"] = smiles
    else:
        st.session_state[f"{prefix}smiles_input"] = ""
        st.session_state[f"{prefix}name_warning"] = (
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


def run_structuremasst_search(
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


STATE_DEFAULTS = {
    "name_query": "",
    "last_fetched_query": None,
    "name_suggestions": [],
    "name_choice": None,
    "usi_input": "",
    "smiles_input": "",
    "name_warning": None,
    "structure_editor_open": False,
    "new_smiles": "",
    "class_label": "",
}


def ensure_state(prefix: str = "lm_") -> None:
    """Seed the session keys these controls read before any widget writes them."""
    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(f"{prefix}{key}", value)


def render_search_inputs(prefix: str = "lm_", molecule_classes=None) -> dict:
    """
    The StructureMASST search controls: name / SMILES / class / USI, a batch
    uploader, and the advanced options. Every widget key is built from
    `prefix`, so two pages can show the same controls without fighting over
    session state.
    """
    ensure_state(prefix)

    uploaded_file = None
    shortcut_smiles_type = None
    effective_smiles = ""
    new_smiles = ""

    # Row 1: name | or | smiles | or | class | or | usi
    col_name, col_or1, col_smiles, col_or2, col_class, col_or3, col_usi = st.columns([4, 1, 4, 1, 4, 1, 4])

    with col_name:
        name_query = st.text_input(
            "Chemical name",
            key=f"{prefix}name_query",
            placeholder="e.g. diazepam, caffeine, surfactin C",
            on_change=lambda: st.session_state.update({
                f"{prefix}structure_editor_open": False,
                f"{prefix}new_smiles": "",
                f"{prefix}smiles_input": "",
                f"{prefix}class_label": "",
                f"{prefix}usi_input": "",
            }),
        )

        if name_query and name_query != st.session_state[f"{prefix}last_fetched_query"]:
            suggestions = pubchem_autocomplete(name_query) or []
            st.session_state[f"{prefix}name_suggestions"] = suggestions
            st.session_state[f"{prefix}last_fetched_query"] = name_query

            if suggestions:
                st.session_state[f"{prefix}name_choice"] = suggestions[0]
                _resolve_name_to_smiles(suggestions[0], prefix)

        suggestions = st.session_state[f"{prefix}name_suggestions"]
        if suggestions:
            def _on_choice_change():
                _resolve_name_to_smiles(st.session_state.get(f"{prefix}name_choice"), prefix)
                st.session_state[f"{prefix}structure_editor_open"] = False
                st.session_state[f"{prefix}new_smiles"] = st.session_state.get(f"{prefix}smiles_input", "")

            st.selectbox(
                "Suggestions",
                options=suggestions,
                key=f"{prefix}name_choice",
                index=(
                    suggestions.index(st.session_state[f"{prefix}name_choice"])
                    if st.session_state[f"{prefix}name_choice"] in suggestions else 0
                ),
                on_change=_on_choice_change,
            )

        if st.session_state[f"{prefix}name_warning"]:
            st.warning(st.session_state[f"{prefix}name_warning"])

    with col_or1:
        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

    with col_smiles:
        smiles_input = st.text_input(
            "SMILES/SMARTS",
            key=f"{prefix}smiles_input",
            placeholder="Enter SMILES or SMARTS",
            on_change=lambda: st.session_state.update({
                f"{prefix}structure_editor_open": False,
                f"{prefix}new_smiles": "",
                f"{prefix}name_query": "",
                f"{prefix}name_suggestions": [],
                f"{prefix}class_label": "",
                f"{prefix}usi_input": "",
            }),
        )
        smiles_input = smiles_input.strip()
        effective_smiles = st.session_state.get(f"{prefix}new_smiles", "") or smiles_input
        shortcut_smiles_type = detect_smiles_or_smarts(effective_smiles) if effective_smiles else None

        if shortcut_smiles_type == "smiles" and effective_smiles:
            edit_button = st.button("Edit structure", key=f"{prefix}edit_structure")
            if edit_button:
                st.session_state[f"{prefix}structure_editor_open"] = True

            if edit_button or st.session_state[f"{prefix}structure_editor_open"]:
                with st.expander("Structure editor", expanded=True):
                    if st.button("Close structure editor", key=f"{prefix}close_editor"):
                        st.session_state[f"{prefix}structure_editor_open"] = False
                        st.rerun()
                    new_smiles = st_ketcher(effective_smiles)
            else:
                new_smiles = effective_smiles

            st.session_state[f"{prefix}new_smiles"] = new_smiles
            effective_smiles = new_smiles

    with col_or2:
        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

    with col_class:
        if molecule_classes is not None:
            class_labels = molecule_classes["class_label"].tolist()
            class_labels.insert(0, "")
            st.selectbox(
                "Class label",
                options=class_labels,
                key=f"{prefix}class_label",
                help="Select a ClassyFire molecule class.",
                on_change=lambda: st.session_state.update({
                    f"{prefix}structure_editor_open": False,
                    f"{prefix}new_smiles": "",
                    f"{prefix}smiles_input": "",
                    f"{prefix}usi_input": "",
                }),
            )

    with col_or3:
        st.markdown("<div style='text-align:center; margin-top:2.5em;'>or</div>", unsafe_allow_html=True)

    with col_usi:
        usi_input = st.text_input(
            "USI",
            key=f"{prefix}usi_input",
            placeholder="mzspec:...",
            on_change=lambda: st.session_state.update({
                f"{prefix}smiles_input": "",
                f"{prefix}new_smiles": "",
                f"{prefix}structure_editor_open": False,
                f"{prefix}class_label": "",
                f"{prefix}name_query": "",
                f"{prefix}name_suggestions": [],
            }),
        ).strip()

    # Row 2: narrow batch uploader at far left, structure preview immediately to its right
    row2_col_left, row2_col_preview, row2_col_rest = st.columns([3, 4, 14])

    with row2_col_left:
        with st.popover("Add batch file", icon=":material/file_upload:"):
            uploaded_file = st.file_uploader(
                "Drop CSV file for batch search (query/name or smiles/name).",
                type=["csv"],
                key=f"{prefix}uploaded_csv",
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
        if st.session_state.get(f"{prefix}class_label", "") != "":
            search_options = ["Exact structure match"]
        elif shortcut_smiles_type == "smarts":
            search_options = ["Substructure match"]

        shortcut_searchtype_ui = st.radio(
            "Library search type",
            search_options,
            horizontal=True,
            index=0,
            key=f"{prefix}searchtype_ui",
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
                    key=f"{prefix}allowed_formula"
                )
            with sc2:
                allowed_elements = st.text_input(
                    "Allowed element difference (optional)",
                    value="",
                    key=f"{prefix}allowed_elements"
                )

        if shortcut_searchtype_ui == "Tanimoto similarity":
            tanimoto_cutoff = st.text_input(
                "Tanimoto threshold",
                value="0.8",
                key=f"{prefix}tanimoto_threshold"
            )

        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            shortcut_mode = st.radio(
                "Mode",
                ["FASSTrecords", "FASST"],
                horizontal=True,
                index=0,
                key=f"{prefix}shortcut_mode",
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
                key=f"{prefix}min_cosine",
            )
        with c2:
            min_peaks = st.number_input(
                "Minimum matching peaks",
                min_value=min_peaks_allowed,
                value=5,
                step=1,
                key=f"{prefix}min_peaks",
            )
        with c3:
            min_rank = st.number_input(
                "Minimum annotation rank (0 disables)",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key=f"{prefix}min_rank",
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
                prec_tol = float(st.text_input("Precursor tolerance (Da)", value="0.02", key=f"{prefix}prec_tol"))
            with f2:
                frag_tol = float(st.text_input("Fragment tolerance (Da)", value="0.02", key=f"{prefix}frag_tol"))

            do_modification_search = st.checkbox(
                "Modification search",
                value=False,
                key=f"{prefix}do_modification_search",
            )

            if do_modification_search:
                m1, m2 = st.columns(2)
                with m1:
                    do_elimination = st.checkbox("Elimination search", value=True, key=f"{prefix}do_elimination")
                with m2:
                    do_addition = st.checkbox("Addition search", value=True, key=f"{prefix}do_addition")

                m3, m4 = st.columns(2)
                with m3:
                    modification_formula = st.text_input("Modification formula", value="", key=f"{prefix}modification_formula")
                with m4:
                    modification_mass_text = st.text_input("Modification mass (Da)", value="", key=f"{prefix}modification_mass")

                if st.checkbox("Only report modified molecules if condition is met", value=False, key=f"{prefix}do_subset_mod"):
                    modification_condition = st.selectbox(
                        "Condition",
                        options=["Raw file", "ATTRIBUTE_DatasetAccession", "NCBITaxonomy"],
                        key=f"{prefix}modification_condition",
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



def has_search_input(prefix: str = "lm_", ui: dict | None = None) -> bool:
    """Whether the user actually filled anything in on these controls."""
    return any([
        bool(st.session_state.get(f"{prefix}usi_input", "").strip()),
        bool((st.session_state.get(f"{prefix}new_smiles", "")
              or st.session_state.get(f"{prefix}smiles_input", "")).strip()),
        bool(st.session_state.get(f"{prefix}class_label", "").strip()),
        (ui or {}).get("uploaded_file") is not None,
        bool(st.session_state.get(f"{prefix}name_query", "").strip()),
    ])


def build_query_table(prefix: str = "lm_", ui: dict | None = None):
    """
    Turn whatever the user filled in into the one query table the search takes.

    Returns (DataFrame, None) or (None, message). The message is the caller's to
    render - this stays out of the page's control flow so both pages can use it.
    """
    ui = ui or {}
    uploaded_file = ui.get("uploaded_file")
    shortcut_searchtype = SEARCHTYPE_UI_TO_ARG[ui.get("shortcut_searchtype_ui", "Exact structure match")]
    allowed_formula = ui.get("allowed_formula", "")
    allowed_elements = ui.get("allowed_elements", "")

    df_input = None

    if st.session_state.get(f"{prefix}usi_input", "").strip():
        df_input = pd.DataFrame([{
            "query": st.session_state[f"{prefix}usi_input"].strip(),
            "name": "USI_Query",
            "type": "usi",
            "searchtype": "usi",
            "formula": "any",
            "allowed_elements": "any",
            "original_smiles": "",
        }])

    elif (st.session_state.get(f"{prefix}new_smiles", "") or st.session_state.get(f"{prefix}smiles_input", "")).strip():
        effective = (st.session_state.get(f"{prefix}new_smiles", "") or st.session_state.get(f"{prefix}smiles_input", "")).strip()
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
                return None, "CSV must contain either ('query','name') or ('smiles','name')."

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

    elif st.session_state.get(f"{prefix}class_label", "").strip():
        class_label = st.session_state[f"{prefix}class_label"].strip()
        df_input = pd.DataFrame([{
            "query": class_label,
            "name": class_label,
            "type": "class_label",
            "searchtype": "class_label",
            "formula": "any",
            "allowed_elements": "any",
            "original_smiles": "",
        }])

    elif st.session_state.get(f"{prefix}name_query", "").strip():
        smi = st.session_state.get(f"{prefix}smiles_input", "").strip()
        if not smi:
            return None, "Name search did not resolve to a usable single-component SMILES."

        effective = tautomerize_neutralize_smiles(smi)
        df_input = pd.DataFrame([{
            "query": effective,
            "name": st.session_state.get(f"{prefix}name_choice") or st.session_state.get(f"{prefix}name_query"),
            "type": "smiles",
            "searchtype": shortcut_searchtype,
            "formula": (allowed_formula if shortcut_searchtype == "substructure" else "any"),
            "allowed_elements": (allowed_elements if shortcut_searchtype == "substructure" else "any"),
            "original_smiles": smi,
        }])
    if df_input is None or df_input.empty:
        return None, "No valid search input could be prepared."
    return df_input, None


def structures_for_results(state=None) -> dict:
    """
    The SMILES behind each result name, for drawing query structures downstream.

    Two records exist and neither is complete on its own. query_by_name is
    written when a search is launched, so it knows the names the user typed.
    query_table also gains a row whenever a substructure hit is split into one
    query per matched molecule - a path that leaves query_by_name naming only the
    original SMARTS, which matches no result and draws nothing. Reading both, with
    query_by_name last, covers either.

    Entries that will not parse (a SMARTS, say) are harmless: the map skips what
    it cannot draw.
    """
    state = st.session_state if state is None else state
    out = {}

    table = state.get("query_table")
    if isinstance(table, pd.DataFrame) and not table.empty and "name" in table.columns:
        for _, row in table.iterrows():
            smiles = str(row.get("original_smiles") or "").strip()
            if not smiles and str(row.get("type", "")).strip() == "smiles":
                smiles = str(row.get("query") or "").strip()
            if smiles:
                out[str(row["name"])] = smiles

    for name, smiles in (state.get("query_by_name") or {}).items():
        if smiles:
            out[str(name)] = str(smiles)
    return out


def search_kwargs(ui: dict) -> dict:
    """The run_structuremasst_search() arguments these controls describe."""
    return {
        "mode": ui["shortcut_mode"],
        "min_cosine": float(ui["min_cosine"]),
        "min_peaks": int(ui["min_peaks"]),
        "min_rank": int(ui["min_rank"]),
        "tanimoto_cutoff": ui["tanimoto_cutoff"],
        "prec_tol": float(ui["prec_tol"]),
        "frag_tol": float(ui["frag_tol"]),
        "do_modification_search": bool(ui["do_modification_search"]),
        "modification_formula": ui["modification_formula"],
        "modification_mass_text": ui["modification_mass_text"],
        "do_elimination": bool(ui["do_elimination"]),
        "do_addition": bool(ui["do_addition"]),
        "modification_condition": ui["modification_condition"],
    }
