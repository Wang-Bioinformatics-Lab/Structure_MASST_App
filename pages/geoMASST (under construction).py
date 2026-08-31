import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components
import sys 


HERE = os.path.dirname(__file__)
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'bin'))
# GeoMASST is its own repository, vendored as a submodule
GEOMASST_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'GeoMASST'))

for _p in (PKG_PATH, GEOMASST_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plotting import raw_data_sankey, export_hits_map, load_environmental_context
from run_script import get_library_spectra_display, get_raw_data_results
from geomasst import build_geomasst_map_html

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


# Write the page label
st.set_page_config(
    page_title="MoleculePlanet",
    page_icon="🌍",
)

left, right = st.columns([6,1])

with left:
    st.title("GeoMASST (preview)")
    st.write("""
            This page maps where a molecule was found. Use the StructureMASST results
            from this session, or search here directly.
    """)

output_folder = st.session_state["_session_output_folder"]


def _run_standalone_search(query, searchtype, mode, min_cos, min_peaks,
                           modification_search, elimination, addition):
    """
    Run a search from this page, so GeoMASST does not require a StructureMASST run
    first. Same pipeline StructureMASST uses, driven through the headless helpers.
    """
    lib = get_library_spectra_display(query=query, searchtype=searchtype)
    spectra = lib["all_spectra"]
    if spectra is None or len(spectra) == 0:
        return None, "No library spectra matched that structure."
    res = get_raw_data_results(
        library_spectra=spectra, mode=mode, min_cos=min_cos, min_peaks=min_peaks,
        modification_search=modification_search,
        elimination=elimination, addition=addition,
    )
    redu = res.get("redu")
    if redu is None or len(redu) == 0:
        return None, f"{len(spectra)} library spectra, but no raw-data matches."
    return redu, f"{len(spectra)} library spectra, {len(redu):,} matched samples."


ALL = "All molecules"
selected_name = None
max_molecule_maps = 5

has_results = bool(st.session_state.get("raw_results"))
SRC_RESULTS = "This session's StructureMASST results"
SRC_SEARCH = "Search from here"

# One source, one button. A separate "search" button made GeoMASST feel like two
# tools; the run button behaves the same whichever source the matches come from.
source = st.radio(
    "Matches to map",
    [SRC_RESULTS, SRC_SEARCH] if has_results else [SRC_SEARCH],
    horizontal=True, key="geo_source",
)

sa_query = ""
sa_searchtype = sa_mode = None
sa_modify = False
sa_cos, sa_peaks = 0.70, 5

if source == SRC_RESULTS:
    names = list(st.session_state.raw_results.keys())
    # A batch search leaves one entry per molecule; mapping them together is the
    # point of that search, so it is the default whenever there is more than one.
    options = ([ALL] + names) if len(names) > 1 else names
    selected_name = st.selectbox("Select chemical", options, key="result_name")
    if selected_name == ALL:
        max_molecule_maps = st.slider(
            "Maximum separate molecule maps", min_value=1, max_value=20, value=5,
            help=("One map shows every molecule at once, sized by how many were "
                  "matched at each site. Pick molecules in the side list to split "
                  "them out; this caps how many get their own map."),
        )
else:
    if not has_results:
        st.caption("No StructureMASST results in this session - search here instead.")
    sa_query = st.text_input("SMILES or SMARTS", key="geo_sa_query",
                             placeholder="CCNc1nc(Cl)nc(NC(C)(C)C)n1")
    c1, c2, c3 = st.columns(3)
    with c1:
        sa_searchtype = st.selectbox("Structure match", ["exact", "substructure", "tanimoto"],
                                     key="geo_sa_type")
    with c2:
        sa_mode = st.selectbox("Search backend", ["fasstrecords", "fasst"], key="geo_sa_mode",
                               help="FASSTrecords reads precomputed matches and is much faster. "
                                    "FASST queries the live API and supports modification search.")
    with c3:
        sa_modify = st.checkbox("Modification search", key="geo_sa_modify",
                                help="FASST only. Splits the map by modification.")
    c4, c5 = st.columns(2)
    with c4:
        sa_cos = st.number_input("Min cosine", 0.0, 1.0, 0.70, 0.05, key="geo_sa_cos")
    with c5:
        sa_peaks = st.number_input("Min matched peaks", 1, 50, 5, 1, key="geo_sa_peaks")

if has_results or source == SRC_SEARCH:
    if st.button("Run GeoMASST", key="run_geomasst_btn"):
        # Tracking this action
        try:
            umami.new_event(event_name="GeoMASST Button Clicked")
        except Exception as e:
            print(f"Error tracking event: {e}")
        

        if source == SRC_SEARCH and not sa_query.strip():
            st.warning("Enter a SMILES or SMARTS to search for.")
            st.stop()

        with st.spinner("Running GeoMASST…"):
            if source == SRC_SEARCH:
                df_redu, msg = _run_standalone_search(
                    sa_query.strip(), sa_searchtype, sa_mode, float(sa_cos), int(sa_peaks),
                    bool(sa_modify) and sa_mode == "fasst", True, True,
                )
                if df_redu is None:
                    st.warning(msg)
                    st.stop()
                st.caption(msg)
                selected_name = sa_query.strip()
            elif selected_name == ALL:
                frames = []
                for _name, _pair in st.session_state.raw_results.items():
                    _df = _pair.get("redu")
                    if _df is None or len(_df) == 0:
                        continue
                    _df = _df.copy()
                    if "query_name" not in _df.columns:
                        _df["query_name"] = _name
                    frames.append(_df)
                df_redu = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                df_redu = st.session_state.raw_results[selected_name]["redu"]

            # background layer: environmental ReDU rows we did NOT hit
            df_context = load_environmental_context(df_redu)

            # SMILES behind each result, so the map can draw the query structures
            structures = dict(st.session_state.get("query_by_name") or {})
            if source == SRC_SEARCH:
                structures = {selected_name: selected_name}   # the query is the SMILES

            st.session_state["_geomasst_html"] = build_geomasst_map_html(
                df_hits=df_redu,
                df_background=df_context,
                compound_name=selected_name,
                max_molecule_maps=max_molecule_maps,
                structures=structures,
            )

    if st.session_state.get("_geomasst_html"):
        # tall enough that the controls and both maps fit without the component
        # scrolling inside itself; the controls are sticky in case it still does
        components.html(st.session_state["_geomasst_html"], height=1600, scrolling=True)

else:
    st.warning("This is a downstream tool. Run StructureMASST first to generate GeoMASST results.")



