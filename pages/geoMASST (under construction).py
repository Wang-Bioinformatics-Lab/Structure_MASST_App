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
from geomasst import build_geomasst_map_html

sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
# first-run data preparation, shared with the LifeMASST page
from bin.streamlit_prepare import ensure_ready_ui, render_status
# searching from here offers exactly the StructureMASST controls, and runs the
# same search behind them - see bin/streamlit_search_ui.py
from bin.streamlit_search_ui import (
    build_query_table,
    has_search_input,
    render_search_inputs,
    run_structuremasst_search,
    search_kwargs,
)
from bin.run_masstRecords_queries import _get_fetcher
from bin.shared_data import get_molecule_classes_cached

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


# the class-label dropdown needs the shared ClassyFire table; without it that one
# control simply does not appear
try:
    import config as _config
    _molecule_classes_cache = get_molecule_classes_cached(
        _get_fetcher(_config.PATH_TO_SQLITE, _config.MASSTRECORDS_ENDPOINT,
                     _config.MASSTRECORDS_TIMEOUT)
    )
except Exception:
    _molecule_classes_cache = None


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

search_ui = None

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
    # the StructureMASST controls verbatim: name / SMILES / class / USI, a batch
    # file, and the same advanced options
    search_ui = render_search_inputs(prefix="geo_", molecule_classes=_molecule_classes_cache)
    max_molecule_maps = st.slider(
        "Maximum separate molecule maps", min_value=1, max_value=20, value=5,
        help=("Only matters for a batch file. One map shows every molecule at "
              "once; this caps how many also get their own."),
        key="geo_max_maps",
    )

render_status("geomasst")

if has_results or source == SRC_SEARCH:
    if st.button("Run GeoMASST", key="run_geomasst_btn"):
        # Tracking this action
        try:
            umami.new_event(event_name="GeoMASST Button Clicked")
        except Exception as e:
            print(f"Error tracking event: {e}")

        # GeoMASST ships its basemap assets, so this normally only verifies them
        if not ensure_ready_ui("geomasst"):
            st.stop()


        if source == SRC_SEARCH and not has_search_input("geo_", search_ui):
            st.warning("Enter a name, SMILES/SMARTS, class or USI to search for.")
            st.stop()

        with st.spinner("Running GeoMASST…"):
            if source == SRC_SEARCH:
                query_table, problem = build_query_table("geo_", search_ui)
                if problem:
                    st.error(problem)
                    st.stop()
                try:
                    result = run_structuremasst_search(
                        df_input=query_table, **search_kwargs(search_ui)
                    )
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                    st.stop()

                frames = []
                for _name, _pair in result["raw_results"].items():
                    _df = _pair.get("redu")
                    if _df is None or len(_df) == 0:
                        continue
                    _df = _df.copy()
                    if "query_name" not in _df.columns:
                        _df["query_name"] = _name
                    frames.append(_df)
                if not frames:
                    st.warning("No raw-data matches for that search.")
                    st.stop()
                df_redu = pd.concat(frames, ignore_index=True)
                st.caption(f"{len(df_redu):,} matched samples "
                           f"across {len(frames)} molecule(s).")

                # the map draws the query structures, so keep the SMILES behind
                # each name - as prepare_lifemasst_input does for LifeMASST
                search_structures = {
                    str(r["name"]): str(r.get("original_smiles") or r["query"])
                    for _, r in query_table.iterrows()
                    if str(r.get("type")) == "smiles"
                }
                selected_name = (query_table["name"].iloc[0]
                                 if len(query_table) == 1 else ALL)
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
                structures = search_structures

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



