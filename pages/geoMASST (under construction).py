import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components
import sys 


HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'bin'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from plotting import raw_data_sankey, export_hits_map, load_environmental_context
from geomasst_map import build_geomasst_map_html

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
            This page lets you further explore your StructureMASST results in the context of environmental distributions across our planet. 
    """)

output_folder = st.session_state["_session_output_folder"]

if st.session_state.get("raw_results"):
    names = list(st.session_state.raw_results.keys())
    ALL = "All molecules"
    # A TSV search leaves one entry per molecule; mapping them together is the
    # point of that search, so it is the default whenever there is more than one.
    options = ([ALL] + names) if len(names) > 1 else names
    selected_name = st.selectbox("Select chemical", options, key="result_name")

    if selected_name == ALL:
        max_molecule_maps = st.slider(
            "Maximum separate molecule maps", min_value=1, max_value=20, value=5,
            help=("One map shows every molecule at once, sized by how many were matched "
                  "at each site. Pick molecules in the side list to split them out; this "
                  "caps how many of the best-matched molecules get their own map."),
        )
    else:
        max_molecule_maps = 5

    selected_tab = None if selected_name == ALL else st.session_state.raw_results[selected_name]

    if st.button("Run GeoMASST", key="run_geomasst_btn"):
        # Tracking this action
        try:
            umami.new_event(event_name="GeoMASST Button Clicked")
        except Exception as e:
            print(f"Error tracking event: {e}")
        

        with st.spinner("Running GeoMASST…"):
            if selected_name == ALL:
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

            st.session_state["_geomasst_html"] = build_geomasst_map_html(
                df_hits=df_redu,
                df_background=df_context,
                compound_name=selected_name,
                max_molecule_maps=max_molecule_maps,
            )

    if st.session_state.get("_geomasst_html"):
        # tall enough that the controls and both maps fit without the component
        # scrolling inside itself; the controls are sticky in case it still does
        components.html(st.session_state["_geomasst_html"], height=1600, scrolling=True)

else:
    st.warning("This is a downstream tool. Run StructureMASST first to generate GeoMASST results.")



