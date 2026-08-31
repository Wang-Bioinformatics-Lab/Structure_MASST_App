import streamlit as st
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
    selected_name = st.selectbox("Select chemical", names, key="result_name")
    
    selected_tab = st.session_state.raw_results[selected_name]

    if st.button("Run GeoMASST", key="run_geomasst_btn"):
        # Tracking this action
        try:
            umami.new_event(event_name="GeoMASST Button Clicked")
        except Exception as e:
            print(f"Error tracking event: {e}")
        

        with st.spinner("Running GeoMASST…"):
            df_redu = st.session_state.raw_results[selected_name]["redu"]

            # background layer: environmental ReDU rows we did NOT hit
            df_context = load_environmental_context(df_redu)

            st.session_state["_geomasst_html"] = build_geomasst_map_html(
                df_hits=df_redu,
                df_background=df_context,
                compound_name=selected_name,
            )

    if st.session_state.get("_geomasst_html"):
        # tall enough that the controls and both maps fit without the component
        # scrolling inside itself; the controls are sticky in case it still does
        components.html(st.session_state["_geomasst_html"], height=1600, scrolling=True)

else:
    st.warning("This is a downstream tool. Run StructureMASST first to generate GeoMASST results.")



