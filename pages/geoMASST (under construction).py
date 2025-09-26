import streamlit as st
import os
import streamlit.components.v1 as components
import sys 

HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'bin'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from plotting import raw_data_sankey, export_hits_map

# Write the page label
st.set_page_config(
    page_title="MoleculePlanet",
    page_icon="🌍",
)

left, right = st.columns([6,1])

with left:
    st.title("GeoMASST (preview)")
    st.write("""
            This page lets you further explore your StructureMASST results in the context of environmental distributions across our plants. 
    """)

output_folder = st.session_state["_session_output_folder"]

if st.session_state.get("raw_results"):
    names = list(st.session_state.raw_results.keys())
    selected_name = st.selectbox("Select result", names, key="result_name")
    selected_tab = st.session_state.raw_results[selected_name]

    df_redu = st.session_state.raw_results[selected_name]["redu"]
    fig_map, _ = export_hits_map(df_redu, engine="mapbox", hover_mri="count", map_style='carto-positron')

    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True}, key=f"map_{names}")

else:
    st.warning("This is a downstream tool. Run StructureMASST first to generate GeoMASST results.")



