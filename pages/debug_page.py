import streamlit as st
import os
import streamlit.components.v1 as components
import sys 


HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'bin'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)



# Write the page label
st.set_page_config(
    page_title="Debug MASST Page",
    page_icon="🛠️",
)

left, right = st.columns([6,1])

with left:
    st.title("Debug MASST Page")
    st.write("""
            Debugging information for StructureMASST runs can be viewed here.
    """)

output_folder = st.session_state["_session_output_folder"]

# read debug_raw_matches.tsv if it exists
debug_raw_matches_path = os.path.join(output_folder, "debug_raw_matches.tsv")
redu_path = os.path.join(output_folder, "redu_table.tsv")
if os.path.exists(debug_raw_matches_path):
    import pandas as pd
    df_raw_matches = pd.read_csv(debug_raw_matches_path, sep="\t")
    st.subheader("Debug MASST Results")
    st.dataframe(df_raw_matches)
    
if os.path.exists(redu_path):
    import pandas as pd
    df_redu = pd.read_csv(redu_path, sep="\t")

    # only keep columns mri, USI, usi, DOIDCommonName if they exist
    cols_to_keep = [col for col in ['mri', 'USI', 'usi', 'DOIDCommonName'] if col in df_redu.columns]
    df_redu = df_redu[cols_to_keep]

    st.subheader("ReDU Table")
    st.dataframe(df_redu)


