import streamlit as st
import os
import streamlit.components.v1 as components

# Write the page label
st.set_page_config(
    page_title="Domain MASST",
    page_icon="🧬",
)

left, right = st.columns([6,1])

with left:
    st.title("Domain MASST (preview)")
    st.write("""
            This page lets you further explore your StructureMASST results in the context of different DomainMASST analyses. 
            You can compare and contextualize findings across plants, microbes, human and mouse tissues, foods, and other domains of interest.
    """)


output_folder = st.session_state["_session_output_folder"]

# dropdown menu showing all html files in folders
# List only files (not subfolders)
files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.endswith(".html")]

if files:
    choice = st.selectbox("Choose a file", files)
    st.write(f"You selected: {choice}")
    
    # show selected file
    with open(os.path.join(output_folder, choice), "r", encoding="utf-8") as f:
        html_content = f.read()

    # Expand to full page
    components.html(html_content, height=1000, width=None, scrolling=True)

else:
    st.warning("This is a downstream tool. Run StructureMASST first and press the populate DomainMASST button to generate DomainMASST results.")



