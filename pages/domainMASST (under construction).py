import streamlit as st
import os
import streamlit.components.v1 as components

# Add a tracking token
from streamlit.components.v1 import html
html('<script async defer data-website-id="74bc9983-13c4-4da0-89ae-b78209c13aaf" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0) # GNPS2 Global
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="74665d88-3b9d-4812-b8fc-7f55ceb08f11"></script>',  width=0, height=0) # Streamlit Apps
html('<script defer src="https://analytics-api.gnps2.org/script.js" data-website-id="032bfca4-a353-4586-b637-8908d8b71c85"></script>',  width=0, height=0) # Structure MASST


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



