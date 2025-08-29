import streamlit as st
import os
import streamlit.components.v1 as components

# Write the page label
st.set_page_config(
    page_title="Domain MASST",
    page_icon="🧬",
)


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
    st.warning("Nothing to show here yet.")



