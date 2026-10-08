import os
import json
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------------------
DATA_FOLDER = "data"
RECIPES_DATA_FILE = os.path.join(DATA_FOLDER, "recipes_book_data.json")

# -------------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="WannabeChef • Antique Fairytale Recipe Book",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit UI Chrome & ensure zero outer scrollbars
st.markdown("""
<style>
    #MainMenu, header, footer {
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Cozy warm rustic kitchen dining table background */
    .stApp {
        background-color: #4A2711 !important;
        background-image: 
            radial-gradient(circle at 50% 35%, rgba(255, 230, 190, 0.16) 0%, transparent 70%),
            repeating-linear-gradient(90deg, rgba(20, 10, 5, 0.08) 0px, rgba(20, 10, 5, 0.08) 2px, transparent 2px, transparent 45px),
            linear-gradient(180deg, #573117 0%, #361908 100%) !important;
        overflow: hidden !important;
    }

    .block-container {
        padding: 0 !important;
        margin: 0 auto !important;
        max-width: 1220px !important;
        overflow: hidden !important;
    }

    iframe {
        border: none !important;
        overflow: hidden !important;
        display: block !important;
        margin: 0 auto !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_recipes_json() -> str:
    if os.path.exists(RECIPES_DATA_FILE):
        with open(RECIPES_DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "[]"

recipes_json_str = get_recipes_json()

# Load HTML template
html_path = os.path.join(os.path.dirname(__file__), "antique_flipbook.html")
if not os.path.exists(html_path):
    html_path = "antique_flipbook.html"

with open(html_path, "r", encoding="utf-8") as f:
    html_template = f.read()

flipbook_html = html_template.replace(
    "/* RECIPES_JSON_PLACEHOLDER */",
    f"let RECIPES = {recipes_json_str};"
)

components.html(flipbook_html, height=720, scrolling=False)
