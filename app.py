import os
import json
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    /* Completely eliminate all Streamlit header, toolbar, status & decorations */
    header,
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    footer,
    #MainMenu, 
    [data-testid="stToolbar"], 
    [data-testid="stDecoration"], 
    [data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        max-height: 0 !important;
        min-height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        position: absolute !important;
        pointer-events: none !important;
        border: none !important;
    }
    
    /* Cozy warm rustic kitchen dining table background */
    html, body, .stApp {
        background-color: #4A2711 !important;
        background-image: 
            radial-gradient(circle at 50% 35%, rgba(255, 230, 190, 0.16) 0%, transparent 70%),
            repeating-linear-gradient(90deg, rgba(20, 10, 5, 0.08) 0px, rgba(20, 10, 5, 0.08) 2px, transparent 2px, transparent 45px),
            linear-gradient(180deg, #573117 0%, #361908 100%) !important;
        overflow: hidden !important;
        height: 100vh !important;
        height: 100dvh !important;
        max-height: 100vh !important;
        max-height: 100dvh !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main,
    .stMain {
        padding: 0 !important;
        margin: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
        max-height: 100vh !important;
        max-height: 100dvh !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stMainBlockContainer"],
    .block-container,
    .stMainBlockContainer,
    [data-testid="stAppViewBlockContainer"] {
        padding: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 auto !important;
        max-width: 100vw !important;
        width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
        max-height: 100vh !important;
        max-height: 100dvh !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="element-container"],
    div[data-testid="stCustomComponentV1"] {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0 !important;
    }

    iframe {
        border: none !important;
        overflow: hidden !important;
        display: block !important;
        margin: 0 auto !important;
        width: 100vw !important;
        max-width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
        max-height: 100vh !important;
        max-height: 100dvh !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_recipes_json() -> str:
    candidates = [
        os.path.join(BASE_DIR, "data", "recipes_book_data.json"),
        os.path.join("data", "recipes_book_data.json"),
        "recipes_book_data.json"
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    content = f.read()
                    if len(content) > 100:
                        return content
            except Exception:
                pass
    
    for root, _, files in os.walk(BASE_DIR):
        if "recipes_book_data.json" in files:
            with open(os.path.join(root, "recipes_book_data.json"), "r", encoding="utf-8") as f:
                return f.read()
                
    return "[]"

recipes_json_str = get_recipes_json()

# Load HTML template
html_candidates = [
    os.path.join(BASE_DIR, "antique_flipbook.html"),
    "antique_flipbook.html"
]
html_path = None
for hc in html_candidates:
    if os.path.exists(hc):
        html_path = hc
        break

if not html_path:
    st.error("antique_flipbook.html not found.")
    st.stop()

with open(html_path, "r", encoding="utf-8") as f:
    html_template = f.read()

flipbook_html = html_template.replace(
    "/* RECIPES_JSON_PLACEHOLDER */",
    f"let RECIPES = {recipes_json_str};"
)

components.html(flipbook_html, height=520, scrolling=False)
