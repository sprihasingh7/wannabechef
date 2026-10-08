import os
import re
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import streamlit as st
from thefuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------------------
DATA_FOLDER = "data"
FEEDBACK_FILE = "feedback.csv"

# Greatly expanded heuristic nutrition database (per 100g approx)
BASE_NUTRITION_DB: Dict[str, Dict[str, float]] = {
    # Meats / Protein
    "chicken": {"calories": 239, "protein": 27, "fat": 14, "carbs": 0},
    "beef": {"calories": 250, "protein": 26, "fat": 15, "carbs": 0},
    "pork": {"calories": 242, "protein": 27, "fat": 14, "carbs": 0},
    "fish": {"calories": 142, "protein": 22, "fat": 6, "carbs": 0},
    "salmon": {"calories": 208, "protein": 20, "fat": 13, "carbs": 0},
    "shrimp": {"calories": 99, "protein": 24, "fat": 0.3, "carbs": 0.2},
    "egg": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1},
    "paneer": {"calories": 265, "protein": 18, "fat": 20, "carbs": 2},
    "tofu": {"calories": 76, "protein": 8, "fat": 5, "carbs": 3},
    "lentils": {"calories": 116, "protein": 9, "fat": 0.4, "carbs": 20},
    "chickpeas": {"calories": 164, "protein": 9, "fat": 2.6, "carbs": 27},
    "beans": {"calories": 130, "protein": 8, "fat": 0.5, "carbs": 24},

    # Vegetables
    "tomato": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9},
    "onion": {"calories": 40, "protein": 1.1, "fat": 0.1, "carbs": 9.3},
    "garlic": {"calories": 149, "protein": 6.4, "fat": 0.5, "carbs": 33},
    "potato": {"calories": 77, "protein": 2, "fat": 0.1, "carbs": 17},
    "carrot": {"calories": 41, "protein": 0.9, "fat": 0.2, "carbs": 10},
    "spinach": {"calories": 23, "protein": 2.9, "fat": 0.4, "carbs": 3.6},
    "broccoli": {"calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7},
    "bell pepper": {"calories": 26, "protein": 1, "fat": 0.2, "carbs": 6},
    "mushroom": {"calories": 22, "protein": 3.1, "fat": 0.3, "carbs": 3.3},
    "peas": {"calories": 81, "protein": 5.4, "fat": 0.4, "carbs": 14},
    "lettuce": {"calories": 15, "protein": 1.4, "fat": 0.2, "carbs": 2.9},
    "ginger": {"calories": 80, "protein": 1.8, "fat": 0.8, "carbs": 18},
    "chilli": {"calories": 40, "protein": 1.9, "fat": 0.4, "carbs": 8.8},

    # Grains / Carbs
    "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
    "pasta": {"calories": 131, "protein": 5, "fat": 1.1, "carbs": 25},
    "flour": {"calories": 364, "protein": 10, "fat": 1, "carbs": 76},
    "bread": {"calories": 265, "protein": 9, "fat": 3.2, "carbs": 49},

    # Dairy / Fats
    "milk": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5},
    "butter": {"calories": 717, "protein": 0.85, "fat": 81, "carbs": 0.06},
    "oil": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0},
    "olive oil": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0},
    "cheese": {"calories": 402, "protein": 25, "fat": 33, "carbs": 1.3},
    "yogurt": {"calories": 59, "protein": 10, "fat": 0.4, "carbs": 3.6},

    # Misc / Spices
    "salt": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
    "sugar": {"calories": 387, "protein": 0, "fat": 0, "carbs": 100},
    "turmeric": {"calories": 312, "protein": 9, "fat": 3.3, "carbs": 67},
    "cumin": {"calories": 375, "protein": 18, "fat": 22, "carbs": 44},
    "paprika": {"calories": 282, "protein": 14, "fat": 13, "carbs": 54},
    "coriander": {"calories": 23, "protein": 2.1, "fat": 0.5, "carbs": 3.7},
}

CUISINE_THEMES: Dict[str, Dict[str, str]] = {
    "Indian": {"emoji": "🍛", "subtitle": "Aromatic spices & timeless heritage"},
    "North Indian Recipes": {"emoji": "🥘", "subtitle": "Rich gravies, buttery naans & warm masalas"},
    "South Indian Recipes": {"emoji": "🥥", "subtitle": "Crispy dosas, coconut chutneys & comforting sambar"},
    "Continental": {"emoji": "🥗", "subtitle": "Herbal delicacies, velvety sauces & bistro classics"},
    "Italian Recipes": {"emoji": "🍝", "subtitle": "Handmade pasta, rustic focaccia & sun-kissed basil"},
    "Bengali Recipes": {"emoji": "🐟", "subtitle": "Panch phoron, mustard pungency & festive treats"},
    "Maharashtrian Recipes": {"emoji": "🌶️", "subtitle": "Spicy goda masala, poha & crunchy bites"},
    "Kerala Recipes": {"emoji": "🌴", "subtitle": "Fresh coastal breezes, coconut milk & spices"},
    "Tamil Nadu": {"emoji": "🍚", "subtitle": "Chettinad zest, tamarind tang & warm comforts"},
    "Karnataka": {"emoji": "🍲", "subtitle": "Bisi bele bath, crispy vadas & wholesome lentils"},
    "Chinese": {"emoji": "🥢", "subtitle": "Wok-fired aromatics, dumplings & savory glazes"},
    "Mexican": {"emoji": "🌮", "subtitle": "Charred peppers, zesty limes & lively tacos"},
    "Thai": {"emoji": "🍜", "subtitle": "Lemongrass, creamy coconut & fragrant Thai basil"},
    "Dessert": {"emoji": "🧁", "subtitle": "Sweet bakes, decadent creams & sugary joy"},
}

# -------------------------------------------------------------------
# Utilities: Data Loading & Cleaning
# -------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

@st.cache_data(show_spinner="Opening the Chef's Recipe Book...")
def load_and_combine_datasets(file_paths: List[str]) -> pd.DataFrame:
    dfs = []
    required_columns = [
        'srno', 'recipename', 'translatedrecipename', 'ingredients', 'translatedingredients',
        'preptimeinmins', 'cooktimeinmins', 'totaltimeinmins', 'servings',
        'cuisine', 'course', 'instructions', 'translatedinstructions', 'url', 'image_url'
    ]
    for file_path in file_paths:
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                df = standardize_columns(df)
                
                if 'recipename' not in df.columns:
                    if 'name' in df.columns:
                        df['recipename'] = df['name']
                    elif 'translatedrecipename' in df.columns:
                        df['recipename'] = df['translatedrecipename']
                    else:
                        df['recipename'] = 'Untitled Recipe'
                
                if 'image_url' not in df.columns:
                    df['image_url'] = None
                
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = None
                
                dfs.append(df[required_columns])
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
    
    if not dfs:
        return pd.DataFrame(columns=required_columns)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=["recipename", "ingredients"], inplace=True)
    combined_df['recipename'] = combined_df['recipename'].fillna('Untitled Recipe')
    
    combined_df['ingredients'] = combined_df['ingredients'].apply(
        lambda x: ", ".join(sorted(list(set(i.strip().lower() for i in str(x).split(","))))) if pd.notna(x) else ""
    )
    
    combined_df['cuisine'] = combined_df['cuisine'].fillna('unknown')
    combined_df['course'] = combined_df['course'].fillna('unknown')
    
    for col in ['preptimeinmins', 'cooktimeinmins', 'totaltimeinmins', 'servings']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    combined_df['totaltimeinmins'] = combined_df['totaltimeinmins'].fillna(0)
    combined_df['servings'] = combined_df['servings'].fillna(1)
    
    return combined_df

# -------------------------------------------------------------------
# Matching & Intelligence Functions
# -------------------------------------------------------------------
def fuzzy_match_score(recipe_ingredients: str, input_ingredients: List[str], threshold=75) -> float:
    if not recipe_ingredients or pd.isna(recipe_ingredients):
        return 0.0
    recipe_list = [r.strip() for r in str(recipe_ingredients).split(",") if r.strip()]
    if not recipe_list or not input_ingredients:
        return 0.0
    
    matches_found = 0
    for ing in input_ingredients:
        best_score = 0
        for rec_ing in recipe_list:
            score = fuzz.partial_ratio(ing, rec_ing)
            if score > best_score:
                best_score = score
        if best_score >= threshold:
            matches_found += 1
            
    return (matches_found / len(input_ingredients)) * 100.0

def get_ingredient_match_details(recipe_ingredients_str: str, user_ingredients: List[str], threshold=75) -> Dict[str, List[str]]:
    if not recipe_ingredients_str or pd.isna(recipe_ingredients_str):
        return {"matched": [], "missing": []}
        
    recipe_ings = [r.strip() for r in str(recipe_ingredients_str).split(",") if r.strip()]
    user_ings_lower = [u.strip().lower() for u in user_ingredients]
    
    matched = []
    missing = []
    
    for r_ing in recipe_ings:
        best_match_score = 0
        for u_ing in user_ings_lower:
            score = fuzz.partial_ratio(u_ing, r_ing)
            if score > best_match_score:
                best_match_score = score
                
        if best_match_score >= threshold:
            matched.append(r_ing.title())
        else:
            missing.append(r_ing.title())
            
    return {"matched": sorted(list(set(matched))), "missing": sorted(list(set(missing)))}

# -------------------------------------------------------------------
# Nutrition Estimator Heuristic
# -------------------------------------------------------------------
def estimate_nutrition_from_ingredients(ingredients: List[str], servings: float = 1.0) -> Dict[str, float]:
    total = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    found = 0
    matched_db_keys: Set[str] = set()

    if not ingredients or (len(ingredients) == 1 and (not ingredients[0] or pd.isna(ingredients[0]))):
        return total

    for ing in ingredients:
        if not ing or pd.isna(ing):
            continue
        key = str(ing).strip().lower()
        best_key = None
        best_score = 0
        
        for dbk in BASE_NUTRITION_DB.keys():
            s = fuzz.partial_ratio(key, dbk)
            if s > best_score:
                best_score = s
                best_key = dbk
        
        if best_score >= 80 and best_key and best_key not in matched_db_keys:
            matched_db_keys.add(best_key)
            found += 1
            nd = BASE_NUTRITION_DB[best_key]
            for k in total:
                total[k] += nd[k]
                
    if found == 0:
        return {k: 0.0 for k in total}

    servings = max(1.0, float(servings) if servings and pd.notna(servings) else 1.0)
    per_serving = {k: round(total[k] / servings, 2) for k in total}
    return per_serving

def healthiness_label(nutrition_per_serving: Dict[str, float]) -> str:
    cal = nutrition_per_serving.get("calories", 0)
    fat = nutrition_per_serving.get("fat", 0)
    protein = nutrition_per_serving.get("protein", 0)
    
    if cal == 0 and fat == 0 and protein == 0:
        return "Chef's Secret ❓"
    if cal > 650 or fat > 35:
        return "Cozy Indulgence 🍕"
    if cal <= 400 and protein >= 15 and fat <= 20:
        return "Wholesome Choice 🥦"
    if cal <= 300 and protein < 10:
        return "Light Delight 🥗"
    return "Balanced Feast ⚖️"

# -------------------------------------------------------------------
# Predictors & Classifiers
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Seasoning AI intuition...")
def train_text_classifier(df: pd.DataFrame, target_col: str) -> Tuple[MultinomialNB, CountVectorizer]:
    df2 = df.copy()
    df2 = df2[df2[target_col].notna()]
    df2 = df2[df2[target_col].str.lower() != "unknown"]
    
    if df2.shape[0] < 20:
        return None, None
    
    vec = CountVectorizer(ngram_range=(1,2), min_df=2, stop_words='english')
    X = vec.fit_transform(df2["ingredients"].astype(str))
    y = df2[target_col].astype(str)
    
    clf = MultinomialNB()
    clf.fit(X, y)
    return clf, vec

def predict_text_label(model_vec_tuple, ingredients: List[str]) -> str:
    model, vec = model_vec_tuple
    if model is None or vec is None:
        return "unknown"
    try:
        X = vec.transform([", ".join(ingredients)])
        return model.predict(X)[0]
    except Exception:
        return "unknown"

@st.cache_resource(show_spinner="Indexing recipe flavors...")
def prepare_tfidf(series: pd.Series) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))
    mat = vec.fit_transform(series.astype(str))
    return vec, mat

# -------------------------------------------------------------------
# Feedback & Scoring
# -------------------------------------------------------------------
def load_feedback_summary(feedback_file: str) -> Dict[str, float]:
    if not os.path.exists(feedback_file):
        return {}
    try:
        fdf = pd.read_csv(feedback_file)
        if fdf.empty:
            return {}
        if 'selected_recipe' in fdf.columns and 'rating' in fdf.columns:
            return fdf.groupby('selected_recipe')['rating'].mean().to_dict()
    except Exception:
        return {}
    return {}

def compute_intelligence_scores(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix,
    input_ingredients: List[str],
    predicted_cuisine: str,
    predicted_course: str,
    user_cuisine_filter: str,
    feedback_summary: Dict[str, float],
    semantic_weight: float = 0.55,
    fuzzy_weight: float = 0.35
) -> pd.DataFrame:
    user_query = " ".join(input_ingredients)
    user_vec = tfidf_vectorizer.transform([user_query])
    semantic_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    fuzzy_scores = df['ingredients'].apply(lambda x: fuzzy_match_score(x, input_ingredients))

    def bonus(row):
        b = 0.0
        c = str(row.get('cuisine', '')).lower()
        co = str(row.get('course', '')).lower() if row.get('course') is not None else ""
        if predicted_cuisine and predicted_cuisine.lower() in c:
            b += 0.05
        if predicted_course and predicted_course.lower() in co:
            b += 0.03
        if user_cuisine_filter and user_cuisine_filter.lower() != "all chapters" and user_cuisine_filter.lower() in c:
            b += 0.07
        return b

    df = df.copy()
    df['semantic_score'] = semantic_scores
    df['fuzzy_score'] = fuzzy_scores
    df['fuzzy_norm'] = df['fuzzy_score'] / 100.0
    df['bonus'] = df.apply(bonus, axis=1)

    def feedback_boost(row):
        name = row.get('recipename')
        if name in feedback_summary:
            return (feedback_summary[name] / 5.0) * 0.08
        return 0.0

    df['feedback_boost'] = df.apply(feedback_boost, axis=1)

    df['intelligence_score'] = (
        (semantic_weight * df['semantic_score']) + 
        (fuzzy_weight * df['fuzzy_norm']) + 
        df['bonus'] + 
        df['feedback_boost']
    )
    return df

def append_feedback(feedback_file: str, selected_recipe: str, user_ingredients: List[str], rating: int):
    rec = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_recipe": selected_recipe,
        "user_ingredients": "|".join(user_ingredients),
        "rating": rating
    }
    if os.path.exists(feedback_file):
        try:
            df = pd.read_csv(feedback_file)
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([rec])
    else:
        df = pd.DataFrame([rec])
    df.to_csv(feedback_file, index=False)

# -------------------------------------------------------------------
# Streamlit App Setup & Self-Contained Open Book Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="WannabeChef • The Cozy Recipe Book",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Open Book CSS: Two-page spread with book cover border and spine crease
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Patrick+Hand&family=Quicksand:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
    }
    
    /* Cozy wooden kitchen countertop background */
    .stApp {
        background-color: #EDE4DC;
        background-image: 
            radial-gradient(#DFD3C8 1px, transparent 1px),
            linear-gradient(135deg, rgba(235, 222, 211, 0.5) 0%, rgba(220, 205, 192, 0.5) 100%);
        background-size: 24px 24px, 100% 100%;
        color: #38302E;
    }

    /* Hide excess padding so entire book fits comfortably on screen without scrolling */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
        max-width: 1400px !important;
    }

    /* Red Silk Bookmark Ribbon */
    .book-ribbon {
        position: relative;
        margin: 0 auto -10px auto;
        width: 240px;
        background: #E63946;
        color: #FFFFFF;
        padding: 6px 14px 12px 14px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        border-radius: 4px 4px 0 0;
        clip-path: polygon(0 0, 100% 0, 100% 82%, 50% 100%, 0 82%);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        z-index: 99;
        text-align: center;
    }

    /* The Main Open Book Spread: Hardcover Leather Outline with Inner Gold Bevel */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        background: #FFFDF9;
        border: 4px solid #5C3024 !important;
        outline: 10px solid #7A4232 !important;
        outline-offset: 2px !important;
        border-radius: 22px !important;
        box-shadow: 
            0 25px 60px rgba(50, 25, 18, 0.4),
            0 8px 25px rgba(0, 0, 0, 0.25) !important;
        padding: 0 !important;
        margin-top: 8px !important;
        position: relative;
    }

    /* Left Page Column */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        background-color: #FFFDF9;
        background-image: linear-gradient(to right, #FFFDF9 92%, #F3EAE1 100%);
        border-right: 2px solid #E2D3C4 !important;
        border-radius: 18px 0 0 18px !important;
        padding: 22px 26px !important;
        height: 780px !important;
        overflow-y: auto !important;
        box-shadow: inset -15px 0 20px -10px rgba(80, 50, 35, 0.08);
    }

    /* Right Page Column */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
        background-color: #FFFDF9;
        background-image: linear-gradient(to left, #FFFDF9 92%, #F3EAE1 100%);
        border-radius: 0 18px 18px 0 !important;
        padding: 22px 26px !important;
        height: 780px !important;
        overflow-y: auto !important;
        box-shadow: inset 15px 0 20px -10px rgba(80, 50, 35, 0.08);
    }

    /* Scrollbars inside the book pages */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]::-webkit-scrollbar-thumb {
        background: #D9C3B0;
        border-radius: 10px;
    }

    /* Book Page Header Banner */
    .page-header-stamp {
        font-family: 'Patrick Hand', cursive;
        font-size: 15px;
        color: #9C7A68;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px dashed #E2D3C4;
        padding-bottom: 6px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
    }

    /* Cute Washi Tape */
    .washi-tape-strip {
        width: 120px;
        height: 22px;
        background: rgba(255, 182, 193, 0.75);
        margin: -8px auto 10px auto;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transform: rotate(-1.5deg);
        border-left: 3px dashed rgba(255,255,255,0.7);
        border-right: 3px dashed rgba(255,255,255,0.7);
    }

    /* Index entry cards */
    .index-entry-card {
        background: #FFFFFF;
        border: 1px solid #F0E5DC;
        border-radius: 12px;
        padding: 9px 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    .index-entry-card:hover {
        border-color: #E07A5F;
        transform: translateX(3px);
        box-shadow: 0 3px 10px rgba(180, 140, 120, 0.12);
    }
    .index-name-text {
        font-weight: 600;
        font-size: 14px;
        color: #38302E;
    }
    .index-meta-text {
        font-size: 12px;
        color: #8C7565;
        background: #FAF3EC;
        padding: 2px 8px;
        border-radius: 10px;
    }

    /* Health & Nutrition Wax Seal */
    .health-stamp-seal {
        display: inline-block;
        border: 2px dashed #E07A5F;
        background: #FFF5F0;
        color: #C85A3F;
        font-family: 'Patrick Hand', cursive;
        font-size: 15px;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 20px;
        margin: 4px 0;
    }

    /* Basket checklist cards */
    .basket-box-have {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    .basket-box-need {
        background-color: #FFF7ED;
        border: 1px solid #FED7AA;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }

    /* Buttons styling */
    .stButton>button {
        border-radius: 18px !important;
        border: 1.5px solid #E07A5F !important;
        color: #E07A5F !important;
        background-color: #FFFFFF !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        padding: 3px 12px !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        border-color: #D95D39 !important;
        color: #FFFFFF !important;
        background-color: #D95D39 !important;
        transform: scale(1.02);
    }
    .stButton>button[kind="primary"] {
        background-color: #D95D39 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #D95D39 !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        color: #D95D39 !important;
        font-size: 19px !important;
    }

    /* Custom tabs for Left Page */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F7EEE5;
        padding: 4px 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Patrick Hand', cursive !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 5px 12px !important;
        color: #6B5B52 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #D95D39 !important;
        box-shadow: 0 2px 6px rgba(180, 140, 120, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Load Recipe Database & Train Models
# -------------------------------------------------------------------
csv_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')] if os.path.exists(DATA_FOLDER) else []
df = load_and_combine_datasets(csv_files)

if df.empty:
    st.error("🚨 Oh no! No recipe pages found in the 'data' shelf. Please add CSV files and reload.")
    st.stop()

tfidf_vectorizer, tfidf_matrix = prepare_tfidf(df['ingredients'])
cuisine_model, cuisine_vectorizer = train_text_classifier(df, 'cuisine')
course_model, course_vectorizer = train_text_classifier(df, 'course')
feedback_summary = load_feedback_summary(FEEDBACK_FILE)

# -------------------------------------------------------------------
# State Management
# -------------------------------------------------------------------
if "active_recipe_index" not in st.session_state:
    st.session_state["active_recipe_index"] = 0

if "pantry_ingredients" not in st.session_state:
    st.session_state["pantry_ingredients"] = ""

def set_pantry_preset(preset: str):
    st.session_state["pantry_ingredients"] = preset

def select_recipe(idx: int):
    st.session_state["active_recipe_index"] = idx

# -------------------------------------------------------------------
# Open Recipe Book Header & Bookmark Ribbon
# -------------------------------------------------------------------
st.markdown("""
<div class="book-ribbon">WANNABECHEF • RECIPE JOURNAL</div>
<div style="text-align: center; margin-bottom: 6px;">
    <h1 style="font-family: 'Playfair Display', serif; font-size: 32px; color: #7A3E2D; margin: 0; display: inline-flex; align-items: center; gap: 8px;">
        <span>📖</span> WannabeChef <span>🍳</span>
    </h1>
    <span style="font-family: 'Patrick Hand', cursive; font-size: 18px; color: #8D6B58; margin-left: 10px;">
        ~ A Cozy Handcrafted Storybook of Recipes & AI Kitchen Intuition ~
    </span>
</div>
""", unsafe_allow_html=True)

# Main Two-Page Spread Columns
col_left_page, col_right_page = st.columns([1, 1.1], gap="small")

# ===================================================================
# LEFT PAGE: Chapters, Index & AI Pantry
# ===================================================================
with col_left_page:
    st.markdown("""
    <div class="page-header-stamp">
        <span>📑 TABLE OF CONTENTS & CHAPTERS</span>
        <span>PAGE I</span>
    </div>
    """, unsafe_allow_html=True)

    left_tab_chapters, left_tab_pantry, left_tab_journal = st.tabs([
        "📖 Chapters Index",
        "🧺 AI Pantry",
        "⭐ Tasting Log"
    ])

    # ---------------------------------------------------------------
    # Tab A: Chapter & Index Browser
    # ---------------------------------------------------------------
    with left_tab_chapters:
        unique_cuisines = sorted([c for c in df["cuisine"].dropna().unique() if c != 'unknown'])
        
        c_sel1, c_sel2 = st.columns([1.3, 1])
        with c_sel1:
            browse_cuisine = st.selectbox("Select Chapter / Cuisine:", options=["All Chapters"] + unique_cuisines, index=0)
        with c_sel2:
            search_title = st.text_input("🔍 Search title:", placeholder="e.g. Tikka, Biryani")

        filtered_chapter_df = df.copy()
        if browse_cuisine != "All Chapters":
            filtered_chapter_df = filtered_chapter_df[filtered_chapter_df['cuisine'] == browse_cuisine]
        if search_title:
            filtered_chapter_df = filtered_chapter_df[filtered_chapter_df['recipename'].str.contains(search_title, case=False, na=False)]

        theme = CUISINE_THEMES.get(browse_cuisine, {"emoji": "🍲", "subtitle": "Curated dishes and homemade specialties"})
        
        st.markdown(f"""
        <div style="font-family: 'Patrick Hand', cursive; font-size: 15px; color: #8D6B58; margin-bottom: 6px;">
            {theme['emoji']} <strong>{browse_cuisine}</strong>: {theme['subtitle']} ({len(filtered_chapter_df):,} dishes)
        </div>
        """, unsafe_allow_html=True)

        if filtered_chapter_df.empty:
            st.warning("No recipes found in this chapter with that title search!")
        else:
            # Pagination
            items_per_page = 7
            total_p = max(1, int(np.ceil(len(filtered_chapter_df) / items_per_page)))
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                cur_page = st.number_input("Index Page:", min_value=1, max_value=total_p, value=1, step=1, key="idx_page_num")
            with p_col2:
                st.caption(f"Showing {(cur_page-1)*items_per_page + 1} - {min(cur_page*items_per_page, len(filtered_chapter_df))} of {len(filtered_chapter_df):,}")

            start_i = (cur_page - 1) * items_per_page
            p_slice = filtered_chapter_df.iloc[start_i:start_i+items_per_page]

            for _, row in p_slice.iterrows():
                r_time = f"⏱️ {int(row['totaltimeinmins'])}m" if pd.notna(row['totaltimeinmins']) and row['totaltimeinmins'] > 0 else "⏱️ Flexible"
                r_course = f"• {row['course'].title()}" if pd.notna(row['course']) and row['course'] != 'unknown' else ""

                entry_col, btn_col = st.columns([3.2, 1])
                with entry_col:
                    st.markdown(f"""
                    <div class="index-entry-card">
                        <span class="index-name-text">✨ {row['recipename']}</span>
                        <span class="index-meta-text">{r_time} {r_course}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with btn_col:
                    st.button("📖 Read", key=f"btn_c_{row.name}", on_click=select_recipe, args=(row.name,), use_container_width=True)

    # ---------------------------------------------------------------
    # Tab B: AI Pantry Finder
    # ---------------------------------------------------------------
    with left_tab_pantry:
        st.markdown("""
        <div style="font-family: 'Patrick Hand', cursive; font-size: 16px; color: #8D6B58; margin-bottom: 6px;">
            🧺 Tell the AI sous-chef what ingredients you have:
        </div>
        """, unsafe_allow_html=True)

        st.caption("✨ Quick pantry ideas:")
        preset_cols = st.columns(4)
        preset_cols[0].button("🍗 Chicken", on_click=set_pantry_preset, args=("chicken, onion, tomato, garlic",), use_container_width=True)
        preset_cols[1].button("🧀 Paneer", on_click=set_pantry_preset, args=("paneer, yogurt, onion, bell pepper",), use_container_width=True)
        preset_cols[2].button("🍝 Pasta", on_click=set_pantry_preset, args=("pasta, olive oil, garlic, chilli, cheese",), use_container_width=True)
        preset_cols[3].button("🥗 Salad", on_click=set_pantry_preset, args=("lettuce, tomato, cucumber, olive oil",), use_container_width=True)

        user_ingredients_input = st.text_input(
            "Pantry ingredients:",
            value=st.session_state.get("pantry_ingredients", ""),
            placeholder="e.g. chicken, tomato, onion, garlic",
            key="pantry_input_field"
        )
        if user_ingredients_input != st.session_state.get("pantry_ingredients", ""):
            st.session_state["pantry_ingredients"] = user_ingredients_input

        # Filters
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            pantry_cuisine = st.selectbox("Filter Cuisine:", ["All Chapters"] + unique_cuisines, key="ai_c_filter")
        with f_col2:
            pantry_time = st.selectbox("Max Time:", ["Any Time", "< 30 mins", "< 60 mins", "< 90 mins"], key="ai_t_filter")

        if user_ingredients_input:
            input_ings = [s.strip().lower() for s in user_ingredients_input.split(",") if s.strip()]
            pred_c = predict_text_label((cuisine_model, cuisine_vectorizer), input_ings)
            pred_co = predict_text_label((course_model, course_vectorizer), input_ings)

            st.markdown(f"""
            <div style="background: #FFFBF5; border: 1.5px dashed #E07A5F; border-radius: 12px; padding: 8px 12px; margin: 8px 0; font-size: 13px;">
                🧑‍🍳 <strong>AI Intuition:</strong> Looks like a <strong>{pred_c.title()} {pred_co.title()}</strong> dish!
            </div>
            """, unsafe_allow_html=True)

            scored = compute_intelligence_scores(
                df,
                tfidf_vectorizer,
                tfidf_matrix,
                input_ings,
                pred_c,
                pred_co,
                pantry_cuisine,
                feedback_summary
            )

            if pantry_cuisine != "All Chapters":
                scored = scored[scored['cuisine'].str.lower() == pantry_cuisine.lower()]
            if pantry_time != "Any Time":
                max_t = int(re.sub(r'\D', '', pantry_time))
                scored = scored[(scored['totaltimeinmins'] > 0) & (scored['totaltimeinmins'] <= max_t)]

            top_ai_results = scored.sort_values(by='intelligence_score', ascending=False).head(8)

            if top_ai_results.empty:
                st.warning("No recipes matched these filters!")
            else:
                st.markdown("##### 🎯 Top AI Recommendations:")
                for _, a_row in top_ai_results.iterrows():
                    match_pct = f"{a_row['fuzzy_score']:.0f}% match"
                    col_ai_info, col_ai_btn = st.columns([3.2, 1])
                    with col_ai_info:
                        st.markdown(f"""
                        <div class="index-entry-card">
                            <span class="index-name-text">✨ {a_row['recipename']}</span>
                            <span class="index-meta-text">{match_pct} • {a_row['cuisine'].title()}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_ai_btn:
                        st.button("📖 Read", key=f"btn_ai_{a_row.name}", on_click=select_recipe, args=(a_row.name,), use_container_width=True)
        else:
            st.info("💡 Type ingredients above or tap one of the quick chips to see AI recipe matches!")

    # ---------------------------------------------------------------
    # Tab C: Chef's Tasting Log
    # ---------------------------------------------------------------
    with left_tab_journal:
        if os.path.exists(FEEDBACK_FILE):
            try:
                fdf = pd.read_csv(FEEDBACK_FILE)
                if not fdf.empty:
                    st.caption(f"⭐ **{len(fdf)} dishes tasted & rated** • Avg score: **{fdf['rating'].mean():.1f} / 5**")
                    for _, j_row in fdf.tail(6).iloc[::-1].iterrows():
                        stars = "⭐" * int(j_row.get('rating', 5))
                        st.markdown(f"""
                        <div class="index-entry-card">
                            <div>
                                <span class="index-name-text">{j_row.get('selected_recipe')}</span>
                                <div style="font-size: 11px; color: #8C7565;">{j_row.get('timestamp', 'Recent')}</div>
                            </div>
                            <span>{stars}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No tastings logged yet! Rate a recipe on the right page.")
            except Exception:
                st.info("Tasting log is ready for your first rating!")
        else:
            st.info("No tastings logged yet! Cook a dish and leave a star rating on the right page.")

# ===================================================================
# RIGHT PAGE: Active Recipe Page (Storybook View)
# ===================================================================
with col_right_page:
    active_idx = st.session_state.get("active_recipe_index", 0)
    if active_idx in df.index:
        active_row = df.loc[active_idx]
    else:
        active_row = df.iloc[0]

    recipename = active_row['recipename']
    cuisine = str(active_row.get('cuisine', 'Specialty')).title()
    course = str(active_row.get('course', 'Dish')).title()
    total_time = active_row.get('totaltimeinmins')
    servings = active_row.get('servings')
    recipe_url = active_row.get('url')
    img_url = active_row.get('image_url')
    rec_ings_str = active_row.get('ingredients', '')
    instructions = active_row.get('instructions')

    rec_ings = [i.strip() for i in str(rec_ings_str).split(",") if i.strip()] if rec_ings_str and pd.notna(rec_ings_str) else []
    nutrition = estimate_nutrition_from_ingredients(rec_ings, servings)
    health_seal = healthiness_label(nutrition)
    theme = CUISINE_THEMES.get(active_row.get('cuisine', ''), {"emoji": "🍲", "subtitle": "A cozy culinary creation"})

    # Right Page Header
    st.markdown(f"""
    <div class="page-header-stamp">
        <span>📖 CHEF'S RECIPE SPREAD</span>
        <span>RECIPE #{active_row.get('srno') or '★'}</span>
    </div>
    """, unsafe_allow_html=True)

    # Washi Tape & Title
    st.markdown(f"""
    <div style="text-align: center;">
        <div class="washi-tape-strip"></div>
        <div style="font-family: 'Patrick Hand', cursive; color: #D95D39; font-size: 15px; font-weight: 700; margin-bottom: 2px;">
            {theme['emoji']} CHAPTER: {cuisine.upper()} • {course.upper()}
        </div>
        <h2 style="font-family: 'Playfair Display', serif; font-size: 26px; color: #38302E; margin: 0 0 6px 0;">{recipename}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Image + Key Info
    col_img, col_metrics = st.columns([1, 1.2])
    with col_img:
        if img_url and pd.notna(img_url) and str(img_url).strip() and str(img_url).startswith("http"):
            st.image(img_url, use_column_width=True)
        else:
            st.image("https://images.unsplash.com/photo-1495521821757-a1efb6729352?auto=format&fit=crop&w=500&q=80", use_column_width=True)

        if recipe_url and pd.notna(recipe_url) and str(recipe_url).startswith("http"):
            st.markdown(f"""
            <div style="text-align: center; margin-top: 4px;">
                <a href="{recipe_url}" target="_blank" style="font-family: 'Patrick Hand', cursive; font-size: 14px; color: #D95D39; text-decoration: none; font-weight: 700;">
                    🔗 Original Recipe Link ↗
                </a>
            </div>
            """, unsafe_allow_html=True)

    with col_metrics:
        t_disp = f"{int(total_time)} min" if pd.notna(total_time) and total_time > 0 else "Flexible"
        s_disp = f"{int(servings)}" if pd.notna(servings) and servings > 0 else "2-4"

        m_c1, m_c2 = st.columns(2)
        m_c1.metric("Cooking Time", t_disp)
        m_c2.metric("Servings", s_disp)

        st.markdown(f'<div class="health-stamp-seal">Seal: {health_seal}</div>', unsafe_allow_html=True)

        st.markdown("##### 🍎 Nutrition (per serving approx):")
        n_c1, n_c2 = st.columns(2)
        n_c1.metric("Calories", f"{nutrition.get('calories', 0):.0f} kcal")
        n_c2.metric("Protein", f"{nutrition.get('protein', 0):.1f} g")
        n_c1.metric("Fats", f"{nutrition.get('fat', 0):.1f} g")
        n_c2.metric("Carbs", f"{nutrition.get('carbs', 0):.1f} g")

    # Basket Checklist if user entered ingredients in Pantry
    pantry_current = st.session_state.get("pantry_ingredients", "").strip()
    if pantry_current and rec_ings_str:
        user_list = [s.strip().lower() for s in pantry_current.split(",") if s.strip()]
        details = get_ingredient_match_details(rec_ings_str, user_list)
        
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            have_txt = ", ".join(details['matched'][:5]) if details['matched'] else "None yet"
            st.markdown(f"""
            <div class="basket-box-have">
                <strong style="color: #166534; font-size: 13px;">🧺 In Basket ({len(details['matched'])}):</strong><br>
                <span style="font-size: 12px; color: #14532D;">{have_txt}</span>
            </div>
            """, unsafe_allow_html=True)
        with b_c2:
            need_txt = ", ".join(details['missing'][:5]) if details['missing'] else "All ready!"
            st.markdown(f"""
            <div class="basket-box-need">
                <strong style="color: #9A3412; font-size: 13px;">🛒 Market List ({len(details['missing'])}):</strong><br>
                <span style="font-size: 12px; color: #7C2D12;">{need_txt}</span>
            </div>
            """, unsafe_allow_html=True)

    # Full Ingredients Accordion
    with st.expander("🧂 Full Ingredients", expanded=False):
        if rec_ings_str and pd.notna(rec_ings_str):
            clean_ings = [i.strip().title() for i in rec_ings_str.split(",") if i.strip()]
            ing_col1, ing_col2 = st.columns(2)
            for idx, item in enumerate(clean_ings):
                col = ing_col1 if idx % 2 == 0 else ing_col2
                col.markdown(f"• {item}")
        else:
            st.write("Ingredient list is tucked away in the chef's secret notes!")

    # Step by Step Instructions
    with st.expander("📋 Step-by-Step Cooking Steps", expanded=True):
        if not instructions or pd.isna(instructions):
            st.info("Instructions are not transcribed in this edition. Check the original source link!")
        else:
            inst_text = str(instructions).strip()
            steps = []
            if "\n" in inst_text and inst_text.count("\n") > 2:
                steps = [s.strip() for s in inst_text.split("\n") if s.strip()]
            else:
                steps = [s.strip() for s in inst_text.split(".") if len(s.strip()) > 5]

            spoon_icons = ["🥄", "🍳", "🔪", "🔥", "🧂", "🍲", "✨", "🥘", "🌿", "🍽️"]
            for i, step in enumerate(steps, start=1):
                icon = spoon_icons[(i - 1) % len(spoon_icons)]
                st.markdown(f"""
                <div style="background: #FFFDF9; border-left: 3px solid #E07A5F; padding: 6px 10px; margin-bottom: 6px; border-radius: 0 8px 8px 0; font-size: 13px;">
                    <strong style="color: #D95D39;">Step {i} {icon}:</strong> {step.rstrip('.')}
                </div>
                """, unsafe_allow_html=True)

    # Rating Form
    with st.form(key=f"taste_form_{active_row.name}"):
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            rating_stars = st.radio("Rate this dish:", [1, 2, 3, 4, 5], index=4, horizontal=True, key=f"star_rad_{active_row.name}")
        with f_col2:
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✍️ Save Note", use_container_width=True)
            if submitted:
                p_save = [s.strip().lower() for s in pantry_current.split(",") if s.strip()] if pantry_current else []
                append_feedback(FEEDBACK_FILE, recipename, p_save, int(rating_stars))
                st.success("Tasting note saved to journal! ⭐")
