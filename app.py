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

@st.cache_data(show_spinner="Opening the Antique Recipe Book...")
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
# Streamlit App Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="WannabeChef • Antique Recipe Book",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Antique Storybook CSS (Aged Parchment, Deckle Edges, Green Ribbons, Wooden Table)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Patrick+Hand&family=Quicksand:wght@500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
    }
    
    /* Warm Rustic Kitchen Wood Table Background */
    .stApp {
        background-color: #B28352;
        background-image: 
            radial-gradient(ellipse at 50% 40%, rgba(255, 235, 200, 0.15) 0%, transparent 70%),
            repeating-linear-gradient(90deg, rgba(70, 40, 20, 0.05) 0px, rgba(70, 40, 20, 0.05) 2px, transparent 2px, transparent 60px),
            linear-gradient(180deg, #9C6F42 0%, #7E542D 100%);
        color: #2F2119;
    }

    /* Keep main container locked on screen - NO SCROLLING */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 1320px !important;
    }

    /* Hide Streamlit header/footer noise for immersion */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Top Book Title Ribbon Header */
    .top-ribbon-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }

    .book-tab-pill {
        background: #FDF4E7;
        color: #6A412A;
        font-family: 'Patrick Hand', cursive;
        font-size: 16px;
        padding: 4px 16px;
        border-radius: 14px 14px 0 0;
        border: 1px solid #C8AC8A;
        border-bottom: none;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.1);
        cursor: pointer;
    }

    /* THE OPEN ANTIQUE BOOK SPREAD */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        background: #F8EFE1 !important;
        /* Stacked book page edges simulation on sides */
        border: 4px solid #75442F !important;
        border-left: 18px solid #C2A57E !important;
        border-right: 18px solid #C2A57E !important;
        outline: 6px solid #4D2619 !important;
        border-radius: 20px !important;
        box-shadow: 
            0 25px 65px rgba(35, 18, 10, 0.6),
            0 8px 25px rgba(0, 0, 0, 0.4) !important;
        padding: 0 !important;
        margin-top: 4px !important;
        position: relative;
    }

    /* Left Aged Parchment Page */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        background: #FBF4EA !important;
        background-image: 
            radial-gradient(circle at 10% 20%, #FFFDF7 0%, transparent 60%),
            linear-gradient(to right, #FBF4EA 92%, #ECDDC7 100%) !important;
        border-right: 2px solid #D6C2A5 !important;
        border-radius: 14px 0 0 14px !important;
        padding: 16px 22px !important;
        height: 690px !important;
        overflow: hidden !important;
        box-shadow: inset -16px 0 22px -10px rgba(80, 50, 30, 0.16) !important;
    }

    /* Right Aged Parchment Page */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
        background: #FBF4EA !important;
        background-image: 
            radial-gradient(circle at 90% 20%, #FFFDF7 0%, transparent 60%),
            linear-gradient(to left, #FBF4EA 92%, #ECDDC7 100%) !important;
        border-radius: 0 14px 14px 0 !important;
        padding: 16px 22px !important;
        height: 690px !important;
        overflow: hidden !important;
        box-shadow: inset 16px 0 22px -10px rgba(80, 50, 30, 0.16) !important;
    }

    /* Page Headings and Vintage Typography */
    .antique-page-header {
        font-family: 'Patrick Hand', cursive;
        font-size: 15px;
        color: #8C6A53;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px dashed #DCC7AD;
        padding-bottom: 4px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }

    .antique-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: #38241B;
        line-height: 1.2;
    }

    /* Cute Washi Tape Strip */
    .washi-tape-strip {
        width: 110px;
        height: 20px;
        background: rgba(255, 182, 193, 0.7);
        margin: -6px auto 8px auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        transform: rotate(-1.5deg);
        border-left: 3px dashed rgba(255,255,255,0.7);
        border-right: 3px dashed rgba(255,255,255,0.7);
    }

    /* Index Card Rows */
    .index-entry-row {
        background: #FFFFFF;
        border: 1px solid #EAD8C3;
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    .index-entry-row:hover {
        border-color: #D95D39;
        background: #FFFDF9;
        transform: translateX(2px);
    }

    /* Health & Nutrition Stamp */
    .health-stamp-seal {
        display: inline-block;
        border: 1.5px dashed #D95D39;
        background: #FFF5EE;
        color: #B54728;
        font-family: 'Patrick Hand', cursive;
        font-size: 14px;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 14px;
    }

    /* Step Cards inside Right Page */
    .step-card {
        background: #FFFDF9;
        border-left: 3px solid #D95D39;
        padding: 6px 10px;
        margin-bottom: 6px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        line-height: 1.35;
        color: #38261D;
        border-top: 1px solid #F5EAE0;
        border-bottom: 1px solid #F5EAE0;
        border-right: 1px solid #F5EAE0;
    }

    /* Page Navigation Flipper Bar */
    .page-flipper-bar {
        position: absolute;
        bottom: 8px;
        left: 20px;
        right: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px dashed #D9C3A8;
        padding-top: 6px;
        font-family: 'Patrick Hand', cursive;
        font-size: 15px;
        color: #7A5B45;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 16px !important;
        border: 1.5px solid #D95D39 !important;
        color: #D95D39 !important;
        background-color: #FFFFFF !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        padding: 2px 10px !important;
        font-size: 12px !important;
    }
    .stButton>button:hover {
        border-color: #B54728 !important;
        color: #FFFFFF !important;
        background-color: #D95D39 !important;
    }

    /* Metric numbers */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        color: #D95D39 !important;
        font-size: 17px !important;
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
# Book State Management (Page Flipping)
# -------------------------------------------------------------------
if "book_mode" not in st.session_state:
    st.session_state["book_mode"] = "recipe" # 'recipe' or 'index' or 'pantry'

if "active_recipe_index" not in st.session_state:
    st.session_state["active_recipe_index"] = 0

if "step_page" not in st.session_state:
    st.session_state["step_page"] = 0

if "chapter_page" not in st.session_state:
    st.session_state["chapter_page"] = 0

if "selected_chapter" not in st.session_state:
    st.session_state["selected_chapter"] = "North Indian Recipes"

if "pantry_ingredients" not in st.session_state:
    st.session_state["pantry_ingredients"] = ""

def set_book_mode(mode: str):
    st.session_state["book_mode"] = mode
    st.session_state["step_page"] = 0

def select_recipe(idx: int):
    st.session_state["active_recipe_index"] = idx
    st.session_state["book_mode"] = "recipe"
    st.session_state["step_page"] = 0

def prev_recipe():
    cur = st.session_state["active_recipe_index"]
    st.session_state["active_recipe_index"] = max(0, cur - 1)
    st.session_state["step_page"] = 0

def next_recipe():
    cur = st.session_state["active_recipe_index"]
    st.session_state["active_recipe_index"] = min(len(df) - 1, cur + 1)
    st.session_state["step_page"] = 0

def prev_steps():
    st.session_state["step_page"] = max(0, st.session_state["step_page"] - 1)

def next_steps(max_p):
    st.session_state["step_page"] = min(max_p, st.session_state["step_page"] + 1)

def prev_chapter_page():
    st.session_state["chapter_page"] = max(0, st.session_state["chapter_page"] - 1)

def next_chapter_page(max_p):
    st.session_state["chapter_page"] = min(max_p, st.session_state["chapter_page"] + 1)

def set_pantry_preset(preset: str):
    st.session_state["pantry_ingredients"] = preset

# -------------------------------------------------------------------
# Top Ribbon Bookmark Tabs (Quick Page Flipper)
# -------------------------------------------------------------------
col_t1, col_t2, col_t3 = st.columns([1, 1.8, 1])
with col_t2:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; gap: 8px;">
        <span style="font-size: 24px;">🎀</span>
        <span style="font-family: 'Playfair Display', serif; font-size: 26px; color: #FFF2DF; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.5);">
            WannabeChef • Antique Recipe Book
        </span>
        <span style="font-size: 24px;">🎀</span>
    </div>
    """, unsafe_allow_html=True)
    
    b_tabs = st.columns(3)
    with b_tabs[0]:
        st.button("📑 Table of Contents", on_click=set_book_mode, args=("index",), use_container_width=True)
    with b_tabs[1]:
        st.button("📖 Current Recipe", on_click=set_book_mode, args=("recipe",), use_container_width=True)
    with b_tabs[2]:
        st.button("🪄 AI Pantry Finder", on_click=set_book_mode, args=("pantry",), use_container_width=True)

# -------------------------------------------------------------------
# Active Recipe Data
# -------------------------------------------------------------------
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

# ===================================================================
# THE TWO-PAGE OPEN BOOK SPREAD
# ===================================================================
page_col_left, page_col_right = st.columns([1, 1], gap="small")

current_mode = st.session_state.get("book_mode", "recipe")

# -------------------------------------------------------------------
# VIEW 1: TABLE OF CONTENTS (INDEX & CHAPTERS)
# -------------------------------------------------------------------
if current_mode == "index":
    with page_col_left:
        st.markdown("""
        <div class="antique-page-header">
            <span>📑 CHAPTER DIRECTORY</span>
            <span>INDEX • LEFT</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 class='antique-title' style='font-size:22px; margin-bottom:4px;'>Chapters & Cuisines</h3>", unsafe_allow_html=True)
        unique_cuisines = sorted([c for c in df["cuisine"].dropna().unique() if c != 'unknown'])
        
        sel_c = st.selectbox("Turn to Chapter:", options=unique_cuisines, index=unique_cuisines.index(st.session_state["selected_chapter"]) if st.session_state["selected_chapter"] in unique_cuisines else 0)
        st.session_state["selected_chapter"] = sel_c

        c_theme = CUISINE_THEMES.get(sel_c, {"emoji": "🍲", "subtitle": "Traditional family specialties"})
        chapter_recipes = df[df['cuisine'] == sel_c]

        st.markdown(f"""
        <div style="background:#FFF9F0; border:1px dashed #D6C2A5; border-radius:10px; padding:10px 14px; margin:10px 0;">
            <div style="font-size:24px; margin-bottom:2px;">{c_theme['emoji']}</div>
            <strong style="color:#7A4232; font-size:16px;">{sel_c}</strong>
            <p style="font-family:'Patrick Hand', cursive; font-size:14px; color:#8C6A53; margin:2px 0 0 0;">
                {c_theme['subtitle']} • <strong>{len(chapter_recipes):,} recipes</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        search_term = st.text_input("🔍 Search recipe in this chapter:", placeholder="e.g. Paneer, Chicken, Soup")
        if search_term:
            chapter_recipes = chapter_recipes[chapter_recipes['recipename'].str.contains(search_term, case=False, na=False)]

        st.caption("👈 Select any recipe on the right page to turn directly to its recipe card!")

    with page_col_right:
        st.markdown("""
        <div class="antique-page-header">
            <span>📖 RECIPES IN CHAPTER</span>
            <span>INDEX • RIGHT</span>
        </div>
        """, unsafe_allow_html=True)

        if chapter_recipes.empty:
            st.warning("No recipes found matching this search in this chapter.")
        else:
            per_page = 5
            total_pages = max(1, int(np.ceil(len(chapter_recipes) / per_page)))
            cur_p = min(st.session_state.get("chapter_page", 0), total_pages - 1)
            
            p_slice = chapter_recipes.iloc[cur_p * per_page : (cur_p + 1) * per_page]

            st.markdown(f"<div style='font-family:Playfair Display, serif; font-size:16px; margin-bottom:8px; color:#5A3626;'>Chapter Index (Page {cur_p + 1} of {total_pages})</div>", unsafe_allow_html=True)

            for _, r in p_slice.iterrows():
                t_str = f"⏱️ {int(r['totaltimeinmins'])}m" if pd.notna(r['totaltimeinmins']) and r['totaltimeinmins'] > 0 else "⏱️ Flexible"
                c_row1, c_row2 = st.columns([3.2, 1])
                with c_row1:
                    st.markdown(f"""
                    <div class="index-entry-row">
                        <span style="font-weight:600; font-size:13px; color:#38241B;">✨ {r['recipename'][:35]}</span>
                        <span style="font-size:11px; color:#8C6A53; background:#F5EDE3; padding:2px 6px; border-radius:8px;">{t_str}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_row2:
                    st.button("📖 Read", key=f"toc_read_{r.name}", on_click=select_recipe, args=(r.name,), use_container_width=True)

            st.markdown("---")
            # Flipper controls
            f1, f2, f3 = st.columns([1, 1.5, 1])
            with f1:
                st.button("◀ Prev Index", on_click=prev_chapter_page, disabled=(cur_p == 0), use_container_width=True)
            with f2:
                st.caption(f"<div style='text-align:center;'>Showing {cur_p*per_page + 1} - {min((cur_p+1)*per_page, len(chapter_recipes))}</div>", unsafe_allow_html=True)
            with f3:
                st.button("Next Index ▶", on_click=next_chapter_page, args=(total_pages - 1,), disabled=(cur_p >= total_pages - 1), use_container_width=True)

# -------------------------------------------------------------------
# VIEW 2: AI PANTRY FINDER
# -------------------------------------------------------------------
elif current_mode == "pantry":
    with page_col_left:
        st.markdown("""
        <div class="antique-page-header">
            <span>🧺 CHEF'S PANTRY COUNTER</span>
            <span>PANTRY • LEFT</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 class='antique-title' style='font-size:20px; margin-bottom:4px;'>What ingredients do you have?</h3>", unsafe_allow_html=True)
        
        st.caption("✨ Quick pantry inspirations:")
        p_row1 = st.columns(2)
        p_row1[0].button("🍗 Chicken Curry", on_click=set_pantry_preset, args=("chicken, onion, tomato, garlic",), use_container_width=True)
        p_row1[1].button("🧀 Paneer Tikka", on_click=set_pantry_preset, args=("paneer, yogurt, onion, bell pepper",), use_container_width=True)
        p_row2 = st.columns(2)
        p_row2[0].button("🍝 Garlic Pasta", on_click=set_pantry_preset, args=("pasta, olive oil, garlic, chilli",), use_container_width=True)
        p_row2[1].button("🥗 Fresh Salad", on_click=set_pantry_preset, args=("lettuce, tomato, cucumber, olive oil",), use_container_width=True)

        user_ingredients_input = st.text_input(
            "Enter ingredients (comma-separated):",
            value=st.session_state.get("pantry_ingredients", ""),
            placeholder="e.g. chicken, tomato, garlic, onion",
            key="pantry_text_field"
        )
        if user_ingredients_input != st.session_state.get("pantry_ingredients", ""):
            st.session_state["pantry_ingredients"] = user_ingredients_input

        unique_cuisines = sorted([c for c in df["cuisine"].dropna().unique() if c != 'unknown'])
        pf_col1, pf_col2 = st.columns(2)
        with pf_col1:
            p_c_filter = st.selectbox("Cuisine:", ["All Chapters"] + unique_cuisines, key="ai_c_box")
        with pf_col2:
            p_t_filter = st.selectbox("Max Time:", ["Any Time", "< 30 mins", "< 60 mins"], key="ai_t_box")

    with page_col_right:
        st.markdown("""
        <div class="antique-page-header">
            <span>🎯 MATCHED CREATIONS</span>
            <span>PANTRY • RIGHT</span>
        </div>
        """, unsafe_allow_html=True)

        if user_ingredients_input:
            input_ings = [s.strip().lower() for s in user_ingredients_input.split(",") if s.strip()]
            pred_c = predict_text_label((cuisine_model, cuisine_vectorizer), input_ings)
            pred_co = predict_text_label((course_model, course_vectorizer), input_ings)

            st.markdown(f"""
            <div style="background:#FFF9F0; border:1px dashed #D95D39; border-radius:8px; padding:6px 10px; font-size:12px; margin-bottom:8px;">
                🧑‍🍳 <strong>AI Intuition:</strong> Authentic <strong>{pred_c.title()} {pred_co.title()}</strong> style!
            </div>
            """, unsafe_allow_html=True)

            scored = compute_intelligence_scores(
                df, tfidf_vectorizer, tfidf_matrix, input_ings, pred_c, pred_co, p_c_filter, feedback_summary
            )
            if p_c_filter != "All Chapters":
                scored = scored[scored['cuisine'].str.lower() == p_c_filter.lower()]
            if p_t_filter != "Any Time":
                max_t = int(re.sub(r'\D', '', p_t_filter))
                scored = scored[(scored['totaltimeinmins'] > 0) & (scored['totaltimeinmins'] <= max_t)]

            top_pantry = scored.sort_values(by='intelligence_score', ascending=False).head(5)

            if top_pantry.empty:
                st.warning("No recipes match these exact filters.")
            else:
                for _, pr in top_pantry.iterrows():
                    match_badge = f"{pr['fuzzy_score']:.0f}% match"
                    col_p1, col_p2 = st.columns([3.2, 1])
                    with col_p1:
                        st.markdown(f"""
                        <div class="index-entry-row">
                            <span style="font-weight:600; font-size:13px; color:#38241B;">✨ {pr['recipename'][:35]}</span>
                            <span style="font-size:11px; color:#8C6A53; background:#F5EDE3; padding:2px 6px; border-radius:8px;">{match_badge}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_p2:
                        st.button("📖 Read", key=f"pan_read_{pr.name}", on_click=select_recipe, args=(pr.name,), use_container_width=True)
        else:
            st.info("💡 Type ingredients on the left page or tap a quick idea to see recommendations here!")

# -------------------------------------------------------------------
# VIEW 3: ACTIVE RECIPE (TWO-PAGE SPREAD - NO SCROLLING, STEP FLIPPING)
# -------------------------------------------------------------------
else:
    # LEFT PAGE: Recipe Story, Photo & Ingredients
    with page_col_left:
        st.markdown(f"""
        <div class="antique-page-header">
            <span>📖 RECIPE OVERVIEW</span>
            <span>CHAPTER {cuisine.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='washi-tape-strip'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:8px;">
            <h2 class="antique-title" style="font-size:22px; margin:0 0 2px 0;">{recipename}</h2>
            <div style="font-family:'Patrick Hand', cursive; font-size:14px; color:#8C6A53;">
                {theme['emoji']} {cuisine} • {course}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_img, c_metrics = st.columns([1, 1.2])
        with c_img:
            if img_url and pd.notna(img_url) and str(img_url).strip() and str(img_url).startswith("http"):
                st.image(img_url, use_column_width=True)
            else:
                st.image("https://images.unsplash.com/photo-1495521821757-a1efb6729352?auto=format&fit=crop&w=400&q=80", use_column_width=True)

        with c_metrics:
            t_disp = f"{int(total_time)}m" if pd.notna(total_time) and total_time > 0 else "Flexible"
            s_disp = f"{int(servings)}" if pd.notna(servings) and servings > 0 else "2-4"

            m1, m2 = st.columns(2)
            m1.metric("Time", t_disp)
            m2.metric("Serves", s_disp)
            st.markdown(f'<div class="health-stamp-seal">Seal: {health_seal}</div>', unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:11px; color:#8C6A53;'>🍎 Calories: <strong>{nutrition.get('calories', 0):.0f} kcal</strong> • Protein: <strong>{nutrition.get('protein', 0):.1f}g</strong></span>", unsafe_allow_html=True)

        # Ingredients (Compact 2-column list)
        st.markdown("<div style='font-family:Patrick Hand, cursive; font-size:16px; font-weight:700; color:#5A3626; margin:8px 0 4px 0;'>🧂 Key Ingredients:</div>", unsafe_allow_html=True)
        if rec_ings:
            ing_cols = st.columns(2)
            for idx, ing_item in enumerate(rec_ings[:8]):
                target_col = ing_cols[idx % 2]
                target_col.markdown(f"<span style='font-size:12px; color:#38241B;'>• {ing_item.title()[:24]}</span>", unsafe_allow_html=True)
            if len(rec_ings) > 8:
                st.caption(f"...and {len(rec_ings) - 8} more pantry spices.")
        else:
            st.caption("Pantry spices listed in notes.")

        # Left Page Bottom Bar: Flip between recipes
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        b_prev, b_toc = st.columns([1, 1])
        b_prev.button("◀ Prev Recipe", on_click=prev_recipe, disabled=(active_idx == 0), use_container_width=True)
        b_toc.button("📑 Flip to Index", on_click=set_book_mode, args=("index",), use_container_width=True)

    # RIGHT PAGE: Cooking Steps (Paginated!) & Tasting Notes
    with page_col_right:
        st.markdown(f"""
        <div class="antique-page-header">
            <span>📋 METHOD & TASTING</span>
            <span>RECIPE #{active_row.get('srno') or '★'}</span>
        </div>
        """, unsafe_allow_html=True)

        # Parse instructions into clean steps
        steps = []
        if instructions and pd.notna(instructions):
            inst_text = str(instructions).strip()
            if "\n" in inst_text and inst_text.count("\n") > 2:
                steps = [s.strip() for s in inst_text.split("\n") if s.strip()]
            else:
                steps = [s.strip() for s in inst_text.split(".") if len(s.strip()) > 5]

        if not steps:
            steps = ["Follow preparation instructions as specified by the chef in the original source link."]

        # Paginate steps so there is ZERO SCROLLING!
        steps_per_page = 4
        total_step_pages = max(1, int(np.ceil(len(steps) / steps_per_page)))
        cur_step_page = min(st.session_state.get("step_page", 0), total_step_pages - 1)

        start_step = cur_step_page * steps_per_page
        end_step = min(start_step + steps_per_page, len(steps))
        page_steps = steps[start_step:end_step]

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-family:'Playfair Display', serif; font-size:16px; color:#5A3626; font-weight:700;">
                Cooking Steps ({cur_step_page + 1}/{total_step_pages})
            </span>
            <span style="font-size:12px; color:#8C6A53; font-family:'Patrick Hand', cursive;">
                Steps {start_step + 1} to {end_step} of {len(steps)}
            </span>
        </div>
        """, unsafe_allow_html=True)

        spoon_icons = ["🥄", "🍳", "🔪", "🔥", "🧂", "🍲", "✨", "🥘", "🌿", "🍽️"]
        for i, step_text in enumerate(page_steps, start=start_step + 1):
            s_icon = spoon_icons[(i - 1) % len(spoon_icons)]
            st.markdown(f"""
            <div class="step-card">
                <strong style="color:#D95D39;">Step {i} {s_icon}:</strong> {step_text.rstrip('.')}
            </div>
            """, unsafe_allow_html=True)

        # Step Flipper buttons
        if total_step_pages > 1:
            s_btn1, s_btn2 = st.columns([1, 1])
            s_btn1.button("◀ Earlier Steps", on_click=prev_steps, disabled=(cur_step_page == 0), use_container_width=True)
            s_btn2.button("Later Steps ▶", on_click=next_steps, args=(total_step_pages - 1,), disabled=(cur_step_page >= total_step_pages - 1), use_container_width=True)

        # Tasting Rating
        st.markdown("<div style='font-family:Patrick Hand, cursive; font-size:14px; font-weight:700; color:#5A3626; margin-top:6px;'>⭐ Rate this Recipe:</div>", unsafe_allow_html=True)
        with st.form(key=f"star_review_form_{active_row.name}"):
            r_c1, r_c2 = st.columns([2, 1])
            with r_c1:
                r_val = st.radio("Rating:", [1, 2, 3, 4, 5], index=4, horizontal=True, label_visibility="collapsed", key=f"r_star_{active_row.name}")
            with r_c2:
                sub = st.form_submit_button("✍️ Save Note", use_container_width=True)
                if sub:
                    p_current = st.session_state.get("pantry_ingredients", "")
                    p_list = [s.strip().lower() for s in p_current.split(",") if s.strip()] if p_current else []
                    append_feedback(FEEDBACK_FILE, recipename, p_list, int(r_val))
                    st.success("Saved! ⭐")

        # Right Page Bottom Bar: Next Recipe
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        if recipe_url and pd.notna(recipe_url) and str(recipe_url).startswith("http"):
            st.markdown(f"<div style='text-align:center; font-size:12px;'><a href='{recipe_url}' target='_blank' style='color:#D95D39; text-decoration:none;'>🔗 View Source Recipe ↗</a></div>", unsafe_allow_html=True)
        st.button("Next Recipe ▶", on_click=next_recipe, disabled=(active_idx >= len(df) - 1), use_container_width=True)
