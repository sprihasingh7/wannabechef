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
                
                # Handle common column variations
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
    
    # Normalize ingredients string
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
    semantic_weight: float,
    fuzzy_weight: float
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
        if user_cuisine_filter and user_cuisine_filter.lower() != "all" and user_cuisine_filter.lower() in c:
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
# Streamlit App Setup & Cute Storybook Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="WannabeChef • The Cozy Recipe Book",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cute Recipe Book CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Patrick+Hand&family=Quicksand:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    /* Global cozy typography & background */
    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
    }
    
    .stApp {
        background-color: #FAF6F0;
        background-image: radial-gradient(#E8DFD8 0.75px, transparent 0.75px);
        background-size: 18px 18px;
    }

    /* Cozy Recipe Book Cover Header */
    .book-cover-header {
        background: linear-gradient(135deg, #FFFDF9 0%, #FFF5ED 100%);
        border: 2px solid #EEDCD0;
        border-radius: 20px;
        padding: 30px 20px 20px 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(184, 137, 110, 0.12);
        position: relative;
        margin-bottom: 25px;
    }
    
    .ribbon-bookmark {
        position: absolute;
        top: -6px;
        right: 45px;
        background: #FF6B6B;
        color: white;
        padding: 8px 14px 14px 14px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        border-radius: 0 0 6px 6px;
        box-shadow: 0 4px 10px rgba(255, 107, 107, 0.35);
        clip-path: polygon(0 0, 100% 0, 100% 85%, 50% 100%, 0 85%);
    }

    .book-badge {
        display: inline-block;
        background: #FFE8D6;
        color: #B5651D;
        font-family: 'Patrick Hand', cursive;
        font-size: 17px;
        font-weight: 600;
        padding: 4px 16px;
        border-radius: 20px;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 46px;
        font-weight: 700;
        color: #D95D39;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .book-subtitle {
        font-family: 'Patrick Hand', cursive;
        font-size: 22px;
        color: #6C584C;
        margin-top: 6px;
        margin-bottom: 12px;
    }

    .book-flourish {
        color: #DDA15E;
        font-size: 20px;
        letter-spacing: 4px;
    }

    /* Washi Tape Strip */
    .washi-tape {
        width: 140px;
        height: 26px;
        background: rgba(255, 182, 193, 0.75);
        margin: -15px auto 12px auto;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transform: rotate(-1.5deg);
        border-left: 4px dashed rgba(255,255,255,0.7);
        border-right: 4px dashed rgba(255,255,255,0.7);
    }
    
    .washi-tape-mint {
        background: rgba(168, 218, 181, 0.75);
        transform: rotate(1.2deg);
    }

    /* Storybook Recipe Card */
    .recipe-book-page {
        background: #FFFFFF;
        border: 2px solid #EFE5DD;
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 8px 24px rgba(160, 130, 110, 0.08);
        position: relative;
        margin-bottom: 25px;
    }

    .chapter-tag {
        display: inline-block;
        background: #FFF1E6;
        color: #D95D39;
        font-family: 'Patrick Hand', cursive;
        font-size: 16px;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 12px;
        border: 1px dashed #F4A261;
        margin-bottom: 8px;
    }

    .recipe-title-storybook {
        font-family: 'Playfair Display', serif;
        font-size: 32px;
        font-weight: 700;
        color: #38302E;
        margin-bottom: 6px;
    }

    /* Checklist Cards */
    .checklist-card-have {
        background-color: #F0FDF4;
        border: 1.5px solid #BBF7D0;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .checklist-card-need {
        background-color: #FFF7ED;
        border: 1.5px solid #FED7AA;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .checklist-title {
        font-family: 'Patrick Hand', cursive;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Health & Nutrition Wax Seal */
    .health-stamp {
        display: inline-block;
        border: 2px dashed #E07A5F;
        background: #FFF5F2;
        color: #C85A3F;
        font-family: 'Patrick Hand', cursive;
        font-size: 18px;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 24px;
        margin-top: 4px;
        margin-bottom: 14px;
    }

    /* Chapter Index Styling */
    .chapter-hero {
        background: #FFFBF5;
        border: 2px dashed #DDBEA9;
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .chapter-hero-emoji {
        font-size: 46px;
        line-height: 1;
    }

    .chapter-hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 26px;
        font-weight: 700;
        color: #4A3E3D;
        margin: 0;
    }

    .chapter-hero-desc {
        font-family: 'Patrick Hand', cursive;
        font-size: 18px;
        color: #8C6D62;
        margin: 4px 0 0 0;
    }

    .index-entry {
        background: #FFFFFF;
        border: 1px solid #F0E6DF;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }

    .index-entry:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(180, 140, 120, 0.12);
        border-color: #E29578;
    }

    .index-entry-name {
        font-weight: 600;
        font-size: 17px;
        color: #38302E;
    }

    .index-entry-meta {
        font-size: 13px;
        color: #8D7B68;
        background: #FAF5F0;
        padding: 4px 10px;
        border-radius: 12px;
    }

    /* Buttons styling */
    .stButton>button {
        border-radius: 20px !important;
        border: 2px solid #E07A5F !important;
        color: #E07A5F !important;
        background-color: #FFFFFF !important;
        font-family: 'Quicksand', sans-serif !important;
        font-weight: 600 !important;
        padding: 6px 18px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        border-color: #D95D39 !important;
        color: #FFFFFF !important;
        background-color: #D95D39 !important;
        transform: scale(1.02);
    }

    /* Primary button */
    .stButton>button[kind="primary"] {
        background-color: #D95D39 !important;
        color: #FFFFFF !important;
        border: 2px solid #D95D39 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFDF9;
        border-right: 2px dashed #EADBCE;
    }

    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        color: #D95D39 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #FAF1E8;
        padding: 6px 10px;
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Patrick Hand', cursive !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 8px 18px !important;
        color: #6B5B52 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #D95D39 !important;
        box-shadow: 0 2px 8px rgba(180, 140, 120, 0.15) !important;
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
# Header: The Cozy Recipe Book Cover
# -------------------------------------------------------------------
st.markdown("""
<div class="book-cover-header">
    <div class="ribbon-bookmark">RECIPE BOOK</div>
    <div class="book-badge">✨ CHEF'S HANDCRAFTED JOURNAL ✨</div>
    <h1 class="book-title">🍳 WannabeChef</h1>
    <p class="book-subtitle">A Cozy Storybook of Culinary Delights, Spiced by AI Magic</p>
    <div class="book-flourish">❧ ✦ ❧</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar: The Kitchen Pantry & Spice Rack
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 12px;">
        <span style="font-size: 38px;">🧺</span>
        <h2 style="font-family: 'Playfair Display', serif; margin: 4px 0 0 0; color: #D95D39;">Kitchen Pantry</h2>
        <p style="font-family: 'Patrick Hand', cursive; font-size: 16px; color: #8D7B68; margin: 0;">Stock your ingredients & spice filters</p>
    </div>
    """, unsafe_allow_html=True)

    # Ingredients Input
    pantry_input = st.text_input(
        "🧂 Ingredients on hand (comma separated):",
        placeholder="e.g. chicken, tomato, onion, garlic",
        key="pantry_ingredients_input"
    )

    # Quick recipe ingredient ideas
    st.caption("✨ Quick pantry ideas:")
    chip_cols = st.columns(2)
    with chip_cols[0]:
        if st.button("🍗 Chicken Curry", use_container_width=True):
            pantry_input = "chicken, onion, tomato, garlic, ginger"
            st.session_state["pantry_ingredients_input"] = pantry_input
            st.rerun()
        if st.button("🍝 Garlic Pasta", use_container_width=True):
            pantry_input = "pasta, olive oil, garlic, chilli, cheese"
            st.session_state["pantry_ingredients_input"] = pantry_input
            st.rerun()
    with chip_cols[1]:
        if st.button("🧀 Paneer Tikka", use_container_width=True):
            pantry_input = "paneer, yogurt, onion, bell pepper, turmeric"
            st.session_state["pantry_ingredients_input"] = pantry_input
            st.rerun()
        if st.button("🥗 Fresh Salad", use_container_width=True):
            pantry_input = "lettuce, tomato, cucumber, carrot, olive oil"
            st.session_state["pantry_ingredients_input"] = pantry_input
            st.rerun()

    st.markdown("---")
    st.markdown("### 🏷️ Filter Chapters")
    
    unique_cuisines = sorted([c for c in df["cuisine"].dropna().unique() if c != 'unknown'])
    sidebar_cuisine_filter = st.selectbox("Cuisine Chapter:", options=["All Cuisines"] + unique_cuisines)
    
    time_options = ["Any Time", "< 30 mins", "< 60 mins", "< 90 mins"]
    sidebar_time_filter = st.selectbox("Cooking Time:", options=time_options)

    # Advanced Scoring Weights Expander
    with st.expander("🎛️ Secret Recipe Formula"):
        st.caption("Fine-tune how your AI sous-chef ranks recipes:")
        semantic_weight = st.slider("Theme / Semantic Weight", 0.0, 1.0, 0.55,
                                    help="Values recipes matching the overarching dish style.")
        fuzzy_weight = st.slider("Pantry Match Weight", 0.0, 1.0, 0.35,
                                 help="Values recipes containing exact ingredients from your pantry.")
        min_ingredient_match = st.slider("Minimum Pantry Match (%)", 0, 100, 10,
                                         help="Filter out recipes that match less than this percentage.")

    st.markdown("---")
    st.markdown("### 📖 Book Library Stats")
    c_s1, c_s2 = st.columns(2)
    c_s1.metric("Total Recipes", f"{len(df):,}")
    c_s2.metric("Chapters", f"{len(unique_cuisines)}")
    st.caption(f"⭐ **Chef's Tastings Logged:** `{len(feedback_summary)}` rated dishes")

# -------------------------------------------------------------------
# Helper: Storybook Recipe Page Renderer
# -------------------------------------------------------------------
def render_recipe_storybook(selected_row: pd.Series, input_ingredients: List[str] = None):
    recipename = selected_row['recipename']
    cuisine = str(selected_row.get('cuisine', 'Specialty')).title()
    course = str(selected_row.get('course', 'Dish')).title()
    total_time = selected_row.get('totaltimeinmins')
    servings = selected_row.get('servings')
    recipe_url = selected_row.get('url')
    img_url = selected_row.get('image_url')
    rec_ings_str = selected_row.get('ingredients', '')
    instructions = selected_row.get('instructions')

    rec_ings = [i.strip() for i in str(rec_ings_str).split(",") if i.strip()] if rec_ings_str and pd.notna(rec_ings_str) else []
    nutrition = estimate_nutrition_from_ingredients(rec_ings, servings)
    health_seal = healthiness_label(nutrition)

    theme = CUISINE_THEMES.get(selected_row.get('cuisine', ''), {"emoji": "🍲", "subtitle": "A cozy culinary creation"})

    # Render Card
    st.markdown(f"""
    <div class="recipe-book-page">
        <div class="washi-tape"></div>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div class="chapter-tag">{theme['emoji']} Chapter: {cuisine} • {course}</div>
            <div style="font-family: 'Patrick Hand', cursive; font-size: 15px; color: #A08272;">Recipe Page # {selected_row.get('srno') or '★'}</div>
        </div>
        <h2 class="recipe-title-storybook">{recipename}</h2>
        <div style="font-family: 'Patrick Hand', cursive; color: #8D7B68; font-size: 18px; margin-bottom: 18px;">
            {theme['subtitle']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Columns: Visuals & Metrics
    col_photo, col_info = st.columns([1.1, 1])

    with col_photo:
        if img_url and pd.notna(img_url) and str(img_url).strip() and str(img_url).startswith("http"):
            st.image(img_url, use_column_width=True, caption=f"🍽️ {recipename} ({cuisine})")
        else:
            # Cute fallback recipe illustration card
            st.image("https://images.unsplash.com/photo-1495521821757-a1efb6729352?auto=format&fit=crop&w=700&q=80",
                     use_column_width=True, caption=f"🍳 Freshly baked {recipename}")

        if recipe_url and pd.notna(recipe_url) and str(recipe_url).startswith("http"):
            st.markdown(f"""
            <div style="text-align: center; margin-top: 8px;">
                <a href="{recipe_url}" target="_blank" style="font-family: 'Patrick Hand', cursive; font-size: 17px; color: #D95D39; text-decoration: none; font-weight: 700;">
                    🔗 Open Original Kitchen Source Recipe ↗
                </a>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown("### ⏱️ Kitchen Timers & Servings")
        m1, m2, m3 = st.columns(3)
        time_display = f"{int(total_time)} min" if pd.notna(total_time) and total_time > 0 else "Flexible"
        servings_display = f"{int(servings)}" if pd.notna(servings) and servings > 0 else "2-4"
        
        m1.metric("Total Time", time_display)
        m2.metric("Yields", f"{servings_display} servings")
        if 'intelligence_score' in selected_row:
            m3.metric("Pantry Match", f"{selected_row['fuzzy_score']:.0f}%")
        else:
            m3.metric("Style", course)

        st.markdown(f'<div class="health-stamp">Seal: {health_seal}</div>', unsafe_allow_html=True)

        st.markdown("### 🍎 Nutrition Estimate <span style='font-size:13px; font-weight:400; color:#8C6D62;'>(per serving)</span>", unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        n1.metric("Calories", f"{nutrition.get('calories', 0):.0f} kcal")
        n2.metric("Protein", f"{nutrition.get('protein', 0):.1f} g")
        n1.metric("Good Fats", f"{nutrition.get('fat', 0):.1f} g")
        n2.metric("Carbs", f"{nutrition.get('carbs', 0):.1f} g")

    st.markdown("---")

    # If user searched with ingredients, show cute matched vs missing checklist
    if input_ingredients and rec_ings_str:
        match_details = get_ingredient_match_details(rec_ings_str, input_ingredients)
        c_have, c_need = st.columns(2)
        with c_have:
            have_items = match_details['matched']
            items_str = "<br>".join([f"✨ {item}" for item in have_items]) if have_items else "<i>None from your pantry list yet!</i>"
            st.markdown(f"""
            <div class="checklist-card-have">
                <div class="checklist-title" style="color: #166534;">🧺 In Your Basket ({len(have_items)})</div>
                <div style="font-size: 15px; color: #14532D; line-height: 1.6;">{items_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_need:
            need_items = match_details['missing']
            items_str = "<br>".join([f"🛒 {item}" for item in need_items]) if need_items else "<i>You have everything needed! Perfect! 🎉</i>"
            st.markdown(f"""
            <div class="checklist-card-need">
                <div class="checklist-title" style="color: #9A3412;">🛒 Market Checklist ({len(need_items)})</div>
                <div style="font-size: 15px; color: #7C2D12; line-height: 1.6;">{items_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # Complete Ingredients Expander
    with st.expander("🧂 Complete Recipe Ingredients List", expanded=False):
        if rec_ings_str and pd.notna(rec_ings_str):
            clean_ings = [i.strip().title() for i in rec_ings_str.split(",") if i.strip()]
            ing_cols = st.columns(2)
            for idx, item in enumerate(clean_ings):
                target_col = ing_cols[idx % 2]
                target_col.markdown(f"• **{item}**")
        else:
            st.write("Ingredient list is tucked away in the chef's secret notes!")

    # Step by Step Instructions
    with st.expander("📋 Step-by-Step Cooking Method", expanded=True):
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
                <div style="background: #FFFDF9; border-left: 3px solid #E07A5F; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 10px 10px 0;">
                    <strong style="color: #D95D39; font-family: 'Patrick Hand', cursive; font-size: 18px;">Step {i} {icon}:</strong>
                    <div style="color: #4A3E3D; font-size: 15px; margin-top: 2px;">{step.rstrip('.')}</div>
                </div>
                """, unsafe_allow_html=True)

    # Tasting Notes & Review Form
    st.markdown("---")
    with st.form(key=f"tasting_review_{recipename}_{selected_row.name}"):
        st.markdown(f"""
        <div style="text-align: center;">
            <h3 style="font-family: 'Playfair Display', serif; color: #D95D39; margin-bottom: 2px;">⭐ Chef's Tasting Review</h3>
            <p style="font-family: 'Patrick Hand', cursive; font-size: 18px; color: #8D7B68;">Leave your rating in the kitchen journal to hone future recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        rating_val = st.radio("Rating:", [1, 2, 3, 4, 5], index=4, horizontal=True, key=f"rad_{recipename}")
        submitted = st.form_submit_button("✍️ Save Tasting Note to Journal", use_container_width=True)
        if submitted:
            user_ings_save = input_ingredients if input_ingredients else []
            append_feedback(FEEDBACK_FILE, recipename, user_ings_save, int(rating_val))
            st.success(f"🎉 Bon appétit! Your {rating_val}-star review for '{recipename}' has been saved in the Chef's Journal!")

# -------------------------------------------------------------------
# Navigation Tabs: Index & Chapters / AI Pantry / Tasting Journal
# -------------------------------------------------------------------
tab_index, tab_ai, tab_journal = st.tabs([
    "📖 Table of Contents (Index & Chapters)",
    "🪄 AI Sous-Chef's Pantry",
    "⭐ Chef's Tasting Journal"
])

# -------------------------------------------------------------------
# TAB 1: Table of Contents & Chapter Index
# -------------------------------------------------------------------
with tab_index:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-family: 'Playfair Display', serif; font-size: 34px; color: #D95D39; margin: 0;">📖 The Master Recipe Book Index</h2>
        <p style="font-family: 'Patrick Hand', cursive; font-size: 20px; color: #7B6858;">Flip through chapters organized by cuisine or search by recipe title</p>
    </div>
    """, unsafe_allow_html=True)

    # Chapter Organization Controls
    col_group, col_filter, col_search = st.columns([1, 1.2, 1.5])
    
    with col_group:
        browse_mode = st.radio("Browse By:", ["Cuisine Chapters", "Course / Meal Type", "All Recipes (A-Z)"], horizontal=False)
    
    with col_filter:
        if browse_mode == "Cuisine Chapters":
            available_cuisines = sorted([c for c in df['cuisine'].dropna().unique() if c != 'unknown'])
            selected_chapter = st.selectbox("Select Chapter:", options=available_cuisines, index=0)
            chapter_df = df[df['cuisine'] == selected_chapter]
            theme = CUISINE_THEMES.get(selected_chapter, {"emoji": "🍲", "subtitle": "Curated dishes and homemade specialties"})
            chapter_title = f"Chapter: {selected_chapter} {theme['emoji']}"
            chapter_desc = theme['subtitle']
        elif browse_mode == "Course / Meal Type":
            available_courses = sorted([c for c in df['course'].dropna().unique() if c != 'unknown'])
            selected_chapter = st.selectbox("Select Course Chapter:", options=available_courses, index=0)
            chapter_df = df[df['course'] == selected_chapter]
            chapter_title = f"Chapter: {selected_chapter} 🍽️"
            chapter_desc = f"Delicious recipes tailored for {selected_chapter}"
        else:
            chapter_df = df.copy()
            chapter_title = "The Complete Alphabetical Index 📚"
            chapter_desc = f"Browsing all {len(df):,} recipes from A to Z"

    with col_search:
        search_query = st.text_input("🔍 Quick search recipe by name:", placeholder="e.g. Biryani, Pasta, Curry...")
        if search_query:
            chapter_df = chapter_df[chapter_df['recipename'].str.contains(search_query, case=False, na=False)]

    # Chapter Banner
    st.markdown(f"""
    <div class="chapter-hero">
        <div class="chapter-hero-emoji">📖</div>
        <div>
            <h3 class="chapter-hero-title">{chapter_title}</h3>
            <p class="chapter-hero-desc">{chapter_desc} • <strong>{len(chapter_df):,} recipes listed</strong></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if chapter_df.empty:
        st.warning("No recipes found matching this chapter or search query. Try another chapter or search term!")
    else:
        # Pagination for clean reading
        recipes_per_page = 10
        total_pages = max(1, int(np.ceil(len(chapter_df) / recipes_per_page)))
        
        c_p1, c_p2 = st.columns([1, 3])
        with c_p1:
            page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)
        with c_p2:
            st.caption(f"Showing page {page_num} of {total_pages} (Recipes {(page_num-1)*recipes_per_page + 1} - {min(page_num*recipes_per_page, len(chapter_df))})")

        start_idx = (page_num - 1) * recipes_per_page
        end_idx = start_idx + recipes_per_page
        page_recipes = chapter_df.iloc[start_idx:end_idx]

        # Render vintage index items
        st.markdown("#### 📜 Chapter Index Entries")
        for idx, row in page_recipes.iterrows():
            r_name = row['recipename']
            r_time = f"⏱️ {int(row['totaltimeinmins'])}m" if pd.notna(row['totaltimeinmins']) and row['totaltimeinmins'] > 0 else "⏱️ Flexible"
            r_serv = f"Yields {int(row['servings'])}" if pd.notna(row['servings']) else "Serves 2-4"
            r_course = f"• {row['course'].title()}" if pd.notna(row['course']) and row['course'] != 'unknown' else ""

            col_entry, col_btn = st.columns([3.5, 1])
            with col_entry:
                st.markdown(f"""
                <div class="index-entry">
                    <span class="index-entry-name">✨ {r_name}</span>
                    <span class="index-entry-meta">{r_time} • {r_serv} {r_course}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                if st.button("📖 Read Page", key=f"btn_read_{row.name}_{idx}", use_container_width=True):
                    st.session_state['active_recipe_row'] = row
                    st.session_state['active_recipe_source'] = "index"

        # Show active selected recipe page if opened from index
        if 'active_recipe_row' in st.session_state and st.session_state.get('active_recipe_source') == "index":
            st.markdown("---")
            st.markdown("<div id='recipe-display-anchor'></div>", unsafe_allow_html=True)
            render_recipe_storybook(st.session_state['active_recipe_row'])

# -------------------------------------------------------------------
# TAB 2: AI Sous-Chef's Pantry (Recommendation System)
# -------------------------------------------------------------------
with tab_ai:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-family: 'Playfair Display', serif; font-size: 34px; color: #D95D39; margin: 0;">🪄 AI Sous-Chef's Magic Pantry</h2>
        <p style="font-family: 'Patrick Hand', cursive; font-size: 20px; color: #7B6858;">Enter whatever ingredients you have in your fridge or cupboards!</p>
    </div>
    """, unsafe_allow_html=True)

    # Main recipe input box
    user_pantry = st.text_input(
        "🍳 What ingredients are on your kitchen counter?",
        value=pantry_input,
        placeholder="e.g. paneer, spinach, garlic, onion, butter",
        key="main_tab_pantry_input"
    )

    if user_pantry:
        input_ingredients = [s.strip().lower() for s in user_pantry.split(",") if s.strip()]
        
        # Tags display
        st.markdown(f"**Ingredients in your basket:** `{'`, `'.join(input_ingredients)}`")

        # Predict cuisine & course from ingredients
        predicted_cuisine = predict_text_label((cuisine_model, cuisine_vectorizer), input_ingredients)
        predicted_course = predict_text_label((course_model, course_vectorizer), input_ingredients)
        
        theme_pred = CUISINE_THEMES.get(predicted_cuisine, {"emoji": "🧑‍🍳", "subtitle": "Fresh culinary creation"})

        st.markdown(f"""
        <div style="background: #FFFDF9; border: 2px dashed #E07A5F; border-radius: 16px; padding: 14px 20px; margin: 15px 0 25px 0; display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 32px;">{theme_pred['emoji']}</span>
            <div>
                <strong style="color: #D95D39; font-size: 16px;">AI Sous-Chef's Intuition:</strong>
                <div style="color: #4A3E3D; font-size: 15px;">Based on your basket, this feels like an authentic <strong>{predicted_cuisine.title()}</strong> <em>{predicted_course.title()}</em> masterpiece!</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Compute intelligence scores
        cuisine_filter_applied = sidebar_cuisine_filter if sidebar_cuisine_filter != "All Cuisines" else None

        scored_df = compute_intelligence_scores(
            df,
            tfidf_vectorizer,
            tfidf_matrix,
            input_ingredients,
            predicted_cuisine,
            predicted_course,
            cuisine_filter_applied,
            feedback_summary,
            semantic_weight,
            fuzzy_weight
        )

        filtered_df = scored_df.copy()

        # Cuisine filter
        if cuisine_filter_applied:
            filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine_filter_applied.lower()]

        # Cooking time filter
        if sidebar_time_filter != "Any Time":
            max_time = int(re.sub(r'\D', '', sidebar_time_filter))
            filtered_df = filtered_df[(filtered_df['totaltimeinmins'] > 0) & (filtered_df['totaltimeinmins'] <= max_time)]

        # Minimum ingredient match %
        if min_ingredient_match > 0:
            filtered_df = filtered_df[filtered_df['fuzzy_score'] >= min_ingredient_match]

        results = filtered_df.sort_values(by='intelligence_score', ascending=False).head(10)

        if results.empty:
            st.warning("⚠️ No recipes found matching these ingredients and filters. Try adding more general ingredients or lowering the minimum match % in the sidebar!")
        else:
            st.markdown(f"""
            <div class="chapter-hero">
                <div class="chapter-hero-emoji">✨</div>
                <div>
                    <h3 class="chapter-hero-title">Special Chapter: Creations from Your Pantry</h3>
                    <p class="chapter-hero-desc">Top 10 recipes handpicked by AI based on your ingredients</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Results selection dropdown
            recipe_titles = results['recipename'].tolist()
            selected_ai_recipe_name = st.selectbox(
                "📖 Choose a recommended recipe to open its storybook page:",
                options=recipe_titles,
                index=0
            )

            selected_row = results[results['recipename'] == selected_ai_recipe_name].iloc[0]

            # Render storybook page
            render_recipe_storybook(selected_row, input_ingredients=input_ingredients)

            st.markdown("---")
            st.markdown("#### 📊 Recommendation Scorecard")
            display_df = results[['recipename', 'cuisine', 'totaltimeinmins', 'intelligence_score', 'fuzzy_score']].copy()
            display_df.columns = ['Recipe Title', 'Cuisine Chapter', 'Cooking Time (mins)', 'AI Score', 'Pantry Match %']
            display_df['AI Score'] = display_df['AI Score'].round(3)
            display_df['Pantry Match %'] = display_df['Pantry Match %'].round(1)
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    else:
        st.info("💡 Enter ingredients above (or click a quick button in the sidebar pantry) to awaken the AI sous-chef!")

# -------------------------------------------------------------------
# TAB 3: Chef's Tasting Journal (Ratings & Reviews)
# -------------------------------------------------------------------
with tab_journal:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-family: 'Playfair Display', serif; font-size: 34px; color: #D95D39; margin: 0;">⭐ Chef's Tasting Journal</h2>
        <p style="font-family: 'Patrick Hand', cursive; font-size: 20px; color: #7B6858;">Every dish you taste and rate shapes your personal recipe intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(FEEDBACK_FILE):
        try:
            fdf = pd.read_csv(FEEDBACK_FILE)
            if not fdf.empty:
                col_j1, col_j2, col_j3 = st.columns(3)
                col_j1.metric("Tastings Recorded", len(fdf))
                col_j2.metric("Average Rating", f"{fdf['rating'].mean():.2f} / 5 ⭐")
                top_recipe = fdf.groupby('selected_recipe')['rating'].mean().idxmax()
                col_j3.metric("Top Rated", str(top_recipe)[:18] + "...")

                st.markdown("---")
                st.markdown("#### 📜 Recent Tasting Entries")
                for _, f_row in fdf.tail(8).iloc[::-1].iterrows():
                    star_str = "⭐" * int(f_row.get('rating', 5))
                    st.markdown(f"""
                    <div class="index-entry">
                        <div>
                            <strong style="color: #38302E; font-size: 16px;">{f_row.get('selected_recipe')}</strong>
                            <div style="font-size: 13px; color: #8D7B68; margin-top: 3px;">
                                Tested on: {f_row.get('timestamp', 'Recent')} • Ingredients: {str(f_row.get('user_ingredients', '')).replace('|', ', ')}
                            </div>
                        </div>
                        <span style="font-size: 18px;">{star_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("🔍 View Raw Journal Data"):
                    st.dataframe(fdf, use_container_width=True)
            else:
                st.info("Your tasting journal is brand new! Cook a recipe and submit a rating to see your entries here.")
        except Exception as e:
            st.error(f"Error reading journal: {e}")
    else:
        st.info("No tasting journal entries found yet. Rate a recipe after cooking to record your first entry!")

# -------------------------------------------------------------------
# Footer: Storybook Colophon
# -------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 25px; border-top: 1px dashed #DDBEA9; font-family: 'Patrick Hand', cursive; color: #8D7B68; font-size: 17px;">
    🍳 <strong>WannabeChef</strong> • Handcrafted with love, spices & AI intelligence.<br>
    <em>"Cooking is at once child's play and adult joy. And cooking with a recipe book is pure storytelling."</em>
</div>
""", unsafe_allow_html=True)
