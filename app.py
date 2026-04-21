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

# ---------------------------
# Config
# ---------------------------
DATA_FOLDER = "data"
FEEDBACK_FILE = "feedback.csv"

# ---------------------------
# Expanded Nutrition Database (Heuristic)
# ---------------------------
# Greatly expanded for better heuristic estimation
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
    
    # Misc
    "salt": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
    "sugar": {"calories": 387, "protein": 0, "fat": 0, "carbs": 100},
    "turmeric": {"calories": 312, "protein": 9, "fat": 3.3, "carbs": 67},
    "cumin": {"calories": 375, "protein": 18, "fat": 22, "carbs": 44},
    "paprika": {"calories": 282, "protein": 14, "fat": 13, "carbs": 54},
    "coriander": {"calories": 23, "protein": 2.1, "fat": 0.5, "carbs": 3.7},
}

# ---------------------------
# Utilities: load + clean
# ---------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

@st.cache_data(show_spinner="Loading recipe database...")
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
                
                # Handle common column name variations
                if 'recipename' not in df.columns:
                    if 'name' in df.columns:
                        df['recipename'] = df['name']
                    elif 'translatedrecipename' in df.columns:
                        df['recipename'] = df['translatedrecipename']
                    else:
                        df['recipename'] = 'Untitled Recipe'
                
                if 'image_url' not in df.columns:
                    df['image_url'] = None
                
                # Fill missing required columns
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
        lambda x: ", ".join(sorted(list(set(i.strip().lower() for i in str(x).split(",")))))
    )
    
    # Fill cuisine/course with 'unknown'
    combined_df['cuisine'] = combined_df['cuisine'].fillna('unknown')
    combined_df['course'] = combined_df['course'].fillna('unknown')
    
    # Ensure time/servings are numeric
    for col in ['preptimeinmins', 'cooktimeinmins', 'totaltimeinmins', 'servings']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Handle NaNs created by coerce
    combined_df['totaltimeinmins'] = combined_df['totaltimeinmins'].fillna(0)
    combined_df['servings'] = combined_df['servings'].fillna(1) # Default to 1 serving
    
    return combined_df

# ---------------------------
# Fuzzy helper & New Feature: Ingredient Match Details
# ---------------------------
def fuzzy_match_score(recipe_ingredients: str, input_ingredients: List[str], threshold=75) -> float:
    """
    Calculates the percentage of user ingredients that find a good match 
    in the recipe's ingredients. (0-100)
    
    This logic is now synchronized with get_ingredient_match_details.
    """
    # Handle potential missing data
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
        
        if best_score >= threshold: # Only count a match if it's a *good* match (same as "You Have" logic)
            matches_found += 1
            
    # Return the percentage of *your* ingredients that were found
    return (matches_found / len(input_ingredients)) * 100.0 

def get_ingredient_match_details(recipe_ingredients_str: str, user_ingredients: List[str], threshold=75) -> Dict[str, List[str]]:
    """
    NEW FEATURE:
    Compares user ingredients to recipe ingredients to find matches and missing items.
    """
    # Handle potential missing data
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

# ---------------------------
# Nutrition estimator + heuristics
# ---------------------------
def estimate_nutrition_from_ingredients(ingredients: List[str], servings: float = 1.0) -> Dict[str, float]:
    """Estimates nutrition using the expanded heuristic DB."""
    total = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    found = 0
    # Use a set to avoid double-counting "chicken, chicken breast" as two chickens
    matched_db_keys: Set[str] = set()

    # Handle empty or invalid ingredient list
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
        
        # Require a decent match, and don't re-add if already matched
        if best_score >= 80 and best_key and best_key not in matched_db_keys:
            matched_db_keys.add(best_key)
            found += 1
            nd = BASE_NUTRITION_DB[best_key]
            for k in total:
                total[k] += nd[k]
                
    if found == 0:
        return {k: 0.0 for k in total}

    # Heuristic adjustment: Assume 100g of each *matched* ingredient per serving
    # This is a very rough heuristic, but better than averaging
    servings = max(1.0, float(servings) if servings and pd.notna(servings) else 1.0)
    per_serving = {k: round(total[k] / servings, 2) for k in total}
    
    return per_serving


def healthiness_label(nutrition_per_serving: Dict[str, float]) -> str:
    cal = nutrition_per_serving.get("calories", 0)
    fat = nutrition_per_serving.get("fat", 0)
    protein = nutrition_per_serving.get("protein", 0)
    
    if cal == 0 and fat == 0 and protein == 0:
        return "Unknown ❓"
    if cal > 650 or fat > 35:
        return "High-Indulgence 🍕"
    if cal <= 400 and protein >= 15 and fat <= 20:
        return "Healthy Choice 🥦"
    if cal <= 300 and protein < 10:
        return "Light Bite 🥗"
    return "Balanced Meal ⚖️"

# ---------------------------
# Train predictors (cuisine + course)
# ---------------------------
@st.cache_data(show_spinner="Training AI classifiers...")
def train_text_classifier(df: pd.DataFrame, target_col: str) -> Tuple[MultinomialNB, CountVectorizer]:
    df2 = df.copy()
    df2 = df2[df2[target_col].notna()]
    
    # --- THIS IS THE FIX ---
    df2 = df2[df2[target_col].str.lower() != "unknown"] 
    # --- END OF FIX ---
    
    if df2.shape[0] < 20:
        return None, None # Not enough data to train
    
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
    except:
        return "unknown"

# ---------------------------
# TF-IDF preparer
# ---------------------------
@st.cache_resource(show_spinner="Preparing semantic search...")
def prepare_tfidf(series: pd.Series) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))
    mat = vec.fit_transform(series.astype(str))
    return vec, mat

# ---------------------------
# Feedback helper: adapt weights
# ---------------------------
def load_feedback_summary(feedback_file: str) -> Dict[str, float]:
    """Return average rating per recipe (if feedback exists)."""
    if not os.path.exists(feedback_file):
        return {}
    try:
        fdf = pd.read_csv(feedback_file)
        if fdf.empty:
            return {}
        if 'selected_recipe' in fdf.columns and 'rating' in fdf.columns:
            summary = fdf.groupby('selected_recipe')['rating'].mean().to_dict()
            return summary
    except Exception:
        return {}
    return {}

# ---------------------------
# Compute CI scores (hybrid)
# ---------------------------
def compute_intelligence_scores(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix,
    input_ingredients: List[str],
    predicted_cuisine: str,
    predicted_course: str,
    user_cuisine_filter: str,
    feedback_summary: Dict[str, float],
    semantic_weight: float, # NEW: Tunable weight
    fuzzy_weight: float     # NEW: Tunable weight
) -> pd.DataFrame:
    
    user_query = " ".join(input_ingredients)
    user_vec = tfidf_vectorizer.transform([user_query])
    semantic_scores = cosine_similarity(user_vec, tfidf_matrix).flatten() # 0..1

    fuzzy_scores = df['ingredients'].apply(lambda x: fuzzy_match_score(x, input_ingredients)) # 0..100

    # cuisine/course bonus
    def bonus(row):
        b = 0.0
        c = str(row.get('cuisine', '')).lower()
        co = str(row.get('course', '')).lower() if row.get('course') is not None else ""
        
        # Bonus if recipe matches AI prediction
        if predicted_cuisine and predicted_cuisine.lower() in c:
            b += 0.05
        if predicted_course and predicted_course.lower() in co:
            b += 0.03
        
        # Larger bonus if recipe matches user's explicit filter
        if user_cuisine_filter and user_cuisine_filter.lower() != "all" and user_cuisine_filter.lower() in c:
            b += 0.07
        return b

    df = df.copy()
    df['semantic_score'] = semantic_scores
    df['fuzzy_score'] = fuzzy_scores
    df['fuzzy_norm'] = df['fuzzy_score'] / 100.0
    df['bonus'] = df.apply(bonus, axis=1)

    # feedback boost: small boost proportional to normalized average rating
    def feedback_boost(row):
        name = row.get('recipename')
        if name in feedback_summary:
            return (feedback_summary[name] / 5.0) * 0.08 # max +0.08 boost
        return 0.0

    df['feedback_boost'] = df.apply(feedback_boost, axis=1)

    # Weighted fusion (tunable) - combining semantic + fuzzy + bonuses + feedback
    # The weights (semantic_weight, fuzzy_weight) now come from the UI
    df['intelligence_score'] = (
        (semantic_weight * df['semantic_score']) + 
        (fuzzy_weight * df['fuzzy_norm']) + 
        df['bonus'] + 
        df['feedback_boost']
    )

    return df

# ---------------------------
# Feedback append
# ---------------------------
def append_feedback(feedback_file: str, selected_recipe: str, user_ingredients: List[str], rating: int):
    rec = {
        "timestamp": pd.Timestamp.now(),
        "selected_recipe": selected_recipe,
        "user_ingredients": "|".join(user_ingredients),
        "rating": rating
    }
    if os.path.exists(feedback_file):
        try:
            df = pd.read_csv(feedback_file)
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([rec]) # Overwrite if corrupted
    else:
        df = pd.DataFrame([rec])
    df.to_csv(feedback_file, index=False)

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="WannabeChef AI", page_icon="🍳", layout="wide")

# --- Custom CSS for "Presentable" Look ---
st.markdown("""
    <style>
        /* Main Title */
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #FF6347; /* Tomato color */
            text-align: center;
            padding-top: 10px;
        }
        /* Subtitle */
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #555;
            margin-bottom: 20px;
        }
        /* Make expanders cleaner */
        .stExpander {
            border: 1px solid #eee;
            border-radius: 10px;
        }
        /* Custom buttons */
        .stButton>button {
            border-radius: 10px;
            border: 2px solid #FF6347;
            color: #FF6347;
            background-color: white;
        }
        .stButton>button:hover {
            border: 2px solid #FF6347;
            color: white;
            background-color: #FF6347;
        }
    </style>
    <div class="title">🍳 WannabeChef</div>
    <div class="subtitle">Your AI-powered sous-chef for finding the perfect recipe!</div>
""", unsafe_allow_html=True)


# --- Load Data ---
csv_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')] if os.path.exists(DATA_FOLDER) else []
df = load_and_combine_datasets(csv_files)

if df.empty:
    st.error("🚨 No recipe data files found in the 'data' folder. Please add CSV files and reload.")
    st.stop()

# --- Prepare Models ---
tfidf_vectorizer, tfidf_matrix = prepare_tfidf(df['ingredients'])
cuisine_model, cuisine_vectorizer = train_text_classifier(df, 'cuisine')
course_model, course_vectorizer = train_text_classifier(df, 'course')
feedback_summary = load_feedback_summary(FEEDBACK_FILE)

# --- Sidebar: Input Controls ---
with st.sidebar:
    st.header("🧑‍🍳 Find Your Recipe")
    user_input = st.text_input("Enter ingredients (comma separated):", placeholder="chicken, tomato, onion")
    
    st.markdown("---")
    st.header("Filtering Options")
    
    # Cuisine Filter
    unique_cuisines = sorted([c for c in df["cuisine"].dropna().unique() if c != 'unknown'])
    cuisine_filter = st.selectbox("Filter by Cuisine", options=["All"] + unique_cuisines)
    
    # NEW: Time Filter
    time_filter = st.selectbox("Max Total Time", 
                               options=["Any", "< 30 min", "< 60 min", "< 90 min"])
    
    st.markdown("---")
    
    # NEW: Tunable Weights
    with st.expander("Advanced Scoring Weights"):
        st.info("Tune how the AI weighs its decisions.")
        semantic_weight = st.slider("Semantic (Topic) Weight", 0.0, 1.0, 0.55, 
                                    help="How much to value recipes with a similar *theme* (e.g., 'pasta' vs 'curry').")
        fuzzy_weight = st.slider("Ingredient (Availability) Weight", 0.0, 1.0, 0.35,
                                 help="How much to value recipes that use the *exact* ingredients you entered.")
        
        # --- NEW SLIDER ---
        min_ingredient_match = st.slider("Minimum Ingredient Match (%)", 0, 100, 10,
                                         help="Filter out recipes that have less than this % of ingredient match. (Set to 0 to see all results).")

    st.markdown("---")
    st.header("Model Info")
    st.write(f"Loaded Recipes: `{len(df)}`")
    st.write(f"Cuisine Model: `{'Trained' if cuisine_model else 'Not Trained'}`")
    st.write(f"Course Model: `{'Trained' if course_model else 'Not Trained'}`")
    st.write(f"Saved Ratings: `{len(feedback_summary)}`")

# --- Main Page: Results ---
if user_input:
    input_ingredients = [s.strip().lower() for s in user_input.split(",") if s.strip()]
    st.markdown(f"**You entered:** `{'`, `'.join(input_ingredients)}`")

    # Predict cuisine & course from ingredients
    predicted_cuisine = predict_text_label((cuisine_model, cuisine_vectorizer), input_ingredients)
    predicted_course = predict_text_label((course_model, course_vectorizer), input_ingredients)
    st.info(f"🤖 **AI Prediction:** Based on your ingredients, this looks like a *{predicted_cuisine.title()}* *{predicted_course.title()}* dish.")

    # Compute intelligence scores
    scored_df = compute_intelligence_scores(
        df,
        tfidf_vectorizer,
        tfidf_matrix,
        input_ingredients,
        predicted_cuisine,
        predicted_course,
        cuisine_filter,
        feedback_summary,
        semantic_weight, # Pass tunable weight
        fuzzy_weight     # Pass tunable weight
    )

    # Apply filters
    filtered_df = scored_df.copy()
    
    if cuisine_filter and cuisine_filter.lower() != "all":
        filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine_filter.lower()]
    
    if time_filter != "Any":
        max_time = int(re.sub(r'\D', '', time_filter))
        filtered_df = filtered_df[(filtered_df['totaltimeinmins'] > 0) & (filtered_df['totaltimeinmins'] <= max_time)]

    # --- NEW FILTER LOGIC ---
    if min_ingredient_match > 0:
        filtered_df = filtered_df[filtered_df['fuzzy_score'] >= min_ingredient_match]

    # Show top results
    results = filtered_df.sort_values(by='intelligence_score', ascending=False).head(10)

    if results.empty:
        st.warning("No good matches found. Try different ingredients or relax your filters.")
    else:
        st.subheader("🎯 Top AI-Recommended Recipes")
        
        # --- Show results table ---
        display_df = results[['recipename', 'cuisine', 'totaltimeinmins', 'intelligence_score', 'fuzzy_score']].copy()
        display_df['intelligence_score'] = display_df['intelligence_score'].round(3)
        display_df['fuzzy_score'] = display_df['fuzzy_score'].round(1)
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

        # --- Show selected recipe details ---
        st.markdown("---")
        selected_name = st.selectbox("Select a recipe to view details", options=results['recipename'].tolist())
        
        selected_row = results[results['recipename'] == selected_name].iloc[0]

        # --- NEW: Redesigned Recipe Layout ---
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.subheader(f"🍽 {selected_row['recipename']}")
            
            # Image with placeholder
            img = selected_row.get('image_url')
            if img and pd.notna(img) and str(img).strip():
                st.image(img, use_column_width=True, caption=f"Cuisine: {selected_row['cuisine'].title()}")
            else:
                st.image("https://placehold.co/600x400/8A9A5B/FFFFFF?text=WannabeChef", 
                         use_column_width=True, caption="No image available")
            
            # Link
            if selected_row.get('url') and pd.notna(selected_row.get('url')):
                st.markdown(f"🔗 [**View Original Recipe**]({selected_row['url']})", unsafe_allow_html=True)

        with right_col:
            st.markdown("### 📊 Key Info")
            
            # Key Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("AI Score", f"{selected_row['intelligence_score']:.2f}")
            c2.metric("Time", f"{selected_row['totaltimeinmins'] or '?'} min")
            c3.metric("Serves", f"{selected_row.get('servings') or '?'}")
            
            st.markdown("### 🍎 Estimated Nutrition")
            st.caption("(per serving, *very* approximate)")
            
            # Nutrition
            rec_ings_str = selected_row.get('ingredients')
            rec_ings = [i.strip() for i in str(rec_ings_str).split(",") if i.strip()] if rec_ings_str and pd.notna(rec_ings_str) else []
            
            servings = selected_row.get('servings')
            nutrition = estimate_nutrition_from_ingredients(rec_ings, servings)
            health_label = healthiness_label(nutrition)
            
            st.info(f"**Healthiness:** {health_label}")
            
            n_col1, n_col2 = st.columns(2)
            n_col1.metric("Calories", f"{nutrition.get('calories', 0):.0f} kcal")
            n_col2.metric("Protein", f"{nutrition.get('protein', 0):.1f} g")
            n_col1.metric("Fat", f"{nutrition.get('fat', 0):.1f} g")
            n_col2.metric("Carbs", f"{nutrition.get('carbs', 0):.1f} g")

        # --- THIS IS THE FIX for the 'None' bug ---
        st.markdown("### 🛒 Ingredient Availability")
        st.caption("How your ingredients match the recipe.")
        
        recipe_ingredients_str = selected_row.get('ingredients')
        
        if recipe_ingredients_str and pd.notna(recipe_ingredients_str):
            match_details = get_ingredient_match_details(recipe_ingredients_str, input_ingredients)
            match_col, missing_col = st.columns(2)
            with match_col:
                st.success(f"**✅ You Have:**\n\n- {', '.join(match_details['matched']) or 'None'}")
            with missing_col:
                st.warning(f"**⚠️ You Might Need:**\n\n- {', '.join(match_details['missing']) or 'None'}")
        else:
            st.error("Ingredient list not available for this recipe.")

        # --- NEW: Expanders for Cleaner Layout ---
        with st.expander("🧂 View Full Ingredient List"):
            if recipe_ingredients_str and pd.notna(recipe_ingredients_str):
                st.markdown("\n".join(f"- {ing.strip().title()}" for ing in recipe_ingredients_str.split(",")))
            else:
                st.markdown("Ingredient list not available.")

        with st.expander("📋 View Instructions"):
            instructions = selected_row.get('instructions')
            
            if not instructions or pd.isna(instructions):
                st.markdown("Instructions not available.")
            # Check if instructions are already well-formatted
            elif "Step 1" in instructions or "\n1." in instructions or (instructions.count('\n') > 5):
                 st.text(instructions)
            else:
                 # Format single-block text into bullet points
                 st.markdown("Here's a step-by-step breakdown:")
                 steps = [step.strip() for step in instructions.split('.') if step.strip()]
                 for step in steps:
                     st.markdown(f"- {step}.")
        # --- END OF FIX ---
        
        # --- NEW: Feedback Form ---
        st.markdown("---")
        with st.form(key=f"feedback_form_{selected_row['recipename']}"):
            st.markdown("### ⭐ Rate this AI Recommendation")
            rating = st.radio("Your rating (1-5)", [1, 2, 3, 4, 5], index=4, horizontal=True, key=f"rating_{selected_row['recipename']}")
            submit_button = st.form_submit_button("Submit Rating")
            
            if submit_button:
                append_feedback(FEEDBACK_FILE, selected_row['recipename'], input_ingredients, int(rating))
                st.success("Thanks — rating saved! (This will improve future recommendations)")

        st.markdown("---")
        st.info(f"**How this was scored:** (Semantic: {semantic_weight*100:.0f}%) + (Ingredient: {fuzzy_weight*100:.0f}%) + (Bonuses & Feedback).")
else:
    st.info("Enter some ingredients in the sidebar to get started!")

# Footer: show feedback file preview if exists
st.sidebar.markdown("---")
st.sidebar.header("Feedback Log")
if os.path.exists(FEEDBACK_FILE):
    if st.sidebar.button("Show/Hide Feedback Data"):
        fdf = pd.read_csv(FEEDBACK_FILE)
        st.sidebar.dataframe(fdf.tail())
