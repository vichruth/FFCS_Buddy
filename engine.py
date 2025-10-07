# File: engine.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

print("Engine starting up...")

# --- SETUP: Load new data and semantic model once ---
try:
    # Load the new, combined FFCS data
    ffcs_df = pd.read_csv("ffcs_data.csv")
    ffcs_df.columns = ffcs_df.columns.str.strip()
    
    # Load the pre-trained semantic model
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Pre-compute embeddings for the style_tags in the new data
    ffcs_embeddings = semantic_model.encode(ffcs_df['style_tags'].astype(str).tolist(), convert_to_tensor=True)
    
    print("Models and FFCS data loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: 'ffcs_data.csv' not found. The engine cannot start.")
    ffcs_df = None

# --- THE NEW MULTI-FILTER RECOMMENDATION FUNCTION ---
def find_faculty(course_code: str = None, slot: str = None, query: str = None, style_prefs: list = None, top_n=10):
    
    if ffcs_df is None:
        return pd.DataFrame()

    # Start with the full list of course offerings
    filtered_df = ffcs_df.copy()

    # --- 1. HARD FILTERING STEP ---
    # Filter by course code if provided
    if course_code and course_code.strip():
        filtered_df = filtered_df[filtered_df['course_code'].str.contains(course_code, case=False, na=False)]

    # Filter by slot if provided
    if slot and slot.strip():
        filtered_df = filtered_df[filtered_df['slot'].str.contains(slot, case=False, na=False)]
        
    # Filter by style tags if provided
    if style_prefs:
        for tag in style_prefs:
            filtered_df = filtered_df[filtered_df['style_tags'].str.contains(tag, case=False, na=False)]
    
    if filtered_df.empty:
        return pd.DataFrame()

    # --- 2. SEMANTIC RANKING STEP ---
    # If a text query is provided, rank the filtered results
    if query and query.strip():
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = ffcs_embeddings[filtered_indices]
        
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(query_embedding, filtered_embeddings).flatten()
        
        filtered_df['similarity_score'] = cosine_scores
        recommendations = filtered_df.sort_values(by='similarity_score', ascending=False)
    else:
        # If no query, just sort by the professor's rating
        recommendations = filtered_df.sort_values(by='rating', ascending=False)
        recommendations['similarity_score'] = 1.0

    return recommendations.head(top_n)