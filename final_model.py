# File: engine.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

print("Engine starting up...")

# --- SETUP: Load data and semantic model once ---
try:
    faculty_df = pd.read_csv("faculty_cleaned.csv")
    faculty_df.columns = faculty_df.columns.str.strip()
    
    # Load the pre-trained semantic model
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Pre-compute embeddings for all faculty to make it fast
    faculty_embeddings = semantic_model.encode(faculty_df['style_tags'].tolist(), convert_to_tensor=True)
    
    print("Models and data loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: 'faculty_cleaned.csv' not found. The engine cannot start.")
    faculty_df = None

# --- THE NEW HYBRID RECOMMENDATION FUNCTION ---
def recommend_faculty_hybrid(query: str = None, style_prefs: list = None, top_n=5):
    
    if faculty_df is None:
        return pd.DataFrame()

    # Start with the full list of professors
    filtered_df = faculty_df.copy()

    # --- 1. FILTERING STEP ---
    # If the user selected any tags, narrow down the list first.
    if style_prefs:
        print(f"Filtering by tags: {style_prefs}")
        for tag in style_prefs:
            # Keep only rows where the 'style_tags' column contains the selected tag
            filtered_df = filtered_df[filtered_df['style_tags'].str.contains(tag, case=False, na=False)]
    
    # If after filtering, no professors are left, return an empty result.
    if filtered_df.empty:
        return pd.DataFrame()

    # --- 2. RANKING STEP ---
    # If the user also typed a query, rank the filtered results semantically.
    if query and query.strip():
        print(f"Ranking filtered results by query: '{query}'")
        # Get the embeddings that correspond to our filtered list of professors
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = faculty_embeddings[filtered_indices]
        
        # Encode the user's query
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        
        # Calculate similarity scores against the FILTERED list
        cosine_scores = util.cos_sim(query_embedding, filtered_embeddings).flatten()
        
        # Add scores to the filtered dataframe and sort
        filtered_df['similarity_score'] = cosine_scores
        recommendations = filtered_df.sort_values(by='similarity_score', ascending=False)
    else:
        # If there's no text query, just sort the filtered results by rating
        recommendations = filtered_df.sort_values(by='rating', ascending=False)
        recommendations['similarity_score'] = 1.0 # Give a default score

    return recommendations.head(top_n)[['faculty_name', 'department', 'rating', 'style_tags', 'similarity_score']]