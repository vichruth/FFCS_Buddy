import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. Load Data and the Pre-trained Semantic Model ---

try:
    faculty_df = pd.read_csv("faculty_cleaned.csv")
    faculty_df.columns = faculty_df.columns.str.strip()
except FileNotFoundError:
    print("Error: 'faculty_cleaned.csv' not found.")
    exit()

print("Loading the semantic search model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully.")

# --- 2. Pre-compute Embeddings for all Faculty (Optimization) ---
print("Creating semantic embeddings for all faculty...")
faculty_embeddings = model.encode(faculty_df['style_tags'].tolist(), convert_to_tensor=True)
print("Embeddings created and ready.")


# --- 3. The New, Smarter Recommendation Function ---

def recommend_faculty_semantic(user_query: str, top_n=5):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, faculty_embeddings)
    top_results = torch.topk(cosine_scores, k=top_n)

    recommendations = []
    for score, idx in zip(top_results[0][0], top_results[1][0]):
        
        # --- THIS IS THE FIX ---
        # Convert the tensor 'idx' to a standard Python integer using .item()
        row_index = idx.item()
        
        recommendations.append({
            'faculty_name': faculty_df.iloc[row_index]['faculty_name'],
            'department': faculty_df.iloc[row_index]['department'],
            'rating': faculty_df.iloc[row_index]['rating'],
            'style_tags': faculty_df.iloc[row_index]['style_tags'],
            'semantic_score': score.item()
        })
    
    return pd.DataFrame(recommendations)


# --- 4. Example Usage for Testing ---

if __name__ == "__main__":
    query = "a relaxed professor who gives hands-on assignments"
    print(f"\nFinding recommendations for user query: '{query}'\n")
    top_faculty = recommend_faculty_semantic(query)
    print("--- Top 5 Semantically Similar Faculty ---")
    print(top_faculty)

    print("\n" + "="*40 + "\n")

    query_2 = "a very hard teacher for a math-heavy course"
    print(f"Finding recommendations for user query: '{query_2}'\n")
    top_faculty_2 = recommend_faculty_semantic(query_2)
    print("--- Top 5 Semantically Similar Faculty ---")
    print(top_faculty_2)