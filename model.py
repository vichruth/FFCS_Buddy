# %%
import pandas as pd

# %%
df = pd.read_csv("/mnt/c/Users/vichr/OneDrive/Desktop/ml projects/FFCS_Buddy/faculty.csv")
df.head()

# %%
from sklearn.preprocessing import MultiLabelBinarizer

# %%
df['style_tags_list'] = df['style_tags'].apply(lambda x: x.split(','))

# 1. Convert the 'style_tags' column to a list of tags
mlb = MultiLabelBinarizer()
encoded_tags = mlb.fit_transform(df['style_tags_list'])

# Create a new DataFrame with the encoded tags
# The column names will be the unique tags themselves
encoded_df = pd.DataFrame(encoded_tags, columns=[f"tag_{cls}" for cls in mlb.classes_])

# 4. Combine the original data with the new encoded columns
final_df = pd.concat([df.drop(['style_tags', 'style_tags_list'], axis=1), encoded_df], axis=1)

# 5. Save the final, numerically-encoded data to a new CSV file
final_df.to_csv("faculty_encoded.csv", index=False)

# %%
final_df.head()


# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    # Use the file that already has the 0s and 1s
    faculty_df = pd.read_csv("faculty_encoded.csv")
except FileNotFoundError:
    print("Error: 'faculty_encoded.csv' not found.")
    print("Make sure you have run the 'encode_data.py' script first.")
    exit()
# Get the list of all possible tag columns (e.g., 'tag_chill', 'tag_strict', etc.)
tag_columns = [col for col in faculty_df.columns if col.startswith('tag_')]

# Separate the numerical tag data from the rest of the info
faculty_tag_matrix = faculty_df[tag_columns].values

def recommend_faculty(user_preferences: list, top_n=5):
    """
    Recommends top N faculty using the pre-encoded numerical data.

    Args:
        user_preferences (list): A list of the user's preferred tags (e.g., ['chill', 'project-based']).
        top_n (int): The number of recommendations to return.

    Returns:
        pandas.DataFrame: A DataFrame with the top N recommended faculty.
    """
    # Create a "user profile" vector of 0s and 1s that matches the faculty data format
    user_vector = np.zeros(len(tag_columns))
    for pref in user_preferences:
        # Create the column name, e.g., 'chill' -> 'tag_chill'
        tag_col_name = f"tag_{pref}"
        if tag_col_name in tag_columns:
            # Find the index of this tag column
            col_index = tag_columns.index(tag_col_name)
            # Set the user's preference for this tag to 1
            user_vector[col_index] = 1

    # Calculate the cosine similarity between the user's vector and ALL faculty members
    # We need to reshape the user_vector to be a 2D array for the function
    cosine_similarities = cosine_similarity(user_vector.reshape(1, -1), faculty_tag_matrix).flatten()

    # Add the similarity scores to the original DataFrame
    faculty_df['similarity_score'] = cosine_similarities

    # Sort the faculty by the similarity score in descending order
    recommendations = faculty_df.sort_values(by=['similarity_score', 'rating'], ascending=False)
    
    # We need to re-add the original 'style_tags' for display purposes.
    # Let's recreate it from the encoded columns.
    def get_tags_from_row(row):
        return [col.replace('tag_', '') for col in tag_columns if row[col] == 1]

    recommendations['style_tags'] = recommendations.apply(get_tags_from_row, axis=1).str.join(',')


    return recommendations.head(top_n)[['faculty_name', 'department', 'rating', 'style_tags', 'similarity_score']]


# --- 4. Example Usage ---

if __name__ == "__main__":
    # Define a sample user's preferences
    user_prefs = ["project-based", "helpful", "lenient-grading"]
    print(f"Finding recommendations for user with preferences: {user_prefs}\n")

    # Get the recommendations
    top_faculty = recommend_faculty(user_prefs)

    # Print the results
    print("--- Top 5 Recommended Faculty ---")
    print(top_faculty)


