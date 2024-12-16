import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from src.utils import load_from_pickle, validate_input

VECTOR_PATH = "model/tfidf_vectorizer.pkl"
MATRIX_PATH = "model/tfidf_matrix.pkl"
DATA_PATH = "data/books_summary.csv"

# 1. Load the pre-trained models and data
print("Loading models and data...")
tfidf_vectorizer = load_from_pickle(VECTOR_PATH)
tfidf_matrix = load_from_pickle(MATRIX_PATH)
books_df = pd.read_csv(DATA_PATH)
print(f"Original dataset shape: {books_df.shape}")

# Group by 'book_name' and 'summaries', and aggregate 'categories' into a single cell

books_df = books_df.groupby(["book_name", "summaries"], as_index=False).agg(
    {"categories": lambda tags: ", ".join(set(tags.dropna()))}
)  # Remove duplicates within tags
print(f"After aggregating categories: {books_df.shape}")

# Drop duplicates (just to be extra cautious)
books_df = books_df.drop_duplicates(subset=["book_name", "summaries"], keep="first")

book_titles = books_df["book_name"].tolist()
print("Models and data loaded successfully!")


# 2. Recommendation Function
def recommend_books(input_book_title):
    """
    Recommends top 5 similar books based on the input book title.

    Args:
        input_book_title (str): The title of the book input by the user.

    Returns:
        List of recommended books with their summaries and tags.
    """
    # Validate input
    if not validate_input(input_book_title, book_titles):
        return "Book title not found in the dataset. Please try another title."

    # Find index of the input book
    book_index = books_df[books_df["book_name"] == input_book_title].index[0]

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(
        tfidf_matrix[book_index], tfidf_matrix
    ).flatten()

    # Sort and get top 5 similar books (excluding the input book itself)
    similar_indices = cosine_similarities.argsort()[-6:-1][::-1]
    recommendations = books_df.iloc[similar_indices]

    """# Format the output
    output = []
    for _, row in recommendations.iterrows():
        output.append(f"**Title:** {row['book_name']}\n**Summary:** {row['summaries']}\n**Tags:** {row['categories']}\n")
    
    return "\n\n".join(output)"""
    # Format the recommendations for the UI
    formatted_books = []
    for _, row in recommendations.iterrows():
        formatted_books.append(
            {
                "title": row["book_name"],
                "description": row["summaries"],
                "categories": row["categories"].split(", "),
            }
        )

    return formatted_books


def display_recommendations(book_title):
    """
    Wrapper function to display recommendations.
    """
    result = recommend_books(book_title)

    if isinstance(result, str):  # If it's an error message
        return result

    # Construct formatted HTML response for book recommendations
    response = ""
    for book in result:
        response += f"""
        <div style='border:1px solid #ddd; border-radius:10px; padding:10px; margin:10px; box-shadow:2px 2px 8px #ccc;'>
            <h2 style='color:#333;'>{book['title']}</h2>
            <p style='color:#555;'>{book['description']}</p>
            <div>
                {" ".join([f"<button style='background-color:#007BFF; color:white; border:none; padding:5px 10px; margin:2px; border-radius:5px;'>{tag}</button>" for tag in book['categories']])}
            </div>
        </div>
        """
    return response


# 3. Gradio Interface
# Gradio UI definition
interface = gr.Interface(
    fn=display_recommendations,
    inputs=gr.Textbox(label="Enter Book Title", placeholder="e.g., The Great Gatsby"),
    outputs=gr.HTML(label="Top 5 Recommendations"),
    title="📚 Book Recommendation System",
    description="Enter the title of a book, and we'll recommend 5 similar books.",
    theme="compact",
)


if __name__ == "__main__":
    # Run the Gradio interface when app.py is executed
    interface.launch()
