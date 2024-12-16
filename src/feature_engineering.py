from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_to_pickle
from data_loader import load_and_clean_data
import os

# Paths
data_path = "data/books_summary.csv"
cleaned_data_path = "data/cleaned_books_summary.csv"
vectorizer_path = "model/tfidf_vectorizer.pkl"
tfidf_matrix_path = "model/tfidf_matrix.pkl"


def train_tfidf_model(data):
    """
    Train a TF-IDF vectorizer on the combined text data and save the model.
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

    # Fit and transform the combined text
    print("Training TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(data["combined_text"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save the TF-IDF vectorizer and matrix
    save_to_pickle(vectorizer, vectorizer_path)
    save_to_pickle(tfidf_matrix, tfidf_matrix_path)
    print(
        f"TF-IDF vectorizer and matrix saved to: {vectorizer_path} and {tfidf_matrix_path}"
    )


def main():
    # Ensure the model directory exists
    os.makedirs("model", exist_ok=True)
    # Load, clean, and prepare data
    books_df = load_and_clean_data(data_path, cleaned_data_path)
    train_tfidf_model(books_df)


if __name__ == "__main__":
    main()
