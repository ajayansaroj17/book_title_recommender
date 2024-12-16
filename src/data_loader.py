import pandas as pd
from utils import clean_text


def load_and_clean_data(data_path, cleaned_data_path):
    """
    Load dataset, aggregate categories, drop duplicates, and preprocess text.
    """
    # Load the dataset
    books_df = pd.read_csv(data_path)
    print(f"Original dataset shape: {books_df.shape}")

    # Group by 'book_name' and 'book_summary', aggregate 'book_tags'
    books_df = books_df.groupby(["book_name", "summaries"], as_index=False).agg(
        {"categories": lambda tags: ", ".join(set(tags.dropna()))}
    )  # Remove duplicates within tags

    print(f"After aggregating categories and removing duplicates: {books_df.shape}")
    books_df = books_df.drop_duplicates(subset=["book_name", "summaries"], keep="first")
    # Combine 'book_summary' and 'book_tags' into a single text field
    books_df["combined_text"] = (
        books_df["summaries"].fillna("") + " " + books_df["categories"].fillna("")
    )

    # Clean the combined text
    books_df["combined_text"] = books_df["combined_text"].apply(clean_text)

    # Save the cleaned dataset
    books_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_data_path}")

    return books_df
