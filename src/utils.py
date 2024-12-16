import re
import string
import pickle
import os

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# 1. Text Cleaning Function
def clean_text(text):
    """
    Preprocesses the input text by removing special characters, punctuation,
    converting to lowercase, and removing stopwords.
    Args:
        text (str): Input text string.
    Returns:
        str: Cleaned and preprocessed text.
    """
    if not isinstance(text, str):
        return ""  # Handle cases where text might not be a string

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return " ".join(cleaned_words)


# 2. Stopwords Loader (Optional, if using a custom stopwords list)
def load_stopwords(file_path="data/custom_stopwords.txt"):
    """
    Loads custom stopwords from a file.
    Args:
        file_path (str): Path to the stopwords file.
    Returns:
        set: Set of stopwords.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            stopwords = set(file.read().splitlines())
        return stopwords
    return set()


# 3. Save to Pickle
def save_to_pickle(obj, file_path):
    """
    Saves an object to a pickle file.
    Args:
        obj: Object to save.
        file_path (str): Path to save the pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


# 4. Load from Pickle
def load_from_pickle(file_path):
    """
    Loads an object from a pickle file.
    Args:
        file_path (str): Path to the pickle file.
    Returns:
        The loaded object.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"Pickle file not found at {file_path}")


# 5. Input Validation
def validate_input(book_title, book_list):
    """
    Validates if the book title exists in the dataset.
    Args:
        book_title (str): Input book title.
        book_list (list): List of all book titles.
    Returns:
        bool: True if book exists, else False.
    """
    return book_title in book_list
