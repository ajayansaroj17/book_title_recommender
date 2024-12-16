** Content-Based Book Recommendation System **

This is a content-based book recommendation system that recommends books similar to an input book title based on the similarity of book summaries. The system uses TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to compare books and find the most relevant recommendations. It provides a user-friendly interface built with Gradio, where users can enter a book title and get recommendations.

Project Structure

.
├── app.py                  # Main script that runs the app
├── utils.py                # Helper functions (data loading, model loading)
├── data/
|   |── books_summary.csv #actual dataset
│   ├── cleaned_books_summary.csv  # Preprocessed dataset
├── model/
│   ├── tfidf_vectorizer.pkl  # Pre-trained TF-IDF vectorizer
│   ├── tfidf_matrix.pkl      # Pre-calculated TF-IDF matrix
├── src/                                 # Source code folder
│   ├── data_loader.py                   # Module to load and preprocess data
│   ├── feature_engineering.py           # Module to create TF-IDF/embedding vectors
│   ├── similarity_calculator.py         # Module to calculate similarity matrix
│   ├── recommender.py                   # Main logic to generate recommendations
│   ├── utils.py                         # Utility functions (e.g., cleaning text)
├── requirement.txt           # List of Python dependencies
└── README.md                # Project overview and setup instructions

Features
Book Recommendation: Enter a book title, and the system will recommend the top 5 similar books based on their summaries.
Categorization: Each recommended book displays its categories as clickable buttons for better user experience.
Interactive UI: Simple and clean interface using Gradio.
Modular Code: The code is modular and easy to extend. Functions for data loading, preprocessing, model training, and similarity calculation are separated into different files.

Technologies Used
Python: The core language used to build the system.
Gradio: Used for creating a simple web-based user interface.
Scikit-learn: Used for TF-IDF Vectorization and Cosine Similarity calculation.
Pandas: Used for data manipulation and preprocessing.
NumPy: Used for numerical operations (like sorting the cosine similarity scores).

Setup Instructions
1. Clone the Repository
First, clone the project repository to your local machine:

bash
Copy code
git clone https://github.com/ajayansaroj17/book_title_recommender.git
cd book_title_recommender
2. Install Dependencies
Make sure you have Python 3.7+ installed. Then, create a virtual environment and install the required libraries:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
3. Download or Prepare the Dataset
The dataset should contain columns for book_name, summaries, and categories. If you don't have the dataset, you can use any book summary dataset or download an example dataset. For simplicity, ensure the dataset is in the CSV format.

The cleaned and preprocessed dataset should be stored as cleaned_books_summary_dataset.csv in the data/ folder.

4. Pre-train the Model
Before running the app, you need to train the TF-IDF model and create the TF-IDF matrix. Run the following script to do this:

bash
Copy code
python train_tfidf_model.py
This will:

Train the TF-IDF vectorizer on the summaries column.
Create the TF-IDF matrix for all books.
Save the trained TF-IDF vectorizer and TF-IDF matrix as model/tfidf_vectorizer.pkl and model/tfidf_matrix.pkl.
5. Run the Application
Once everything is set up, you can start the Gradio-based web interface by running:

bash
Copy code
python app.py
The application will launch in your browser, where you can enter a book title and receive the top 5 recommendations.

How the System Works
User Input: The user enters a book title in the input field.
Recommendation Logic:
The system searches for the input book in the dataset.
It calculates the TF-IDF vector of the input book's summary and compares it with the summaries of all other books in the dataset using cosine similarity.
The top 5 books with the highest similarity scores are returned.
Output: The system displays the recommendations in a clean and structured format, including the book's title, summary, and categories (displayed as clickable buttons).
File Descriptions
app.py
The main script to launch the Gradio UI. It takes the user's input (book title), computes recommendations, and displays them. It also handles the core logic of book recommendation using TF-IDF and cosine similarity.

utils.py
Contains helper functions for loading the pre-trained model (load_from_pickle), data preprocessing, and other utilities.

feature_engineering.py
This script trains the TF-IDF model on the dataset's book summaries, creates the TF-IDF matrix, and saves the model and matrix to disk.

data/cleaned_books_summary_dataset.csv
The cleaned dataset containing book names, summaries, and categories. This is the dataset used for training the model.

model/tfidf_vectorizer.pkl and model/tfidf_matrix.pkl
These files contain the pre-trained TF-IDF vectorizer and the TF-IDF matrix, which are used by the recommendation system for similarity calculation.

Dependencies
This project requires the following Python packages. You can install them using pip:

gradio: For the user interface.
sklearn: For TF-IDF vectorization and cosine similarity.
pandas: For data manipulation.
numpy: For numerical operations.
To install the required dependencies, run:

pip install -r requirements.txt

Potential Extensions and Improvements
Category-Based Filtering: You could filter recommendations based on specific categories (e.g., "Fantasy" or "Science Fiction").
More Advanced NLP Techniques: For better performance, you could use word embeddings (Word2Vec, GloVe) or transformer-based models like BERT to improve similarity calculations.
Personalization: Implement a user profiling system to tailor recommendations based on user preferences or past interactions.
Scalability: If the dataset grows large, consider using techniques like Approximate Nearest Neighbors (ANN) for faster similarity calculation.
Conclusion
This project demonstrates how to build a content-based book recommendation system using TF-IDF and cosine similarity. It provides an easy-to-use interface for users to input a book title and receive relevant book recommendations. The modular design ensures that the system is easy to extend and maintain, and the use of Gradio makes the deployment and user interaction seamless.

Feel free to modify and extend the system based on your needs!