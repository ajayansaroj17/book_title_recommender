# 📚 Content-Based Book Recommendation System 📖

This is a content-based book recommendation system that recommends books 📕📗 similar to an input book title based on the similarity of book summaries. The system uses **TF-IDF (📊 Term Frequency-Inverse Document Frequency)** and **Cosine Similarity 🧮** to compare books and find the most relevant recommendations. It provides a user-friendly interface built with **Gradio 💻**, where users can enter a book title and get recommendations.

## 📂 Project Structure

```
.
├── app.py                         # 🚀 Main script that runs the app
├── utils.py                       # 🛠️ Helper functions (data loading, model loading)
├── data/
│   ├── books_summary.csv          # 📑 Actual dataset
│   ├── cleaned_books_summary.csv  # 🧹 Preprocessed dataset
├── model/
│   ├── tfidf_vectorizer.pkl       # 🤖 Pre-trained TF-IDF vectorizer
│   ├── tfidf_matrix.pkl           # 🗂️ Pre-calculated TF-IDF matrix
├── src/                           # 📦 Source code folder
│   ├── data_loader.py             # 📥 Module to load and preprocess data
│   ├── feature_engineering.py     # 🧬 Module to create TF-IDF/embedding vectors
│   ├── similarity_calculator.py   # 🧮 Module to calculate similarity matrix
│   ├── recommender.py             # 📚 Main logic to generate recommendations
│   ├── utils.py                   # ⚙️ Utility functions (e.g., cleaning text)
├── requirements.txt               # 📜 List of Python dependencies
└── README.md                      # 📝 Project overview and setup instructions
```

## 🌟 Features

- **📚 Book Recommendation:** Enter a book title, and the system will recommend the top 5 similar books based on their summaries.
- **🏷️ Categorization:** Each recommended book displays its categories as clickable buttons for better user experience.
- **💻 Interactive UI:** Simple and clean interface using Gradio.
- **🔧 Modular Code:** Functions for data loading, preprocessing, model training, and similarity calculation are separated into different files.

## 💻 Technologies Used

- **🐍 Python:** Core language used to build the system.
- **💻 Gradio:** For creating a web-based user interface.
- **📊 Scikit-learn:** For TF-IDF Vectorization and Cosine Similarity calculation.
- **🗂️ Pandas:** For data manipulation and preprocessing.
- **🔢 NumPy:** For numerical operations.

## ⚙️ Setup Instructions

### 1. 🧬 Clone the Repository

```bash
git clone https://github.com/ajayansaroj17/book_title_recommender.git
cd book_title_recommender
```

### 2. 📦 Install Dependencies

Make sure you have Python 3.7+ installed. Then, create a virtual environment and install the required libraries:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. 📚 Download or Prepare the Dataset

The dataset should contain columns for `book_name`, `summaries`, and `categories`. Store the preprocessed dataset as `cleaned_books_summary.csv` in the `data/` folder.

### 4. 🏋️‍♂️ Pre-train the Model

Run the following script to train the TF-IDF model and create the TF-IDF matrix:

```bash
python train_tfidf_model.py
```

This will:

- 🧠 Train the TF-IDF vectorizer on the summaries column.
- 🗂️ Create the TF-IDF matrix for all books.
- 💾 Save the trained model and matrix as `model/tfidf_vectorizer.pkl` and `model/tfidf_matrix.pkl`.

### 5. 🚀 Run the Application

Launch the Gradio-based web interface by running:

```bash
python app.py
```

The application will open in your browser, allowing you to enter a book title and receive recommendations.

## 🤔 How the System Works

1. **👤 User Input:** The user enters a book title in the input field.
2. **🔍 Recommendation Logic:**
   - The system searches for the input book in the dataset.
   - It calculates the TF-IDF vector of the input book's summary and compares it with the summaries of all other books using cosine similarity.
   - The top 5 books with the highest similarity scores are returned.
3. **📊 Output:** Recommendations are displayed, including book titles, summaries, and categories as clickable buttons.

## 📄 File Descriptions

- **app.py:** 🚀 Main script launching the Gradio UI and handling book recommendations.
- **utils.py:** 🛠️ Helper functions for loading models, data preprocessing, and utilities.
- **feature_engineering.py:** 🧬 Trains the TF-IDF model and creates the TF-IDF matrix.
- **data/cleaned_books_summary.csv:** 📚 Cleaned dataset used for training.
- **model/tfidf_vectorizer.pkl** and **model/tfidf_matrix.pkl:** 🤖 Pre-trained TF-IDF model and matrix.

## 📦 Dependencies

Install the following Python packages using:

```bash
pip install -r requirements.txt
```

- **💻 gradio:** For the web interface.
- **📊 sklearn:** For TF-IDF and cosine similarity calculations.
- **🗂️ pandas:** For data manipulation.
- **🔢 numpy:** For numerical operations.

## 🚀 Potential Extensions and Improvements

- **🏷️ Category-Based Filtering:** Filter recommendations by specific categories.
- **🤖 Advanced NLP Techniques:** Use embeddings like Word2Vec, GloVe, or transformer-based models like BERT.
- **👥 Personalization:** Implement a user profiling system for personalized recommendations.
- **⚡ Scalability:** Use Approximate Nearest Neighbors (ANN) for faster similarity calculation on large datasets.

## 🏁 Conclusion

This project demonstrates building a content-based book recommendation system using TF-IDF and cosine similarity. The modular design ensures easy maintenance and extension, while Gradio simplifies deployment and user interaction!🚀
