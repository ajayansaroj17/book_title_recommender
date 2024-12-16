def recommend_books(book_title, df, similarity_matrix, top_n=5):
    if book_title not in df["book_name"].values:
        return "Book not found. Please check the title."

    index = df.index[df["book_name"] == book_title][0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, _ in sorted_scores[1 : top_n + 1]:  # Exclude input book
        recommendations.append(df["book_name"].iloc[idx])
    return recommendations
