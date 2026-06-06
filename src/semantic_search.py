"""Semantic retrieval using TF-IDF vectors and cosine similarity."""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from vectorization import vectorize_query


def semantic_search(query: str, df: pd.DataFrame, vectorizer, tfidf_matrix, top_n: int = 10) -> pd.DataFrame:
    """Retrieve songs that semantically match a user query."""
    query_vector = vectorize_query(query, vectorizer)
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    result_indices = scores.argsort()[::-1][:top_n]

    results = df.iloc[result_indices].copy()
    results["semantic_score"] = scores[result_indices]
    return results.reset_index(drop=True)


def metadata_filter(
    df: pd.DataFrame,
    genre: str | None = None,
    mood: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """Filter retrieval context using optional metadata constraints."""
    filtered = df.copy()

    if genre and genre != "All":
        filtered = filtered[filtered["genre"].str.lower() == genre.lower()]
    if mood and mood != "All":
        filtered = filtered[filtered["mood"].str.lower() == mood.lower()]
    if year:
        filtered = filtered[filtered["release_year"].astype(int) == int(year)]

    return filtered
