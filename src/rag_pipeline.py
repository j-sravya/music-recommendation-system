"""RAG-inspired retrieval workflow without LLM generation."""

import pandas as pd

from recommendation_engine import recommend_songs
from semantic_search import semantic_search


def retrieve_context(query: str, df: pd.DataFrame, vectorizer, tfidf_matrix, top_n: int = 8) -> pd.DataFrame:
    """Retrieve semantically relevant song context from vector similarity."""
    return semantic_search(query, df, vectorizer, tfidf_matrix, top_n=top_n)


def generate_recommendations_from_context(
    query: str,
    df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve context first, then recommend from the strongest matched song."""
    context = retrieve_context(query, df, vectorizer, tfidf_matrix, top_n=8)
    if context.empty:
        return context, pd.DataFrame()

    anchor_song = context.iloc[0]
    recommendations = recommend_songs(
        df=df,
        song_name=anchor_song["song_name"],
        year=int(anchor_song["release_year"]),
        top_n=top_n,
    )
    return context, recommendations


def explain_retrieved_context(query: str, context: pd.DataFrame) -> str:
    """Explain how retrieved context supports recommendation generation."""
    if context.empty:
        return "No matching context was retrieved for this query."

    top_song = context.iloc[0]
    return (
        f"The query '{query}' retrieved '{top_song['song_name']}' by {top_song['artist']} "
        f"as the strongest context because its metadata, genre, mood, and feature terms "
        f"had the highest TF-IDF cosine similarity."
    )
