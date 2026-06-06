"""Vector generation for semantic search and recommendation retrieval."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import build_metadata_text


def build_tfidf_vectors(df: pd.DataFrame):
    """Create TF-IDF vectors from song metadata."""
    metadata_text = build_metadata_text(df)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=800,
    )
    vectors = vectorizer.fit_transform(metadata_text)
    return vectorizer, vectors, metadata_text


def vectorize_query(query: str, vectorizer: TfidfVectorizer):
    """Transform a free-text query into the fitted TF-IDF vector space."""
    return vectorizer.transform([query])
