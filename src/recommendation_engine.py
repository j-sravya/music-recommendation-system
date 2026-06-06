"""Content-based recommendation engine."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import NUMERIC_FEATURES, prepare_feature_matrix


def compute_similarity_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for song feature vectors."""
    return cosine_similarity(feature_matrix)


def find_song_index(df: pd.DataFrame, song_name: str, year: int | None = None) -> int:
    """Find a song by name and optional release year."""
    matches = df[df["song_name"].str.lower() == song_name.lower()]
    if year is not None:
        matches = matches[matches["release_year"].astype(int) == int(year)]
    if matches.empty:
        raise ValueError(f"Song not found: {song_name}")
    return int(matches.index[0])


def recommend_songs(
    df: pd.DataFrame,
    song_name: str,
    year: int | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-N content-based recommendations for a selected song."""
    feature_matrix, _ = prepare_feature_matrix(df)
    similarity_matrix = compute_similarity_matrix(feature_matrix)
    song_index = find_song_index(df, song_name, year)

    similar_indices = similarity_matrix[song_index].argsort()[::-1]
    similar_indices = [idx for idx in similar_indices if idx != song_index][:top_n]

    recommendations = df.iloc[similar_indices].copy()
    recommendations["similarity_score"] = similarity_matrix[song_index][similar_indices]
    recommendations["similarity_percentage"] = (recommendations["similarity_score"] * 100).round(2)
    recommendations["recommendation_reason"] = recommendations.apply(
        lambda row: explain_recommendation(df.iloc[song_index], row),
        axis=1,
    )
    return recommendations.reset_index(drop=True)


def explain_recommendation(source_song: pd.Series, recommended_song: pd.Series) -> str:
    """Generate a concise explanation for a recommendation."""
    reasons = []

    if source_song["genre"] == recommended_song["genre"]:
        reasons.append("same genre")
    if source_song["mood"] == recommended_song["mood"]:
        reasons.append("matching mood")
    if abs(source_song["tempo"] - recommended_song["tempo"]) <= 12:
        reasons.append("similar tempo")
    if abs(source_song["energy"] - recommended_song["energy"]) <= 0.15:
        reasons.append("close energy")
    if abs(source_song["danceability"] - recommended_song["danceability"]) <= 0.15:
        reasons.append("similar danceability")

    if not reasons:
        reasons.append("overall feature similarity")

    return "Recommended because this song shares " + ", ".join(reasons) + "."


def add_pca_and_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Add PCA coordinates and K-Means cluster labels to the dataset."""
    feature_matrix, _ = prepare_feature_matrix(df)
    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(feature_matrix)

    cluster_count = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_matrix)

    enriched = df.copy()
    enriched["pca_1"] = pca_components[:, 0]
    enriched["pca_2"] = pca_components[:, 1]
    enriched["cluster"] = clusters
    return enriched


def feature_profile(song: pd.Series) -> dict[str, float]:
    """Return the key recommendation features for a song."""
    return {feature: float(song[feature]) for feature in NUMERIC_FEATURES}
