"""Data loading, cleaning, and feature engineering utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

NUMERIC_FEATURES = [
    "release_year",
    "popularity",
    "tempo",
    "danceability",
    "energy",
    "acousticness",
    "valence",
]

METADATA_FEATURES = ["song_name", "artist", "genre", "mood"]


def load_songs(path: str | Path | None = None) -> pd.DataFrame:
    """Load the main songs dataset."""
    dataset_path = Path(path) if path else DATA_DIR / "songs_dataset.csv"
    return pd.read_csv(dataset_path)


def clean_songs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and normalize text columns for search."""
    cleaned = df.copy()

    for column in METADATA_FEATURES:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["song_key"] = (
        cleaned["song_name"].str.lower()
        + " - "
        + cleaned["artist"].str.lower()
        + " - "
        + cleaned["release_year"].astype(int).astype(str)
    )
    return cleaned


def build_metadata_text(df: pd.DataFrame) -> pd.Series:
    """Combine metadata into a searchable semantic text field."""
    return (
        df["song_name"].astype(str)
        + " by "
        + df["artist"].astype(str)
        + " genre "
        + df["genre"].astype(str)
        + " mood "
        + df["mood"].astype(str)
        + " year "
        + df["release_year"].astype(int).astype(str)
        + " energy "
        + pd.cut(df["energy"], bins=[0, 0.4, 0.7, 1.0], labels=["low", "medium", "high"]).astype(str)
        + " danceability "
        + pd.cut(df["danceability"], bins=[0, 0.4, 0.7, 1.0], labels=["low", "medium", "high"]).astype(str)
    )


def prepare_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Create a scaled numeric feature matrix for recommendations."""
    features = df[NUMERIC_FEATURES].copy()
    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(features)
    return matrix, NUMERIC_FEATURES
