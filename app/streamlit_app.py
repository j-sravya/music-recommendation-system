"""Streamlit interface for the music recommendation system."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import clean_songs, load_songs
from rag_pipeline import explain_retrieved_context, generate_recommendations_from_context
from recommendation_engine import add_pca_and_clusters, recommend_songs
from vectorization import build_tfidf_vectors


st.set_page_config(
    page_title="AI Music Recommendation System",
    page_icon="music",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #0e1117; color: #f5f7fb; }
    [data-testid="stSidebar"] { background: #151a24; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stSlider label, .stSelectbox label, .stTextInput label {
        color: #f5f7fb !important;
    }
    [data-testid="stAlert"] {
        color: #dbeafe;
    }
    .metric-card {
        background: #151a24;
        border: 1px solid #283244;
        border-radius: 8px;
        padding: 16px;
    }
    .song-card {
        background: #151a24;
        border: 1px solid #293348;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0;
    }
    .score { color: #6ee7b7; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_project_data() -> pd.DataFrame:
    songs = clean_songs(load_songs())
    return add_pca_and_clusters(songs)


@st.cache_resource
def load_vectors(df: pd.DataFrame):
    return build_tfidf_vectors(df)


df = load_project_data()
vectorizer, tfidf_matrix, metadata_text = load_vectors(df)

st.title("AI-Powered Music Recommendation System")
st.caption("Content-based recommendations with TF-IDF semantic retrieval and RAG-inspired vector search concepts.")

left, mid, right = st.columns(3)
left.metric("Songs", len(df))
mid.metric("Genres", df["genre"].nunique())
right.metric("Artists", df["artist"].nunique())

tab_recommend, tab_semantic, tab_visuals = st.tabs(
    ["Recommendations", "Semantic Search", "Visual Insights"]
)

with tab_recommend:
    st.subheader("Song Similarity Recommendations")
    col_a, col_b, col_c = st.columns([2, 1, 1])
    song_name = col_a.selectbox("Choose a song", sorted(df["song_name"].unique()))
    available_years = sorted(df[df["song_name"] == song_name]["release_year"].astype(int).unique())
    year = col_b.selectbox("Release year", available_years)
    top_n = col_c.slider("Top N", 3, 15, 8)

    recommendations = recommend_songs(df, song_name=song_name, year=int(year), top_n=top_n)
    source = df[(df["song_name"] == song_name) & (df["release_year"].astype(int) == int(year))].iloc[0]
    st.markdown(
        f"""
        <div class="song-card">
            Anchor: <strong>{source['song_name']}</strong> by {source['artist']} |
            {source['genre']} | {source['mood']} mood | popularity {int(source['popularity'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for _, row in recommendations.iterrows():
        st.markdown(
            f"""
            <div class="song-card">
                <strong>{row['song_name']}</strong> by {row['artist']}<br>
                {row['genre']} | {row['mood']} | {int(row['release_year'])}<br>
                Similarity: <span class="score">{row['similarity_percentage']}%</span><br>
                {row['recommendation_reason']}
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_semantic:
    st.subheader("RAG-Inspired Semantic Retrieval")
    query = st.text_input("Search by mood, genre, artist, era, or energy", "energetic pop dance songs")
    context, rag_recommendations = generate_recommendations_from_context(
        query, df, vectorizer, tfidf_matrix, top_n=6
    )

    st.write(explain_retrieved_context(query, context))
    st.markdown("**Retrieved Context**")
    st.dataframe(
        context[["song_name", "artist", "genre", "mood", "release_year", "semantic_score"]]
        .assign(semantic_score=lambda x: (x["semantic_score"] * 100).round(2)),
        width="stretch",
    )

    st.markdown("**Recommendations Generated From Retrieved Context**")
    for _, row in rag_recommendations.iterrows():
        st.markdown(
            f"""
            <div class="song-card">
                <strong>{row['song_name']}</strong> by {row['artist']} |
                <span class="score">{row['similarity_percentage']}%</span><br>
                {row['recommendation_reason']}
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_visuals:
    st.subheader("Exploratory Analysis")
    visual_files = [
        "genre_distribution.png",
        "popularity_analysis.png",
        "pca_visualization.png",
        "cluster_visualization.png",
        "recommendation_similarity_chart.png",
        "mood_analysis.png",
        "artist_analysis.png",
        "song_feature_distributions.png",
        "top_genres.png",
        "energy_vs_popularity.png",
    ]
    cols = st.columns(2)
    for index, visual in enumerate(visual_files):
        path = PROJECT_DIR / "visuals" / visual
        if path.exists():
            cols[index % 2].image(str(path), caption=visual.replace("_", " ").replace(".png", "").title())
