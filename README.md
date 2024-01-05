# AI-Powered Music Recommendation System

## Overview

This project is a content-based AI music recommendation system enhanced with semantic retrieval and RAG-inspired recommendation workflows.

The system analyzes song features, metadata, genres, moods, and vector similarities to generate personalized music recommendations.

## Objective

The goal is to build an intelligent recommendation engine capable of retrieving semantically similar songs using vector similarity and machine learning techniques.

## Features

* Content-based recommendation
* Semantic song retrieval
* Cosine similarity engine
* PCA dimensionality reduction
* K-Means clustering
* Streamlit web application
* Recommendation explanation system
* Vector similarity search
* Metadata-aware recommendations
* Mood and genre similarity search
* Top-N recommendation output
* Search by song name and release year
* Similarity score display

## RAG-Inspired Workflow

This project demonstrates RAG-inspired retrieval concepts using vector embeddings and semantic similarity search without relying on large language models.

1. Song metadata is cleaned and combined into searchable text.
2. TF-IDF creates vector representations of song metadata.
3. A query or selected song retrieves semantically related songs.
4. Cosine similarity ranks the retrieved context.
5. The recommendation engine generates context-aware recommendations from the most relevant retrieved songs.

## How RAG Concepts Are Applied

This is not a full LLM RAG system. It uses a lightweight retrieval-first recommendation workflow inspired by RAG architecture:

* **Retrieval step:** song metadata and feature descriptions are searched using TF-IDF vectors.
* **Similarity search:** cosine similarity identifies songs with related genre, mood, artist, year, and audio-feature patterns.
* **Vector representation:** every song is represented as a semantic vector built from metadata.
* **Context retrieval:** the strongest matches are treated as recommendation context.
* **Recommendation generation:** the content-based engine recommends songs from the retrieved context and explains why each song is relevant.

## Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Machine Learning
* NLP concepts
* Recommendation Systems

## Skills Demonstrated

* Recommendation systems
* Vector similarity search
* Feature engineering
* NLP preprocessing
* Machine learning pipelines
* Semantic retrieval
* Streamlit deployment
* Exploratory data analysis
* Clustering
* Dimensionality reduction

## Key Insights

* Songs with similar mood and energy cluster together.
* Cosine similarity effectively identifies semantically related songs.
* PCA helps reduce dimensionality while preserving major recommendation patterns.
* Semantic retrieval improves recommendation relevance by using metadata before recommendation generation.
* Metadata-aware retrieval enhances personalization across mood, genre, artist, and release year.

## Project Structure

```text
music-recommendation-system/
├── README.md
├── requirements.txt
├── data/
│   ├── songs_dataset.csv
│   ├── artists_dataset.csv
│   └── genres_dataset.csv
├── notebooks/
│   └── music_recommendation_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── recommendation_engine.py
│   ├── semantic_search.py
│   └── rag_pipeline.py
├── app/
│   └── streamlit_app.py
├── visuals/
│   └── generated charts and screenshots
├── insights/
│   └── recommendation_insights.md
└── docs/
    └── architecture.md
```

## How to Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis notebook:

```bash
jupyter notebook notebooks/music_recommendation_analysis.ipynb
```

4. Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Future Improvements

* LLM integration
* Real-time recommendation APIs
* Audio embeddings
* User personalization
* Deep learning recommendation models

## Disclaimer

This project is based on academic research and enhanced into a portfolio-grade AI recommendation system for educational and demonstration purposes.