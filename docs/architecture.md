# Architecture

## Recommendation Pipeline

The system follows a content-based recommendation design. Each song is represented with numeric audio-style features such as tempo, danceability, energy, acousticness, valence, popularity, and release year.

1. Load song metadata from CSV files.
2. Clean text and numeric fields.
3. Normalize numeric features with `MinMaxScaler`.
4. Compute pairwise cosine similarity between song feature vectors.
5. Return the highest scoring songs as Top-N recommendations.
6. Generate a human-readable explanation for each recommendation.

## Semantic Retrieval Flow

The semantic search pipeline converts metadata into a searchable vector space.

1. Song name, artist, genre, mood, year, energy level, and danceability level are merged into text.
2. `TfidfVectorizer` converts this text into sparse vectors.
3. A user query is transformed into the same vector space.
4. Cosine similarity ranks songs by semantic match.
5. Retrieved songs become context for recommendation generation.

## Vector Search

Vector search allows the system to compare songs and queries mathematically instead of relying only on exact keywords. Songs with overlapping metadata terms and related descriptive features appear closer in the TF-IDF vector space.

The project uses two vector views:

* **Semantic vectors:** TF-IDF vectors created from song metadata.
* **Feature vectors:** scaled numeric vectors created from audio-style features.

## Cosine Similarity

Cosine similarity measures the angle between two vectors. A score closer to `1.0` means the songs are highly similar, while a score closer to `0.0` means they are less related.

The system uses cosine similarity for:

* song-to-song content recommendation
* query-to-song semantic retrieval
* ranking recommendation results

## PCA Workflow

PCA reduces the high-dimensional feature matrix into two components for visualization. This helps show how songs with similar energy, danceability, mood, and popularity patterns group together.

1. Scale numeric recommendation features.
2. Apply PCA with two components.
3. Plot songs in a 2D space.
4. Compare PCA patterns with K-Means clusters.

## K-Means Clustering

K-Means groups songs into clusters based on their numeric feature profiles. These clusters help identify listening patterns such as high-energy dance tracks, acoustic calm tracks, and balanced mainstream tracks.

## RAG-Inspired Recommendation Flow

This project does not use LLM fine-tuning, GPT training, or a production RAG architecture. It demonstrates a RAG-inspired semantic retrieval and recommendation workflow using vector similarity.

```text
User query or selected song
        |
        v
Metadata preprocessing
        |
        v
TF-IDF vector embeddings
        |
        v
Cosine similarity retrieval
        |
        v
Retrieved song context
        |
        v
Content-based recommendation engine
        |
        v
Recommendations with similarity scores and explanations
```

## Recommendation Reasoning

Each recommendation includes an explanation based on shared metadata and feature similarity. The engine checks genre, mood, tempo, energy, and danceability to produce reasoning such as:

```text
Recommended because this song shares same genre, matching mood, similar tempo, and close energy.
```

This makes the recommendation process more transparent and recruiter-friendly.
