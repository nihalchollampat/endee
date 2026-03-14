# Semantic Movie Search System with Endee Vector DB

This project demonstrates how to build a practical **Semantic Search / Recommendation System** using the [Endee](https://github.com/endee-io/endee) vector database.

## Project Overview and Problem Statement

### Problem
Traditional databases rely on exact keyword matching for search. If a user searches for "a story about dreams within dreams," a SQL database might struggle if the plot summary doesn't explicitly contain all those keywords. Standard full-text search often fails for abstract concepts, user intents, or synonyms.

### Solution
This project builds a **Semantic Search** application using a high-performance vector database, Endee. By generating mathematical representations (embeddings) of text elements (movie plots), we capture the underlying meaning using a pre-trained SentenceTransformer AI model (HuggingFace `all-MiniLM-L6-v2`).

Users can enter free-form queries (e.g., "A movie about space exploration and black holes") and Endee retrieves the closest matching movie plots using **Cosine Similarity**, ensuring much better contextual relevance than traditional SQL searches.

## System Design and Technical Approach

1. **Embedding Generation**: We use `sentence-transformers` locally to compute context-aware dense vector embeddings (384 dimensions) from the movie plot strings.
2. **Vector Storage Setup**: We leverage Endee to create a high-performance index with `FP32` precision metrics and `cosine` distance tracking.
3. **Data Ingestion**: Each vector is bulk-upserted into the Endee index alongside metadata (`title`, `plot`, `year`, `genre`).
4. **Search/Retrieval**: The system continually waits for user input, computes the vector for the user query, and passes it to Endee's `query()` mechanism which utilizes its optimized SIMD loops for near-instant retrieval.

## Explanation of how Endee is used

In this application, Endee provides the underlying engine that makes retrieval possible:
- **`client.create_index()`**: We provision a semantic index named `movie_recommendations` with the `cosine` space type suited for our sentence-transformers.
- **`index.upsert()`**: We store the plot embeddings tightly joined with their JSON-capable payload metadata (`meta`). Endee takes care of optimizing the graphs in-memory to accelerate search later.
- **`index.query()`**: This computes the exact top-K most similar movies inside Endee, avoiding the need to process elements client-side. The database scales efficiently and directly returns payloads and scores to our console.

## Clear Setup and Execution Instructions

### Prerequisites
1. **Python 3.8+** installed on your system.
2. **Endee Vector Database** running locally or remotely (default: `localhost:8080`).

If you haven't started Endee yet, you can use Docker:
```bash
docker run -p 8080:8080 -v ./endee-data:/data endeeio/endee-server:latest
```

### Installation

1. Navigate to this directory in your terminal:
```bash
cd projects/semantic_movie_search
```

2. (Optional but recommended) Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### Execution

Run the interactive Python script:
```bash
python main.py
```

### Usage
- Upon startup, the script connects to your Endee instance.
- It will instantly drop and recreate its `movie_recommendations` index, generating and embedding 13 classic movies into the vector space.
- The console will display an interactive prompt. Enter what kind of movie you'd like to see.
  - **Example Input:** `"Something with astronauts and gravity"` -> Returns *Interstellar*
  - **Example Input:** `"Mob boss and crime"` -> Returns *The Godfather*, *Goodfellas*
- Type `quit` or press `Ctrl+C` to end.
