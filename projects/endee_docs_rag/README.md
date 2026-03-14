# Endee Knowledge Base (RAG Search)

This project demonstrates a production-grade Semantic Search / Retrieval-Augmented Generation (RAG) system using **Endee** as the core vector database. 

It indexes a directory of unstructured markdown documents (in this case, Endee's own official documentation), chunks the text, creates dense vector embeddings via Hugging Face Transformers, and inserts them into an Endee Vector Database. Finally, it provides a functional Gradio Web UI to perform natural language queries securely over the local knowledge base.

## Project Overview

**Problem Statement:** Large organizations often have extensive internal documentation, making it difficult for employees and agents to quickly find the right information using traditional keyword matching. 
**Solution:** By converting text into dense embeddings and indexing them using **Endee**—a high-performance, single-node vector repository—we enable instantaneous, domain-specific semantic search without having to build complex indexing infrastructure from scratch.

## System Design and Technical Approach

This project consists of two core components representing the standard RAG capability pipeline:

1. **Indexer (`indexer.py`)**:
   - Parses standard `.md` knowledge base files via `BeautifulSoup`.
   - Chunks text into normalized blocks of words (100-word blocks with 20-word overlaps to maintain contextual boundaries).
   - Generates 384-dimensional dense vectors using the `sentence-transformers/all-MiniLM-L6-v2` local model, requiring no API keys.
   - Pushes the vectors alongside raw text metadata into Endee.

2. **Web Application (`app.py`)**:
   - Runs a fast, user-friendly interactive web interface built with `Gradio`.
   - On each natural language query, it encodes the user prompt into a search vector.
   - Queries the local Endee instance performing sub-millisecond retrieval.
   - Returns the highly relevant context chunks formatted cleanly into the UI.

### How Endee is Used

Endee is placed perfectly in the center of the application lifecycle:
- Serves as the high-availability vector backbone.
- Stores custom data explicitly mapped into metadata dictionaries via `upsert()`.
- Supports the fundamental retrieval system via `query(vector=..., top_k=5)` exposing its native sub-millisecond vector capability across locally hosted infrastructure.

## Setup and Execution Instructions

### 1. Start the Endee Server
If you don't already have an Endee instance running locally, use Docker to spin one up instantly:
```bash
docker run -p 8080:8080 -v ./endee-data:/data endeeio/endee-server:latest
```

### 2. Setup the Project Environment
Ensure you have Python 3.10+ installed. Provide a local virtual environment and install the required packages.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the Knowledge Base
By default, the `knowledge/` folder contains Markdown documentation ready to be indexed. Run the script below:
```bash
python indexer.py
```
This script will construct the `.md` blocks, transform the embeddings, initialize an Endee index named `endee_docs`, and push all data chunks.

### 4. Run the Search UI
Finally, launch the local Web Portal:
```bash
python app.py
```
Open the provided local URL (e.g., `http://localhost:7860/`) in your browser to start querying the internal Endee knowledge base semantically!
