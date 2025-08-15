This project implements a Retrieval-Augmented Generation (RAG) system for searching and summarising content from a collection of documents, with a focus on extracting meaningful quotes and insights.

## Features
* **Semantic Search**: Retrieve the most relevant content using vector embeddings.
* **Intelligent Chunking**: Break documents into context-aware chunks for better retrieval.
* **Quote Extraction**: Automatically identifies meaningful quotes and key points.
* **Hybrid Relevance Ranking**: Combines embeddings with cross-encoder reranking for high-quality results.
* **Insight Summarisation**: Generates coherent summaries from multiple sources.
* **Fully Local**: Runs entirely on your machine using HuggingFace models.

## Components
### 1. `create_database.py`
* Loads Markdown files from `highlights/`
* Splits text into semantic chunks
* Generates embeddings with `sentence-transformers/all-mpnet-base-v2`
* Stores vectors in ChromaDB for efficient local retrieval

### 2. `query_data.py`
* Performs semantic search with optional hybrid reranking
* Extracts and filters high-quality quotes
* Generates concise, context-aware summaries
* Formats output for readability

## Setup
1. Install dependencies:
```bash
pip install langchain langchain-community chromadb sentence-transformers transformers torch
```

2. Place Markdown documents in a `highlights/` directory.

3. Create the vector database:
```bash
python create_database.py
```

4. Query the knowledge base:
```bash
python query_data.py "your question about the content"
```

## Usage Examples
* **Search key concepts:**
```bash
python query_data.py "what are the main ideas about productivity?"
```

* **Generate summarised insights:**
```bash
python query_data.py "main takeaways from psychology research"
```

## Implementation Notes
* **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
* **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Summarization**: `sshleifer/distilbart-cnn-12-6`
* Processes content locally without API calls
* Special handling for quotes and text formatting

## Customization
* Modify **quote filtering rules** in `query_data.py`
* Swap models for embeddings, reranking, or summarisation

## Limitations
* Works best with text-heavy documents
* Requires Markdown format
* Initial database creation may be memory-intensive

## Inspiration
Inspired by [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I), with enhanced quote extraction and summarization.
