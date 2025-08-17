This project implements a Retrieval-Augmented Generation (RAG) system for searching and summarising content from a collection of documents, with a focus on extracting meaningful quotes and insights.

> **Note:** I’ve included a `highlights/` folder for your convenience—place your Markdown files there to get started.

## Features

* **Semantic Search**: Retrieve the most relevant content using vector embeddings.
* **Intelligent Chunking**: Break documents into context-aware chunks for better retrieval.
* **Quote Extraction**: Automatically identifies meaningful quotes and key points.
* **Hybrid Relevance Ranking**: Combines embeddings with cross-encoder reranking for high-quality results.
* **Insight Summarisation**: Generates coherent summaries from multiple sources.
* **Fully Local**: Runs entirely on your machine using HuggingFace models.

## Components

### 1. `python run.py database create`

* Loads Markdown files from `highlights/`
* Splits text into semantic chunks
* Generates embeddings with `sentence-transformers/all-mpnet-base-v2`
* Stores vectors in ChromaDB for efficient local retrieval

### 2. `python run.py query`

* Performs semantic search with optional hybrid reranking
* Extracts and filters high-quality quotes
* Generates concise, context-aware summaries
* Formats output for readability

## Setup

1. Create Python Environment:

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Install `libmagic` depending on your machine:

```bash
brew install libmagic # for mac
sudo apt install libmagic # debian-based linux
```

4. Place Markdown documents in the `highlights/` directory (included for convenience).

5. Create the vector database:

```bash
python run.py database create
```

6. Query the knowledge base:

## Usage Examples

### Example 1: Searching for productivity insights

```bash
python run.py query "how to be productive"
# Or just: python run.py query   (for interactive mode)
```

**Output:**

```
🔍 Top 5 Results for: 'how to be productive'

📌 Result 1 (Relevance: 1.00)
📚 Source: 18 Minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To get the right things done, choosing what to ignore is as important as choosing where to focus.
You can be relatively certain that if you decide when and where you’re going to do those things, you’ll actually, reliably and predictably, get them done.
...
📌 Result 5 (Relevance: 0.51)
📚 Source: Feel-Good Productivity
════════════════════════════════════════════════
✨ Insights ✨
1. The less distracted you are, the more productive you’ll be.
2. What can you realistically accomplish that will further your focus for the year and allow you to leave at the end of the day feeling that you’ve been productive and successful?
3. My life has changed. These days, I know that productivity isn’t about discipline; it’s about doing more of what makes you feel happier, less stressed, more energised.
4. Productivity can be achieved only through imperfection.
5. You need to have a reason for doing something in order to make performing that task worth your time and effort.
```

### Example 2: Searching for creativity insights

```bash
python run.py query "how to be creative"
```

**Output:**

```
🔍 Top 5 Results for: 'how to be creative'

📌 Result 1 (Relevance: 1.00)
📚 Source: Designing Your Life
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
And if you accept this idea—that there are multiple great designs for your life, though you’ll still only get to live one—it is rather liberating.
Do not fall in love with your first idea.
...
📌 Result 5 (Relevance: 0.74)
📚 Source: 18 Minutes
════════════════════════════════════════════════
✨ Insights ✨
1. So be a genius at your life design; just don’t think you have to be one of those lone geniuses.
2. Artists should focus on mastering their own territory or turf—the place where they are experts, or have control or mastery.
3. Choice is yours: either master your mind to create the life you want, or remain mired in frustration and failure.
4. Realize that “we are the makers of ourselves” through the power of the thoughts we choose and encourage.
5. To home in on your passion, think about what you love doing—what’s important enough to you that you’re willing to persist over the year, even when it feels like you’re not succeeding at it.
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
