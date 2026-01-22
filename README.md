This project implements a Retrieval-Augmented Generation (RAG) system for searching and summarising content from a collection of documents, with a focus on extracting meaningful quotes and insights.

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

### 3. `streamlit run streamlit_app.py`

* Launches a modern web interface
* Provides visual search results and insights
* **Incremental Updates**: Upload new books and add them instantly without rebuilding the whole database
* **Full Rebuild**: Option to completely reset and re-index the library

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

   **Option A: CLI**
   ```bash
   python run.py query
   ```

   **Option B: Modern UI**
   ```bash
   streamlit run streamlit_app.py
   ```
   *Or use the Makefile:* `make run`

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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To get the right things done, choosing what to ignore is as important as choosing where to focus.
You can be relatively certain that if you decide when and where you’re going to do those things, you’ll actually, reliably and predictably, get them done.
...
📌 Result 5 (Relevance: 0.53)
📚 Source: Feel-Good Productivity
════════════════════════════════════════════════
✨ Top Insights About 'how to be productive' ✨

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
📚 Source: As a Man Thinketh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A blessed life is the sure result of right thought.
The power of thought multiplies through focus and concentration.
As a person strives and fails, again and again, in the service of a worthy goal, their character is strengthened and deepened. They turn obstacles into steppingstones. This is the measure of true success.
...
📌 Result 5 (Relevance: 0.52)
📚 Source: Designing Your Life
════════════════════════════════════════════════
✨ Top Insights About 'how to be creative' ✨

1. Allied with purpose, your every thought is energized, giving you the courage to face and overcome any obstacle. Purposedriven thought is a creative force.
2. So be a genius at your life design; just don’t think you have to be one of those lone geniuses.
3. Artists should focus on mastering their own territory or turf—the place where they are experts, or have control or mastery.
4. Work is fun when you are actually leaning into your strengths and are deeply engaged and energized by what you’re doing.
5. And if you accept this idea—that there are multiple great designs for your life, though you’ll still only get to live one—it is rather liberating.
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
