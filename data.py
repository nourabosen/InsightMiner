from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from utils import clean_text, CleanTextLoader
import re

# --- Constants ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Initialize models (Lazy Loading) ---
class SearchEngine:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            print("⏳ Loading models... (this might take a moment)")
            cls._instance = cls()
            print("✅ Models loaded!")
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful before rebuilding DB)."""
        if cls._instance:
            print("🔄 Resetting search engine instance...")
            # Try to release resources if possible
            del cls._instance
            cls._instance = None

    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = Chroma(persist_directory="chroma_db", embedding_function=self.embedding_function)
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def rebuild(self):
        """Rebuild the database using the existing connection."""
        print("🔄 Rebuilding database...")
        
        # 1. Clear existing data
        try:
            # Try to delete the collection entirely
            self.db.delete_collection()
            print("🗑️ Collection deleted.")
        except Exception as e:
            print(f"⚠️ Warning during collection deletion: {e}")
            # Fallback: Try to delete all documents if collection deletion failed
            try:
                # Empty dictionary means match all in some versions, or might need workarounds
                # But safer to just try delete_collection. 
                # If that failed, maybe the DB connection is stale.
                pass
            except:
                pass

        # 2. Re-initialize DB (creates new collection if deleted, or reconnects)
        # Force a reload of the client if possible
        self.db = Chroma(persist_directory="chroma_db", embedding_function=self.embedding_function)

        # 3. Load and Process Documents
        print("📂 Loading documents...")
        loader = DirectoryLoader("highlights", glob="**/*.md", loader_cls=CleanTextLoader)
        docs = loader.load()

        if not docs:
            print("⚠️ No documents found in 'highlights/'")
            return

        print("✂️ Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # 4. Add to DB
        print(f"💾 Adding {len(chunks)} chunks to database...")
        self.db.add_documents(chunks)
        print("✅ Database rebuild complete!")

    def add_documents_from_files(self, file_paths):
        """Add specific files to the database, removing old versions first."""
        print(f"📂 Processing {len(file_paths)} new documents...")
        
        # 1. Remove existing chunks for these files to avoid duplicates
        for file_path in file_paths:
            source_path = str(file_path)
            try:
                # Clean up old entries for this file
                # The 'source' metadata field typically stores the relative path
                print(f"🧹 Cleaning existing entries for: {source_path}")
                self.db.delete(where={"source": source_path})
            except Exception as e:
                print(f"⚠️ Note: Could not clean existing entries for {source_path} (might be new): {e}")

        # 2. Load new content
        docs = []
        for file_path in file_paths:
            try:
                loader = CleanTextLoader(str(file_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
        
        if not docs:
            print("⚠️ No valid documents loaded.")
            return

        print("✂️ Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        print(f"💾 Adding {len(chunks)} chunks to database...")
        self.db.add_documents(chunks)
        print("✅ Added new documents successfully!")

# --- Helper functions ---
def format_content(content, width=80):
    """Format text nicely with one quote per line."""
    if not content:
        return ""

    # Remove unwanted artifacts
    content = clean_text(content)
    content = re.sub(r"\bQuotes\b", "", content)

    # Split quotes into separate lines
    quotes = re.findall(r'"([^"]+)"', content)
    if quotes:
        cleaned_quotes = []
        seen = set()
        for q in quotes:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                if not q.endswith((".", "!", "?")):
                    q += "."
                q = q[0].upper() + q[1:]
                cleaned_quotes.append(q)
        return "\n".join(cleaned_quotes)

    # Fallback: split into paragraphs
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    lines = []
    for para in paragraphs:
        words = para.split()
        line = []
        for word in words:
            if len(" ".join(line + [word])) <= width:
                line.append(word)
            else:
                lines.append(" ".join(line))
                line = [word]
        if line:
            lines.append(" ".join(line))
        lines.append("")

    return "\n".join(lines)


def extract_and_filter_quotes(text):
    """Extract and filter quotes from text."""
    cleaned = clean_text(str(text))
    if not cleaned:
        return []

    quotes = re.findall(r'"([^"]+)"', cleaned)
    quotes += re.findall(r"•\s*(.+?)(?=\n|$)", cleaned)

    valid_quotes = []
    for q in quotes:
        q = q.strip()
        words = q.split()

        if (
            (6 <= len(words) <= 40)
            and not q.lower().startswith(("chapter", "section", "note"))
            and not q.endswith((":", "-", "..."))
            and not any(word.isupper() for word in words[:3])
        ):
            q = q.rstrip(",.;")
            valid_quotes.append(q)

    return valid_quotes


def generate_quality_insight(texts, query):
    """Generate clean, coherent, relevant insights from sources."""
    if not texts or not query:
        return "🔍 No content provided for summarization."

    valid_quotes = []
    for t in texts:
        valid_quotes.extend(extract_and_filter_quotes(t))

    if not valid_quotes:
        return "🔍 No strong insights found in the sources."

    seen = set()
    unique_quotes = []
    for q in valid_quotes:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_quotes.append(q)

    if len(unique_quotes) > 5:
        try:
            engine = SearchEngine.get()
            pairs = [(query, q) for q in unique_quotes]
            scores = engine.reranker.predict(pairs)
            scored_quotes = sorted(
                zip(unique_quotes, scores), key=lambda x: x[1], reverse=True
            )
            top_quotes = [q for q, _ in scored_quotes[:5]]
        except Exception as e:
            print(f"⚠️ Reranking error: {str(e)}")
            top_quotes = unique_quotes[:5]
    else:
        top_quotes = unique_quotes

    insight = f"✨ Top Insights About '{query}' ✨\n\n"
    for i, q in enumerate(top_quotes, 1):
        q = re.sub(r"\s+", " ", q).strip()
        if q and not q.endswith((".", "!", "?")):
            q += "."
        q = q[0].upper() + q[1:]
        insight += f"{i}. {q}\n"

    if len(top_quotes) >= 2:
        insight += "\n💡 Which of these resonates most with you?"

    return insight


def search_database(query, top_k=5, book_filter=None):
    """Retrieve top results and generate insights via return values."""
    try:
        engine = SearchEngine.get()
        
        filter_dict = None
        if book_filter:
            # Construct the source path as expected in metadata
            # usually "highlights/Filename.md"
            sources = [f"highlights/{b}" for b in book_filter]
            if len(sources) == 1:
                filter_dict = {"source": sources[0]}
            else:
                filter_dict = {"source": {"$in": sources}}
            
        results = engine.db.similarity_search_with_score(query, k=top_k * 3, filter=filter_dict)
    except Exception as e:
        return [], f"❌ Search error: {e}"

    processed = []
    for doc, score in results:
        content = doc.page_content
        if content and len(clean_text(content).split()) > 10:
            source = doc.metadata.get("source", "unknown")
            filename = source.split("/")[-1].replace(".md", "").replace("Book ", "")
            processed.append(
                {"content": content, "raw_score": float(score), "source": filename}
            )

    if not processed:
        return [], "🔍 No quality results found"

    if len(processed) > 2:
        pairs = [(query, r["content"]) for r in processed]
        rerank_scores = engine.reranker.predict(pairs)
        min_s, max_s = min(rerank_scores), max(rerank_scores)
        for i, score in enumerate(rerank_scores):
            processed[i]["score"] = (
                0.1 + 0.9 * ((score - min_s) / (max_s - min_s))
                if max_s > min_s
                else 0.5
            )
        processed.sort(key=lambda x: x["score"], reverse=True)
    else:
        for r in processed:
            r["score"] = r["raw_score"]

    top_results = processed[:top_k]
    insight = generate_quality_insight([r["content"] for r in top_results], query)
    
    return top_results, insight


def query_database(query, top_k=5):
    """CLI wrapper for search_database."""
    results, insight = search_database(query, top_k)
    
    if not results:
        print(insight) # Will contain error or "no results" message
        return

    print(f"\n🔍 Top {len(results)} Results for: '{query}'\n")
    for i, res in enumerate(results, 1):
        print(f"📌 Result {i} (Relevance: {res['score']:.2f})")
        print(f"📚 Source: {res['source']}")
        print("━" * 80)
        print(format_content(res["content"]))
        print()

    print("═" * 80)
    print("✨ Insights ✨".center(80))
    print("═" * 80)
    print(insight)
    print("\n" + "═" * 80 + "\n")
