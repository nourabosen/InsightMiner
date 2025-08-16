from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import re
import sys

# --- Constants ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Initialize models ---
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
reranker = CrossEncoder(RERANKER_MODEL)

# --- Helper functions ---
def clean_text(text):
    """Clean text but preserve quotes."""
    if not text or not isinstance(text, str):
        return ""
    
    if "## Quotes" in text:
        return text
    
    text = re.sub(r'background-color:: \w+', '', text)
    text = re.sub(r'==|[\*\_\[\]\(\)#-]', '', text)
    text = re.sub(r'(\w) "(\w)', r'\1 "\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_content(content, width=80):
    """Format text nicely with one quote per line."""
    if not content:
        return ""
    
    # Remove unwanted artifacts
    content = re.sub(r'background-color:: \w+', '', content)
    content = re.sub(r'\bQuotes\b', '', content)
    content = re.sub(r'==|[\*\_\[\]\(\)#-]', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Split quotes into separate lines
    quotes = re.findall(r'"([^"]+)"', content)
    if quotes:
        cleaned_quotes = []
        seen = set()
        for q in quotes:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                if not q.endswith(('.', '!', '?')):
                    q += '.'
                q = q[0].upper() + q[1:]
                cleaned_quotes.append(q)
        return "\n".join(cleaned_quotes)
    
    # Fallback: split into paragraphs
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    lines = []
    for para in paragraphs:
        words = para.split()
        line = []
        for word in words:
            if len(' '.join(line + [word])) <= width:
                line.append(word)
            else:
                lines.append(' '.join(line))
                line = [word]
        if line:
            lines.append(' '.join(line))
        lines.append('')
    
    return "\n".join(lines)

def extract_and_filter_quotes(text):
    """Extract and filter quotes from text."""
    cleaned = clean_text(str(text))
    if not cleaned:
        return []
    
    quotes = re.findall(r'"([^"]+)"', cleaned)
    quotes += re.findall(r'•\s*(.+?)(?=\n|$)', cleaned)
    
    valid_quotes = []
    for q in quotes:
        q = q.strip()
        words = q.split()
        
        if (6 <= len(words) <= 40) and \
           not q.lower().startswith(("chapter", "section", "note")) and \
           not q.endswith((":", "-", "...")) and \
           not any(word.isupper() for word in words[:3]):
            q = q.rstrip(",.;")
            valid_quotes.append(q)
    
    return valid_quotes

def generate_quality_summary(texts, query):
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
            pairs = [(query, q) for q in unique_quotes]
            scores = reranker.predict(pairs)
            scored_quotes = sorted(zip(unique_quotes, scores), key=lambda x: x[1], reverse=True)
            top_quotes = [q for q, _ in scored_quotes[:5]]
        except Exception as e:
            print(f"⚠️ Reranking error: {str(e)}")
            top_quotes = unique_quotes[:5]
    else:
        top_quotes = unique_quotes

    summary = f"✨ Top Insights About '{query}' ✨\n\n"
    for i, q in enumerate(top_quotes, 1):
        q = re.sub(r'\s+', ' ', q).strip()
        if q and not q.endswith(('.', '!', '?')):
            q += '.'
        q = q[0].upper() + q[1:]
        summary += f"{i}. {q}\n"

    if len(top_quotes) >= 2:
        summary += "\n💡 Which of these resonates most with you?"

    return summary

def query_database(query, top_k=5):
    """Retrieve top results and generate a summary."""
    try:
        results = db.similarity_search_with_score(query, k=top_k*3)
    except Exception as e:
        print(f"❌ Search error: {e}")
        return
    
    processed = []
    for doc, score in results:
        content = doc.page_content
        if content and len(clean_text(content).split()) > 10:
            source = doc.metadata.get('source', 'unknown')
            filename = source.split('/')[-1].replace('.md', '').replace('Book ', '')
            processed.append({'content': content, 'raw_score': float(score), 'source': filename})
    
    if not processed:
        print("🔍 No quality results found")
        return
    
    if len(processed) > 2:
        pairs = [(query, r['content']) for r in processed]
        rerank_scores = reranker.predict(pairs)
        min_s, max_s = min(rerank_scores), max(rerank_scores)
        for i, score in enumerate(rerank_scores):
            processed[i]['score'] = 0.1 + 0.9 * ((score - min_s) / (max_s - min_s)) if max_s > min_s else 0.5
        processed.sort(key=lambda x: x['score'], reverse=True)
    else:
        for r in processed:
            r['score'] = r['raw_score']
    
    print(f"\n🔍 Top {min(top_k, len(processed))} Results for: '{query}'\n")
    for i, res in enumerate(processed[:top_k], 1):
        print(f"📌 Result {i} (Relevance: {res['score']:.2f})")
        print(f"📚 Source: {res['source']}")
        print("━" * 80)
        print(format_content(res['content']))
        print()
    
    print("═" * 80)
    print("✨ Summary ✨".center(80))
    print("═" * 80)
    summary = generate_quality_summary([r['content'] for r in processed[:top_k]], query)
    print(summary)
    print("\n" + "═" * 80 + "\n")

# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_data.py \"your question\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    query_database(query)
