import streamlit as st
import os
import shutil
from pathlib import Path
import warnings
import gc

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="InsightMiner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI (Dark/Light mode compatible)
st.markdown("""
<style>
    .card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .source-tag {
        background-color: rgba(128, 128, 128, 0.2);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-color);
        display: inline-block;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .score-tag {
        float: right;
        color: #10b981;
        font-weight: bold;
    }
    .insight-box {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📚 InsightMiner")
st.sidebar.markdown("Explore your reading highlights insightfully.")

highlights_dir = Path("highlights")
highlights_dir.mkdir(exist_ok=True)

# 1. Add New Content
st.sidebar.header("📥 Add New Content")
uploaded_files = st.sidebar.file_uploader("Upload Book Highlights (.md)", accept_multiple_files=True, type=['md'])

if uploaded_files:
    if st.sidebar.button("Add to Library"):
        with st.spinner("Processing and adding to database..."):
            try:
                new_file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = highlights_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    new_file_paths.append(file_path)
                
                import data
                engine = data.SearchEngine.get()
                engine.add_documents_from_files(new_file_paths)
                
                st.sidebar.success(f"Successfully added {len(new_file_paths)} books!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error adding files: {e}")

st.sidebar.divider()

# 2. Database Management
st.sidebar.header("⚙️ Database Management")

# Statistics
if highlights_dir.exists():
    num_books = len(list(highlights_dir.glob("*.md")))
    st.sidebar.metric("Books Indexed", num_books)
else:
    st.sidebar.warning("No highlights folder found!")

if st.sidebar.button("♻️ Full Rebuild", help="Clear and re-process ALL markdown files"):
    with st.spinner("Rebuilding database from scratch..."):
        try:
            import data
            engine = data.SearchEngine.get()
            engine.rebuild()
            st.sidebar.success("Database fully rebuilt!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error rebuilding database: {e}")

# --- Main Content ---

st.title("🧠 Mining...")

# Lazy load the search engine to avoid UI freeze on startup if possible
@st.cache_resource
def get_search_engine():
    try:
        import data
        # Ensure models are loaded
        data.SearchEngine.get()
        return data
    except Exception as e:
        print(f"Error loading engine: {e}")
        return None

if not Path("chroma_db").exists():
    st.info("👋 Welcome! It looks like your database hasn't been created yet.")
    st.info("Please click '♻️ Rebuild Database' in the sidebar to process your highlights.")
else:
    data_module = get_search_engine()

    if not data_module:
        st.error("Could not load the search engine. Check logs for details.")
    else:
        # Book Filter
        available_books = [f.name for f in list(highlights_dir.glob("*.md"))]
        selected_books = st.multiselect("Filter by Book (Optional)", options=available_books, placeholder="Select books to search within...")
        
        query = st.text_input("Ask a question to your library...", placeholder="e.g., How to build better habits?")

        if query:
            with st.spinner("Thinking..."):
                try:
                    # search_database returns (results, insight_string)
                    results, insight = data_module.search_database(query, book_filter=selected_books)
                    
                    if not results:
                        st.warning("No results found. Try a different query.")
                    else:
                        # Display Insights
                        st.markdown(f"""
                    <div class="insight-box">
                        <h3>✨ Insights</h3>
                        {insight.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 🔍 Relevant Highlights")
                    
                    for res in results:
                        source = res['source']
                        score = res['score']
                        content = res['content']
                        
                        st.markdown(f"""
                        <div class="card">
                            <span class="source-tag">📖 {source}</span>
                            <span class="score-tag">Match: {int(score * 100)}%</span>
                            <div style="margin-top: 10px; line-height: 1.6;">
                                {content.replace(chr(10), '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during search: {e}")

st.markdown("---")
st.markdown("InsightMiner © 2026")
