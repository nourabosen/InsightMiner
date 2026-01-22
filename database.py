from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils import clean_text
from shutil import rmtree
import time
import os


# --- Load documents ---
class CleanTextLoader(TextLoader):
    def load(self):
        docs = super().load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        return docs


def clean_database():
    if os.path.exists("chroma_db"):
        try:
            rmtree("chroma_db")
        except OSError as e:
            print(f"⚠️ Error cleaning DB: {e}. Retrying...")
            time.sleep(1)
            try:
                rmtree("chroma_db")
            except OSError:
                print("❌ Failed to fully clean DB. Attempting to proceed anyway.")


def create_database():
    loader = DirectoryLoader("highlights", glob="**/*.md", loader_cls=CleanTextLoader)
    docs = loader.load()

    # --- Semantic chunking for better context ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # --- Local embedding model ---
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # --- Create Chroma DB ---
    Chroma.from_documents(chunks, embedding_function, persist_directory="chroma_db")

    print(f"✅ Database created with {len(chunks)} chunks")
