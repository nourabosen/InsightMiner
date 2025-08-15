from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load documents
loader = DirectoryLoader("highlights", glob="**/*.md")
docs = loader.load()

# 2. Semantic chunking for better context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(docs)

# 3. Local embedding model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 4. Create Chroma DB
db = Chroma.from_documents(chunks, embedding_function, persist_directory="chroma_db")
db.persist()

print(f"✅ Database created with {len(chunks)} chunks")