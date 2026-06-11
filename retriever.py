from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_filings
from pathlib import Path

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "embeddings/nvda_index"

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "])
    chunks = []
    for doc in docs:
        splits = splitter.create_documents([doc["text"]], metadatas=[{"ticker": doc["ticker"], "filename": doc["filename"], "filing_date": doc["filing_date"]}])
        chunks.extend(splits)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_index(chunks):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Building FAISS index...")
    index = FAISS.from_documents(chunks, embeddings)
    Path("embeddings").mkdir(exist_ok=True)
    index.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}")
    return index

def load_index():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    docs = load_filings("NVDA")
    chunks = chunk_docs(docs)
    build_index(chunks)