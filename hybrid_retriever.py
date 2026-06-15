from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from ingest import load_filings
from pathlib import Path
import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "embeddings/nvda_index"

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for doc in docs:
        splits = splitter.create_documents(
            [doc["text"]],
            metadatas=[{
                "ticker": doc["ticker"],
                "filename": doc["filename"],
                "filing_date": doc["filing_date"],
                "section": doc["section"]
            }]
        )
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
    return index, chunks

def load_index():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def hybrid_search(query, index, all_chunks, k=5):
    # Vector search
    vector_docs = index.similarity_search(query, k=k*2)
    
    # BM25 keyword search
    tokenized_chunks = [chunk.page_content.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_docs = [all_chunks[i] for i in top_bm25_indices]
    
    # Merge and deduplicate
    seen = set()
    merged = []
    for doc in vector_docs + bm25_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    
    return merged[:k]

if __name__ == "__main__":
    docs = load_filings("NVDA")
    chunks = chunk_docs(docs)
    index, _ = build_index(chunks)