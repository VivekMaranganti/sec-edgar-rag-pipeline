from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import load_filings
import pickle
from pathlib import Path

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "embeddings/nvda_index"
BM25_PATH = "embeddings/bm25_retriever.pkl"

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", " ", ""]
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

def build_hybrid_retriever(chunks):
    Path("embeddings").mkdir(exist_ok=True)
    
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Building FAISS index...")
    faiss_db = FAISS.from_documents(chunks, embeddings)
    faiss_db.save_local(INDEX_PATH)
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
    
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def load_production_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    faiss_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
    
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )

if __name__ == "__main__":
    docs = load_filings("NVDA")
    chunks = chunk_docs(docs)
    retriever = build_hybrid_retriever(chunks)
    print("Hybrid production retriever compiled successfully.")