import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import load_index
import ollama
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor(max_workers=2)
print("Loading index...")
index = load_index()
print("Index loaded.")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    try:
        print(f"--- Starting similarity search for: {query.question} ---")
        
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(
            executor, 
            lambda: index.similarity_search(query.question, k=5)
        )
        
        print(f"Search successful. Found {len(docs)} documents.")
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a financial analyst assistant. Use the following excerpts from SEC 10-K filings to answer the question. Be specific and cite numbers where possible. 
Context: {context}
Question: {query.question}
Answer:"""

        print("Sending payload to Ollama...")
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        print("Ollama responded successfully.")

        return {
            "answer": response['message']['content'], 
            "sources": [doc.metadata for doc in docs]
        }

    except Exception as e:
        print(f"CATCHABLE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))