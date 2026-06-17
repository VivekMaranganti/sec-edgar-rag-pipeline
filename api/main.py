import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hybrid_retriever import load_production_retriever
import ollama

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor(max_workers=2)
print("Loading index...")
retriever = load_production_retriever()
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
            lambda: retriever.invoke(query.question)
        )
        
        print(f"Search successful. Found {len(docs)} documents.")
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a financial analyst reviewing SEC 10-K filings. A colleague has asked you a question and you need to find the answer directly from the filing excerpts below. Only use what's explicitly written in the text. If you see a table, read it row by row and pull the exact number for the year being asked about. Don't estimate or do math — just find and report the figure.
Context: {context}
Question: {query.question}
Answer: """

        print("Sending payload to Ollama...")
        response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
        print("Ollama responded successfully.")

        return {
            "answer": response.message.content, 
            "sources": [doc.metadata for doc in docs]
        }

    except Exception as e:
        print(f"CATCHABLE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))