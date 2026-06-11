import os
from sec_edgar_downloader import Downloader
from pdfplumber import open as plumber_open
from pathlib import Path

dl = Downloader("Vivek Maranganti", "vivek.maranganti@gmail.com", "data")

def download_filings(ticker, form_type="10-K", limit=3):
    print(f"Downloading {ticker} {form_type} filings...")
    dl.get(form_type, ticker, limit=limit)
    print("Done.")

def extract_text(filepath):
    text = ""
    if filepath.suffix == ".pdf":
        with plumber_open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text

def load_filings(ticker, form_type="10-K"):
    docs = []
    base = Path(f"data/sec-edgar-filings/{ticker}/{form_type}")

    if not base.exists():
        print(f"Nothing at {base} — run download_filings first")
        return docs

    for filing_dir in base.iterdir():
        for filepath in filing_dir.iterdir():
            if filepath.suffix in [".htm", ".txt", ".pdf"]:
                print(f"Reading {filepath.name}...")
                text = extract_text(filepath)
                if text.strip():
                    docs.append({
                        "ticker": ticker,
                        "form": form_type,
                        "filename": filepath.name,
                        "text": text
                    })
    return docs

if __name__ == "__main__":
    download_filings("NVDA", limit=3)
    docs = load_filings("NVDA")
    print(f"\nGot {len(docs)} docs")
    for doc in docs:
        print(f"  {doc['filename']} — {len(doc['text'])} chars")