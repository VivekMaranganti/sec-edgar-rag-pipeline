import os
from sec_edgar_downloader import Downloader
from pdfplumber import open as plumber_open
from pathlib import Path
from bs4 import BeautifulSoup
import re
from docling.document_converter import DocumentConverter

def strip_html(text):
    return BeautifulSoup(text, "lxml").get_text(separator=" ")

converter = DocumentConverter()
dl = Downloader("Vivek Maranganti", "vivek.maranganti@gmail.com", "data")

def download_filings(ticker, form_type="10-K", limit=3):
    print(f"Downloading {ticker} {form_type} filings...")
    dl.get(form_type, ticker, limit=limit)
    print("Done.")

def extract_text(filepath):
    if filepath.suffix == ".pdf" or filepath.suffix in [".htm", ".txt"]:
        try:
            result = converter.convert(str(filepath))
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Docling failed on {filepath.name}: {e}, falling back to plain text")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return strip_html(f.read())
    return ""

def extract_sections(text):
    """Split SEC filing into meaningful sections by Item headers."""
    pattern = r'(Item\s+\d+[A-Za-z]?\.?\s+[A-Z][A-Z\s,]+)'
    splits = re.split(pattern, text, flags=re.IGNORECASE)
    
    sections = []
    for i in range(1, len(splits), 2):
        header = splits[i].strip()
        content = splits[i+1].strip() if i+1 < len(splits) else ""
        if len(content) > 200:
            sections.append({
                "header": header,
                "content": content
            })
    return sections

def load_filings(ticker, form_type="10-K"):
    docs = []
    base = Path(f"data/sec-edgar-filings/{ticker}/{form_type}")

    if not base.exists():
        print(f"Nothing at {base} — run download_filings first")
        return docs

    for filing_dir in base.iterdir():
        for filepath in filing_dir.iterdir():
            if filepath.suffix in [".htm", ".txt", ".pdf"]:
                print(f"Reading {filepath.name} from {filing_dir.name}...")
                text = extract_text(filepath)
                sections = extract_sections(text)
                print(f"Found {len(sections)} sections")
                for section in sections:
                    docs.append({
                        "ticker": ticker,
                        "form": form_type,
                        "filename": filepath.name,
                        "filing_date": filing_dir.name,
                        "section": section["header"],
                        "text": section["content"]
                    })
    return docs

if __name__ == "__main__":
    download_filings("NVDA", limit=3)
    docs = load_filings("NVDA")
    print(f"\nGot {len(docs)} docs")
    for doc in docs:
        print(f"  {doc['filename']} — {len(doc['text'])} chars")