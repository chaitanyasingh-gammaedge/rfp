# ingest.py
import os, math
import pdfplumber
import docx
from vector_store import VectorStore
from tqdm import tqdm

def read_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=800, overlap=200):
    # chunk_size in characters; simple splitter
    start = 0
    L = len(text)
    chunks = []
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def ingest_file(path, store: VectorStore, source_name=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = read_pdf(path)
    elif ext in [".docx", ".doc"]:
        text = read_docx(path)
    else:
        text = read_txt(path)
    chunks = chunk_text(text)
    metadatas = []
    for i, ch in enumerate(chunks):
        metadatas.append({
            "source": source_name or os.path.basename(path),
            "chunk_id": i,
            "text": ch
        })
    store.add_texts(chunks, metadatas)
    print(f"Indexed {len(chunks)} chunks from {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--index", default="faiss.index")
    args = parser.parse_args()
    store = VectorStore(index_path=args.index)
    for f in args.files:
        ingest_file(f, store)
