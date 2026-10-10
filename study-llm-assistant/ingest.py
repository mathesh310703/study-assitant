"""
ingest.py
---------
Reads all PDFs from the `notes/` folder, splits them into overlapping text
chunks, embeds each chunk locally with a sentence-transformers model, and
stores everything in a local ChromaDB collection.

Run this once whenever you add new lecture notes:
    python ingest.py
"""

import os
import re
import uuid
import pymupdf  # PyMuPDF (import name matches package name in current versions)
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

NOTES_DIR = "notes"
DB_DIR = "chroma_db"
COLLECTION_NAME = "study_notes"

# Chunking parameters — tuned for lecture-slide style PDFs (short, dense pages)
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap so context isn't cut mid-thought

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs fine on CPU or GPU


def extract_text_by_page(pdf_path):
    """Return a list of (page_number, text) tuples for a given PDF."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window character chunker with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks


def build_index():
    if not os.path.isdir(NOTES_DIR):
        raise FileNotFoundError(f"Put your PDF notes inside a '{NOTES_DIR}/' folder first.")

    pdf_files = [f for f in os.listdir(NOTES_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in '{NOTES_DIR}/'. Add some lecture PDFs and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to ingest.")

    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Fresh collection each run keeps things simple and avoids stale duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    all_ids, all_docs, all_metas = [], [], []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(NOTES_DIR, pdf_file)
        pages = extract_text_by_page(pdf_path)

        for page_num, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk in chunks:
                all_ids.append(str(uuid.uuid4()))
                all_docs.append(chunk)
                all_metas.append({"source": pdf_file, "page": page_num})

    if not all_docs:
        print("No extractable text found. If these are scanned/image PDFs, OCR is needed first.")
        return

    # Chroma batches internally, but we add in slices to keep memory predictable
    # on large note collections.
    BATCH = 100
    for i in range(0, len(all_docs), BATCH):
        collection.add(
            ids=all_ids[i:i + BATCH],
            documents=all_docs[i:i + BATCH],
            metadatas=all_metas[i:i + BATCH],
        )

    print(f"Indexed {len(all_docs)} chunks from {len(pdf_files)} PDF(s) into '{DB_DIR}/'.")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    build_index()
