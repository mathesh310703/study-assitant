# Study Assistant — Local RAG-based Study Tool

A fully local, offline-capable study assistant. Point it at your lecture PDFs,
ask questions in plain English, and get answers grounded in your own notes
with page-level citations — no cloud API, no internet dependency, no cost.

Built as a personal tool to study faster by querying notes directly instead
of manually searching through PDFs, and doubles as a demonstration of
practical, hands-on LLM engineering (local model deployment, retrieval-
augmented generation, and workflow automation) rather than just using a
chat interface.

---

## How it works

```
notes/*.pdf
     │
     ▼
[ingest.py]  → extract text per page → chunk with overlap → embed locally
     │                                                    (sentence-transformers)
     ▼
[chroma_db/]  ← persistent local vector store (ChromaDB)
     │
     ▼
[rag_engine.py]  ← question → embed → retrieve top-k chunks → build grounded
     │              prompt → send to local LLM (Ollama)
     ▼
[app.py]  ← Streamlit chat UI with source citations + quiz generator
```

Everything — embeddings, vector search, and text generation — runs on your
own machine. Nothing is sent to an external API.

---

## Setup

### 1. Install Ollama and pull a model
Download Ollama from https://ollama.com, then:
```bash
ollama pull llama3.1:8b
```
(Any Ollama-supported model works — swap the `LLM_MODEL` constant in
`rag_engine.py` if you prefer a smaller/faster model like `phi3:mini`.)

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your notes
Drop your lecture PDFs into the `notes/` folder.

### 4. Build the index
```bash
python ingest.py
```
Re-run this any time you add new PDFs.

### 5. Launch the app
```bash
streamlit run app.py
```

---

## Features

- **Grounded Q&A** — answers are generated only from retrieved note content,
  with the LLM instructed to say "not enough information" rather than
  hallucinate, and every answer shows which PDF/page it came from.
- **Quiz generator** — generate exam-style practice questions on any topic
  covered in your notes, useful for active recall instead of passive re-reading.
- **Fully local** — works offline once the model is pulled; no API keys, no
  per-query cost, no data leaving your machine.
- **Swappable model** — change `LLM_MODEL` in `rag_engine.py` to trade off
  speed vs. quality depending on your hardware.

---

## Design notes / things I'd improve with more time

- **Chunking** currently uses a simple fixed-size sliding window. A
  layout-aware chunker (splitting on slide boundaries or headings) would
  improve retrieval quality for slide-deck-style PDFs.
- **Evaluation**: currently qualitative only. A proper next step would be a
  small labeled Q&A set to measure retrieval precision and answer accuracy
  across different embedding models / chunk sizes.
- **Scanned/handwritten PDFs** aren't supported yet — would need an OCR step
  (e.g. Tesseract) before chunking.
- **Multi-model benchmarking**: comparing 2–3 local models (e.g. Llama 3.1 8B
  vs. Phi-3-mini vs. Mistral 7B) on latency and answer quality would be a
  good extension for a more rigorous writeup.

---

## Tech stack

Python · PyMuPDF (PDF parsing) · sentence-transformers (local embeddings) ·
ChromaDB (local vector store) · Ollama (local LLM inference) · Streamlit (UI)
