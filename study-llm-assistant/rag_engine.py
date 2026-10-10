"""
rag_engine.py
-------------
Core retrieval-augmented generation logic:
  1. Embed the user's question with the same local embedding model used at
     ingest time.
  2. Retrieve the most relevant note chunks from ChromaDB.
  3. Build a grounded prompt and send it to a locally running LLM (via Ollama).
  4. Return the answer plus the source chunks used, so answers are always
     traceable back to the actual PDF/page they came from.

This file has no UI code — app.py imports these functions.
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama

DB_DIR = "chroma_db"
COLLECTION_NAME = "study_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Change this to whatever model you've pulled with `ollama pull <model>`.
# llama3.1:8b is a good default for a laptop with a dedicated GPU.
LLM_MODEL = "llama3.1:8b"

TOP_K = 5  # how many chunks to retrieve per question


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_chunks(question, top_k=TOP_K):
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "distance": dist,
        })
    return chunks


def build_prompt(question, chunks):
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a study assistant. Answer the question using ONLY the
context below, which comes from the student's own lecture notes. If the
context does not contain enough information to answer, say so clearly instead
of guessing. Keep the answer concise and exam-relevant. When useful, mention
which source/page the information came from.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return prompt


def ask(question, top_k=TOP_K, model=LLM_MODEL):
    """Full RAG pipeline: retrieve -> build prompt -> query local LLM."""
    chunks = retrieve_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No notes have been indexed yet. Run `python ingest.py` first.",
            "sources": [],
        }

    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
    }


def generate_quiz(topic, num_questions=5, model=LLM_MODEL):
    """
    Bonus automation feature: auto-generate quiz questions from the notes
    relevant to a topic, instead of just answering a direct question.
    """
    chunks = retrieve_chunks(topic, top_k=8)
    if not chunks:
        return "No notes have been indexed yet. Run `python ingest.py` first."

    context = "\n\n---\n\n".join(c["text"] for c in chunks)

    prompt = f"""Using ONLY the context below from the student's lecture notes,
write {num_questions} exam-style quiz questions with answers on the topic
"{topic}". Format each as:

Q: <question>
A: <answer>

CONTEXT:
{context}
"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
