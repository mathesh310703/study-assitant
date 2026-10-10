"""
app.py
------
Streamlit UI for the local study assistant.

Run with:
    streamlit run app.py

Prerequisites:
    1. Ollama installed and running (https://ollama.com)
    2. Model pulled once:  ollama pull llama3.1:8b
    3. Notes ingested once (or whenever new PDFs are added):  python ingest.py
"""

import streamlit as st
from rag_engine import ask, generate_quiz

st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="centered")

st.title("📚 My Study Assistant")
st.caption("Runs fully locally — your notes and questions never leave your machine.")

tab_chat, tab_quiz = st.tabs(["Ask a question", "Generate quiz"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                src_text = ", ".join(
                    f"{s['source']} (p.{s['page']})" for s in turn["sources"]
                )
                st.caption(f"Sources: {src_text}")

    question = st.chat_input("Ask something from your notes...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(question)
            st.write(result["answer"])
            if result["sources"]:
                src_text = ", ".join(
                    f"{s['source']} (p.{s['page']})" for s in result["sources"]
                )
                st.caption(f"Sources: {src_text}")
        st.session_state.history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })

with tab_quiz:
    st.write("Generate practice questions from your notes on a specific topic.")
    topic = st.text_input("Topic (e.g. 'RANSAC', 'gradient descent', 'DBSCAN')")
    num_q = st.slider("Number of questions", min_value=3, max_value=10, value=5)

    if st.button("Generate quiz", type="primary") and topic:
        with st.spinner("Generating quiz from your notes..."):
            quiz = generate_quiz(topic, num_questions=num_q)
        st.markdown(quiz)

st.sidebar.header("Setup checklist")
st.sidebar.markdown(
    """
1. Put your lecture PDFs in the `notes/` folder
2. Run `python ingest.py` once (re-run when you add new PDFs)
3. Make sure Ollama is running locally
4. Ask away — answers are grounded in your own notes with page citations
"""
)
