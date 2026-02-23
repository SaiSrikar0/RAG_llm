"""Streamlit frontend for a strict Retrieval-Augmented Generation (RAG) app."""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.embeddings import EmbeddingService
from src.qa import AnswerGenerator
from src.retriever import parse_results
from src.utils import chunk_text, read_pdf_bytes, read_txt_bytes
from src.vector_store import VectorStore

st.set_page_config(page_title="Strict RAG Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def get_services() -> tuple[EmbeddingService, VectorStore, AnswerGenerator]:
    embedder = EmbeddingService()
    store = VectorStore()
    answerer = AnswerGenerator()
    return embedder, store, answerer


def ingest_text(source_name: str, text: str) -> int:
    embedder, store, _ = get_services()
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    embeddings = embedder.embed_documents(chunks)
    store.add_chunks(chunks, embeddings, source=source_name)
    return len(chunks)


def ingest_uploaded_file(uploaded_file) -> int:
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        text = read_pdf_bytes(data)
    else:
        text = read_txt_bytes(data)
    return ingest_text(uploaded_file.name, text)


def ask_question(question: str) -> tuple[str, float, str, str]:
    embedder, store, answerer = get_services()
    query_vector = embedder.embed_query(question)
    raw = store.query(query_embedding=query_vector, top_k=settings.top_k)
    result = parse_results(raw)

    if result.category == "Not Found":
        return (
            "The information for your question is not present in the provided data.",
            result.score,
            result.category,
            "",
        )

    answer = answerer.answer_from_context(question, result.context_chunks)
    context_preview = "\n\n".join(result.context_chunks)

    if result.category == "Similar":
        answer = f"Similar results are given below to your question.\n\n{answer}"

    return answer, result.score, result.category, context_preview


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_state()

st.title("📚 Strict RAG Chat Assistant")
st.caption("Upload your data, then ask questions. Answers are constrained to retrieved context only.")

with st.sidebar:
    st.header("Data Upload")
    st.write("Upload .txt or .pdf files and/or paste raw text.")

    if st.button("Reset Knowledge Base", use_container_width=True):
        _, store, _ = get_services()
        store.reset_collection()
        st.session_state.messages = []
        st.success("Knowledge base reset successfully.")

    files = st.file_uploader(
        "Upload files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    pasted_text = st.text_area("Or paste custom text", height=180)

    if st.button("Process Data", type="primary", use_container_width=True):
        processed = 0
        try:
            if files:
                for file in files:
                    processed += ingest_uploaded_file(file)
            if pasted_text.strip():
                processed += ingest_text("pasted_text", pasted_text)
        except Exception as exc:
            st.error(f"Data ingestion failed: {exc}")
        else:
            if processed:
                st.success(f"Processed and stored {processed} chunks in ChromaDB.")
            else:
                st.warning("No input provided. Please upload a file or paste text.")

    _, store, _ = get_services()
    st.metric("Stored chunks", store.count())

st.subheader("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Ask a question grounded in your uploaded data...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    _, store, _ = get_services()
    if store.count() == 0:
        output = "Please upload or paste data first so I can retrieve grounded context."
        score = 0.0
        category = "Not Found"
        context_preview = ""
    else:
        with st.spinner("Retrieving context and generating grounded answer..."):
            try:
                output, score, category, context_preview = ask_question(prompt)
            except Exception as exc:
                output = f"Question handling failed: {exc}"
                score = 0.0
                category = "Not Found"
                context_preview = ""

    similarity_label = f"Similarity Score: {score:.2f} | Category: {category}"
    st.session_state.messages.append(
        {"role": "assistant", "content": output, "meta": similarity_label}
    )

    with st.chat_message("assistant"):
        st.markdown(output)
        st.caption(similarity_label)
        if context_preview:
            with st.expander("Retrieved context"):
                st.write(context_preview)
