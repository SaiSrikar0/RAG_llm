# Strict RAG Streamlit App

A production-style Retrieval-Augmented Generation (RAG) app with a Streamlit frontend, ChromaDB vector store, configurable embeddings (SentenceTransformers or OpenAI), and strict context-grounded answers.

## Features

- Upload `.txt` and `.pdf` files from the sidebar.
- Paste custom text directly in the sidebar.
- Chunk and embed content, then store it in ChromaDB.
- Ask questions in a chat-style interface.
- Similarity threshold policy:
  - `>= 0.90` → **High Match**
  - `0.50 - 0.89` → **Similar** and prefixed advisory message
  - `< 0.50` → **Not Found**
- Guardrails to avoid hallucination: answers are instructed to use retrieved context only.

## Project Structure

```text
.
├── app.py
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── qa.py
│   ├── retriever.py
│   ├── utils.py
│   └── vector_store.py
├── data/
│   └── sample_test_data.txt
├── requirements.txt
└── .env.example
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
cp .env.example .env
```

4. (Optional) Use OpenAI embeddings + GPT model:
   - Set `EMBEDDING_PROVIDER=openai`
   - Set `OPENAI_API_KEY=...`

By default the app uses `sentence-transformers` embeddings locally.

## Run

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## How the RAG Flow Works

1. Ingestion:
   - Read text from uploaded PDF/TXT or pasted text.
   - Chunk text with overlap for better retrieval.
   - Generate embeddings.
   - Save chunks and vectors in ChromaDB.

2. Retrieval:
   - Embed user question.
   - Perform top-k similarity search in ChromaDB.
   - Convert cosine distance to similarity score.

3. Answering:
   - Apply threshold category logic.
   - If eligible, call LLM with strict context-only prompt.
   - If context is weak/missing, return the "not present" message.

## Example quick test

- Upload `data/sample_test_data.txt`
- Ask: `How many paid leave days are employees entitled to?`
- Ask: `What is the maternity leave policy?` (should likely return not present)
