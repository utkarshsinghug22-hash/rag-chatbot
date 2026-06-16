# 🔍 RAG Chatbot — Chat with Your Documents

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-czaiqcna6dqnnbuappy3wxe.streamlit.app/)

**🚀 Live Demo:** [https://rag-chatbot-czaiqcna6dqnnbuappy3wxe.streamlit.app/](https://rag-chatbot-czaiqcna6dqnnbuappy3wxe.streamlit.app/)

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions over your PDF documents using LangChain, FAISS, and Groq.

## What Problem Does This Solve?

**Plain LLMs** (like ChatGPT, Llama) can only answer from their training data — they hallucinate when asked about your private documents, recent reports, or domain-specific content.

**RAG (Retrieval-Augmented Generation)** solves this by:
1. **Retrieving** relevant chunks from YOUR documents first
2. **Generating** an answer grounded ONLY in those chunks
3. **Citing sources** so you can verify every answer

This means: accurate, verifiable answers from your own PDFs — with zero hallucination on out-of-scope questions.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  PDF Files   │────▶│  Text Chunks  │────▶│   Embeddings    │
│  (docs/)     │     │  (500 chars)  │     │  (bge-base-en)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Answer +   │◀────│   Groq LLM   │◀────│   FAISS Index   │
│   Sources    │     │ (Llama 3.3)  │     │  (vector store) │
└─────────────┘     └──────────────┘     └─────────────────┘
                           ▲
                           │
                    ┌──────┴──────┐
                    │  User Query  │
                    │ (Streamlit)  │
                    └─────────────┘
```

**Flow:** PDF → chunks → embeddings → FAISS index → user query → retrieve top-4 chunks → Groq LLM generates answer → display with sources

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| **Orchestration** | LangChain | Industry-standard RAG pipeline framework |
| **Vector Store** | FAISS (CPU) | Fast local vector search, no server needed |
| **Embeddings** | BAAI/bge-base-en-v1.5 | Top-tier retrieval model, runs locally, free |
| **LLM** | Groq (Llama 3.3 70B) | Free API, 30 req/min, fast inference |
| **PDF Loading** | pypdf | Modern replacement for deprecated PyPDF2 |
| **UI** | Streamlit | Python-native UI, free cloud deployment |
| **Deployment** | Streamlit Community Cloud | One-click deploy from GitHub |

## Project Structure

```
rag-chatbot/
├── app.py                  # Streamlit chat UI
├── rag_pipeline.py         # RAG chain (retriever + LLM)
├── ingest.py               # PDF → chunks → FAISS index
├── evaluate.py             # Automated evaluation script
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore              # Security + cleanliness
├── .streamlit/
│   └── secrets.toml.example  # Streamlit Cloud secrets template
├── docs/
│   └── (your PDFs here)    # Document corpus
└── README.md               # This file
```

## Setup and Run Locally

### Prerequisites
- Python 3.9+
- Free Groq API key ([get one here](https://console.groq.com) — no credit card needed)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/joyboy034/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv rag-env
   rag-env\Scripts\activate

   # macOS/Linux
   python -m venv rag-env
   source rag-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   ```bash
   # Windows
   copy .env.example .env

   # macOS/Linux
   cp .env.example .env
   ```
   Open `.env` and replace `your_groq_api_key_here` with your actual Groq key.

5. **Add your PDFs**
   Place at least one PDF file in the `docs/` folder.

6. **Build the FAISS index**
   ```bash
   python ingest.py
   ```
   First run downloads the embedding model (~440MB, cached after that).

7. **Launch the chatbot**
   ```bash
   streamlit run app.py
   ```
   The app opens in your browser at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push your repo to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your GitHub repo
4. Set the main file to `app.py`
5. Go to **Settings → Secrets** and add:
   ```
   GROQ_API_KEY = "your_actual_key_here"
   ```
6. Click **Deploy** — your app is live!

**✅ Live app:** [https://rag-chatbot-czaiqcna6dqnnbuappy3wxe.streamlit.app/](https://rag-chatbot-czaiqcna6dqnnbuappy3wxe.streamlit.app/)

> **Note:** You'll need to include the `faiss_index/` folder in your repo for cloud deployment, or modify `app.py` to build the index on startup.

## Features

- 💬 **Chat interface** with persistent session history
- 📄 **PDF upload** — bring your own documents without re-deploying
- 📎 **Source citations** — every answer shows which PDF page it came from
- 👍👎 **Feedback logging** — thumbs up/down saved to CSV for analysis
- 🛡️ **Anti-hallucination prompt** — model refuses to answer if context is insufficient
- ⚙️ **Model info sidebar** — shows what's running under the hood
- 📊 **Session stats** — tracks helpfulness rating in real-time
- 🧪 **Evaluation script** — measure accuracy and latency with `evaluate.py`

## What I Tried and Learned

### Chunk Size Tuning
- Started with 1000-char chunks — too much noise in retrieval
- Settled on **500 chars with 50 overlap** — better precision without losing context

### Embedding Model Comparison
- Tried `all-MiniLM-L6-v2` — fast but lower retrieval quality
- **BAAI/bge-base-en-v1.5** gave significantly better results for QA tasks
- Requires `normalize_embeddings=True` (trained with cosine similarity)

### Hallucination Guard
- Without the anti-hallucination prompt, the model would fabricate answers for out-of-scope questions
- The explicit "say I don't know" instruction solved this reliably

### Limitations
- **Groq free tier**: 30 requests/minute, 1000/day — sufficient for personal use
- **CPU-only FAISS**: Fine for <10K chunks; for larger corpora, consider GPU FAISS
- **No multi-modal**: Only handles text in PDFs, not images/tables

## Possible Extensions

- 🌍 **Multilingual support** — swap embedding model for multilingual-e5-large
- 📏 **RAGAS evaluation** — automated faithfulness + relevance scoring
- 🔄 **LLM comparison** — add Gemini, Claude, or GPT-4 as alternative generators
- 📊 **Analytics dashboard** — visualize feedback trends and query patterns
- 🔐 **Authentication** — add user login for private deployments

## Built By

**Utkarsh Singh**
NSUT Delhi

- 🔗 [LinkedIn](https://www.linkedin.com/in/utkarsh-singh-347609366)
- 💻 [GitHub](https://github.com/joyboy034)

---

*Built with LangChain, FAISS, Groq, and Streamlit. June 2026.*
