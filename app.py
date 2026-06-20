"""
app.py — Streamlit Chat UI

Features:
  - Chat history (session-scoped)
  - PDF uploader (users can bring their own docs)
  - Source document expander (shows retrieved chunks + page numbers)
  - Thumbs up/down feedback logging to CSV
  - Sidebar with model info and session stats

Run: streamlit run app.py
Deploy free: share.streamlit.io
"""

import os
import csv
import time
import tempfile
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message=".*langchain-community.*being sunset.*")

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline import load_rag_chain, get_answer, get_api_key, EMBEDDING_MODEL

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages"        not in st.session_state:
    st.session_state.messages = []
if "chain"           not in st.session_state:
    st.session_state.chain = None
if "using_upload"    not in st.session_state:
    st.session_state.using_upload = False
if "feedback_counts" not in st.session_state:
    st.session_state.feedback_counts = {"up": 0, "down": 0}


@st.cache_resource(show_spinner=False)
def get_default_chain():
    return load_rag_chain()


def build_chain_from_upload(uploaded_files):
    docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        os.unlink(tmp_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    from langchain_groq import ChatGroq
    from langchain_classic.chains import RetrievalQA
    from rag_pipeline import RAG_PROMPT, GROQ_MODEL, TEMPERATURE

    api_key = get_api_key()
    llm = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE,
                   api_key=api_key, max_tokens=1024)
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return chain, len(chunks)


def log_feedback(question, answer, rating):
    log_file = "feedback_log.csv"
    exists   = os.path.exists(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "question", "answer", "rating"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "question":  question,
            "answer":    answer[:500],
            "rating":    rating
        })


# ── Sidebar ──
with st.sidebar:
    st.title("🔍 RAG Chatbot")
    st.caption("LangChain · FAISS · Groq · HuggingFace")
    st.divider()

    st.subheader("📄 Upload your own PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDFs", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_files:
        if st.button("Index uploaded PDFs", type="primary", use_container_width=True):
            with st.spinner(f"Indexing {len(uploaded_files)} file(s)..."):
                try:
                    chain, n = build_chain_from_upload(uploaded_files)
                    st.session_state.chain        = chain
                    st.session_state.using_upload = True
                    st.session_state.messages     = []
                    st.success(f"Indexed {n} chunks. Start chatting!")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.using_upload:
        if st.button("↩ Use default corpus", use_container_width=True):
            st.session_state.chain        = None
            st.session_state.using_upload = False
            st.session_state.messages     = []
            st.rerun()

    st.divider()
    st.subheader("⚙️ Model info")
    st.markdown("""
| Component | Details |
|-----------|---------|
| **LLM** | Llama 3.3 70B |
| **Provider** | Groq (free tier) |
| **Embeddings** | BAAI/bge-base-en-v1.5 |
| **Vector DB** | FAISS (local) |
| **Retrieval k** | 4 chunks/query |
""")
    st.divider()
    st.subheader("📊 Session feedback")
    up, down = st.session_state.feedback_counts["up"], st.session_state.feedback_counts["down"]
    total = up + down
    if total > 0:
        st.metric("Helpfulness", f"{round(up/total*100)}%", f"{total} ratings")
    else:
        st.caption("No ratings yet.")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main ──
st.title("Chat with your documents")
st.caption("📎 Using uploaded documents" if st.session_state.using_upload
           else "📚 Using default document corpus")

if st.session_state.chain is None:
    with st.spinner("Loading model and index..."):
        try:
            st.session_state.chain = get_default_chain()
        except FileNotFoundError:
            st.error("**FAISS index not found.** Run `python ingest.py` or upload PDFs in the sidebar.")
            st.stop()
        except EnvironmentError as e:
            st.error(f"**API key error:** {e}")
            st.stop()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for j, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {j}:** `{os.path.basename(src['file'])}` · Page {src['page']}")
                    st.caption(src["snippet"] + "...")
                    if j < len(msg["sources"]):
                        st.divider()
        if msg["role"] == "assistant":
            c1, c2, _ = st.columns([1, 1, 8])
            with c1:
                if st.button("👍", key=f"up_{i}"):
                    user_q = st.session_state.messages[i-1]["content"] if i > 0 else "unknown"
                    log_feedback(user_q, msg["content"], "positive")
                    st.session_state.feedback_counts["up"] += 1
                    st.toast("Thanks!")
            with c2:
                if st.button("👎", key=f"dn_{i}"):
                    user_q = st.session_state.messages[i-1]["content"] if i > 0 else "unknown"
                    log_feedback(user_q, msg["content"], "negative")
                    st.session_state.feedback_counts["down"] += 1
                    st.toast("Noted!")

if query := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                start   = time.time()
                result  = get_answer(st.session_state.chain, query)
                elapsed = time.time() - start
                answer, sources = result["answer"], result["sources"]
            except Exception as e:
                answer, sources, elapsed = f"Error: {e}", [], 0

        st.write(answer)
        st.caption(f"_{elapsed:.1f}s · {len(sources)} chunks retrieved_")

        if sources:
            with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                for j, src in enumerate(sources, 1):
                    st.markdown(f"**Source {j}:** `{os.path.basename(src['file'])}` · Page {src['page']}")
                    st.caption(src["snippet"] + "...")
                    if j < len(sources):
                        st.divider()

        c1, c2, _ = st.columns([1, 1, 8])
        with c1:
            if st.button("👍", key="up_new"):
                log_feedback(query, answer, "positive")
                st.session_state.feedback_counts["up"] += 1
                st.toast("Thanks!")
        with c2:
            if st.button("👎", key="dn_new"):
                log_feedback(query, answer, "negative")
                st.session_state.feedback_counts["down"] += 1
                st.toast("Noted!")

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
