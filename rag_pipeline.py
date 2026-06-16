"""
rag_pipeline.py — RAG Chain
Verified (June 2026):
  - Groq free tier: llama-3.3-70b-versatile, 30 RPM, 1000 req/day
  - BAAI/bge-base-en-v1.5 must match the model used in ingest.py
  - allow_dangerous_deserialization=True required for local FAISS indexes
  - temperature=0.2: keeps answers factual and grounded
  - k=4: retrieves 4 most similar chunks (~2000 tokens of context)
"""

import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*langchain-community.*being sunset.*")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

INDEX_PATH      = "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
GROQ_MODEL      = "llama-3.3-70b-versatile"
RETRIEVER_K     = 4
TEMPERATURE     = 0.2

# Anti-hallucination prompt: instructs the model to use ONLY retrieved context.
# The fallback sentence handles out-of-scope questions gracefully.
RAG_PROMPT = PromptTemplate(
    template="""You are a precise and helpful assistant.
Answer the question using ONLY the context provided below.

Rules:
1. Use ONLY information from the context — not your training knowledge.
2. If the answer is not in the context, say:
   "I don't have enough information in the provided documents to answer that."
3. Be concise and cite relevant details from the context.
4. Do not make up facts, names, dates, or statistics.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


def load_rag_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found.\n"
            "1. Get a free key at https://console.groq.com\n"
            "2. Copy .env.example to .env and add your key."
        )
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found. Run: python ingest.py"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        api_key=api_key,
        max_tokens=1024
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return chain


def get_answer(chain, query: str) -> dict:
    result  = chain.invoke({"query": query})
    sources = [
        {
            "file":    doc.metadata.get("source", "Unknown"),
            "page":    doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:300]
        }
        for doc in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}
