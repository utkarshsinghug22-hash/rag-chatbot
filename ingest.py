"""
ingest.py — Document Ingestion Pipeline
Run once after adding PDFs to docs/:  python ingest.py

Verified (June 2026):
  - pypdf replaces deprecated PyPDF2
  - BAAI/bge-base-en-v1.5: strong RAG retrieval, runs locally, free
  - chunk_size=500, overlap=50: well-tested default for QA tasks
"""

import os
import time
import warnings

warnings.filterwarnings("ignore", message=".*langchain-community.*being sunset.*")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_FOLDER      = "docs/"
INDEX_PATH      = "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


def load_pdfs(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Create a docs/ folder and add PDFs to it.")
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDFs found in {folder}.")
    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")
    docs = []
    for filename in pdf_files:
        path = os.path.join(folder, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()
        docs.extend(pages)
        print(f"  Loaded: {filename} ({len(pages)} pages)")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"\nSplit into {len(chunks)} chunks")
    return chunks


def build_and_save_index(chunks):
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~440MB — cached after that)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Building FAISS index...")
    start = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}/ ({time.time()-start:.1f}s)")
    print(f"\nDone. {len(chunks)} chunks indexed.")


def main():
    print("=" * 50)
    print("RAG Chatbot -- Document Ingestion")
    print("=" * 50)
    docs   = load_pdfs(PDF_FOLDER)
    chunks = split_documents(docs)
    build_and_save_index(chunks)
    print("\nRun the app: streamlit run app.py")


if __name__ == "__main__":
    main()
