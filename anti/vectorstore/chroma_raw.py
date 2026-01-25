import os
from pathlib import Path

PERSIST_DIR = str(Path(__file__).resolve().parents[1] / "chroma_raw")

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

def _get_embedding():
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def save_raw_docs(docs):
    from langchain_chroma import Chroma

    db = Chroma.from_documents(
        docs,
        _get_embedding(),
        persist_directory=PERSIST_DIR
    )
    return db

def get_chroma():
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_get_embedding()
    )
