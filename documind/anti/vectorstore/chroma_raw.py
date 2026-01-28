import os
from pathlib import Path

from documind.utils.pydantic_compat import patch_pydantic_v1_for_chromadb


PERSIST_DIR = str(Path(__file__).resolve().parents[3] / "chroma_raw")


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

def _get_embedding():
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def save_raw_docs(docs):
    patch_pydantic_v1_for_chromadb()
    from langchain_community.vectorstores import Chroma

    embedding = _get_embedding()
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
        collection_name="langchain",
    )
    db.add_documents(docs)
    return db

def get_chroma():
    patch_pydantic_v1_for_chromadb()
    from langchain_community.vectorstores import Chroma

    embedding = _get_embedding()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
        collection_name="langchain",
    )
