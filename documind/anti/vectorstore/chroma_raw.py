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
    import chromadb
    
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # Use 'langchain' collection to match previous behavior
    collection = client.get_or_create_collection(name="langchain")
    
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    # Simple ID generation
    import uuid
    ids = [str(uuid.uuid4()) for _ in docs]
    
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    # Persist data explicitly if needed (0.3.x sometimes requires it)
    try:
        client.persist()
    except AttributeError:
        pass 
        
    return collection

def get_chroma():
    patch_pydantic_v1_for_chromadb()
    import chromadb

    return chromadb.PersistentClient(path=PERSIST_DIR)
