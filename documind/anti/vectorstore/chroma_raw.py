import os
from pathlib import Path


PERSIST_DIR = str(Path(__file__).resolve().parents[3] / "chroma_raw")


os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

def _get_embedding():
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def save_raw_docs(docs):
    import chromadb
    from chromadb.config import Settings
    
    # ChromaDB 0.3.x style persistence
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    ))
    
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
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    ))
    return client
