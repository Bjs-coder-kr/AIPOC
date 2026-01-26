import os
# Hotfix for ChromaDB < 0.4.0 compatibility with Pydantic v2
os.environ.setdefault("CLICKHOUSE_HOST", "localhost")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")
os.environ.setdefault("CHROMA_SERVER_HOST", "localhost")
os.environ.setdefault("CHROMA_SERVER_HTTP_PORT", "8000")
os.environ.setdefault("CHROMA_SERVER_GRPC_PORT", "50051")

try:
    import pydantic
    if int(pydantic.VERSION.split('.')[0]) >= 2:
        from pydantic_settings import BaseSettings
        pydantic.BaseSettings = BaseSettings
except ImportError:
    pass

import chromadb
from chromadb.config import Settings

PERSIST_DIR = os.path.abspath("chroma_raw")
print(f"Checking Persistence Directory: {PERSIST_DIR}")

try:
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    ))
    print("Client initialized successfully.")
    
    # Try different ways to list collections
    try:
        # In 0.3.x, getting a collection usually doesn't raise error if it doesn't exist?
        # But get_or_create does. get_collection implies it must exist.
        col = client.get_collection("langchain")
        print(f"Found collection: {col.name}")
        print(f"Count: {col.count()}")
    except Exception as e:
        print(f"Error accessing collection 'langchain': {e}")
        # Try to list if possible (this method might not exist in old client)
        if hasattr(client, 'list_collections'):
             print(f"List collections: {client.list_collections()}")

except Exception as e:
    print(f"Init Error: {e}")
