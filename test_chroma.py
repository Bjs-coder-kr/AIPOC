import chromadb
from chromadb.config import Settings
import os

PERSIST_DIR = os.path.abspath("chroma_raw")
print(f"Checking Persistence Directory: {PERSIST_DIR}")

try:
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    ))
    print("Client initialized successfully.")
    
    # Try different ways to list collections in 0.3.23
    try:
        if hasattr(client, 'list_collections'):
            print(f"Collections: {client.list_collections()}")
        else:
            # Fallback or internal method check
            print("client.list_collections() not found. Trying to get 'langchain' directly...")
            col = client.get_collection("langchain")
            print(f"Found collection: {col.name}, count: {col.count()}")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        
except Exception as e:
    print(f"Init Error: {e}")
