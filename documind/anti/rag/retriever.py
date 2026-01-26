from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_retriever():
    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_raw",
        embedding_function=embedding
    )

    return db.as_retriever(search_kwargs={"k": 5})
