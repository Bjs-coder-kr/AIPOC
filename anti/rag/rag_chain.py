from documind.anti.rag.claude import get_claude
from documind.anti.rag.chain import get_rag_chain
from documind.anti.vectorstore.chroma_raw import get_chroma

# 🔒 전역 캐시 (중요)
_rag_chain = None


def ask(question: str) -> str:
    global _rag_chain

    # 1️⃣ 최초 1회만 초기화
    if _rag_chain is None:
        llm = get_claude()
        db = get_chroma()
        retriever = db.as_retriever(search_kwargs={"k": 3})
        _rag_chain = get_rag_chain(llm, retriever)

    # 2️⃣ 질문 실행
    return _rag_chain.invoke(question)

def format_docs(docs):
    print("=== RETRIEVED DOCS ===")
    for d in docs:
        print(d.page_content)
        print("------")
    return "\n\n".join(d.page_content for d in docs)
