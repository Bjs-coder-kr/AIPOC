from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 문서를 기반으로 질문에 답하는 AI야.
        반드시 아래 제공된 문서 내용만 사용해서 답변해.
        모르면 모른다고 말해.

        [문서]
        {context}

        [질문]
        {question}

        [답변]
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ============================
# 🔥 안티테제 체인 (비판 분석)
# ============================
def get_antithesis_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 비판적 사고를 하는 리뷰어야.

        아래 문서를 읽고,
        1. 논리적 약점
        2. 과장된 표현
        3. 근거가 부족한 주장
        4. 반대 관점에서의 비판

        을 중심으로 안티테제(반론)를 제시해.

        문서에 없는 내용은 상상하지 마.
        반드시 문서 내용 기반으로만 비판해.

        [문서]
        {context}

        [안티테제 분석]
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
