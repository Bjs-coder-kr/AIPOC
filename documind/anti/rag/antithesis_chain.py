from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_antithesis_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 문서를 비판적으로 분석하는 AI다.
        반드시 아래 제공된 문서 내용만 기반으로 답변하라.
        추측하거나 없는 사실을 만들지 마라.

        다음 관점에서 문서를 분석하라:
        1. 논리적으로 부족하거나 모호한 부분
        2. 주장에 비해 근거가 약한 부분
        3. 실무 관점에서 의문이 드는 부분
        4. 개선하면 좋을 점

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
