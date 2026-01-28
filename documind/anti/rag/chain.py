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
        너는 냉정하고 비판적인 리뷰어야. 아래 문서를 근거로만 안티테제(반론)를 작성해.
        문서에 없는 정보는 상상하지 말고, 추정이 필요한 경우 "추정"이라고 표시해.

        다음 항목 중심으로 지적해:
        1) 논리적 약점/비약
        2) 과장/모호/검증 불가 표현
        3) 근거 부족/증거 누락
        4) 반대 관점에서의 반론

        출력 형식(한국어):
        - 항목은 5~10개, 중요도 높은 순서
        - 각 항목에 근거 스니펫 1개 포함(문서에서 그대로 인용, 1~2문장)
        - 근거에는 반드시 페이지 표기 포함: (pX)

        형식:
        1. [한줄 요약] (중요도: 높음/중간/낮음)
           - 비판: ...
           - 근거: "..." (pX)
           - 반대 관점: ...
           - 개선 제안: ...

        [문서]
        {context}

        [안티테제 분석]
        """
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[p{page}] " if page is not None else ""
            formatted.append(f"{prefix}{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_antithesis_critic_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 비판 분석 결과를 검수하는 리뷰어야.
        아래 문서와 안티테제 결과를 비교해서 근거 일치성과 과장 여부를 점검해.
        문서에 없는 내용이 있으면 지적하고, 개선점을 제시해.

        출력 형식(한국어):
        - verdict: PASS 또는 FAIL
        - score: 0~100
        - issues: 3개 이내 불릿
        - suggestions: 3개 이내 불릿

        [문서]
        {context}

        [안티테제 결과]
        {antithesis}

        [검수 결과]
        """
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[p{page}] " if page is not None else ""
            formatted.append(f"{prefix}{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "antithesis": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_antithesis_refine_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 비판 분석 결과를 개선하는 편집자야.
        아래 문서와 기존 안티테제, 검수 피드백을 참고해서 더 정확하고 근거 기반으로 재작성해.
        문서에 없는 내용은 제거하고, 근거가 약한 부분은 약하게 표현해.

        출력은 안티테제 본문만.

        [문서]
        {context}

        [기존 안티테제]
        {antithesis}

        [검수 피드백]
        {review}

        [개선된 안티테제]
        """
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[p{page}] " if page is not None else ""
            formatted.append(f"{prefix}{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "antithesis": lambda x: x["antithesis"],
            "review": lambda x: x["review"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_revision_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 문서를 개선해 다시 쓰는 편집자야.
        아래 문서 원문과 안티테제(비판 분석)를 반영해서 더 명확하고 설득력 있게 재작성해.
        문서에 없는 사실은 추가하지 마.

        출력은 개선된 문서 본문만.

        [문서]
        {context}

        [안티테제]
        {antithesis}

        [개선된 문서]
        """
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[p{page}] " if page is not None else ""
            formatted.append(f"{prefix}{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "antithesis": lambda x: x["antithesis"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_revision_critic_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 개선된 문서를 검수하는 리뷰어야.
        아래 문서 원문, 안티테제, 개선된 문서를 비교해 근거 일치성과 과장 여부를 점검해.
        문서에 없는 내용이 있으면 지적하고 개선 방향을 제시해.

        출력 형식(한국어):
        - verdict: PASS 또는 FAIL
        - score: 0~100
        - issues: 3개 이내 불릿
        - suggestions: 3개 이내 불릿

        [문서]
        {context}

        [안티테제]
        {antithesis}

        [개선된 문서]
        {revision}

        [검수 결과]
        """
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[p{page}] " if page is not None else ""
            formatted.append(f"{prefix}{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "antithesis": lambda x: x["antithesis"],
            "revision": lambda x: x["revision"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_revision_refine_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        """
        너는 개선된 문서를 한 번 더 다듬는 편집자야.
        아래 문서 원문, 안티테제, 개선된 문서, 검수 피드백을 반영해서 더 정확하고 자연스럽게 재작성해.
        문서에 없는 내용은 제거하고, 근거가 약한 부분은 약하게 표현해.

        출력은 최종 개선본만.

        [문서]
        {context}

        [안티테제]
        {antithesis}

        [개선된 문서]
        {revision}

        [검수 피드백]
        {review}

        [최종 개선본]
        """
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[p{page}] " if page is not None else ""
            formatted.append(f"{prefix}{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "antithesis": lambda x: x["antithesis"],
            "revision": lambda x: x["revision"],
            "review": lambda x: x["review"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
