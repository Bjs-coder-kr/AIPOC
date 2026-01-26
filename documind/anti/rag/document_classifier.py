from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_document_type_chain(llm):
    prompt = ChatPromptTemplate.from_template(
        """
        아래 문서를 분석해서
        가장 적절한 문서 유형을 하나만 선택해.

        선택지는 다음 중 하나야:
        - 자기소개서
        - 기획서
        - 기술 문서
        - 보고서
        - 논문
        - 기타

        판단 기준:
        - 문서의 목적
        - 문체 (1인칭/객관)
        - 구성 방식
        - 반복적으로 등장하는 주제

        ⚠️ 결과는 반드시 위 선택지 중
        하나의 단어로만 출력해.

        [문서]
        {context}

        [문서 유형]
        """
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain
