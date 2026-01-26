import os
import streamlit as st
import tempfile

from documind.anti.ingest.pdf_loader import load_pdf_with_ocr
from documind.anti.ingest.splitter import split_docs
from documind.anti.vectorstore.chroma_raw import save_raw_docs, get_chroma
from documind.anti.rag.claude import get_claude
from documind.anti.rag.chain import get_rag_chain
from documind.anti.rag.document_classifier import get_document_type_chain


if os.getenv("DOCUMIND_UNIFIED_APP") != "1":
    st.set_page_config(page_title="📄 문서 Q&A")
st.title("📄 PDF 문서 Q&A (OCR 지원)")

uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 PDF 처리 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        docs = load_pdf_with_ocr(tmp_path)
        chunks = split_docs(docs)
        save_raw_docs(chunks)

    st.success("✅ 문서 인덱싱 완료!")

    # 🔥 OCR / 텍스트 미리보기
    st.subheader("📄 OCR / 텍스트 추출 결과 미리보기")

    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "pdf")

        label = "🧠 OCR" if source == "ocr" else "📄 PDF 텍스트"

        with st.expander(f"{label} | Page {page}"):
            st.text(doc.page_content[:3000])

    # RAG 준비
    llm = get_claude()
    db = get_chroma()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag_chain = get_rag_chain(llm, retriever)


    # =========================
    # 🔍 일반 질문 Q&A
    # =========================
    question = st.text_input("문서에 대해 질문하세요")

    if question:
        with st.spinner("🤖 답변 생성 중..."):
            answer = rag_chain.invoke(question)
        st.markdown("### 💡 답변")
        st.write(answer)

    # =========================
    # 🧠 문서 분석 (요약 / 안티테제)
    # =========================
    st.divider()
    st.subheader("🧠 문서 분석")

    

    

    col1, col2, col3 = st.columns(3)

    # 1️⃣ 요약
    with col1:
        if st.button("📌 핵심 요약"):
            with st.spinner("요약 중..."):
                answer = rag_chain.invoke("이 문서의 핵심 내용을 요약해줘")
            st.write(answer)

    # 2️⃣ 안티테제
    with col2:
        if st.button("⚠️ 안티테제 (비판 분석)"):
            from documind.anti.rag.chain import get_antithesis_chain

            antithesis_chain = get_antithesis_chain(llm, retriever)

            with st.spinner("비판적으로 분석 중..."):
                antithesis = antithesis_chain.invoke(
                    "이 문서 전체를 비판적으로 분석해줘"
                )

            st.session_state["antithesis"] = antithesis
            st.markdown("### ⚠️ 안티테제 분석")
            st.write(antithesis)

    # 3️⃣ 개선된 문서 재작성
    with col3:
        if st.button("✨ 개선된 문서 재작성"):
            if "antithesis" not in st.session_state:
                st.warning("먼저 안티테제를 생성해주세요.")
            else:
                from documind.anti.rag.chain import get_revision_chain

                revision_chain = get_revision_chain(llm, retriever)

                with st.spinner("문서 개선 중..."):
                    revised = revision_chain.invoke({
                        "antithesis": st.session_state["antithesis"]
                    })

                st.markdown("### ✨ 개선된 문서")
                st.write(revised)
