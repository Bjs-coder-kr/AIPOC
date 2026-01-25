from documind.anti.ingest.pdf_loader import load_pdf
from documind.anti.ingest.splitter import split_docs
from documind.anti.vectorstore.chroma_raw import save_raw_docs

print("PDF 로딩 중...")
docs = load_pdf("sample.pdf")

print("문서 분할 중...")
chunks = split_docs(docs)

print(f"총 chunk 수: {len(chunks)}")

print("Chroma에 저장 중...")
save_raw_docs(chunks)

print("✅ 인덱싱 완료")
