import re
import pdfplumber
import pytesseract
from langchain_core.documents import Document  # ✅ 이거 유지

def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()

        if len(line) < 5:
            continue

        if re.fullmatch(r"[.,·\s]+", line):
            continue

        special_ratio = sum(1 for c in line if not c.isalnum()) / len(line)
        if special_ratio > 0.6:
            continue

        lines.append(line)

    return "\n".join(lines)


def load_pdf_with_ocr(path: str):
    docs = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            source = "pdf"

            if not text or len(text.strip()) < 30:
                image = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(image, lang="kor+eng")
                source = "ocr"

            # 🔥 여기서 정제
            text = clean_text(text)

            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": i + 1,
                            "source": source
                        }
                    )
                )

    return docs


def load_pdf(path: str):
    return load_pdf_with_ocr(path)
