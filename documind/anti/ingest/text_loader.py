from langchain_core.documents import Document


def load_text(text: str):
    return [Document(page_content=text)]
