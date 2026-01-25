
import os
import sys
# Add current directory to path so we can import AIPOC
sys.path.append(os.getcwd())

from documind.ingest.loader import load_document
import docx

def create_samples():
    # TXT
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample text file.")

    # MD
    with open("sample.md", "w", encoding="utf-8") as f:
        f.write("# Sample Markdown\nThis is a markdown file.")

    # DOCX
    doc = docx.Document()
    doc.add_paragraph("This is a sample docx file.")
    doc.save("sample.docx")

def verify():
    create_samples()
    
    # Verify TXT
    with open("sample.txt", "rb") as f:
        data = f.read()
    res = load_document(data, "sample.txt")
    print(f"TXT Result: {res['pages'][0]['text'].strip()}")
    assert "sample text file" in res["pages"][0]["text"]

    # Verify MD
    with open("sample.md", "rb") as f:
        data = f.read()
    res = load_document(data, "sample.md")
    print(f"MD Result: {res['pages'][0]['text'].strip()}")
    assert "Sample Markdown" in res["pages"][0]["text"]

    # Verify DOCX
    with open("sample.docx", "rb") as f:
        data = f.read()
    res = load_document(data, "sample.docx")
    print(f"DOCX Result: {res['pages'][0]['text'].strip()}")
    assert "sample docx file" in res["pages"][0]["text"]
    
    # Cleanup
    os.remove("sample.txt")
    os.remove("sample.md")
    os.remove("sample.docx")
    print("SUCCESS: All file formats loaded correctly via ingest/loader.py")

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
