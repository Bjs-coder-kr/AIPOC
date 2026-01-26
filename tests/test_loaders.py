
import pytest
from unittest.mock import MagicMock, patch
from documind.ingest.loader import load_document
from documind.ingest.text_loader import load_text, load_docx

def test_load_text_utf8():
    data = b"Hello World"
    res = load_text(data, "test.txt")
    assert res["meta"]["file_name"] == "test.txt"
    assert res["pages"][0]["text"] == "Hello World"
    assert res["meta"]["raw_char_count"] == 11

def test_load_text_latin1():
    # 'Hello' in latin-1 is same, but let's try something that might fail utf-8 decoding if strictly checked
    # But python bytes decode defaults to utf-8. 
    # Invalid utf-8 sequence: 0xFF
    data = b"Hello \xff" 
    res = load_text(data, "test.txt")
    # Should fall back to latin-1
    assert "Hello" in res["pages"][0]["text"]

def test_loader_dispatch_txt():
    # Mock load_text to verify dispatch
    with patch("documind.ingest.loader.load_text") as mock_load:
        mock_load.return_value = {"pages": [], "meta": {}}
        load_document(b"data", "file.txt")
        mock_load.assert_called_once()

def test_loader_dispatch_docx():
    with patch("documind.ingest.loader.load_docx") as mock_load:
        mock_load.return_value = {"pages": [], "meta": {}}
        load_document(b"data", "file.docx")
        mock_load.assert_called_once()

def test_loader_dispatch_pdf():
    with patch("documind.ingest.loader.load_pdf") as mock_load:
        mock_load.return_value = {"pages": [], "meta": {}}
        load_document(b"data", "file.pdf")
        mock_load.assert_called_once()
