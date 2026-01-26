"""
Unified document loader facade.
Dispatches to specific loaders based on file extension.
"""
from __future__ import annotations

from .pdf_loader import load_pdf
from .text_loader import load_text, load_docx

def load_document(file_bytes: bytes, file_name: str) -> dict:
    """
    Load a document from bytes, routing to the appropriate loader.
    Supports PDF, DOCX, TXT, MD.
    
    Returns:
        dict: {"pages": [...], "meta": ...} structure.
    """
    lower_name = file_name.lower()
    
    if lower_name.endswith(".pdf"):
        return load_pdf(file_bytes, file_name)
    elif lower_name.endswith(".docx"):
        return load_docx(file_bytes, file_name)
    else:
        # Default to text loader (txt, md, log, etc.)
        return load_text(file_bytes, file_name)
