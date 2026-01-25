"""
Text/Markdown/DOCX loader for AIPOC.
Adapts various text formats into the standard document structure.
"""
from __future__ import annotations

import logging
from io import BytesIO
import docx

logger = logging.getLogger(__name__)

def load_text(file_bytes: bytes, file_name: str) -> dict:
    """Load text/markdown files."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {file_name}, trying latin-1")
        text = file_bytes.decode("latin-1")

    # Treat entire text as a single page for now
    # TODO: Could implement smart splitting by headers for MD
    pages = [{"page_number": 1, "text": text}]
    
    meta = {
        "file_name": file_name,
        "page_count": 1,
        "textless_pages": 0,
        "raw_char_count": len(text),
        "scan_like": False,
        "scan_like_ratio": 0.0,
        "scan_level": "NONE",
    }
    
    return {"pages": pages, "meta": meta}

def load_docx(file_bytes: bytes, file_name: str) -> dict:
    """Load DOCX files using python-docx."""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        text = "\n".join(full_text)
        
        # Treat as single page
        pages = [{"page_number": 1, "text": text}]
        
        meta = {
            "file_name": file_name,
            "page_count": 1, # DOCX doesn't have fixed pages
            "textless_pages": 0 if text.strip() else 1,
            "raw_char_count": len(text),
            "scan_like": False,
            "scan_like_ratio": 0.0 if text.strip() else 1.0,
            "scan_level": "NONE",
        }
        
        return {"pages": pages, "meta": meta}
        
    except Exception as e:
        logger.error(f"Failed to load DOCX {file_name}: {e}")
        # Return empty structure on error
        return {
            "pages": [{"page_number": 1, "text": ""}],
            "meta": {
                "file_name": file_name,
                "page_count": 1,
                "textless_pages": 1,
                "raw_char_count": 0,
                "scan_like": False,
                "scan_like_ratio": 0.0,
                "scan_level": "none",
            }
        }
