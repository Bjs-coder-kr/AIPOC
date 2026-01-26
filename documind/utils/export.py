"""
Export utilities for creating downloadable files (PDF, DOCX, TXT, ZIP).
"""
import io
import zipfile
import docx
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_txt_bytes(text: str) -> bytes:
    """Create TXT file bytes with UTF-8 BOM for better compatibility."""
    # UTF-8 BOM helps Korean text display correctly in Windows Notepad and similar editors
    UTF8_BOM = b'\xef\xbb\xbf'
    return UTF8_BOM + text.encode("utf-8")

def create_docx_bytes(text: str) -> bytes:
    """Create DOCX file bytes."""
    doc = docx.Document()
    for paragraph in text.split('\n'):
        if paragraph.strip():
            doc.add_paragraph(paragraph)
    
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def create_pdf_bytes(text: str) -> bytes:
    """Create PDF file bytes with Korean support."""
    bio = io.BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )

    # Register Korean Font (NanumGothic TTF)
    font_name = 'NanumGothic'
    try:
        # Resolve path relative to this file: export.py is in documind/utils/
        # Assets are in assets/fonts/ (root/assets/fonts)
        # So we go up 2 levels (documind/utils -> documind -> root)
        from pathlib import Path
        import os
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]  # documind/utils -> documind -> root
        font_path = project_root / "assets" / "fonts" / "NanumGothic.ttf"
        
        if not font_path.exists():
             # Try one level up if in documind/app case? No, structure is fixed.
             # Just log if missing
             print(f"Font not found at: {font_path}")
             raise FileNotFoundError(f"Font missing: {font_path}")

        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    except Exception as e:
        # Fallback to CID if TTF fails
        print(f"Font loading failed: {e}")
        try:
             font_name = 'HYSMyeongJo-Medium' # Fallback name
             from reportlab.pdfbase.cidfonts import UnicodeCIDFont
             pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        except:
             pass

    styles = getSampleStyleSheet()
    # Modify Normal style to use Korean font
    style = ParagraphStyle(
        'KoreanNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=10
    )

    story = []
    for paragraph in text.split('\n'):
        if paragraph.strip():
            # Replace newlines with <br/> if needed, or just let Paragraph handle wrapping
            # Paragraph expects XML-like tags, need to escape special chars?
            # Reportlab Paragraph handles simplistic HTML-like tags.
            # We should escape <, >, & at least.
            safe_text = paragraph.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_text, style))
    
    doc.build(story)
    return bio.getvalue()

def create_zip_bytes(files: dict[str, bytes]) -> bytes:
    """
    Create a ZIP file containing provided files.
    files: dict {filename: bytes}
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return bio.getvalue()
