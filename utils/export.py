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
    """Create TXT file bytes."""
    return text.encode("utf-8")

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
        # Assuming AIPOC/assets/fonts/NanumGothic.ttf exists relative to the current working directory
        # We need absolute path or correct relative path. 
        # Since we run from "Project AI POC" (root), and file is in "AIPOC/assets/fonts/..."
        # We'll try to find it.
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # AIPOC root
        font_path = os.path.join(base_dir, "assets", "fonts", "NanumGothic.ttf")
        
        from reportlab.pdfbase.ttfonts import TTF
        pdfmetrics.registerFont(TTF(font_name, font_path))
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
