"""
Document processing utilities for extracting text from various file formats.
"""
from pypdf import PdfReader
from docx import Document
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_bytes: PDF file content as bytes
        
    Returns:
        Extracted text as a string
    """
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_bytes: DOCX file content as bytes
        
    Returns:
        Extracted text as a string
    """
    try:
        docx_file = io.BytesIO(file_bytes)
        doc = Document(docx_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {e}")

def process_uploaded_file(uploaded_file) -> str:
    """
    Process an uploaded file and extract its text content.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Extracted text as a string
        
    Raises:
        ValueError: If file type is unsupported or processing fails
    """
    file_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
