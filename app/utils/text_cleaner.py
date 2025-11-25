"""
Text cleaning and preprocessing utilities.
"""
import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters but keep printable ones
    # This is safer than a whitelist approach for general text
    text = "".join(ch for ch in text if ch.isprintable())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting (can be improved with spaCy)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
