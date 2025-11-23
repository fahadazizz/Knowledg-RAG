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
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
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
