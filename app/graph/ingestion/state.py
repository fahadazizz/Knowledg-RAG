from typing import TypedDict, List, Dict, Any, Annotated
import operator

class IngestionState(TypedDict):
    """
    State for the document ingestion process.
    """
    raw_documents: List[str]
    cleaned_documents: List[str]
    chunks: List[Dict[str, Any]]
    extraction_results: List[Dict[str, Any]]
    doc_ids: List[str]
    status: str
    error: str
