from app.graph.ingestion.state import IngestionState
from app.utils.text_cleaner import clean_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.kg_builder import KnowledgeGraphBuilder
import uuid
from typing import Dict, Any

def clean_node(state: IngestionState) -> Dict[str, Any]:
    """
    Node to clean raw documents.
    """
    print("--- CLEANING DOCUMENTS ---")
    raw_docs = state["raw_documents"]
    cleaned_docs = [clean_text(doc) for doc in raw_docs]
    return {"cleaned_documents": cleaned_docs, "status": "cleaned"}

def chunk_node(state: IngestionState) -> Dict[str, Any]:
    """
    Node to split documents into structural chunks.
    """
    print("--- CHUNKING DOCUMENTS ---")
    cleaned_docs = state["cleaned_documents"]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    all_chunks = []
    doc_ids = []
    
    for doc in cleaned_docs:
        # Generate a Document ID for each document
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        
        chunks = text_splitter.split_text(doc)
        # Tag chunks with their parent doc_id
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "id": str(uuid.uuid4()),
                "doc_id": doc_id
            })
            
    return {"chunks": all_chunks, "doc_ids": doc_ids, "status": "chunked"}

def extract_node(state: IngestionState) -> Dict[str, Any]:
    """
    Node to extract entities and relations from chunks.
    """
    print("--- EXTRACTING ENTITIES & RELATIONS ---")
    chunks = state["chunks"]
    kg_builder = KnowledgeGraphBuilder()
    
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    
    # Use existing parallel extraction logic
    results = kg_builder.extract_from_chunks(chunk_texts, chunk_ids)
    
    # Add simple canonicalization here or in a separate node
    # For now, we'll do a simple pass
    for res in results:
        for entity in res.get("entities", []):
            entity["name"] = entity["name"].strip()
            
    return {"extraction_results": results, "status": "extracted"}

def build_node(state: IngestionState) -> Dict[str, Any]:
    """
    Node to build the graph in Neo4j.
    """
    print("--- BUILDING GRAPH ---")
    chunks = state["chunks"]
    extraction_results = state["extraction_results"]
    doc_ids = state["doc_ids"]
    
    kg_builder = KnowledgeGraphBuilder()
    
    try:
        # We need to group chunks by doc_id to call build_pure_graph per document
        # or refactor build_pure_graph to handle batch.
        # Let's refactor build_pure_graph usage here.
        
        # Group chunks and results by doc_id
        doc_chunks_map = {did: [] for did in doc_ids}
        doc_results_map = {did: [] for did in doc_ids}
        
        # Map chunk_id to doc_id
        chunk_to_doc = {c["id"]: c["doc_id"] for c in chunks}
        
        for chunk in chunks:
            doc_chunks_map[chunk["doc_id"]].append(chunk)
            
        for res in extraction_results:
            c_id = res.get("chunk_id")
            if c_id and c_id in chunk_to_doc:
                d_id = chunk_to_doc[c_id]
                doc_results_map[d_id].append(res)
        
        # Build graph for each document
        for d_id in doc_ids:
            kg_builder.build_pure_graph(d_id, doc_chunks_map[d_id], doc_results_map[d_id])
            
        return {"status": "completed"}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}
