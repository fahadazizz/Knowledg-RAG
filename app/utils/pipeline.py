"""
Pure Knowledge Graph RAG Pipeline.
"""
from typing import List, Dict, Any
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.text_cleaner import clean_text
from app.utils.kg_builder import KnowledgeGraphBuilder
from app.utils.config import settings

class DocumentPipeline:
    """
    Pure Knowledge Graph document processing pipeline.
    Stages:
    1. Clean Text
    2. Chunk Text (Structural)
    3. Extract KG (Entities + Relations)
    4. Canonicalize Entities
    5. Validate Relations
    6. Build Graph (Neo4j)
    """
    
    def __init__(self):
        # Structural chunking instead of semantic
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.kg_builder = KnowledgeGraphBuilder()
        
    def clean_text(self, text: str) -> str:
        """Stage 1: Clean text."""
        return clean_text(text)
        
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Stage 2: Chunk text structurally.
        Returns list of dicts with 'text' and 'id'.
        """
        chunks = self.text_splitter.split_text(text)
        return [{"text": chunk, "id": str(uuid.uuid4())} for chunk in chunks]

    def extract_kg(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: Extract Entities and Relations from chunks.
        """
        print(f"Extracting KG from {len(chunks)} chunks...")
        # We'll use the KG builder's extraction logic but manage it here
        # to allow for intermediate steps like canonicalization
        
        # Prepare data for parallel extraction
        chunk_texts = [c["text"] for c in chunks]
        chunk_ids = [c["id"] for c in chunks]
        
        # Use KG Builder's parallel extraction
        # We need to modify KG Builder to return results without writing to DB yet
        # For now, we'll assume we can use a method that returns results
        return self.kg_builder.extract_from_chunks(chunk_texts, chunk_ids)

    def canonicalize_entities(self, extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 4: Canonicalize Entities.
        Merge similar entities (e.g., "Google" vs "Google Inc").
        Simple implementation: Lowercase normalization and type matching.
        """
        print("Canonicalizing entities...")
        # This is a simplified version. In a real prod system, this would use
        # vector similarity or fuzzy matching.
        
        # Map normalized name -> canonical name
        entity_map = {}
        
        for res in extraction_results:
            for entity in res.get("entities", []):
                norm_name = entity["name"].lower().strip()
                if norm_name not in entity_map:
                    entity_map[norm_name] = entity["name"]
                
                # Update entity name to canonical
                entity["name"] = entity_map[norm_name]
                
            # Update relations to use canonical names
            for rel in res.get("relations", []):
                source_norm = rel["source"].lower().strip()
                target_norm = rel["target"].lower().strip()
                
                if source_norm in entity_map:
                    rel["source"] = entity_map[source_norm]
                if target_norm in entity_map:
                    rel["target"] = entity_map[target_norm]
                    
        return extraction_results

    def validate_relations(self, extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 5: Validate Relations.
        Ensure relations have valid source/target and types.
        """
        print("Validating relations...")
        for res in extraction_results:
            valid_relations = []
            entities_in_chunk = {e["name"] for e in res.get("entities", [])}
            
            for rel in res.get("relations", []):
                # Rule 1: Source and Target must exist (or at least be non-empty)
                if not rel["source"] or not rel["target"]:
                    continue
                
                # Rule 2: No self-loops unless specific types
                if rel["source"] == rel["target"]:
                    continue
                
                # Rule 3: Ensure type exists
                if not rel.get("type"):
                    rel["type"] = "RELATED_TO"
                    
                valid_relations.append(rel)
            
            res["relations"] = valid_relations
            
        return extraction_results

    def build_graph(self, doc_id: str, chunks: List[Dict[str, Any]], extraction_results: List[Dict[str, Any]]) -> None:
        """
        Stage 6: Build Graph in Neo4j.
        Schema:
        (Document {id}) -[:HAS_SECTION]-> (Section {id, text})
        (Section) -[:MENTIONS]-> (Entity)
        (Entity) -[:RELATION]-> (Entity)
        """
        print("Building graph in Neo4j...")
        self.kg_builder.build_pure_graph(doc_id, chunks, extraction_results)

    def process_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Orchestrate the pipeline.
        """
        results = {
            "status": "pending",
            "total_documents": len(documents),
            "total_chunks": 0,
            "error": None
        }
        
        try:
            for doc_text in documents:
                # 1. Clean
                cleaned_text = self.clean_text(doc_text)
                
                # Generate Document ID
                doc_id = str(uuid.uuid4())
                
                # 2. Chunk
                chunks = self.chunk_text(cleaned_text)
                results["total_chunks"] += len(chunks)
                
                # 3. Extract
                extraction_results = self.extract_kg(chunks)
                
                # 4. Canonicalize
                extraction_results = self.canonicalize_entities(extraction_results)
                
                # 5. Validate
                extraction_results = self.validate_relations(extraction_results)
                
                # 6. Build
                self.build_graph(doc_id, chunks, extraction_results)
            
            results["status"] = "success"
            return results
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def clear_all_data(self) -> None:
        """Clear all data from Neo4j (graph and vectors)."""
        self.kg_builder.clear_graph()
        print("All data cleared from Neo4j")
