"""
Knowledge Graph construction and management using Neo4j.
"""
from typing import List, Dict, Any
from app.tools.graph_store import get_graph_store
from langchain_ollama import ChatOllama
from app.utils.config import settings
import json

class KnowledgeGraphBuilder:
    """Handles construction of knowledge graphs in Neo4j."""
    
    def __init__(self):
        self.graph = get_graph_store()
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
    
    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using LLM.
        
        Args:
            text: Text chunk to process
            
        Returns:
            Dictionary containing entities and relations
        """
        prompt = f"""Extract entities and relationships from the following text using the Universal Knowledge Graph Schema.

You MUST obey the following constraints:

ENTITY TYPES (Strict):
DOCUMENT, SECTION, PERSON, ORGANIZATION, CONCEPT,
TECHNOLOGY, COMPONENT, PROCESS, ATTRIBUTE,
EVENT, LOCATION, OUTCOME

RELATION TYPES (Strict):
HAS_SECTION, MENTIONS, AUTHORED_BY, WORKS_FOR,
USES, IMPLEMENTS, HAS_ATTRIBUTE, PRODUCES,
INTERACTS_WITH, LOCATED_AT, PART_OF, DEFINES

MANDATORY EXTRACTION RULES:
1. ALWAYS create exactly ONE DOCUMENT entity.
   - Use the provided document title or infer a short name.
   - All other entities MUST connect back to this DOCUMENT.

2. Extract SECTION entities whenever the text shows structure 
   (headers, bullets, subsections).  
   - Connect them using HAS_SECTION from DOCUMENT → SECTION.

3. Extract ONLY explicit entities or relations directly stated in the text.  
   - No assumptions. No filling gaps.

4. Every entity MUST have:
   - "name"
   - "type"
   - "properties": include a "source_span" with the exact phrase extracted.

5. Every relation MUST use ONLY allowed relation types.

6. Every extracted entity (except DOCUMENT) MUST be connected using at least ONE relation.
   - No floating nodes.

7. Normalize similar entities:
   - lowercase technology names
   - merge repeated names into a single canonical representation


OUTPUT FORMAT (Strict JSON):
{{
  "entities": [
    {{"name": "string", "type": "ENTITY_TYPE", "properties": {{"source_span": "text"}}}},
    ...
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "type": "RELATION_TYPE"}},
    ...
  ]
}}

Text:
{text}

Return ONLY valid JSON."""

        try:
            response = self.llm.invoke(prompt).content
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            result = json.loads(response)
            return result
        except Exception as e:
            print(f"Error extracting entities and relations: {e}")
            return {"entities": [], "relations": []}
    
    def create_entity(self, entity: Dict[str, Any]) -> None:
        """
        Create an entity node in Neo4j.
        
        Args:
            entity: Entity dictionary with name, type, and properties
        """
        try:
            name = entity.get("name", "")
            entity_type = entity.get("type", "CONCEPT")
            properties = entity.get("properties", {})
            
            # Create node with properties
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            if props_str:
                props_str = ", " + props_str
            
            query = f"""
            MERGE (e:{entity_type} {{name: $name{props_str}}})
            RETURN e
            """
            
            params = {"name": name, **properties}
            self.graph.query(query, params)
        except Exception as e:
            print(f"Error creating entity {entity}: {e}")
    
    def create_relation(self, relation: Dict[str, Any]) -> None:
        """
        Create a relationship between entities in Neo4j.
        
        Args:
            relation: Relation dictionary with source, target, and type
        """
        try:
            source = relation.get("source", "")
            target = relation.get("target", "")
            rel_type = relation.get("type", "RELATED_TO")
            
            query = f"""
            MATCH (a {{name: $source}})
            MATCH (b {{name: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            RETURN r
            """
            
            self.graph.query(query, {"source": source, "target": target})
        except Exception as e:
            print(f"Error creating relation {relation}: {e}")
    
    def build_graph_from_text(self, text_chunks: List[str]) -> None:
        """
        Process text chunks and build knowledge graph using parallel processing.
        
        Args:
            text_chunks: List of text chunks to process
        """
        print(f"Building knowledge graph from {len(text_chunks)} chunks...")
        
        import concurrent.futures
        
        # Function to process a single chunk (extraction only)
        def process_chunk_extraction(chunk):
            return self.extract_entities_and_relations(chunk)

        # Parallel extraction
        results = []
        print("Extracting entities and relations in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_chunk = {executor.submit(process_chunk_extraction, chunk): i for i, chunk in enumerate(text_chunks)}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                i = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Processed chunk {i+1}/{len(text_chunks)}")
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {e}")

        # Sequential writing to avoid database locking issues
        print("Writing to Neo4j...")
        for i, result in enumerate(results):
            # Create entities
            for entity in result.get("entities", []):
                self.create_entity(entity)
            
            # Create relations
            for relation in result.get("relations", []):
                self.create_relation(relation)
        
        print("Knowledge graph construction complete!")
    
    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("Graph cleared successfully!")
        except Exception as e:
            print(f"Error clearing graph: {e}")
