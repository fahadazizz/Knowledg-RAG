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
        prompt = f"""Extract entities and relationships from the following text.
        
Format your response as valid JSON with this structure:
{{
    "entities": [
        {{"name": "entity_name", "type": "entity_type", "properties": {{}}}},
        ...
    ],
    "relations": [
        {{"source": "entity1", "target": "entity2", "type": "relation_type"}},
        ...
    ]
}}

Entity types can be: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, etc.
Relation types can be: WORKS_FOR, LOCATED_IN, PART_OF, CREATED, RELATED_TO, etc.

Text: {text}

Return ONLY the JSON, no other text."""

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
        Process text chunks and build knowledge graph.
        
        Args:
            text_chunks: List of text chunks to process
        """
        print(f"Building knowledge graph from {len(text_chunks)} chunks...")
        
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}...")
            
            # Extract entities and relations
            result = self.extract_entities_and_relations(chunk)
            
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
