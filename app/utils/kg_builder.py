"""
Knowledge Graph construction and management using Neo4j.
"""
from typing import List, Dict, Any
from app.tools.graph_store import get_graph_store
from langchain_ollama import ChatOllama
from app.utils.config import settings
import json
import concurrent.futures

class KnowledgeGraphBuilder:
    """Handles construction of knowledge graphs in Neo4j."""
    
    def __init__(self):
        self.graph = get_graph_store()
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0  # Deterministic output
        )
    
    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using LLM with improved prompting.
        """
        prompt = f"""
        You are an expert Knowledge Graph Engineer. Your task is to extract structured information from the provided text.
        Identify key entities and the relationships between them.
        
        Guidelines:
        1. **Entities**: Extract significant entities (Person, Organization, Location, Concept, Event, Technology, etc.).
           - Normalize names (e.g., "Google Inc." -> "Google").
           - Avoid generic terms like "company", "user", "it".
        2. **Relations**: Extract meaningful relationships between these entities.
           - Use active verbs (e.g., WORKS_FOR, LOCATED_IN, USES, DEVELOPED_BY).
        3. **Output**: Return ONLY a valid JSON object.
        
        Text:
        {text}
        
        JSON Structure:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "ENTITY_TYPE", "properties": {{"description": "brief description"}}}},
                ...
            ],
            "relations": [
                {{"source": "Entity Name 1", "target": "Entity Name 2", "type": "RELATION_TYPE", "properties": {{"context": "brief context"}}}},
                ...
            ]
        }}
        """

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

    
    def extract_from_chunks(self, chunk_texts: List[str], chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities and relations from chunks in parallel without writing to DB.
        Returns list of results with chunk_id included.
        """
        print(f"Extracting from {len(chunk_texts)} chunks...")
        
        # Function to process a single chunk
        def process_chunk_extraction(chunk_data):
            text, chunk_id = chunk_data
            result = self.extract_entities_and_relations(text)
            result["chunk_id"] = chunk_id
            return result

        results = []
        chunk_data_list = list(zip(chunk_texts, chunk_ids))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {executor.submit(process_chunk_extraction, data): i for i, data in enumerate(chunk_data_list)}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                i = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Extracted from chunk {i+1}/{len(chunk_texts)}")
                except Exception as e:
                    print(f"Error extracting from chunk {i+1}: {e}")
                    
        return results

    def build_pure_graph(self, doc_id: str, chunks: List[Dict[str, Any]], extraction_results: List[Dict[str, Any]]) -> None:
        """
        Build the pure knowledge graph in Neo4j.
        Schema:
        (Document {id}) -[:HAS_SECTION]-> (Section {id, text})
        (Section) -[:MENTIONS]-> (Entity)
        (Entity) -[:RELATION]-> (Entity)
        """
        print(f"Building pure graph for Document {doc_id}...")
        
        try:
            # 1. Create Document Node
            self.graph.query(
                "MERGE (d:Document {id: $id})", 
                {"id": doc_id}
            )
            
            # 2. Create Section Nodes and link to Document
            # Batch this
            section_batch = [{"id": c["id"], "text": c["text"], "doc_id": doc_id} for c in chunks]
            
            query_sections = """
            UNWIND $batch as row
            MATCH (d:Document {id: row.doc_id})
            MERGE (s:Section {id: row.id})
            SET s.text = row.text
            MERGE (d)-[:HAS_SECTION]->(s)
            """
            self.graph.query(query_sections, {"batch": section_batch})
            
            # 3. Create Entities and Relations (using batch writer)
            self._batch_write_pure_graph(extraction_results)
            
            print(f"✅ Graph built for Document {doc_id}")
            
        except Exception as e:
            print(f"Error building pure graph: {e}")
            raise e

    def _batch_write_pure_graph(self, results: List[Dict[str, Any]]) -> None:
        """
        Write extracted entities and relations to Neo4j, linking to Sections.
        """
        all_entities = {}
        all_relations = []
        mentions = [] # (chunk_id, entity_name, entity_type)
        
        for res in results:
            chunk_id = res.get("chunk_id")
            
            for entity in res.get("entities", []):
                key = (entity["name"], entity["type"])
                if key in all_entities:
                    all_entities[key]["properties"].update(entity.get("properties", {}))
                else:
                    all_entities[key] = entity
                
                if chunk_id:
                    mentions.append({"chunk_id": chunk_id, "entity_name": entity["name"], "entity_type": entity["type"]})
            
            for rel in res.get("relations", []):
                all_relations.append(rel)
        
        # 1. Batch create entities
        entity_list = [{"name": k[0], "type": k[1], "properties": v.get("properties", {})} for k, v in all_entities.items()]
        
        entities_by_type = {}
        for ent in entity_list:
            etype = ent["type"]
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(ent)
            
        for etype, batch in entities_by_type.items():
            query = f"""
            UNWIND $batch AS row
            MERGE (e:`{etype}` {{name: row.name}})
            SET e += row.properties
            """
            try:
                self.graph.query(query, {"batch": batch})
            except Exception as e:
                print(f"Error batch creating entities of type {etype}: {e}")

        # 2. Batch create relations
        relations_by_type = {}
        for rel in all_relations:
            rtype = rel.get("type", "RELATED_TO")
            if rtype not in relations_by_type:
                relations_by_type[rtype] = []
            relations_by_type[rtype].append(rel)
            
        for rtype, batch in relations_by_type.items():
            query = f"""
            UNWIND $batch AS row
            MATCH (source {{name: row.source}})
            MATCH (target {{name: row.target}})
            MERGE (source)-[r:`{rtype}`]->(target)
            SET r += row.properties
            """
            try:
                self.graph.query(query, {"batch": batch})
            except Exception as e:
                print(f"Error batch creating relations of type {rtype}: {e}")
                
        # 3. Batch create MENTIONS relationships (Section -> Entity)
        if mentions:
            query = """
            UNWIND $batch AS row
            MATCH (s:Section {id: row.chunk_id})
            MATCH (e {name: row.entity_name})
            MERGE (s)-[:MENTIONS]->(e)
            """
            try:
                batch_size = 1000
                for i in range(0, len(mentions), batch_size):
                    batch = mentions[i:i + batch_size]
                    self.graph.query(query, {"batch": batch})
            except Exception as e:
                print(f"Error linking sections to entities: {e}")

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("Graph cleared successfully!")
        except Exception as e:
            print(f"Error clearing graph: {e}")
