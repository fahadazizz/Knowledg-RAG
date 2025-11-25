from langchain_neo4j import Neo4jGraph
from app.utils.config import settings

def get_graph_store():
    """
    Establishes and returns a connection to the Neo4j graph database.
    """
    try:
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
        )
        return graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        raise e

def retrieve_documents(query: str, k: int = 4):
    """
    Retrieves documents using keyword search on Section nodes.
    Pure Graph RAG approach (no vectors).
    """
    try:
        graph = get_graph_store()
        # Simple keyword search using CONTAINS
        # In production, use a FullText index
        cypher = f"""
        MATCH (s:Section)
        WHERE toLower(s.text) CONTAINS toLower($query)
        RETURN s.text as text
        LIMIT $k
        """
        # If query is long, this might be inefficient/ineffective without NLP extraction
        # But it satisfies the "no vector" requirement.
        
        # Better approach: Extract entities from query (using LLM ideally, but here simple)
        # and find Sections mentioning those entities.
        
        results = graph.query(cypher, {"query": query, "k": k})
        return [r['text'] for r in results]
    except Exception as e:
        print(f"Error retrieving from Neo4j: {e}")
        return []

def query_graph(query: str):
    """
    Executes a Cypher query against the Neo4j graph.
    
    Args:
        query (str): The Cypher query string.
        
    Returns:
        List[Dict]: The query results.
    """
    try:
        graph = get_graph_store()
        return graph.query(query)
    except Exception as e:
        print(f"Error executing graph query: {e}")
        return []

def get_graph_visualization_data(limit: int = 100):
    """
    Fetches nodes and relationships for visualization.
    
    Args:
        limit (int): Maximum number of relationships to fetch.
        
    Returns:
        Tuple[List[dict], List[dict]]: Lists of nodes and edges.
    """
    try:
        graph = get_graph_store()
        # Query to get nodes and relationships
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT {limit}
        """
        data = graph.query(query)
        
        nodes = {}
        edges = []
        
        for record in data:
            source = record['n']
            target = record['m']
            rel = record['r']
            
            # Process source node
            source_id = source.get('name', 'Unknown')
            source_type = list(source.get('labels', ['Concept']))[0] if hasattr(source, 'labels') else 'Concept'
            nodes[source_id] = {'id': source_id, 'label': source_id, 'type': source_type}
            
            # Process target node
            target_id = target.get('name', 'Unknown')
            target_type = list(target.get('labels', ['Concept']))[0] if hasattr(target, 'labels') else 'Concept'
            nodes[target_id] = {'id': target_id, 'label': target_id, 'type': target_type}
            
            # Process edge
            edges.append({
                'source': source_id,
                'target': target_id,
                'label': rel[1] if isinstance(rel, tuple) else rel.get('type', 'RELATED_TO')
            })
            
        return list(nodes.values()), edges
    except Exception as e:
        print(f"Error fetching graph data: {e}")
        return [], []
