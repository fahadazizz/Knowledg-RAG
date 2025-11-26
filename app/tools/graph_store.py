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

def retrieve_subgraph(query: str, limit: int = 5):
    """
    Retrieves a structured subgraph based on the query.
    1. Finds entities matching the query.
    2. Traverses 1-hop neighbors.
    3. Fetches linked text sections.
    """
    try:
        graph = get_graph_store()
        
        # 1. Find relevant entities and their 1-hop neighborhood
        # We search for entities whose names contain the query keywords
        # and return the entity, its relationships, and neighbors
        cypher_graph = f"""
        MATCH (e)
        WHERE e.name IS NOT NULL AND toLower(e.name) CONTAINS toLower($query)
        WITH e LIMIT {limit}
        MATCH (e)-[r]-(neighbor)
        RETURN e.name as entity, type(r) as relation, neighbor.name as neighbor, neighbor.type as neighbor_type
        LIMIT 20
        """
        
        graph_results = graph.query(cypher_graph, {"query": query})
        
        # 2. Find relevant text sections linked to these entities
        cypher_text = f"""
        MATCH (e)
        WHERE e.name IS NOT NULL AND toLower(e.name) CONTAINS toLower($query)
        WITH e LIMIT {limit}
        MATCH (s:Section)-[:MENTIONS]->(e)
        RETURN DISTINCT s.text as text
        LIMIT {limit}
        """
        
        text_results = graph.query(cypher_text, {"query": query})
        
        # Format the output
        structured_info = []
        
        if graph_results:
            structured_info.append("### Knowledge Graph Connections:")
            for row in graph_results:
                structured_info.append(f"- {row['entity']} --[{row['relation']}]--> {row['neighbor']} ({row['neighbor_type']})")
        
        if text_results:
            structured_info.append("\n### Relevant Source Text:")
            for row in text_results:
                structured_info.append(f"- ...{row['text'][:300]}...")
                
        return structured_info
        
    except Exception as e:
        print(f"Error retrieving subgraph: {e}")
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
