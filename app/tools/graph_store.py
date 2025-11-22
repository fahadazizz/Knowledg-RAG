from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import OllamaEmbeddings
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

def get_vector_store():
    """
    Returns a Neo4jVector store instance using Ollama embeddings.
    """
    try:
        embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        # Using Neo4j as a vector store
        return Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            index_name="vector", # Default index name
            node_label="Document", # Assuming nodes are labeled 'Document'
            text_node_properties=["text"], # Assuming text is in 'text' property
            embedding_node_property="embedding",
        )
    except Exception as e:
        print(f"Failed to initialize Neo4jVector: {e}")
        raise e

def retrieve_documents(query: str, k: int = 4):
    """
    Retrieves documents similar to the query using vector search.
    
    Args:
        query (str): The search query.
        k (int): Number of documents to retrieve.
        
    Returns:
        List[Document]: List of retrieved documents.
    """
    # For now, we use vector search on Neo4j. 
    # In a real GraphRAG, we might use graph traversal here too.
    try:
        vector_store = get_vector_store()
        return vector_store.similarity_search(query, k=k)
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
