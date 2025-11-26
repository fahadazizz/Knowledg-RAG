from langchain_core.tools import tool
from typing import List
from app.tools.graph_store import retrieve_subgraph

@tool
def search_knowledge_graph(query: str) -> List[str]:
    """
    Search the Knowledge Graph for relevant entities, relationships, and text.
    Use this tool to find structured connections and source context.
    
    Args:
        query (str): The search query (keywords or entity names).
        
    Returns:
        List[str]: A list of structured graph connections and relevant text sections.
    """
    try:
        results = retrieve_subgraph(query)
        if not results:
            return ["No relevant information found in the knowledge graph."]
        return results
    except Exception as e:
        return [f"Error searching knowledge graph: {e}"]
