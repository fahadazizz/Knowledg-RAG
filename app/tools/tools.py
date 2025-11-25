from langchain_core.tools import tool
from typing import List
from app.tools.graph_store import retrieve_documents

@tool
def search_knowledge_graph(query: str) -> List[str]:
    """
    Search the Knowledge Graph for relevant documents and entities.
    Use this tool when you need to answer questions based on the uploaded documents.
    
    Args:
        query (str): The search query (keywords or entity names).
        
    Returns:
        List[str]: A list of relevant text sections from the documents.
    """
    try:
        results = retrieve_documents(query)
        if not results:
            return ["No relevant information found in the knowledge graph."]
        return results
    except Exception as e:
        return [f"Error searching knowledge graph: {e}"]
