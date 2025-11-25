from langgraph.graph import StateGraph, END
from app.graph.ingestion.state import IngestionState
from app.graph.ingestion.nodes import clean_node, chunk_node, extract_node, build_node

def create_ingestion_graph():
    """
    Creates the Ingestion Graph.
    """
    workflow = StateGraph(IngestionState)
    
    # Add nodes
    workflow.add_node("clean", clean_node)
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("build", build_node)
    
    # Set entry point
    workflow.set_entry_point("clean")
    
    # Add edges
    workflow.add_edge("clean", "chunk")
    workflow.add_edge("chunk", "extract")
    workflow.add_edge("extract", "build")
    workflow.add_edge("build", END)
    
    # Compile
    app = workflow.compile()
    return app

ingestion_graph = create_ingestion_graph()
