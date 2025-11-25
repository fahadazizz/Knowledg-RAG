from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from app.tools.tools import search_knowledge_graph
from app.utils.config import settings

def create_graph():
    """
    Creates a ReAct-style agent graph with persistence.
    """
    # 1. Initialize Model
    model = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0
    )
    
    # 2. Define Tools
    tools = [search_knowledge_graph]
    
    # 3. Initialize Checkpointer for persistence
    memory = MemorySaver()
    
    # 4. Create ReAct Agent
    # This prebuilt function creates a graph with 'agent' and 'tools' nodes
    # and handles the conditional routing.
    app = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory
    )
    
    return app

# Compile the graph
graph = create_graph()
