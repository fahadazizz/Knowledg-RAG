from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from app.tools.tools import search_knowledge_graph
from app.utils.config import settings

def create_graph(with_memory: bool = False):
    """
    Creates a ReAct-style agent graph.
    Args:
        with_memory (bool): If True, adds a MemorySaver checkpointer. 
                            Set to False for LangGraph Studio (which handles persistence).
    """
    # 1. Initialize Model
    model = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0
    )
    
    # 2. Define Tools
    tools = [search_knowledge_graph]
    
    # 3. Initialize Checkpointer for persistence (if requested)
    memory = MemorySaver() if with_memory else None
    
    # 4. Create ReAct Agent
    system_prompt = """You are a helpful Knowledge Graph RAG Assistant.
    Your goal is to answer user questions based on the provided Knowledge Graph.
    
    INSTRUCTIONS:
    1. ALWAYS use the `search_knowledge_graph` tool to find information. Do not answer from your own training data.
    2. The tool returns structured data: Entities, Relationships, and Source Text.
    3. Use this structure to explain connections (e.g., "X is related to Y because...").
    4. Cite the source text when possible.
    5. If the tool returns "No relevant information", state that clearly.
    """
    
    app = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory
    )
    
    return app

# Export graphs
# 'graph' is for LangGraph Studio (no built-in memory, platform handles it)
graph = create_graph(with_memory=False)

# 'local_graph' is for local execution (e.g., Streamlit) where we need explicit memory
local_graph = create_graph(with_memory=True)
