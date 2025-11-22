from app.graph.state import AgentState
from app.tools.graph_store import retrieve_documents
from app.chains.rag_chain import get_rag_chain

def retrieve(state: AgentState):
    """
    Retrieve documents based on the current question.
    """
    question = state["question"]
    documents = retrieve_documents(question)
    # Extract content from documents
    doc_contents = [doc.page_content for doc in documents]
    return {"documents": doc_contents, "steps": ["retrieve"]}

def generate(state: AgentState):
    """
    Generate answer using RAG chain.
    """
    question = state["question"]
    documents = state["documents"]
    
    # Combine documents into a single context string
    context = "\n\n".join(documents)
    
    chain = get_rag_chain()
    generation = chain.invoke({"context": context, "question": question})
    
    return {"generation": generation, "steps": ["generate"]}
