from app.graph.state import AgentState
from app.tools.graph_store import retrieve_documents
from app.chains.rag_chain import get_rag_chain

def retrieve(state: AgentState):
    """
    Retrieve documents based on the current question.
    """
    question = state["question"]
    try:
        documents = retrieve_documents(question)
        # Extract content from documents
        doc_contents = [doc.page_content for doc in documents] if documents else []
        return {"documents": doc_contents, "steps": ["retrieve"]}
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"documents": [], "steps": ["retrieve_error"]}

def generate(state: AgentState):
    """
    Generate answer using RAG chain.
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"generation": "I couldn't find any relevant documents to answer your question.", "steps": ["generate_no_docs"]}
    
    # Combine documents into a single context string
    context = "\n\n".join(documents)
    
    try:
        chain = get_rag_chain()
        generation = chain.invoke({"context": context, "question": question})
        return {"generation": generation, "steps": ["generate"]}
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"generation": "An error occurred while generating the answer.", "steps": ["generate_error"]}
