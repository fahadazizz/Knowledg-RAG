from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import shutil
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from app.graph.workflow import local_graph as agent_graph
from app.graph.ingestion.workflow import ingestion_graph
from app.utils.document_processor import process_uploaded_file

app = FastAPI(
    title="Knowledge RAG Agent API",
    description="API for Knowledge Graph RAG Agent",
    version="1.0.0"
)

# --- Data Models ---
class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class IngestResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    doc_ids: List[str]

# --- System Prompt ---
SYSTEM_PROMPT = """You are a helpful Knowledge Graph RAG Assistant.
Your goal is to answer user questions based on the provided Knowledge Graph.

INSTRUCTIONS:
1. ALWAYS use the `search_knowledge_graph` tool to find information. Do not answer from your own training data.
2. The tool returns structured data: Entities, Relationships, and Source Text.
3. Use this structure to explain connections (e.g., "X is related to Y because...").
4. Cite the source text when possible.
5. If the tool returns "No relevant information", state that clearly.
"""

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Knowledge RAG Agent API is running"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload and ingest documents into the Knowledge Graph.
    """
    all_text = []
    try:
        for file in files:
            # Save temp file to process
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process file
            with open(temp_path, "rb") as f:
                # We need to mock a streamlit uploaded file or adjust process_uploaded_file
                # process_uploaded_file expects a file-like object with .name
                # Let's just read text directly if possible or use the helper
                # Re-opening as binary for the helper
                pass
            
            # Clean up is tricky with the helper expecting a specific object
            # Let's just read the file content based on extension manually here for simplicity
            # or better, adapt process_uploaded_file to work with standard file objects
            
            # For now, let's assume text/plain for simplicity or read content
            content = await file.read()
            
            if file.filename.endswith(".pdf"):
                from pypdf import PdfReader
                import io
                pdf = PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                all_text.append(text)
            elif file.filename.endswith(".docx"):
                import docx
                import io
                doc = docx.Document(io.BytesIO(content))
                text = "\n".join([para.text for para in doc.paragraphs])
                all_text.append(text)
            else:
                # Assume text
                all_text.append(content.decode("utf-8"))
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

    # Run Ingestion Graph
    initial_state = {
        "raw_documents": all_text,
        "cleaned_documents": [],
        "chunks": [],
        "extraction_results": [],
        "doc_ids": [],
        "status": "started",
        "error": ""
    }
    
    try:
        result = ingestion_graph.invoke(initial_state)
        
        if result.get("status") == "completed":
            return IngestResponse(
                status="success",
                total_documents=len(result.get("doc_ids", [])),
                total_chunks=len(result.get("chunks", [])),
                doc_ids=result.get("doc_ids", [])
            )
        else:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {result.get('error')}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running ingestion graph: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the Knowledge Graph Agent.
    """
    try:
        # Prepare messages
        messages: List[BaseMessage] = []
        
        # Add System Prompt
        messages.append(SystemMessage(content=SYSTEM_PROMPT))
        
        # Add History (if any)
        if request.history:
            for msg in request.history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # We use SystemMessage or AIMessage for assistant? AIMessage is better
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add Current Query
        messages.append(HumanMessage(content=request.query))
        
        # Config for persistence
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Invoke Agent Graph
        # Note: We pass the full message history including system prompt
        # The graph state has 'messages', so we pass {"messages": messages}
        # But since we are using persistence, we might duplicate history if we pass it all every time.
        # If thread_id is new, we pass all. If existing, we should only pass new messages.
        # For simplicity in this API, we'll assume the client manages history OR we rely on thread_id.
        # If we rely on thread_id, we should only pass the NEW message.
        # BUT we need to ensure System Prompt is always there or at least initially.
        # Let's just pass the NEW message and the System Prompt (if it's not persisted, but MemorySaver persists everything).
        # Actually, if we pass SystemPrompt every time, it might get appended multiple times.
        # Best practice: Check if thread exists? No easy way.
        # Let's just pass the System Prompt + Query. The agent should handle it.
        # OR, simpler: Just pass Query. The System Prompt is missing from the graph definition now.
        # So we MUST pass it.
        # Let's pass [SystemMessage, HumanMessage].
        
        inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=request.query)]}
        
        result = agent_graph.invoke(inputs, config=config)
        
        # Get response
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            return ChatResponse(
                response=last_message.content,
                thread_id=request.thread_id
            )
        else:
            return ChatResponse(response="No response generated.", thread_id=request.thread_id)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
