from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from app.utils.config import settings

def get_rag_chain():
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful and accurate assistant. 
        Answer the user's question based ONLY on the provided context below. 
        If the answer is not present in the context, simply state "I don't know based on the available information." 
        Do not make up answers or use outside knowledge.
        
        Context:
        {context}
        
        Question: {question}
        """
    )
    
    model = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )
    
    chain = prompt | model | StrOutputParser()
    return chain
