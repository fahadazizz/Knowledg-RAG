from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from app.utils.config import settings

def get_rag_chain():
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
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
