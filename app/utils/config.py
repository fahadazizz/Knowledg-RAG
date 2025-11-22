import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3-vl:235b-cloud"
    OLLAMA_EMBEDDING_MODEL: str = "embeddinggemma:latest" # Default, user will provide later
    NEO4J_URI: str = "neo4j://127.0.0.1:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str = "knowledge-rag"

    class Config:
        env_file = ".env"

settings = Settings()
