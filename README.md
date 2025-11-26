# Knowledge RAG 🧠

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.26-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-5.18-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local-black)
![Docker](https://img.shields.io/badge/Docker-Container-blue)

**Knowledge RAG** is a cutting-edge Retrieval-Augmented Generation (RAG) application that leverages **Knowledge Graphs** and **Local LLMs** to provide accurate, context-aware answers. Built with **LangGraph** for agentic workflows and **Neo4j** for hybrid vector and graph storage, it offers a robust solution for complex query answering.

## 🚀 Features

- **Graph RAG**: Combines the power of semantic vector search with structured knowledge graph traversal using **Neo4j**.
- **Agentic Workflow**: Orchestrated by **LangGraph**, allowing for multi-step reasoning, retrieval, and generation.
- **Local Privacy**: Fully supports local LLMs via **Ollama** (e.g., Llama 3, Qwen, Mistral) for both embeddings and generation.
- **Interactive UI**: Clean and responsive chat interface built with **Streamlit**.
- **Production Ready**: Containerized with **Docker** and **Docker Compose** for easy deployment.
- **Observability**: Integrated with **LangSmith** for full trace visibility and debugging.

## 🛠️ Tech Stack

- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://www.langchain.com/)
- **Database**: [Neo4j](https://neo4j.com/) (Graph + Vector)
- **LLM Serving**: [Ollama](https://ollama.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Containerization**: Docker

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1.  **Docker & Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/)
2.  **Ollama**: [Install Ollama](https://ollama.com/)
3.  **Git**: [Install Git](https://git-scm.com/)

## 🚀 Running the API

### 1. Local Development
Start the FastAPI server:
```bash
python -m uvicorn app.server:app --reload
```
The API will be available at `http://localhost:8000`.
- **Docs**: `http://localhost:8000/docs`

### 2. Docker Deployment
Build and run the container:
```bash
docker build -t knowledge-rag .
docker run -p 8000:8000 --env-file .env knowledge-rag
```

## 📡 API Endpoints

### `POST /ingest`
Upload documents (PDF, DOCX, TXT) to the Knowledge Graph.
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/document.pdf"
```

### `POST /chat`
Chat with the agent.
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "thread_id": "session_123"
  }'
```

## 🖥️ Running the UI (Streamlit)
To run the Streamlit interface locally:
```bash
streamlit run app/ui/streamlit_app.py
```

## ⚡ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/fahadazizz/Knowledg-RAG.git
cd knowledge_rag
```

### 2. Configure Environment

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

Update the `.env` file with your configuration. **Note**: Since we are using Ollama and local Neo4j, most defaults will work out of the box, but you may need to pull your specific Ollama models first.

```ini
# .env configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3-vl:235b-cloud
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest

NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
```

> **Important**: If running in Docker, use `host.docker.internal` to access Ollama running on your host machine.

### 3. Pull Ollama Models

Make sure you have the models pulled locally:

```bash
ollama pull qwen3-vl:235b-cloud
ollama pull embeddinggemma:latest
```

### 4. Run with Docker

Start the application and the Neo4j database:

```bash
docker-compose up --build
```

- **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474)

## 🏃‍♂️ Running Locally (Development)

If you prefer to run the Python app locally while keeping Neo4j in Docker:

1.  **Start Neo4j**:
    ```bash
    docker-compose up neo4j -d
    ```

2.  **Install Dependencies**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app/ui/streamlit_app.py
    ```

## 📂 Project Structure

```
knowledge_rag/
├── app/
│   ├── chains/         # LangChain definitions
│   ├── graph/          # LangGraph state and workflow
│   ├── tools/          # Neo4j and retrieval tools
│   ├── ui/             # Streamlit frontend
│   └── utils/          # Configuration and helpers
├── docker-compose.yml  # Docker services config
├── Dockerfile          # App container definition
├── langgraph.json      # LangGraph Studio config
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
