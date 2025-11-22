"""
Document processing pipeline orchestrator.
Handles the complete flow: Text Extraction → Cleaning → Chunking → KG Construction → Vector Storage
"""
from typing import List, Any
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
from app.utils.text_cleaner import clean_text
from app.utils.kg_builder import KnowledgeGraphBuilder
from app.utils.config import settings

class DocumentPipeline:
    """Orchestrates the complete document processing pipeline."""
    
    def __init__(self):
        self.kg_builder = KnowledgeGraphBuilder()
        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Semantic chunking based on embedding similarity
        # This splits text where semantic meaning changes
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",  # Use percentile-based threshold
            breakpoint_threshold_amount=95  # Split at 95th percentile of similarity scores
        )
    
    def process_documents(self, documents: List[str]) -> dict:
        """
        Process uploaded documents through the complete pipeline.
        
        Pipeline steps:
        1. Text Cleaning
        2. Semantic Chunking
        3. Entity & Relation Extraction
        4. Knowledge Graph Construction
        5. Vector Embedding & Storage
        
        Args:
            documents: List of document text strings
            
        Returns:
            Processing results summary
        """
        results = {
            "total_documents": len(documents),
            "total_chunks": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "status": "success"
        }
        
        try:
            all_chunks = []
            
            # Step 1 & 2: Clean and chunk documents
            print("Step 1-2: Cleaning and chunking documents...")
            for doc in documents:
                cleaned_text = clean_text(doc)
                chunks = self.text_splitter.split_text(cleaned_text)
                all_chunks.extend(chunks)
            
            results["total_chunks"] = len(all_chunks)
            print(f"Created {len(all_chunks)} chunks")
            
            # Step 3-4: Build knowledge graph
            print("Step 3-4: Building knowledge graph...")
            self.kg_builder.build_graph_from_text(all_chunks)
            
            # Step 5: Create vector embeddings and store in Neo4j
            print("Step 5: Creating vector embeddings...")
            self._store_vectors(all_chunks)
            
            print("✅ Pipeline processing complete!")
            return results
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def _store_vectors(self, chunks: List[str]) -> None:
        """
        Create and store vector embeddings in Neo4j.
        
        Args:
            chunks: Text chunks to embed
        """
        try:
            # Create Document nodes with embeddings
            for i, chunk in enumerate(chunks):
                # Store chunk as a Document node in Neo4j
                vector_store = Neo4jVector.from_texts(
                    texts=[chunk],
                    embedding=self.embeddings,
                    url=settings.NEO4J_URI,
                    username=settings.NEO4J_USERNAME,
                    password=settings.NEO4J_PASSWORD,
                    index_name="vector",
                    node_label="Document",
                    text_node_property="text",
                    embedding_node_property="embedding",
                )
                
                if (i + 1) % 10 == 0:
                    print(f"Embedded {i + 1}/{len(chunks)} chunks...")
            
            print(f"✅ Stored {len(chunks)} embeddings in Neo4j")
            
        except Exception as e:
            print(f"Error storing vectors: {e}")
            raise e
    
    def clear_all_data(self) -> None:
        """Clear all data from Neo4j (graph and vectors)."""
        self.kg_builder.clear_graph()
        print("All data cleared from Neo4j")
