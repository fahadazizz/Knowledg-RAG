"""
Knowledge RAG - Enhanced Streamlit Application
Multi-page UI with file upload, chat, and graph visualization.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from app.graph.workflow import graph
from app.utils.document_processor import process_uploaded_file

st.set_page_config(
    page_title="Knowledge RAG", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=Knowledge+RAG", use_container_width=True)
    st.markdown("### 🧭 Navigation")
    
    if st.button("📤 Upload Documents", use_container_width=True):
        st.session_state.current_page = "upload"
    
    if st.button("💬 Chat with Agent", use_container_width=True, disabled=not st.session_state.documents_uploaded):
        st.session_state.current_page = "chat"
    
    if st.button("🗺️ Graph Visualization", use_container_width=True, disabled=not st.session_state.documents_uploaded):
        st.session_state.current_page = "graph"
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    if st.session_state.documents_uploaded:
        st.success("✅ Documents Loaded")
    else:
        st.warning("⏳ No Documents")

# Main content area
if st.session_state.current_page == "upload":
    st.markdown('<h1 class="main-header">📤 Upload Your Documents</h1>', unsafe_allow_html=True)
    st.markdown("Upload your documents (PDF, Word, DOCX) to build your knowledge base.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or Word documents"
        )
        
        if uploaded_files:
            st.markdown("### 📝 Uploaded Files")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.2f} KB)")
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    try:
                        # Import pipeline
                        from app.utils.pipeline import DocumentPipeline
                        from app.utils.document_processor import process_uploaded_file
                        
                        # Extract text from all files
                        all_text = []
                        for file in uploaded_files:
                            file.seek(0)  # Reset file pointer
                            text = process_uploaded_file(file)
                            all_text.append(text)
                        
                        # Store in session state
                        st.session_state.document_text = "\n\n".join(all_text)
                        
                        # Process through pipeline
                        pipeline = DocumentPipeline()
                        
                        # Show progress
                        with st.status("Processing pipeline...", expanded=True) as status:
                            st.write("📄 Cleaning documents...")
                            st.write("✂️ Creating semantic chunks...")
                            st.write("🔍 Extracting entities and relations...")
                            st.write("🗺️ Building knowledge graph...")
                            st.write("🧮 Creating vector embeddings...")
                            
                            results = pipeline.process_documents(all_text)
                            
                            if results["status"] == "success":
                                status.update(label="✅ Processing complete!", state="complete")
                            else:
                                status.update(label="❌ Processing failed", state="error")
                        
                        # Display results
                        if results["status"] == "success":
                            st.session_state.documents_uploaded = True
                            st.success("✅ Documents processed successfully!")
                            
                            # Show stats
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Chunks", results["total_chunks"])
                            with col2:
                                st.metric("Documents", results["total_documents"])
                            
                            st.balloons()
                            st.info("🎯 Navigate to **Chat with Agent** to start asking questions or view the **Graph Visualization**.")
                        else:
                            st.error(f"❌ Error: {results.get('error', 'Unknown error')}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {e}")
                        import traceback
                        st.code(traceback.format_exc())

elif st.session_state.current_page == "chat":
    st.markdown('<h1 class="main-header">💬 Chat with Knowledge RAG Agent</h1>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    inputs = {"question": prompt}
                    result = graph.invoke(inputs)
                    response = result.get("generation", "I couldn't generate an answer.")
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Show retrieval details in expander
                    with st.expander("🔍 View Retrieved Documents"):
                        if "documents" in result and result["documents"]:
                            for i, doc in enumerate(result["documents"], 1):
                                st.markdown(f"**Document {i}:**")
                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                        else:
                            st.info("No documents were retrieved.")
                            
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif st.session_state.current_page == "graph":
    st.markdown('<h1 class="main-header">🗺️ Knowledge Graph Visualization</h1>', unsafe_allow_html=True)
    st.info("📊 This is a temporary visualization. The actual graph will be built from your documents.")
    
    # Temporary graph visualization
    nodes = [
        Node(id="AI", label="Artificial Intelligence", size=25, color="#667eea"),
        Node(id="ML", label="Machine Learning", size=20, color="#764ba2"),
        Node(id="DL", label="Deep Learning", size=20, color="#f093fb"),
        Node(id="NLP", label="Natural Language Processing", size=18, color="#4facfe"),
        Node(id="CV", label="Computer Vision", size=18, color="#00f2fe"),
        Node(id="RAG", label="Retrieval Augmented Generation", size=22, color="#43e97b"),
    ]
    
    edges = [
        Edge(source="AI", target="ML", label="includes"),
        Edge(source="ML", target="DL", label="subset of"),
        Edge(source="ML", target="NLP", label="enables"),
        Edge(source="ML", target="CV", label="enables"),
        Edge(source="NLP", target="RAG", label="used in"),
    ]
    
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
    )
    
    agraph(nodes=nodes, edges=edges, config=config)
    
    st.markdown("### 📈 Graph Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", len(nodes))
    with col2:
        st.metric("Edges", len(edges))
    with col3:
        st.metric("Depth", "3")
