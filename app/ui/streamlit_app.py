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
from app.ui.views.upload import render_upload_page
from app.ui.views.chat import render_chat_page
from app.ui.views.graph import render_graph_page

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
    
    if st.button("💬 Chat with Agent", use_container_width=True):
        st.session_state.current_page = "chat"
    
    if st.button("🗺️ Graph Visualization", use_container_width=True):
        st.session_state.current_page = "graph"
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    if st.session_state.documents_uploaded:
        st.success("✅ Documents Loaded")
    else:
        st.warning("⏳ No Documents")

# Main content area
if st.session_state.current_page == "upload":
    render_upload_page()

elif st.session_state.current_page == "chat":
    render_chat_page()

elif st.session_state.current_page == "graph":
    render_graph_page()
