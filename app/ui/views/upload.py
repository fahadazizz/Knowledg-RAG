import streamlit as st
from app.utils.pipeline import DocumentPipeline
from app.utils.document_processor import process_uploaded_file

def render_upload_page():
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
                        # Extract text from all files
                        all_text = []
                        for file in uploaded_files:
                            file.seek(0)  # Reset file pointer
                            text = process_uploaded_file(file)
                            all_text.append(text)
                        
                        # Store in session state
                        st.session_state.document_text = "\n\n".join(all_text)
                        
                        # Initialize Ingestion Graph
                        from app.graph.ingestion.workflow import ingestion_graph
                        
                        # Show progress
                        with st.status("Processing Ingestion Graph...", expanded=True) as status:
                            st.write("🚀 Starting ingestion workflow...")
                            
                            # Initial state
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
                                    st.success("✅ Processing complete!")
                                    st.session_state.documents_uploaded = True
                                    
                                    # Show stats
                                    total_chunks = len(result.get("chunks", []))
                                    total_docs = len(result.get("doc_ids", []))
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Total Chunks", total_chunks)
                                    with col2:
                                        st.metric("Documents", total_docs)
                                    
                                    st.balloons()
                                    st.info("🎯 Navigate to **Chat with Agent** to start asking questions or view the **Graph Visualization**.")
                                    
                                else:
                                    st.error(f"❌ Processing failed: {result.get('error')}")
                                    
                            except Exception as e:
                                st.error(f"❌ An error occurred: {e}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {e}")
                        import traceback
                        st.code(traceback.format_exc())
