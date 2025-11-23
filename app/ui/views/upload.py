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
