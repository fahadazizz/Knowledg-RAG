import streamlit as st
from app.graph.workflow import graph

def render_chat_page():
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
