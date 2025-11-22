import streamlit as st
import os
from app.graph.workflow import graph

st.set_page_config(page_title="Knowledge RAG", page_icon="🤖")

st.title("Knowledge RAG Agent 🤖")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.info("Ensure your API keys are set in the .env file or environment variables.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Invoke the graph
        inputs = {"question": prompt}
        try:
            # Run the graph
            result = graph.invoke(inputs)
            full_response = result.get("generation", "I couldn't generate an answer.")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Optional: Show steps or retrieved documents in an expander
            with st.expander("View Retrieval Details"):
                if "documents" in result:
                    for i, doc in enumerate(result["documents"]):
                        st.markdown(f"**Document {i+1}:**")
                        st.text(doc)
                        
        except Exception as e:
            st.error(f"An error occurred: {e}")
