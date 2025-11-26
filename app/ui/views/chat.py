import streamlit as st
from app.graph.workflow import local_graph as graph
from langchain_core.messages import HumanMessage
import uuid

def render_chat_page():
    st.markdown('<h1 class="main-header">💬 Chat with Knowledge RAG Agent</h1>', unsafe_allow_html=True)
    
    # Initialize thread_id for persistence if not present
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
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
                    # Prepare config with thread_id
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    # Prepare input
                    inputs = {"messages": [HumanMessage(content=prompt)]}
                    
                    # Stream the graph execution
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # We'll capture the final response
                    final_response = None
                    
                    # Invoke the graph
                    # Using invoke for simplicity, but could use stream for token streaming if supported by model
                    result = graph.invoke(inputs, config=config)
                    
                    # Get the last message from the agent
                    if "messages" in result and result["messages"]:
                        last_message = result["messages"][-1]
                        final_response = last_message.content
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})
                    else:
                        st.error("No response generated.")
                    
                    # Show tool usage details if available (optional)
                    # We can inspect intermediate steps if we used stream()
                            
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
