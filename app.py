import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"], exist_ok=True)

import streamlit as st
from agent import Agent
from index_manager_pinecone import IndexManagerPinecone
from constants import embedding_model, llm_model
import logging
import time

logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_agent():
    """Initialize the agent with status feedback"""
    with st.status("Initializing AI Research Assistant...", expanded=True) as status:
        st.write("🛠️ Connecting to Pinecone database...")
        try:
            # Add explicit logging
            logger.info("Starting Pinecone connection...")
            
            index_manager = IndexManagerPinecone(
                embedding_model,
                index_name="ai-research-paper-assistant"
            )
            
            st.write("📄 Loading research papers...")
            time.sleep(0.5)  # For better UX flow
            
            st.write("🔍 Building query engine...")
            index_manager.create_index()
            
            status.update(label="✅ Initialization complete!", state="complete")
            return Agent(index_manager, llm_model)
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            st.error(f"Initialization error: {str(e)}")
            status.update(label="❌ Initialization failed", state="error")
            return None

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
# Main chat interface
    st.title("📚 AI Research Paper Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with enhanced processing
if prompt := st.chat_input("Ask me anything about research papers"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with status
    with st.chat_message("assistant"):
        with st.status("🧠 Analyzing research papers...", expanded=False):
            try:
                st.write("🔍 Searching knowledge base...")
                time.sleep(0.3)  # Simulate processing steps
                
                if "agent" not in st.session_state or st.session_state.agent is None:
                    raise RuntimeError("Research assistant not initialized")
                
                st.write("📚 Processing relevant papers...")
                response = st.session_state.agent.chat(prompt)
                
                answer = response.response if hasattr(response, 'response') else str(response)
                
                # Stream the response
                response_container = st.empty()
                full_response = ""
                for chunk in answer.split(" "):
                    full_response += chunk + " "
                    response_container.markdown(full_response + "▌")
                    time.sleep(0.02)  # Adjust speed as needed
                
                response_container.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )