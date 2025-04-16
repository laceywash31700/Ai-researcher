import streamlit as st
from agent import Agent
from index_manager import IndexManager
from constants import embedding_model, llm_model

@st.cache_resource
def initialize_agent(refresh_index: bool = False):
    """
    Initialize the agent with controlled index loading
    Args:
        refresh_index: If True, forces a fresh index rebuild
    """
    index_manager = IndexManager(embedding_model)
    
    # Use explicit load/rebuild based on parameter
    index = (index_manager.build_index(refresh=True) 
             if refresh_index 
             else index_manager.load_index())
    
    return Agent(index, llm_model)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_refreshed" not in st.session_state:
    st.session_state.index_refreshed = False

# UI Controls
st.title("AI Research Paper Assistant")

# Add index refresh button in sidebar
with st.sidebar:
    if st.button("🔄 Refresh Research Index"):
        st.session_state.agent = initialize_agent(refresh_index=True)
        st.session_state.index_refreshed = True
        st.rerun()

    if st.session_state.index_refreshed:
        st.success("Index refreshed successfully!")
        st.session_state.index_refreshed = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about research papers"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing research..."):
            try:
                response = st.session_state.agent.chat(prompt)
                answer = response.response if hasattr(response, 'response') else str(response)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )