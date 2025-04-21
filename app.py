import streamlit as st
from agent import Agent
from index_manager_pinecone import IndexManagerPinecone
from constants import embedding_model, llm_model

@st.cache_resource
def initialize_agent(refresh_index: bool = False):
    """
    Initialize the agent with connection verification
    """
    index_manager = IndexManagerPinecone(embedding_model, index_name="ai-research-paper-assistant")
    
    try:
        if refresh_index:
            st.toast("Rebuilding index...", icon="🔄")
            index = index_manager.create_index(refresh=True)
        else:
            st.toast("Loading existing index...", icon="📚")
            index = index_manager.retrieve_index()
        
        # Verify connection
        if not index_manager.is_index_connected():
            raise ConnectionError("Failed to connect to vector index")
            
        return Agent(index, llm_model)
        
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()  # Halt the app if initialization fails

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
        st.divider()
        if "agent" in st.session_state:
                try:
                    if st.session_state.agent.index_manager.is_index_connected():
                        st.success("✅ Index connected")
                        stats = st.session_state.agent.index_manager.pinecone_index.describe_index_stats()
                        st.caption(f"Vectors: {stats['total_vector_count']:,}")
                    else:
                        st.warning("⚠️ Index not connected")
                except:
                    st.error("❌ Connection check failed")

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