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
    # st.title("🔍 Paper Browser")
    
    # # PDF Download Control
    # include_pfetch_and_update_papers,_and_update_papers,st.checkbox(
    #     "Include PDF content", 
    #     value=True,
    #     help="Process full PDF text for more comprehensive answers"
    # )
    
    # if st.session_state.agent and st.session_state.agent.index:  # <-- SAFE CHECK
    #     manager = st.session_state.agent.index
    
    #     if st.button("🔄 Refresh Index"):
    #         with st.status("Rebuilding index..."):
    #             manager = st.session_state.agent.index
    #             manager.create_documents()
    #             manager.build_index()
    #             st.success("Index refreshed successfully!")
    #     else:
    #         st.warning("Index features not available")
    
    # # Paper Browser Section
    # st.subheader("📚 Indexed Papers")
    # if st.session_state.agent:
    #     manager = st.session_state.agent.index
    #     search_term = st.text_input("Search papers in index")
        
    #     # Display papers with filtering
    #     filtered_papers = [
    #         p for p in manager.documents 
    #         if not search_term or 
    #         search_term.lower() in p['title'].lower() or 
    #         search_term.lower() in p['summary'].lower()
    #     ]
        
    #     for paper in filtered_papers[:10]:  # Limit to 10 for performance
    #         with st.expander(f"{paper['title'][:50]}..."):
    #             st.write(f"**Authors**: {', '.join(paper['authors'])}")
    #             st.write(f"**Published**: {paper['published']}")
    #             st.write(f"**Summary**: {paper['summary'][:200]}...")
    #             if st.button("View Details", key=paper['arxiv_id']):
    #                 st.session_state.selected_paper = paper
                    
    # # Show selected paper details in main area if clicked
    # if "selected_paper" in st.session_state:
    #     paper = st.session_state.selected_paper
    #     st.sidebar.subheader(paper['title'])
    #     st.sidebar.write(f"**ArXiv ID**: {paper['arxiv_id']}")
    #     st.sidebar.write(f"**PDF**: [Download]({paper['pdf_url']})")

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