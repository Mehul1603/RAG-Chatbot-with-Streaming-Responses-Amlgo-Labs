import os
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src import test_similarity_search, create_rag_prompt_with_sources
from vectordb import get_index
from sentence_transformers import SentenceTransformer
import time
from typing import Iterator, List, Dict, Any

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with Streaming",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_components():
    """Initialize all components with caching for better performance"""
    
    # Initialize ChatGroq with streaming enabled
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=1024,
        streaming=True  # Enable streaming
    )
    
    # Initialize embedding model (same as used for indexing)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Initialize Pinecone index
    index = get_index()
    
    return llm, embedding_model, index

def stream_response_from_llm(llm: ChatGroq, formatted_prompt: str) -> Iterator[str]:
    """
    Stream response from ChatGroq LLM token by token
    """
    try:
        for chunk in llm.stream(formatted_prompt):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def retrieve_relevant_chunks(user_query: str, index, embedding_model, top_k=5):
    """
    Retrieve relevant chunks using the existing test_similarity_search function
    """
    try:
        # Use existing test_similarity_search function
        query_response = test_similarity_search(index, user_query, top_k)
        
        # Extract matches from QueryResponse
        if hasattr(query_response, 'matches'):
            return query_response.matches
        else:
            return query_response
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []

def display_source_references(source_references: List[Dict[str, Any]]):
    """
    Display source references in an organized, expandable format
    """
    if not source_references:
        return
    
    st.subheader("📚 Source References")
    st.write(f"*Retrieved {len(source_references)} relevant document chunks*")
    
    for source in source_references:
        # Create expandable section for each source
        with st.expander(
            f"📄 Source {source['source_number']}: {source['chunk_id']}", 
            expanded=False
        ):
            # Display metadata
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Similarity Score:** {source['similarity_score']}")
            with col2:
                st.write(f"**Word Count:** {source['word_count']}")
            
            # Content preview
            st.write("**Content Preview:**")
            st.write(source['content_preview'])
            
            # Option to show full content
            if st.button(f"Show Full Content", key=f"full_content_{source['source_number']}"):
                st.write("**Full Content:**")
                st.text_area(
                    "Full chunk content",
                    value=source['full_content'],
                    height=200,
                    key=f"full_text_{source['source_number']}"
                )

def get_index_stats(index):
    """
    Get statistics about the Pinecone index
    """
    try:
        stats = index.describe_index_stats()
        return {
            'total_documents': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', 0),
            'index_fullness': stats.get('index_fullness', 0.0)
        }
    except Exception as e:
        return {
            'total_documents': 'Unknown',
            'dimension': 'Unknown', 
            'index_fullness': 'Unknown'
        }

def clear_chat_history():
    """Clear chat history and reset session state"""
    st.session_state.messages = []
    st.session_state.last_sources = []
    st.rerun()

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    
    # Initialize components
    llm, embedding_model, index = initialize_components()
    
    # Get index statistics
    index_stats = get_index_stats(index)
    
    # Main title
    st.title("🤖 AMLGO RAG Chatbot with Real-Time Streaming")
    st.markdown("Ask questions about your documents and get real-time responses with source citations!")
    
    # Create layout columns
    main_col, sidebar_col = st.columns([2, 1])
    
    # Sidebar with model information and controls
    with st.sidebar:
        st.header("🔧 System Information")
        
        # Model information
        st.subheader("Current Model")
        st.write("**LLM:** llama-3.1-8b-instant")
        st.write("**Embeddings:** all-MiniLM-L6-v2")
        st.write("**Vector DB:** Pinecone")
        
        st.divider()
        
        # Index statistics
        st.subheader("📊 Document Statistics")
        st.write(f"**Total Documents:** {index_stats['total_documents']}")
        st.write(f"**Vector Dimension:** {index_stats['dimension']}")
        st.write(f"**Index Status:** {'Ready' if index_stats['total_documents'] != 'Unknown' else 'Error'}")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider("Retrieved Documents", 1, 10, 5)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            clear_chat_history()
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "What is the main topic of this document?",
            "What are the arbitration procedures?",
            "How does batch arbitration work?",
            "What are the class action limitations?"
        ]
        
        for query in example_queries:
            if st.button(f"📝 {query[:30]}...", key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.example_query = query
    
    # Main chat interface
    with main_col:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle example query
        if hasattr(st.session_state, 'example_query'):
            user_input = st.session_state.example_query
            delattr(st.session_state, 'example_query')
        else:
            user_input = None
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents...") or user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and stream assistant response
            with st.chat_message("assistant"):
                # Show thinking indicator
                with st.spinner("🔍 Searching for relevant information..."):
                    # Retrieve relevant chunks
                    search_results = retrieve_relevant_chunks(
                        prompt, 
                        index, 
                        embedding_model, 
                        top_k=top_k
                    )
                
                if not search_results:
                    no_results_msg = "I don't have any relevant information to answer your question. Please try rephrasing your question or asking about different topics."
                    st.markdown(no_results_msg)
                    st.session_state.messages.append({"role": "assistant", "content": no_results_msg})
                    st.session_state.last_sources = []
                else:
                    with st.spinner("📝 Preparing response..."):
                        # Generate prompt and source references
                        formatted_prompt, source_references = create_rag_prompt_with_sources(
                            search_results, 
                            prompt
                        )
                        
                        # Store sources for sidebar display
                        st.session_state.last_sources = source_references
                    
                    # Stream the response in real-time
                    try:
                        # Use st.write_stream for real-time token-by-token streaming
                        streamed_response = st.write_stream(
                            stream_response_from_llm(llm, formatted_prompt)
                        )
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": streamed_response
                        })
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Display source references in sidebar
    with sidebar_col:
        if st.session_state.last_sources:
            display_source_references(st.session_state.last_sources)
        else:
            st.info("Source references will appear here after asking a question.")

if __name__ == "__main__":
    main()
