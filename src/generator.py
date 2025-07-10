from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def create_rag_prompt_with_sources(
    search_results: List[Dict], 
    user_query: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a formatted prompt template with source references for simplified metadata.
    
    Args:
        search_results (List[Dict]): Results from Pinecone query with 'id', 'score', 'metadata'
        user_query (str): User's question
        
    Returns:
        tuple: (formatted_prompt_string, source_references_list)
    """
    
    # Input validation
    if not search_results:
        error_prompt = f"""No relevant context found for the query: "{user_query}"
        
Please respond that you don't have sufficient information to answer this question and suggest the user try rephrasing their question."""
        return error_prompt, []
    
    if not user_query.strip():
        return "Please provide a valid question.", []
    
    # Format retrieved chunks with simplified source attribution
    context_parts = []
    source_references = []
    
    for i, result in enumerate(search_results, 1):
        # Extract data from Pinecone search result
        chunk_id = result.get('id', f'unknown_{i}')
        score = result.get('score', 0.0)
        text_content = result.get('metadata', {}).get('text', '')
        
        # Handle case where text might be stored as Document object
        if hasattr(text_content, 'page_content'):
            actual_text = text_content.page_content
        else:
            actual_text = str(text_content)
        
        word_count = len(actual_text.split())
        
        # Create source reference for display
        source_ref = {
            'source_number': i,
            'chunk_id': chunk_id,
            'similarity_score': round(score, 4),
            'word_count': word_count,
            'content_preview': actual_text[:200] + "..." if len(actual_text) > 200 else actual_text,
            'full_content': actual_text
        }
        source_references.append(source_ref)
        
        # Format chunk for prompt (simplified since no file/page info)
        chunk_header = f"[Source {i} - Chunk ID: {chunk_id}]"
        formatted_chunk = f"{chunk_header}\n{actual_text}\n"
        context_parts.append(formatted_chunk)
    
    # Combine all context
    retrieved_context = "\n".join(context_parts)
    
    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant that provides accurate, comprehensive answers based on provided context documents. 

Key Instructions:
- Use ONLY the information provided in the context documents
- Always cite sources using [Source X] notation when referencing specific information
- If multiple sources support the same point, cite all relevant sources
- Provide detailed, well-structured responses
- If the context doesn't fully answer the question, clearly state what information is missing"""),
        
        ("human", f"""Context Documents:
{retrieved_context}

Question: {user_query}

Please provide a comprehensive answer based on the context above. Remember to:
1. Cite sources using [Source X] notation
2. Include specific details and examples from the source materials
3. Structure your response clearly
4. Acknowledge any limitations in the available information

Your response:""")
    ])
    
    # Convert to string format for the LLM
    formatted_prompt = prompt_template.format()
    
    logger.info(f"Generated prompt for query: '{user_query[:50]}...' with {len(source_references)} sources")
    
    return formatted_prompt, source_references

def get_retrieved_chunks_from_pinecone(index, query_embedding, top_k=5):
    """
    Helper function to retrieve chunks from Pinecone based on your metadata structure.
    
    Args:
        index: Pinecone index object
        query_embedding: Encoded query vector
        top_k: Number of chunks to retrieve
        
    Returns:
        List[Dict]: Search results with id, score, and metadata
    """
    
    # Query Pinecone
    search_results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    return search_results['matches']
