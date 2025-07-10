from sentence_transformers import SentenceTransformer

def test_similarity_search(index, query_text, top_k=5):
    """Test similarity search with a query"""
    
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding[0].tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    print(f"Query: '{query_text}'")
    print(f"Found {len(results['matches'])} similar chunks:\n")
    
    for i, match in enumerate(results['matches'], 1):
        print(f"--- Result {i} (Score: {match['score']:.4f}) ---")
        print(f"Text: {match['metadata']['text'][:200]}...")
        print()
    
    return results

