import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

def get_index():
    # Load environment variables
    load_dotenv()

    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    # Configuration
    index_name = "amlgo-chatbot-embeddings"
    dimension = 384 

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Using existing index: {index_name}")

    # Connect to the index
    index = pc.Index(index_name)
    return index
