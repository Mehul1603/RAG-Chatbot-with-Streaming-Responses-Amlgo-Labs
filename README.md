# Deployed Application Link
https://rag-chatbot-with-streaming-responses-amlgo-labs.streamlit.app/

# 🤖 AMLGO RAG Chatbot with Real-Time Streaming

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that provides real-time streaming responses with source citations. Built with Streamlit, LangChain, Pinecone vector database, and Groq's fast LLM API.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture & Flow](#architecture--flow)
- [Model & Technology Choices](#model--technology-choices)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Sample Queries](#sample-queries)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Demo Video](#demo-video)
- [Project Report](#project-report)

## 🎯 Project Overview {#project-overview}

This RAG chatbot demonstrates a complete document Q&A system with the following features:

- **Real-time streaming responses** using Groq's fast LLM API
- **Semantic search** with Pinecone vector database
- **Source citations** with expandable references
- **Modern UI** built with Streamlit
- **Document preprocessing** pipeline for PDF ingestion
- **Configurable retrieval** parameters

## 🏗️ Architecture & Flow {#architecture--flow}

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Document  │    │  Preprocessing  │    │  Vector Store   │
│   (Input Data)  │───▶│  & Chunking     │───▶│   (Pinecone)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Query     │───▶│  Embedding      │───▶│  Similarity     │
│  (Streamlit UI) │    │  Generation     │    │  Search         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streaming      │◀───│  RAG Prompt     │◀───│  Retrieved      │
│  Response       │    │  Generation     │    │  Chunks         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Processing Pipeline

1. **Document Ingestion**: PDF documents are loaded and processed
2. **Text Chunking**: Documents are split into 300-word chunks with 100-word overlap
3. **Embedding Generation**: Chunks are converted to 384-dimensional vectors
4. **Vector Storage**: Embeddings are stored in Pinecone with metadata
5. **Query Processing**: User queries are embedded and used for similarity search
6. **Response Generation**: Retrieved chunks are formatted into prompts for the LLM
7. **Streaming Output**: Real-time token-by-token response generation

## 🤖 Model & Technology Choices {#model--technology-choices}

### Language Model
- **Groq Llama-3.1-8b-instant**: Fast inference with streaming support
- **Temperature**: 0.5 (balanced creativity and accuracy)
- **Max Tokens**: 1024 (sufficient for detailed responses)

### Embedding Model
- **Sentence Transformers**: `all-MiniLM-L6-v2`
- **Dimensions**: 384 (optimized for speed and quality)
- **Normalization**: Cosine similarity for better matching

### Vector Database
- **Pinecone**: Serverless vector database
- **Index Type**: Cosine similarity
- **Region**: AWS us-east-1
- **Cloud**: AWS (serverless)

### Frontend
- **Streamlit**: Modern web interface with real-time updates
- **Chat Interface**: Native Streamlit chat components
- **Sidebar**: System information and controls

## 🚀 Setup Instructions {#setup-instructions}

### Prerequisites

- Python 3.8+
- Groq API key
- Pinecone API key

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd langchain-practice

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Document Preprocessing

#### Step 1: Prepare Your Documents
Place your PDF documents in the `data/` directory.

#### Step 2: Run Preprocessing Notebook
```bash
cd notebooks
jupyter notebook doc_embedding.ipynb
```

The notebook will:
- Load PDF documents
- Clean and merge text
- Split into chunks (300 words, 100-word overlap)
- Generate embeddings using all-MiniLM-L6-v2
- Store in Pinecone vector database

#### Step 3: Verify Setup
Check that your documents are properly indexed:
```python
from vectordb import get_index
index = get_index()
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

### 4. Run the Chatbot

```bash
# From the root directory
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide {#usage-guide}

### Chat Interface

1. **Ask Questions**: Type your question in the chat input
2. **Real-time Responses**: Watch responses stream in real-time
3. **Source References**: Expand source citations in the sidebar
4. **Settings**: Adjust retrieval parameters in the sidebar

### Sidebar Features

- **System Information**: Model details and index statistics
- **Settings**: Configure number of retrieved documents (1-10)
- **Example Queries**: Quick-start questions
- **Clear Chat**: Reset conversation history

### Response Features

- **Streaming**: Real-time token-by-token generation
- **Citations**: Source references with similarity scores
- **Expandable Content**: View full source chunks
- **Metadata**: Word count and relevance scores

## 💡 Sample Queries {#sample-queries}

### Example Questions

1. **General Information**
   - "What is the main topic of this document?"
   - "What are the key terms and conditions?"

2. **Specific Procedures**
   - "What are the arbitration procedures?"
   - "How does batch arbitration work?"
   - "What are the class action limitations?"

3. **Policy Details**
   - "What are the fees and taxes policies?"
   - "How does the eBay Money Back Guarantee work?"
   - "What are the vehicle purchase conditions?"

### Expected Response Format

```
The main topic of this document is [topic], as outlined in the [document type]. The document provides details on [key aspects], including [specific procedures].

According to [Source X - Chunk ID: doc_Y], [specific information from source]. [Additional context from the same source].

The document also outlines [another aspect], including [specific details] ([Source Z - Chunk ID: doc_W]). [More information about this aspect].

In addition, the document discusses [third aspect], which [explanation] ([Source A - Chunk ID: doc_B]). [Further details about this aspect].

The document also addresses [fourth aspect], stating that [specific policy or rule] ([Source C - Chunk ID: doc_D]). [Additional context about this policy].

In terms of limitations, [acknowledgment of information gaps or limitations]. [Additional limitations or caveats].

In conclusion, [restatement of main topic and key points].

Sources: [Source 1 - Chunk ID: doc_X] [Source 2 - Chunk ID: doc_Y] [Source 3 - Chunk ID: doc_Z] [Source 4 - Chunk ID: doc_W] [Source 5 - Chunk ID: doc_V]
```

## 📁 Project Structure {#project-structure}

```
Langchain Practice/
├── app.py                          # Main Streamlit application
├── src/
│   ├── __init__.py                 # Module exports
│   ├── retriever.py                # Similarity search functions
│   └── generator.py                # RAG prompt generation
├── vectordb/
│   ├── __init__.py
│   └── pinecone.py                 # Pinecone index management
├── notebooks/
│   ├── doc_embedding.ipynb         # Document preprocessing
│   └── documents.txt               # Processed text
├── data/
│   └── AI Training Document.pdf    # Source documents
├── chunks/
│   ├── chunked_documents           # Pickled chunk data
│   └── chunked_documents.txt       # Text chunks
└── README.md                       # This file
```

## 🔧 Troubleshooting {#troubleshooting}

### Common Issues

1. **API Key Errors**
   ```
   Error: Invalid API key
   Solution: Verify your .env file has correct API keys
   ```

2. **Pinecone Index Not Found**
   ```
   Error: Index does not exist
   Solution: Run the preprocessing notebook to create the index
   ```

3. **Streaming Not Working**
   ```
   Error: Streaming response failed
   Solution: Check Groq API key and model availability
   ```

4. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce chunk size or use smaller documents
   ```

### Performance Tips

- **Chunk Size**: 300 words with 100-word overlap works well
- **Retrieval Count**: 5-7 documents typically provides good coverage
- **Model Selection**: Llama-3.1-8b-instant offers good speed/quality balance
- **Index Optimization**: Use cosine similarity for better semantic matching

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check index statistics:
```python
from vectordb import get_index
index = get_index()
print(index.describe_index_stats())
```

## 🎥 Demo Video {#demo-video}
![RAG Chatbot Demo](./demo.gif)

## 🎯 Project Report {#project-report}
https://drive.google.com/file/d/12iIsFcLoLZfAQOIRiMgpR2SavAhcbSQT/view?usp=drive_link

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

