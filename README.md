# RAG Chatbot with Open-Source LLM

A Retrieval-Augmented Generation (RAG) chatbot that answers user queries based on custom documents using FAISS vector store and open-source language models.

## 🚀 Features

- **Document Ingestion**: Supports PDF and TXT file uploads with automatic processing
- **RAG Pipeline**: Retrieves relevant document chunks and generates contextual responses
- **Open-Source LLM**: Uses DialoGPT-small for local inference (no API keys required)
- **Vector Store**: FAISS for efficient similarity search
- **Web Interface**: Streamlit UI with chat history
- **REST API**: FastAPI backend with `/chat` and `/upload` endpoints
- **Dockerized**: Complete containerized deployment

## 📋 Requirements

- Python 3.10+
- 8GB+ RAM recommended
- CPU/GPU support (works on both)

## 🛠️ Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd chatbot-rag
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Prepare Documents
Place your documents (PDF/TXT) in the `data/your_documents/` folder. Sample documents are already included:
- `ai_overview.txt` - Introduction to Artificial Intelligence
- `llm_basics.txt` - Large Language Models basics
- `rag.txt` - Retrieval-Augmented Generation concepts
- `machine_learning.txt` - Machine Learning fundamentals
- `fastapi_intro.txt` - FastAPI introduction

### 4. Generate Embeddings
```bash
python -c "
from app.loader import load_documents
from app.rag_utils import split_docs, embed_documents
docs = load_documents()
chunks = split_docs(docs)
embed_documents(chunks)
print('Embeddings generated successfully!')
"
```

### 5. Start the Application

#### Option A: Run Both Services Locally
```bash
# Terminal 1: Start FastAPI backend
python -m app.main

# Terminal 2: Start Streamlit UI
streamlit run ui/streamlit_app.py --server.port 8502
```

#### Option B: Docker Deployment
```bash
# Build and run with Docker
docker build -t rag-chatbot .
docker run -p 8001:8001 -p 8501:8501 rag-chatbot
```

## 🌐 Usage

### Web Interface
1. Open your browser to `http://localhost:8502` (Streamlit UI)
2. Type your questions in the chat input
3. Upload new documents using the file uploader
4. View chat history and responses

### API Usage
```bash
# Chat endpoint
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?"}'

# Upload endpoint
curl -X POST "http://localhost:8001/upload" \
  -F "file=@your_document.txt"
```

## 📁 Project Structure

```
chatbot-rag/
├── app/                    # Backend application
│   ├── main.py            # FastAPI server
│   ├── rag_utils.py       # RAG pipeline utilities
│   ├── llm_model.py       # LLM integration
│   └── loader.py          # Document loading
├── ui/                    # Frontend interface
│   └── streamlit_app.py   # Streamlit UI
├── data/                  # Document storage
│   └── your_documents/    # Sample documents
├── models/                # Generated models
│   └── faiss_index/       # FAISS vector store
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Technical Architecture

### RAG Pipeline
1. **Document Processing**: Text chunking with 200 characters and 50 character overlap
2. **Embedding Generation**: SentenceTransformers (`all-MiniLM-L6-v2`)
3. **Vector Storage**: FAISS IndexFlatL2 for similarity search
4. **Retrieval**: Semantic similarity matching with relevance filtering
5. **Generation**: DialoGPT-small for contextual response generation

### Key Components
- **FastAPI Backend**: REST API with async endpoints
- **Streamlit Frontend**: Interactive chat interface
- **FAISS Vector Store**: Efficient similarity search
- **Sentence Transformers**: Text embedding generation
- **DialoGPT**: Conversational AI model

## 📊 Sample Queries

Try these example questions:
- "What is artificial intelligence?"
- "Explain machine learning concepts"
- "How does RAG work?"
- "What are large language models?"
- "Tell me about FastAPI"

## 🐳 Docker Information

The application is fully dockerized with:
- Multi-service container (FastAPI + Streamlit)
- Automatic dependency installation
- Port exposure for both services
- Volume mounting for persistent data

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Download Issues**: The sentence transformer model downloads automatically on first run

3. **Memory Issues**: Ensure at least 8GB RAM for optimal performance

4. **Port Conflicts**: Check if ports 8001 and 8502 are available

## 🚀 Performance Notes

- **First Run**: Initial model download may take time
- **Embedding Generation**: One-time process per document set
- **Response Time**: ~1-3 seconds per query
- **Memory Usage**: ~2-4GB during operation

## 📈 Future Enhancements

- Support for more file formats (Word, PowerPoint)
- Multiple LLM model options
- Advanced chunking strategies
- Vector database alternatives (Chroma, Pinecone)
- Authentication and user management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Python, FastAPI, Streamlit, and open-source AI models.**
