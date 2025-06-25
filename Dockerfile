# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/your_documents models/faiss_index

# Pre-download the sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Generate embeddings for existing documents
RUN python -c "from app.loader import load_documents; from app.rag_utils import split_docs, embed_documents; docs = load_documents(); chunks = split_docs(docs) if docs else []; embed_documents(chunks) if chunks else print('No documents found')"

# Expose ports for FastAPI and Streamlit
EXPOSE 8001 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Start both services
CMD ["bash", "-c", "python -m app.main & streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & wait"] 