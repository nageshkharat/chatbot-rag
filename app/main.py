"""
RAG Chatbot FastAPI Backend

This module provides a REST API for the RAG (Retrieval-Augmented Generation) chatbot.
It includes endpoints for chat interactions and document uploads.

Features:
- /chat: Process user queries and return contextual answers
- /upload: Accept document uploads and re-index the knowledge base
- Automatic background processing for uploaded documents
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from app.rag_utils import load_faiss_index, retrieve
from app.llm_model import generate_answer
import shutil
import os
from app.loader import load_documents
from app.rag_utils import split_docs, embed_documents

# Initialize FastAPI application
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot API that answers questions based on custom documents",
    version="1.0.0"
)

# Configuration
UPLOAD_DIR = "data/your_documents/"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Process a user query and return a contextual answer using RAG.
    
    Args:
        request: ChatRequest containing the user's query
        
    Returns:
        dict: Response containing the generated answer
        
    Process:
        1. Load FAISS vector index and documents
        2. Retrieve relevant document chunks using semantic similarity
        3. Generate response using retrieved context and LLM
        4. Return formatted answer to user
    """
    try:
        query = request.query
        if not query.strip():
            return {"answer": "Please ask a question."}
        
        # Load vector store and documents
        index, docs, metadata = load_faiss_index()
        
        if index is None:
            return {"answer": "Please upload some documents first to build the knowledge base."}
        
        # Retrieve relevant chunks and their sources using semantic similarity
        retrieved_chunks, chunk_sources = retrieve(query, index, docs, metadata)
        
        # Check if we found any relevant information
        if not retrieved_chunks or (len(retrieved_chunks) == 1 and 
            "don't have information" in retrieved_chunks[0]):
            return {"answer": "I don't have information about that in my knowledge base."}
        
        # Combine the retrieved chunks into context for the LLM
        context = "\n".join(retrieved_chunks)
        
        # Generate answer using the LLM with retrieved context
        answer = generate_answer(context, query, chunk_sources)
        
        return {"answer": answer}
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"answer": "I apologize, but I encountered an error while processing your question. Please try again."}

@app.post("/upload")
def upload_doc(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a new document and trigger re-indexing of the knowledge base.
    
    Args:
        file: Uploaded file (PDF or TXT format)
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        dict: Success message confirming upload
        
    Process:
        1. Validate file format (PDF/TXT only)
        2. Save file to documents directory
        3. Trigger background re-embedding of all documents
    """
    # Validate file format
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger background re-embedding
    if background_tasks:
        background_tasks.add_task(reembed_all)
    
    return {"message": "File uploaded and will be processed."}

def reembed_all():
    """
    Re-embed all documents in the knowledge base.
    
    This function is called in the background after new document uploads
    to regenerate the FAISS index with updated document collection.
    
    Process:
        1. Load all documents from the documents directory
        2. Split documents into chunks with overlap
        3. Generate embeddings and update FAISS index
    """
    try:
        docs = load_documents()
        chunks = split_docs(docs)
        embed_documents(chunks)
        print("Successfully re-embedded documents")
    except Exception as e:
        print(f"Error during re-embedding: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API",
        "endpoints": {
            "/chat": "POST - Send queries to the chatbot",
            "/upload": "POST - Upload new documents",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG Chatbot API server...")
    print("FastAPI server will be available at: http://localhost:8000")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
