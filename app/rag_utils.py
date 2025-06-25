from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

try:
    if os.path.exists(r"C:\\Users\\Asus\\Desktop\\models\\all-MiniLM-L6-v2"):
        EMBEDDING_MODEL = SentenceTransformer(r"C:\\Users\\Asus\\Desktop\\models\\all-MiniLM-L6-v2")
    else:
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return splitter.split_documents(documents)

def embed_documents(docs):
    texts = [doc.page_content for doc in docs]
    # Store metadata for each chunk
    metadata = []
    for doc in docs:
        # Extract source file name from document metadata
        source = doc.metadata.get('source', 'unknown')
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            source = doc.metadata['source']
        metadata.append(source)
    
    embeddings = EMBEDDING_MODEL.encode(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    os.makedirs("models/faiss_index", exist_ok=True)
    faiss.write_index(index, "models/faiss_index/index.faiss")

    # Save both texts and metadata
    with open("models/faiss_index/docs.pkl", "wb") as f:
        pickle.dump({'texts': texts, 'metadata': metadata}, f)

def load_faiss_index():
    try:
        if not os.path.exists("models/faiss_index/index.faiss") or not os.path.exists("models/faiss_index/docs.pkl"):
            return None, [], []
        
        index = faiss.read_index("models/faiss_index/index.faiss")
        with open("models/faiss_index/docs.pkl", "rb") as f:
            data = pickle.load(f)
            # Handle both old format (just texts) and new format (dict with texts and metadata)
            if isinstance(data, dict):
                docs = data['texts']
                metadata = data.get('metadata', ['unknown'] * len(docs))
            else:
                docs = data
                metadata = ['unknown'] * len(docs)
        return index, docs, metadata
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None, [], []

def retrieve(query, index, docs, metadata=None, k=5):
    if not docs or index.ntotal == 0:
        return ["No documents available."], ["unknown"]
    
    query_vec = EMBEDDING_MODEL.encode([query])
    scores, I = index.search(query_vec, k)
    
    # Filter results based on similarity score
    # Lower scores are better for FAISS (L2 distance)
    # We'll use a threshold to determine relevance
    relevant_chunks = []
    chunk_sources = []
    
    for i, score in enumerate(scores[0]):
        # FAISS returns L2 distances, so lower is better
        # Convert to similarity score (0-1, where 1 is most similar)
        # Assuming embeddings are normalized, max distance is ~2
        similarity = max(0, 1 - score / 2)
        
        # Only include if similarity is above threshold
        if similarity > 0.25:  # Slightly lower threshold for more diverse results
            relevant_chunks.append(docs[I[0][i]])
            if metadata and I[0][i] < len(metadata):
                chunk_sources.append(metadata[I[0][i]])
            else:
                chunk_sources.append('unknown')
    
    if not relevant_chunks:
        return ["I don't have information about that in my knowledge base."], ["unknown"]
    
    return relevant_chunks, chunk_sources
