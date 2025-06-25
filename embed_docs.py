from app.loader import load_documents
from app.rag_utils import split_docs, embed_documents

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_docs()docs
    embed_documents(chunks)
    print("✅ Documents embedded and index saved.")
