import os
import glob
from langchain.schema import Document

def load_documents():
    """Load documents from the data directory using manual text loading"""
    documents = []
    data_dir = "data/your_documents/"
    
    if not os.path.exists(data_dir):
        return documents
    
    # Get all txt files
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Create a Document object similar to what LangChain loaders return
                doc = Document(
                    page_content=content,
                    metadata={'source': file_path}
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # For PDF files, we'll skip them for now due to the langchain_community issue
    # You can add manual PDF parsing later if needed
    
    return documents
