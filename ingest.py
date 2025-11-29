import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List

# Initialize Embeddings
def get_embeddings(api_key=None):
    if api_key:
        return OpenAIEmbeddings(openai_api_key=api_key)
    return OpenAIEmbeddings()

# Vector Store Path
VECTOR_STORE_PATH = "chroma_db"

def load_local_docs(directory: str) -> List[Document]:
    """Loads PDF and TXT files from a directory."""
    documents = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} documents from {directory}")
    return documents

def load_website(url: str) -> List[Document]:
    """Loads content from a website URL."""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        print(f"Loaded content from {url}")
        return documents
    except Exception as e:
        print(f"Error loading website {url}: {e}")
        return []

def ingest_documents(documents: List[Document], api_key=None):
    """Splits documents and stores them in ChromaDB."""
    if not documents:
        print("No documents to ingest.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    if not splits:
        print("No text splits created.")
        return

    # Create or update vector store
    embeddings = get_embeddings(api_key)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print(f"Ingested {len(splits)} chunks into ChromaDB at {VECTOR_STORE_PATH}")
    return vectorstore

def get_vectorstore(api_key=None):
    """Returns the existing vector store."""
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = get_embeddings(api_key)
        return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    return None

if __name__ == "__main__":
    # Example usage for testing
    # os.environ["OPENAI_API_KEY"] = "sk-..." # Set your key here or in env
    
    # Create a dummy docs folder for testing if it doesn't exist
    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
        with open("test_docs/sample.txt", "w") as f:
            f.write("This is a sample document for the Agentic RAG system.")
    
    print("Testing local doc ingestion...")
    docs = load_local_docs("test_docs")
    ingest_documents(docs)
    
    print("Testing website ingestion...")
    web_docs = load_website("https://example.com")
    ingest_documents(web_docs)
