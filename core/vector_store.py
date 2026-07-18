import os 
import shutil
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": 'cpu'}
    )

def clear_vector_store():
    """Forcibly clears out the local persistent directory to prevent session cross-contamination."""
    if os.path.exists(CHROMA_DIR):
        print(f"🧹 Purging old session database directory: {CHROMA_DIR}")
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception as e:
            print(f"Warning during vector cleanup: {e}")

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector Store...")
    
    # ── FIX: Clear previous video contexts out entirely
    clear_vector_store()

    # ── FIX: Increased chunk dimensions for expansive conversational flow
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    # Note: Modern versions of LangChain Chroma auto-load persistence directories 
    # directly upon initialization layout.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 6):
    # Defaulting k to 6 seamlessly aligns with your updated rag_engine requirements
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )