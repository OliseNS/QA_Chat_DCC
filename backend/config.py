import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Healthcare AI Assistant."""
    
    # API Keys
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY') or 'your_openrouter_api_key_here'
    
    # Model settings
    LLM_MODEL = os.environ.get('LLM_MODEL', 'google/gemma-3-27b-it:free')
    
    # Data directories
    DOCS_DIR = os.environ.get('DOCS_DIR', 'docs')
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings', 'faiss')
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-for-healthcare-ai'
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # RAG settings
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 200))
    OVERLAP_SIZE = int(os.environ.get('OVERLAP_SIZE', 50))
    TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', 4))
    SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', 0.15))