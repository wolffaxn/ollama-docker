import os
from dotenv import load_dotenv

# load enviromental variables from .env file
load_dotenv()

class RAGConfig:

    CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
    CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP")
    DOCS_PATH = os.environ.get("DOCS_PATH")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
    OPEN_API_BASE_URL = os.environ.get("OPEN_API_BASE_URL")
    OPEN_API_KEY = os.environ.get("OPEN_API_KEY")
    OPEN_API_VERSION = os.environ.get("OPEN_API_VERSION")
    QDRANT_URL = os.environ.get("QDRANT_URL")
    QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")
    REDIS_URL=os.environ.get("REDIS_URL")
    REDIS_COLLECTION_NAME=os.environ.get("REDIS_COLLECTION_NAME")
    REQUEST_TIMEOUT = 120

    def __setattr__(self, name, value):
        raise AttributeError(f"Can't reassign constant '{name}'")
