import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)

# load enviromental variables from .env file
load_dotenv()

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

def query(query_text, collection_name):

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=600
    )
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        top_k=1,
        request_timeout=600
    )

    # initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60
    )
    # initialize vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(
        similarity_top_k=1,
        vector_store_query_mode="default",
        streaming=True
    )
    result = query_engine.query(query_text)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
        collection_name = QDRANT_COLLECTION_NAME

        print(query(query_text, collection_name))
