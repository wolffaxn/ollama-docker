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

load_dotenv()

def retrieve():

    if len(sys.argv) == 1:
        exit()
    user_message = sys.argv[1]

    Settings.embed_model = OllamaEmbedding(
        model_name=os.environ.get("EMBEDDING_MODEL"),
        base_url=os.environ.get("OLLAMA_BASE_URL")
    )
    Settings.llm = Ollama(
        model=os.environ.get("OLLAMA_MODEL"),
        base_url=os.environ.get("OLLAMA_BASE_URL")
    )

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        timeout=60
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.environ.get("QDRANT_COLLECTION_NAME")
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(user_message)
    print(response)

if __name__ == "__main__":
    retrieve()
