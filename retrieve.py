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
        base_url=os.environ.get("OLLAMA_BASE_URL"),
        request_timeout=600
    )
    Settings.llm = Ollama(
        model=os.environ.get("OLLAMA_MODEL"),
        base_url=os.environ.get("OLLAMA_BASE_URL"),
        temperature=0,
        request_timeout=600
    )

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        timeout=600
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.environ.get("QDRANT_COLLECTION_NAME")
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(
        similarity_top_k=1,
        vector_store_query_mode="default",
        streaming=True
    )
    response = query_engine.query(user_message)
    print(response)

if __name__ == "__main__":
    retrieve()
