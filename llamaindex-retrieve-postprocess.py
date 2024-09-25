import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex
)
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
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
REQUEST_TIMEOUT = 120

def query(query_text, collection_name):

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=REQUEST_TIMEOUT
    )
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        # The temperature of the model. Increasing the temperature will
        # make the model answer more creatively.
        temperature=0.0,
        # Reduces the probability of generating nonsense. A higher value
        # (e.g. 100) will give more diverse answers, while a lower value
        # (e.g. 10) will be more conservative.
        top_k=10,
        # Works together with top-k. A higher value (e.g., 0.95) will lead
        # to more diverse text, while a lower value (e.g., 0.5) will generate
        # more focused and conservative text.
        top_p=0.2,
        request_timeout=REQUEST_TIMEOUT
    )

    # initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=REQUEST_TIMEOUT
    )
    # initialize vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )

    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=0.60
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor]
    )
    result = query_engine.query(query_text)
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
        collection_name = QDRANT_COLLECTION_NAME

        result = query(query_text, collection_name)
        pprint_response(result, show_source=True)
