import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import RAGConfig
from embeddings import Embedding, EmbeddingProvider
from llm import LLM, LLMProvider
from util import QdrantUtil

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)

# load enviromental variables from .env file
load_dotenv()

config = RAGConfig

def query(query_text: str) -> RESPONSE_TYPE:

    Settings.embed_model = Embedding(config).get_embedding_model(EmbeddingProvider.OLLAMA)
    Settings.llm = LLM(config).get_llm(LLMProvider.OLLAMA)

    qdrant_client = QdrantUtil.get_client(
        url=config.QDRANT_URL,
        timeout=config.REQUEST_TIMEOUT
    )
    # initialize vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_NAME
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )

    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=0.20
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor]
    )
    result = query_engine.query(query_text)
    return result

def main():
    if len(sys.argv) > 1:
        query_text = sys.argv[1]

        result = query(query_text)
        pprint_response(result, show_source=True)

if __name__ == "__main__":
    main()
