import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
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
    level=logging.WARN
)

# load enviromental variables from .env file
load_dotenv()

config = RAGConfig

def get_chat_engine() -> BaseChatEngine:

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

    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)
    return chat_engine

def main():
    chat_engine = get_chat_engine()
    chat_engine.streaming_chat_repl()

if __name__ == "__main__":
    main()
