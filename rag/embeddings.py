import logging
from enum import Enum
from typing import Optional

from config import RAGConfig
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    OLLAMA = "ollama"
    OPENAPI = "openapi"

class Embedding:
    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embed_model = None

    def _initialize_ollama_embedding(self) -> None:
        self._embed_model = OllamaEmbedding(
            base_url= self._config.OLLAMA_BASE_URL,
            model_name=self._config.EMBEDDING_MODEL,
            request_timeout=self._config.REQUEST_TIMEOUT
        )

    def _initialize_openai_embedding(self) -> None:
        self._embed_model = OpenAIEmbedding(
            api_base=f"{self._config.OPEN_API_BASE_URL}/v1",
            api_key=self._config.OPEN_API_KEY
        )

    def get_embedding_model(
        self,
        provider: Optional[EmbeddingProvider] = EmbeddingProvider.OPENAPI
    ) -> OllamaEmbedding | OpenAIEmbedding:
        if provider is EmbeddingProvider.OLLAMA:
            self._initialize_ollama_embedding()
        elif provider is EmbeddingProvider.OPENAPI:
            self._initialize_openai_embedding()
        return self._embed_model
