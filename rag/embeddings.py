from enum import Enum
import logging
from typing import Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from config import RAGConfig

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    OLLAMA = "ollama"
    OPENAPI = "openapi"

class Embedding:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None

    def _initialize_ollama_embedding(self):
        self.embedding_model = OllamaEmbedding(
            base_url= self.config.OLLAMA_BASE_URL,
            model_name=self.config.EMBEDDING_MODEL,
            request_timeout=self.config.REQUEST_TIMEOUT
        )

    def _initialize_openai_embedding(self):
        self.embedding_model = OpenAIEmbedding(
            api_base=f"{self.config.OPEN_API_BASE_URL}/v1",
            api_key=self.config.OPEN_API_KEY
        )

    def get_embedding_model(
        self,
        provider: Optional[EmbeddingProvider] = EmbeddingProvider.OPENAPI
    ) -> BaseEmbedding:
        if provider is EmbeddingProvider.OLLAMA:
            self._initialize_ollama_embedding()
        elif provider is EmbeddingProvider.OPENAPI:
            self._initialize_openai_embedding()
        return self.embedding_model
