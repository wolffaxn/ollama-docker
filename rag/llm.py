from enum import Enum
import logging
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from typing import Optional
from config import RAGConfig

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAPI = "openapi"

class LLM:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = None

    def _initialize_ollama(self):
        self.llm = Ollama(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.OLLAMA_MODEL,
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
            request_timeout=self.config.REQUEST_TIMEOUT
        )

    def _initialize_openai(self):
        self.llm = OpenAI(
            api_base=f"{self.config.OPEN_API_BASE_URL}/v1",
            api_key=self.config.OPEN_API_KEY,
            api_version=self.config.OPEN_API_VERSION,
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
            request_timeout=self.config.REQUEST_TIMEOUT
        )

    def get_llm(
        self,
        provider: Optional[LLMProvider] = LLMProvider.OPENAPI
    ):
        if provider is LLMProvider.OLLAMA:
            self._initialize_ollama()
        elif provider is LLMProvider.OPENAPI:
            self._initialize_openai()
        return self.llm

