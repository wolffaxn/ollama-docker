"""
title: Llama Index Ollama Pipeline
author: Alexander Wolff
date: 2024-08-30
version: 0.1
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index-core, llama-index-embeddings-ollama, llama-index-llms-ollama, llama-index-vector-stores-qdrant, qdrant_client, nltk==3.9b1
"""

import os

from pydantic import BaseModel
from schemas import OpenAIChatMessage
from typing import List, Union, Generator, Iterator

class Pipeline:

    class Valves(BaseModel):
        OLLAMA_BASE_URL: str
        OLLAMA_MODEL: str
        EMBEDDING_MODEL: str
        QDRANT_URL: str
        QDRANT_COLLECTION_NAME: str

    def __init__(self):
        self.index = None

        self.valves = self.Valves(
            **{
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/"),
                "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "jina/jina-embeddings-v2-base-de:latest"),
                "QDRANT_URL": os.getenv("QDRANT_URL", "http://qdrant:6333/"),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "documents")
            }
        )

    async def on_startup(self):
        from llama_index.core import VectorStoreIndex
        from llama_index.core import (
            Settings,
            VectorStoreIndex
        )
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.EMBEDDING_MODEL,
            base_url=self.valves.OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.OLLAMA_MODEL,
            base_url=self.valves.OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global index

        client = QdrantClient(
            url=self.valves.QDRANT_URL,
            timeout=60
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.valves.QDRANT_COLLECTION_NAME
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
