from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from redis import Redis
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)
from typing import Optional

class QdrantUtil:
    @retry(
        retry=retry_if_exception_type(ResponseHandlingException),
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=10, multiplier=1)
    )
    @staticmethod
    def get_client(
        url: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> QdrantClient:
        qdrant_client = None
        try:
            # initialize Qdrant client
            qdrant_client = QdrantClient(
                url=url,
                timeout=timeout
            )
        except Exception:
            raise
        return qdrant_client

    @staticmethod
    def get_vectorstore(
        client: QdrantClient,
        collection_name: str
    ) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )

class RedisUtil:
    @staticmethod
    def get_client(
        url: str
    ) -> Redis:
        return Redis.from_url(url=url)

    @staticmethod
    def get_kvstore(
        client: Redis
    ) -> RedisKVStore:
        return RedisKVStore(async_redis_client=client)
