from typing import Optional

from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models, QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from redis import Redis
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
        collection_name: str,
        enable_hybrid: Optional[bool] = False
    ) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=enable_hybrid
        )

    @staticmethod
    def recreate_collection(
        client: QdrantClient,
        collection_name: str
    ) -> None:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    distance=models.Distance.COSINE,
                    size=768
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams()
                )
            }
        )

class RedisUtil:
    @staticmethod
    def get_client(url: str) -> Redis:
        return Redis.from_url(url=url)

    @staticmethod
    def get_kvstore(client: Redis) -> RedisKVStore:
        return RedisKVStore(async_redis_client=client)
