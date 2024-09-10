import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import (
    QdrantClient,
    models
)
from qdrant_client.http.exceptions import ResponseHandlingException
from redis import Redis

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)

# load enviromental variables from .env file
load_dotenv()

DOCS_PATH = os.environ.get("DOCS_PATH")
CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")
REDIS_URL=os.environ.get("REDIS_URL")
REDIS_COLLECTION_NAME=os.environ.get("REDIS_COLLECTION_NAME")
VECTOR_LENGTH = os.environ.get("VECTOR_LENGTH")
REQUEST_TIMEOUT = 120

def ingest(input_dir, qdrant_collection_name, redis_collection_name):

    if not os.path.isdir(input_dir):
        logging.error(f"Directory '{input_dir}' doesn't exist")
        return

    # read in PDF documents from filesystem using SimpleDirectoryReader
    logging.info(f"Load documents from '{input_dir}'.")
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=[".pdf"]
    ).load_data()
    logging.info(f"{len(documents)} page(s) found.")

    # initialize Qdrant client
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        timeout=REQUEST_TIMEOUT
    )
    # initialize redis client
    redis_kvstore = RedisKVStore(
        async_redis_client=Redis.from_url(
            url=REDIS_URL
        )
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            ),
            OllamaEmbedding(
                model_name=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=REQUEST_TIMEOUT
            )
        ],
        cache=IngestionCache(
            cache=redis_kvstore,
            collection=redis_collection_name
        ),
        docstore=RedisDocumentStore(
            redis_kvstore=redis_kvstore,
            namespace=redis_collection_name
        ),
        docstore_strategy=DocstoreStrategy.UPSERTS,
        vector_store=QdrantVectorStore(
            client=qdrant_client,
            collection_name=qdrant_collection_name
        )
    )
    logging.info(f"Using SentenceSplitter (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")
    logging.info(f"Using embedding model '{EMBEDDING_MODEL}'.")

    nodes = pipeline.run(documents=documents)

    logging.info(f"Ingested {len(nodes)} nodes.")
    logging.info("Ingestion process completed.")

if __name__ == "__main__":
    input_dir=os.path.join(os.getcwd(), DOCS_PATH)
    ingest(
        input_dir=input_dir,
        qdrant_collection_name=QDRANT_COLLECTION_NAME,
        redis_collection_name=REDIS_COLLECTION_NAME
    )
