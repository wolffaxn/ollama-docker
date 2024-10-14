import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

from config import RAGConfig
from embeddings import Embedding, EmbeddingProvider
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionCache,
    IngestionPipeline,
)
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import BaseNode, Document
from llama_index.storage.docstore.redis import RedisDocumentStore
from llm import LLM, LLMProvider
from util import QdrantUtil, RedisUtil

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

config = RAGConfig

def get_documents(
    input_dir: Optional[Union[Path, str]] = None
) -> List[Document]:
    documents = None

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory '{input_dir}' doesn't exist")

    # read in PDF documents from filesystem using SimpleDirectoryReader
    logger.info(f"Load documents from '{input_dir}'")
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=[".pdf"]
    ).load_data()
    logger.info(f"Found {len(documents)} page(s)")
    return documents

def run_pipeline(
    documents: List[Document]
) -> Sequence[BaseNode]:

    qdrant_client = QdrantUtil.get_client(
        url=config.QDRANT_URL,
        timeout=config.REQUEST_TIMEOUT
    )

    QdrantUtil.recreate_collection(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        size=config.VECTOR_LENGTH,
        enable_hybrid=config.ENABLE_HYBRID
    )

    redis_kvstore = RedisUtil.get_kvstore(
        client=RedisUtil.get_client(url=config.REDIS_URL)
    )

    llm = LLM(config).get_llm(LLMProvider.OLLAMA)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            ),
            KeywordExtractor(
                llm,
                show_progress=False
            ),
            Embedding(config).get_embedding_model(EmbeddingProvider.OLLAMA)
        ],
        cache=IngestionCache(
            cache=redis_kvstore,
            collection=config.REDIS_COLLECTION_NAME
        ),
        docstore=RedisDocumentStore(
            redis_kvstore=redis_kvstore,
            namespace=config.REDIS_COLLECTION_NAME
        ),
        docstore_strategy=DocstoreStrategy.UPSERTS,
        vector_store=QdrantUtil.get_vectorstore(
            client=qdrant_client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            enable_hybrid=config.ENABLE_HYBRID
        )
    )
    nodes = pipeline.run(documents=documents)
    return nodes

def main():
    logger.info("Starting ingestion process")
    logger.info(f"Using SentenceSplitter (chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP})")
    logger.info(f"Using LLM '{config.OLLAMA_MODEL}'")
    logger.info(f"Using embedding model '{config.EMBEDDING_MODEL}'")

    documents = get_documents(input_dir=os.path.join(os.getcwd(), config.DOCS_PATH))
    nodes = run_pipeline(documents=documents)

    logger.info(f"Ingested {len(nodes)} node(s)")
    logger.info("Ingestion process completed")

if __name__ == "__main__":
    main()
