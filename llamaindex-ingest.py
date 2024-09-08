import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import (
    QdrantClient,
    models
)
from qdrant_client.http.exceptions import ResponseHandlingException

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
VECTOR_LENGTH = os.environ.get("VECTOR_LENGTH")

def ingest(input_dir, collection_name):

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

    logging.info(f"Splitting using SentenceSplitter (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    logging.info(f"{len(nodes)} chunk(s) created.")

    logging.info(f"Generate text embeddings for {len(nodes)} chunk(s) using embedding model '{EMBEDDING_MODEL}'.")
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=60
    )
    Settings.embed_model = embed_model

    for node_count, node in enumerate(nodes):
        logging.info(f"Processing chunk {node_count+1} ...")
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    logging.info("Checks whether the collection already exists in Qdrant.")
    # initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60
    )
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_LENGTH,
                    distance=models.Distance.COSINE
                )
            )
            logging.info(f"Created collection '{collection_name}'.")
        else:
            logging.info(f"Collection 'collection_name' already exists.")
    except ResponseHandlingException as e:
        print(f"Error checking or creating collection: {e}")

    logging.info("Store text embeddings into Qdrant.")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )
    vector_store.add(nodes)

    logging.info("Ingestion process completed.")

if __name__ == "__main__":
    input_dir=os.path.join(os.getcwd(), DOCS_PATH)
    ingest(
        input_dir=input_dir,
        collection_name=QDRANT_COLLECTION_NAME
    )
