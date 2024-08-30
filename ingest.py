import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex
)
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)

load_dotenv()

def ingest():

    # read in PDF documents from filesystem using SimpleDirectoryReader
    input_dir = os.path.join(os.getcwd(), os.environ.get("DOCS_PATH"))
    if not os.path.isdir(input_dir):
        logging.error("Directory '%s' doesn't exist", input_dir)
        exit(1)

    logging.info("Load documents from '%s'.", input_dir)
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=[".pdf"]
    ).load_data()
    logging.info("%s page(s) found.", len(documents))

    node_parser = SentenceSplitter(
        chunk_size=os.environ.get("CHUNK_SIZE"),
        chunk_overlap=os.environ.get("CHUNK_OVERLAP")
    )
    logging.info("Splitting using SentenceSplitter (chunk_size=%s, chunk_overlap=%s).",
        node_parser.chunk_size,
        node_parser.chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    logging.info("%s chunk(s) created.", len(nodes))

    # generate embedding usind ollama embedding
    embed_model_name = os.environ.get("EMBEDDING_MODEL")
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=os.environ.get("OLLAMA_BASE_URL")
    )
    Settings.embed_model = embed_model

    logging.info("Generate text embeddings for %s chunk(s) using embedding model '%s'.",
        len(nodes),
        embed_model_name
    )
    for node_count, node in enumerate(nodes):
        logging.info("Processing chunk %s ...", node_count+1)
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    # create a vector store collection using qdrant vector db
    logging.info("Checks whether the collection already exists in Qdrant.")
    client = QdrantClient(os.environ.get("QDRANT_URL"))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.environ.get("QDRANT_COLLECTION_NAME")
    )

    # add nodes into the vector store
    logging.info("Store text embeddings into Qdrant.")
    vector_store.add(nodes)

    # create an index from vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

if __name__ == "__main__":
    ingest()
