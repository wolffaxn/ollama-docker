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
    reader = SimpleDirectoryReader(
        input_dir=os.environ.get("DOCS_PATH"),
        recursive=True,
        required_exts=[".pdf"]
    )
    documents = reader.load_data()
    logging.info("%s pages(s) found", len(documents))

    splitter = SentenceSplitter(
        chunk_size=os.environ.get("CHUNK_SIZE"),
        chunk_overlap=os.environ.get("CHUNK_OVERLAP")
    )

    # create chunks from all document pages
    chunks = []
    chunks_with_page = []
    for page_no, page in enumerate(documents):
        chunk = splitter.split_text(page.text)
        chunks.extend(chunk)
        chunks_with_page.extend([page_no] * len(chunk))

        logging.info(
            "Splitting page %s into %s chunks (chunk_size=%s, chunk_overlap=%s)",
            page_no+1,
            len(chunk),
            splitter.chunk_size,
            splitter.chunk_overlap
        )

    # construct text nodes from chunks
    nodes = []
    for idx, chunk in enumerate(chunks):
        node = TextNode(
            text=chunk
        )
        node.metadata = documents[chunks_with_page[idx]].metadata
        nodes.append(node)

    # generate embedding usind ollama embedding
    embed_model_name = os.environ.get("EMBEDDING_MODEL")
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=os.environ.get("OLLAMA_BASE_URL")
    )

    logging.info("Generate text embeddings for %s chunks using embedding model '%s'",
        len(nodes),
        embed_model_name
    )
    for count, node in enumerate(nodes):
        logging.info("Processing chunk %s ...", count+1)
        embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = embedding

    Settings.llm = Ollama(
        model=os.environ.get("OLLAMA_MODEL"),
        base_url=os.environ.get("OLLAMA_BASE_URL")
    )
    Settings.embed_model = embed_model

    # create a vector store collection using qdrant vector db
    logging.info("Checks whether the collection already exists in Qdrant")
    client = QdrantClient(os.environ.get("QDRANT_URL"))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.environ.get("QDRANT_COLLECTION_NAME")
    )

    # add nodes into the vector store
    logging.info("Store text embeddings into Qdrant")
    vector_store.add(nodes)

    # create an index from vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

def main():
    ingest()

if __name__ == "__main__":
    main()
