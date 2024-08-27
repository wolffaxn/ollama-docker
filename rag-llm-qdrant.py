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

class RAG:
    def ingest(self):
        documents = SimpleDirectoryReader(os.environ.get("DOCS_PATH")).load_data()
        logging.info("%s pages(s) found", len(documents))

        # split the documents into small chunks
        chunk_size=os.environ.get("CHUNK_SIZE")
        chunk_overlap=os.environ.get("CHUNK_OVERLAP")
        text_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = []
        idxs = []
        for idx, doc in enumerate(documents):
            logging.info(
                "Splitting page %s into chunks using chunk_size=%s and chunk_overlap=%s",
                idx+1,
                text_parser.chunk_size,
                text_parser.chunk_overlap
            )
            doc_chunks = text_parser.split_text(doc.text)
            chunks.extend(doc_chunks)
            idxs.extend([idx] * len(doc_chunks))

        # construct text nodes from chunks
        nodes = []
        for idx, chunk in enumerate(chunks):
            node = TextNode(
                text=chunk
            )
            node.metadata = documents[idxs[idx]].metadata
            nodes.append(node)

        # generate embedding usind ollama embedding
        embed_model = OllamaEmbedding(
            model_name=os.environ.get("OLLAMA_MODEL"),
            base_url=os.environ.get("OLLAMA_BASE_URL")
        )

        logging.info("Get text embeddings for %s chunks", len(chunks))
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
    RAG().ingest()

if __name__ == "__main__":
    main()
