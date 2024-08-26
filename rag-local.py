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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

class RAG:
    def ingest(self):
        # load documents
        documents = SimpleDirectoryReader(os.environ.get("DOCS_PATH")).load_data()

        # split the documents into small chunks
        text_parser = SentenceSplitter(
            chunk_size=os.environ.get("CHUNK_SIZE"),
            chunk_overlap=os.environ.get("CHUNK_OVERLAP")
        )

        chunks = []
        idxs = []
        for idx, doc in enumerate(documents):
            chunk = text_parser.split_text(doc.text)
            chunks.extend(chunk)
            idxs.extend([idx] * len(chunk))

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

        for node in nodes:
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
        client = QdrantClient(os.environ.get("QDRANT_URL"))
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=os.environ.get("QDRANT_COLLECTION_NAME")
        )

        # add nodes into the vector store
        vector_store.add(nodes)

        # create an index from vector store
        index = VectorStoreIndex.from_vector_store(vector_store)

def main():
    RAG().ingest()

if __name__ == "__main__":
    main()
