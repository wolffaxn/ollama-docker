import logging
import os
import sys
import uuid

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)


load_dotenv()

DOCS_PATH = os.environ.get("DOCS_PATH")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")
VECTOR_LENGTH = os.environ.get("VECTOR_LENGTH")

def ingest(input_dir, collection_name):

    if not os.path.isdir(input_dir):
        logging.error(f"Directory '{input_dir}' doesn't exist")
        return

    logging.info(f"Load documents from '{input_dir}'.")
    loader = PyPDFDirectoryLoader(input_dir)
    pages = loader.load()
    logging.info(f"Loaded {len(pages)} page(s).")

    # Split the documents into chunks using SentenceTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    logging.info(f"Created {len(chunks)} chunk(s) (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")

    # Generate embeddings using OllamaEmbeddings
    ollama_embed = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    logging.info(f"Generate text embeddings for {len(chunks)} chunk(s) using embedding model '{EMBEDDING_MODEL}'.")
    points = []
    for chunk_count, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {chunk_count+1} ...")
        chunk_embedding = ollama_embed.embed_documents(chunk)
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=chunk_embedding[0],
            payload={"text": chunk.page_content}
        ))

    logging.info("Connecting to Qdrant.")
    client = QdrantClient(QDRANT_URL)

    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_LENGTH,
                distance=models.Distance.COSINE
            )
        )
        logging.info(f"Created collection '{QDRANT_COLLECTION_NAME}'.")
    else:
        logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")

    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
        wait=True
    )

    logging.info("Ingestion process completed.")

if __name__ == "__main__":
    input_dir=os.path.join(os.getcwd(), DOCS_PATH)
    ingest(
        input_dir=input_dir,
        collection_name=QDRANT_COLLECTION_NAME
    )
