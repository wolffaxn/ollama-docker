import logging
import os
import sys

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO
)

load_dotenv()

def query(query_text):

    # set up the embedding model
    embeddings = OllamaEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL"),
        base_url=os.environ.get("OLLAMA_BASE_URL")
    )

    llm = ChatOllama(
        model=os.environ.get("OLLAMA_MODEL"),
        base_url=os.environ.get("OLLAMA_BASE_URL"),
#        temperatur="0.0",
#        top_k=1
    )

    logging.info("Connecting to Qdrant.")
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL")
    )
    # load the vector store from Qdrant
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=os.environ.get("QDRANT_COLLECTION_NAME"),
        embedding=embeddings
    )

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    response = qa_chain.invoke(query_text)
    print(response)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query(sys.argv[1])
