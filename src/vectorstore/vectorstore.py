"""Vector store module using Pinecone + HuggingFace embeddings."""

from typing import List, Optional
import os

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config.config import Config


class VectorStore:
    """
    Manages Pinecone vector store operations.

    Uses:
      - HuggingFaceEmbeddings
      - PineconeVectorStore (langchain-pinecone)
    """

    def __init__(self, namespace: Optional[str] = None):
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},  # change to 'cuda' if GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index_name = Config.PINECONE_INDEX_NAME
        self.namespace = namespace or Config.PINECONE_NAMESPACE

        # this will internally connect to Pinecone using PINECONE_API_KEY + PINECONE_INDEX_NAME
        # index must already exist with correct dimension
        self.vectorstore = PineconeVectorStore(
            embedding=self.embedding,
            index_name=self.index_name,
            text_key="text",
            namespace=self.namespace,
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        )
        self.retriever = None

    def add_documents(self, documents: List[Document]):
        """
        Add documents to Pinecone index and set retriever.
        """
        if not documents:
            raise ValueError("No documents provided to add to vector store.")

        # PineconeVectorStore expects content in metadata[text_key]; we just use add_documents directly.
        self.vectorstore.add_documents(documents)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            # still allow retriever if docs were already in index
            self.retriever = self.vectorstore.as_retriever()
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        retriever = self.get_retriever()
        return retriever.invoke(query)

    def switch_namespace(self, namespace: str):
        """Switch Pinecone namespace on-the-fly."""
        self.namespace = namespace
        self.vectorstore = PineconeVectorStore(
            embedding=self.embedding,
            index_name=self.index_name,
            text_key="text",
            namespace=self.namespace,
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        )
        self.retriever = self.vectorstore.as_retriever()
