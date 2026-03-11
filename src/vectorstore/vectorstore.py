"""Vector store module using Pinecone + HuggingFace embeddings."""

from typing import List, Optional
import os

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from src.config.config import Config


class VectorStore:
    """
    Manages FAISS vector store operations.

    Uses:
      - HuggingFaceEmbeddings
      - FAISS (langchain-community)
    """

    def __init__(self, namespace: Optional[str] = None):
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},  # change to 'cuda' if GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        self.namespace = namespace  # Not strictly needed for FAISS in-memory but kept for compatibility
        self.vectorstore = None
        self.retriever = None

    def add_documents(self, documents: List[Document]):
        """
        Add documents to FAISS index (creates new index) and set retriever.
        """
        if not documents:
            raise ValueError("No documents provided to add to vector store.")

        # Create FAISS index from documents
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            # If no docs added yet, this might return None or raise error depending on usage
            return None
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        retriever = self.get_retriever()
        if retriever:
            return retriever.invoke(query)
        return []

    def switch_namespace(self, namespace: str):
        """
        For FAISS, switching namespace effectively means resetting/clearing the current index 
        or preparing for a new one. Since we are in-memory, we just reset.
        """
        self.namespace = namespace
        self.vectorstore = None
        self.retriever = None
