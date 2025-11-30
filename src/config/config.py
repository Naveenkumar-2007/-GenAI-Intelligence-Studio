"""Configuration module for Agentic RAG system using Groq + Pinecone + HF."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class Config:
    """Configuration class for Agentic RAG system."""

    # API keys
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")

    # Models
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
    )

    # Pinecone
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "genai-intel-index")
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "docs")

    # Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    @classmethod
    def get_llm(cls):
        """Initialize and return a Groq chat model."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        return ChatGroq(
            model_name=cls.GROQ_MODEL_NAME,
            groq_api_key=cls.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=None,
        )
