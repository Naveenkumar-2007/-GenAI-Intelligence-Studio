"""Configuration module for Agentic RAG system using Groq + Pinecone + HF."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read from env vars first, then Streamlit secrets (for Cloud deployment)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


class Config:
    """Configuration class for Agentic RAG system."""

    # API keys — env vars first, then st.secrets for Streamlit Cloud
    GROQ_API_KEY: str | None = _get_secret("GROQ_API_KEY")
    TAVILY_API_KEY: str | None = _get_secret("TAVILY_API_KEY")
    
    # Models
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
    )
    # Pinecone (Deprecated / Optional)
    # PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "genai-intel-index")
    # PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "docs")

    # Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    @classmethod
    def get_llm(cls):
        """Initialize and return a Groq chat model."""
        # Re-read at call time in case secrets loaded after import
        if not cls.GROQ_API_KEY:
            cls.GROQ_API_KEY = _get_secret("GROQ_API_KEY")
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Add it to .env or Streamlit Secrets.")

        return ChatGroq(
            model_name=cls.GROQ_MODEL_NAME,
            groq_api_key=cls.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=None,
        )
