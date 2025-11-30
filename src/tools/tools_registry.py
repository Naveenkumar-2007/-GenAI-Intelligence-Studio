"""
Tools registry for ReAct agent with mode-specific tool sets.

Provides specialized tools for:
- Docs Mode: Retriever, Wikipedia, Code Explainer
- Video Mode: Transcript Search, Timestamp Lookup, Summarizer, Chapter Search
- Product Mode: Feature Generator, User Persona, System Architect, Competitor Analysis
- Research Mode: Web Search, Web Scrape, Price Extractor, Comparison Formatter
"""

from typing import List

from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from src.tools.web_search import get_wikipedia_tool
from src.tools.code_tools import build_code_explainer_tool


def build_retriever_tool(retriever) -> Tool:
    """
    Wrap retriever as a Tool for the agent.
    """

    def _retriever_tool_fn(query: str) -> str:
        docs: List[Document] = retriever.invoke(query)
        if not docs:
            return "No documents found in the corpus."
        merged = []
        for i, d in enumerate(docs[:8], start=1):
            title = d.metadata.get("source") or f"doc_{i}"
            merged.append(f"[{i}] {title}\n{d.page_content}")
        return "\n\n".join(merged)

    return Tool(
        name="corpus_retriever",
        description="Fetch relevant text passages from uploaded / indexed documents.",
        func=_retriever_tool_fn,
    )


def build_all_tools(retriever, llm: BaseLanguageModel) -> List[Tool]:
    """
    Build complete toolset for DOCS mode:
      - corpus_retriever (Pinecone)
      - wikipedia
      - code_explainer
    """
    corpus_tool = build_retriever_tool(retriever)
    wikipedia_query = get_wikipedia_tool()
    wikipedia_tool = Tool(
        name="wikipedia",
        description="Search Wikipedia for general/world knowledge.",
        func=wikipedia_query.run,
    )
    code_tool = build_code_explainer_tool(llm)
    return [corpus_tool, wikipedia_tool, code_tool]


def build_docs_tools(retriever, llm: BaseLanguageModel) -> List[Tool]:
    """Build tools specifically for Docs mode (same as build_all_tools)."""
    return build_all_tools(retriever, llm)


def build_video_tools(retriever, llm: BaseLanguageModel) -> List[Tool]:
    """
    Build tools specifically for Video mode:
      - transcript_search
      - timestamp_lookup
      - video_summarizer
      - chapter_search
    """
    try:
        from src.tools.video_tools import build_video_tools as _build_video_tools
        return _build_video_tools(retriever, llm)
    except ImportError:
        # Fallback to basic retriever tool
        return [build_retriever_tool(retriever)]


def build_product_tools(llm: BaseLanguageModel) -> List[Tool]:
    """
    Build tools specifically for Product mode:
      - feature_generator
      - user_persona_generator
      - system_architect
      - competitor_analyzer
      - tech_stack_recommender
    """
    try:
        from src.tools.product_tools import build_product_tools as _build_product_tools
        return _build_product_tools(llm)
    except ImportError:
        # Fallback to empty list (product mode uses direct LLM)
        return []


def build_research_tools() -> List[Tool]:
    """
    Build tools specifically for Research mode:
      - web_search: Search the web via DuckDuckGo
      - web_scrape: Extract content from webpages
      - price_extractor: Parse prices from text
      - comparison_formatter: Format comparison tables
    """
    try:
        from src.tools.web_research import build_all_web_research_tools
        return build_all_web_research_tools()
    except ImportError as e:
        print(f"Warning: Could not import web_research tools: {e}")
        return []


def get_tools_for_mode(mode: str, retriever, llm: BaseLanguageModel) -> List[Tool]:
    """
    Get the appropriate toolset based on mode.
    
    Args:
        mode: "docs", "video", "product", or "research"
        retriever: The retriever instance
        llm: The language model instance
    
    Returns:
        List of tools appropriate for the mode
    """
    if mode == "docs":
        return build_docs_tools(retriever, llm)
    elif mode == "video":
        return build_video_tools(retriever, llm)
    elif mode == "product":
        return build_product_tools(llm)
    elif mode == "research":
        return build_research_tools()
    else:
        # Default to docs tools
        return build_docs_tools(retriever, llm)

