"""State definition for LangGraph Agentic workflow."""

from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    State object for multi-agent workflow.

    mode:
      - 'docs'     → document RAG
      - 'product'  → MVP product builder
      - 'video'    → YouTube video brain
      - 'research' → Auto web research agent
    """

    # Core
    question: str
    user_id: str
    mode: str

    # Routing / intent
    intent: Optional[str]

    # RAG
    retrieved_docs: List[Document]

    # Pre-context from tools / analysis
    tool_context: str

    # Intermediate answer from ReAct or product builder
    intermediate_answer: str

    # Final
    answer: str

    # Memory
    memory_snippet: Optional[str]
    memory_to_save: Optional[str]

    # Video
    video_url: str
    video_chapters: List[str]

    # Research Agent
    research_links: List[str]
    research_raw_contents: List[str]
    research_plan: str
