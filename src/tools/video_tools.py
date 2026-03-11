"""Video-specific tools for ReAct agent in Video Brain mode."""

from typing import List
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel


def build_transcript_search_tool(retriever) -> Tool:
    """Search through video transcript for specific content."""
    
    def _search_transcript(query: str) -> str:
        docs: List[Document] = retriever.invoke(query)
        if not docs:
            return "No matching transcript segments found."
        
        results = []
        for i, d in enumerate(docs[:8], start=1):
            timestamp = d.metadata.get("timestamp_start", 0)
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            results.append(f"[{minutes}m{seconds:02d}s] {d.page_content[:300]}")
        
        return "\n\n".join(results)
    
    return Tool(
        name="transcript_search",
        description="Search the video transcript for specific topics, keywords, or concepts. Returns timestamped segments.",
        func=_search_transcript,
    )


def build_timestamp_lookup_tool(retriever) -> Tool:
    """Find content at specific timestamps in the video."""
    
    def _lookup_timestamp(time_query: str) -> str:
        """
        Look up content near a specific timestamp.
        Input can be like '5:30', '5m30s', '330' (seconds), or '5 minutes'.
        """
        # Parse the time query
        import re
        
        seconds = 0
        # Try different formats
        if ':' in time_query:
            parts = time_query.split(':')
            if len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
        elif 'm' in time_query.lower():
            match = re.search(r'(\d+)\s*m', time_query.lower())
            if match:
                seconds = int(match.group(1)) * 60
            match = re.search(r'(\d+)\s*s', time_query.lower())
            if match:
                seconds += int(match.group(1))
        else:
            try:
                seconds = int(time_query)
            except ValueError:
                return f"Could not parse timestamp: {time_query}. Use format like '5:30' or '5m30s'."
        
        # Search for content near this timestamp
        docs: List[Document] = retriever.invoke(f"content at {seconds} seconds")
        
        # Filter by timestamp proximity
        relevant = []
        for d in docs:
            ts = d.metadata.get("timestamp_start", 0)
            if abs(ts - seconds) < 120:  # Within 2 minutes
                relevant.append(d)
        
        if not relevant:
            return f"No content found near timestamp {seconds}s."
        
        results = []
        for d in relevant[:5]:
            ts = d.metadata.get("timestamp_start", 0)
            minutes = int(ts // 60)
            secs = int(ts % 60)
            results.append(f"[{minutes}m{secs:02d}s] {d.page_content[:400]}")
        
        return "\n\n".join(results)
    
    return Tool(
        name="timestamp_lookup",
        description="Look up video content at a specific timestamp. Input: time like '5:30', '5m30s', or seconds like '330'.",
        func=_lookup_timestamp,
    )


def build_video_summarizer_tool(llm: BaseLanguageModel, retriever) -> Tool:
    """Summarize sections of the video transcript."""
    
    def _summarize_video(topic: str) -> str:
        # Get relevant transcript sections
        docs: List[Document] = retriever.invoke(topic)
        if not docs:
            return "No content found to summarize."
        
        transcript = "\n".join(d.page_content for d in docs[:10])
        
        prompt = f"""Summarize the following video transcript section about "{topic}":

{transcript}

Provide a concise summary (3-5 bullet points) of the key information:"""
        
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    
    return Tool(
        name="video_summarizer",
        description="Summarize specific topics or sections from the video. Input: topic or concept to summarize.",
        func=_summarize_video,
    )


def build_chapter_search_tool(retriever) -> Tool:
    """Search for specific chapters or sections in the video."""
    
    def _search_chapters(topic: str) -> str:
        docs: List[Document] = retriever.invoke(topic)
        if not docs:
            return "No chapters found matching this topic."
        
        # Group by timestamp ranges to identify chapters
        chapters = []
        seen_timestamps = set()
        
        for d in docs[:10]:
            ts = d.metadata.get("timestamp_start", 0)
            # Round to nearest minute to group
            rounded = int(ts // 60) * 60
            if rounded not in seen_timestamps:
                seen_timestamps.add(rounded)
                minutes = int(ts // 60)
                chapters.append(f"[{minutes}m00s] {d.page_content[:150]}...")
        
        return "Relevant sections:\n" + "\n".join(chapters[:8])
    
    return Tool(
        name="chapter_search",
        description="Find specific chapters or sections in the video that discuss a topic.",
        func=_search_chapters,
    )


def build_video_tools(retriever, llm: BaseLanguageModel) -> List[Tool]:
    """Build all video-specific tools for ReAct agent."""
    return [
        build_transcript_search_tool(retriever),
        build_timestamp_lookup_tool(retriever),
        build_video_summarizer_tool(llm, retriever),
        build_chapter_search_tool(retriever),
    ]
