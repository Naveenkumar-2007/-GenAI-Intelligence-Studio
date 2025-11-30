"""Web search tool using Wikipedia (simple + free)."""

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


def get_wikipedia_tool() -> WikipediaQueryRun:
    """
    Return a Wikipedia query tool instance.
    """
    wrapper = WikipediaAPIWrapper(top_k_results=3, lang="en")
    return WikipediaQueryRun(api_wrapper=wrapper)
