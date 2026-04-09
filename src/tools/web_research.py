"""
Web Research Tools: Search + Scrape real webpages.

Tools:
- tavily_search: Search the web using Tavily API for latest results
- web_search: Fallback search using DuckDuckGo
- web_scrape: Fetch and extract readable text from any webpage
- price_extractor: Extract price information from scraped content
"""

import os
from typing import List, Dict, Any, Optional
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import Tool

# Silence downstream warning and identify requests when USER_AGENT is not explicitly set.
os.environ.setdefault("USER_AGENT", "GenAI-Intelligence-Studio/1.0")


def tavily_live_search(query: str) -> Dict[str, Any]:
    """
    Perform a direct Tavily search and return structured results
    with images, sources, and AI answer — for the Streamlit UI.
    Returns empty dict if Tavily is unavailable.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("TAVILY_API_KEY")
        except Exception:
            pass
    if not api_key:
        return {}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=8,
            include_images=True,
            include_answer=True,
        )
        return {
            "answer": response.get("answer", ""),
            "images": response.get("images", []),
            "results": response.get("results", []),
            "follow_up_questions": response.get("follow_up_questions", []),
            "response_time": response.get("response_time", 0),
        }
    except Exception as e:
        print(f"Tavily live search error: {e}")
        return {}


# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

REQUEST_TIMEOUT = 12


def _get_headers(idx: int = 0) -> Dict[str, str]:
    return {
        "User-Agent": USER_AGENTS[idx % len(USER_AGENTS)],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _search_duckduckgo(query: str) -> List[Dict]:
    """Search using DuckDuckGo HTML."""
    results = []
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers=_get_headers(0),
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for result in soup.select(".result")[:8]:
            link = result.select_one("a.result__a")
            snippet = result.select_one(".result__snippet")
            if link:
                href = link.get("href", "")
                if href.startswith("http"):
                    results.append({
                        "title": link.get_text(strip=True),
                        "url": href,
                        "snippet": snippet.get_text(strip=True)[:180] if snippet else ""
                    })
    except Exception as e:
        print(f"DuckDuckGo error: {e}")
    return results


def _search_bing(query: str) -> List[Dict]:
    """Search using Bing as fallback."""
    results = []
    try:
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "count": "8"},
            headers=_get_headers(1),
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for result in soup.select("li.b_algo")[:8]:
            link = result.select_one("h2 a")
            snippet = result.select_one(".b_caption p")
            if link:
                href = link.get("href", "")
                if href.startswith("http"):
                    results.append({
                        "title": link.get_text(strip=True),
                        "url": href,
                        "snippet": snippet.get_text(strip=True)[:180] if snippet else ""
                    })
    except Exception as e:
        print(f"Bing error: {e}")
    return results


def build_tavily_search_tool() -> Optional[Tool]:
    """Build a Tavily-powered web search tool for live, up-to-date results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("TAVILY_API_KEY")
        except Exception:
            pass
    if not api_key:
        return None

    def _tavily_search(query: str) -> str:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=6,
                include_answer=True,
            )

            output_parts = []

            # Include Tavily's AI-generated answer summary if available
            ai_answer = response.get("answer")
            if ai_answer:
                output_parts.append(f"**AI Summary:** {ai_answer}\n")

            results = response.get("results", [])
            if not results:
                return f"No results found for: {query}"

            output_parts.append(f"🔍 Found {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                url = r.get("url", "")
                snippet = r.get("content", "")[:300]
                score = r.get("score", 0)
                output_parts.append(
                    f"{i}. **{title}** (relevance: {score:.2f})\n"
                    f"   URL: {url}\n"
                    f"   {snippet}\n"
                )

            return "\n".join(output_parts)
        except Exception as e:
            return f"Tavily search error: {e}"

    return Tool(
        name="tavily_search",
        description=(
            "Search the internet using Tavily for the latest, most relevant results. "
            "Returns AI-summarized answers plus source URLs. Use this as the PRIMARY search tool."
        ),
        func=_tavily_search,
    )


def build_web_search_tool() -> Tool:
    """Enhanced web search using duckduckgo-search library."""

    def _search(query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            results = DDGS().text(keywords=query, region='wt-wt', safesearch='moderate', max_results=6)
            
            if not results:
                return f"No results found for: {query}"
            
            output = f"🔍 Found {len(results)} results:\n\n"
            for i, r in enumerate(results, 1):
                title = r.get('title', 'No Title')
                link = r.get('href', '')
                snippet = r.get('body', '')
                output += f"{i}. **{title}**\n   URL: {link}\n   {snippet}\n\n"
            
            return output
        except Exception as e:
            return f"Search error: {e}"

    return Tool(
        name="web_search",
        description="Search the internet. Input: search query. Output: list of URLs with titles. Use this first, then scrape 1-2 best URLs.",
        func=_search,
    )


def build_web_scraper_tool() -> Tool:
    """Improved web scraper with smart content extraction."""

    def _scrape(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"
        
        try:
            resp = requests.get(url, headers=_get_headers(0), timeout=REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "noscript", "iframe", "svg", "nav", "footer"]):
                tag.decompose()
            
            # Get title
            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else "No title"
            
            # Detect e-commerce sites
            is_ecommerce = any(x in url.lower() for x in ["amazon", "flipkart", "ebay", "walmart", "bestbuy", "croma"])
            
            content = [f"📄 **{title_text}**\n🔗 {url}\n"]
            
            # Extract prices for e-commerce
            if is_ecommerce:
                prices = []
                price_selectors = [".a-price-whole", ".a-offscreen", "._30jeq3", ".price", ".product-price", "[data-price]"]
                for sel in price_selectors:
                    for elem in soup.select(sel)[:5]:
                        text = elem.get_text(strip=True)
                        if re.search(r'[₹$€£]\s*[\d,]+', text):
                            prices.append(text)
                if prices:
                    content.append(f"\n💰 **Prices:** {', '.join(list(dict.fromkeys(prices))[:4])}")
            
            # Find main content
            main = soup.select_one("article, main, .content, #content, .post-content") or soup.body or soup
            
            # Extract headings
            for h in main.find_all(["h1", "h2"])[:5]:
                text = h.get_text(strip=True)
                if text and len(text) > 5:
                    content.append(f"\n## {text}")
            
            # Extract paragraphs
            texts = []
            for p in main.find_all(["p", "li"])[:25]:
                text = p.get_text(strip=True)
                if text and len(text) > 20:
                    texts.append(text)
            
            # Deduplicate
            seen = set()
            unique = []
            for t in texts:
                t_lower = t.lower()[:50]
                if t_lower not in seen:
                    seen.add(t_lower)
                    unique.append(t)
            
            content.append("\n**Content:**\n" + "\n".join(unique[:20]))
            
            result = "\n".join(content)
            return result[:6000] if len(result) > 6000 else result
            
        except requests.Timeout:
            return f"Error: Request timed out for {url}"
        except requests.HTTPError as e:
            return f"Error: HTTP {e.response.status_code} for {url}"
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

    return Tool(
        name="web_scrape",
        description="Extract content from a webpage. Input: URL. Output: page title and main content. Use after web_search.",
        func=_scrape,
    )


def build_price_extractor_tool() -> Tool:
    """Extract prices from text."""

    def _extract_prices(text: str) -> str:
        patterns = [
            r'₹\s*[\d,]+(?:\.\d{2})?',
            r'Rs\.?\s*[\d,]+(?:\.\d{2})?',
            r'\$\s*[\d,]+(?:\.\d{2})?',
            r'€\s*[\d,]+(?:\.\d{2})?',
            r'£\s*[\d,]+(?:\.\d{2})?',
        ]
        
        prices = []
        for pattern in patterns:
            prices.extend(re.findall(pattern, text, re.IGNORECASE))
        
        if not prices:
            return "No prices found in the text."
        
        unique = list(dict.fromkeys(prices))
        result = "💰 **Prices Found:**\n"
        for i, p in enumerate(unique[:10], 1):
            result += f"{i}. {p}\n"
        
        return result

    return Tool(
        name="price_extractor",
        description="Extract prices from text. Input: text with prices. Output: list of prices found.",
        func=_extract_prices,
    )


def build_all_web_research_tools() -> List[Tool]:
    """Build toolset for Auto Research Agent. Tavily is preferred when available."""
    tools = []

    # Primary: Tavily search (live, high-quality results)
    tavily_tool = build_tavily_search_tool()
    if tavily_tool:
        tools.append(tavily_tool)

    # Fallback: DuckDuckGo search
    tools.append(build_web_search_tool())
    tools.append(build_web_scraper_tool())
    tools.append(build_price_extractor_tool())

    return tools
