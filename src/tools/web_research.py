"""
Web Research Tools: Search + Scrape real webpages.

Tools:
- web_search: Search the web using multiple engines (DuckDuckGo + Bing fallback)
- web_scrape: Fetch and extract readable text from any webpage
- price_extractor: Extract price information from scraped content
"""

from typing import List, Dict, Any, Optional
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import Tool


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


def build_web_search_tool() -> Tool:
    """Enhanced web search with multiple engines."""

    def _search(query: str) -> str:
        seen_urls = set()
        all_results = []
        
        # Try DuckDuckGo first
        for r in _search_duckduckgo(query):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)
        
        # Add Bing results if needed
        if len(all_results) < 4:
            for r in _search_bing(query):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        
        if not all_results:
            return f"No results found for: {query}. Try different keywords."
        
        output = f"🔍 Found {len(all_results)} results:\n\n"
        for i, r in enumerate(all_results[:6], 1):
            output += f"{i}. **{r['title']}**\n"
            output += f"   URL: {r['url']}\n"
            if r['snippet']:
                output += f"   {r['snippet']}\n"
            output += "\n"
        
        return output

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
    """Build toolset for Auto Research Agent."""
    return [
        build_web_search_tool(),
        build_web_scraper_tool(),
        build_price_extractor_tool(),
    ]
