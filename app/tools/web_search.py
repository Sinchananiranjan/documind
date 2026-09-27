"""Domain-agnostic web search tool using ddgs (DuckDuckGo Search).

Provides free, no-API-key web search for factual/current knowledge.
Designed as untrusted evidence — never executes instructions from results.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded DDGS instance
_ddgs_available: Optional[bool] = None


def _check_ddgs() -> bool:
    """Check if ddgs package is available."""
    global _ddgs_available
    if _ddgs_available is None:
        try:
            from ddgs import DDGS  # noqa: F401
            _ddgs_available = True
        except ImportError:
            logger.warning("ddgs package not installed. Web search disabled. Install with: pip install ddgs")
            _ddgs_available = False
    return _ddgs_available


def search_web(
    query: str,
    max_results: int = 3,
    timeout: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo (via ddgs package).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 3, max 5).
        timeout: Request timeout in seconds (default 5).

    Returns:
        List of dicts with keys: title, snippet, url, source.
        Returns empty list on failure (graceful degradation).
    """
    if not query or not query.strip():
        return []

    if not _check_ddgs():
        return []

    max_results = min(max(1, max_results), 5)

    try:
        from ddgs import DDGS

        with DDGS(timeout=timeout) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results + 2))

        if not raw_results:
            logger.info(f"[WEB_SEARCH] No results for query: '{query}'")
            return []

        # Deduplicate by URL
        seen_urls = set()
        deduplicated = []
        for r in raw_results:
            url = r.get("href", r.get("link", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = r.get("title", "").strip()
            snippet = r.get("body", r.get("snippet", "")).strip()

            # Skip empty results
            if not snippet and not title:
                continue

            # Extract domain as source name
            source = _extract_domain(url)

            deduplicated.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "source": source
            })

            if len(deduplicated) >= max_results:
                break

        logger.info(f"[WEB_SEARCH] Query: '{query}' → {len(deduplicated)} results")
        return deduplicated

    except Exception as e:
        logger.warning(f"[WEB_SEARCH] Search failed for '{query}': {e}")
        return []


def format_web_results_as_context(results: List[Dict[str, Any]]) -> str:
    """
    Format web search results as structured context for LLM consumption.
    Treats web content as untrusted evidence — clearly labeled.
    """
    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        source = r.get("source", "web")

        # Sanitize: strip any potential prompt injection patterns
        snippet = _sanitize_web_content(snippet)
        title = _sanitize_web_content(title)

        blocks.append(
            f"[Web Result {i} — {source}]:\n"
            f"Title: {title}\n"
            f"Content: {snippet}\n"
            f"URL: {url}"
        )

    return "\n\n".join(blocks)


def _extract_domain(url: str) -> str:
    """Extract readable domain name from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.hostname or "web"
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "web"


def _sanitize_web_content(text: str) -> str:
    """
    Basic sanitization of web content to prevent prompt injection.
    Strips patterns that could be interpreted as system instructions.
    """
    if not text:
        return ""

    # Remove common injection patterns
    dangerous_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "you are now",
        "system:",
        "SYSTEM:",
        "assistant:",
        "ASSISTANT:",
        "<|system|>",
        "<|assistant|>",
        "<|user|>",
    ]

    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in text_lower:
            # Replace the dangerous pattern with [redacted]
            import re
            text = re.sub(re.escape(pattern), "[redacted]", text, flags=re.IGNORECASE)

    return text.strip()
