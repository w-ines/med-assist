# huggingsmolagent/tools/search/news_search.py

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from .endpoints import search_web

logger = logging.getLogger(__name__)


def search_news_specific(query: str, location: str = None, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Performs a news-specific search with better targeting for current events.
    
    Args:
        query: The search query
        location: Optional location to focus on (e.g., "Paris", "London")
        max_results: Maximum number of results
        
    Returns:
        List of news results with better URLs
    """
    current_date = datetime.now()
    month_year = current_date.strftime("%B %Y")

    q = (query or "").strip()
    q_lower = q.lower()
    loc = (location or "").strip()
    loc_lower = loc.lower()

    if loc and q_lower.startswith(loc_lower):
        q = q[len(loc):].strip()
        q_lower = q.lower()
    for prefix in ("news ", "latest news "):
        if q_lower.startswith(prefix):
            q = q[len(prefix):].strip()
            q_lower = q.lower()

    base = f"{loc} news".strip() if loc else "news"
    enhanced_query = f"{base} {q} {month_year}".strip()
    
    logger.info(f"News search with enhanced query: '{enhanced_query}'")
    
    preferred_sites = [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "france24.com",
        "lemonde.fr",
        "lefigaro.fr",
        "liberation.fr",
        "francetvinfo.fr",
        "bfmtv.com",
        "rfi.fr",
        "theguardian.com",
    ]
    blocked_domains = {
        "sante-medecine.journaldesfemmes.fr",
        "journaldesfemmes.fr",
        "doctissimo.fr",
        "pinterest.com",
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
        "reddit.com",
        "quora.com",
        "shein.com",
    }

    site_boost = " OR ".join([f"site:{d}" for d in preferred_sites[:6]])
    strategies = [
        enhanced_query,
        f"{enhanced_query} today",
        f"{enhanced_query} site:reuters.com",
        f"{enhanced_query} ({site_boost})",
    ]

    if "yesterday" in q_lower:
        y = (current_date - timedelta(days=1))
        y_iso = y.strftime("%Y-%m-%d")
        y_long = y.strftime("%B %d, %Y")
        strategies.insert(0, f"{base} {q} {y_iso}")
        strategies.insert(1, f"{base} {q} {y_long}")
    
    all_results = []
    seen_urls = set()
    
    for strategy in strategies:
        try:
            results = search_web(strategy, max_results=max_results)

            for result in results:
                url = (result.get("link", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                url_lower = url.lower()
                if any(bad in url_lower for bad in blocked_domains):
                    continue
                if _is_homepage_url(url):
                    continue
                all_results.append(result)
                seen_urls.add(url)

            if len(all_results) >= max_results:
                break

        except Exception as e:
            logger.warning(f"Strategy '{strategy}' failed: {str(e)}")
            continue
    
    # Limit to max_results
    final_results = all_results[:max_results]
    
    logger.info(f"Found {len(final_results)} news articles")
    return final_results


def _is_homepage_url(url: str) -> bool:
    """
    Checks if a URL is likely a homepage rather than a specific article.
    
    Args:
        url: The URL to check
        
    Returns:
        True if it's likely a homepage
    """
    url_lower = url.lower()
    
    # Homepage indicators
    homepage_patterns = [
        "/?$",  # Ends with just /
        "/index",
        "/home",
        "msockid=",  # Tracking parameters without article path
    ]
    
    # Article indicators (should NOT be homepage)
    article_indicators = [
        "/article",
        "/news/",
        "/story",
        "/post",
        "/blog",
        "/world",
        "/politics",
        "/business",
        "/sports",
        "/entertainment",
        "/live-news",
        "/breaking",
        "-20",  # Date patterns like -2025-
    ]
    
    # Check if URL has article indicators
    has_article_indicator = any(indicator in url_lower for indicator in article_indicators)
    
    # Check if URL is very short (likely homepage)
    # e.g., https://www.cnn.com/ or https://www.bbc.com/news
    path_after_domain = url.split("//")[-1].split("/", 1)
    if len(path_after_domain) > 1:
        path = path_after_domain[1]
        is_short = len(path.strip("/")) < 10
    else:
        is_short = True
    
    # Homepage if: short URL AND no article indicators
    return is_short and not has_article_indicator
