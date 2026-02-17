import requests
import httpx
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from smolagents import tool
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import re
import json
import random
from firecrawl import FirecrawlApp
from .search.endpoints import search_web
from .search.generate_query import generate_query
import os
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Simple in-memory caches (process-local)
# -----------------------------------------------------------------------------
# These caches are intentionally lightweight and best-effort. They reduce latency
# when the agent repeats the same web_search call across steps.
_WEB_SEARCH_CACHE: dict[str, tuple[float, dict]] = {}
_WEB_SEARCH_CACHE_LOCK = threading.Lock()

def is_content_relevant(content: str, title: str, query: str) -> tuple[bool, str]:
    """
    Evaluates if the scraped content is relevant to the query.
    
    Args:
        content: The scraped text
        title: The page title
        query: The original search query
        
    Returns:
        tuple: (is_relevant: bool, reason: str)
    """
    # Rejection criteria
    MIN_CONTENT_LENGTH = 100
    MAX_COOKIE_RATIO = 0.3  # Max 30% of content = cookies/consent
    
    # 1. Content too short
    if len(content) < MIN_CONTENT_LENGTH:
        return False, f"Content too short ({len(content)} chars)"
    
    # 2. Too many cookie/consent/GDPR mentions
    cookie_keywords = ["cookie", "consent", "gdpr", "privacy policy", "accept", "we use cookies"]
    cookie_count = sum(content.lower().count(kw) for kw in cookie_keywords)
    if cookie_count > len(content.split()) * MAX_COOKIE_RATIO:
        return False, f"Too many cookie/consent mentions ({cookie_count})"
    
    # 3. Mostly links/navigation content
    link_indicators = ["login", "register", "subscribe", "sign up", "menu", "navigation"]
    link_count = sum(content.lower().count(ind) for ind in link_indicators)
    if link_count > 10 and len(content.split()) < 200:
        return False, "Mostly navigation/links"
    
    # 4. Page error or empty content
    error_indicators = ["404", "not found", "page not found", "access denied", "forbidden"]
    if any(err in content.lower()[:500] for err in error_indicators):
        return False, "Error page detected"
    
    # 5. Check if query keywords are present
    query_words = set(query.lower().split())
    # Ignore common words
    stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "what", "how"}
    query_words = query_words - stop_words
    
    if query_words:
        content_lower = content.lower()
        matches = sum(1 for word in query_words if word in content_lower)
        match_ratio = matches / len(query_words)
        
        if match_ratio < 0.3:  # Less than 30% of keywords present
            return False, f"Low keyword match ({match_ratio:.1%})"
    
    return True, "Content appears relevant"

# Configuration for scraping strategy
SCRAPING_CONFIG = {
    'prefer_firecrawl': True,  # Keep Firecrawl active
    'firecrawl_timeout': 30,   # Shorter timeout for Firecrawl (30s instead of 45s)
    'firecrawl_wait_for': 5,   # Wait time for page loading (5s)
    'requests_timeout': 25,    # Timeout for BeautifulSoup (increased for retries)
    'selenium_timeout': 30,    # Timeout for Selenium
    'max_retries': 2,          # 2 attempts per method
    'retry_delay': 1,          # Short delay between retries
    'fallback_enabled': True,  # Fallback to other methods
    'fast_fallback': True,     # Quickly switch to alternatives on timeout
}

@tool
def webscraper(url: str, css_selector: str = "", extraction_prompt: str = "", prefer_method: str = "auto") -> dict:
    """
    Enhanced web scraper with improved error handling and fallback mechanisms.
    
    IMPORTANT: The return structure varies depending on which scraping method succeeds.
    
    Args:
        url: The URL of the webpage to scrape
        css_selector: CSS selector to extract specific elements (BeautifulSoup/Selenium only)
        extraction_prompt: Natural language instructions (Firecrawl only)
        prefer_method: Preferred method - "jina_reader", "firecrawl", "selenium", "beautifulsoup", or "auto"
                      - "auto": intelligent routing (jina_reader for news/blogs, firecrawl for complex sites)
                      - "jina_reader": Ultra-fast (1-3s), clean markdown, perfect for articles/blogs
    
    Returns:
        dict: Structure depends on the successful scraping method:
        
        **BeautifulSoup response:**
        {
            'title': str,
            'full_text': str (JSON string containing extracted content),
            'articles': list[dict] with 'link' and 'title' keys,
            'selected_elements': list,
            'extracted_data': dict with 'method', 'content_length', 'status_code',
            'summary': str,
            'scraping_method': 'beautifulsoup'
        }
        
        **Selenium response:**
        {
            'title': str,
            'full_text': str (plain text),
            'selected_elements': list[str] (CSS selected elements),
            'extracted_data': dict,
            'summary': str,
            'scraping_method': 'selenium'
        }
        
        **Firecrawl response:**
        {
            'markdown': str (page content in markdown),
            'html': str (raw HTML),
            'metadata': dict,
            'scraping_method': 'firecrawl'
        }
        
        **Jina Reader response:**
        {
            'title': str,
            'full_text': str (clean markdown),
            'selected_elements': [],
            'articles': [],
            'extracted_data': dict with 'markdown', 'full_text', 'method',
            'summary': str,
            'scraping_method': 'jina_reader'
        }
    
    Example - Safe way to handle all methods:
        result = webscraper(url="https://example.com", prefer_method="auto")
        
        # Check which method succeeded
        method = result.get('scraping_method')
        
        if method == 'beautifulsoup':
            articles = result.get('articles', [])
            if articles:
                first_link = articles[0]['link']
        elif method == 'selenium':
            text = result.get('full_text', '')
        elif method == 'firecrawl':
            markdown = result.get('markdown', '')
        
        # Or use common fields that usually exist:
        summary = result.get('summary', '')  # Available in BS and Selenium
    
    Recommendation:
        For simple content extraction, use visit_webpage() instead - it returns
        consistent plain text regardless of scraping method.
    """
    logger.info(f"=== STARTING webscraper for URL: {url} ===")
    
    methods = {
        "jina_reader": lambda: use_jina_reader(url),
        "firecrawl": lambda: use_firecrawl_optimized(url, extraction_prompt, css_selector),
        "selenium": lambda: use_selenium_optimized(url, css_selector),
        "beautifulsoup": lambda: use_beautifulsoup_optimized(url, css_selector)
    }
    
    if prefer_method != "auto":
        if prefer_method in methods:
            try:
                logger.info(f"Attempting with {prefer_method}")
                result = methods[prefer_method]()
                # If result is already a properly formatted dict (from jina_reader), return it directly
                if isinstance(result, dict) and 'scraping_method' not in result:
                    result['scraping_method'] = prefer_method
                    return result
                return {
                    "title": "",
                    "full_text": result if isinstance(result, str) else json.dumps(result),
                    "selected_elements": [],
                    "articles": [],
                    "extracted_data": result if isinstance(result, dict) else {},
                    "summary": f"Successfully scraped using {prefer_method}",
                    "scraping_method": prefer_method
                }
            except Exception as e:
                logger.warning(f"{prefer_method} failed: {str(e)}")
                # In fast/robust mode we prefer falling back rather than failing the whole tool call.
                # This prevents a single blocked URL/method from aborting the agent step.
                force_fallback = os.getenv("SCRAPE_FORCE_FALLBACK", "true").lower() == "true"
                if not force_fallback:
                    raise
    
    # Try methods based on the (auto) strategy.
    # NOTE: dict order is NOT a strategy; use determine_scraping_strategy() for smarter ordering.
    strategy = determine_scraping_strategy(url, css_selector, extraction_prompt, prefer_method)
    for method_name in strategy:
        method_func = methods.get(method_name)
        if not method_func:
            continue
        try:
            logger.info(f"Attempting with {method_name}")
            result = method_func()
            # If result is already a properly formatted dict (from jina_reader), return it directly
            if isinstance(result, dict) and 'scraping_method' not in result:
                result['scraping_method'] = method_name
                return result
            return {
                "title": "",
                "full_text": result if isinstance(result, str) else json.dumps(result),
                "selected_elements": [],
                "articles": [],
                "extracted_data": result if isinstance(result, dict) else {},
                "summary": f"Successfully scraped using {method_name}",
                "scraping_method": method_name
            }
        except Exception as e:
            logger.warning(f"{method_name} failed: {str(e)}")
            continue
    
    # If all methods fail
    error_msg = "All scraping methods failed"
    logger.error(error_msg)
    return {
        "title": "",
        "full_text": error_msg,
        "selected_elements": [],
        "articles": [],
        "extracted_data": {},
        "summary": error_msg,
        "scraping_method": "failed",
        "error": error_msg
    }

def determine_scraping_strategy(url: str, css_selector: str, extraction_prompt: str, prefer_method: str) -> list:
    """Determines the optimal order of scraping methods based on URL and requirements."""
    # Allow disabling Firecrawl (LLM-based extraction) to reduce latency and avoid external failures.
    # This is particularly useful for local-only deployments or "fast mode".
    disable_firecrawl = os.getenv("SCRAPE_DISABLE_FIRECRAWL", "true").lower() == "true"
    
    if prefer_method != "auto":
        if prefer_method == "jina_reader":
            base = ["jina_reader", "beautifulsoup", "selenium"]
            return base if disable_firecrawl else ["jina_reader", "firecrawl", "beautifulsoup", "selenium"]
        elif prefer_method == "firecrawl":
            return ["jina_reader", "beautifulsoup", "selenium"] if disable_firecrawl else ["firecrawl", "jina_reader", "beautifulsoup", "selenium"]
        elif prefer_method == "beautifulsoup":
            base = ["beautifulsoup", "jina_reader", "selenium"]
            return base if disable_firecrawl else ["beautifulsoup", "jina_reader", "selenium", "firecrawl"]
        elif prefer_method == "selenium":
            base = ["selenium", "jina_reader", "beautifulsoup"]
            return base if disable_firecrawl else ["selenium", "jina_reader", "firecrawl", "beautifulsoup"]
    
    # Intelligent automatic logic
    domain = url.split('/')[2].lower()
    
    # News/blog sites - Jina Reader is PERFECT for these (10x faster)
    news_blog_domains = [
        'forbes', 'techcrunch', 'medium', 'substack', 'blog', 'news', 'article', 
        'post', 'bbc', 'cnn', 'nytimes', 'theguardian', 'reuters', 'bloomberg',
        'wired', 'verge', 'arstechnica', 'engadget', 'mashable', 'venturebeat',
        'investopedia', 'cnbc', 'wsj', 'ft.com', 'economist'
    ]
    is_news_blog = any(keyword in domain for keyword in news_blog_domains)
    
    # Sites that often require JavaScript
    js_heavy_domains = ['spa', 'react', 'angular', 'vue', 'twitter', 'facebook', 'instagram', 'tiktok','flightstats', 'flightaware', 'flightview']
    needs_js = any(keyword in domain for keyword in js_heavy_domains)
    
    # Flight tracking sites that often have timeout issues
    flight_domains = ['flightstats', 'flightaware', 'flightradar24', 'planefinder']
    is_flight_site = any(keyword in domain for keyword in flight_domains)
    
    # Sites that often block scrapers
    blocking_domains = ['cloudflare', 'bot-protection', 'captcha']
    likely_blocked = any(keyword in domain for keyword in blocking_domains)
    
    # 🚀 PRIORITY 1: News/blog sites → Jina Reader first (ultra-fast, clean markdown)
    if is_news_blog and not extraction_prompt:
        base = ["jina_reader", "beautifulsoup", "selenium"]
        return base if disable_firecrawl else ["jina_reader", "firecrawl", "beautifulsoup", "selenium"]
    
    # Special handling for flight status sites (prefer Selenium for real-time data)
    if is_flight_site:
        base = ["selenium", "jina_reader", "beautifulsoup"]
        return base if disable_firecrawl else ["selenium", "jina_reader", "firecrawl", "beautifulsoup"]
    
    # If we have specific instructions, prefer Firecrawl (LLM extraction)
    if extraction_prompt:
        return ["jina_reader", "selenium", "beautifulsoup"] if disable_firecrawl else ["firecrawl", "jina_reader", "selenium", "beautifulsoup"]
    
    # If site requires JS, start with Selenium
    if needs_js or likely_blocked:
        base = ["selenium", "jina_reader", "beautifulsoup"]
        return base if disable_firecrawl else ["selenium", "jina_reader", "firecrawl", "beautifulsoup"]
    
    # For static sites, try Jina first (fastest), then BeautifulSoup
    if not css_selector:
        base = ["jina_reader", "beautifulsoup", "selenium"]
        return base if disable_firecrawl else ["jina_reader", "beautifulsoup", "firecrawl", "selenium"]
    
    # Default: Jina Reader first for speed, with fallbacks
    base = ["jina_reader", "beautifulsoup", "selenium"]
    return base if disable_firecrawl else ["jina_reader", "firecrawl", "beautifulsoup", "selenium"]

def retry_with_backoff(func, *args, max_retries=3, initial_delay=1, **kwargs):
    """
    Retry function with exponential backoff for handling timeouts.
    """
    # For Firecrawl, use more conservative retry
    func_name = func.__name__ if hasattr(func, '__name__') else str(func)
    if 'firecrawl' in func_name.lower():
        max_retries = min(max_retries, 2)  # Maximum 2 attempts for Firecrawl
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a timeout or connection error
            is_timeout = any(keyword in error_str for keyword in [
                'timeout', 'timed out', 'connection timeout', '408', 
                'connection error', 'connection failed', 'read timeout',
                'request timeout', 'failed to scrape'
            ])
            
            # For Firecrawl with fast_fallback, fail faster after 1 attempt
            if ('firecrawl' in func_name.lower() and is_timeout and 
                SCRAPING_CONFIG.get('fast_fallback', False) and attempt >= 0):
                logger.warning(f"Firecrawl timeout - quickly switching to next method: {str(e)}")
                raise  # Quickly switch to next method
            
            if not is_timeout or attempt == max_retries - 1:
                raise  # Re-raise if not timeout or last attempt
            
            delay = initial_delay * (1.2 ** attempt)  # Very moderate backoff
            logger.warning(f"Timeout error on attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s: {str(e)}")
            time.sleep(delay)
    
    raise Exception(f"All {max_retries} attempts failed")

def use_jina_reader(url: str) -> dict:
    """
    Ultra-fast scraping using Jina AI Reader API.
    Converts any URL to clean markdown in 1-3 seconds.
    
    Advantages:
    - 10x faster than Firecrawl (1-3s vs 10-30s)
    - No API key required (free tier: 20 req/s)
    - Clean markdown output perfect for RAG
    - Handles JavaScript automatically
    - Removes ads, navigation, cookie banners automatically
    
    Args:
        url: The URL to scrape
        
    Returns:
        dict: Scraped content in markdown format
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            'X-Return-Format': 'markdown',  # Force markdown output
            'X-Timeout': '10',  # 10s timeout
        }
        
        logger.info(f"🚀 Using Jina Reader for {url}")
        
        # Use httpx for better performance
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(jina_url, headers=headers)
            response.raise_for_status()
        
        markdown_content = response.text
        
        # Validate content
        if not markdown_content or len(markdown_content) < 100:
            raise Exception("Jina Reader returned empty or too short content")
        
        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
        title = title_match.group(1) if title_match else ""
        
        logger.info(f"✅ Jina Reader success: {len(markdown_content)} chars in ~2s")
        
        # IMPORTANT: Avoid returning the full markdown in tool output.
        # The agent/UI logs often print the entire dict, which can explode in size.
        markdown_preview = truncate_content(markdown_content, 2000)
        
        return {
            'title': title,
            'full_text': truncate_content(markdown_content, 1500),
            'selected_elements': [],
            'articles': [],
            'extracted_data': {
                'method': 'jina_reader',
                'content_length': len(markdown_content),
                'markdown_preview': markdown_preview,
            },
            'summary': f"Jina Reader: {len(markdown_content)} chars extracted in ~2s"
        }
    
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Jina Reader HTTP error {e.response.status_code}: {str(e)}")
        raise Exception(f"Jina Reader HTTP {e.response.status_code} error: {str(e)}")
    except httpx.TimeoutException as e:
        logger.error(f"❌ Jina Reader timeout: {str(e)}")
        raise Exception(f"Jina Reader timeout after 15s: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Jina Reader failed: {str(e)}")
        raise Exception(f"Jina Reader error: {str(e)}")

def use_firecrawl_optimized(url: str, extraction_prompt: str = None, css_selector: str = None) -> dict:
    """Optimized Firecrawl implementation with correct v1 API usage and enhanced error handling."""
    try:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise Exception("FIRECRAWL_API_KEY not found in environment variables. Please set it in your .env file or environment.")
        
        app = FirecrawlApp(api_key=api_key)
        
        logger.debug(f"🔧 Firecrawl extraction_prompt: {extraction_prompt}")
        
        # Check which method is available (API compatibility)
        scrape_method = None
        if hasattr(app, 'scrape_url'):
            scrape_method = 'scrape_url'
        elif hasattr(app, 'scrape'):
            scrape_method = 'scrape'
        else:
            raise Exception("Firecrawl SDK version incompatible - no scrape method found. Please update: pip install --upgrade firecrawl")
        
        logger.debug(f"Using Firecrawl method: {scrape_method}")
        
        if extraction_prompt:
            # ✅ Use scrape method with json format for extraction
            logger.info(f"🔥 Calling Firecrawl {scrape_method} with JSON extraction...")
            
            # Format the prompt properly for jsonOptions
            optimized_prompt_text = f"""
            Extract the following information from this webpage:
            {extraction_prompt}

            Please structure the response as JSON with clear field names.
            Focus on extracting only the most relevant and accurate information.
            If the page contains prices, include currency information.
            If the page contains links, include relevant URLs.
            """

            # Call the appropriate method
            if scrape_method == 'scrape_url':
                result = app.scrape_url(
                    url=url,
                    formats=["json"],
                    json_options={
                        "prompt": optimized_prompt_text.strip()
                    },
                    only_main_content=True,
                    timeout=SCRAPING_CONFIG.get('firecrawl_timeout', 30) * 1000,  # Convert to milliseconds
                    waitFor=SCRAPING_CONFIG.get('firecrawl_wait_for', 5) * 1000,  # Use configurable wait time in milliseconds
                    skip_tls_verification=True  # Ignore SSL issues that can cause timeouts
                )
            else:  # scrape method (newer API - v1.4.0+)
                # The scrape() method takes only URL as positional argument
                # All options must be passed as keyword arguments
                result = app.scrape(
                    url,
                    formats=["json"],
                    extract={
                        "prompt": optimized_prompt_text.strip()
                    },
                    only_main_content=True,
                    timeout=SCRAPING_CONFIG.get('firecrawl_timeout', 30) * 1000,
                    wait_for=SCRAPING_CONFIG.get('firecrawl_wait_for', 5) * 1000
                )

            # Enhanced response processing with better error handling
            if hasattr(result, 'success') and not result.success:
                error_message = getattr(result, 'error', 'Unknown Firecrawl extraction error')
                logger.error(f"Firecrawl scrape_url with JSON failed: {error_message}")
                raise Exception(f"Firecrawl extraction failed: {error_message}")
            
            # Validate result structure
            if not result:
                raise Exception("Firecrawl returned empty result")
            
            # Extract data from the result with better error handling
            data = None
            if hasattr(result, 'data'):
                data = result.data
            elif isinstance(result, dict) and 'data' in result:
                data = result['data']
            elif isinstance(result, dict):
                data = result
            else:
                # Handle Firecrawl ScrapeResponse object
                logger.debug(f"Handling Firecrawl response object: {type(result)}")
                if hasattr(result, '__dict__'):
                    # Convert object to dict
                    data = result.__dict__
                else:
                    data = str(result)

            if not data:
                raise Exception("No data found in Firecrawl response")

            # Get the extracted JSON data with fallbacks
            extracted_data = {}
            if isinstance(data, dict):
                extracted_data = data.get('json', data.get('extract', data.get('content', {})))
            
            metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
            
            # Better title extraction
            title = (metadata.get('title') or 
                    extracted_data.get('title') or 
                    extracted_data.get('name') or 
                    extracted_data.get('heading') or '')
            
            content = str(extracted_data) if extracted_data else str(data)
            
            if not content or content == "{}":
                raise Exception("Firecrawl extraction returned empty content")
            
            logger.info(f"✅ Firecrawl JSON extraction successful: {len(content)} characters")

            return {
                'title': title,
                'full_text': truncate_content(content, 1500),
                'selected_elements': [],  
                'articles': [],
                # Keep extracted JSON (already structured), but avoid accidental huge payloads
                # by not embedding any raw html/markdown here.
                'extracted_data': extracted_data,
                'summary': f"Firecrawl JSON extraction successful - {len(content)} characters extracted"
            }
        else:
            # ✅ Standard scraping without extraction
            logger.info(f"🔥 Calling Firecrawl {scrape_method} for standard scraping...")
            
            if scrape_method == 'scrape_url':
                result = app.scrape_url(
                    url=url,
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=SCRAPING_CONFIG.get('firecrawl_timeout', 30) * 1000,  # Convert to milliseconds
                    waitFor=SCRAPING_CONFIG.get('firecrawl_wait_for', 5) * 1000,  # Use configurable wait time in milliseconds
                    skip_tls_verification=True  # Ignore SSL issues
                )
            else:  # scrape method (newer API - v1.4.0+)
                # The scrape() method takes only URL as positional argument
                result = app.scrape(
                    url,
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=SCRAPING_CONFIG.get('firecrawl_timeout', 30) * 1000,
                    wait_for=SCRAPING_CONFIG.get('firecrawl_wait_for', 5) * 1000
                )
            
            # Process the response
            if hasattr(result, 'success') and not result.success:
                error_message = getattr(result, 'error', 'Unknown Firecrawl scraping error')
                logger.error(f"Firecrawl scrape_url failed: {error_message}")
                raise Exception(f"Firecrawl scraping failed: {error_message}")
            
            # Extract data from the result
            if hasattr(result, 'data'):
                data = result.data
            elif isinstance(result, dict) and 'data' in result:
                data = result['data']
            elif isinstance(result, dict):
                data = result
            else:
                # Handle Firecrawl ScrapeResponse object
                logger.debug(f"Handling Firecrawl response object: {type(result)}")
                if hasattr(result, '__dict__'):
                    # Convert object to dict
                    data = result.__dict__
                else:
                    data = str(result)

            markdown_content = data.get('markdown', '') if isinstance(data, dict) else str(data)
            metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
            title_from_scrape = metadata.get('title', '')

            logger.info(f"✅ Firecrawl standard scraping successful: {len(markdown_content)} characters")

            return {
                'title': title_from_scrape,
                'full_text': truncate_content(markdown_content, 1500),
                'selected_elements': [],
                'articles': [],
                # IMPORTANT: Do NOT return full Firecrawl payload (may contain full markdown/html).
                'extracted_data': {
                    "method": "firecrawl",
                    "content_length": len(markdown_content),
                    "metadata": metadata,
                },
                'summary': f"Firecrawl standard scraping successful - {len(markdown_content)} characters"
            }
    
    except Exception as e:
        error_msg = str(e)
        
        # Enhanced error categorization and logging
        error_type = "unknown"
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_type = "timeout"
        elif "400" in error_msg or "bad request" in error_msg.lower():
            error_type = "bad_request"
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            error_type = "auth_error"
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            error_type = "forbidden"
        elif "404" in error_msg or "not found" in error_msg.lower():
            error_type = "not_found"
        elif "500" in error_msg or "internal server error" in error_msg.lower():
            error_type = "server_error"
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            error_type = "rate_limit"
        elif "waitfor" in error_msg.lower() or "invalid_type" in error_msg.lower():
            error_type = "parameter_error"
        
        logger.error(f"❌ Firecrawl {error_type} error for URL '{url}': {error_msg}")
        
        # Log the URL and parameters for debugging
        current_params = {
            'extraction_prompt': extraction_prompt if extraction_prompt else None,
            'css_selector': css_selector if css_selector else None,
            'error_type': error_type
        }
        logger.warning(f"💥 Firecrawl failure for URL '{url}' with params {current_params}: {error_msg}")
        
        # Provide more specific error messages for common issues
        if error_type == "parameter_error":
            raise Exception(f"Firecrawl parameter error (likely waitFor format): {error_msg}")
        elif error_type == "timeout":
            raise Exception(f"Firecrawl timeout after {SCRAPING_CONFIG.get('firecrawl_timeout', 45)}s: {error_msg}")
        elif error_type == "auth_error":
            raise Exception(f"Firecrawl authentication error - check API key: {error_msg}")
        elif error_type == "rate_limit":
            raise Exception(f"Firecrawl rate limit exceeded: {error_msg}")
        else:
            raise Exception(f"Firecrawl {error_type} error: {error_msg}")

def use_beautifulsoup_optimized(url: str, css_selector: str = None) -> dict:
    """Optimized BeautifulSoup implementation with httpx for better performance and enhanced error handling."""
    
    client = None
    
    try:
        # Enhanced headers with more realistic browser simulation
        headers = get_enhanced_headers()
        
        # Configure timeout
        timeout = SCRAPING_CONFIG.get('requests_timeout', 20)
        
        # Retry logic with different strategies using httpx
        max_attempts = 3
        last_error = None
        response = None
        
        for attempt in range(max_attempts):
            try:
                logger.debug(f"BeautifulSoup (httpx) attempt {attempt + 1}/{max_attempts} for {url}")
                
                # Try different request strategies
                if attempt == 0:
                    # Standard request with SSL verification
                    client = httpx.Client(
                        timeout=timeout,
                        follow_redirects=True,
                        verify=True,
                        headers=headers
                    )
                    response = client.get(url)
                elif attempt == 1:
                    # Retry with different headers and no SSL verification
                    if client:
                        client.close()
                    headers = get_enhanced_headers()
                    client = httpx.Client(
                        timeout=timeout + 5,
                        follow_redirects=True,
                        verify=False,  # Skip SSL verification
                        headers=headers
                    )
                    response = client.get(url)
                else:
                    # Final attempt with minimal headers
                    if client:
                        client.close()
                    minimal_headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }
                    client = httpx.Client(
                        timeout=timeout + 10,
                        follow_redirects=True,
                        verify=False,
                        headers=minimal_headers
                    )
                    response = client.get(url)
                
                response.raise_for_status()
                break  # Success, exit retry loop
                
            except httpx.HTTPError as e:
                last_error = e
                error_msg = str(e).lower()
                
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    break
                
                # Determine if we should retry
                should_retry = any(keyword in error_msg for keyword in [
                    'timeout', 'connection', 'ssl', 'certificate', 'handshake'
                ])
                
                if not should_retry:
                    break  # Don't retry for non-recoverable errors
                
                logger.warning(f"BeautifulSoup (httpx) attempt {attempt + 1} failed, retrying: {str(e)}")
                time.sleep(1)  # Brief delay before retry
        
        if not response or response.status_code != 200:
            raise Exception(f"Failed to fetch content after {max_attempts} attempts: {str(last_error)}")
        
        # Enhanced content validation
        if not response.text or len(response.text) < 100:
            raise Exception("Response content is too short or empty")
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'text/plain' not in content_type:
            raise Exception(f"Unexpected content type: {content_type}")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Enhanced content validation
        if not soup or not soup.find():
            raise Exception("Failed to parse HTML content")
        
        # Intelligent detection of blocked content with enhanced checks
        if is_content_blocked_enhanced(soup, response):
            raise Exception("Content appears to be blocked or requires JavaScript")
        
        # Enhanced extraction with error handling
        title = ""
        full_text = ""
        articles = []
        selected_elements = []
        
        try:
            title = extract_title(soup)
        except Exception as e:
            logger.warning(f"Error extracting title: {str(e)}")
        
        try:
            full_text = extract_main_content(soup)
        except Exception as e:
            logger.warning(f"Error extracting main content: {str(e)}")
        
        try:
            articles = extract_articles_bs(soup, url)
        except Exception as e:
            logger.warning(f"Error extracting articles: {str(e)}")
        
        try:
            selected_elements = extract_css_elements(soup, css_selector) if css_selector else []
        except Exception as e:
            logger.warning(f"Error extracting CSS elements: {str(e)}")
        
        # Validate that we extracted meaningful content
        if not full_text and not articles and not selected_elements:
            raise Exception("No meaningful content extracted from page")
        
        if full_text and len(full_text.strip()) < 50:
            raise Exception("Extracted content is too short, likely blocked or empty page")
        
        return {
            'title': title,
            'full_text': truncate_content(full_text, 1500),
            'selected_elements': selected_elements,
            'articles': articles,
            'extracted_data': {
                'method': 'beautifulsoup',
                'content_length': len(full_text),
                'response_size': len(response.text),
                'status_code': response.status_code
            },
            'summary': f"BeautifulSoup: {len(articles)} articles, {len(selected_elements)} selected elements"
        }
    
    except Exception as e:
        error_msg = str(e)
        
        # Enhanced error categorization for BeautifulSoup
        error_type = "unknown"
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_type = "timeout"
        elif "connection" in error_msg.lower():
            error_type = "connection_error"
        elif "ssl" in error_msg.lower() or "certificate" in error_msg.lower():
            error_type = "ssl_error"
        elif "blocked" in error_msg.lower() or "javascript" in error_msg.lower():
            error_type = "content_blocked"
        elif "404" in error_msg or "not found" in error_msg.lower():
            error_type = "not_found"
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            error_type = "forbidden"
        elif "500" in error_msg or "server error" in error_msg.lower():
            error_type = "server_error"
        elif "content type" in error_msg.lower():
            error_type = "content_type_error"
        elif "too short" in error_msg.lower() or "empty" in error_msg.lower():
            error_type = "empty_content"
        
        logger.error(f"❌ BeautifulSoup {error_type} error for URL '{url}': {error_msg}")
        
        # Provide specific error messages
        if error_type == "content_blocked":
            raise Exception(f"BeautifulSoup content blocked - site requires JavaScript: {error_msg}")
        elif error_type == "timeout":
            raise Exception(f"BeautifulSoup timeout after {SCRAPING_CONFIG.get('requests_timeout', 20)}s: {error_msg}")
        elif error_type == "ssl_error":
            raise Exception(f"BeautifulSoup SSL/Certificate error: {error_msg}")
        elif error_type == "connection_error":
            raise Exception(f"BeautifulSoup connection error: {error_msg}")
        elif error_type == "empty_content":
            raise Exception(f"BeautifulSoup extracted empty or insufficient content: {error_msg}")
        else:
            raise Exception(f"BeautifulSoup {error_type} error: {error_msg}")
    
    finally:
        if client:
            try:
                client.close()
            except Exception as cleanup_error:
                logger.warning(f"Error closing httpx client: {str(cleanup_error)}")

def use_selenium_optimized(url: str, css_selector: str = None) -> dict:
    """Optimized Selenium implementation with enhanced error handling and resource management."""
    
    driver = None
    
    try:
        options = get_optimized_chrome_options()
        service = Service(ChromeDriverManager().install())
        
        # Enhanced driver initialization with retry logic
        max_init_attempts = 3
        for attempt in range(max_init_attempts):
            try:
                driver = webdriver.Chrome(service=service, options=options)
                break
            except Exception as init_error:
                logger.warning(f"Selenium driver init attempt {attempt + 1}/{max_init_attempts} failed: {str(init_error)}")
                if attempt == max_init_attempts - 1:
                    raise Exception(f"Failed to initialize Chrome driver after {max_init_attempts} attempts: {str(init_error)}")
                time.sleep(1)  # Wait before retry
        
        # Set timeouts
        driver.set_page_load_timeout(SCRAPING_CONFIG.get('selenium_timeout', 30))
        driver.implicitly_wait(10)
        
        # Anti-detection setup
        stealth_setup(driver)
        
        logger.info(f"Selenium navigating to {url}")
        
        # Enhanced navigation with timeout handling
        try:
            driver.get(url)
        except Exception as nav_error:
            if "timeout" in str(nav_error).lower():
                logger.warning(f"Page load timeout, but continuing: {str(nav_error)}")
                # Continue with partial page load
            else:
                raise
        
        # Intelligent waiting
        wait_for_content_load(driver)
        
        # Handle popups
        handle_consent_popups_optimized(driver)
        
        # Content extraction with error handling
        title = ""
        full_content = ""
        articles = []
        selected_elements = []
        
        try:
            title = driver.title or ""
        except Exception as e:
            logger.warning(f"Error getting title: {str(e)}")
        
        try:
            full_content = extract_body_content(driver)
        except Exception as e:
            logger.warning(f"Error extracting content: {str(e)}")
        
        try:
            articles = extract_articles_selenium(driver)
        except Exception as e:
            logger.warning(f"Error extracting articles: {str(e)}")
        
        try:
            selected_elements = extract_css_elements_selenium(driver, css_selector) if css_selector else []
        except Exception as e:
            logger.warning(f"Error extracting CSS elements: {str(e)}")
        
        # Validate that we got some content
        if not full_content and not articles and not selected_elements:
            raise Exception("No content extracted from page")
        
        return {
            'title': title,
            'full_text': truncate_content(full_content, 1500),
            'selected_elements': selected_elements,
            'articles': articles,
            'extracted_data': {
                'method': 'selenium',
                'page_loaded': True,
                'content_length': len(full_content)
            },
            'summary': f"Selenium: {len(articles)} articles, {len(selected_elements)} selected elements"
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Enhanced error categorization for Selenium
        error_type = "unknown"
        if "session not created" in error_msg.lower():
            error_type = "session_creation"
        elif "unable to discover open pages" in error_msg.lower():
            error_type = "page_discovery"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_type = "timeout"
        elif "chromedriver" in error_msg.lower():
            error_type = "driver_issue"
        elif "no such element" in error_msg.lower():
            error_type = "element_not_found"
        elif "connection refused" in error_msg.lower():
            error_type = "connection_error"
        
        logger.error(f"❌ Selenium {error_type} error for URL '{url}': {error_msg}")
        
        # Provide specific error messages
        if error_type == "session_creation" or error_type == "page_discovery":
            raise Exception(f"Selenium Chrome session creation failed - possible Chrome/driver compatibility issue: {error_msg}")
        elif error_type == "timeout":
            raise Exception(f"Selenium timeout after {SCRAPING_CONFIG.get('selenium_timeout', 30)}s: {error_msg}")
        elif error_type == "driver_issue":
            raise Exception(f"Selenium ChromeDriver issue - may need driver update: {error_msg}")
        else:
            raise Exception(f"Selenium {error_type} error: {error_msg}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as cleanup_error:
                logger.warning(f"Error during Selenium cleanup: {str(cleanup_error)}")

# Optimized utility functions

def get_random_headers():
    """Returns random headers to avoid detection."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

def get_enhanced_headers():
    """Returns enhanced headers with more realistic browser simulation."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',  # Do Not Track
        'Sec-CH-UA': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-CH-UA-Mobile': '?0',
        'Sec-CH-UA-Platform': '"Windows"'
    }

def is_content_blocked(soup):
    """Detects if content is blocked or requires JavaScript."""
    indicators = [
        lambda: len(soup.get_text()) < 500,  # Content too short
        lambda: any(word in soup.get_text().lower() for word in ['javascript', 'enable js', 'blocked']),
        lambda: soup.find('noscript') and len(soup.find('noscript').get_text()) > 100,
        lambda: any(word in soup.get_text().lower() for word in ['consent', 'cookie', 'gdpr']) and len(soup.get_text()) < 2000
    ]
    
    return any(check() for check in indicators)

def is_content_blocked_enhanced(soup, response):
    """Enhanced detection of blocked content with response analysis."""
    if not soup or not response:
        return True
    
    text_content = soup.get_text().strip()
    text_lower = text_content.lower()
    
    # Basic content length check
    if len(text_content) < 100:
        return True
    
    # JavaScript requirement indicators
    js_indicators = [
        'javascript is disabled',
        'enable javascript',
        'please enable javascript',
        'javascript required',
        'js is disabled',
        'turn on javascript',
        'javascript must be enabled',
        'this site requires javascript'
    ]
    
    if any(indicator in text_lower for indicator in js_indicators):
        return True
    
    # Bot detection indicators
    bot_indicators = [
        'access denied',
        'blocked',
        'bot detected',
        'automated requests',
        'unusual traffic',
        'captcha',
        'verify you are human',
        'security check',
        'cloudflare',
        'ddos protection'
    ]
    
    if any(indicator in text_lower for indicator in bot_indicators):
        return True
    
    # Check for redirect pages or loading pages
    redirect_indicators = [
        'redirecting',
        'please wait',
        'loading',
        'you will be redirected',
        'if you are not redirected'
    ]
    
    if any(indicator in text_lower for indicator in redirect_indicators) and len(text_content) < 1000:
        return True
    
    # Check for consent/cookie walls that block content
    consent_indicators = ['consent', 'cookie', 'gdpr', 'privacy policy', 'accept cookies']
    if (any(indicator in text_lower for indicator in consent_indicators) and 
        len(text_content) < 2000 and
        not any(word in text_lower for word in ['article', 'content', 'news', 'blog', 'post'])):
        return True
    
    # Check response headers for additional clues
    content_type = response.headers.get('content-type', '').lower()
    if 'application/json' in content_type:
        # Might be an API response instead of HTML
        return True
    
    # Check for minimal HTML structure
    if not soup.find('body') or not soup.find('head'):
        return True
    
    # Check for single-page applications with minimal server-side content
    script_tags = soup.find_all('script')
    if (len(script_tags) > 10 and  # Lots of scripts
        len(text_content) < 1000 and  # But little text content
        any('react' in str(script).lower() or 'vue' in str(script).lower() or 'angular' in str(script).lower() 
            for script in script_tags)):
        return True
    
    # Check for noscript content that's longer than main content
    noscript = soup.find('noscript')
    if noscript and len(noscript.get_text()) > len(text_content) * 0.5:
        return True
    
    return False

def extract_title(soup):
    """Extracts page title with fallbacks."""
    if soup.title:
        return soup.title.get_text(strip=True)
    
    # Fallback to h1 or other title elements
    for selector in ['h1', '.title', '#title', '[data-title]']:
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
    
    return ""

def extract_main_content(soup):
    """Intelligently extracts main content."""
    # Priority order for main content
    content_selectors = [
        'main',
        'article',
        '[role="main"]',
        '.content',
        '.main-content',
        '#content',
        '.post-content',
        '.entry-content'
    ]
    
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            return ' '.join(el.get_text(separator=' ', strip=True) for el in elements)
    
    # Fallback: entire body minus header/footer/nav
    for tag in soup(['header', 'footer', 'nav', 'script', 'style']):
        tag.decompose()
    
    return soup.get_text(separator=' ', strip=True)

def extract_articles_bs(soup, base_url):
    """Extracts article links using BeautifulSoup."""
    articles = []
    
    # Look for article links in common patterns
    article_elements = (
        soup.find_all('article') or 
        soup.find_all('div', class_=lambda x: x and ('post' in x.lower() or 'article' in x.lower())) or
        soup.find_all('a', href=lambda x: x and ('/20' in x or '/article' in x or '/post' in x))
    )
    
    for element in article_elements[:10]:  # Limit to 10 articles
        article = {}
        
        # Try to find title and link in various ways
        if element.name == 'article':
            link_elem = element.find('a')
            title_elem = element.find(['h1', 'h2', 'h3']) or link_elem
        else:
            link_elem = element if element.name == 'a' else element.find('a')
            title_elem = element.find(['h1', 'h2', 'h3']) or link_elem
            
        if link_elem and link_elem.get('href'):
            href = link_elem['href']
            # Make relative URLs absolute
            if href.startswith('/'):
                href = '/'.join(base_url.split('/')[:3]) + href
            article['link'] = href
            
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title and len(title) > 5:  # Ignore very short titles
                article['title'] = title
                
        if article.get('title') and article.get('link'):
            articles.append(article)
    
    return articles

def extract_css_elements(soup, css_selector):
    """Extracts elements matching CSS selector."""
    if not css_selector:
        return []
    
    try:
        elements = soup.select(css_selector)
        if elements:
            # Limit to 5 elements max, 200 chars each
            max_elements = 5
            selected_elements = [
                el.get_text(strip=True)[:200] + "..." if len(el.get_text(strip=True)) > 200 
                else el.get_text(strip=True)
                for el in elements[:max_elements]
            ]
            if len(elements) > max_elements:
                selected_elements.append(f"...and {len(elements) - max_elements} more elements")
            return selected_elements
        
        # Try alternative selector if no match
        try:
            elements = soup.select(f".{css_selector}")
            if elements:
                return [el.get_text(strip=True)[:200] for el in elements[:5]]
        except:
            pass
            
    except Exception as e:
        logger.warning(f"CSS selector error: {str(e)}")
    
    return []

def extract_articles_from_structured(structured_data):
    """Extracts articles from Firecrawl structured data."""
    articles = []
    
    # Different ways Firecrawl might structure articles
    if 'links' in structured_data:
        for link in structured_data['links'][:10]:
            if isinstance(link, dict) and 'text' in link and 'url' in link:
                articles.append({
                    'title': link['text'],
                    'link': link['url']
                })
    
    if 'articles' in structured_data:
        articles.extend(structured_data['articles'][:10])
    
    return articles

def truncate_content(content, max_length=1000):
    """Intelligently truncates content."""
    if len(content) <= max_length:
        return content
    
    # Cut at last space to avoid cutting in middle of word
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If we find space in last 20%
        truncated = truncated[:last_space]
    
    return truncated + "..."

def get_optimized_chrome_options():
    """Chrome options optimized for performance, stealth, and stability."""
    options = Options()
    
    # Core headless configuration
    options.add_argument("--headless=new")  # Use new headless mode for better stability
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # GPU and rendering optimizations
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    
    # Performance optimizations
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-images")  # Faster loading
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-default-apps")
    
    # Memory and process optimizations
    options.add_argument("--memory-pressure-off")
    options.add_argument("--max_old_space_size=4096")
    options.add_argument("--single-process")  # Can help with session creation issues
    
    # Network and security
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--ignore-certificate-errors-spki-list")
    
    # Anti-detection measures
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Stability improvements for session creation
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-ipc-flooding-protection")
    options.add_argument("--disable-hang-monitor")
    options.add_argument("--disable-client-side-phishing-detection")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-prompt-on-repost")
    
    # Logging and debugging (can help identify issues)
    options.add_argument("--enable-logging")
    options.add_argument("--log-level=3")  # Only fatal errors
    options.add_argument("--silent")
    
    # Additional prefs for stability
    prefs = {
        "profile.default_content_setting_values": {
            "notifications": 2,  # Block notifications
            "geolocation": 2,    # Block location sharing
        },
        "profile.managed_default_content_settings": {
            "images": 2  # Block images for faster loading
        }
    }
    options.add_experimental_option("prefs", prefs)
    
    return options

def stealth_setup(driver):
    """Sets up anti-detection measures for Selenium."""
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

def wait_for_content_load(driver):
    """Intelligently waits for content to load with enhanced error handling."""
    try:
        selenium_timeout = SCRAPING_CONFIG.get('selenium_timeout', 30)
        
        # Wait for basic page structure
        try:
            WebDriverWait(driver, min(selenium_timeout, 15)).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            logger.debug("✅ Body element found")
        except Exception as e:
            logger.warning(f"Body element not found within timeout: {str(e)}")
            return  # Continue without body if needed
        
        # Wait for document ready state
        try:
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.debug("✅ Document ready state complete")
        except Exception as e:
            logger.warning(f"Document ready state timeout: {str(e)}")
        
        # Additional wait for dynamic content
        time.sleep(2)  # Reduced from 3 to 2 seconds
        
        # Try to wait for meaningful content
        try:
            WebDriverWait(driver, 5).until(
                lambda d: len(d.find_element(By.TAG_NAME, "body").text.strip()) > 50
            )
            logger.debug("✅ Meaningful content detected")
        except Exception as e:
            logger.debug(f"Meaningful content timeout (continuing anyway): {str(e)}")
        
        # Wait for any JavaScript to finish (if present)
        try:
            WebDriverWait(driver, 3).until(
                lambda d: d.execute_script("return jQuery.active == 0") if d.execute_script("return typeof jQuery !== 'undefined'") else True
            )
        except:
            pass  # jQuery might not be present
            
    except Exception as e:
        logger.warning(f"Content load error: {str(e)}")

def extract_body_content(driver):
    """Extracts body content using Selenium."""
    try:
        return driver.find_element(By.TAG_NAME, "body").text
    except Exception as e:
        logger.warning(f"Error extracting body content: {str(e)}")
        return ""

def extract_articles_selenium(driver):
    """Extracts articles using Selenium with optimized element finding."""
    articles = []
    
    # Store original implicit wait and temporarily disable it for faster element searches
    original_implicit_wait = driver.timeouts.implicit_wait
    driver.implicitly_wait(0)  # Disable implicit wait to avoid 10s delays
    
    try:
        # Try different strategies to find articles
        article_elements = (
            driver.find_elements(By.TAG_NAME, "article") or 
            driver.find_elements(By.XPATH, "//div[contains(@class, 'post') or contains(@class, 'article')]") or
            driver.find_elements(By.XPATH, "//a[contains(@href, '/20') or contains(@href, '/article') or contains(@href, '/post')]")
        )
        
        for element in article_elements[:10]:  # Limit to 10 articles
            try:
                if element.tag_name == "article":
                    # Find link element
                    link_elems = element.find_elements(By.TAG_NAME, "a")
                    link_elem = link_elems[0] if link_elems else None
                    
                    # Find heading element using find_elements (no implicit wait)
                    heading_elems = element.find_elements(By.XPATH, ".//h1 | .//h2 | .//h3")
                    title_elem = heading_elems[0] if heading_elems else link_elem
                else:
                    # For non-article elements
                    if element.tag_name == "a":
                        link_elem = element
                    else:
                        link_elems = element.find_elements(By.TAG_NAME, "a")
                        link_elem = link_elems[0] if link_elems else None
                    
                    # Find heading element using find_elements (no implicit wait)
                    heading_elems = element.find_elements(By.XPATH, ".//h1 | .//h2 | .//h3")
                    title_elem = heading_elems[0] if heading_elems else link_elem
                        
                href = link_elem.get_attribute("href") if link_elem else None
                title = title_elem.text if title_elem else None
                
                if href and title and len(title) > 5:  # Ignore very short titles
                    articles.append({
                        "title": title,
                        "link": href
                    })
            except Exception as e:
                continue  # Skip this article on error
    except Exception as e:
        logger.warning(f"Error extracting articles: {str(e)}")
    finally:
        # Restore original implicit wait
        driver.implicitly_wait(original_implicit_wait)
    
    return articles

def extract_css_elements_selenium(driver, css_selector):
    """Extracts elements matching CSS selector using Selenium."""
    if not css_selector:
        return []
    
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, css_selector)
        if elements:
            # Limit to 5 elements max, 200 chars each
            max_elements = 5
            selected_elements = [
                el.text[:200] + "..." if len(el.text) > 200 else el.text 
                for el in elements[:max_elements]
            ]
            if len(elements) > max_elements:
                selected_elements.append(f"...and {len(elements) - max_elements} more elements")
            return selected_elements
    except Exception as e:
        logger.warning(f"Error with CSS selector '{css_selector}': {str(e)}")
    
    return []

def handle_consent_popups_optimized(driver):
    """Handles common consent popups more efficiently with reduced timeouts."""
    try:
        # Set implicit wait to 0 temporarily to speed up popup detection
        original_implicit_wait = driver.timeouts.implicit_wait
        driver.implicitly_wait(0)
        
        # Common consent button XPath patterns (reduced to most common)
        consent_patterns = [
            "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
            "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
        ]
        
        for xpath in consent_patterns:
            try:
                # Very short timeout - if popup exists it should be immediately visible
                button = WebDriverWait(driver, 0.5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                # Scroll to button and click
                driver.execute_script("arguments[0].scrollIntoView();", button)
                time.sleep(0.2)
                button.click()
                logger.info(f"✓ Clicked consent button with XPath: {xpath}")
                time.sleep(0.5)  # Brief wait after click
                driver.implicitly_wait(original_implicit_wait)
                return True
            except:
                continue
        
        driver.implicitly_wait(original_implicit_wait)
        logger.debug("No consent popups found (this is normal)")
        return False
    except Exception as e:
        logger.warning(f"Error handling consent popups: {str(e)}")
        return False

# =============================================================================
# NEWS-SPECIFIC SEARCH TOOL
# =============================================================================

@tool
def search_news(query: str, location: str = "") -> dict:
    """
    Performs a news-specific search optimized for finding current news articles.
    This is better than web_search for finding today's news or recent events.
    
    Args:
        query: The search query (e.g., "latest news", "breaking news")
        location: Optional location to focus on (e.g., "Paris", "London", "New York")
    
    Returns:
        dict: Structured results with context and sources
    """
    from .search.news_search import search_news_specific
    from datetime import datetime
    
    logger.info(f"🔍 search_news called: query='{query}', location='{location}'")
    
    try:
        # Use the specialized news search
        results = search_news_specific(query, location=location if location else None, max_results=10)
        
        if not results:
            return {
                "ok": False,
                "search_query": query,
                "results": [],
                "sources": [],
                "context": "",
                "error": f"No news articles found for '{query}' in {location if location else 'general search'}.",
            }
        
        # Format results as markdown context (but keep structured payload)
        output = f"## News Results ({len(results)} articles found)\n\n"
        
        current_date = datetime.now().strftime("%B %d, %Y")
        output += f"*Search performed on {current_date}*\n\n"
        
        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("link", "")
            snippet = result.get("text", [""])[0] if result.get("text") else ""
            
            output += f"### [{idx}] {title}\n"
            output += f"**URL:** {url}\n"
            if snippet:
                output += f"**Summary:** {snippet}\n"
            output += "\n"
        
        logger.info(f"✅ search_news returned {len(results)} articles")
        sources = []
        for idx, r in enumerate(results, 1):
            sources.append(
                {
                    "index": idx,
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": (r.get("text", [""])[0] if r.get("text") else ""),
                }
            )

        return {
            "ok": True,
            "search_query": query,
            "location": location,
            "results": results,
            "sources": sources,
            "context": output,
            "instructions": "When answering the question, reference the sources inline by wrapping the index in brackets like this: [1]. If multiple sources are used, reference each without commas like this: [1][2][3].",
        }
        
    except Exception as e:
        error_msg = f"Error searching news: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            "ok": False,
            "search_query": query,
            "location": location,
            "results": [],
            "sources": [],
            "context": "",
            "error": error_msg,
        }

# Keep your existing web_search function but make it synchronous for smolagents
@tool
def web_search_ctx(query: str = None, messages: list = None, allowed_domains: list = None, blocked_domains: list = None, max_results: int = 8) -> dict:
    """
    Searches for information on the web and extracts it to provide relevant results.
    
    Args:
        query (str, optional): The search query. If None, will be generated from messages.
        messages (list, optional): Message history to generate query if needed.
        allowed_domains (list, optional): List of allowed domains to filter results (IGNORED - too restrictive).
        blocked_domains (list, optional): List of domains to block in results.
        max_results (int, optional): Maximum number of results to return. Default 8.
        
    Returns:
        dict: Structured results with context and sources
    """
    logger.info("=== STARTING web_search ===")

    try:
        # Generate or use provided query
        search_query = query
        if not search_query and messages:
            try:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    search_query = last_message["content"]
                elif isinstance(last_message, str):
                    search_query = last_message
                else:
                    search_query = str(last_message)
            except Exception:
                search_query = None
        
        if not search_query:
            return {
                "ok": False,
                "error": "No query provided and couldn't generate one from messages",
                "results": [],
                "sources": [],
                "context": "",
            }
        
        # 🔧 FIX 1: IGNORE allowed_domains (too restrictive)
        # The agent often chooses domains that block scraping (ESPN, Google News)
        if allowed_domains:
            logger.warning(f"⚠️ Ignoring allowed_domains={allowed_domains} (prevents finding results)")
        
        # Build domain filters ONLY for blocked domains
        domain_filters = ""
        if blocked_domains:
            domain_filters = " ".join([f"-site:{domain}" for domain in blocked_domains])
        
        # 🔧 FIX: Enhance query for academic papers to prefer English research sources
        enhanced_query = search_query
        if any(keyword in search_query.lower() for keyword in ["paper", "research", "architecture", "model", "ai", "ml", "transformer"]):
            # Add academic domain preferences for research queries
            enhanced_query = f"{search_query} (site:arxiv.org OR site:paperswithcode.com OR site:huggingface.co OR site:github.com OR site:openreview.net)"
            logger.info(f"📚 Enhanced academic query: {enhanced_query}")
        
        # Combine filters with query (without allowed_domains)
        full_query = f"{enhanced_query} {domain_filters}".strip()
        cache_key = f"{full_query}|max_results={max_results}"

        # Best-effort cache to avoid repeated web_search+scrape calls across agent steps.
        cache_ttl = int(os.getenv("WEB_SEARCH_CACHE_TTL_SECONDS", "600"))
        if cache_ttl > 0:
            with _WEB_SEARCH_CACHE_LOCK:
                cached = _WEB_SEARCH_CACHE.get(cache_key)
            if cached:
                ts, payload = cached
                if time.time() - ts <= cache_ttl:
                    logger.info(f"⚡ web_search cache hit (ttl={cache_ttl}s)")
                    return payload
        
        logger.info(f"🔍 Executing search: '{full_query}' (max_results={max_results})")
        
        # Execute web search
        search_results = search_web(full_query, max_results)
        
        # Filter out problematic domains that require CAPTCHA (after search)
        # Also filter non-English sites and low-quality content
        # Note: Only block domains that consistently return bad content
        blocked_after_search = [
            "reddit.com", "twitter.com", "x.com",  # Social media (CAPTCHA)
            "baidu.com", "zhihu.com", "weibo.com", "zhidao.baidu.com",  # Chinese sites
            "pinterest.com", "instagram.com", "facebook.com",  # Image/social sites
            "bilibili.com", "qq.com", "sina.com.cn",  # More Chinese sites
            # Removed skyscrapercity.com and skyscraperpage.com - they can have valid data
            # Removed quora.com - sometimes has good answers
        ]
        if search_results:
            original_count = len(search_results)
            search_results = [
                result for result in search_results 
                if not any(blocked in result.get("link", "") for blocked in blocked_after_search)
            ]
            if len(search_results) < original_count:
                logger.info(f"🚫 Filtered out {original_count - len(search_results)} blocked domains")
        
        # Check if search returned any results
        if not search_results:
            # 🔧 FIX: If enhanced query failed, try a simpler fallback without site restrictions
            if enhanced_query != search_query:
                logger.warning(f"⚠️ Enhanced query returned no results, trying simpler query...")
                full_query = f"{search_query} {domain_filters}".strip()
                logger.info(f"🔍 Fallback search: '{full_query}'")
                search_results = search_web(full_query, max_results)
                
                # Filter blocked domains again
                if search_results:
                    original_count = len(search_results)
                    search_results = [
                        result for result in search_results 
                        if not any(blocked in result.get("link", "") for blocked in blocked_after_search)
                    ]
                    if len(search_results) < original_count:
                        logger.info(f"🚫 Filtered out {original_count - len(search_results)} blocked domains in fallback")
            
            # If still no results, return error
            if not search_results:
                logger.warning(f"❌ No search results for: '{search_query}'")
                return {
                    "ok": False,
                    "search_query": search_query,
                    "results": [],
                    "sources": [],
                    "context": "",
                    "error": "No search results found. Try a different query or check your connection.",
                    "instructions": "When answering the question, reference the sources inline by wrapping the index in brackets like this: [1]. If multiple sources are used, reference each without commas like this: [1][2][3]."
                }
        
        logger.info(f"✅ Found {len(search_results)} search results")
        
        # Scraping: run multiple candidate URLs in parallel,
        # then keep the first N relevant pages.
        scraped_results = []
        sources = []
        MAX_RELEVANT_PAGES = 2  # Target
        MAX_ATTEMPTS = 6  # 🔧 Increased to 6 (from 4)
        attempts = search_results[:MAX_ATTEMPTS]
        attempts_made = len(attempts)
        parallelism = int(os.getenv("SCRAPE_PARALLELISM", "3"))
        parallelism = max(1, min(parallelism, 6))
        logger.info(f"⚡ Parallel scraping enabled: attempts={attempts_made} workers={parallelism}")

        def _safe_scrape(u: str):
            try:
                return {"ok": True, "data": webscraper(u, prefer_method="auto")}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        executor = ThreadPoolExecutor(max_workers=parallelism)
        futures = {}
        try:
            for idx, result in enumerate(attempts):
                url = result.get("link", "")
                title = result.get("title", "No title")
                if not url:
                    continue
                logger.info(f"📄 [queue {idx + 1}/{attempts_made}] Scraping: {title[:60]}...")
                futures[executor.submit(_safe_scrape, url)] = {"idx": idx, "url": url, "title": title}

            for fut in as_completed(futures):
                meta = futures[fut]
                idx = meta["idx"]
                url = meta["url"]
                title = meta["title"]

                res = fut.result()
                if not res.get("ok"):
                    err = (res.get("error") or "")[:140]
                    logger.warning(f"  ✗ [{idx + 1}/{attempts_made}] failed: {err}")
                    continue

                scraped_data = res.get("data") or {}
                content = (scraped_data.get("full_text") or "").strip()
                if not content:
                    logger.warning(f"  ✗ [{idx + 1}/{attempts_made}] empty content")
                    continue

                page_title = scraped_data.get("title") or title
                is_relevant, reason = is_content_relevant(content, page_title, search_query)
                if not is_relevant:
                    logger.warning(f"  ✗ [{idx + 1}/{attempts_made}] rejected: {reason}")
                    continue

                scraped_results.append({"title": page_title, "content": content, "url": url})
                sources.append({"title": page_title, "url": url, "snippet": content[:200] + "..."})
                logger.info(f"  ✅ [{idx + 1}/{attempts_made}] accepted: {len(content)} chars")

                if len(scraped_results) >= MAX_RELEVANT_PAGES:
                    logger.info(f"✅ Reached target of {MAX_RELEVANT_PAGES} relevant pages (stopping early)")
                    break
        finally:
            # Best-effort cancel pending tasks (won't stop tasks already running).
            for f in futures:
                if not f.done():
                    f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        
        # 🔧 FIX 5: Check we have at least 1 result
        if not scraped_results:
            logger.error(f"❌ No content could be extracted from {attempts_made} attempts")
            return {
                "ok": False,
                "search_query": search_query,
                "results": [],
                "sources": [],
                "context": "",
                "error": "Found search results but couldn't extract content. Sites may be blocking scraping or require JavaScript.",
                "instructions": "When answering the question, reference the sources inline by wrapping the index in brackets like this: [1]. If multiple sources are used, reference each without commas like this: [1][2][3]."
            }
        
        # Build context from scraped results
        context = ""
        for idx, result in enumerate(scraped_results):
            context += f"Source [{idx + 1}]: {result['title']}\n{result['content']}\n\n----------\n\n"
        
        logger.info(f"✅ SUCCESS: {len(scraped_results)} pages scraped from {attempts_made} attempts")
        
        payload = {
            "ok": True,
            "search_query": search_query,
            "results": scraped_results,
            "sources": sources,
            "context": context,
            "instructions": "When answering the question, reference the sources inline by wrapping the index in brackets like this: [1]. If multiple sources are used, reference each without commas like this: [1][2][3]."
        }
        if cache_ttl > 0:
            with _WEB_SEARCH_CACHE_LOCK:
                _WEB_SEARCH_CACHE[cache_key] = (time.time(), payload)
        return payload
    except Exception as e:
        logger.error(f"❌ Web search error: {str(e)}")
        return {
            "ok": False,
            "error": str(e),
            "results": [],
            "sources": [],
            "context": "",
        }


# =============================================================================
# CUSTOM visit_webpage TOOL WITH CONTENT TRUNCATION
# =============================================================================
# This overrides the default smolagents visit_webpage to prevent context overflow

@tool
def visit_webpage(url: str) -> dict:
    """
    Visits a webpage at the given URL and reads its content as a markdown string.
    Content is automatically truncated to prevent context overflow.
    
    Args:
        url: The URL of the webpage to visit.
        
    Returns:
        dict: Structured content payload with url/title/content
    """
    MAX_CONTENT_LENGTH = 3000  # Limit to prevent LLM context overflow
    
    logger.info(f"🌐 visit_webpage called for: {url}")
    
    try:
        # Use BeautifulSoup for fast content extraction
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = session.get(url, headers=headers, timeout=20, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, nav, footer, header elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
            element.decompose()
        
        # Try to find main content
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.article']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Get title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ""
        
        # Truncate content
        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH]
            # Cut at last complete sentence or paragraph
            last_period = text.rfind('.')
            last_newline = text.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > MAX_CONTENT_LENGTH * 0.7:
                text = text[:cut_point + 1]
            text += "\n\n[... content truncated for brevity ...]"
        
        content = f"# {title_text}\n\n{text}" if title_text else text
        
        logger.info(f"✅ visit_webpage extracted {len(content)} chars from {url}")
        return {
            "ok": True,
            "url": url,
            "title": title_text,
            "content": content,
            "context": content,
            "sources": [{"title": title_text, "url": url}],
        }
        
    except Exception as e:
        error_msg = f"Error visiting {url}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            "ok": False,
            "url": url,
            "title": "",
            "content": "",
            "context": "",
            "sources": [],
            "error": error_msg,
        }