#SEMAINE 1 (Backend core)
#├── 1.1 search_pubmed      ← COMMENCE ICI
#├── 1.2 extract_medical_entities
#├── 1.3 build_knowledge_graph
#└── 1.4 requirements.txt

#SEMAINE 2 (API + Frontend setup)
#├── 2.1-2.3 Endpoints FastAPI
#├── 3.1 npm install libs
#└── 3.2 GraphViewer.tsx (basic)

#SEMAINE 3 (UI complète)
#├── 3.3-3.5 Composants dashboard
#├── 4.1 Intégration UI ↔ API
#└── 4.2 Test use-case Alzheimer

#SEMAINE 4 (Polish)
#├── 4.3 UX improvements
#└── 4.4 Documentation + démo

"""
PubMed Search Tool for Medical Literature Retrieval

Uses NCBI E-utilities API (ESearch + EFetch).

Environment variables (.env):
    NCBI_API_KEY        - NCBI API key (increases rate limit from 3 to 10 req/sec)
    PUBMED_API_KEY      - Fallback api key name if you stored the NCBI key under this name
    NCBI_EMAIL          - Your email (recommended by NCBI)
    NCBI_TOOL           - Tool name identifier (default: med-assist)
    NCBI_BASE_URL       - NCBI base URL (default: https://eutils.ncbi.nlm.nih.gov/entrez/eutils)
    PUBMED_USE_NCBI     - Toggle to use NCBI API (default: true)
"""

import os
import logging
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from smolagents import tool

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration helpers
# =============================================================================

def _env(name: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(name, default)


# =============================================================================
# NCBI E-utilities backend
# =============================================================================

def _ncbi_base_url() -> str:
    base = _env("NCBI_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils").strip()
    return base.rstrip("/")


def ncbi_research(
    query: str,
    retmax: int = 20,
    retstart: int = 0,
    sort: str = "relevance",
    mindate: str = "",
    maxdate: str = "",
) -> Dict[str, Any]:
    """
    Search PubMed via NCBI ESearch API.
    Returns PMIDs matching the query.
    """
    url = f"{_ncbi_base_url()}/esearch.fcgi"
    
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max(1, min(int(retmax), 10000)),
        "retstart": max(0, int(retstart)),
    }
    
    # Sort options: relevance, pub_date, Author, JournalName
    if sort:
        params["sort"] = sort
    
    # Date filtering
    if mindate or maxdate:
        params["datetype"] = "pdat"  # Publication date
        if mindate:
            params["mindate"] = mindate
        if maxdate:
            params["maxdate"] = maxdate
    
    # NCBI recommends identifying your tool
    email = _env("NCBI_EMAIL")
    api_key = _env("NCBI_API_KEY")
    tool_name = _env("NCBI_TOOL", "med-assist")
    
    if email:
        params["email"] = email
    if tool_name:
        params["tool"] = tool_name
    if api_key:
        params["api_key"] = api_key
    
    logger.info(f"🔍 NCBI ESearch: {query[:50]}... (max={retmax}, start={retstart})")
    
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def ncbi_fetch(pmids: List[str], rettype: str = "abstract") -> str:
    """
    Fetch article details from PubMed via NCBI EFetch API.
    Returns XML with abstracts and metadata.
    """
    if not pmids:
        return ""
    
    url = f"{_ncbi_base_url()}/efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": ",".join(pmids[:200]),  # Max 200 per request
        "retmode": "xml",
        "rettype": rettype,
    }
    
    email = _env("NCBI_EMAIL")
    api_key = _env("NCBI_API_KEY")
    tool_name = _env("NCBI_TOOL", "med-assist")
    
    if email:
        params["email"] = email
    if tool_name:
        params["tool"] = tool_name
    if api_key:
        params["api_key"] = api_key
    
    logger.info(f"📄 NCBI EFetch: {len(pmids)} PMIDs")
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def ncbi_parse_efetch_xml(xml_text: str) -> List[Dict[str, Any]]:
    """
    Parse EFetch XML response into structured article data.
    """
    import xml.etree.ElementTree as ET
    
    articles = []
    
    try:
        root = ET.fromstring(xml_text)
        
        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
            
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            article_elem = medline.find("Article")
            if article_elem is None:
                continue
            
            # Title
            title_elem = article_elem.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_elem = article_elem.find("Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Journal
            journal_elem = article_elem.find("Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Publication date
            pub_date = ""
            date_elem = article_elem.find("Journal/JournalIssue/PubDate")
            if date_elem is not None:
                year = date_elem.find("Year")
                month = date_elem.find("Month")
                if year is not None:
                    pub_date = year.text
                    if month is not None:
                        pub_date = f"{month.text} {pub_date}"
            
            # Authors
            authors = []
            for author in article_elem.findall("AuthorList/Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None:
                    name = lastname.text
                    if forename is not None:
                        name = f"{forename.text} {name}"
                    authors.append(name)
            
            # MeSH terms
            mesh_terms = []
            for mesh in medline.findall("MeshHeadingList/MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_date": pub_date,
                "authors": authors,
                "mesh_terms": mesh_terms,
            })
    
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
    
    return articles



# =============================================================================
# Main search_pubmed tool
# =============================================================================

@tool
def search_pubmed(
    query: str,
    max_results: int = 20,
    start: int = 0,
    sort: str = "relevance",
    mindate: str = "",
    maxdate: str = "",
    fetch_details: bool = True,
) -> dict:
    """
    Search PubMed for medical literature.
    
    Args:
        query: Search query (supports PubMed syntax: MeSH terms, [Title/Abstract], etc.)
               Examples: "alzheimer treatment", "COVID-19[Title] AND vaccine[MeSH]"
        max_results: Maximum number of results to return (1-10000, default: 20)
        start: Starting index for pagination (default: 0)
        sort: Sort order - "relevance", "pub_date", "Author", "JournalName"
        mindate: Minimum publication date (format: YYYY or YYYY/MM or YYYY/MM/DD)
        maxdate: Maximum publication date (format: YYYY or YYYY/MM or YYYY/MM/DD)
        fetch_details: If True, fetch article details (title, abstract, authors)
    
    Returns:
        dict: {
            "provider": "ncbi",
            "query": str,
            "total": int,
            "start": int,
            "max_results": int,
            "pmids": list[str],
            "articles": list[dict],  # If fetch_details=True
            "error": str  # If error occurred
        }
    
    Examples:
        >>> search_pubmed("alzheimer treatment", max_results=10)
        >>> search_pubmed("COVID-19 vaccine", mindate="2023", maxdate="2024")
        >>> search_pubmed("cancer[MeSH] AND immunotherapy", sort="pub_date")
    """
    use_ncbi_setting = _env("PUBMED_USE_NCBI", "true").lower().strip()
    if use_ncbi_setting in {"false", "0", "no"}:
        return {
            "error": "PUBMED_USE_NCBI is disabled (no custom connector is implemented)",
            "query": query,
            "provider": "ncbi",
            "pmids": [],
            "articles": [],
            "total": 0,
        }

    if _env("PUBMED_API_KEY") and not _env("NCBI_API_KEY"):
        logger.warning(
            "PUBMED_API_KEY is set but NCBI_API_KEY is not set. "
            "This tool uses NCBI E-utilities and will only send an API key if NCBI_API_KEY is provided."
        )

    # Validate parameters
    max_results = max(1, min(int(max_results), 10000))
    start = max(0, int(start))
    
    try:
        # Search PubMed via NCBI ESearch
        esearch_result = ncbi_research(
            query=query,
            retmax=max_results,
            retstart=start,
            sort=sort,
            mindate=mindate,
            maxdate=maxdate,
        )
        
        result = esearch_result.get("esearchresult", {})
        pmids = result.get("idlist", []) or []
        total = int(result.get("count", 0) or 0)
        
        response = {
            "provider": "ncbi",
            "query": query,
            "total": total,
            "start": start,
            "max_results": max_results,
            "pmids": pmids,
            "articles": [],
        }
        
        # Fetch article details if requested
        if fetch_details and pmids:
            xml_data = ncbi_fetch(pmids)
            articles = ncbi_parse_efetch_xml(xml_data)
            response["articles"] = articles
            logger.info(f"✅ Found {total} results, fetched {len(articles)} articles")
        else:
            logger.info(f"✅ Found {total} results, {len(pmids)} PMIDs")
        
        return response
    
    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Timeout querying PubMed: {query[:50]}...")
        return {
            "error": "Timeout while querying PubMed",
            "query": query,
            "provider": "ncbi",
            "pmids": [],
            "articles": [],
            "total": 0,
        }
    
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", "")[:500]
        logger.error(f"❌ HTTP {status} from PubMed: {text}")
        return {
            "error": f"HTTP error {status} from PubMed",
            "status_code": status,
            "details": text,
            "query": query,
            "provider": "ncbi",
            "pmids": [],
            "articles": [],
            "total": 0,
        }
    
    except Exception as e:
        logger.error(f"❌ PubMed search error: {e}")
        return {
            "error": str(e),
            "query": query,
            "provider": "ncbi",
            "pmids": [],
            "articles": [],
            "total": 0,
        }
