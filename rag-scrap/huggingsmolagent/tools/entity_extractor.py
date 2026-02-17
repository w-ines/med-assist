"""
GLiNER2 Entity Extraction Module

This module provides entity extraction capabilities using GLiNER2 model
to enhance RAG retrieval with semantic entity understanding.

Features:
- Singleton pattern for efficient model loading
- Query entity extraction for enriched retrieval
- Document entity extraction for metadata enrichment
- Entity-aware chunking for better context preservation
- Entity overlap scoring for relevance ranking

Usage:
    from huggingsmolagent.tools.entity_extractor import (
        get_entity_extractor,
        extract_query_entities,
        enrich_document_metadata,
        entity_aware_chunking
    )
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

# Lazy import to avoid loading model at module import
_gliner_instance = None
_gliner_available = None


def _check_gliner_available() -> bool:
    """Check if GLiNER2 package is installed."""
    global _gliner_available
    if _gliner_available is None:
        try:
            from gliner2 import GLiNER2
            _gliner_available = True
        except ImportError:
            print("⚠️  GLiNER2 not available - install with: pip install gliner2")
            _gliner_available = False
    return _gliner_available


def get_entity_extractor():
    """
    Get or create the GLiNER2 entity extractor singleton.
    
    Uses lazy loading to avoid slow startup times.
    Model is cached in memory after first load (~2-3 seconds).
    
    Returns:
        GLiNER2 instance or None if not available
    """
    global _gliner_instance
    
    if not _check_gliner_available():
        return None
    
    if _gliner_instance is None:
        try:
            from gliner2 import GLiNER2
            print("🔄 Loading GLiNER2 model (fastino/gliner2-base-v1)...")
            start = time.time()
            _gliner_instance = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
            print(f"✅ GLiNER2 loaded in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load GLiNER2: {e}")
            return None
    
    return _gliner_instance


# ============================================================================
# ENTITY TYPES - Customizable for different domains
# ============================================================================

# Default entity types for general RAG use
DEFAULT_ENTITY_TYPES = [
    "person",
    "company", 
    "organization",
    "product",
    "technology",
    "location",
    "date",
    "event"
]

# Scientific/research document entity types
RESEARCH_ENTITY_TYPES = [
    "person",
    "organization",
    "method",
    "dataset",
    "metric",
    "technology",
    "concept",
    "location"
]

# Business document entity types
BUSINESS_ENTITY_TYPES = [
    "person",
    "company",
    "product",
    "price",
    "date",
    "location",
    "contract",
    "regulation"
]


def extract_query_entities(
    query: str,
    entity_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract entities from a user query for enhanced retrieval.
    
    This enables:
    - Query understanding (what is the user looking for?)
    - Query enrichment (add entity context to search)
    - Intent classification based on entity types
    
    Args:
        query: The user's search query
        entity_types: List of entity types to extract (default: DEFAULT_ENTITY_TYPES)
    
    Returns:
        Dict with:
            - entities: Dict mapping entity_type -> list of extracted values
            - query_intent: Inferred intent based on entities
            - enriched_query: Query enhanced with entity context
            - extraction_time: Time taken for extraction
    
    Example:
        >>> result = extract_query_entities("What is Apple's iPhone 15 price?")
        >>> print(result['entities'])
        {'company': ['Apple'], 'product': ['iPhone 15']}
    """
    extractor = get_entity_extractor()
    
    if extractor is None:
        # Fallback: return empty result if GLiNER2 not available
        return {
            "entities": {},
            "query_intent": "general",
            "enriched_query": query,
            "extraction_time": 0.0,
            "gliner_available": False
        }
    
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    start_time = time.time()
    
    try:
        # Extract entities using GLiNER2
        result = extractor.extract_entities(query, entity_types)
        entities = result.get("entities", {})
        
        # Infer query intent based on detected entity types
        query_intent = _infer_query_intent(entities)
        
        # Create enriched query with entity context
        enriched_query = _enrich_query_with_entities(query, entities)
        
        extraction_time = time.time() - start_time
        
        return {
            "entities": entities,
            "query_intent": query_intent,
            "enriched_query": enriched_query,
            "extraction_time": extraction_time,
            "gliner_available": True
        }
    except Exception as e:
        print(f"⚠️  Entity extraction failed: {e}")
        return {
            "entities": {},
            "query_intent": "general",
            "enriched_query": query,
            "extraction_time": time.time() - start_time,
            "error": str(e),
            "gliner_available": True
        }


def _infer_query_intent(entities: Dict[str, List[str]]) -> str:
    """
    Infer query intent based on extracted entity types.
    
    Returns one of: technical, business, research, news, general
    """
    # Count entity types
    has_tech = bool(entities.get("technology") or entities.get("product"))
    has_business = bool(entities.get("company") or entities.get("price"))
    has_research = bool(entities.get("method") or entities.get("dataset") or entities.get("metric"))
    has_temporal = bool(entities.get("date") or entities.get("event"))
    
    if has_research:
        return "research"
    elif has_business and has_tech:
        return "business_tech"
    elif has_business:
        return "business"
    elif has_tech:
        return "technical"
    elif has_temporal:
        return "news"
    else:
        return "general"


def _enrich_query_with_entities(query: str, entities: Dict[str, List[str]]) -> str:
    """
    Enrich query with entity type context for better embedding retrieval.
    
    Example:
        "iPhone 15 price" -> "iPhone 15 price [product: iPhone 15]"
    """
    entity_context = []
    for entity_type, values in entities.items():
        if values:
            entity_context.append(f"{entity_type}: {', '.join(values)}")
    
    if entity_context:
        return f"{query} [{' | '.join(entity_context)}]"
    return query


# ============================================================================
# DOCUMENT ENRICHMENT - Add entity metadata during ingestion
# ============================================================================

def enrich_document_metadata(
    doc: Document,
    entity_types: Optional[List[str]] = None,
    max_content_length: int = 3000
) -> Document:
    """
    Enrich a document's metadata with extracted entities.
    
    This enables:
    - Entity-based filtering during retrieval
    - Better relevance scoring
    - Automatic document classification
    
    Args:
        doc: LangChain Document to enrich
        entity_types: Entity types to extract
        max_content_length: Max chars to analyze (for speed)
    
    Returns:
        Document with enriched metadata containing:
            - auto_entities: Dict of extracted entities
            - auto_domain: Inferred document domain
            - entity_count: Total number of entities found
    """
    extractor = get_entity_extractor()
    
    if extractor is None:
        return doc
    
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    content = doc.page_content[:max_content_length]
    
    try:
        result = extractor.extract_entities(content, entity_types)
        entities = result.get("entities", {})
        
        # Calculate entity count
        entity_count = sum(len(v) for v in entities.values())
        
        # Infer document domain
        domain = _infer_query_intent(entities)
        
        # Create enriched metadata
        enriched_meta = dict(doc.metadata or {})
        enriched_meta.update({
            "auto_entities": entities,
            "auto_domain": domain,
            "entity_count": entity_count,
            "gliner_enriched": True
        })
        
        return Document(page_content=doc.page_content, metadata=enriched_meta)
    
    except Exception as e:
        print(f"⚠️  Document enrichment failed: {e}")
        return doc


def enrich_documents_batch(
    docs: List[Document],
    entity_types: Optional[List[str]] = None,
    max_content_length: int = 3000
) -> List[Document]:
    """
    Batch enrich multiple documents with entity metadata.
    
    Args:
        docs: List of documents to enrich
        entity_types: Entity types to extract
        max_content_length: Max chars per doc to analyze
    
    Returns:
        List of enriched documents
    """
    extractor = get_entity_extractor()
    
    if extractor is None:
        print("⚠️  GLiNER2 not available - skipping batch enrichment")
        return docs
    
    print(f"🔍 Enriching {len(docs)} documents with GLiNER2...")
    start = time.time()
    
    enriched = []
    for doc in docs:
        enriched.append(
            enrich_document_metadata(doc, entity_types, max_content_length)
        )
    
    print(f"✅ Enriched {len(enriched)} documents in {time.time() - start:.2f}s")
    return enriched


# ============================================================================
# ENTITY-AWARE CHUNKING - Preserve entity boundaries
# ============================================================================

def entity_aware_chunking(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    entity_types: Optional[List[str]] = None
) -> List[str]:
    """
    Split text into chunks while respecting entity boundaries.
    
    This prevents cutting entities in half, which improves:
    - Entity extraction from chunks
    - Embedding quality (entities stay intact)
    - Retrieval accuracy
    
    Args:
        text: Full text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        entity_types: Entity types to detect for boundary preservation
    
    Returns:
        List of text chunks with preserved entity boundaries
    """
    extractor = get_entity_extractor()
    
    if extractor is None:
        # Fallback to simple sentence-based chunking
        return _simple_chunking(text, chunk_size, chunk_overlap)
    
    entity_types = entity_types or ["person", "company", "product", "location"]
    
    try:
        # Extract all entities from the full text
        result = extractor.extract_entities(text, entity_types)
        entities = result.get("entities", {})
        
        # Get all entity values for boundary checking
        all_entity_values = []
        for values in entities.values():
            all_entity_values.extend(values)
        
        # Chunk with entity awareness
        return _chunk_with_entity_awareness(
            text, 
            all_entity_values, 
            chunk_size, 
            chunk_overlap
        )
    
    except Exception as e:
        print(f"⚠️  Entity-aware chunking failed: {e}")
        return _simple_chunking(text, chunk_size, chunk_overlap)


def _simple_chunking(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple sentence-based chunking fallback."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _chunk_with_entity_awareness(
    text: str,
    entity_values: List[str],
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Chunk text while avoiding cutting through entity values.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        potential_chunk = current_chunk + sentence + " "
        
        if len(potential_chunk) < chunk_size:
            current_chunk = potential_chunk
        else:
            # Check if ending here would cut an entity
            if not _cuts_entity(current_chunk, entity_values):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                # Continue to avoid cutting entity
                if len(potential_chunk) < chunk_size * 1.5:  # Allow 50% overflow
                    current_chunk = potential_chunk
                else:
                    # Too long, force chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _cuts_entity(text: str, entity_values: List[str]) -> bool:
    """Check if text ends in the middle of an entity."""
    text_end = text.rstrip()[-50:] if len(text) > 50 else text.rstrip()
    
    for entity in entity_values:
        if len(entity) > 3:  # Ignore very short entities
            # Check if entity is partially at the end
            for i in range(1, len(entity)):
                partial = entity[:i]
                if text_end.endswith(partial) and not text_end.endswith(entity):
                    return True
    return False


# ============================================================================
# ENTITY OVERLAP SCORING - Relevance ranking enhancement
# ============================================================================

def compute_entity_overlap_score(
    query_entities: Dict[str, List[str]],
    doc_entities: Dict[str, List[str]],
    weighted: bool = True
) -> float:
    """
    Compute similarity score based on entity overlap.
    
    This can be combined with vector similarity for hybrid ranking.
    
    Args:
        query_entities: Entities from user query
        doc_entities: Entities from document
        weighted: Apply type-based weighting (person > location)
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Entity type weights (higher = more important for matching)
    weights = {
        "person": 1.5,
        "company": 1.3,
        "product": 1.3,
        "technology": 1.2,
        "organization": 1.1,
        "location": 0.8,
        "date": 0.7,
        "event": 1.0
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for entity_type, query_values in query_entities.items():
        if not query_values:
            continue
        
        doc_values = doc_entities.get(entity_type, [])
        
        # Normalize for case-insensitive matching
        query_set = {v.lower() for v in query_values}
        doc_set = {v.lower() for v in doc_values}
        
        # Calculate overlap
        matches = query_set & doc_set
        
        if query_set:
            type_score = len(matches) / len(query_set)
            weight = weights.get(entity_type, 1.0) if weighted else 1.0
            total_score += type_score * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_score / total_weight


def enhance_retrieval_results(
    query: str,
    docs: List[Document],
    entity_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Enhance retrieval results with entity extraction and overlap scoring.
    
    Args:
        query: Original user query
        docs: Retrieved documents
        entity_types: Entity types to extract
    
    Returns:
        List of enhanced result dicts with:
            - content: Document content
            - metadata: Original metadata
            - query_entities: Entities from query
            - doc_entities: Entities from document
            - entity_overlap_score: Similarity based on entities
            - entity_highlights: Text snippets around matching entities
    """
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    
    # Extract query entities
    query_result = extract_query_entities(query, entity_types)
    query_entities = query_result.get("entities", {})
    
    enhanced_results = []
    
    for doc in docs:
        # Check if document was already enriched during ingestion
        doc_entities = doc.metadata.get("auto_entities")
        
        if doc_entities is None:
            # Extract entities from document content
            extractor = get_entity_extractor()
            if extractor:
                try:
                    result = extractor.extract_entities(
                        doc.page_content[:2000], 
                        entity_types
                    )
                    doc_entities = result.get("entities", {})
                except Exception:
                    doc_entities = {}
            else:
                doc_entities = {}
        
        # Compute entity overlap score
        overlap_score = compute_entity_overlap_score(query_entities, doc_entities)
        
        # Create highlights around matching entities
        highlights = _create_entity_highlights(
            doc.page_content, 
            query_entities, 
            doc_entities
        )
        
        enhanced_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "query_entities": query_entities,
            "doc_entities": doc_entities,
            "entity_overlap_score": overlap_score,
            "entity_highlights": highlights
        })
    
    return enhanced_results


def _create_entity_highlights(
    text: str,
    query_entities: Dict[str, List[str]],
    doc_entities: Dict[str, List[str]],
    context_chars: int = 60
) -> List[str]:
    """Create text snippets around matching entities."""
    highlights = []
    
    # Find matching entities
    for entity_type in query_entities:
        query_values = set(v.lower() for v in query_entities.get(entity_type, []))
        doc_values = doc_entities.get(entity_type, [])
        
        for doc_value in doc_values:
            if doc_value.lower() in query_values:
                # Find the entity in text and extract context
                idx = text.lower().find(doc_value.lower())
                if idx != -1:
                    start = max(0, idx - context_chars)
                    end = min(len(text), idx + len(doc_value) + context_chars)
                    snippet = text[start:end]
                    
                    # Clean up snippet
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(text):
                        snippet = snippet + "..."
                    
                    highlights.append(snippet.strip())
    
    return highlights[:5]  # Return top 5 highlights


# ============================================================================
# DOCUMENT CLASSIFICATION - Auto-classify document type and domain
# ============================================================================

def classify_document(
    content: str,
    max_content_length: int = 2000
) -> Dict[str, Any]:
    """
    Automatically classify a document's type and domain using entity analysis.
    
    Args:
        content: Document text content
        max_content_length: Max chars to analyze
    
    Returns:
        Dict with:
            - doc_type: Inferred document type
            - domain: Subject domain
            - main_entities: Most prominent entities
            - confidence: Classification confidence
    """
    extractor = get_entity_extractor()
    
    if extractor is None:
        return {
            "doc_type": "unknown",
            "domain": "general",
            "main_entities": {},
            "confidence": 0.0
        }
    
    content = content[:max_content_length]
    
    try:
        # Extract entities with extended types
        extended_types = [
            "person", "company", "organization", "product", "technology",
            "location", "date", "method", "dataset", "regulation", "law"
        ]
        
        result = extractor.extract_entities(content, extended_types)
        entities = result.get("entities", {})
        
        # Determine domain based on entity patterns
        domain = _infer_query_intent(entities)
        
        # Determine document type based on patterns
        doc_type = _infer_doc_type(content, entities)
        
        # Get main entities (most frequent/important)
        main_entities = _get_main_entities(entities)
        
        # Calculate confidence based on entity density
        total_entities = sum(len(v) for v in entities.values())
        confidence = min(1.0, total_entities / 10)  # More entities = higher confidence
        
        return {
            "doc_type": doc_type,
            "domain": domain,
            "main_entities": main_entities,
            "confidence": confidence
        }
    
    except Exception as e:
        print(f"⚠️  Document classification failed: {e}")
        return {
            "doc_type": "unknown",
            "domain": "general",
            "main_entities": {},
            "confidence": 0.0,
            "error": str(e)
        }


def _infer_doc_type(content: str, entities: Dict[str, List[str]]) -> str:
    """Infer document type from content patterns and entities."""
    content_lower = content.lower()
    
    # Check for document type indicators
    if any(kw in content_lower for kw in ["abstract", "methodology", "references", "conclusion"]):
        return "research_paper"
    elif any(kw in content_lower for kw in ["article", "section", "pursuant", "hereby"]):
        return "legal_document"
    elif any(kw in content_lower for kw in ["installation", "configuration", "api", "function"]):
        return "technical_manual"
    elif any(kw in content_lower for kw in ["quarterly", "fiscal", "revenue", "growth"]):
        return "business_report"
    elif any(kw in content_lower for kw in ["dear", "regards", "sincerely", "attached"]):
        return "email"
    elif entities.get("date") and len(entities.get("person", [])) > 2:
        return "news_article"
    else:
        return "general_document"


def _get_main_entities(entities: Dict[str, List[str]], max_per_type: int = 3) -> Dict[str, List[str]]:
    """Get the most prominent entities per type."""
    main = {}
    for entity_type, values in entities.items():
        if values:
            main[entity_type] = values[:max_per_type]
    return main

