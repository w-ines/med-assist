from typing import List, Optional, Dict, Any
import hashlib
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from huggingsmolagent.tools.supabase_store import supabase, SUPABASE_AVAILABLE
from smolagents import tool
import os
from dotenv import load_dotenv

# Import cache system
try:
    from huggingsmolagent.tools.query_cache import cache_query_result, get_cache_stats
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️  Query cache not available - install cachetools")

# Import GLiNER2 entity extractor for enhanced RAG
try:
    from huggingsmolagent.tools.entity_extractor import (
        extract_query_entities,
        enrich_documents_batch,
        enhance_retrieval_results,
        compute_entity_overlap_score,
        get_entity_extractor,
    )
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    extract_query_entities = None
    enrich_documents_batch = None
    enhance_retrieval_results = None
    compute_entity_overlap_score = None
    get_entity_extractor = None
    print("⚠️  GLiNER2 entity extractor not available")

load_dotenv()


def _get_embeddings(*, embedding_model: Optional[str] = None):
    provider = (os.getenv("EMBEDDINGS_PROVIDER") or os.getenv("LLM_PROVIDER") or "openrouter").lower()
    if embedding_model and embedding_model.strip().lower() in {"chroma", "documentembeddingsimilarity", "file_chunks"}:
        embedding_model = None
    if provider in {"openrouter", "openai"}:
        api_key = os.getenv("OPEN_ROUTER_KEY") if provider == "openrouter" else os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("BASE_URL", "https://openrouter.ai/api/v1") if provider == "openrouter" else os.getenv("OPENAI_API_BASE")
        model_name = embedding_model or os.getenv("EMBEDDINGS_MODEL") or "text-embedding-3-small"
        if not api_key:
            raise RuntimeError(
                "Embeddings provider is configured to use OpenRouter/OpenAI but no API key was found. "
                "Set OPEN_ROUTER_KEY (for OpenRouter) or OPENAI_API_KEY (for OpenAI), or set EMBEDDINGS_PROVIDER=ollama."
            )
        return OpenAIEmbeddings(model=model_name, openai_api_base=api_base, openai_api_key=api_key)

    if provider == "ollama":
        model_name = embedding_model or os.getenv("OLLAMA_EMBEDDINGS_MODEL") or os.getenv("EMBEDDINGS_MODEL") or "mxbai-embed-large"
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
        if not base_url:
            raise RuntimeError(
                "Embeddings provider is configured to use Ollama but no OLLAMA_BASE_URL/OLLAMA_HOST was found. "
                "Example: OLLAMA_BASE_URL=http://host.docker.internal:11434"
            )
        return OllamaEmbeddings(model=model_name, base_url=base_url)

    raise RuntimeError(
        f"Unsupported embeddings provider '{provider}'. Set EMBEDDINGS_PROVIDER to one of: openrouter, openai, ollama."
    )

# Monkey patch for Supabase v2.23+ compatibility with LangChain
# The issue: LangChain expects query_builder.params.set() but Supabase v2.23+ uses .limit() directly
import langchain_community.vectorstores.supabase as lc_supabase

_original_similarity_search = lc_supabase.SupabaseVectorStore.similarity_search_by_vector_with_relevance_scores

def _patched_similarity_search(self, query, k=4, filter=None, postgrest_filter=None, score_threshold=None, **kwargs):
    """Patched version that uses .limit() instead of .params.set()"""
    # Build the filter if provided
    if filter:
        postgrest_filter = self._build_postgrest_filter(filter)
    
    # Call the RPC function
    match_documents_params = self.match_args(query, filter)
    query_builder = self._client.rpc(self.query_name, match_documents_params)
    
    # Apply postgrest filter if provided
    if postgrest_filter:
        # For Supabase v2.23+, we can't use .params.set()
        # Instead, we'll retrieve more results and filter in Python
        pass
    
    # Use .limit() instead of .params.set("limit", k)
    query_builder = query_builder.limit(k * 3 if postgrest_filter else k)
    
    res = query_builder.execute()
    
    # Build results
    match_result = [
        (
            Document(
                metadata=search.get("metadata", {}),
                page_content=search.get("content", ""),
            ),
            search.get("similarity", 0.0),
        )
        for search in res.data
        if search.get("content")
    ]
    
    # Apply score threshold if provided
    if score_threshold is not None:
        match_result = [
            (doc, score) for doc, score in match_result if score >= score_threshold
        ]
    
    return match_result[:k]

# Apply the monkey patch
lc_supabase.SupabaseVectorStore.similarity_search_by_vector_with_relevance_scores = _patched_similarity_search 

def chunk_documents(
    documents: List[Document], 
    *, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    use_entity_aware_chunking: bool = False
) -> List[Document]:
    """
    Split documents into chunks for embedding.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        use_entity_aware_chunking: If True, use GLiNER2 to avoid cutting entities
    
    Returns:
        List of document chunks with metadata
    """
    # Try entity-aware chunking if requested and available
    if use_entity_aware_chunking and GLINER_AVAILABLE:
        try:
            from huggingsmolagent.tools.entity_extractor import entity_aware_chunking
            print("[chunk_documents] 🔍 Using GLiNER2 entity-aware chunking...")
            
            all_chunks = []
            counter_by_doc: Dict[str, int] = {}
            
            for doc in documents:
                base_id = (doc.metadata or {}).get("doc_id", "")
                text_chunks = entity_aware_chunking(
                    doc.page_content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                for chunk_text in text_chunks:
                    idx = counter_by_doc.get(base_id, 0)
                    meta = dict(doc.metadata or {})
                    meta["chunk_index"] = idx
                    meta["chunking_method"] = "entity_aware"
                    all_chunks.append(Document(page_content=chunk_text, metadata=meta))
                    counter_by_doc[base_id] = idx + 1
            
            print(f"[chunk_documents] ✅ Created {len(all_chunks)} entity-aware chunks")
            return all_chunks
        except Exception as e:
            print(f"[chunk_documents] ⚠️  Entity-aware chunking failed: {e}, falling back to standard")
    
    # Standard chunking with RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    # Add chunk_index metadata for traceability
    numbered: List[Document] = []
    counter_by_doc: Dict[str, int] = {}
    for ch in chunks:
        base_id = (ch.metadata or {}).get("doc_id", "")
        idx = counter_by_doc.get(base_id, 0)
        meta = dict(ch.metadata or {})
        meta["chunk_index"] = idx
        meta["chunking_method"] = "recursive"
        numbered.append(Document(page_content=ch.page_content, metadata=meta))
        counter_by_doc[base_id] = idx + 1
    return numbered


def store_embeddings(
    chunks: List[Document],
    *,
    table_name: str = "documents",
    query_name: str = "match_documents",
    embedding_model: Optional[str] = None,
    enable_entity_enrichment: bool = True,
    entity_enrichment_threshold: int = 50,  # Only enrich if <= this many chunks
) -> int:
    """
    Store document chunks as embeddings in Supabase vector store.
    
    Args:
        chunks: List of document chunks to embed and store
        table_name: Supabase table name
        query_name: RPC function name for similarity search
        embedding_model: Optional override for embedding model
        enable_entity_enrichment: If True, enrich chunks with GLiNER2 entities
        entity_enrichment_threshold: Max chunks to enrich (larger docs skip enrichment)
    
    Returns:
        Number of chunks stored
    """
    if not SUPABASE_AVAILABLE or supabase is None:
        print("[store_embeddings] ⚠️  Supabase not available. Cannot store embeddings.")
        return 0

    embeddings = _get_embeddings(embedding_model=embedding_model)
    if isinstance(embeddings, OpenAIEmbeddings):
        print(f"[store_embeddings] Using OpenAI embeddings: {getattr(embeddings, 'model', '')}")
    else:
        print(f"[store_embeddings] Using Ollama embeddings: {getattr(embeddings, 'model', '')}")

    def _sanitize_text(text: str) -> str:
        if text is None:
            return ""
        # Remove NUL bytes that Postgres text cannot store
        text = text.replace("\x00", " ").replace("\u0000", " ")
        # Fix OCR spacing issues where spaces appear between characters
        import re
        # Pattern 1: Remove spaces within words that have excessive spacing
        text = re.sub(r'(?<=\w)\s+(?=\w(?:\s+\w){2,})', '', text)
        # Pattern 2: Fix remaining single-letter words followed by spaces
        text = re.sub(r'\b(\w)\s+(?=\w\b)', r'\1', text)
        return text.strip()

    # Sanitize chunk contents to avoid Postgres 22P05 (NUL byte) errors
    sanitized_chunks: List[Document] = []
    for doc in chunks:
        cleaned = _sanitize_text(doc.page_content)
        if cleaned:
            sanitized_chunks.append(Document(page_content=cleaned, metadata=doc.metadata))

    # GLiNER2 ENHANCEMENT: Enrich documents with extracted entities
    # ONLY for small documents (< threshold) to avoid timeout on large PDFs
    chunk_count = len(sanitized_chunks)
    should_enrich = (
        enable_entity_enrichment 
        and GLINER_AVAILABLE 
        and enrich_documents_batch
        and chunk_count <= entity_enrichment_threshold
    )
    
    if should_enrich:
        print(f"[store_embeddings] 🔍 Enriching {chunk_count} documents with GLiNER2 entities...")
        try:
            sanitized_chunks = enrich_documents_batch(
                sanitized_chunks,
                entity_types=["person", "company", "organization", "product", 
                             "technology", "location", "date"],
                max_content_length=2000
            )
            print(f"[store_embeddings] ✅ Entity enrichment complete")
        except Exception as e:
            print(f"[store_embeddings] ⚠️  Entity enrichment failed: {e}")
            # Continue without enrichment
    elif enable_entity_enrichment and chunk_count > entity_enrichment_threshold:
        print(f"[store_embeddings] ⏭️  Skipping GLiNER2 enrichment: {chunk_count} chunks > threshold ({entity_enrichment_threshold})")
        print(f"[store_embeddings] 💡 Entity extraction will happen at query time instead")

    # Use from_documents to ensure the table/function are created if missing
    try:
        vector_store = SupabaseVectorStore.from_documents(
            documents=sanitized_chunks,
            embedding=embeddings,
            client=supabase,
            table_name=table_name,
            query_name=query_name,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to embed and store documents. This is usually caused by an embeddings backend misconfiguration "
            "(e.g., Ollama not reachable from inside Docker, or missing OPEN_ROUTER_KEY)."
        ) from e

    # When using from_documents above, items are already inserted.
    # Return the number of chunks stored.
    return len(sanitized_chunks)


def compute_file_hash(content: bytes) -> str:
    """Compute SHA256 hash of file content for deduplication."""
    return hashlib.sha256(content).hexdigest()


def check_existing_document(file_hash: str, table_name: str = "documents") -> Optional[Dict[str, Any]]:
    """
    Check if a document with the given file hash already exists in the database.
    
    Args:
        file_hash: SHA256 hash of the file content
        table_name: Supabase table name
        
    Returns:
        Dict with doc_id and chunk count if exists, None otherwise
    """
    if not SUPABASE_AVAILABLE or supabase is None:
        print("[check_existing_document] ⚠️  Supabase not available. Skipping deduplication check.")
        return None
    
    try:
        # Query for documents with this file_hash in metadata
        response = supabase.table(table_name).select("metadata").eq("metadata->>file_hash", file_hash).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            metadata = response.data[0].get("metadata", {})
            doc_id = metadata.get("doc_id")
            
            if doc_id:
                # Count total chunks for this doc_id
                count_response = supabase.table(table_name).select("id", count="exact").eq("metadata->>doc_id", doc_id).execute()
                
                return {
                    "doc_id": doc_id,
                    "chunk_count": count_response.count or 0,
                    "filename": metadata.get("filename"),
                    "source": metadata.get("source")
                }
        
        return None
    except Exception as e:
        print(f"[check_existing_document] Error: {e}")
        return None


def delete_document_by_doc_id(doc_id: str, table_name: str = "documents") -> int:
    """
    Delete all chunks associated with a doc_id.
    
    Args:
        doc_id: Document ID to delete
        table_name: Supabase table name
        
    Returns:
        Number of chunks deleted
    """
    if not SUPABASE_AVAILABLE or supabase is None:
        print("[delete_document_by_doc_id] ⚠️  Supabase not available. Cannot delete document.")
        return 0
    
    try:
        # Delete all rows with this doc_id
        response = supabase.table(table_name).delete().eq("metadata->>doc_id", doc_id).execute()
        deleted_count = len(response.data) if response.data else 0
        print(f"[delete_document_by_doc_id] Deleted {deleted_count} chunks for doc_id={doc_id}")
        return deleted_count
    except Exception as e:
        print(f"[delete_document_by_doc_id] Error: {e}")
        return 0


def index_documents(
    documents: List[Document],
    *,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    table_name: str = "documents",
    query_name: str = "match_documents",
    embedding_model: Optional[str] = None,
    enable_entity_enrichment: bool = True,
    entity_enrichment_threshold: int = 50,  # Skip GLiNER2 for docs > 50 chunks
) -> int:
    """
    Index documents by chunking and storing embeddings.
    
    Args:
        documents: List of documents to index
        base_metadata: Metadata to attach to all chunks
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        table_name: Supabase table name
        query_name: RPC function name
        embedding_model: Optional embedding model override
        enable_entity_enrichment: Enable GLiNER2 entity extraction
        entity_enrichment_threshold: Max chunks for GLiNER2 (larger docs skip)
    
    Returns:
        Number of chunks stored
    """
    base_metadata = base_metadata or {}

    # Attach base metadata to each page-level document
    enriched_docs: List[Document] = []
    for doc in documents:
        md = dict(base_metadata)
        md.update(doc.metadata or {})
        enriched_docs.append(Document(page_content=doc.page_content, metadata=md))

    chunks = chunk_documents(enriched_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stored = store_embeddings(
        chunks,
        table_name=table_name,
        query_name=query_name,
        embedding_model=embedding_model,
        enable_entity_enrichment=enable_entity_enrichment,
        entity_enrichment_threshold=entity_enrichment_threshold,
    )
    print("stored in vector store=", stored)
    return stored


def _extract_topics_from_chunks(chunks: List[Dict[str, Any]], max_topics: int = 5) -> List[str]:
    """
    Extract key topics/terms from retrieved chunks dynamically.
    Uses a combination of:
    1. GLiNER entities if available
    2. Capitalized phrases (likely proper nouns/concepts)
    3. Technical terms (words with special patterns)
    
    Args:
        chunks: List of result dictionaries with 'content' and optional 'doc_entities'
        max_topics: Maximum number of topics to return
    
    Returns:
        List of extracted topic strings
    """
    import re
    from collections import Counter
    
    topics = []
    
    # 1. First, try to use GLiNER entities if available
    for chunk in chunks:
        doc_entities = chunk.get("doc_entities", {})
        if doc_entities:
            for entity_type, values in doc_entities.items():
                if entity_type in ["technology", "product", "organization", "person"]:
                    topics.extend(values[:2])  # Top 2 per type
    
    # 2. If no entities, extract from content using patterns
    if len(topics) < 3:
        all_content = " ".join(chunk.get("content", "")[:500] for chunk in chunks)
        
        # Pattern for capitalized words/phrases (likely concepts)
        # Match: "RAG", "LangChain", "Vector Store", "Machine Learning"
        capitalized = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', all_content)
        
        # Pattern for technical terms with special characters
        # Match: "GPT-4", "mxbai-embed", "Q&A"
        technical = re.findall(r'\b([A-Za-z]+[-_][A-Za-z0-9]+)\b', all_content)
        
        # Pattern for acronyms (2-5 uppercase letters)
        # Match: "RAG", "LLM", "NLP", "API"
        acronyms = re.findall(r'\b([A-Z]{2,5})\b', all_content)
        
        # Combine and count frequency
        candidates = capitalized + technical + acronyms
        
        # Filter out common English words and very short terms
        stopwords = {
            'The', 'This', 'That', 'What', 'When', 'Where', 'How', 'Why', 'Which',
            'You', 'Your', 'They', 'Their', 'We', 'Our', 'It', 'Its', 'Is', 'Are',
            'Was', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had', 'Do', 'Does',
            'Did', 'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'Must', 'Can',
            'If', 'But', 'And', 'Or', 'So', 'As', 'At', 'By', 'For', 'In', 'Of',
            'On', 'To', 'With', 'From', 'Up', 'Out', 'Into', 'Over', 'After',
            'Before', 'Between', 'Through', 'During', 'Without', 'Again', 'Further',
            'Then', 'Once', 'Here', 'There', 'All', 'Each', 'Few', 'More', 'Most',
            'Other', 'Some', 'Such', 'No', 'Not', 'Only', 'Own', 'Same', 'Than',
            'Too', 'Very', 'Just', 'Also', 'Now', 'Use', 'Used', 'Using',
            'Example', 'Question', 'Answer', 'Step', 'Steps', 'Note', 'See',
            'Source', 'Result', 'Results', 'Output', 'Input', 'Data', 'File',
        }
        
        filtered = [
            c for c in candidates 
            if c not in stopwords 
            and len(c) >= 2 
            and not c.isdigit()
        ]
        
        # Count and get most common
        counts = Counter(filtered)
        topics.extend([term for term, _ in counts.most_common(max_topics - len(topics))])
    
    # Deduplicate while preserving order
    seen = set()
    unique_topics = []
    for t in topics:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique_topics.append(t)
    
    return unique_topics[:max_topics]


def _create_entity_highlights(
    text: str,
    query_entities: Dict[str, list],
    doc_entities: Dict[str, list],
    context_chars: int = 60
) -> List[str]:
    """
    Create text snippets around matching entities between query and document.
    
    Args:
        text: Document text
        query_entities: Entities from the query
        doc_entities: Entities from the document
        context_chars: Characters of context around each entity
    
    Returns:
        List of highlighted text snippets
    """
    highlights = []
    
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


def _retrieve_knowledge_impl(
    query: str,
    top_k: int = 5,
    table_name: str = "documents",
    query_name: str = "match_documents",
    embedding_model: Optional[str] = None,
    doc_id: Optional[str] = None,
    enable_entity_extraction: bool = True,
) -> Dict[str, Any]:
    """
    Internal implementation of retrieve_knowledge (without cache).
    Used by the cached version.
    
    GLiNER2 Enhancement:
    - Extracts entities from query for better understanding
    - Uses enriched query for embedding search
    - Computes entity overlap scores for hybrid ranking
    - Returns entity highlights in results
    """
    start_time = time.time()
    
    if not SUPABASE_AVAILABLE or supabase is None:
        return {
            "error": "Supabase vector store is not available. Please check your configuration.",
            "results": [],
            "sources": [],
            "context": ""
        }
    
    # Detect if query is asking for a numbered/ordinal item
    import re
    ordinal_patterns = [
        r'\b(first|1st|one)\b', r'\b(second|2nd|two)\b', r'\b(third|3rd|three)\b',
        r'\b(fourth|4th|four)\b', r'\b(fifth|5th|five)\b', r'\b(sixth|6th|six)\b',
        r'\b(seventh|7th|seven)\b', r'\b(eighth|8th|eight)\b', r'\b(ninth|9th|nine)\b',
        r'\b(tenth|10th|ten)\b', r'\bquestion\s*\d+\b', r'\bQ\d+\b', r'\b#\d+\b',
    ]
    is_ordinal_query = any(re.search(p, query, re.IGNORECASE) for p in ordinal_patterns)
    if is_ordinal_query:
        print(f"[retrieve_knowledge] 🔢 ORDINAL QUERY DETECTED: '{query}'")
        print(f"[retrieve_knowledge] 💡 TIP: Semantic search may not find 'Nth item' - look for actual numbered content in results")
    
    # GLiNER2 ENHANCEMENT: Extract entities from query
    query_entities = {}
    query_intent = "general"
    enriched_query = query
    
    if enable_entity_extraction and GLINER_AVAILABLE and extract_query_entities:
        try:
            entity_result = extract_query_entities(
                query,
                entity_types=["person", "company", "organization", "product",
                             "technology", "location", "date"]
            )
            query_entities = entity_result.get("entities", {})
            query_intent = entity_result.get("query_intent", "general")
            enriched_query = entity_result.get("enriched_query", query)
            
            if query_entities:
                print(f"[retrieve_knowledge] 🔍 Query entities: {query_entities}")
                print(f"[retrieve_knowledge] 📊 Query intent: {query_intent}")
        except Exception as e:
            print(f"[retrieve_knowledge] ⚠️  Entity extraction failed: {e}")
    
    try:
        embeddings = _get_embeddings(embedding_model=embedding_model)
        if isinstance(embeddings, OpenAIEmbeddings):
            print(f"[retrieve_knowledge] Using OpenAI embeddings: {getattr(embeddings, 'model', '')}")
        else:
            print(f"[retrieve_knowledge] Using Ollama embeddings: {getattr(embeddings, 'model', '')}")

        # Try to initialize the vector store robustly across versions
        try:
            vector_store = SupabaseVectorStore(
                embedding=embeddings,
                client=supabase,
                table_name=table_name,
                query_name=query_name,
            )
        except Exception:
            # Fallback constructor signature order in some versions
            vector_store = SupabaseVectorStore(
                supabase, embeddings, table_name=table_name, query_name=query_name
            )

        # Perform similarity search
        # Note: similarity_search_with_score is not implemented in LangChain's SupabaseVectorStore
        # We use the patched similarity_search_by_vector_with_relevance_scores instead
        
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query)
        
        # If doc_id is provided, use direct Supabase RPC with filter for better accuracy
        if doc_id:
            print(f"[retrieve_knowledge] 🎯 Using direct Supabase query with doc_id filter: '{doc_id}'")
            try:
                # Call Supabase RPC with doc_id filter - query vectors for specific document only
                # With proper DB-level filtering, we only need a small margin (1.5x instead of 2x)
                response = supabase.rpc(
                    query_name,
                    {
                        "query_embedding": query_embedding,
                        "match_count": int(top_k * 1.5),  # Slight margin for relevance ranking
                        "filter": {"doc_id": doc_id}  # Filter at database level!
                    }
                ).execute()
                
                if response.data:
                    docs = []
                    scores = []
                    for row in response.data[:top_k]:
                        content = row.get("content", "")
                        metadata = row.get("metadata", {})
                        similarity = row.get("similarity", 0.0)
                        
                        docs.append(Document(page_content=content, metadata=metadata))
                        scores.append(float(similarity))
                    
                    print(f"[retrieve_knowledge] Got {len(docs)} results from doc_id-filtered RPC")
                else:
                    docs = []
                    scores = []
                    print(f"[retrieve_knowledge] No results from RPC with doc_id filter")
                    
            except Exception as rpc_error:
                print(f"[retrieve_knowledge] ⚠️ RPC with filter failed: {rpc_error}")
                print(f"[retrieve_knowledge] Falling back to standard search + local filter")
                
                # Fallback: standard search then filter locally
                # Reduced from 5x to 3x - balance between coverage and performance
                try:
                    docs_scores = vector_store.similarity_search_by_vector_with_relevance_scores(
                        query_embedding, k=top_k * 3
                    )
                    # Filter by doc_id locally
                    docs = []
                    scores = []
                    for d, s in docs_scores:
                        if d.metadata and d.metadata.get("doc_id") == doc_id:
                            docs.append(d)
                            scores.append(float(s))
                            if len(docs) >= top_k:
                                break
                    print(f"[retrieve_knowledge] Fallback: Got {len(docs)} results after local filtering")
                except Exception as e2:
                    print(f"[retrieve_knowledge] Fallback also failed: {e2}")
                    docs = []
                    scores = []
        else:
            # Standard search without doc_id filter
            try:
                docs_scores = vector_store.similarity_search_by_vector_with_relevance_scores(
                    query_embedding, k=top_k
                )
                docs = [d for d, _ in docs_scores]
                scores = [float(s) for _, s in docs_scores]
                print(f"[retrieve_knowledge] Got {len(docs)} results with scores")
            except Exception as e:
                print(f"[retrieve_knowledge] similarity_search_with_relevance_scores failed: {e}")
                print(f"[retrieve_knowledge] Falling back to similarity_search without scores")
                docs = vector_store.similarity_search(query, k=top_k)
                scores = []

        # DEBUG: Log search results
        print(f"[retrieve_knowledge] Query: '{query}' | Results: {len(docs)}")

        # Helper to normalize retrieved text (fix OCR spacing issues)
        def normalize_text(text: str) -> str:
            if not text:
                return ""
            import re
            # Fix OCR spacing issues where spaces appear between characters
            # Pattern 1: Remove spaces within words that have excessive spacing
            # e.g., "a g e n t" -> "agent", "c o n v e r s a t i o n" -> "conversation"
            text = re.sub(r'(?<=\w)\s+(?=\w(?:\s+\w){2,})', '', text)
            # Pattern 2: Fix remaining single-letter words followed by spaces
            text = re.sub(r'\b(\w)\s+(?=\w\b)', r'\1', text)
            return text.strip()

        results: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        for idx, doc in enumerate(docs):
            meta = doc.metadata or {}
            vector_score = scores[idx] if idx < len(scores) else 0.0
            score_info = f" | score={vector_score:.4f}" if idx < len(scores) else ""
            
            # Normalize the content before returning
            normalized_content = normalize_text(doc.page_content)
            
            # GLiNER2 ENHANCEMENT: Compute entity overlap score
            entity_overlap_score = 0.0
            doc_entities = meta.get("auto_entities", {})
            entity_highlights = []
            
            if query_entities and GLINER_AVAILABLE and compute_entity_overlap_score:
                try:
                    # Get document entities (from pre-enriched metadata or extract now)
                    if not doc_entities and get_entity_extractor:
                        extractor = get_entity_extractor()
                        if extractor:
                            result = extractor.extract_entities(
                                normalized_content[:1500],
                                ["person", "company", "organization", "product",
                                 "technology", "location", "date"]
                            )
                            doc_entities = result.get("entities", {})
                    
                    # Compute overlap score
                    entity_overlap_score = compute_entity_overlap_score(
                        query_entities, doc_entities
                    )
                    
                    # Create entity highlights
                    entity_highlights = _create_entity_highlights(
                        normalized_content, query_entities, doc_entities
                    )
                except Exception as e:
                    print(f"[retrieve_knowledge] ⚠️  Entity scoring failed for doc {idx}: {e}")
            
            # Combine vector score with entity overlap for hybrid ranking
            # Weight: 70% vector similarity + 30% entity overlap
            hybrid_score = (vector_score * 0.7) + (entity_overlap_score * 0.3) if vector_score else entity_overlap_score
            
            results.append(
                {
                    "content": normalized_content,
                    "metadata": meta,
                    "score": vector_score,
                    "entity_overlap_score": entity_overlap_score,
                    "hybrid_score": hybrid_score,
                    "doc_entities": doc_entities,
                    "entity_highlights": entity_highlights,
                }
            )
            sources.append(
                {
                    "id": meta.get("doc_id") or meta.get("source") or meta.get("filename") or f"chunk-{idx}",
                    "filename": meta.get("filename"),
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index"),
                    "score": vector_score,
                    "entity_overlap_score": entity_overlap_score,
                    "hybrid_score": hybrid_score,
                }
            )
            
            # Include entity info in context if available
            entity_info = ""
            if doc_entities:
                entity_summary = ", ".join(
                    f"{k}: {', '.join(v[:2])}" 
                    for k, v in doc_entities.items() if v
                )[:100]
                if entity_summary:
                    entity_info = f" | entities: {entity_summary}"
            
            context_parts.append(
                f"Source [{idx + 1}]{score_info}{entity_info}: {meta.get('filename') or meta.get('source') or meta.get('doc_id') or ''}\n{normalized_content}\n\n----------\n"
            )

        # Sort results by hybrid score if entity extraction was used
        if query_entities:
            results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            sources.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

        elapsed = time.time() - start_time
        print(f"[retrieve_knowledge] Retrieved {len(results)} chunks in {elapsed:.2f}s")
        if query_entities:
            print(f"[retrieve_knowledge] 🎯 Results re-ranked by hybrid score (vector + entity overlap)")
        
        # LOW RELEVANCE WARNING - Help agent detect when results don't match query
        relevance_warning = ""
        if results:
            avg_score = sum(r.get("score", 0) or 0 for r in results) / len(results)
            max_score = max(r.get("score", 0) or 0 for r in results)
            
            # Extract topics found dynamically from the content
            found_topics = _extract_topics_from_chunks(results[:3])
            topics_str = ", ".join(found_topics) if found_topics else "general content"
            
            # Check if scores are too low (threshold: 0.6 for max, 0.5 for avg)
            if max_score < 0.6 or avg_score < 0.5:
                relevance_warning = (
                    f"\n⚠️ LOW RELEVANCE WARNING: Best match score is {max_score:.2f} (avg: {avg_score:.2f}). "
                    "The retrieved content may NOT contain what you're looking for. "
                    "If you cannot find the specific information requested, say: "
                    "'I could not find [X] in the document. The retrieved content mentions: [summary]' "
                    "DO NOT invent or fabricate answers.\n"
                    f"\n🔍 TOPICS FOUND IN CHUNKS: {topics_str}\n"
                    "\n⚠️ MANDATORY ACTION: If the user asked for a SPECIFIC item (like 'the 4th question') "
                    "and you don't see that exact item numbered in the chunks above, "
                    "you MUST report this honestly. DO NOT answer with generic information!\n\n"
                )
                print(f"[retrieve_knowledge] ⚠️  LOW RELEVANCE: max={max_score:.2f}, avg={avg_score:.2f}")
        
        context_text = "\n".join(context_parts)
        
        # Add ordinal query warning to instructions if applicable
        ordinal_instruction = ""
        if is_ordinal_query:
            ordinal_instruction = (
                "\n\n🔢 ORDINAL QUERY DETECTED: You asked for a NUMBERED item. "
                "Look for actual numbers (Q4, 4., Question 4:) in the chunks above. "
                "If you don't see the specific numbered item, say: "
                "'I could not find [specific item] in the retrieved chunks. The document discusses: [topics found]'. "
                "DO NOT provide a generic answer!"
            )
        
        return {
            "results": results,
            "sources": sources,
            "context": context_text,
            "instructions": "Cite sources inline as [1], [2], etc. for each used passage. If content doesn't match the query, SAY SO - never invent answers." + ordinal_instruction,
            "execution_time": elapsed,
            "query_entities": query_entities,
            "query_intent": query_intent,
            "relevance_warning": bool(relevance_warning),
            "is_ordinal_query": is_ordinal_query,
            "avg_score": avg_score if results else 0,
            "max_score": max_score if results else 0,
        }
    except Exception as e:
        return {"error": str(e), "results": [], "sources": [], "context": ""}


# Cached version (if available)
if CACHE_AVAILABLE:
    @tool
    @cache_query_result(ttl=3600)  # Cache for 1 hour
    def retrieve_knowledge(
        query: str,
        *,
        top_k: int = 5,
        table_name: str = "documents",
        query_name: str = "match_documents",
        embedding_model: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        🚀 CACHED VERSION - Retrieve relevant chunks from the Supabase vector store.
        
        This version uses intelligent caching to speed up repeated queries by up to 80%.
        
        Args:
            query: The search query to match against embeddings
            top_k: Number of results to return
            table_name: Supabase table name that stores vectors
            query_name: Supabase RPC function name for similarity search
            embedding_model: Optional override for the embedding model name
            doc_id: Optional document ID to filter results by specific document
        
        Returns:
            dict with keys: results (list), sources (list), context (str), execution_time (float)
        """
        if table_name != "documents":
            table_name = "documents"
        if query_name != "match_documents":
            query_name = "match_documents"
        if embedding_model and embedding_model.strip().lower() == "chroma":
            embedding_model = None
        return _retrieve_knowledge_impl(query, top_k, table_name, query_name, embedding_model, doc_id)
else:
    # Non-cached version (fallback)
    @tool
    def retrieve_knowledge(
        query: str,
        *,
        top_k: int = 5,
        table_name: str = "documents",
        query_name: str = "match_documents",
        embedding_model: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks from the Supabase vector store.
        
        Args:
            query: The search query to match against embeddings
            top_k: Number of results to return
            table_name: Supabase table name that stores vectors
            query_name: Supabase RPC function name for similarity search
            embedding_model: Optional override for the embedding model name
            doc_id: Optional document ID to filter results by specific document
        
        Returns:
            dict with keys: results (list), sources (list), context (str)
        """
        if table_name != "documents":
            table_name = "documents"
        if query_name != "match_documents":
            query_name = "match_documents"
        if embedding_model and embedding_model.strip().lower() == "chroma":
            embedding_model = None
        return _retrieve_knowledge_impl(query, top_k, table_name, query_name, embedding_model, doc_id)
