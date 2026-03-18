import io
import sys
import os
import asyncio

# Set environment variables FIRST
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
os.environ.setdefault("SMOLAGENTS_VERBOSITY", "0")
os.environ.setdefault("RICH_FORCE_TERMINAL", "false")
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import locale
import sys
from dotenv import load_dotenv
import httpx
import uuid
import logging
import json


_original_dumps = json.dumps

def utf8_dumps(*args, **kwargs):
    kwargs["ensure_ascii"] = False
    return _original_dumps(*args, **kwargs)

json.dumps = utf8_dumps


from huggingsmolagent.agent import app as smolagent_router
from huggingsmolagent.tools.supabase_store import store_pdf
from huggingsmolagent.tools.pdf_loader import parse_pdf
from huggingsmolagent.tools.vector_store import (
    index_documents, 
    retrieve_knowledge, 
    compute_file_hash, 
    check_existing_document
)
from huggingsmolagent.tools.summarizer import summarize
from huggingsmolagent.tools.scraper import web_search_ctx
from pydantic import BaseModel

from api.agent_lc import router as agent_lc_router

# NEW: Deep Agents router (migration from smolagents)
try:
    from deepagents.router import router as deepagent_router
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False
    print("[startup] ⚠️  Deep Agents not available - run: pip install deepagents")

load_dotenv() 
app = FastAPI()
print("[startup] FastAPI app initialized")

app.include_router(agent_lc_router)

# Mount Deep Agents router (NEW - parallel to smolagents)
if DEEPAGENTS_AVAILABLE:
    app.include_router(deepagent_router)
    print("[startup] ✅ Deep Agents router mounted at /agent-deep")
else:
    print("[startup] ⚠️  Deep Agents router not available")

# Reduce extremely verbose DEBUG logs (httpcore/openai/etc.) that make responses look "very long".
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
for noisy in (
    "openai",
    "httpx",
    "httpcore",
    "rquest",
    "primp",
    "cookie_store",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# CORS for frontend imports/uploads
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# Add health check before mounting
@app.get("/health")
async def health_check():
    print("[health] /health called")
    return {"status": "ok"}


# Cache stats endpoint
@app.get("/cache/stats")
async def cache_stats():
    """Returns query cache statistics"""
    try:
        from huggingsmolagent.tools.query_cache import get_cache_stats
        stats = get_cache_stats()
        return stats
    except ImportError:
        return {"error": "Cache not available", "enabled": False}


@app.post("/cache/clear")
async def clear_cache():
    """Clears the query cache"""
    try:
        from huggingsmolagent.tools.query_cache import clear_cache
        clear_cache()
        return {"status": "cache cleared", "success": True}
    except ImportError:
        return {"error": "Cache not available", "success": False}


# Knowledge Graph endpoints
@app.get("/kg/stats")
async def kg_stats():
    """Returns Knowledge Graph statistics"""
    try:
        from tools.kg_tool import stats
        return stats()
    except Exception as e:
        return {"error": str(e), "node_count": 0, "edge_count": 0}


@app.get("/kg/graph")
async def kg_graph(
    entity_type: str = None,
    max_nodes: int = 100,
    min_frequency: int = 1
):
    """
    Returns Knowledge Graph in node-link format for visualization.
    
    Query params:
    - entity_type: Filter by entity type (DRUG, DISEASE, GENE, etc.)
    - max_nodes: Maximum number of nodes to return (default: 100)
    - min_frequency: Minimum frequency for nodes (default: 1)
    """
    try:
        from tools.kg_tool import get_graph
        import networkx as nx
        
        G = get_graph()
        
        # Filter by entity type if specified
        if entity_type:
            nodes_to_keep = [
                n for n, d in G.nodes(data=True)
                if d.get('entity_type', '').upper() == entity_type.upper()
            ]
            G = G.subgraph(nodes_to_keep).copy()
        
        # Filter by frequency
        if min_frequency > 1:
            nodes_to_keep = [
                n for n, d in G.nodes(data=True)
                if d.get('frequency', 0) >= min_frequency
            ]
            G = G.subgraph(nodes_to_keep).copy()
        
        # Limit number of nodes (take top by frequency)
        if G.number_of_nodes() > max_nodes:
            nodes_sorted = sorted(
                G.nodes(data=True),
                key=lambda x: x[1].get('frequency', 0),
                reverse=True
            )
            top_nodes = [n[0] for n in nodes_sorted[:max_nodes]]
            G = G.subgraph(top_nodes).copy()
        
        # Convert to node-link format
        nodes = []
        for node_id, data in G.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get('label', node_id),
                "type": data.get('entity_type', 'UNKNOWN'),
                "frequency": data.get('frequency', 1),
                "degree": G.degree(node_id)
            })
        
        links = []
        for source, target, data in G.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "weight": data.get('weight', 1),
                "relation_type": data.get('relation_type', 'co_occurrence')
            })
        
        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "filtered": entity_type is not None or min_frequency > 1
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}


@app.get("/kg/node/{node_id}")
async def kg_node_details(node_id: str):
    """Get details for a specific node and its neighbors"""
    try:
        from tools.kg_tool import query_node
        return query_node(node_id)
    except Exception as e:
        return {"error": str(e)}


@app.get("/kg/top-nodes")
async def kg_top_nodes(n: int = 20, sort_by: str = "frequency"):
    """Get top N nodes by frequency or degree"""
    try:
        from tools.kg_tool import query_top_nodes
        return {"nodes": query_top_nodes(n=n, sort_by=sort_by)}
    except Exception as e:
        return {"error": str(e), "nodes": []}


# ============================================================================
# CONVERSATION MEMORY ENDPOINTS
# ============================================================================

@app.get("/conversations")
async def list_conversations():
    """List all active conversations with metadata."""
    try:
        from deepagents.memory import ConversationMemoryManager
        conversations = ConversationMemoryManager.list_conversations()
        stats = ConversationMemoryManager.get_stats()
        return {
            "conversations": conversations,
            "stats": stats
        }
    except Exception as e:
        return {"error": str(e), "conversations": []}


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str, limit: int = 50):
    """Get conversation history for a specific conversation."""
    try:
        from deepagents.memory import ConversationMemoryManager
        history = ConversationMemoryManager.get_history(conversation_id, limit=limit)
        return {
            "conversation_id": conversation_id,
            "message_count": len(history),
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content
                }
                for msg in history
            ]
        }
    except Exception as e:
        return {"error": str(e), "messages": []}


@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history for a specific conversation."""
    try:
        from deepagents.memory import ConversationMemoryManager
        ConversationMemoryManager.clear_conversation(conversation_id)
        return {"status": "success", "message": f"Conversation {conversation_id} cleared"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.post("/conversations/cleanup")
async def cleanup_conversations(keep_last_n: int = 100):
    """Cleanup old conversations, keeping only the N most active ones."""
    try:
        from deepagents.memory import ConversationMemoryManager
        ConversationMemoryManager.cleanup_old_conversations(keep_last_n=keep_last_n)
        stats = ConversationMemoryManager.get_stats()
        return {
            "status": "success",
            "message": f"Kept {keep_last_n} most active conversations",
            "stats": stats
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


"""
Mount the smolagent streaming app at /agent (POST /agent/) to keep SSE streaming.
We implement a unified /ask below that returns JSON and orchestrates upload/summarize/RAG/scrape.
"""
app.mount("/agent", smolagent_router)


@app.post("/ask")
async def ask(request: Request):
    """
    Unified Ask endpoint:
    - If files are attached: upload+index them first, then pass query to smolagent
    - If no files: directly pass query to smolagent
    
    smolagent decides which tools to use (RAG, summarize, scrape) based on the query context.
    Returns JSON response from smolagent.
    """
    content_type = request.headers.get("content-type", "")
    is_multipart = "multipart/form-data" in content_type
    is_json = "application/json" in content_type
    print(f"[ask] called content_type={content_type} is_multipart={is_multipart} is_json={is_json}")

    try:
        query = ""
        conversation_id = None
        uploaded_context = []  # Track uploaded files for context
        
        # STEP 1: Handle file uploads if present
        if is_multipart:
            form = await request.form()
            query = (form.get("query") or "").strip()
            conversation_id = (form.get("conversation_id") or "").strip() or None
            files = form.getlist("files")
            print(f"[ask] multipart received query='{query[:80] if query else ''}' conversation_id={conversation_id} files_count={len(files) if files else 0}")

            if files:
                # Process uploads: store, parse, index
                print(f"[ask] processing {len(files)} file(s)")
                for f in files:
                    print(f"[ask] processing file name={getattr(f, 'filename', None)}")
                    
                    # Read file content for hashing
                    file_content = await f.read()
                    await f.seek(0)  # Reset file pointer for subsequent reads
                    
                    # Compute file hash for deduplication
                    file_hash = compute_file_hash(file_content)
                    print(f"[ask] computed file_hash={file_hash[:16]}...")
                    
                    # Check if this file already exists
                    existing = check_existing_document(file_hash)
                    
                    if existing:
                        print(f"[ask] ⚠️  File already indexed! doc_id={existing['doc_id']}, chunks={existing['chunk_count']}")
                        uploaded_context.append({
                            "filename": f.filename,
                            "doc_id": existing["doc_id"],
                            "chunks": existing["chunk_count"],
                            "reused": True
                        })
                        continue
                    
                    # New file - proceed with storage and indexing
                    file_url = store_pdf(f)
                    print(f"[ask] stored file_url={file_url}")
                    documents = parse_pdf(f)
                    print(f"[ask] parsed documents_count={len(documents) if isinstance(documents, list) else 'n/a'}")
                    doc_id = str(uuid.uuid4())
                    stored = index_documents(
                        documents,
                        base_metadata={
                            "source": file_url, 
                            "filename": f.filename, 
                            "doc_id": doc_id,
                            "file_hash": file_hash
                        },
                    )
                    print(f"[ask] indexed doc_id={doc_id} stored={stored}")
                    uploaded_context.append({
                        "filename": f.filename,
                        "doc_id": doc_id,
                        "chunks": stored,
                        "reused": False
                    })
                print(f"[ask] upload complete. {len(uploaded_context)} file(s) processed")
        else:
            # JSON body
            body = await request.json() if is_json else {}
            query = (body.get("query") if isinstance(body, dict) else None) or ""
            conversation_id = (body.get("conversation_id") if isinstance(body, dict) else None) or None
            print(f"[ask] json body parsed query='{query[:120] if query else ''}' conversation_id={conversation_id}")

        # If files were uploaded but no query, return upload success
        if not query and uploaded_context:
            return JSONResponse({
                "answer": f"✅ {len(uploaded_context)} file(s) uploaded and indexed successfully. You can now ask questions about them.",
                "uploaded_files": uploaded_context,
                "status": "success"
            }, status_code=200)
        
        if not query:
            return JSONResponse({"answer": "Please provide a query."}, status_code=400)

        # STEP 2: Delegate to agent
        print(f"[ask] delegating to agent with query='{query[:100]}'")
        print(f"[ask] uploaded_files_context={len(uploaded_context)} files")

        if not uploaded_context:
            # Use Deep Agent for queries without uploaded files
            from deepagents.agents.main_agent import create_medAssist_agent
            from fastapi.responses import StreamingResponse

            async def _deep_agent_events():
                import queue
                import threading
                
                # Queue pour recevoir les événements de streaming
                event_queue = queue.Queue()
                final_answer = None
                
                def stream_callback(event):
                    """Callback appelé par l'agent pour streamer les événements"""
                    event_queue.put(event)
                
                # Fonction pour exécuter l'agent dans un thread
                def run_agent():
                    nonlocal final_answer
                    from deepagents.agents.main_agent import create_medAssist_agent
                    
                    # Créer l'agent avec le callback de streaming
                    agent = create_medAssist_agent()
                    if hasattr(agent, 'stream_callback'):
                        agent.stream_callback = stream_callback
                    if conversation_id and hasattr(agent, 'conversation_id'):
                        agent.conversation_id = conversation_id
                    
                    # Exécuter l'agent
                    result = agent.invoke({"input": query})
                    final_answer = result.get("output", str(result)) if isinstance(result, dict) else str(result)
                    event_queue.put({"type": "done"})
                
                # Démarrer l'agent dans un thread
                agent_thread = threading.Thread(target=run_agent)
                agent_thread.start()
                
                # Streamer les événements au fur et à mesure
                yield json.dumps({"step": "� Starting Deep Agent..."}, ensure_ascii=False) + "\n"
                
                while True:
                    try:
                        event = event_queue.get(timeout=0.1)
                        
                        if event["type"] == "done":
                            break
                        elif event["type"] == "thought":
                            yield json.dumps({"step": event["content"]}, ensure_ascii=False) + "\n"
                        elif event["type"] == "action":
                            yield json.dumps({"step": event["content"]}, ensure_ascii=False) + "\n"
                        elif event["type"] == "observation":
                            step_data = {"step": event["content"]}
                            if "preview" in event:
                                step_data["preview"] = event["preview"]
                            yield json.dumps(step_data, ensure_ascii=False) + "\n"
                        elif event["type"] == "error":
                            yield json.dumps({"step": event["content"]}, ensure_ascii=False) + "\n"
                        elif event["type"] == "answer":
                            # L'agent a la réponse finale
                            pass
                            
                    except queue.Empty:
                        # Pas d'événement, continuer à attendre
                        await asyncio.sleep(0.05)
                        continue
                
                # Attendre que le thread se termine
                agent_thread.join()
                
                # Envoyer la réponse finale
                if final_answer:
                    yield json.dumps({"response": final_answer, "canHandle": True}, ensure_ascii=False) + "\n"
                else:
                    yield json.dumps({"response": "No response generated", "canHandle": False}, ensure_ascii=False) + "\n"

            return StreamingResponse(
                _deep_agent_events(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        
        # Build enhanced context message for Deep Agent
        context_msg = ""
        doc_id_param = None
        
        if uploaded_context:
            # Build the context message with explicit instructions for the agent
            filenames = [ctx["filename"] for ctx in uploaded_context]
            doc_ids = [ctx["doc_id"] for ctx in uploaded_context]
            total_chunks = sum(ctx["chunks"] for ctx in uploaded_context)
            
            # For single file upload, provide the doc_id
            if len(doc_ids) == 1:
                doc_id_param = doc_ids[0]
            
            context_msg = f"""

[CONTEXT: User uploaded {len(uploaded_context)} file(s): {', '.join(filenames)}]
[Total chunks indexed: {total_chunks}]

The uploaded document(s) are now in the vector database. Use the retrieve_knowledge tool to access the content.
{f'Document ID: {doc_id_param}' if doc_id_param else ''}
"""
        
        # Use Deep Agent with uploaded files context
        from deepagents.agents.main_agent import create_medAssist_agent
        from fastapi.responses import StreamingResponse
        
        agent_query = query + context_msg if context_msg else query
        print(f"[ask] calling Deep Agent with query (length={len(agent_query)})")
        print(f"[ask] query preview: '{agent_query[:200]}...'")
        
        async def _deep_agent_with_files():
            import queue
            import threading
            
            # Queue pour recevoir les événements de streaming
            event_queue = queue.Queue()
            final_answer = None
            
            def stream_callback(event):
                """Callback appelé par l'agent pour streamer les événements"""
                event_queue.put(event)
            
            # Fonction pour exécuter l'agent dans un thread
            def run_agent():
                nonlocal final_answer
                from deepagents.agents.main_agent import create_medAssist_agent
                
                # Créer l'agent avec le callback de streaming
                agent = create_medAssist_agent()
                if hasattr(agent, 'stream_callback'):
                    agent.stream_callback = stream_callback
                if conversation_id and hasattr(agent, 'conversation_id'):
                    agent.conversation_id = conversation_id
                
                # Exécuter l'agent
                result = agent.invoke({"input": agent_query})
                final_answer = result.get("output", str(result)) if isinstance(result, dict) else str(result)
                event_queue.put({"type": "done"})
            
            # Démarrer l'agent dans un thread
            agent_thread = threading.Thread(target=run_agent)
            agent_thread.start()
            
            # Streamer les événements au fur et à mesure
            yield json.dumps({"step": "🚀 Starting Deep Agent with uploaded files..."}, ensure_ascii=False) + "\n"
            
            while True:
                try:
                    event = event_queue.get(timeout=0.1)
                    
                    if event["type"] == "done":
                        break
                    elif event["type"] == "thought":
                        yield json.dumps({"step": event["content"]}, ensure_ascii=False) + "\n"
                    elif event["type"] == "action":
                        yield json.dumps({"step": event["content"]}, ensure_ascii=False) + "\n"
                    elif event["type"] == "observation":
                        step_data = {"step": event["content"]}
                        if "preview" in event:
                            step_data["preview"] = event["preview"]
                        yield json.dumps(step_data, ensure_ascii=False) + "\n"
                    elif event["type"] == "error":
                        yield json.dumps({"step": event["content"]}, ensure_ascii=False) + "\n"
                    elif event["type"] == "answer":
                        # L'agent a la réponse finale
                        pass
                        
                except queue.Empty:
                    # Pas d'événement, continuer à attendre
                    await asyncio.sleep(0.05)
                    continue
            
            # Attendre que le thread se termine
            agent_thread.join()
            
            # Envoyer la réponse finale
            if final_answer:
                yield json.dumps({"response": final_answer, "canHandle": True}, ensure_ascii=False) + "\n"
            else:
                yield json.dumps({"response": "No response generated", "canHandle": False}, ensure_ascii=False) + "\n"
        
        # Return streaming response
        return StreamingResponse(
            _deep_agent_with_files(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print("[ask] error:", e)
        import traceback
        traceback.print_exc()
        return JSONResponse({"answer": "Error processing request.", "error": str(e)}, status_code=500)





# =============================================================================
# PubMed Search endpoint
# =============================================================================

class PubMedSearchRequest(BaseModel):
    query: str
    max_results: int = 20
    start: int = 0
    sort: str = "relevance"
    mindate: str = ""
    maxdate: str = ""
    fetch_details: bool = True
    publication_types: list = None
    journals: list = None
    language: str = ""
    species: list = None

@app.post("/pubmed/search")
async def pubmed_search(request: PubMedSearchRequest):
    """Search PubMed with advanced filters."""
    try:
        from tools.pubmed_tool import PubMedSearchEngine, ncbi_fetch, ncbi_parse_efetch_xml
        engine = PubMedSearchEngine()
        search_result = engine.search(
            query=request.query,
            max_results=request.max_results,
            start=request.start,
            sort=request.sort,
            mindate=request.mindate,
            maxdate=request.maxdate,
            publication_types=request.publication_types,
            journals=request.journals,
            language=request.language if request.language else None,
            species=request.species,
            use_cache=True,
        )
        result = search_result.get("esearchresult", {})
        pmids = result.get("idlist", []) or []
        total = int(result.get("count", 0) or 0)

        articles = []
        if request.fetch_details and pmids:
            articles = engine.fetch_articles(pmids)

        return {
            "provider": "ncbi",
            "query": request.query,
            "total": total,
            "start": request.start,
            "max_results": request.max_results,
            "pmids": pmids,
            "articles": articles,
        }
    except Exception as e:
        print(f"[pubmed/search] error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e), "pmids": [], "articles": [], "total": 0}, status_code=500)


# =============================================================================
# NER Extract endpoint
# =============================================================================

class NerExtractRequest(BaseModel):
    text: str
    entity_types: list = None
    provider: str = None

@app.post("/ner/extract")
async def ner_extract(request: NerExtractRequest):
    """Extract medical entities from text."""
    try:
        from ner.router import extract_from_text
        result = extract_from_text(
            request.text,
            entity_types=request.entity_types,
            provider=request.provider,
        )
        return result.to_dict()
    except Exception as e:
        print(f"[ner/extract] error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e), "entities": {}}, status_code=500)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print("[upload] /upload called")
    print ("filename", file.filename, "content_type" ,file.content_type)

    # Read file content for hashing
    file_content = await file.read()
    await file.seek(0)  # Reset file pointer for subsequent reads
    
    # Compute file hash for deduplication
    file_hash = compute_file_hash(file_content)
    print(f"[upload] computed file_hash={file_hash[:16]}...")
    
    # Check if this file already exists
    existing = check_existing_document(file_hash)
    
    if existing:
        print(f"[upload] ⚠️  File already indexed! doc_id={existing['doc_id']}, chunks={existing['chunk_count']}")
        return {
            "file_url": existing.get("source", ""),
            "doc_id": existing["doc_id"],
            "chunks_indexed": existing["chunk_count"],
            "summary": f"File '{file.filename}' was already indexed. Reusing existing document.",
            "reused": True
        }

    # New file - proceed with storage and indexing
    # 1. Save in supabase storage
    file_url = store_pdf(file)
    print("[upload] stored file_url", file_url)
    
    # 2. Text Extraction
    documents = parse_pdf(file)
    try:
        print("[upload] documents_count", len(documents))
    except Exception:
        print("[upload] documents_count unavailable")

    # 3. Vector Supabase Indexation
    doc_id = str(uuid.uuid4())
    print("[upload] doc_id", doc_id)
    stored = index_documents(
        documents,
        base_metadata={
            "source": file_url, 
            "filename": file.filename, 
            "doc_id": doc_id,
            "file_hash": file_hash
        },
    )
    print("[upload] indexed stored=", stored)
    
    # 4. Summarization
    summary = summarize(documents)
    print("[upload] summary generated length=", (len(summary) if isinstance(summary, str) else "n/a"))
    # 5. notify n8n webhook 
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    if webhook_url:
        print("webhook_url", webhook_url)
        try:
            payload = {
                "doc_id": doc_id,
                "file_url": file_url,
                "filename": file.filename,
                "chunks_indexed": stored,
                "summary": summary,
            }
            print("payload", payload)
            
            # non-blocking fire-and-forget
            verify = os.getenv("N8N_WEBHOOK_VERIFY", "true").lower() != "false"
            with httpx.Client(timeout=5.0, verify=verify) as client:
                client.post(webhook_url, json=payload)
        except Exception as e:
            print("n8n webhook notify failed:", e)

    return {"file_url": file_url, "doc_id": doc_id, "chunks_indexed": stored, "summary": summary}


 
if __name__ == "__main__":
    import uvicorn
    print("[startup] Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)