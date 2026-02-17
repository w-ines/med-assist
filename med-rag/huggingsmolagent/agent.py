import io
import sys
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import locale
import json
# Force UTF-8 encoding globally (must happen before importing smolagents/rich)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
os.environ.setdefault("SMOLAGENTS_VERBOSITY", "0")
os.environ.setdefault("RICH_FORCE_TERMINAL", "false")

# Force UTF-8 streams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_original_dumps = json.dumps

def utf8_dumps(*args, **kwargs):
    kwargs["ensure_ascii"] = False
    return _original_dumps(*args, **kwargs)

json.dumps = utf8_dumps



# Debug encoding (helps diagnose persistent ASCII issues)
try:
    _stdout_enc = getattr(sys.stdout, "encoding", None)
    _stderr_enc = getattr(sys.stderr, "encoding", None)
except Exception:
    _stdout_enc = None
    _stderr_enc = None

try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except Exception:
        pass


def _build_ascii_default_headers() -> dict[str, str]:
    """Build ASCII-safe default headers for OpenAI/OpenRouter clients.

    httpx header values are normalized with ASCII by default; any non-ASCII in headers will crash.
    """
    headers: dict[str, str] = {}

    referer = (
        os.getenv("OPENROUTER_HTTP_REFERER")
        or os.getenv("OPENROUTER_REFERER")
        or os.getenv("OPENROUTER_SITE_URL")
        or os.getenv("HTTP_REFERER")
    )
    if referer:
        headers["HTTP-Referer"] = _ascii_only(referer)

    title = (
        os.getenv("OPENROUTER_X_TITLE")
        or os.getenv("OPENROUTER_APP_NAME")
        or os.getenv("OPENROUTER_APP_TITLE")
        or os.getenv("X_TITLE")
    )
    if title:
        headers["X-Title"] = _ascii_only(title)

    return headers

import asyncio
import time
import logging
import traceback
from typing import Optional, Dict, Any, List
from smolagents import CodeAgent, Tool, OpenAIServerModel
from smolagents.utils import AgentGenerationError

from huggingsmolagent.agent_factory import (
    AGENT_INSTRUCTIONS,
    create_agent,
    _ascii_only,
)

from huggingsmolagent.response_formatting import extract_final_answer


def _patch_httpx_headers_ascii() -> None:
    """Prevent httpx from crashing on non-ASCII header values.

    httpx normalizes header values using ASCII by default.
    Some upstream clients (OpenAI/OpenRouter wrappers) may inject metadata headers
    from env/config containing accents. This patch sanitizes header values to ASCII.
    """
    enabled = os.getenv("HTTPX_FORCE_ASCII_HEADERS", "true").lower() == "true"
    if not enabled:
        return

    try:
        import httpx

        if getattr(httpx, "_ASCII_HEADERS_PATCHED", False):
            return

        _OriginalHeaders = httpx.Headers

        class ASCIIHeaders(_OriginalHeaders):
            def __init__(self, headers=None, encoding=None):
                if isinstance(headers, dict):
                    safe_headers = {}
                    for k, v in headers.items():
                        if isinstance(v, str) and any(ord(ch) > 127 for ch in v):
                            safe_headers[k] = _ascii_only(v)
                        else:
                            safe_headers[k] = v
                    headers = safe_headers
                super().__init__(headers, encoding=encoding)

        httpx.Headers = ASCIIHeaders
        httpx._ASCII_HEADERS_PATCHED = True
        logger.info("Applied httpx.Headers ASCII sanitization patch")
    except Exception:
        pass


_patch_httpx_headers_ascii()


def _get_openai_max_tokens() -> int:
    try:
        value = int(os.getenv("OPENAI_MAX_TOKENS") or os.getenv("OPEN_AI_MAX_TOKENS") or "2048")
    except Exception:
        value = 2048
    if value < 1:
        return 2048
    if value > 8192:
        return 8192
    return value


def _build_openai_server_model() -> OpenAIServerModel:
    max_tokens = _get_openai_max_tokens()
    base_kwargs = {
        "model_id": os.getenv("OPEN_AI_MODEL"),
        "api_base": os.getenv("BASE_URL"),
        "api_key": os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY"),
        "organization": _ascii_only(os.getenv("OPENAI_ORGANIZATION") or "") or None,
        "project": _ascii_only(os.getenv("OPENAI_PROJECT") or "") or None,
        "client_kwargs": {"default_headers": _build_ascii_default_headers()},    }

    # Some smolagents versions may not accept max_tokens; keep runtime safe.
    try:
        return OpenAIServerModel(**base_kwargs, max_tokens=max_tokens)
    except TypeError:
        return OpenAIServerModel(**base_kwargs)

for k, v in os.environ.items():
    if isinstance(v, str) and any(ord(ch) > 127 for ch in v):
        os.environ[k] = _ascii_only(v)

def _sanitize_http_header_env() -> None:
    """Sanitize env vars that may be forwarded as HTTP headers by OpenAI/OpenRouter clients."""
    # OpenRouter commonly recommends these headers.
    # Various SDKs / wrappers may read them from env and pass them to HTTP clients.
    candidate_keys = {
        "HTTP_REFERER",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_REFERER",
        "OPENROUTER_SITE_URL",
        "X_TITLE",
        "OPENROUTER_X_TITLE",
        "OPENROUTER_APP_NAME",
        "OPENROUTER_APP_TITLE",
        # OpenAI SDK commonly used metadata headers
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
    }

    for k in list(os.environ.keys()):
        if k in candidate_keys or k.startswith(("OPENROUTER_", "OPENAI_", "HTTP_", "X_")):
            v = os.getenv(k)
            if not v:
                continue
            ascii_v = _ascii_only(v)
            if ascii_v != v:
                os.environ[k] = ascii_v

def _safe_utf8_str(value: Any) -> str:
    """
    Safely convert any value to a UTF-8 encoded string, handling Unicode characters.
    This prevents 'ascii' codec errors when processing model outputs.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        # Ensure the string is properly encoded as UTF-8
        try:
            # If it's already a valid string, return it
            return value
        except Exception:
            # If there's an encoding issue, try to fix it
            return value.encode('utf-8', errors='replace').decode('utf-8')
    try:
        # Convert to string first, then ensure UTF-8 encoding
        str_value = str(value)
        # Handle any encoding issues by replacing problematic characters
        return str_value.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeEncodeError:
        # If encoding fails, use error replacement
        return str(value).encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        # Fallback: return a safe representation
        return repr(value).encode('utf-8', errors='replace').decode('utf-8')


def _log_non_ascii_env() -> None:
    """Log which env vars still contain non-ASCII characters (values masked for secrets)."""
    try:
        suspect = []
        for k, v in os.environ.items():
            if not isinstance(v, str):
                continue
            if any(ord(ch) > 127 for ch in v):
                is_secret = any(tok in k.upper() for tok in ["KEY", "TOKEN", "SECRET", "PASSWORD"])
                if is_secret:
                    suspect.append(f"{k}=<redacted>")
                else:
                    # Keep a short preview without leaking too much
                    preview = _ascii_only(v)[:80]
                    suspect.append(f"{k}~='{preview}'")
        if suspect:
            logger.warning("Non-ASCII env values detected (may break httpx headers): %s", ", ".join(suspect[:30]))
    except Exception:
        pass

# Monkey patch to fix encoding issues in smolagents
import builtins
original_print = builtins.print

def utf8_print(*args, **kwargs):
    try:
        # Convert all args to string with UTF-8 encoding
        utf8_args = []
        for arg in args:
            if isinstance(arg, str):
                utf8_args.append(arg.encode('utf-8', 'ignore').decode('utf-8'))
            else:
                utf8_args.append(str(arg).encode('utf-8', 'ignore').decode('utf-8'))
        original_print(*utf8_args, **kwargs)
    except Exception:
        original_print(*args, **kwargs)

builtins.print = utf8_print
try:
    from .streaming_handler import streaming_manager
except Exception:
    streaming_manager = None
from huggingsmolagent.tools.scraper import webscraper, web_search_ctx, visit_webpage, search_news
from huggingsmolagent.tools.vector_store import retrieve_knowledge
from huggingsmolagent.tools.weather import get_weather, get_weather_simple
import os.path
import logging
import os
import time
import yaml 
import re
import queue
import threading
from typing import AsyncGenerator
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smolagents.agent")
logger.info(f"[encoding] PYTHONUTF8={os.getenv('PYTHONUTF8')} PYTHONIOENCODING={os.getenv('PYTHONIOENCODING')} LANG={os.getenv('LANG')} LC_ALL={os.getenv('LC_ALL')}")
logger.info(f"[encoding] sys.stdout.encoding={_stdout_enc} sys.stderr.encoding={_stderr_enc}")

# ============================================================================
# SHARED AGENT CONFIGURATION - Centralized to avoid duplication
# ============================================================================

app = FastAPI()

# Custom Log Handler to capture steps
class ListLogHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_records = []

    def emit(self, record):
        self.log_records.append(self.format(record))

class ComplexRequest(BaseModel):
    chatSettings: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    selectedTools: Optional[List[Dict[str, Any]]] = None
    toolsQuery: Optional[str] = None
    conversationId: Optional[str] = None

# Keep the original model for backward compatibility
class QueryRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    response: str
    steps: Optional[List[str]] = None
    paper: Optional[Dict[str, Any]] = None
    canHandle: bool = False


# =============================================================================
# ENHANCED CONVERSATION MEMORY SYSTEM
# =============================================================================
# Modern agent memory with: entities, facts, preferences, summaries, tool cache

from datetime import datetime
from collections import defaultdict

# In-memory per-conversation memory store
conversation_memory: Dict[str, Dict[str, Any]] = {}

# Tool results cache (avoid re-calling expensive APIs)
tool_results_cache: Dict[str, Dict[str, Any]] = {}

def update_conversation_memory(conversation_id: Optional[str], query: str, final_text: str, 
                                tool_calls: Optional[List[str]] = None):
    """
    Enhanced memory extraction with multiple modern techniques:
    1. Entity Memory - Named entities (people, places, orgs, products)
    2. Fact Memory - Key-value facts from the response
    3. Q&A Memory - Query-answer pairs for retrieval
    4. Reflection Memory - What tools worked/failed
    5. User Preference Detection - Implicit preferences
    """
    if not conversation_id or not final_text:
        return
    
    mem = conversation_memory.setdefault(conversation_id, {
        "entities": [],      # Named entities mentioned
        "facts": [],         # Extracted facts (subject-predicate-object)
        "qa_pairs": [],      # Query-answer pairs for retrieval
        "preferences": {},   # Detected user preferences
        "summaries": [],     # Conversation summaries
        "successful_tools": defaultdict(int),  # Tools that worked well
        "last_updated": None
    })
    
    # =========================================================================
    # 1. ENTITY EXTRACTION - People, places, organizations, products, dates
    # =========================================================================
    entity_patterns = [
        # Products/Technologies
        (r"\b(React|Vue|Angular|Next\.?js|Python|JavaScript|TypeScript|Node\.?js|Django|Flask|FastAPI|TensorFlow|PyTorch|Kubernetes|Docker|AWS|Azure|GCP)\b", "technology"),
        # Companies/Organizations
        (r"\b(Google|Microsoft|Apple|Meta|Amazon|OpenAI|Anthropic|Netflix|Spotify|Tesla|NVIDIA|Intel|AMD)\b", "organization"),
        # Dates and times
        (r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "date"),
        (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b", "date"),
        # Numbers with units (prices, stats, metrics)
        (r"\b(\d+(?:\.\d+)?\s*(?:million|billion|thousand|k|M|B|%|USD|EUR|\$|€))\b", "metric"),
        # URLs
        (r"(https?://[^\s]+)", "url"),
        # Email patterns (for contact info)
        (r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email"),
    ]
    
    entities = mem.get("entities", [])
    existing_entities = {(e.get("value"), e.get("type")) for e in entities}
    
    for pattern, entity_type in entity_patterns:
        for match in re.finditer(pattern, final_text, re.IGNORECASE):
            value = match.group(1).strip()
            if (value, entity_type) not in existing_entities:
                entities.append({
                    "type": entity_type,
                    "value": value,
                    "context": final_text[max(0, match.start()-50):match.end()+50],
                    "timestamp": datetime.now().isoformat()
                })
                existing_entities.add((value, entity_type))
    
    # =========================================================================
    # 2. FACT EXTRACTION - Subject-Predicate-Object triples
    # =========================================================================
    fact_patterns = [
        # "X is Y" patterns
        (r"([A-Z][a-zA-Z0-9]+)\s+is\s+(?:a|an|the)?\s*([^.!?]{10,100})", "definition"),
        # "X has Y" patterns  
        (r"([A-Z][a-zA-Z0-9]+)\s+has\s+([^.!?]{5,100})", "attribute"),
        # "X costs Y" patterns
        (r"([A-Z][a-zA-Z0-9\s]+)\s+(?:costs?|priced? at|is\s+\$)\s*([\d.,]+\s*(?:USD|EUR|\$|€|k|M)?)", "price"),
        # "X was founded in Y" patterns
        (r"([A-Z][a-zA-Z0-9]+)\s+was\s+(?:founded|created|started|launched)\s+(?:in\s+)?(\d{4})", "founding_date"),
        # Statistics patterns
        (r"([A-Z][a-zA-Z0-9\s]+?)\s*(?:has|have|with|reached)\s*([\d.,]+[kKmMbB]?)\s+(users?|downloads?|stars?|followers?|subscribers?)", "statistic"),
    ]
    
    facts = mem.get("facts", [])
    existing_facts = {(f.get("subject"), f.get("type")) for f in facts}
    
    for pattern, fact_type in fact_patterns:
        for match in re.finditer(pattern, final_text):
            try:
                subject = match.group(1).strip()
                value = match.group(2).strip() if len(match.groups()) >= 2 else ""
                # Add metric type if present
                metric = match.group(3).strip() if len(match.groups()) >= 3 else ""
                
                if subject and value and (subject, fact_type) not in existing_facts:
                    facts.append({
                        "type": fact_type,
                        "subject": subject,
                        "value": f"{value} {metric}".strip(),
                        "timestamp": datetime.now().isoformat()
                    })
                    existing_facts.add((subject, fact_type))
            except Exception:
                continue
    
    # =========================================================================
    # 3. Q&A MEMORY - Store query-answer pairs for future retrieval
    # =========================================================================
    if query and final_text and len(final_text) > 50:
        qa_pairs = mem.get("qa_pairs", [])
        # Create summary of answer (first 200 chars + key points)
        answer_summary = final_text[:300].strip()
        if len(final_text) > 300:
            answer_summary += "..."
        
        qa_pairs.append({
            "query": query[:200],
            "answer_summary": answer_summary,
            "full_answer_length": len(final_text),
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 20 Q&A pairs
        mem["qa_pairs"] = qa_pairs[-20:]
    
    # =========================================================================
    # 4. USER PREFERENCE DETECTION - Learn from implicit signals
    # =========================================================================
    preferences = mem.get("preferences", {})
    
    # Detect language preference from query
    french_indicators = ["comment", "pourquoi", "quoi", "quel", "est-ce", "qu'est"]
    english_indicators = ["what", "how", "why", "when", "where", "which"]
    
    query_lower = query.lower() if query else ""
    if any(ind in query_lower for ind in french_indicators):
        preferences["language"] = "french"
    elif any(ind in query_lower for ind in english_indicators):
        preferences["language"] = "english"
    
    # Detect topic interests from entities
    tech_count = sum(1 for e in entities if e.get("type") == "technology")
    if tech_count > 2:
        preferences["interest_area"] = "technology"
    
    # Detect detail preference (short vs detailed answers)
    if "brief" in query_lower or "short" in query_lower or "résumé" in query_lower:
        preferences["detail_level"] = "concise"
    elif "detail" in query_lower or "explain" in query_lower or "expliquer" in query_lower:
        preferences["detail_level"] = "detailed"
    
    mem["preferences"] = preferences
    
    # =========================================================================
    # 5. TOOL REFLECTION - Track which tools were useful
    # =========================================================================
    if tool_calls:
        successful_tools = mem.get("successful_tools", defaultdict(int))
        for tool in tool_calls:
            successful_tools[tool] += 1
        mem["successful_tools"] = dict(successful_tools)
    
    # =========================================================================
    # CLEANUP & SAVE
    # =========================================================================
    # Keep memory bounded
    if len(entities) > 100:
        entities = entities[-100:]
    if len(facts) > 50:
        facts = facts[-50:]
    
    mem["entities"] = entities
    mem["facts"] = facts
    mem["last_updated"] = datetime.now().isoformat()
    conversation_memory[conversation_id] = mem
    
    logger.debug(f"Memory updated: {len(entities)} entities, {len(facts)} facts, {len(mem.get('qa_pairs', []))} Q&A pairs")


def build_memory_context(conversation_id: Optional[str]) -> str:
    """
    Build a rich memory context for the LLM including:
    - Recent entities mentioned
    - Key facts learned
    - User preferences
    - Relevant past Q&A
    """
    if not conversation_id:
        return ""
    
    mem = conversation_memory.get(conversation_id)
    if not mem:
        return ""
    
    lines: List[str] = []
    
    # User preferences
    preferences = mem.get("preferences", {})
    if preferences:
        pref_str = ", ".join(f"{k}: {v}" for k, v in preferences.items())
        lines.append(f"[User preferences: {pref_str}]")
    
    # Recent entities (last 10)
    entities = mem.get("entities", [])[-10:]
    if entities:
        entity_str = ", ".join(f"{e['value']} ({e['type']})" for e in entities)
        lines.append(f"[Recent topics: {entity_str}]")
    
    # Key facts (last 5)
    facts = mem.get("facts", [])[-5:]
    if facts:
        lines.append("[Known facts:]")
        for f in facts:
            lines.append(f"  - {f.get('subject')}: {f.get('value')} ({f.get('type')})")
    
    # Recent Q&A summaries (last 3)
    qa_pairs = mem.get("qa_pairs", [])[-3:]
    if qa_pairs:
        lines.append("[Recent questions in this conversation:]")
        for qa in qa_pairs:
            lines.append(f"  - Q: {qa['query'][:80]}...")
    
    return "\n".join(lines) if lines else ""


def build_history_context(
    messages: Optional[List[Dict[str, Any]]],
    current_query: Optional[str] = None,
    max_turns: int = 8,
    max_chars_per_turn: int = 300,
    max_total_chars: int = 2000,
) -> str:
    """
    Construct a compact, model-friendly conversation context from past turns.

    - Uses only user/assistant roles
    - Excludes the current user message if duplicated in history
    - Truncates each turn and caps total context length
    """
    if not messages:
        return ""

    # Keep only user/assistant roles and strip empty content
    filtered: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            filtered.append({"role": role, "content": content})

    if not filtered:
        return ""

    # Drop the last user message if it's exactly the current query (to avoid duplication)
    if current_query:
        for i in range(len(filtered) - 1, -1, -1):
            if filtered[i]["role"] == "user" and filtered[i]["content"] == current_query:
                filtered.pop(i)
                break

    # Take last N relevant turns
    recent = filtered[-(max_turns * 2) :]

    # Format chronologically
    lines: List[str] = []
    total_chars = 0
    for turn in recent:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        # Trim overly long turns
        if len(content) > max_chars_per_turn:
            content = content[: max_chars_per_turn - 1] + "…"

        candidate_line = f"- {prefix}: {content}"
        if total_chars + len(candidate_line) > max_total_chars:
            break
        lines.append(candidate_line)
        total_chars += len(candidate_line)

    if not lines:
        return ""

    return "\n".join(lines)

def is_simple_query(query: str) -> bool:
    """
    Detects if a query is simple enough to bypass the agent workflow.
    
    Simple queries include:
    - Greetings (hi, hello, hey)
    - Basic questions (how are you, what's up)
    - Thanks (thank you, thanks)
    """
    simple_patterns = [
        r"^(hi|hello|hey|greetings)(\s+there)?(!|\.|)?$",
        r"^(how are you|what'?s up|how'?s it going|how do you do)(\?|\.|!)?$",
        r"^(thanks|thank you|ty)(\s+so much)?(!|\.|)?$"
    ]
    
    query = query.lower().strip()
    return any(re.match(pattern, query) for pattern in simple_patterns)

# Add simple responses for direct handling
SIMPLE_RESPONSES = {
    "greeting": "Hello! I'm your AI assistant. How can I help you today?",
    "how_are_you": "I'm doing well, thank you for asking! How can I assist you?",
    "thanks": "You're welcome! Is there anything else I can help you with?"
}

def get_simple_response(query: str) -> str:
    """Returns an appropriate simple response based on query type"""
    query = query.lower().strip()
    logger.debug(f"query: {query}")

    if re.match(r"^(hi|hello|hey|greetings)(\s+there)?(!|\.|)?$", query):
        return SIMPLE_RESPONSES["greeting"]
    elif re.match(r"^(how are you|what'?s up|how'?s it going|how do you do)(\?|\.|!)?$", query):
        return SIMPLE_RESPONSES["how_are_you"]
    elif re.match(r"^(thanks|thank you|ty)(\s+so much)?(!|\.|)?$", query):
        return SIMPLE_RESPONSES["thanks"]
    
    return None 


def parse_agent_steps(agent_output: str) -> List[str]:
    """
    Parses the agent's raw output string to extract human-readable steps.
    """    
    if not agent_output:
        return []
        
    steps = []
    current_step = []
    
    for line in agent_output.strip().split('\n'):
        if line.startswith(("Thought:", "Action:", "Observation:")):
            if current_step:
                formatted_step = format_step(" ".join(current_step))
                steps.append(formatted_step)
            current_step = [line]
        elif line.strip():
            current_step.append(line)
            
    if current_step:
        formatted_step = format_step(" ".join(current_step))
        steps.append(formatted_step)
        
    return steps
def format_step(step: str) -> str:
    """
    Formats and cleans a step for display by removing unwanted patterns and Python objects.
    """
    if not step or not isinstance(step, str):
        return ""
        
    # Convert to string if it's not already
    step_str = str(step)
    
    # Remove common unwanted patterns
    patterns_to_remove = [
        r"ActionStep\([^)]*\)",
        r"<MessageRole\.[^>]*>",
        r"MessageRole\.[A-Z_]+",
        r"tool_calls=\[[^\]]*\]",
        r"model_input_messages=\[[^\]]*\]",
        r"start_time=[\d.]+",
        r"end_time=[\d.]+",
        r"step_number=\d+",
        r"duration=[\d.]+",
        r"observations_images=None",
        r"action_output=None",
        r"error=[^,)]*",
        r"'role': <[^>]*>",
        r"'content': \[.*?\]",
        r"ToolCall\([^)]*\)",
        r"ChatMessage\([^)]*\)",
        r"ChatCompletion\([^)]*\)",
        r"CompletionUsage\([^)]*\)",
    ]
    
    # Apply all removal patterns
    for pattern in patterns_to_remove:
        step_str = re.sub(pattern, "", step_str, flags=re.DOTALL)
    
    # Extract useful information
    if "Thought:" in step_str and "Code:" in step_str:
        # Extract Thought and Code sections
        thought_match = re.search(r"Thought:\s*([^C]*?)(?=Code:|$)", step_str, re.DOTALL)
        code_match = re.search(r"Code:\s*```(?:python|py)?\s*(.*?)```", step_str, re.DOTALL)
        
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = ""
            
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = ""
        
        if thought or code:
            formatted = ""
            if thought:
                formatted += f"💭 **Thought:** {thought}\n"
            if code:
                formatted += f"💻 **Code:**\n```python\n{code}\n```"
            return formatted
    
    # Try to extract just the meaningful text
    if "model_output=" in step_str:
        output_match = re.search(r"model_output='([^']*)'", step_str)
        if output_match:
            return f"🤖 **Agent Output:** {output_match.group(1)}"
    
    # If it's just a simple string, clean and return it
    step_str = re.sub(r"\s+", " ", step_str)  # Normalize whitespace
    step_str = step_str.strip()
    
    # Remove very technical/debug info
    if any(term in step_str.lower() for term in ["actionstep", "messagero", "toolcall", "chatcompletion"]):
        return ""
    
    # If the step is too short or empty after cleaning, return empty
    if len(step_str) < 10:
        return ""
    
    return f"📝 **Step:** {step_str}"

class StepTracker:
    """
    Tracks and formats ReAct agent steps for user-friendly display.
    ReAct Pattern: Thought → Action → Observation → ... → Final Answer
    """
    def __init__(self):
        self.steps = []
        self.iteration = 0
        
    def __call__(self, step: str):
        """Called by the agent at each step"""
        formatted_steps = self.format_step_realtime(step)
        for step in formatted_steps:
            if step:
                self.steps.append(step)
    
    def _get_action_description(self, code: str) -> tuple[str, str]:
        """
        Convert code to human-readable action description.
        Returns (emoji, description)
        """
        code_lower = code.lower()
        
        if 'final_answer' in code_lower:
            return "✅", "Generating final answer..."
        elif 'get_weather' in code_lower:
            # Extract location if possible
            loc_match = re.search(r'location=["\']([^"\']+)["\']', code)
            loc = loc_match.group(1) if loc_match else "requested location"
            return "🌤️", f"Fetching weather for {loc}..."
        elif ('web_search_ctx' in code_lower) or ('web_search' in code_lower):
            # Extract query if possible
            query_match = re.search(r'query=["\']([^"\']+)["\']', code)
            query = query_match.group(1)[:50] if query_match else "your question"
            return "🔎", f"Searching the web for: \"{query}\"..."
        elif 'webscraper' in code_lower:
            url_match = re.search(r'url=["\']([^"\']+)["\']', code)
            url = url_match.group(1)[:40] + "..." if url_match else "webpage"
            return "🌐", f"Scraping content from {url}"
        elif 'visit_webpage' in code_lower:
            return "🌐", "Visiting webpage to extract content..."
        elif 'retrieve_knowledge' in code_lower:
            query_match = re.search(r'query=["\']([^"\']+)["\']', code)
            query = query_match.group(1)[:40] if query_match else "your question"
            return "📚", f"Searching knowledge base for: \"{query}\"..."
        elif 'print(' in code_lower:
            return "", ""  # Skip print statements
        else:
            # Show short code snippet for transparency
            if len(code) < 80:
                return "⚙️", f"Executing: `{code.strip()}`"
            return "⚙️", "Executing code..."
    
    def _format_observation(self, observation: str) -> str:
        """Format observation result for user display"""
        # Ensure observation is safely encoded as UTF-8
        observation = _safe_utf8_str(observation)
        observation = re.sub(r'\s+', ' ', observation).strip()
        
        # Weather results
        if 'temperature' in observation.lower() or 'weather' in observation.lower():
            result = observation[:200] if len(observation) > 200 else observation
            return _safe_utf8_str(result)
        
        # Retrieved chunks
        if 'retrieved' in observation.lower() and 'chunk' in observation.lower():
            chunk_match = re.search(r'(\d+)\s*chunk', observation.lower())
            if chunk_match:
                return f"Found {chunk_match.group(1)} relevant document sections"
            return "Retrieved relevant document sections"
        
        # Web content
        if 'webpage' in observation.lower() or 'html' in observation.lower() or len(observation) > 300:
            return "Content retrieved successfully"
        
        # Error
        if 'error' in observation.lower() or 'failed' in observation.lower():
            result = observation[:150] + "..." if len(observation) > 150 else observation
            return _safe_utf8_str(result)
        
        # Default: truncate if too long
        if len(observation) > 150:
            result = observation[:150] + "..."
            return _safe_utf8_str(result)
        return _safe_utf8_str(observation)
            
    def format_step_realtime(self, step_content) -> List[str]:
        """
        Format steps following ReAct pattern for clear user display.
        Returns list of formatted step strings.
        """
        if not step_content:
            return []
        
        # Convert to string with safe UTF-8 encoding
        if hasattr(step_content, 'model_output'):
            step_str = _safe_utf8_str(step_content.model_output)
        elif hasattr(step_content, 'content'):
            step_str = _safe_utf8_str(step_content.content)
        else:
            step_str = _safe_utf8_str(step_content).strip()
        
        if not isinstance(step_str, str):
            step_str = _safe_utf8_str(step_str)
        
        formatted_steps = []
        
        # === THOUGHT (Reasoning) ===
        if "Thought:" in step_str:
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nCode:|\n\nCode:|\nAction:|\nObservation:|$)", step_str, re.DOTALL)
            if thought_match:
                thought = re.sub(r'\s+', ' ', thought_match.group(1).strip())
                # Clean up thought - remove technical jargon
                thought = thought.replace("I need to", "I'll")
                thought = thought.replace("I will", "I'll")
                if len(thought) > 200:
                    thought = thought[:200] + "..."
                formatted_steps.append(f"💭 **Thought:** {thought}")
        
        # === ACTION (Tool execution) ===
        if "Code:" in step_str:
            code_match = re.search(r"Code:\s*```(?:python|py)?\s*(.*?)```", step_str, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                emoji, action_desc = self._get_action_description(code)
                if action_desc:  # Skip if empty (e.g., print statements)
                    formatted_steps.append(f"{emoji} **Action:** {action_desc}")
        
        # === OBSERVATION (Result) ===
        if "Observation:" in step_str:
            obs_match = re.search(r"Observation:\s*(.+?)(?=\nThought:|\nCode:|\nAction:|$)", step_str, re.DOTALL)
            if obs_match:
                observation = _safe_utf8_str(obs_match.group(1).strip())
                formatted_obs = self._format_observation(observation)
                # Ensure formatted observation is safe
                formatted_obs = _safe_utf8_str(formatted_obs)
                if 'error' in formatted_obs.lower():
                    formatted_steps.append(f"❌ **Result:** {formatted_obs}")
                else:
                    formatted_steps.append(f"👁️ **Observation:** {formatted_obs}")
        
        # === FINAL ANSWER detection ===
        if "final_answer" in step_str.lower():
            if not any("final answer" in s.lower() for s in formatted_steps):
                formatted_steps.append("✅ **Generating final answer...**")
        
        # === Output lines (execution results) ===
        for line in step_str.split('\n'):
            line = line.strip()
            if line.startswith('Out - ') and 'Final answer' not in line:
                result = line.replace('Out - ', '').strip()
                if result and len(result) < 200:
                    formatted_steps.append(f"📤 **Output:** {result}")
        
        return formatted_steps
            
    def get_steps(self) -> List[str]:
        return self.steps


# =============================================================================
# 🚀 FAST RAG PATH - Direct document summarization without agent loop
# =============================================================================
# This bypasses the full ReAct agent for simple summary questions,
# reducing response time from ~600s to ~60s (1 LLM call instead of 7+)

def is_summary_question(query: str) -> bool:
    """
    Detect if query is a simple summary/overview question about a document.
    These can be handled with direct RAG retrieval + single LLM call.
    
    English only - comprehensive coverage of summary-type questions.
    """
    q = query.lower().strip()
    
    summary_patterns = [
        # =================================================================
        # DIRECT SUMMARY REQUESTS
        # =================================================================
        "summarize", "summary", "summarise", "sum up", "sumup",
        "give me a summary", "provide a summary", "can you summarize",
        "make a summary", "write a summary", "create a summary",
        "brief summary", "quick summary", "short summary",
        "executive summary", "general summary",
        "could you summarize", "would you summarize", "please summarize",
        "i need a summary", "i want a summary",
        
        # =================================================================
        # OVERVIEW REQUESTS
        # =================================================================
        "overview", "give me an overview", "provide an overview",
        "general overview", "quick overview", "brief overview",
        "high level overview", "high-level overview",
        "broad overview", "comprehensive overview",
        
        # =================================================================
        # "WHAT IS THIS ABOUT" VARIATIONS
        # =================================================================
        "what is it about", "what's it about", "whats it about",
        "what is this about", "what's this about", "whats this about",
        "what is this document about", "what's this document about",
        "what is this file about", "what's this file about",
        "what is the document about", "what's the document about",
        "what is the file about", "what's the file about",
        "what is this pdf about", "what's this pdf about",
        "what is the pdf about", "what's the pdf about",
        "what is it all about", "what's it all about",
        
        # =================================================================
        # "WHAT DOES IT TALK/DISCUSS/COVER" VARIATIONS
        # =================================================================
        "what does it talk about", "what does this talk about",
        "what does it discuss", "what does this discuss",
        "what does it cover", "what does this cover",
        "what does it contain", "what does this contain",
        "what does it say", "what does this say",
        "what does the document talk about", "what does the file talk about",
        "what does the document cover", "what does the file cover",
        "what does the document discuss", "what does the file discuss",
        "what is being discussed", "what is discussed",
        "what is covered", "what is being covered",
        "what does it address", "what does this address",
        "what does it explain", "what does this explain",
        
        # =================================================================
        # "TELL ME ABOUT" VARIATIONS
        # =================================================================
        "tell me about this", "tell me about it", "tell me about the document",
        "tell me about the file", "tell me about the content",
        "tell me about the pdf", "tell me about this pdf",
        "tell me what this is", "tell me what it is",
        "tell me what's in it", "tell me whats in it",
        "can you tell me about", "could you tell me about",
        "please tell me about", "i'd like to know about",
        
        # =================================================================
        # EXPLAIN/DESCRIBE VARIATIONS
        # =================================================================
        "explain this", "explain the document", "explain the content",
        "explain what this is", "explain to me", "explain it",
        "explain the file", "explain the pdf",
        "can you explain", "could you explain", "please explain",
        "describe this", "describe the document", "describe the content",
        "describe what this is", "describe it", "describe the file",
        "can you describe", "could you describe", "please describe",
        "give me a description", "provide a description",
        
        # =================================================================
        # CONTENT QUESTIONS
        # =================================================================
        "what are the contents", "what's the content", "whats the content",
        "what is the content", "show me the content",
        "what's inside", "whats inside", "what is inside",
        "what's in this", "whats in this", "what is in this",
        "what's in the document", "what's in the file", "what's in the pdf",
        "what's in there", "whats in there", "what is in there",
        "what's in here", "whats in here", "what is in here",
        "what do we have", "what have we got", "what do i have",
        "what am i looking at", "what do we have here",
        
        # =================================================================
        # MAIN TOPICS/POINTS/IDEAS
        # =================================================================
        "main topics", "main points", "main ideas", "main themes",
        "key topics", "key points", "key ideas", "key themes",
        "important topics", "important points", "important ideas",
        "central topics", "central points", "central ideas",
        "what are the main", "what are the key", "what are the important",
        "list the main", "list the key", "list topics", "list the topics",
        "what topics", "which topics", "what themes",
        "give me the main points", "give me the key points",
        "what are the takeaways", "key takeaways", "main takeaways",
        
        # =================================================================
        # GENERAL UNDERSTANDING / GIST
        # =================================================================
        "help me understand", "i want to understand", "help understand",
        "what should i know", "what do i need to know",
        "give me the gist", "the gist", "in a nutshell",
        "tl;dr", "tldr", "too long didn't read", "too long didnt read",
        "quick read", "brief read", "quick take",
        "high level", "high-level", "at a high level",
        "big picture", "the big picture", "give me the big picture",
        "bottom line", "the bottom line", "what's the bottom line",
        "in short", "in brief", "briefly",
        "long story short", "cut to the chase",
        "essence", "the essence", "what's the essence",
        "core message", "main message", "key message",
        
        # =================================================================
        # SIMPLE QUESTIONS
        # =================================================================
        "what is this", "what's this", "whats this",
        "what is it", "what's it", "whats it",
        "what have you got", "what do you have",
        "can you tell me what this is", "what am i reading",
        "what's going on here", "whats going on here",
        "break it down", "break this down", "break down the document",
        
        # =================================================================
        # ANALYSIS REQUESTS
        # =================================================================
        "analyze this", "analyse this", "analysis",
        "give me an analysis", "provide an analysis",
        "analyze the document", "analyse the document",
        "analyze the content", "analyse the content",
        "document analysis", "content analysis",
        "review this", "review the document", "give me a review",
        
        # =================================================================
        # RUNDOWN / BREAKDOWN
        # =================================================================
        "give me a rundown", "rundown", "the rundown",
        "give me a breakdown", "breakdown", "the breakdown",
        "walk me through", "walk through this",
        "go through this", "go over this", "run through this",
        "take me through", "guide me through",
    ]
    
    return any(pattern in q for pattern in summary_patterns)


async def try_fast_rag_path(query: str, request_data) -> Optional[str]:
    """
    🚀 Fast RAG Path: Handle simple document questions directly.
    
    Instead of:
      Agent Step 1 → LLM call (90s)
      Agent Step 2 → retrieve_knowledge
      Agent Step 3 → LLM call (90s)
      Agent Step 4 → retrieve_knowledge again
      Agent Step 5 → LLM call (90s)
      ...
      Total: ~630s
    
    We do:
      1. retrieve_knowledge (10s)
      2. Single LLM call (90s)
      Total: ~100s ✅
    
    Returns:
        NDJSON string to yield, or None if not applicable
    """
    # Check if this is a summary question
    if not is_summary_question(query):
        return None
    
    # Check if we have document context (from uploaded file)
    has_doc_context = (
        "[context: user just uploaded" in query.lower() or
        "doc_id" in query.lower() or
        (request_data.selectedTools and any(
            t.get("name") == "rag" for t in request_data.selectedTools
        ))
    )
    
    if not has_doc_context:
        return None
    
    try:
        from huggingsmolagent.tools.vector_store import retrieve_knowledge
        import httpx
        
        # Step 1: Send initial step
        step1 = {
            "steps": ["🚀 **Fast Mode:** Direct document retrieval..."],
            "response": None
        }
        # Step 2: Retrieve relevant chunks (top_k=15 for good coverage)
        logger.info("🚀 Fast RAG: Retrieving document chunks...")
        rag_result = retrieve_knowledge(
            query="document overview summary main topics content",
            top_k=15
        )
        
        if not rag_result or not rag_result.get("results"):
            logger.warning("🚀 Fast RAG: No results from retrieve_knowledge")
            return None
        
        # Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(rag_result["results"][:15]):
            content = result.get("content", "")
            if content:
                context_parts.append(f"[Chunk {i+1}]: {content[:800]}")
        
        context = "\n\n".join(context_parts)
        
        if not context.strip():
            logger.warning("🚀 Fast RAG: Empty context")
            return None
        
        logger.info("🚀 Fast RAG: Generating summary with single LLM call...")
        
        summary_prompt = f"""Based on the following document excerpts, provide a comprehensive summary.
Answer the user's question: "{query}"

Document excerpts:
{context}

Provide a clear, well-structured summary covering the main topics and key points."""

        # Call Ollama directly for the summary
        ollama_url = os.getenv("BASE_URL", "http://localhost:11434/v1")
        model = os.getenv("OPEN_AI_MODEL")
        
        # Increased timeout to 360s (6 min) for slow LLMs like qwen2.5:7b-instruct
        async with httpx.AsyncClient(timeout=360.0) as client:
            response = await client.post(
                f"{ollama_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that summarizes documents clearly and concisely."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            
            if response.status_code != 200:
                logger.error(f"🚀 Fast RAG: LLM call failed with status {response.status_code}")
                return None
            
            result = response.json()
            summary = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not summary:
                logger.warning("🚀 Fast RAG: Empty summary from LLM")
                return None
        
        # Build final response
        logger.info(f"🚀 Fast RAG: Success! Summary length: {len(summary)} chars")
        
        final_response = {
            "steps": [
                "🚀 **Fast Mode:** Direct document analysis",
                "📄 **Retrieved:** 15 relevant document sections",
                "✅ **Generated:** Document summary"
            ],
            "response": summary
        }
        
        return f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"🚀 Fast RAG Path error: {e}")
        import traceback
        traceback.print_exc()
        return None  # Fall back to normal agent


async def generate_streaming_response(request_data: ComplexRequest):
    """
    Generator function for streaming steps and final response in real-time.
    """
    # Initialize step communication system
    step_queue = queue.Queue()
    agent_finished = threading.Event()
    
    # Create step tracker with queue communication and deduplication
    class QueueStepTracker(StepTracker):
        def __init__(self, step_queue):
            super().__init__()
            self.step_queue = step_queue
            self.step_counter = 0
            self.sent_steps = set()  # Track sent steps to avoid duplicates
            
        def __call__(self, step):
            """This method is called by the agent at each step"""
            try:
                # Safely sanitize step content before processing to prevent encoding errors
                if hasattr(step, '__dict__'):
                    for attr_name in ['model_output', 'content', 'observation', 'action_output']:
                        if hasattr(step, attr_name):
                            try:
                                attr_value = getattr(step, attr_name)
                                if attr_value is not None:
                                    setattr(step, attr_name, _safe_utf8_str(attr_value))
                            except Exception:
                                pass
                
                formatted_steps = self.format_step_realtime(step)
                
                for formatted_step in formatted_steps:
                    if formatted_step:
                        # Ensure formatted step is UTF-8 safe
                        formatted_step = _safe_utf8_str(formatted_step)
                        
                        # Skip duplicate steps (compare normalized version)
                        step_key = formatted_step.strip().lower()
                        if step_key in self.sent_steps:
                            continue
                        self.sent_steps.add(step_key)
                        
                        self.steps.append(formatted_step)
                        self.step_counter += 1
                        # Use safe UTF-8 for print statement
                        safe_step_preview = _safe_utf8_str(formatted_step[:80])
                        print(f"🔍 Step {self.step_counter}: {safe_step_preview}...")
                        
                        # Send to queue immediately
                        self.step_queue.put(formatted_step)
            except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError) as unicode_err:
                # Catch all Unicode-related errors in step processing
                logger.error(f"Unicode error in step callback: {unicode_err}")
                try:
                    error_step = "⚠️ Step processing error: Unable to display step due to encoding issue"
                    self.step_queue.put(error_step)
                except Exception:
                    pass
            except Exception as e:
                # Catch any other errors in step processing
                logger.error(f"Error in step callback: {e}")
                try:
                    error_msg = _safe_utf8_str(str(e))[:100]
                    error_step = f"⚠️ Step processing error: {error_msg}"
                    self.step_queue.put(error_step)
                except Exception:
                    pass
    
    step_tracker = QueueStepTracker(step_queue)
    
    # Configure logging with UTF-8 safe handler
    agent_logger = logging.getLogger("smolagents")
    
    # Create a UTF-8 safe formatter
    class UTF8SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                # Ensure all record attributes are UTF-8 safe
                if hasattr(record, 'msg') and isinstance(record.msg, str):
                    record.msg = _safe_utf8_str(record.msg)
                if hasattr(record, 'args') and record.args:
                    safe_args = []
                    for arg in record.args:
                        if isinstance(arg, str):
                            safe_args.append(_safe_utf8_str(arg))
                        else:
                            safe_args.append(arg)
                    record.args = tuple(safe_args)
                return super().format(record)
            except Exception:
                # Fallback to basic formatting
                try:
                    return f"{record.levelname}: {_safe_utf8_str(str(record.msg))}"
                except Exception:
                    return f"{record.levelname}: [Error formatting log message]"
    
    # Create a UTF-8 safe stream handler
    class UTF8SafeStreamHandler(logging.StreamHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFormatter(UTF8SafeFormatter())
        
        def emit(self, record):
            try:
                # Ensure the message is UTF-8 safe
                msg = self.format(record)
                safe_msg = _safe_utf8_str(msg)
                stream = self.stream
                stream.write(safe_msg + self.terminator)
                self.flush()
            except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError):
                # If encoding fails, try to log a safe version
                try:
                    safe_record = record
                    safe_record.msg = _safe_utf8_str(str(record.msg))
                    msg = self.format(safe_record)
                    self.stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    pass  # Silently ignore if even the safe version fails
    
    # Remove existing handlers to avoid duplicates
    agent_logger.handlers.clear()
    
    # Add UTF-8 safe handlers
    list_handler = ListLogHandler()
    list_handler.setFormatter(UTF8SafeFormatter())
    utf8_handler = UTF8SafeStreamHandler()
    agent_logger.addHandler(list_handler)
    agent_logger.addHandler(utf8_handler)
    agent_logger.setLevel(logging.DEBUG) 
    agent_logger.propagate = False
    
    # Also patch the root logger used by smolagents
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(UTF8SafeFormatter())

    try:
        # Extract query from different possible formats in the request
        query = None
        if request_data.toolsQuery:
            query = request_data.toolsQuery
        elif request_data.messages and len(request_data.messages) > 0:
            # Get the last user message if available
            for message in reversed(request_data.messages):
                if message.get("role") == "user":
                    query = message.get("content")
                    break
        elif request_data.chatSettings and "query" in request_data.chatSettings:
            query = request_data.chatSettings["query"]
            
        if not query:
            raise HTTPException(status_code=400, detail="No query found in request")
            
        logger.info(f"Processing query: {query}")

        # Check if this is a simple query that can bypass the agent workflow
        if is_simple_query(query):
            logger.info(f"Detected simple query, bypassing agent workflow")
            simple_response = get_simple_response(query)
            # Send simple response via streaming
            simple_data = {
                "steps": ["Simple query handled directly."],
                "response": simple_response,
                "canHandle": True
            }
            try:
                json_str = json.dumps(simple_data, ensure_ascii=False)
                yield f"data: {json_str}\n\n"
            except Exception as e:
                logger.error(f"Error encoding simple response to JSON: {e}")
                # Fallback
                fallback_data = {
                    "steps": ["Simple query handled."],
                    "response": "Hello!",
                    "canHandle": True
                }
                yield f"data: {json.dumps(fallback_data)}\n\n"
            return

        # =====================================================================
        # 🚀 FAST RAG PATH: Bypass agent for simple document summary questions
        # =====================================================================
        # Detect if this is a simple summary question about an uploaded document
        # and handle it directly with ONE LLM call instead of 7+ agent steps
        fast_rag_result = await try_fast_rag_path(query, request_data)
        if fast_rag_result is not None:
            logger.info("🚀 Fast RAG path handled the query directly")
            yield fast_rag_result
            return

        # Build conversation context from history (only if multi-turn conversation)
        history_context = ""
        memory_context = ""
        
        # Skip context building for first message or simple queries (optimization)
        has_conversation = (
            request_data and 
            hasattr(request_data, "messages") and 
            request_data.messages and 
            len(request_data.messages) > 1
        )
        
        if has_conversation:
            history_context = build_history_context(
                request_data.messages,
                current_query=query,
            )
            memory_context = build_memory_context(getattr(request_data, "conversationId", None))
        
        enhanced_query = query

        # Intent classification to hint tool choice
        def classify_query_intent(user_query: str, selected_tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            intent = "scrape"  # Default to scrape (web search) instead of rag
            reason = []
            detected_url = None
            q = (user_query or "").lower()
            m = re.search(r"https?://\S+", user_query or "")
            if m:
                detected_url = m.group(0)
                intent = "scrape"
                reason.append("url detected")
            
            # Weather keywords - should use get_weather_simple(), NOT web_search_ctx
            weather_keywords = [
                "weather", "temperature", "forecast", "how is it in", "what's the weather",
                "how's the weather", "is it raining", "is it sunny", "is it cold", "is it hot"
            ]
            if any(k in q for k in weather_keywords):
                intent = "weather"
                reason.append("weather keywords")
                return {"intent": intent, "url": detected_url, "reason": ", ".join(reason) or "heuristics"}
            
            scrape_keywords = [
                "scrape", "crawl", "website", "webpage", "from site", "from web", "google", "search the web",
                "news", "current", "today", "latest"  # Removed "weather" - handled separately
            ]
            if any(k in q for k in scrape_keywords):
                intent = "scrape"
                reason.append("scrape keywords")
            rag_keywords = [
                "pdf", "document", "knowledge base", "kb", "vector", "embedding", "retrieve", "chunks", "supabase", "stored docs", "my documents"
            ]
            if any(k in q for k in rag_keywords) and not detected_url:
                intent = "rag"
                reason.append("rag keywords")
            if selected_tools:
                names = [t.get("name", "").lower() for t in selected_tools if isinstance(t, dict)]
                if any(n in names for n in ["webscraper", "web_search_ctx", "web_search"]):
                    intent = "scrape"
                    reason.append("selectedTools web")
                if any(n in names for n in ["retrieve_knowledge", "retriever", "rag"]):
                    intent = "rag"
                    reason.append("selectedTools rag")
            return {"intent": intent, "url": detected_url, "reason": ", ".join(reason) or "heuristics"}

        intent_info = classify_query_intent(query, getattr(request_data, "selectedTools", None))
        
        # EARLY RETURN: If weather intent but no city specified, ask user to specify
        if intent_info['intent'] == "weather":
            # Check if query contains a recognizable city/location
            q_lower = query.lower()
            # Common patterns that indicate a city IS specified
            city_patterns = [
                r'\bin\s+\w+',  # "in London", "in Paris"
                r'\bfor\s+\w+',  # "for Tokyo"
                r'\bat\s+\w+',  # "at New York"
            ]
            # List of common cities to check
            common_cities = [
                "london", "paris", "tokyo", "new york", "berlin", "madrid", "rome",
                "amsterdam", "barcelona", "moscow", "sydney", "melbourne", "toronto",
                "vancouver", "chicago", "los angeles", "san francisco", "seattle",
                "boston", "miami", "dubai", "singapore", "hong kong", "beijing",
                "shanghai", "mumbai", "delhi", "cairo", "lagos", "nairobi"
            ]
            
            has_city = any(city in q_lower for city in common_cities)
            has_pattern = any(re.search(p, q_lower) for p in city_patterns)
            
            if not has_city and not has_pattern:
                # No city detected - return early asking user to specify
                logger.info("Weather query without city detected - asking user to specify location")
                no_city_response = {
                    "type": "final",
                    "answer": "I'd be happy to help with weather information! 🌤️\n\nPlease specify a city or location, for example:\n- \"Weather in Paris\"\n- \"How is it in London?\"\n- \"Temperature in Tokyo\"",
                    "steps": [{
                        "type": "step",
                        "emoji": "🌤️",
                        "description": "Weather query detected but no city specified"
                    }]
                }
                yield f"data: {json.dumps(no_city_response)}\n\n"
                return
        
        # Generate intent hint based on detected intent type
        if intent_info['intent'] == "weather":
            intent_hint_tools = "Preferred tool: get_weather_simple(location). DO NOT use web_search_ctx for weather!"
        elif intent_info['intent'] == "rag":
            intent_hint_tools = "Preferred tools: retrieve_knowledge (RAG)."
        else:
            intent_hint_tools = "Preferred tools: web_search_ctx then webscraper (web)."
        
        intent_hint = (
            f"Detected intent: {intent_info['intent']}. "
            + (f"URL: {intent_info['url']}. " if intent_info.get("url") else "")
            + intent_hint_tools + " "
            + f"Reason: {intent_info['reason']}."
        )

        if history_context:
            intro = (
                "You are in a multi-turn conversation. Maintain continuity and carry implied parameters "
                "(topic, location, filters) from prior turns unless the user changes them."
            )
            enhanced_query = intro + "\n" + intent_hint + "\n"
            if memory_context:
                enhanced_query += "Known conversation memory:\n" + memory_context + "\n"
            enhanced_query += "Conversation so far:\n" + history_context + "\n\nCurrent user message:\n" + query
        else:
            enhanced_query = intent_hint + "\n\n" + query

        try:
            max_enhanced_chars = int(os.getenv("AGENT_ENHANCED_QUERY_MAX_CHARS", "6000"))
        except Exception:
            max_enhanced_chars = 6000
        if max_enhanced_chars > 0 and len(enhanced_query) > max_enhanced_chars:
            enhanced_query = enhanced_query[:max_enhanced_chars] + "\n[Prompt truncated]\n"

        # Start timing the agent execution
        start_time = time.time()
        logger.debug("Initializing OpenAIServerModel")

        # Configure the language model (Ollama/OpenAI-compatible server)
        # Fix encoding issue for international characters
        import locale
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        
        _sanitize_http_header_env()
        llm_model = _build_openai_server_model()
        
        # Load prompt templates from YAML file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "tools", "prompt.yaml")
        prompt_templates = {}
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as stream:
                    prompt_templates = yaml.safe_load(stream) or {}
            else:
                logger.info(f"prompt.yaml not found at {prompt_path}, continuing without it")
        except Exception as e:
            logger.warning(f"Failed to load prompt.yaml: {e}")
        
        # Configure the agent with tools and settings
        logger.debug("Setting up tools for CodeAgent")
        # FAST MODE: Reduce the number of tools available to the agent to minimize tool thrashing
        # (e.g., web_search_ctx -> webscraper -> visit_webpage loops) and reduce total LLM steps.
        fast_mode = os.getenv("AGENT_FAST_MODE", "true").lower() == "true"

        # Base tools (kept small on purpose).
        tools_list = [
            get_weather_simple,
            search_news,  # Optimized for current news articles
            web_search_ctx,   # Preferred for most web questions (search + scrape + relevance filter)
        ]

        # Only expose direct scraping tools when a specific URL is present (or fast mode is off).
        has_explicit_url = bool(intent_info.get("url")) or ("http://" in query.lower()) or ("https://" in query.lower())
        if not fast_mode or has_explicit_url:
            tools_list.extend([
                webscraper,
                visit_webpage,  # Custom tool with truncation - overrides default
            ])
        
        # Check if RAG/document retrieval is needed
        has_rag_tool = False
        if request_data.selectedTools:
            for tool in request_data.selectedTools:
                if tool.get("name") == "rag" or "rag" in str(tool).lower():
                    has_rag_tool = True
                    break
        
        # Also check if there's a doc_id in the query context (from uploaded files)
        if has_rag_tool or "doc_id" in query.lower() or "[context: user just uploaded" in query.lower():
            tools_list.append(retrieve_knowledge)
            logger.info("✅ retrieve_knowledge tool added (RAG mode active)")
        else:
            logger.info("ℹ️ retrieve_knowledge tool NOT added (no files uploaded)")
        
        agent = create_agent(
            tools=tools_list,
            model=llm_model,
            step_callbacks=[step_tracker],
            max_steps=5
        )
        
        # Run agent in thread and stream steps
        agent_result = None
        agent_error = None
        agent_start_time = time.time()
        # Allow configurable timeout via environment variable
        # Reduced to 300s (5 min) since Fast RAG Path handles most summary queries
        AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT_SECONDS", "600"))
        logger.info(f"Agent timeout set to {AGENT_TIMEOUT}s")
        
     
        # Send initial message
        initial_data = {
            "steps": ["🚀 **Starting ReAct Agent...** Analyzing your question"],
            "response": None
        }
        yield f"data: {json.dumps(initial_data, ensure_ascii=False)}\n\n"
        
        def run_agent():
            nonlocal agent_result, agent_error
            # Save original encoding settings
            original_stdout_encoding = getattr(sys.stdout, 'encoding', None)
            original_stderr_encoding = getattr(sys.stderr, 'encoding', None)
            
            try:
                # Force UTF-8 encoding in this thread (threads may not inherit encoding)
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                if hasattr(sys.stderr, 'reconfigure'):
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                
                logger.info("Running agent with query")
                _sanitize_http_header_env()
                for k, v in os.environ.items():
                    if isinstance(v, str) and any(ord(ch) > 127 for ch in v):
                        os.environ[k] = _ascii_only(v)
                # Note: sys.stdout/stderr are already configured with UTF-8 at the top of the file
                # No more generic "Initializing" message - let actual steps speak
                # Wrap in try-except to catch encoding errors from smolagents
                try:
                    agent_result = agent.run(_ascii_only(enhanced_query))
                except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError, AgentGenerationError) as unicode_err:
                    # Catch all Unicode-related errors (UnicodeEncodeError is a subclass of UnicodeError)
                    logger.error(f"Unicode error in agent.run(): {unicode_err}")
                    if "ascii" in str(unicode_err).lower() and "codec" in str(unicode_err).lower():
                        _log_non_ascii_env()
                    # Try to get partial result if available
                    if 'agent_result' in locals() and agent_result is not None:
                        try:
                            # Safely convert result to string
                            agent_result = _safe_utf8_str(agent_result)
                        except Exception:
                            agent_result = "Error: Unable to process model output due to encoding issues."
                    else:
                        # Use safe UTF-8 for error message
                        error_msg = _safe_utf8_str(str(unicode_err))
                        agent_result = f"Error in generating model output: {error_msg}"
                    raise  # Re-raise to be caught by outer except
                agent_finished.set()  # Signal completion
                print("🔍 Agent execution completed")
            except Exception as e:
                agent_error = e
                agent_finished.set()  # Signal completion even on error
                # Use safe UTF-8 for all error logging
                try:
                    safe_error_msg = _safe_utf8_str(str(e))
                    logger.error(f"🔍 Agent execution failed: {safe_error_msg}")
                    safe_traceback = _safe_utf8_str(traceback.format_exc())
                    logger.error(safe_traceback)
                except Exception:
                    logger.error("Agent execution failed (error message encoding failed)")
                # Ensure error message is UTF-8 safe
                error_msg = _safe_utf8_str(str(e))
                print(f"🔍 Agent execution failed: {error_msg}")
            finally:
                # Restore original encoding if possible (though it may not be necessary)
                try:
                    if original_stdout_encoding and hasattr(sys.stdout, 'reconfigure'):
                        sys.stdout.reconfigure(encoding=original_stdout_encoding)
                    if original_stderr_encoding and hasattr(sys.stderr, 'reconfigure'):
                        sys.stderr.reconfigure(encoding=original_stderr_encoding)
                except Exception:
                    pass  # Ignore errors when restoring
        # Start agent in background thread
        agent_thread = threading.Thread(target=run_agent)
        agent_thread.start()
        
        # Stream steps as they come from the queue
        last_progress_log = time.time()
        last_heartbeat = time.time()
        heartbeat_interval = 45  # ⭐ OPTIMIZED: Send heartbeat only every 45s to reduce noise
        sent_heartbeat_once = False  # Only send ONE heartbeat message to avoid spam
        
        while not agent_finished.is_set() or not step_queue.empty():
            # Check for timeout
            elapsed = time.time() - agent_start_time
            
            # Log progress every 30 seconds
            if elapsed - (last_progress_log - agent_start_time) >= 30:
                logger.info(f"⏱️ Agent still running... {elapsed:.1f}s elapsed (timeout: {AGENT_TIMEOUT}s)")
                last_progress_log = time.time()
            
            if elapsed > AGENT_TIMEOUT:
                logger.error(f"⏱️ Agent timeout after {elapsed:.1f}s (limit: {AGENT_TIMEOUT}s)")
                agent_error = TimeoutError(f"Agent execution exceeded {AGENT_TIMEOUT}s timeout")
                agent_finished.set()
                break
            
            # Send ONE heartbeat if no activity for a while (avoid spam)
            time_since_last_heartbeat = time.time() - last_heartbeat
            if time_since_last_heartbeat >= heartbeat_interval and not agent_finished.is_set() and not sent_heartbeat_once:
                # Send only ONE progress message to show the agent is still working
                progress_msg = "⏳ **Still working...** The agent is processing your request"
                sent_heartbeat_once = True  # Only send once
                
                progress_data = {
                    "steps": [progress_msg],
                    "response": None
                }
                json_str = json.dumps(progress_data, ensure_ascii=False)
                yield f"data: {json_str}\n\n"
                print(f"💓 Sent single heartbeat to frontend")
                last_heartbeat = time.time()
            
            try:
                # Try to get step from queue with timeout
                step = step_queue.get(timeout=0.1)
                
                # Send step immediately
                steps_data = {
                    "steps": [step],
                    "response": None
                }
                json_str = json.dumps(steps_data, ensure_ascii=False)
                yield f"data: {json_str}\n\n"
                # Use safe UTF-8 for print statement
                safe_step_preview = _safe_utf8_str(step[:50])
                print(f"🔍 Streamed step to frontend: {safe_step_preview}...")
                
                # Reset heartbeat timer when we send a real step
                last_heartbeat = time.time()
                
            except queue.Empty:
                # No step available, continue waiting
                await asyncio.sleep(0.1)
                continue
        
        # Wait for agent to complete (with timeout)
        agent_thread.join(timeout=5)
        
        # Check for errors
        if agent_error:
            # Send error message to frontend with safe UTF-8 encoding
            error_msg = _safe_utf8_str(str(agent_error))
            if isinstance(agent_error, TimeoutError):
                error_msg = f"⏱️ The task took too long to complete (>{AGENT_TIMEOUT}s). Try simplifying your question or asking for specific information."
            elif isinstance(agent_error, UnicodeEncodeError):
                # Handle encoding errors specifically
                error_msg = "Error processing response: Unable to handle special characters in the output. Please try rephrasing your question."
            
            error_data = {
                "steps": [],
                "response": error_msg,
                "error": error_msg  # Send the actual error message, not just True
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return

        # Process final response
        # Convert output to string for processing with safe UTF-8 encoding
        if not isinstance(agent_result, str):
            agent_result_str = _safe_utf8_str(agent_result)
        else:
            agent_result_str = _safe_utf8_str(agent_result)

        logger.debug(f"Raw agent output string from agent.run():\n{agent_result_str}")

        # Get steps directly from the step tracker
        parsed_display_steps = step_tracker.get_steps()
        
        # If no steps were captured, try traditional parsing
        if not parsed_display_steps:
            parsed_display_steps = parse_agent_steps(agent_result_str)
        
        # Extract the final answer - handle different return types
        final_answer_text = ""
        
        # Try to get action_output from the agent result if it's an object
        if hasattr(agent_result, 'action_output') and agent_result.action_output:
            final_answer_text = _safe_utf8_str(agent_result.action_output)
        else:
            # Fallback to string parsing
            final_answer_text = extract_final_answer(agent_result_str, _safe_utf8_str)

        # Process tool results
        tool_results = []
        for step in parsed_display_steps:
            if any(marker in step.lower() for marker in ["result:", "results:", "weather forecast"]):
                tool_results.append(step)
        
        # Determine the final response
        if not final_answer_text or final_answer_text == "Unable to extract final answer":
            if tool_results:
                final_answer_text = "\n\n".join(tool_results)
            elif parsed_display_steps:
                final_answer_text = parsed_display_steps[-1]
            else:
                final_answer_text = "I couldn't process this request with the available tools."

        # Update conversation memory with extracted facts, entities, and Q&A pairs
        try:
            update_conversation_memory(
                conversation_id=getattr(request_data, "conversationId", None),
                query=query,  # The original user query
                final_text=final_answer_text,
                tool_calls=None  # TODO: Extract tool names from agent execution
            )
        except Exception as mem_err:
            logger.warning(f"Memory update failed: {mem_err}")

        # Check if the agent can handle the query
        result_str_lower = final_answer_text.lower()
        cannot_handle_patterns = [
            "cannot complete this task with the available tools",
            "unable to fulfill this request",
            "don't have the tools",
            "cannot access",
            "cannot retrieve",
            "cannot get information about",
            "not possible to get",
            "i'm unable to",
            "beyond my capabilities",
            "i don't have access to",
            "i don't have the ability to",
            "i can't perform this action",
            "sorry, i cannot",
            "i am unable to"
        ]
        
        # Determine if the query was handled successfully
        is_unhandled = any(pattern in result_str_lower for pattern in cannot_handle_patterns)
        
        # No separate completion message - the final response is enough
        
        # Send final response
        final_data = {
            "steps": [],
            "response": final_answer_text,
            "canHandle": not is_unhandled
        }
        # Ensure JSON is properly encoded
        try:
            json_str = json.dumps(final_data, ensure_ascii=False)
            yield f"data: {json_str}\n\n"
        except Exception as e:
            logger.error(f"Error encoding final response to JSON: {e}")
            # Fallback with error response
            error_data = {
                "steps": [],
                "response": "Error encoding response",
                "canHandle": False,
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

        # Log execution time
        end_time = time.time()
        logger.info(f"Agent processing time: {end_time - start_time:.2f} seconds")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        error_data = {
            "steps": [],
            "response": None,
            "error": str(http_exc.detail)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        error_str = _safe_utf8_str(str(e))
        
        # Handle specific error cases
        if "401 Client Error: Unauthorized" in error_str:
            error_detail = "Authentication error with Hugging Face API. Please check your token validity."
        elif "403 Forbidden" in error_str and "does not have sufficient permissions" in error_str:
            error_detail = "Insufficient permissions to use the model. Please check your HF token permissions."
        elif "ascii" in error_str.lower() and "codec" in error_str.lower():
            error_detail = "Error processing response: Unable to handle special characters in the output. Please try rephrasing your question."
        else:
            error_detail = f"Error during agent execution: {error_str}"
        
        # Handle general errors
        logger.error(f"Error in run_agent: {error_str}", exc_info=True)
        error_data = {
            "steps": [],
            "response": None,
            "error": _safe_utf8_str(error_detail)
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

@app.post("/")
async def run_agent_streaming(request_data: ComplexRequest):
    """
    Main entry point that returns a StreamingResponse.
    """
    return StreamingResponse(
        generate_streaming_response(request_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        )


def run_agent_sync(query: str) -> str:
    """
    Synchronous agent execution for /ask endpoint.
    The agent will decide which tools to use (RAG, summarize, scrape) based on the query.
    
    Args:
        query: User question/request (may include context from main.py)
        
    Returns:
        str: Agent's final answer
    """
    print(f"[agent_sync] Starting with query: '{query[:100]}'")
    
    try:
        # Initialize the agent with all tools
        # Fix encoding issue for international characters
        import locale
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        
        _sanitize_http_header_env()
        model = _build_openai_server_model()
        
        # All available tools - keep a smaller set in fast mode to reduce latency.
        # Note: web_search_ctx already performs search + scraping + relevance filtering.
        fast_mode = os.getenv("AGENT_FAST_MODE", "true").lower() == "true"
        has_explicit_url = ("http://" in query.lower()) or ("https://" in query.lower())

        tools = [get_weather_simple, search_news, web_search_ctx]
        if not fast_mode or has_explicit_url:
            tools.extend([webscraper, visit_webpage])
        
        # Add retrieve_knowledge only if documents are mentioned in query
        # This matches the streaming version's behavior for consistency
        if "doc_id" in query.lower() or "[context: user just uploaded" in query.lower() or "document" in query.lower():
            tools.append(retrieve_knowledge)
            print(f"[agent_sync] ✅ retrieve_knowledge tool added (document context detected)")
        else:
            print(f"[agent_sync] ℹ️ retrieve_knowledge tool NOT added (no document context)")
        
        # Use factory function for consistent configuration (uses AGENT_INSTRUCTIONS from top of file)
        agent = create_agent(
            tools=tools,
            model=model,
            max_steps=5,
        )
        
        print(f"[agent_sync] Agent initialized with {len(tools)} tools")
        
        # Run the agent with error handling for encoding issues
        try:
            _sanitize_http_header_env()
            for k, v in os.environ.items():
                if isinstance(v, str) and any(ord(ch) > 127 for ch in v):
                    os.environ[k] = _ascii_only(v)
            result = agent.run(_ascii_only(query))
        except UnicodeEncodeError as enc_err:
            # Handle encoding errors from smolagents library
            print(f"[agent_sync] Unicode encoding error: {enc_err}")
            if "ascii" in str(enc_err).lower() and "codec" in str(enc_err).lower():
                _log_non_ascii_env()
            return f"Error in generating model output: Unable to process response with special characters. Please try rephrasing your question."
        except AgentGenerationError as gen_err:
            print(f"[agent_sync] Agent generation error: {gen_err}")
            if "ascii" in str(gen_err).lower() and "codec" in str(gen_err).lower():
                _log_non_ascii_env()
            return "Error in generating model output: Unable to process HTTP headers with non-ASCII characters. Please remove accents from OPENAI/OPENROUTER metadata env vars and retry."
        
        # Extract final answer with safe UTF-8 encoding
        if hasattr(result, 'action_output') and result.action_output:
            final_answer = _safe_utf8_str(result.action_output)
        else:
            final_answer = _safe_utf8_str(result)
        
        print(f"[agent_sync] Agent completed. Answer length: {len(final_answer)}")
        return final_answer
        
    except Exception as e:
        error_msg = _safe_utf8_str(str(e))
        print(f"[agent_sync] Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return f"Error running agent: {error_msg}"

