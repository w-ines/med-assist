"""
MedAssist API — Entry point.
All route logic lives in api/routes/. This file only wires things together.
"""

# Environment & encoding must be configured before any other import
from core.config import setup_encoding, patch_json_ascii, setup_logging, get_cors_origins
setup_encoding()
patch_json_ascii()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from api.agent_lc import router as agent_lc_router

# Smolagent (legacy — will be removed after full migration to deepagents)
try:
    from huggingsmolagent.agent import app as smolagent_router
    SMOLAGENT_AVAILABLE = True
except ImportError:
    SMOLAGENT_AVAILABLE = False
    print("[startup] ⚠️  huggingsmolagent not available (expected if fully migrated to deepagents)")

# Deep Agents router (optional)
try:
    from deepagents.router import router as deepagent_router
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False
    print("[startup] ⚠️  Deep Agents not available - run: pip install deepagents")

setup_logging()

# =============================================================================
# Create FastAPI app
# =============================================================================

app = FastAPI(title="MedAssist API", version="1.0.0")
print("[startup] FastAPI app initialized")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Mount routers
# =============================================================================

# Main API router (health, ask, upload, kg, pubmed, ner, cache, conversations, users, topics)
app.include_router(api_router)

# Legacy LangChain agent router
app.include_router(agent_lc_router)

# Smolagent streaming (mounted as sub-app for SSE) — legacy
if SMOLAGENT_AVAILABLE:
    app.mount("/agent", smolagent_router)
    print("[startup] ✅ Smolagent router mounted at /agent")
else:
    print("[startup] ⚠️  Smolagent router not available")

# Deep Agents router
if DEEPAGENTS_AVAILABLE:
    app.include_router(deepagent_router)
    print("[startup] ✅ Deep Agents router mounted at /agent-deep")
else:
    print("[startup] ⚠️  Deep Agents router not available")


if __name__ == "__main__":
    import uvicorn
    print("[startup] Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)