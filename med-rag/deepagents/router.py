"""
FastAPI router for Deep Agents endpoints.
Provides streaming SSE responses compatible with the existing frontend.
"""

import json
import asyncio
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class DeepAgentRequest(BaseModel):
    """Request model for Deep Agent queries."""
    query: str
    conversationId: Optional[str] = None
    selectedTools: Optional[list] = None
    doc_id: Optional[str] = None


async def stream_agent_response(
    query: str,
    conversation_id: Optional[str] = None,
    doc_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Stream Deep Agent execution with SSE format.
    Compatible with existing frontend that expects:
    - data: {"steps": [...], "response": null} for progress
    - data: {"steps": [], "response": "...", "canHandle": true} for final answer
    """
    try:
        from deepagents.agents.main_agent import create_medAssist_agent
        
        # Send initial message
        yield f"data: {json.dumps({'steps': ['🚀 Starting Deep Agent...'], 'response': None}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)
        
        # Create agent
        agent = create_medAssist_agent()
        
        yield f"data: {json.dumps({'steps': ['🤖 Agent initialized'], 'response': None}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)
        
        # Prepare input
        input_data = {"messages": [{"role": "user", "content": query}]}
        
        # Add doc_id context if provided
        if doc_id:
            query_with_context = f"{query}\n[Context: Search in document {doc_id}]"
            input_data = {"messages": [{"role": "user", "content": query_with_context}]}
        
        # Execute agent (AgentExecutor doesn't support streaming in the same way)
        yield f"data: {json.dumps({'steps': ['💭 Processing query...'], 'response': None}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)
        
        # Prepare input for AgentExecutor (uses "input" key instead of "messages")
        agent_input = {"input": query}
        
        # Invoke agent synchronously in thread
        result = await asyncio.to_thread(agent.invoke, agent_input)
        
        # Extract response from result
        if isinstance(result, dict):
            final_response = result.get("output", str(result))
        else:
            final_response = str(result)
        
        # Send final response
        final_data = {
            "steps": [],
            "response": final_response,
            "canHandle": True
        }
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        error_msg = f"Error in Deep Agent execution: {str(e)}"
        error_data = {
            "steps": [],
            "response": error_msg,
            "error": error_msg,
            "canHandle": False
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@router.post("/agent-deep")
async def agent_deep(req: DeepAgentRequest):
    """
    Deep Agents endpoint with SSE streaming.
    
    This endpoint runs in parallel with the existing /agent endpoint (smolagents)
    to allow gradual migration and A/B testing.
    """
    return StreamingResponse(
        stream_agent_response(
            query=req.query,
            conversation_id=req.conversationId,
            doc_id=req.doc_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/agent-deep-simple")
async def agent_deep_simple(req: DeepAgentRequest):
    """
    Simple non-streaming endpoint for Deep Agents.
    Returns JSON response directly (useful for testing).
    """
    try:
        from deepagents.agents.main_agent import create_medAssist_agent
        
        agent = create_medAssist_agent()
        
        # Prepare query with doc_id context if provided
        query = req.query
        if req.doc_id:
            query = f"{req.query}\n[Context: Search in document {req.doc_id}]"
        
        # AgentExecutor uses "input" key
        agent_input = {"input": query}
        
        # Invoke agent synchronously
        result = await asyncio.to_thread(agent.invoke, agent_input)
        
        # Extract response (AgentExecutor returns dict with "output" key)
        if isinstance(result, dict):
            response = result.get("output", str(result))
        else:
            response = str(result)
        
        return {
            "response": response,
            "canHandle": True,
            "agent_type": "deep_agents"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-deep/health")
async def health_check():
    """Health check endpoint for Deep Agents."""
    try:
        from deepagents.agents.main_agent import create_medAssist_agent
        
        # Try to create agent
        agent = create_medAssist_agent()
        
        return {
            "status": "healthy",
            "agent_type": "deep_agents",
            "message": "Deep Agents is operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
