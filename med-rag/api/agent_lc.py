#endpoint SSE /agent-lc qui stream des events (même “fake steps” au début)

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.langchain_agent import run as run_langchain


router = APIRouter()


class AgentLCRequest(BaseModel):
    query: str
    conversationId: Optional[str] = None


async def _sse_events(*, req: AgentLCRequest) -> AsyncGenerator[str, None]:
    conversation_id = req.conversationId or "default"
    query = (req.query or "").strip()

    yield f"data: {json.dumps({'steps': ['🧠 Loading conversation memory...'], 'response': None}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0)

    yield f"data: {json.dumps({'steps': ['🤖 Calling LLM...'], 'response': None}, ensure_ascii=False)}\n\n"

    answer = await asyncio.to_thread(
        run_langchain,
        question=query,
        conversation_id=conversation_id,
    )

    yield f"data: {json.dumps({'steps': ['💾 Saving messages...'], 'response': None}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0)

    yield f"data: {json.dumps({'steps': [], 'response': answer, 'canHandle': True}, ensure_ascii=False)}\n\n"


@router.post("/agent-lc")
async def agent_lc(req: AgentLCRequest):
    return StreamingResponse(
        _sse_events(req=req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )