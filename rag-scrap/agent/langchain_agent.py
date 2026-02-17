#un run(question, conversation_id) qui :
#  charge history depuis memory/store.py
#  appelle une chaîne simple (même sans tools au début)
#  sauvegarde le nouveau tour


from __future__ import annotations

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from memory.store import load_history, save_message


def _build_llm() -> ChatOpenAI:
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPEN_AI_MODEL") or os.getenv("OPENAI_MODEL")
    if not api_key:
        raise RuntimeError("Missing OPEN_ROUTER_KEY or OPENAI_API_KEY")
    if not model:
        raise RuntimeError("Missing OPEN_AI_MODEL / OPENAI_MODEL")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
    )


def _history_to_messages(history_limit: int, conversation_id: str) -> List[BaseMessage]:
    history = load_history(conversation_id=conversation_id, limit=history_limit)
    messages: List[BaseMessage] = []
    for m in history:
        if m.role == "user":
            messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            messages.append(AIMessage(content=m.content))
    return messages


def run(
    *,
    question: str,
    conversation_id: str,
    system_prompt: Optional[str] = None,
    history_limit: int = 20,
) -> str:
    llm = _build_llm()
    system_prompt = system_prompt or (
        "You are a helpful medical assistant. Answer clearly and concisely. "
        "If you are unsure, say so."
    )

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(_history_to_messages(history_limit=history_limit, conversation_id=conversation_id))
    messages.append(HumanMessage(content=question))

    save_message(conversation_id=conversation_id, role="user", content=question)
    resp = llm.invoke(messages)
    answer = getattr(resp, "content", None) or str(resp)
    save_message(conversation_id=conversation_id, role="assistant", content=answer)
    return answer
