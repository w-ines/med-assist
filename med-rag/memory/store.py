#save_message(conversation_id, role, content, ts)
#load_history(conversation_id, limit)
#implémentation Redis (et éventuellement “dual write” vers Supabase)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class StoredMessage:
    conversation_id: str
    role: str
    content: str
    created_at: str


_memory_fallback: Dict[str, List[StoredMessage]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    created_at: Optional[str] = None,
    limit_fallback_history: int = 50,
) -> StoredMessage:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    created_at = created_at or _now_iso()
    msg = StoredMessage(
        conversation_id=conversation_id,
        role=role,
        content=content or "",
        created_at=created_at,
    )

    try:
        from memory.repository_supabase import insert_message

        insert_message(
            conversation_id=conversation_id,
            role=role,
            content=content or "",
            created_at=created_at,
        )
    except Exception:
        bucket = _memory_fallback.setdefault(conversation_id, [])
        bucket.append(msg)
        if len(bucket) > limit_fallback_history:
            _memory_fallback[conversation_id] = bucket[-limit_fallback_history:]

    return msg


def load_history(
    *,
    conversation_id: str,
    limit: int = 20,
) -> List[StoredMessage]:
    limit = max(1, min(int(limit), 200))

    try:
        from memory.repository_supabase import fetch_messages

        rows = fetch_messages(conversation_id=conversation_id, limit=limit)
        out: List[StoredMessage] = []
        for r in rows:
            out.append(
                StoredMessage(
                    conversation_id=r.get("conversation_id") or conversation_id,
                    role=r.get("role") or "user",
                    content=r.get("content") or "",
                    created_at=r.get("created_at") or _now_iso(),
                )
            )
        return out
    except Exception:
        return list(_memory_fallback.get(conversation_id, [])[-limit:])
