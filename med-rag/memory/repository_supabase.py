from __future__ import annotations

from typing import Any, Dict, List, Optional

from storage.supabase_client import SupabaseNotConfigured, get_supabase_client


def insert_message(
    *,
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    created_at: Optional[str] = None,
    table: str = "conversation_messages",
) -> Dict[str, Any]:
    client = get_supabase_client()
    payload: Dict[str, Any] = {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }
    if created_at:
        payload["created_at"] = created_at

    res = client.table(table).upsert(payload, on_conflict="message_id").execute()
    data = getattr(res, "data", None)
    if isinstance(data, list) and data:
        return data[0]
    return payload


def fetch_messages(
    *,
    conversation_id: str,
    limit: int = 20,
    table: str = "conversation_messages",
) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    limit = max(1, min(int(limit), 200))
    res = (
        client.table(table)
        .select("message_id,conversation_id,role,content,created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    data = getattr(res, "data", None)
    return data if isinstance(data, list) else []
