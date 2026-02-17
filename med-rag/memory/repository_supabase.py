from __future__ import annotations

from typing import Any, Dict, List, Optional


def _get_supabase_client():
    """Return a Supabase client suitable for server-side inserts.

    Prefers SUPABASE_SERVICE_ROLE_KEY when available to avoid RLS issues.
    """

    import os
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY")

    return create_client(url, key)


def insert_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    created_at: Optional[str] = None,
    table: str = "conversation_messages",
) -> Dict[str, Any]:
    client = _get_supabase_client()
    payload: Dict[str, Any] = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }
    if created_at:
        payload["created_at"] = created_at

    res = client.table(table).insert(payload).execute()
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
    client = _get_supabase_client()
    limit = max(1, min(int(limit), 200))
    res = (
        client.table(table)
        .select("conversation_id,role,content,created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    data = getattr(res, "data", None)
    return data if isinstance(data, list) else []
