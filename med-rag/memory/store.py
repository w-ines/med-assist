#save_message(conversation_id, role, content, ts)
#load_history(conversation_id, limit)
#implémentation Redis (et éventuellement “dual write” vers Supabase)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4


def _is_network_error(err: BaseException) -> bool:
    try:
        import httpx

        return isinstance(err, httpx.RequestError)
    except Exception:
        return False


def _is_missing_supabase_table_error(err: BaseException) -> bool:
    """Return True when PostgREST reports missing schema cache/table.

    This happens when the Supabase project does not have the expected table
    (default: public.conversation_messages).
    """

    try:
        # postgrest.exceptions.APIError typically stores a dict in args[0]
        args = getattr(err, "args", None)
        if args and isinstance(args[0], dict):
            code = args[0].get("code")
            if code == "PGRST205":
                return True
        if "PGRST205" in str(err):
            return True
    except Exception:
        return False
    return False


@dataclass
class StoredMessage:
    message_id: str
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
        message_id=str(uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content or "",
        created_at=created_at,
    )

    try:
        from memory.repository_supabase import SupabaseNotConfigured, insert_message

        insert_message(
            message_id=msg.message_id,
            conversation_id=conversation_id,
            role=role,
            content=content or "",
            created_at=created_at,
        )

        try:
            flush_fallback_to_supabase(conversation_id=conversation_id)
        except (SupabaseNotConfigured,) as e:
            pass
        except Exception as e:
            if _is_network_error(e):
                pass
            else:
                raise
    except (SupabaseNotConfigured,) as e:
        bucket = _memory_fallback.setdefault(conversation_id, [])
        bucket.append(msg)
        if len(bucket) > limit_fallback_history:
            _memory_fallback[conversation_id] = bucket[-limit_fallback_history:]
    except Exception as e:
        if _is_missing_supabase_table_error(e):
            bucket = _memory_fallback.setdefault(conversation_id, [])
            bucket.append(msg)
            if len(bucket) > limit_fallback_history:
                _memory_fallback[conversation_id] = bucket[-limit_fallback_history:]
            return msg

        if not _is_network_error(e):
            raise
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
        from memory.repository_supabase import SupabaseNotConfigured, fetch_messages

        rows = fetch_messages(conversation_id=conversation_id, limit=limit)
        out: List[StoredMessage] = []
        for r in rows:
            out.append(
                StoredMessage(
                    message_id=r.get("message_id") or str(uuid4()),
                    conversation_id=r.get("conversation_id") or conversation_id,
                    role=r.get("role") or "user",
                    content=r.get("content") or "",
                    created_at=r.get("created_at") or _now_iso(),
                )
            )
        return out
    except (SupabaseNotConfigured,) as e:
        return list(_memory_fallback.get(conversation_id, [])[-limit:])
    except Exception as e:
        if _is_missing_supabase_table_error(e):
            return list(_memory_fallback.get(conversation_id, [])[-limit:])
        if not _is_network_error(e):
            raise
        return list(_memory_fallback.get(conversation_id, [])[-limit:])


def flush_fallback_to_supabase(
    *,
    conversation_id: Optional[str] = None,
    max_messages: Optional[int] = None,
) -> int:
    """Try to persist in-memory fallback messages to Supabase.

    Returns the number of messages successfully flushed.
    """

    from memory.repository_supabase import SupabaseNotConfigured, insert_message

    flushed = 0
    conversation_ids = [conversation_id] if conversation_id else list(_memory_fallback.keys())

    for cid in conversation_ids:
        bucket = _memory_fallback.get(cid)
        if not bucket:
            continue

        to_flush = bucket if max_messages is None else bucket[: max(0, int(max_messages))]
        remaining = bucket[len(to_flush) :]

        for msg in to_flush:
            try:
                insert_message(
                    message_id=msg.message_id,
                    conversation_id=msg.conversation_id,
                    role=msg.role,
                    content=msg.content,
                    created_at=msg.created_at,
                )
                flushed += 1
            except (SupabaseNotConfigured,) as e:
                return flushed
            except Exception as e:
                if _is_missing_supabase_table_error(e):
                    return flushed
                if _is_network_error(e):
                    return flushed
                raise

        if remaining:
            _memory_fallback[cid] = remaining
        else:
            _memory_fallback.pop(cid, None)

    return flushed
