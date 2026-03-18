from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional
from ner import router

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _entity_to_dict(entity: Any) -> Dict[str, Any]:
    if isinstance(entity, dict):
        text = entity.get("text") or entity.get("entity") or entity.get("label") or ""
        out: Dict[str, Any] = {
            "text": _safe_text(text).strip(),
            "confidence": entity.get("confidence") or entity.get("score"),
        }
        if "start" in entity:
            out["start"] = entity["start"]
        if "end" in entity:
            out["end"] = entity["end"]
        return out

    text = getattr(entity, "text", None) or getattr(entity, "entity", None) or getattr(entity, "label", None)
    score = getattr(entity, "confidence", None)
    if score is None:
        score = getattr(entity, "score", None)

    out = {
        "text": _safe_text(text).strip(),
        "confidence": score,
    }
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    if start is not None:
        out["start"] = start
    if end is not None:
        out["end"] = end
    return out


def extract_medical_entities_from_text(
    text: str,
    *,
    entity_types: Optional[Iterable[str]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract medical entities from a free text.

    Returns a dict:
      {"entities": {"DISEASE": [...], ...}, "provider": str, "error": str|None}
    """

    text = _safe_text(text).strip()
    out = router.extract_from_text(text, entity_types=entity_types, provider=provider)
    payload = out.to_dict()
    entities = payload.get("entities") or {}
    if isinstance(entities, dict):
        entities = {k: [_entity_to_dict(e) for e in (v or [])] for k, v in entities.items()}
    return {
        "entities": entities,
        "provider": payload.get("provider") or out.provider,
        "error": payload.get("error"),
    }


def extract_medical_entities_from_article(
    article: Mapping[str, Any],
    *,
    entity_types: Optional[Iterable[str]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract entities from an article dict (expects title/abstract/pmid keys)."""

    return router.extract_from_article(article, entity_types=entity_types, provider=provider)


def extract_medical_entities_batch(
    articles: List[Mapping[str, Any]],
    *,
    entity_types: Optional[Iterable[str]] = None,
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Batch extraction.

    Tries OpenMed batch API first; if not available, falls back to per-article extraction.
    """

    return router.extract_batch(articles, entity_types=entity_types, provider=provider)
