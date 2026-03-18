from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ner.schemas import NerResult


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _select_provider(provider: Optional[str] = None) -> str:
    raw = (provider or _env("NER_PROVIDER", "openmed")).strip().lower()
    return raw or "openmed"


def extract_from_text(
    text: str,
    *,
    entity_types: Optional[Iterable[str]] = None,
    provider: Optional[str] = None,
) -> NerResult:
    p = _select_provider(provider)

    if p == "openmed":
        from ner.backends import openmed_backend

        return openmed_backend.extract(text, entity_types=entity_types)

    if p == "gliner":
        from ner.backends import gliner_backend

        return gliner_backend.extract(text, entity_types=entity_types)

    return NerResult(
        entities={str(t).strip().upper(): [] for t in (entity_types or []) if str(t).strip()},
        provider=p,
        error=f"ValueError: Unknown NER provider '{p}'",
    )


def extract_from_article(
    article: Mapping[str, Any],
    *,
    entity_types: Optional[Iterable[str]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    title = _safe_text(article.get("title"))
    abstract = _safe_text(article.get("abstract"))
    pmid = _safe_text(article.get("pmid"))

    text = (title + "\n\n" + abstract).strip()
    out = extract_from_text(text, entity_types=entity_types, provider=provider)

    result: Dict[str, Any] = {
        "pmid": pmid,
        "title": title,
        "entities": out.to_dict().get("entities", {}),
        "provider": out.provider,
    }
    if out.error:
        result["error"] = out.error
    return result


def extract_batch(
    articles: List[Mapping[str, Any]],
    *,
    entity_types: Optional[Iterable[str]] = None,
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:
    texts: List[str] = []
    for a in articles:
        title = _safe_text(a.get("title"))
        abstract = _safe_text(a.get("abstract"))
        texts.append((title + "\n\n" + abstract).strip())

    p = _select_provider(provider)
    if p == "openmed":
        from ner.backends import openmed_backend

        results = openmed_backend.extract_batch(texts, entity_types=entity_types)
    elif p == "gliner":
        from ner.backends import gliner_backend

        results = gliner_backend.extract_batch(texts, entity_types=entity_types)
    else:
        results = [extract_from_text(t, entity_types=entity_types, provider=p) for t in texts]

    out: List[Dict[str, Any]] = []
    for idx, article in enumerate(articles):
        pmid = _safe_text(article.get("pmid"))
        title = _safe_text(article.get("title"))
        r = results[idx] if idx < len(results) else None
        if r is None:
            out.append({"pmid": pmid, "title": title, "entities": {}, "provider": p, "error": "IndexError: missing batch result"})
            continue

        payload = r.to_dict()
        item: Dict[str, Any] = {
            "pmid": pmid,
            "title": title,
            "entities": payload.get("entities", {}),
            "provider": payload.get("provider", p),
        }
        if payload.get("error"):
            item["error"] = payload["error"]
        out.append(item)

    return out
