from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ner.schemas import NerEntity, NerResult


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


@dataclass(frozen=True)
class GLiNERNerConfig:
    model_name: str = "fastino/gliner2-base-v1"
 

_gliner_instance = None
_gliner_available: Optional[bool] = None


def _check_gliner_available() -> bool:
    global _gliner_available
    if _gliner_available is None:
        try:
            from gliner2 import GLiNER2

            _gliner_available = True
        except ImportError:
            _gliner_available = False
    return bool(_gliner_available)


def _get_gliner(cfg: GLiNERNerConfig):
    global _gliner_instance
    if not _check_gliner_available():
        return None

    if _gliner_instance is None:
        from gliner2 import GLiNER2

        _gliner_instance = GLiNER2.from_pretrained(cfg.model_name)

    return _gliner_instance


def load_gliner_config_from_env() -> GLiNERNerConfig:
    model_name = _env("GLINER_MODEL_NAME", "fastino/gliner2-base-v1").strip() or "fastino/gliner2-base-v1"
    return GLiNERNerConfig(model_name=model_name)


def extract(
    text: str,
    *,
    entity_types: Optional[Iterable[str]] = None,
    config: Optional[GLiNERNerConfig] = None,
) -> NerResult:
    cfg = config or load_gliner_config_from_env()
    extractor = _get_gliner(cfg)

    labels = [str(t).strip() for t in (entity_types or []) if str(t).strip()]
    if not labels:
        labels = ["DISEASE", "DRUG", "GENE", "ANATOMY"]

    text = _safe_text(text).strip()
    if not text:
        return NerResult(entities={t: [] for t in labels}, provider="gliner", error=None)

    if extractor is None:
        return NerResult(
            entities={t: [] for t in labels},
            provider="gliner",
            error="ImportError: gliner2 is not installed",
        )

    try:
        res = extractor.extract_entities(text, labels)
        entities_raw: Dict[str, List[Any]] = (res or {}).get("entities", {}) or {}

        entities: Dict[str, List[NerEntity]] = {}
        for label in labels:
            values = entities_raw.get(label) or []
            entities[label] = [NerEntity(text=_safe_text(v).strip(), confidence=None) for v in values if _safe_text(v).strip()]

        return NerResult(entities=entities, provider="gliner", error=None)
    except Exception as e:
        return NerResult(
            entities={t: [] for t in labels},
            provider="gliner",
            error=f"{type(e).__name__}: {e}",
        )


def extract_batch(
    texts: List[str],
    *,
    entity_types: Optional[Iterable[str]] = None,
    config: Optional[GLiNERNerConfig] = None,
) -> List[NerResult]:
    return [extract(t, entity_types=entity_types, config=config) for t in texts]
