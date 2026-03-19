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
    enable_assertion: bool = False,
    custom_labels: Optional[List[str]] = None,
    config: Optional[GLiNERNerConfig] = None,
) -> NerResult:
    """
    Extract entities using GLiNER.
    
    Args:
        text: Input text
        entity_types: Entity types to extract
        enable_assertion: Whether to compute assertion status (F2c)
        custom_labels: Custom zero-shot labels (F2b)
        config: GLiNER configuration
    """
    cfg = config or load_gliner_config_from_env()
    extractor = _get_gliner(cfg)

    # Use custom labels if provided (zero-shot mode)
    labels = custom_labels if custom_labels else [str(t).strip() for t in (entity_types or []) if str(t).strip()]
    if not labels:
        labels = ["DISEASE", "DRUG", "GENE", "ANATOMY", "PROTEIN", "CHEMICAL"]

    text = _safe_text(text).strip()
    if not text:
        return NerResult(
            entities={t: [] for t in labels}, 
            provider="gliner", 
            error=None,
            custom_labels=custom_labels,
            assertion_enabled=enable_assertion
        )

    if extractor is None:
        return NerResult(
            entities={t: [] for t in labels},
            provider="gliner",
            error="ImportError: gliner2 is not installed",
            custom_labels=custom_labels,
            assertion_enabled=enable_assertion
        )

    try:
        res = extractor.extract_entities(text, labels)
        entities_raw: Dict[str, List[Any]] = (res or {}).get("entities", {}) or {}

        entities: Dict[str, List[NerEntity]] = {}
        for label in labels:
            values = entities_raw.get(label) or []
            entity_list = []
            for v in values:
                entity_text = _safe_text(v).strip()
                if not entity_text:
                    continue
                
                # Compute assertion status if enabled (F2c)
                assertion = None
                if enable_assertion:
                    assertion = _compute_assertion_status(text, entity_text)
                
                entity_list.append(NerEntity(
                    text=entity_text, 
                    confidence=None,
                    assertion_status=assertion,
                    label=label
                ))
            
            entities[label] = entity_list

        return NerResult(
            entities=entities, 
            provider="gliner", 
            error=None,
            custom_labels=custom_labels,
            assertion_enabled=enable_assertion
        )
    except Exception as e:
        return NerResult(
            entities={t: [] for t in labels},
            provider="gliner",
            error=f"{type(e).__name__}: {e}",
            custom_labels=custom_labels,
            assertion_enabled=enable_assertion
        )


def _compute_assertion_status(text: str, entity: str) -> str:
    """
    Simple heuristic-based assertion status detection (F2c).
    
    For production, this should use OpenMed Assertion model.
    This is a basic implementation for MVP.
    """
    text_lower = text.lower()
    entity_lower = entity.lower()
    
    # Find entity context (sentence containing the entity)
    sentences = text.split('.')
    context = ""
    for sent in sentences:
        if entity_lower in sent.lower():
            context = sent.lower()
            break
    
    if not context:
        return "PRESENT"
    
    # Negation patterns
    negation_patterns = [
        "no ", "not ", "without ", "absence of", "lack of", 
        "negative for", "ruled out", "excluded", "denied"
    ]
    for pattern in negation_patterns:
        if pattern in context:
            return "NEGATED"
    
    # Hypothetical patterns
    hypothetical_patterns = [
        "may ", "might ", "could ", "would ", "should ",
        "potential", "possible", "hypothesis", "suggest",
        "further studies", "needs to be", "remains to be"
    ]
    for pattern in hypothetical_patterns:
        if pattern in context:
            return "HYPOTHETICAL"
    
    # Historical patterns
    historical_patterns = [
        "history of", "previous", "prior", "past", "former",
        "previously", "had been", "was diagnosed"
    ]
    for pattern in historical_patterns:
        if pattern in context:
            return "HISTORICAL"
    
    return "PRESENT"


def extract_batch(
    texts: List[str],
    *,
    entity_types: Optional[Iterable[str]] = None,
    config: Optional[GLiNERNerConfig] = None,
) -> List[NerResult]:
    return [extract(t, entity_types=entity_types, config=config) for t in texts]
