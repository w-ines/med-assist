from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ner.schemas import NerEntity, NerResult


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass(frozen=True)
class OpenMedNerConfig:
    confidence_threshold: float = 0.7
    group_entities: bool = True
    use_medical_tokenizer: bool = True


def load_openmed_config_from_env() -> OpenMedNerConfig:
    return OpenMedNerConfig(
        confidence_threshold=_env_float("OPENMED_CONFIDENCE_THRESHOLD", 0.7),
        group_entities=_env_bool("OPENMED_GROUP_ENTITIES", True),
        use_medical_tokenizer=_env_bool("OPENMED_USE_MEDICAL_TOKENIZER", True),
    )


DEFAULT_OPENMED_MODELS: Dict[str, str] = {
    "DISEASE": "disease_detection_superclinical",
    "DRUG": "pharma_detection_superclinical",
    "GENE": "genomic_detection",
    "ANATOMY": "anatomy_detection",
}


def models_from_env() -> Dict[str, str]:
    models = dict(DEFAULT_OPENMED_MODELS)
    for k in list(models.keys()):
        override = _env(f"OPENMED_MODEL_{k}", "").strip()
        if override:
            models[k] = override
    return models


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _to_entity(obj: Any) -> NerEntity:
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("entity") or obj.get("label") or ""
        return NerEntity(
            text=_safe_text(text).strip(),
            confidence=obj.get("confidence") or obj.get("score"),
            start=obj.get("start"),
            end=obj.get("end"),
        )

    text = getattr(obj, "text", None) or getattr(obj, "entity", None) or getattr(obj, "label", None)
    score = getattr(obj, "confidence", None)
    if score is None:
        score = getattr(obj, "score", None)

    return NerEntity(
        text=_safe_text(text).strip(),
        confidence=score,
        start=getattr(obj, "start", None),
        end=getattr(obj, "end", None),
    )


def _build_openmed_config(cfg: OpenMedNerConfig):
    from openmed import OpenMedConfig

    return OpenMedConfig(
        use_medical_tokenizer=cfg.use_medical_tokenizer,
        confidence_threshold=cfg.confidence_threshold,
        group_entities=cfg.group_entities,
    )


def extract(
    text: str,
    *,
    entity_types: Optional[Iterable[str]] = None,
    config: Optional[OpenMedNerConfig] = None,
) -> NerResult:
    config = config or load_openmed_config_from_env()
    models = models_from_env()

    requested = [t.strip().upper() for t in (entity_types or models.keys()) if str(t).strip()]
    requested = [t for t in requested if t in models]
    if not requested:
        requested = list(models.keys())

    text = _safe_text(text).strip()
    if not text:
        return NerResult(entities={t: [] for t in requested}, provider="openmed", error=None)

    try:
        from openmed import analyze_text

        om_cfg = _build_openmed_config(config)

        entities: Dict[str, List[NerEntity]] = {t: [] for t in requested}
        for t in requested:
            model_name = models[t]
            results = analyze_text(text, model_name=model_name, config=om_cfg)
            if results is None:
                continue
            entities[t] = [_to_entity(e) for e in results]

        return NerResult(entities=entities, provider="openmed", error=None)
    except Exception as e:
        return NerResult(
            entities={t: [] for t in requested},
            provider="openmed",
            error=f"{type(e).__name__}: {e}",
        )


def extract_batch(
    texts: List[str],
    *,
    entity_types: Optional[Iterable[str]] = None,
    config: Optional[OpenMedNerConfig] = None,
) -> List[NerResult]:
    config = config or load_openmed_config_from_env()
    models = models_from_env()

    requested = [t.strip().upper() for t in (entity_types or models.keys()) if str(t).strip()]
    requested = [t for t in requested if t in models]
    if not requested:
        requested = list(models.keys())

    try:
        from openmed import batch_process

        om_cfg = _build_openmed_config(config)

        per_type_results: Dict[str, List[Any]] = {}
        for t in requested:
            model_name = models[t]
            per_type_results[t] = batch_process(texts, model_name=model_name, config=om_cfg)

        out: List[NerResult] = []
        for idx in range(len(texts)):
            entities_by_type: Dict[str, List[NerEntity]] = {}
            for t in requested:
                batch_res = per_type_results.get(t) or []
                entities = batch_res[idx] if idx < len(batch_res) else []
                entities_by_type[t] = [_to_entity(e) for e in (entities or [])]
            out.append(NerResult(entities=entities_by_type, provider="openmed", error=None))
        return out
    except Exception as e:
        return [extract(t, entity_types=requested, config=config) for t in texts]
