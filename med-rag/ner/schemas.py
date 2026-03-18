from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class NerEntity:
    text: str
    confidence: Optional[float] = None
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.start is not None:
            out["start"] = self.start
        if self.end is not None:
            out["end"] = self.end
        return out


@dataclass(frozen=True)
class NerResult:
    entities: Dict[str, List[NerEntity]]
    provider: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: [e.to_dict() for e in v] for k, v in self.entities.items()},
            "provider": self.provider,
            "error": self.error,
        }


TextLike = Any
ArticleLike = Mapping[str, Any]
EntityTypes = Optional[Iterable[str]]
