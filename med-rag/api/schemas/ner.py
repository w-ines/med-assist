"""Pydantic schemas for NER endpoints."""

from pydantic import BaseModel
from typing import Optional, List


class NerExtractRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None
    provider: Optional[str] = None
