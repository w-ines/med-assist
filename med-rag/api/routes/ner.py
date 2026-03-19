"""NER (Named Entity Recognition) endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas.ner import NerExtractRequest

router = APIRouter()


@router.post("/extract")
async def ner_extract(request: NerExtractRequest):
    """Extract medical entities from text."""
    try:
        from ner.router import extract_from_text
        result = extract_from_text(
            request.text,
            entity_types=request.entity_types,
            provider=request.provider,
        )
        return result.to_dict()
    except Exception as e:
        print(f"[ner/extract] error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e), "entities": {}}, status_code=500)
