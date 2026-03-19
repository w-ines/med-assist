"""
Signals endpoints — detection and retrieval of emerging signals.
Stub implementation — to be completed when signals/ module is built.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/")
async def list_signals(week: str = None, signal_type: str = None):
    """
    List detected emerging signals.
    
    Query params:
        week: Filter by week label (e.g. "2026-W12")
        signal_type: Filter by type ('emerging', 'accelerating', 'declining', 'contradictory')
    """
    # TODO: Implement when signals/ module is ready
    return {
        "signals": [],
        "message": "Signals module not yet implemented. See PROJECT_SPECIFICATION.md Phase 2."
    }


@router.get("/{signal_id}")
async def get_signal(signal_id: int):
    """Get a specific signal by ID with full details (articles, consensus, timeline)."""
    # TODO: Implement when signals/ module is ready
    raise HTTPException(status_code=501, detail="Signals module not yet implemented")


@router.get("/consensus/{relation_id}")
async def get_consensus(relation_id: int):
    """
    Get consensus score for a specific relation.
    Returns % positive / negative / hypothetical.
    """
    # TODO: Implement when signals/consensus.py is ready
    raise HTTPException(status_code=501, detail="Consensus scoring not yet implemented")
