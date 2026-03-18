"""
Deep Agents module for medAssist - Biomedical Intelligence Agent.

This module provides a LangChain Deep Agents implementation to replace smolagents,
with enhanced capabilities for biomedical research, RAG, and Knowledge Graph integration.
"""

from deepagents.agents.main_agent import create_medAssist_agent
from deepagents.router import router as deepagent_router

__all__ = [
    "create_medAssist_agent",
    "deepagent_router",
]
