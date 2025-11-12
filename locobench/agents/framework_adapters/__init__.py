"""
Agent Framework Adapters for LoCoBench-Agent

This module provides adapters for integrating existing agent frameworks
into the LoCoBench-Agent evaluation system.
"""

from .autogen_adapter import AutoGenAdapter
from .langchain_adapter import LangChainAdapter
from .crewai_adapter import CrewAIAdapter
from .swarm_adapter import SwarmAdapter

__all__ = [
    "AutoGenAdapter",
    "LangChainAdapter", 
    "CrewAIAdapter",
    "SwarmAdapter"
]
