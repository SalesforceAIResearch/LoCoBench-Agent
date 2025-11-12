"""
LoCoBench Agent System

This module provides agent interfaces and implementations for evaluating
LLM agents in long-context software development scenarios.
"""

from .base_agent import BaseAgent, AgentResponse, AgentMessage, ToolCall, ToolResponse
from .agent_factory import AgentFactory, AgentConfig, AgentType
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent
from .google_agent import GoogleAgent
from .custom_agent import CustomAgent, TemplateCustomAgent, ResearchCustomAgent, CustomAgentFactory
# Framework adapters
from .framework_adapters.autogen_adapter import AutoGenAdapter
from .framework_adapters.langchain_adapter import LangChainAdapter
from .framework_adapters.crewai_adapter import CrewAIAdapter
from .framework_adapters.swarm_adapter import SwarmAdapter

__all__ = [
    "BaseAgent",
    "AgentResponse", 
    "AgentMessage",
    "ToolCall",
    "ToolResponse",
    "AgentFactory",
    "AgentConfig",
    "AgentType",
    "OpenAIAgent",
    "AnthropicAgent",
    "GoogleAgent",
    "CustomAgent",
    "TemplateCustomAgent",
    "ResearchCustomAgent",
    "CustomAgentFactory",
    "AutoGenAdapter",
    "LangChainAdapter",
    "CrewAIAdapter",
    "SwarmAdapter"
]
