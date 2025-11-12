"""
Base Agent Interface for LoCoBench-Agent

This module defines the abstract base class and data structures for all agents
in the LoCoBench-Agent evaluation framework.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

from ..core.task import TaskCategory, DifficultyLevel


class MessageRole(Enum):
    """Message roles in agent conversations"""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """Represents a tool call made by an agent"""
    call_id: str
    tool_name: str
    function_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "function_name": self.function_name,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ToolResponse:
    """Represents the response from a tool call"""
    call_id: str
    result: Any
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "result": self.result,
            "success": self.success,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentMessage:
    """Represents a message in the agent conversation"""
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_responses: Optional[List[ToolResponse]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None  # For tool role messages
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in (self.tool_calls or [])],
            "tool_responses": [tr.to_dict() for tr in (self.tool_responses or [])],
            "timestamp": self.timestamp.isoformat(),
            "context_tokens": self.context_tokens,
            "metadata": self.metadata
        }
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class AgentResponse:
    """Response from an agent after processing a turn"""
    message: AgentMessage
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None
    confidence: float = 1.0
    processing_time: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in LoCoBench-Agent
    
    This class defines the interface that all agents must implement,
    whether they are LLM-based agents or framework-based agents.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.conversation_history: List[AgentMessage] = []
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.session_start_time = datetime.now()
        
    @abstractmethod
    async def process_turn(
        self, 
        message: str, 
        available_tools: List['Tool'] = None,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """
        Process a single turn in the conversation
        
        Args:
            message: The user message to process
            available_tools: List of tools available for this turn
            context: Additional context information
            
        Returns:
            AgentResponse containing the agent's response and any tool calls
        """
        pass
    
    @abstractmethod
    async def initialize_session(
        self, 
        scenario_context: Dict[str, Any],
        available_tools: List['Tool'] = None
    ) -> bool:
        """
        Initialize a new evaluation session
        
        Args:
            scenario_context: Context information for the scenario
            available_tools: List of tools available for this session
            
        Returns:
            True if initialization was successful
        """
        pass
    
    @abstractmethod
    async def finalize_session(self) -> Dict[str, Any]:
        """
        Finalize the evaluation session and return summary statistics
        
        Returns:
            Dictionary containing session statistics and metadata
        """
        pass
    
    def add_message_to_history(self, message: AgentMessage) -> None:
        """Add a message to the conversation history"""
        self.conversation_history.append(message)
        self.total_tokens_used += message.context_tokens
    
    def get_conversation_history(self) -> List[AgentMessage]:
        """Get the full conversation history"""
        return self.conversation_history.copy()
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            "agent_name": self.name,
            "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "total_messages": len(self.conversation_history),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "session_start_time": self.session_start_time.isoformat()
        }
    
    def clear_history(self) -> None:
        """Clear conversation history (useful for new sessions)"""
        self.conversation_history.clear()
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.session_start_time = datetime.now()
    
    async def compress_context(self, max_tokens: int = 100000) -> None:
        """
        Compress conversation history to stay within token limits
        
        This is a basic implementation that can be overridden by specific agents
        for more sophisticated context management.
        
        Args:
            max_tokens: Maximum number of tokens to keep in history
        """
        if self.total_tokens_used <= max_tokens:
            return
            
        # Simple compression: keep first message (usually system) and recent messages
        if len(self.conversation_history) > 2:
            # Keep first message (system prompt) and compress middle
            first_message = self.conversation_history[0]
            recent_messages = []
            current_tokens = first_message.context_tokens
            
            # Add messages from the end until we hit the token limit
            for message in reversed(self.conversation_history[1:]):
                if current_tokens + message.context_tokens > max_tokens:
                    break
                recent_messages.insert(0, message)
                current_tokens += message.context_tokens
            
            # Add compression marker
            if len(recent_messages) < len(self.conversation_history) - 1:
                compression_message = AgentMessage(
                    role=MessageRole.SYSTEM,
                    content=f"[CONTEXT COMPRESSED: {len(self.conversation_history) - len(recent_messages) - 1} messages omitted]",
                    context_tokens=10
                )
                self.conversation_history = [first_message, compression_message] + recent_messages
            else:
                self.conversation_history = [first_message] + recent_messages
            
            # Recalculate total tokens
            self.total_tokens_used = sum(msg.context_tokens for msg in self.conversation_history)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"


class AgentCapability(Enum):
    """Capabilities that an agent can have"""
    FUNCTION_CALLING = "function_calling"
    TOOL_USAGE = "tool_usage"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    COLLABORATION = "collaboration"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"
    MEMORY = "memory"


@dataclass
class AgentCapabilities:
    """Describes what capabilities an agent has"""
    supported_capabilities: List[AgentCapability] = field(default_factory=list)
    max_context_tokens: int = 128000
    supports_streaming: bool = False
    supports_function_calling: bool = False
    supports_tool_usage: bool = False
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if the agent has a specific capability"""
        return capability in self.supported_capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported_capabilities": [cap.value for cap in self.supported_capabilities],
            "max_context_tokens": self.max_context_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
            "supports_tool_usage": self.supports_tool_usage,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens
        }
