"""
Agent-Environment Interaction Protocol for LoCoBench-Agent

This module defines the communication protocol between agents and the evaluation
environment, including message formats, tool calling conventions, and response handling.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the interaction protocol"""
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR_MESSAGE = "error_message"
    STATUS_UPDATE = "status_update"


class MessageRole(Enum):
    """Roles in the conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class InteractionMessage:
    """Base message in the interaction protocol"""
    
    message_id: str
    message_type: MessageType
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_message_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "parent_message_id": self.parent_message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionMessage':
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            parent_message_id=data.get("parent_message_id")
        )


@dataclass
class ToolCallMessage(InteractionMessage):
    """Message representing a tool call"""
    
    tool_name: str = ""
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    
    def __post_init__(self):
        self.message_type = MessageType.TOOL_CALL
        self.role = MessageRole.ASSISTANT
        
        # Set content to JSON representation of tool call
        self.content = json.dumps({
            "tool_name": self.tool_name,
            "arguments": self.tool_arguments,
            "call_id": self.call_id
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCallMessage':
        content_data = json.loads(data["content"])
        
        return cls(
            message_id=data["message_id"],
            tool_name=content_data["tool_name"],
            tool_arguments=content_data["arguments"],
            call_id=content_data["call_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            parent_message_id=data.get("parent_message_id")
        )


@dataclass
class ToolResponseMessage(InteractionMessage):
    """Message representing a tool response"""
    
    call_id: str = ""
    tool_name: str = ""
    result: Any = None
    success: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        self.message_type = MessageType.TOOL_RESPONSE
        self.role = MessageRole.TOOL
        
        # Set content to JSON representation of tool response
        self.content = json.dumps({
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "result": self.result,
            "success": self.success,
            "error_message": self.error_message
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResponseMessage':
        content_data = json.loads(data["content"])
        
        return cls(
            message_id=data["message_id"],
            call_id=content_data["call_id"],
            tool_name=content_data["tool_name"],
            result=content_data["result"],
            success=content_data["success"],
            error_message=content_data.get("error_message"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            parent_message_id=data.get("parent_message_id")
        )


class InteractionProtocolHandler(ABC):
    """Abstract base class for handling interaction protocol"""
    
    @abstractmethod
    async def process_message(self, message: InteractionMessage) -> Optional[InteractionMessage]:
        """Process an incoming message and optionally return a response"""
        pass
    
    @abstractmethod
    async def handle_tool_call(self, tool_call: ToolCallMessage) -> ToolResponseMessage:
        """Handle a tool call and return the response"""
        pass
    
    @abstractmethod
    def validate_message(self, message: InteractionMessage) -> bool:
        """Validate a message according to protocol rules"""
        pass


@dataclass
class ConversationState:
    """Maintains the state of a conversation"""
    
    conversation_id: str
    messages: List[InteractionMessage] = field(default_factory=list)
    active_tool_calls: Dict[str, ToolCallMessage] = field(default_factory=dict)
    
    # Conversation metadata
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0
    total_cost: float = 0.0
    
    def add_message(self, message: InteractionMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Track tool calls
        if isinstance(message, ToolCallMessage):
            self.active_tool_calls[message.call_id] = message
        elif isinstance(message, ToolResponseMessage):
            # Remove completed tool call
            if message.call_id in self.active_tool_calls:
                del self.active_tool_calls[message.call_id]
    
    def get_messages_by_type(self, message_type: MessageType) -> List[InteractionMessage]:
        """Get all messages of a specific type"""
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage in this conversation"""
        tool_calls = self.get_messages_by_type(MessageType.TOOL_CALL)
        tool_responses = self.get_messages_by_type(MessageType.TOOL_RESPONSE)
        
        tool_stats = {}
        
        for call in tool_calls:
            if isinstance(call, ToolCallMessage):
                tool_name = call.tool_name
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "calls": 0,
                        "successful": 0,
                        "failed": 0
                    }
                tool_stats[tool_name]["calls"] += 1
        
        for response in tool_responses:
            if isinstance(response, ToolResponseMessage):
                tool_name = response.tool_name
                if tool_name in tool_stats:
                    if response.success:
                        tool_stats[tool_name]["successful"] += 1
                    else:
                        tool_stats[tool_name]["failed"] += 1
        
        return {
            "total_tool_calls": len(tool_calls),
            "total_tool_responses": len(tool_responses),
            "active_calls": len(self.active_tool_calls),
            "tool_breakdown": tool_stats
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "tool_usage_stats": self.get_tool_usage_stats()
        }


class AgentEnvironmentProtocol(InteractionProtocolHandler):
    """
    Implementation of the interaction protocol for agent-environment communication
    
    This class manages the communication between agents and the evaluation environment,
    handling message routing, tool execution, and state management.
    """
    
    def __init__(self, tool_registry=None):
        from .tool_registry import get_tool_registry
        self.tool_registry = tool_registry or get_tool_registry()
        
        # Active conversations
        self.conversations: Dict[str, ConversationState] = {}
        
        logger.info("AgentEnvironmentProtocol initialized")
    
    def create_conversation(self, conversation_id: str) -> ConversationState:
        """Create a new conversation"""
        conversation = ConversationState(conversation_id=conversation_id)
        self.conversations[conversation_id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get an existing conversation"""
        return self.conversations.get(conversation_id)
    
    async def process_message(self, message: InteractionMessage) -> Optional[InteractionMessage]:
        """Process an incoming message"""
        
        if not self.validate_message(message):
            return InteractionMessage(
                message_id=f"error_{message.message_id}",
                message_type=MessageType.ERROR_MESSAGE,
                role=MessageRole.SYSTEM,
                content="Invalid message format",
                parent_message_id=message.message_id
            )
        
        # Handle different message types
        if message.message_type == MessageType.TOOL_CALL:
            if isinstance(message, ToolCallMessage):
                return await self.handle_tool_call(message)
        
        # For other message types, just acknowledge
        return None
    
    async def handle_tool_call(self, tool_call: ToolCallMessage) -> ToolResponseMessage:
        """Handle a tool call and return the response"""
        
        try:
            # Get the tool
            tool = self.tool_registry.get_tool(tool_call.tool_name)
            
            if not tool:
                return ToolResponseMessage(
                    message_id=f"response_{tool_call.call_id}",
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    result=None,
                    success=False,
                    error_message=f"Tool '{tool_call.tool_name}' not found",
                    parent_message_id=tool_call.message_id
                )
            
            # Execute the tool
            result = await tool.execute(tool_call.tool_arguments)
            
            return ToolResponseMessage(
                message_id=f"response_{tool_call.call_id}",
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                result=result,
                success=True,
                parent_message_id=tool_call.message_id
            )
            
        except Exception as e:
            # Handle FileNotFoundError as debug (expected when agents request files not in scenario context)
            if isinstance(e, FileNotFoundError):
                logger.debug(f"Tool {tool_call.tool_name} file not found (expected in scenario context): {e}")
            else:
                logger.error(f"Error executing tool {tool_call.tool_name}: {e}")
            
            return ToolResponseMessage(
                message_id=f"response_{tool_call.call_id}",
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                result=None,
                success=False,
                error_message=str(e),
                parent_message_id=tool_call.message_id
            )
    
    def validate_message(self, message: InteractionMessage) -> bool:
        """Validate a message according to protocol rules"""
        
        # Basic validation
        if not message.message_id or not message.content:
            return False
        
        # Type-specific validation
        if message.message_type == MessageType.TOOL_CALL:
            if not isinstance(message, ToolCallMessage):
                return False
            
            # Validate tool call structure
            if not message.tool_name or not message.call_id:
                return False
        
        elif message.message_type == MessageType.TOOL_RESPONSE:
            if not isinstance(message, ToolResponseMessage):
                return False
            
            # Validate tool response structure
            if not message.call_id or not message.tool_name:
                return False
        
        return True
    
    def get_conversation_history(
        self,
        conversation_id: str,
        message_types: List[MessageType] = None
    ) -> List[InteractionMessage]:
        """Get conversation history with optional filtering"""
        
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages
        
        if message_types:
            messages = [msg for msg in messages if msg.message_type in message_types]
        
        return messages
    
    def format_conversation_for_llm(self, conversation_id: str) -> List[Dict[str, str]]:
        """Format conversation history for LLM consumption"""
        
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        formatted_messages = []
        
        for message in conversation.messages:
            # Skip internal messages
            if message.message_type in [MessageType.STATUS_UPDATE, MessageType.ERROR_MESSAGE]:
                continue
            
            # Format based on role
            if message.role == MessageRole.USER:
                formatted_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif message.role == MessageRole.ASSISTANT:
                formatted_messages.append({
                    "role": "assistant", 
                    "content": message.content
                })
            elif message.role == MessageRole.SYSTEM:
                formatted_messages.append({
                    "role": "system",
                    "content": message.content
                })
            elif message.role == MessageRole.TOOL:
                # Tool responses are typically embedded in assistant messages
                # or handled separately by the LLM provider
                if isinstance(message, ToolResponseMessage):
                    formatted_messages.append({
                        "role": "tool",
                        "content": json.dumps(message.result),
                        "tool_call_id": message.call_id
                    })
        
        return formatted_messages
    
    async def cleanup_conversation(self, conversation_id: str):
        """Clean up a completed conversation"""
        
        if conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            
            # Log final statistics
            stats = conversation.get_tool_usage_stats()
            logger.info(f"Conversation {conversation_id} completed. Tool calls: {stats['total_tool_calls']}")
            
            # Remove from active conversations
            del self.conversations[conversation_id]
    
    def get_protocol_statistics(self) -> Dict[str, Any]:
        """Get overall protocol usage statistics"""
        
        total_conversations = len(self.conversations)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())
        total_tool_calls = sum(
            conv.get_tool_usage_stats()["total_tool_calls"] 
            for conv in self.conversations.values()
        )
        
        return {
            "active_conversations": total_conversations,
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "conversations": {
                conv_id: {
                    "messages": len(conv.messages),
                    "duration_seconds": (conv.last_activity - conv.start_time).total_seconds(),
                    "tool_usage": conv.get_tool_usage_stats()
                }
                for conv_id, conv in self.conversations.items()
            }
        }
