"""
OpenAI Agent Implementation for LoCoBench-Agent

This module implements agents that use OpenAI's function calling capabilities
to interact with tools in the LoCoBench-Agent evaluation framework.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None

from .base_agent import (
    BaseAgent, AgentResponse, AgentMessage, MessageRole, 
    ToolCall, ToolResponse, AgentCapabilities, AgentCapability
)
from ..core.tool_registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    provider: str = "OpenAI"
):
    """
    Retry a function with exponential backoff for rate limit and transient errors.
    
    This ensures that API calls are resilient to:
    - Rate limit errors (429)
    - Temporary network issues
    - Server errors (500, 502, 503, 504)
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay cap in seconds
        provider: API provider name for logging
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is a retryable error
            is_rate_limit = any(pattern in error_str for pattern in [
                "rate limit", "rate_limit", "ratelimit", "429", "too many requests"
            ])
            is_server_error = any(pattern in error_str for pattern in [
                "500", "502", "503", "504", "internal error", "server error",
                "connection", "timeout", "network"
            ])
            is_retryable = is_rate_limit or is_server_error
            
            if is_retryable and attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                
                error_type = "Rate limit" if is_rate_limit else "Server error"
                logger.warning(
                    f"{provider} {error_type} on attempt {attempt + 1}/{max_retries}. "
                    f"Retrying in {delay:.1f}s... Error: {str(e)[:100]}"
                )
                
                await asyncio.sleep(delay)
                continue
            else:
                # Non-retryable error or max retries exceeded
                if attempt == max_retries - 1:
                    logger.error(
                        f"{provider} API call failed after {max_retries} attempts. "
                        f"Final error: {str(e)}"
                    )
                raise
    
    raise Exception(f"{provider} API call failed after {max_retries} attempts")


class OpenAIAgent(BaseAgent):
    """
    Agent implementation using OpenAI's function calling capabilities
    """
    
    # Model configurations
    # Updated: October 2025 with latest OpenAI models
    MODEL_CONFIGS = {
        # GPT-5 Series (Released August 2025)
        "gpt-5": {
            "max_tokens": 200000,
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",  # Fallback until GPT-5 encoding released
            "use_max_completion_tokens": True,  # GPT-5 uses new parameter name
            "fixed_temperature": 1.0,  # GPT-5 only supports temperature=1 (cannot customize)
        },
        "gpt-5-mini": {
            "max_tokens": 200000,
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",
            "use_max_completion_tokens": True,
            "fixed_temperature": 1.0,  # GPT-5 only supports temperature=1 (cannot customize)
        },
        # Note: gpt-5-codex uses different API endpoint (v1/responses), not supported in chat completions
        
        # GPT-4.1 Series (Released April 2025)
        "gpt-4.1": {
            "max_tokens": 1000000,  # 1M token context window
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",
            "use_max_completion_tokens": True,
            # Temperature 0-2 supported, default 1.0 (customizable)
        },
        "gpt-4.1-mini": {
            "max_tokens": 1000000,  # 1M token context window
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",
            "use_max_completion_tokens": True,
            # Temperature 0-2 supported, default 1.0 (customizable)
        },
        
        # GPT-4o Series (2024)
        "gpt-4o": {
            "max_tokens": 128000,
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",
        },
        "gpt-4o-mini": {
            "max_tokens": 128000,
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",
        },
        
        # GPT-4 Series (Legacy)
        "gpt-4-turbo": {
            "max_tokens": 128000,
            "supports_functions": True,
            "tiktoken_encoding": "cl100k_base",
        },
        
        # o-Series Reasoning Models
        "o3": {
            "max_tokens": 200000,
            "supports_functions": True,  # o3 supports function calling
            "tiktoken_encoding": "o200k_base",
            "use_max_completion_tokens": True,
            "omit_temperature": True,  # o-series does NOT support temperature parameter at all
        },
        "o3-pro": {
            "max_tokens": 200000,
            "supports_functions": True,
            "tiktoken_encoding": "o200k_base",
            "use_max_completion_tokens": True,
            "omit_temperature": True,  # o-series does NOT support temperature parameter at all
        },
        "o3-mini": {
            "max_tokens": 200000,
            "supports_functions": True,
            "tiktoken_encoding": "o200k_base",
            "use_max_completion_tokens": True,
            "omit_temperature": True,  # o-series does NOT support temperature parameter at all
        },
        "o4-mini": {
            "max_tokens": 200000,
            "supports_functions": True,
            "tiktoken_encoding": "o200k_base",
            "use_max_completion_tokens": True,
            "omit_temperature": True,  # o-series does NOT support temperature parameter at all
        },
        
        # o1 Series (Legacy - Limited function calling)
        "o1-preview": {
            "max_tokens": 128000,
            "supports_functions": False,  # o1 models have limited function calling
            "tiktoken_encoding": "o200k_base",
            "omit_temperature": True,  # o1 series does NOT support temperature parameter
        },
        "o1-mini": {
            "max_tokens": 128000,
            "supports_functions": False,
            "tiktoken_encoding": "o200k_base",
            "omit_temperature": True,  # o1 series does NOT support temperature parameter
        }
    }
    
    def __init__(
        self,
        name: str = "OpenAI Agent",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        config: Dict[str, Any] = None,
        base_url: Optional[str] = None
    ):
        if openai is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        super().__init__(name, config)
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_config = self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS["gpt-4o"])
        
        # Initialize OpenAI client with optional custom base URL (for gateway support)
        # Gateway URL format: https://gateway.salesforceresearch.ai/openai/process/v1/
        client_kwargs = {}
        
        if base_url:
            # Salesforce Research Gateway requires X-Api-Key header and dummy api_key
            # See: https://gateway.salesforceresearch.ai/docs#/openai
            client_kwargs["base_url"] = base_url
            client_kwargs["api_key"] = "dummy"  # Required by OpenAI SDK
            client_kwargs["default_headers"] = {"X-Api-Key": api_key}
            logger.info(f"Using OpenAI Gateway: {base_url} with X-Api-Key authentication")
        else:
            # Direct OpenAI API uses standard api_key
            client_kwargs["api_key"] = api_key
        
        self.client = AsyncOpenAI(**client_kwargs)
        
        # Tool registry for function calling
        self.tool_registry: Optional[ToolRegistry] = None
        self.available_tools: List[Tool] = []
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.FUNCTION_CALLING,
                AgentCapability.CODE_GENERATION,
                AgentCapability.REASONING,
                AgentCapability.LONG_CONTEXT
            ],
            max_context_tokens=self.model_config["max_tokens"],
            supports_function_calling=self.model_config["supports_functions"],
            supports_tool_usage=self.model_config["supports_functions"]
        )
        
        logger.info(f"Initialized OpenAI agent with model {model}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize a new evaluation session"""
        try:
            self.available_tools = available_tools or []
            
            # Test OpenAI connection
            await self._test_connection()
            
            logger.info(f"OpenAI agent session initialized with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI agent session: {e}")
            return False
    
    async def process_turn(
        self,
        message: str,
        available_tools: List[Tool] = None,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process a single turn in the conversation"""
        start_time = time.time()
        
        try:
            # Update available tools if provided
            if available_tools is not None:
                self.available_tools = available_tools
            
            # Prepare messages for OpenAI
            messages = self._prepare_messages(message, context)
            
            # Prepare functions for function calling
            functions = self._prepare_functions() if self.model_config["supports_functions"] else None
            
            # Make the API call
            response = await self._call_openai_api(messages, functions)
            
            # Process the response
            agent_response = await self._process_openai_response(response)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            agent_response.processing_time = processing_time
            
            # Update token usage
            usage = response.usage if hasattr(response, 'usage') else None
            if usage:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                agent_response.tokens_used = input_tokens + output_tokens
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error processing turn in OpenAI agent: {e}")
            
            # Return error response
            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"I encountered an error: {str(e)}",
                timestamp=datetime.now()
            )
            
            return AgentResponse(
                message=error_message,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def finalize_session(self) -> Dict[str, Any]:
        """Finalize the evaluation session"""
        return {
            "model": self.model,
            "total_api_calls": len([msg for msg in self.conversation_history if msg.role == MessageRole.ASSISTANT]),
            "total_function_calls": sum(len(msg.tool_calls or []) for msg in self.conversation_history),
            "final_token_usage": self.total_tokens_used,
            "capabilities": self.capabilities.to_dict()
        }
    
    def _prepare_messages(self, current_message: str = None, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Prepare messages in OpenAI format using managed conversation history"""
        messages = []
        
        # Check if managed conversation history is provided (context management active)
        if context and 'managed_conversation' in context:
            logger.debug("Using managed conversation history from context management")
            managed_history = context['managed_conversation']
            
            for msg_data in managed_history:
                # Skip metadata-only messages
                if msg_data.get('metadata', {}).get('type') == 'conversation_summary':
                    # Add summary as system message
                    messages.append({
                        "role": "system",
                        "content": msg_data['content']
                    })
                else:
                    # Regular conversation message
                    messages.append({
                        "role": msg_data['role'],
                        "content": msg_data['content']
                    })
            
            # Add architectural summary if available
            if context.get('architectural_summary'):
                messages.append({
                    "role": "system",
                    "content": f"Project architecture summary:\n{context['architectural_summary']}"
                })
        
        else:
            # Use normal conversation history (default behavior)
            logger.debug(f"Using normal conversation history ({len(self.conversation_history)} messages)")
            for msg in self.conversation_history:
                openai_msg = {
                    "role": msg.role.value,
                    "content": msg.content
                }
                
                # Add tool calls if present
                if msg.tool_calls:
                    openai_msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": f"{tc.tool_name}_{tc.function_name}",
                                "arguments": json.dumps(tc.parameters)
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                
                # Add tool_call_id if present (for tool messages)
                if msg.tool_call_id:
                    openai_msg["tool_call_id"] = msg.tool_call_id
                
                messages.append(openai_msg)
        
        # Add current message if provided
        if current_message:
            messages.append({
                "role": "user",
                "content": current_message
            })
        
        return messages
    
    def _prepare_functions(self) -> Optional[List[Dict[str, Any]]]:
        """Prepare functions for OpenAI function calling"""
        if not self.available_tools:
            return None
        
        functions = []
        
        for tool in self.available_tools:
            if not tool.enabled:
                continue
                
            for func in tool.get_functions():
                # Modify function name to include tool name (use underscore instead of dot for OpenAI compatibility)
                openai_func = func.to_openai_format()
                openai_func["name"] = f"{tool.name}_{func.name}"
                functions.append(openai_func)
        
        return functions if functions else None
    
    async def _call_openai_api(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Make the actual API call to OpenAI with automatic retry on rate limits"""
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        
        # Handle temperature parameter based on model requirements:
        # - omit_temperature: Do NOT include parameter at all (o-series, o1-series)
        # - fixed_temperature: Must use specific value (GPT-5 series only supports 1.0)
        # - default: Use configured temperature (GPT-4o, GPT-4.1 support 0-2)
        if self.model_config.get("omit_temperature", False):
            # o-series and o1-series: temperature parameter is not supported, omit it
            pass  # Do not add temperature to kwargs
        elif "fixed_temperature" in self.model_config:
            # GPT-5 series: only supports temperature=1.0
            kwargs["temperature"] = self.model_config["fixed_temperature"]
        else:
            # GPT-4o, GPT-4.1, GPT-4: support temperature 0-2
            kwargs["temperature"] = self.temperature
        
        # Use max_completion_tokens for newer models (GPT-5, o3 series)
        # Use max_tokens for older models (GPT-4o, GPT-4, etc.)
        if self.model_config.get("use_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        
        # Add functions if available and model supports them
        if functions and self.model_config["supports_functions"]:
            kwargs["tools"] = [{"type": "function", "function": func} for func in functions]
            kwargs["tool_choice"] = "auto"
        
        # Wrap API call in retry logic to handle rate limits automatically
        async def make_api_call():
            return await self.client.chat.completions.create(**kwargs)
        
        response = await retry_with_exponential_backoff(
            make_api_call,
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            provider=f"OpenAI-{self.model}"
        )
        return response
    
    async def _process_openai_response(self, response: Any) -> AgentResponse:
        """Process the response from OpenAI API"""
        choice = response.choices[0]
        message = choice.message
        
        # Create agent message with improved content handling
        content = message.content or ""
        
        # Process tool calls first to get parsed tool information
        tool_calls = []
        tool_responses = []
        
        # Process tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                # Parse tool and function name (format: tool_name_function_name)
                # Tool names can contain underscores, so we need to match against registered tools
                full_function_name = tool_call.function.name
                tool_name = "unknown"
                function_name = full_function_name
                
                # Try to match against registered tool names (longest match first)
                available_tool_names = [t.name for t in self.available_tools]
                available_tool_names.sort(key=len, reverse=True)  # Longest first to avoid partial matches
                
                for registered_tool_name in available_tool_names:
                    if full_function_name.startswith(registered_tool_name + '_'):
                        # Clean the tool name by removing _copy_ suffix for storage
                        clean_tool_name = registered_tool_name
                        if '_copy_' in clean_tool_name:
                            clean_tool_name = clean_tool_name.split('_copy_')[0]
                        tool_name = clean_tool_name
                        function_name = full_function_name[len(registered_tool_name) + 1:]  # +1 for underscore
                        break
                
                # Parse parameters
                try:
                    parameters = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    parameters = {}
                
                # Create tool call
                tc = ToolCall(
                    call_id=tool_call.id,
                    tool_name=tool_name,
                    function_name=function_name,
                    parameters=parameters,
                    timestamp=datetime.now()
                )
                tool_calls.append(tc)
                
                # Execute tool call
                tool_response = await self._execute_tool_call(tc)
                tool_responses.append(tool_response)
        
        # If content is empty but there are tool calls, add explanatory text using parsed tool info
        if not content and tool_calls:
            # Group actions by type for cleaner descriptions
            action_groups = {}
            for tc in tool_calls:
                clean_function_name = tc.function_name.replace('_', ' ')
                if clean_function_name not in action_groups:
                    action_groups[clean_function_name] = 0
                action_groups[clean_function_name] += 1
            
            # Create natural descriptions
            actions = []
            for action_type, count in action_groups.items():
                if count == 1:
                    actions.append(action_type)
                else:
                    actions.append(f"{action_type} ({count} files)")
            
            # Create more natural descriptions
            if len(actions) == 1:
                content = f"I'll help you with this task. Let me start by {actions[0]}."
            elif len(actions) == 2:
                content = f"I'll help you with this task. Let me start by {actions[0]} and {actions[1]}."
            else:
                content = f"I'll help you with this task. Let me start by {', '.join(actions[:-1])}, and {actions[-1]}."
        
        agent_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            timestamp=datetime.now()
        )
        
        # Add tool calls and responses to assistant message
        if tool_calls:
            agent_message.tool_calls = tool_calls
        if tool_responses:
            agent_message.tool_responses = tool_responses
        
        # Add assistant message to history
        self.add_message_to_history(agent_message)
        
        # Add separate tool response messages to history
        if tool_responses:
            for tool_response in tool_responses:
                # Truncate large tool results to prevent context overflow
                # Even though OpenAI has 10MB per-message limit, we need to be aggressive
                # to avoid filling the entire context window with tool results
                if tool_response.success:
                    result_str = json.dumps(tool_response.result)
                    MAX_TOOL_RESULT_SIZE = 50_000  # 50KB limit (~12K tokens, reasonable for file contents)
                    if len(result_str) > MAX_TOOL_RESULT_SIZE:
                        logger.warning(f"Tool result too large ({len(result_str):,} bytes), truncating to {MAX_TOOL_RESULT_SIZE:,}")
                        result_str = result_str[:MAX_TOOL_RESULT_SIZE] + f"\n\n... [TRUNCATED: {len(result_str) - MAX_TOOL_RESULT_SIZE:,} bytes omitted]\n\nNote: Use more specific queries or read files individually to get complete content."
                    content = result_str
                else:
                    content = f"Error: {tool_response.error_message}"
                
                tool_message = AgentMessage(
                    role=MessageRole.TOOL,
                    content=content,
                    timestamp=datetime.now(),
                    tool_call_id=tool_response.call_id
                )
                self.add_message_to_history(tool_message)
        
        # Create agent response
        return AgentResponse(
            message=agent_message,
            tool_calls=tool_calls,
            reasoning=self._extract_reasoning(message.content),
            confidence=1.0  # Could be enhanced with confidence scoring
        )
    
    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """Execute a tool call and return the response"""

        start_time = time.time()
        
        try:
            # Find the tool - handle both exact match and _copy_ suffix match
            tool = None
            for t in self.available_tools:
                # First try exact match
                if t.name == tool_call.tool_name:
                    tool = t
                    break
                # Then try matching the base name (removing _copy_ suffix)
                base_name = t.name.split('_copy_')[0] if '_copy_' in t.name else t.name
                if base_name == tool_call.tool_name:
                    tool = t
                    break
            
            if not tool:
                # Debug info for troubleshooting
                available_names = [t.name for t in self.available_tools]
                logger.debug(f"Tool '{tool_call.tool_name}' not found. Available tools: {available_names}")
                return ToolResponse(
                    call_id=tool_call.call_id,
                    result=None,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Tool '{tool_call.tool_name}' not found"
                )
            
            # Execute the function
            result = await tool.call_function(tool_call.function_name, tool_call.parameters)
            
            execution_time = time.time() - start_time
            
            return ToolResponse(
                call_id=tool_call.call_id,
                result=result.get("result"),
                success=result.get("success", False),
                execution_time=execution_time,
                error_message=result.get("error"),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error executing tool call {tool_call.call_id}: {e}")
            
            return ToolResponse(
                call_id=tool_call.call_id,
                result=None,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from the agent's response"""
        if not content:
            return None
        
        # Look for common reasoning patterns
        reasoning_markers = [
            "I need to", "First, I'll", "Let me", "I should", "My approach is",
            "I think", "I believe", "Based on", "Given that", "Since"
        ]
        
        sentences = content.split('.')
        reasoning_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(marker.lower() in sentence.lower() for marker in reasoning_markers):
                reasoning_sentences.append(sentence)
        
        return '. '.join(reasoning_sentences) if reasoning_sentences else None
    
    async def _test_connection(self) -> None:
        """Test the connection to OpenAI API"""
        try:
            # Simple test call with correct parameter name for the model
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}]  # Short message for quick test
            }
            
            # Handle temperature parameter (same logic as _call_openai_api)
            if self.model_config.get("omit_temperature", False):
                # o-series and o1-series: temperature parameter is not supported, omit it
                pass  # Do not add temperature to kwargs
            elif "fixed_temperature" in self.model_config:
                # GPT-5 series: only supports temperature=1.0
                kwargs["temperature"] = self.model_config["fixed_temperature"]
            else:
                # GPT-4o, GPT-4.1, GPT-4: support temperature 0-2
                kwargs["temperature"] = self.temperature
            
            # Use appropriate parameter name based on model
            # CRITICAL: o-series reasoning models (o3, o4-mini, o1-mini) generate
            # internal reasoning tokens that count toward max_completion_tokens!
            # Even a simple "Hi" prompt can use 1000+ tokens (reasoning + response)
            # Regular models only need ~100 tokens for connection test
            if self.model_config.get("use_max_completion_tokens", False):
                # o-series and GPT-5 series (reasoning models): need 2000+ tokens
                kwargs["max_completion_tokens"] = 2000
            else:
                # Regular models (GPT-4o, GPT-4.1): 100 tokens is sufficient
                kwargs["max_tokens"] = 100
            
            response = await self.client.chat.completions.create(**kwargs)
            logger.debug("OpenAI connection test successful")
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            raise
