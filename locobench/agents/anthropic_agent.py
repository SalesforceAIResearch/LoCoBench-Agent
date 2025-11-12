"""
Anthropic Agent Implementation for LoCoBench-Agent

This module implements agents that use Anthropic's Claude tool usage capabilities
to interact with tools in the LoCoBench-Agent evaluation framework.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None

from .base_agent import (
    BaseAgent, AgentResponse, AgentMessage, MessageRole, 
    ToolCall, ToolResponse, AgentCapabilities, AgentCapability
)
from ..core.tool_registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)


async def retry_with_exponential_backoff(
    func,
    max_retries: int = 10,  # Increased from 5 to 10 for throttling resilience
    base_delay: float = 3.0,  # Increased from 2.0 to 3.0
    max_delay: float = 120.0,  # Increased from 60.0 to 120.0 for severe throttling
    provider: str = "Anthropic"
):
    """
    Retry a function with exponential backoff for rate limit and transient errors.
    
    ENHANCED FOR THROTTLING RESILIENCE:
    - Increased max_retries to 10 (from 5)
    - Longer delays for throttling (up to 120s)
    - Special handling for AWS ThrottlingException
    
    This ensures that API calls are resilient to:
    - Rate limit errors (429)
    - AWS ThrottlingException (org-wide rate limits)
    - Temporary network issues
    - Server errors (500, 502, 503, 504)
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 10)
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay cap in seconds (default: 120s)
        provider: API provider name for logging
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is a throttling error (AWS-specific)
            is_throttling = any(pattern in error_str for pattern in [
                "throttlingexception", "throttling", "too many tokens"
            ])
            
            # Check if this is a rate limit error
            is_rate_limit = any(pattern in error_str for pattern in [
                "rate limit", "rate_limit", "ratelimit", "429", "too many requests"
            ])
            
            # Check if this is a server error
            is_server_error = any(pattern in error_str for pattern in [
                "500", "502", "503", "504", "internal error", "server error",
                "connection", "timeout", "network"
            ])
            
            is_retryable = is_throttling or is_rate_limit or is_server_error
            
            if is_retryable and attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                # Use even longer delays for throttling errors
                if is_throttling:
                    delay = min(base_delay * (2.5 ** attempt), max_delay)
                else:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                
                error_type = "Throttling" if is_throttling else ("Rate limit" if is_rate_limit else "Server error")
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


class AnthropicAgent(BaseAgent):
    """
    Agent implementation using Anthropic's Claude tool usage capabilities
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "claude-3-5-sonnet-20241022": {
            "max_tokens": 200000,
            "supports_tools": True,
        },
        "claude-3-5-haiku-20241022": {
            "max_tokens": 200000,
            "supports_tools": True,
        },
        "claude-3-opus-20240229": {
            "max_tokens": 200000,
            "supports_tools": True,
        },
        "claude-sonnet-4": {  # Future model placeholder
            "max_tokens": 200000,
            "supports_tools": True,
        },
        "claude-opus-4": {  # Future model placeholder
            "max_tokens": 200000,
            "supports_tools": True,
        }
    }
    
    def __init__(
        self,
        name: str = "Anthropic Agent",
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        config: Dict[str, Any] = None
    ):
        if anthropic is None:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        super().__init__(name, config)
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_config = self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS["claude-3-5-sonnet-20241022"])
        
        # Initialize Anthropic/Bedrock client
        # Support both Anthropic API key and AWS Bedrock Bearer Token
        import os
        auth_token = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_BEARER_TOKEN") or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        
        if not auth_token:
            raise ValueError("No authentication provided. Set ANTHROPIC_API_KEY, CLAUDE_BEARER_TOKEN, AWS_BEARER_TOKEN_BEDROCK, or pass api_key parameter")
        
        # Check if using AWS Bedrock Bearer Token (different from standard Anthropic API)
        # Bearer tokens for Bedrock start with "ABSK" while standard API keys start with "sk-ant-"
        if auth_token.startswith("ABSK"):
            # For AWS Bedrock, we MUST use boto3, not Anthropic SDK
            logger.info("Using AWS Bedrock Bearer Token authentication for Claude")
            
            # CRITICAL: Set the environment variable BEFORE creating boto3 client
            # boto3 will automatically use this for authentication
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = auth_token
            logger.info(f"Set AWS_BEARER_TOKEN_BEDROCK environment variable")
            
            # Get AWS region (default to us-east-1)
            region = os.getenv("AWS_REGION", "us-east-1")
            logger.info(f"Initializing Bedrock client in region: {region}")
            
            # Setup for Bedrock (defer actual client creation to avoid blocking __init__)
            self.use_bedrock = True
            self.bedrock_region = region
            self.bedrock_client = None  # Will be created on first use
            self.client = None  # We won't use AsyncAnthropic for Bedrock
            logger.info(f"Bedrock configuration set (client will be created on first use)")
        else:
            # Standard Anthropic API key authentication (sk-ant-...)
            logger.info("Using standard Anthropic API key authentication")
            self.use_bedrock = False
            self.bedrock_client = None
            self.client = AsyncAnthropic(api_key=auth_token)
        
        # Tool registry for tool usage
        self.tool_registry: Optional[ToolRegistry] = None
        self.available_tools: List[Tool] = []
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.TOOL_USAGE,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.REASONING,
                AgentCapability.LONG_CONTEXT
            ],
            max_context_tokens=self.model_config["max_tokens"],
            supports_function_calling=False,  # Claude uses tool usage, not function calling
            supports_tool_usage=self.model_config["supports_tools"]
        )
        
        # Throttling tracking for fair evaluation
        self.throttling_error_count = 0
        self.infrastructure_error_count = 0
        self.model_error_count = 0
        
        logger.info(f"Initialized Anthropic agent with model {model}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize a new evaluation session"""
        try:
            self.available_tools = available_tools or []
            
            # Test Anthropic connection
            await self._test_connection()
            
            logger.info(f"Anthropic agent session initialized with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic agent session: {e}")
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
            
            # Prepare messages (format depends on whether using Bedrock)
            messages = self._prepare_messages(message, for_bedrock=self.use_bedrock)
            
            # Prepare tools for tool usage (format depends on whether using Bedrock)
            # Note: tools are prepared inside _call_anthropic_api/_call_bedrock_api now
            tools = None  # Will be prepared inside API call methods
            
            # Make the API call
            response = await self._call_anthropic_api(messages, tools)
            
            # Process the response
            agent_response = await self._process_anthropic_response(response)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            agent_response.processing_time = processing_time
            
            # Update token usage
            usage = response.usage if hasattr(response, 'usage') else None
            if usage:
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                
                agent_response.tokens_used = input_tokens + output_tokens
            
            return agent_response
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Classify error type for fair evaluation
            is_throttling = any(pattern in error_str for pattern in [
                "throttling", "rate limit", "rate_limit", "too many tokens", "too many requests", "429"
            ])
            is_infrastructure = any(pattern in error_str for pattern in [
                "500", "502", "503", "504", "connection", "timeout", "network", "server error"
            ])
            
            # Track errors for session-level reporting
            if is_throttling:
                self.throttling_error_count += 1
                logger.warning(f"Throttling error in Anthropic agent (count: {self.throttling_error_count}): {e}")
                
                # THROTTLING-RESILIENT STRATEGY: Pause and wait before returning
                # This allows the agent to recover without counting as a failed turn
                if self.throttling_error_count < 10:  # Only for first 10 throttles
                    wait_time = min(60 + (self.throttling_error_count * 10), 180)  # 60s to 180s
                    logger.info(f"🔄 THROTTLING RECOVERY: Waiting {wait_time}s before continuing...")
                    await asyncio.sleep(wait_time)
                    
                    # Return a recovery message instead of error
                    recovery_message = AgentMessage(
                        role=MessageRole.ASSISTANT,
                        content="I'm experiencing API rate limits. Continuing after brief pause...",
                        timestamp=datetime.now()
                    )
                    
                    return AgentResponse(
                        message=recovery_message,
                        processing_time=time.time() - start_time,
                        metadata={
                            "throttling_recovery": True,
                            "wait_time": wait_time,
                            "throttling_count": self.throttling_error_count
                        }
                    )
                
            elif is_infrastructure:
                self.infrastructure_error_count += 1
                logger.warning(f"Infrastructure error in Anthropic agent (count: {self.infrastructure_error_count}): {e}")
            else:
                self.model_error_count += 1
                logger.error(f"Model error in Anthropic agent (count: {self.model_error_count}): {e}")
            
            # Return error response (for non-recoverable or excessive errors)
            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"I encountered an error: {str(e)}",
                timestamp=datetime.now()
            )
            
            return AgentResponse(
                message=error_message,
                processing_time=time.time() - start_time,
                metadata={
                    "error": str(e),
                    "error_type": "throttling" if is_throttling else ("infrastructure" if is_infrastructure else "model")
                }
            )
    
    async def finalize_session(self) -> Dict[str, Any]:
        """Finalize the evaluation session with throttling metadata"""
        return {
            "model": self.model,
            "total_api_calls": len([msg for msg in self.conversation_history if msg.role == MessageRole.ASSISTANT]),
            "total_tool_calls": sum(len(msg.tool_calls or []) for msg in self.conversation_history),
            "final_token_usage": self.total_tokens_used,
            "capabilities": self.capabilities.to_dict(),
            
            # Throttling resilience metadata for fair evaluation
            "throttling_error_count": self.throttling_error_count,
            "infrastructure_error_count": self.infrastructure_error_count,
            "model_error_count": self.model_error_count,
            "throttling_affected": self.throttling_error_count > 0,
            "evaluation_validity": "valid" if self.throttling_error_count < 5 else "throttling_impacted"
        }
    
    def _prepare_messages(self, current_message: str = None, for_bedrock: bool = False) -> List[Dict[str, Any]]:
        """Prepare messages in Anthropic or Bedrock format"""
        messages = []
        
        for msg in self.conversation_history:
            # Skip system messages - they'll be handled separately
            if msg.role == MessageRole.SYSTEM:
                continue
                
            message_dict = {
                "role": msg.role.value if msg.role != MessageRole.ASSISTANT else "assistant",
                "content": []
            }
            
            # Add text content
            if msg.content:
                if for_bedrock:
                    # Bedrock format: no "type" field, just {"text": "..."}
                    message_dict["content"].append({"text": msg.content})
                else:
                    # Anthropic format: {"type": "text", "text": "..."}
                    message_dict["content"].append({"type": "text", "text": msg.content})
            
            # Add tool use if present
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if for_bedrock:
                        # Bedrock format: {"toolUse": {...}}
                        # Bedrock requires tool names to match regex [a-zA-Z0-9_-]+
                        # Replace dots with underscores
                        tool_function_name = f"{tc.tool_name}_{tc.function_name}"
                        message_dict["content"].append({
                            "toolUse": {
                                "toolUseId": tc.call_id,
                                "name": tool_function_name,
                                "input": tc.parameters
                            }
                        })
                    else:
                        # Anthropic format: {"type": "tool_use", ...}
                        message_dict["content"].append({
                            "type": "tool_use",
                            "id": tc.call_id,
                            "name": f"{tc.tool_name}.{tc.function_name}",
                            "input": tc.parameters
                        })
            
            messages.append(message_dict)
            
            # Add tool results if present
            if msg.tool_responses:
                tool_result_msg = {
                    "role": "user",
                    "content": []
                }
                
                for tr in msg.tool_responses:
                    if for_bedrock:
                        # Bedrock format: {"toolResult": {...}}
                        tool_result_msg["content"].append({
                            "toolResult": {
                                "toolUseId": tr.call_id,
                                "content": [{"text": json.dumps(tr.result) if tr.success else f"Error: {tr.error_message}"}]
                            }
                        })
                    else:
                        # Anthropic format: {"type": "tool_result", ...}
                        tool_result_msg["content"].append({
                            "type": "tool_result",
                            "tool_use_id": tr.call_id,
                            "content": json.dumps(tr.result) if tr.success else f"Error: {tr.error_message}"
                        })
                
                if tool_result_msg["content"]:
                    messages.append(tool_result_msg)
        
        # Add current message if provided
        if current_message:
            if for_bedrock:
                # Bedrock format
                messages.append({
                    "role": "user",
                    "content": [{"text": current_message}]
                })
            else:
                # Anthropic format
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": current_message}]
                })
        
        return messages
    
    def _prepare_tools(self, for_bedrock: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Prepare tools for Anthropic or Bedrock tool usage"""
        if not self.available_tools:
            return None
        
        tools = []
        
        for tool in self.available_tools:
            if not tool.enabled:
                continue
                
            for func in tool.get_functions():
                tool_name = f"{tool.name}.{func.name}"
                input_schema = {
                    "type": "object",
                    "properties": {
                        p.name: {
                            "type": p.type,
                            "description": p.description
                        }
                        for p in func.parameters
                    },
                    "required": [p.name for p in func.parameters if p.required]
                }
                
                if for_bedrock:
                    # Bedrock Converse API format: wrap in toolSpec
                    # Bedrock requires tool names to match regex [a-zA-Z0-9_-]+
                    # Replace dots with underscores
                    bedrock_tool_name = tool_name.replace(".", "_")
                    bedrock_tool = {
                        "toolSpec": {
                            "name": bedrock_tool_name,
                            "description": func.description,
                            "inputSchema": {
                                "json": input_schema
                            }
                        }
                    }
                    tools.append(bedrock_tool)
                else:
                    # Standard Anthropic format
                    anthropic_tool = {
                        "name": tool_name,
                        "description": func.description,
                        "input_schema": input_schema
                    }
                    tools.append(anthropic_tool)
        
        return tools if tools else None
    
    def _get_system_message(self) -> str:
        """Get the system message from conversation history"""
        for msg in self.conversation_history:
            if msg.role == MessageRole.SYSTEM:
                return msg.content
        return "You are a helpful assistant."
    
    def _ensure_bedrock_client(self):
        """Lazily create the Bedrock client on first use"""
        if self.use_bedrock and self.bedrock_client is None:
            try:
                import boto3
                from botocore.config import Config
                logger.info(f"Creating Bedrock client for region: {self.bedrock_region}")
                
                # Configure longer timeout for Bedrock API calls
                config = Config(
                    read_timeout=300,  # 5 minutes
                    connect_timeout=60,  # 1 minute
                    retries={'max_attempts': 3, 'mode': 'adaptive'}
                )
                
                self.bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.bedrock_region,
                    config=config
                )
                logger.info("Bedrock client created successfully with extended timeout")
            except ImportError:
                raise ImportError("boto3 is required for AWS Bedrock authentication. Install it with: pip install boto3")
            except Exception as e:
                logger.error(f"Failed to create Bedrock client: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    
    def _convert_model_to_bedrock_id(self, model: str) -> str:
        """Convert Anthropic model name to Bedrock model ID"""
        # Map Anthropic model names to Bedrock model IDs
        bedrock_model_map = {
            "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-haiku-20241022": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "claude-3-opus-20240229": "us.anthropic.claude-3-opus-20240229-v1:0",
            "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-3.7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "claude-3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        }
        return bedrock_model_map.get(model, f"us.anthropic.{model}-v1:0")
    
    async def _call_anthropic_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Make the actual API call to Anthropic or Bedrock with automatic retry on rate limits"""
        
        if self.use_bedrock:
            # Use AWS Bedrock API
            return await self._call_bedrock_api(messages, tools)
        else:
            # Use Anthropic API
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": self._get_system_message()
            }
            
            # Add tools if available and model supports them
            if self.model_config["supports_tools"]:
                anthropic_tools = self._prepare_tools(for_bedrock=False)
                if anthropic_tools:
                    kwargs["tools"] = anthropic_tools
            
            # Wrap API call in retry logic to handle rate limits automatically
            async def make_api_call():
                return await self.client.messages.create(**kwargs)
            
            response = await retry_with_exponential_backoff(
                make_api_call,
                max_retries=5,
                base_delay=2.0,
                max_delay=60.0,
                provider=f"Anthropic-{self.model}"
            )
            return response
    
    async def _call_bedrock_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Call AWS Bedrock API (synchronous boto3 wrapped in async)"""
        import asyncio
        
        # Ensure Bedrock client is created
        self._ensure_bedrock_client()
        
        # Convert model name to Bedrock model ID
        model_id = self._convert_model_to_bedrock_id(self.model)
        
        # Prepare request for Bedrock
        request = {
            "modelId": model_id,
            "messages": messages,
        }
        
        # Add system message if present
        system_msg = self._get_system_message()
        if system_msg:
            request["system"] = [{"text": system_msg}]
        
        # Add inference configuration
        inference_config = {
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
        }
        request["inferenceConfig"] = inference_config
        
        # Add tools if available (Bedrock uses toolConfig with toolSpec format)
        if self.model_config["supports_tools"]:
            # Prepare tools in Bedrock format
            bedrock_tools = self._prepare_tools(for_bedrock=True)
            if bedrock_tools:
                request["toolConfig"] = {
                    "tools": bedrock_tools
                }
        
        # Wrap boto3 call (synchronous) in executor for async compatibility
        def make_bedrock_call():
            return self.bedrock_client.converse(**request)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, make_bedrock_call)
        
        # Convert Bedrock response format to Anthropic format for compatibility
        return self._convert_bedrock_response(response)
    
    def _convert_bedrock_response(self, bedrock_response: Dict[str, Any]) -> Any:
        """Convert Bedrock response to Anthropic-compatible format"""
        # Create a mock Anthropic response object
        class BedrockResponse:
            def __init__(self, bedrock_data):
                self.content = []
                self.stop_reason = bedrock_data.get("stopReason", "end_turn")
                
                # Convert usage dict to object with attributes
                usage_data = bedrock_data.get("usage", {})
                class Usage:
                    def __init__(self, data):
                        self.input_tokens = data.get("inputTokens", 0)
                        self.output_tokens = data.get("outputTokens", 0)
                self.usage = Usage(usage_data)
                
                # Parse output content
                output = bedrock_data.get("output", {})
                message = output.get("message", {})
                content_blocks = message.get("content", [])
                
                for block in content_blocks:
                    if "text" in block:
                        # Text content
                        class TextBlock:
                            def __init__(self, text):
                                self.type = "text"
                                self.text = text
                        self.content.append(TextBlock(block["text"]))
                    elif "toolUse" in block:
                        # Tool use content
                        tool_use = block["toolUse"]
                        class ToolUseBlock:
                            def __init__(self, tool_data):
                                self.type = "tool_use"
                                self.id = tool_data.get("toolUseId", "")
                                # Bedrock uses underscores, convert back to dots for consistency
                                # Format: tool_name_function_name -> tool.name_function.name
                                name = tool_data.get("name", "")
                                # Find the last underscore that separates tool from function
                                # We need to be careful since tool names can contain underscores
                                # The format from Bedrock will be: toolname_functionname
                                # We need to convert back to: toolname.functionname
                                # Strategy: Split on first underscore after the tool name portion
                                self.name = name  # Keep as-is for now, will be parsed in _process_anthropic_response
                                self.input = tool_data.get("input", {})
                        self.content.append(ToolUseBlock(tool_use))
        
        return BedrockResponse(bedrock_response)
    
    async def _process_anthropic_response(self, response: Any) -> AgentResponse:
        """Process the response from Anthropic API"""
        # Extract content
        content_text = ""
        tool_calls = []
        tool_responses = []
        
        for content_block in response.content:
            if content_block.type == "text":
                content_text += content_block.text
            elif content_block.type == "tool_use":
                # Parse tool and function name
                full_function_name = content_block.name
                if '.' in full_function_name:
                    # Anthropic format: tool.function
                    tool_name, function_name = full_function_name.split('.', 1)
                elif '_' in full_function_name:
                    # Bedrock format: tool_function (where dot was replaced with underscore)
                    # CRITICAL FIX: Bedrock adds copy_{id}_ prefix to tool names
                    # Format: copy_23172915920_tool_name_function_name
                    # Strip the copy_ prefix if present
                    clean_name = full_function_name
                    if clean_name.startswith('copy_'):
                        # Remove copy_{id}_ prefix (format: copy_DIGITS_rest)
                        parts = clean_name.split('_', 2)  # Split into ['copy', 'ID', 'rest']
                        if len(parts) >= 3:
                            clean_name = parts[2]  # Get everything after copy_ID_
                    
                    # Now match against available tools
                    tool_name = None
                    function_name = None
                    for tool in self.available_tools:
                        tool_base = tool.name
                        if clean_name.startswith(tool_base + "_"):
                            tool_name = tool_base
                            function_name = clean_name[len(tool_base) + 1:]
                            break
                    
                    if not tool_name:
                        # Fallback: assume last underscore is the separator
                        parts = clean_name.rsplit('_', 1)
                        if len(parts) == 2:
                            tool_name, function_name = parts
                        else:
                            tool_name = "unknown"
                            function_name = clean_name
                else:
                    tool_name = "unknown"
                    function_name = full_function_name
                
                # Create tool call
                tc = ToolCall(
                    call_id=content_block.id,
                    tool_name=tool_name,
                    function_name=function_name,
                    parameters=content_block.input,
                    timestamp=datetime.now()
                )
                tool_calls.append(tc)
                
                # Execute tool call
                tool_response = await self._execute_tool_call(tc)
                tool_responses.append(tool_response)
        
        # Create agent message
        agent_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
            tool_responses=tool_responses if tool_responses else None,
            timestamp=datetime.now()
        )
        
        # Add message to history
        self.add_message_to_history(agent_message)
        
        # Create agent response
        return AgentResponse(
            message=agent_message,
            tool_calls=tool_calls,
            reasoning=self._extract_reasoning(content_text),
            confidence=1.0  # Could be enhanced with confidence scoring
        )
    
    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """Execute a tool call and return the response"""
        start_time = time.time()
        
        try:
            # Find the tool - handle exact match
            tool = None
            for t in self.available_tools:
                if t.name == tool_call.tool_name:
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
        """Test the connection to Anthropic or Bedrock API"""
        try:
            if self.use_bedrock:
                # Ensure Bedrock client is created
                self._ensure_bedrock_client()
                
                # Test Bedrock connection
                import asyncio
                model_id = self._convert_model_to_bedrock_id(self.model)
                
                def test_bedrock():
                    return self.bedrock_client.converse(
                        modelId=model_id,
                        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
                        inferenceConfig={"maxTokens": 10, "temperature": 0.1}
                    )
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, test_bedrock)
                logger.debug("Bedrock connection test successful")
            else:
                # Test Anthropic connection
                response = await self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                    max_tokens=10
                )
                logger.debug("Anthropic connection test successful")
            
        except Exception as e:
            provider = "Bedrock" if self.use_bedrock else "Anthropic"
            logger.error(f"{provider} connection test failed: {e}")
            raise
