"""
Google Agent Implementation for LoCoBench-Agent

This module implements agents that use Google's Gemini function calling capabilities
to interact with tools in the LoCoBench-Agent evaluation framework.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None

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
    provider: str = "Google"
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
                "rate limit", "rate_limit", "ratelimit", "429", "too many requests",
                "quota", "resource_exhausted"  # Google-specific
            ])
            is_server_error = any(pattern in error_str for pattern in [
                "500", "502", "503", "504", "internal error", "server error",
                "connection", "timeout", "network", "unavailable"
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


class GoogleAgent(BaseAgent):
    """
    Agent implementation using Google's Gemini function calling capabilities
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "gemini-2.0-flash-exp": {
            "max_tokens": 1000000,
            "supports_functions": True,
        },
        "gemini-1.5-pro": {
            "max_tokens": 2000000,
            "supports_functions": True,
        },
        "gemini-1.5-flash": {
            "max_tokens": 1000000,
            "supports_functions": True,
        },
        "gemini-2.5-pro": {  # Future model placeholder
            "max_tokens": 1000000,
            "supports_functions": True,
        },
        "gemini-2.5-flash": {  # Future model placeholder
            "max_tokens": 1000000,
            "supports_functions": True,
        }
    }
    
    def __init__(
        self,
        name: str = "Google Agent",
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        config: Dict[str, Any] = None
    ):
        if genai is None:
            raise ImportError("Google AI package not installed. Run: pip install google-generativeai")
        
        super().__init__(name, config)
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_config = self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS["gemini-2.0-flash-exp"])
        
        # Initialize Google AI client
        if api_key:
            genai.configure(api_key=api_key)
        
        # Store model name for later reconstruction with tools
        self._model_name = model
        self._generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        } if HarmCategory else {}
        
        # Will be initialized with tools in initialize_session
        self.client = None
        
        # Tool registry for function calling
        self.tool_registry: Optional[ToolRegistry] = None
        self.available_tools: List[Tool] = []
        self.chat_session = None
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.FUNCTION_CALLING,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.REASONING,
                AgentCapability.LONG_CONTEXT,
                AgentCapability.MULTIMODAL
            ],
            max_context_tokens=self.model_config["max_tokens"],
            supports_function_calling=self.model_config["supports_functions"],
            supports_tool_usage=self.model_config["supports_functions"]
        )
        
        logger.info(f"Initialized Google agent with model {model}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize a new evaluation session"""
        try:
            self.available_tools = available_tools or []
            
            # Prepare tools for the chat session
            tools = self._prepare_tools() if self.model_config["supports_functions"] else None
            
            # Create GenerativeModel with tools (tools go in the model, not start_chat)
            if tools:
                self.client = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config=self._generation_config,
                    safety_settings=self._safety_settings,
                    tools=tools  # Tools passed here, not to start_chat
                )
            else:
                self.client = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config=self._generation_config,
                    safety_settings=self._safety_settings
                )
            
            # Create a new chat session (without tools parameter)
            self.chat_session = self.client.start_chat(history=[])
            
            # Test Google AI connection
            await self._test_connection()
            
            logger.info(f"Google agent session initialized with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google agent session: {e}")
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
                # Recreate model and chat session with new tools
                tools = self._prepare_tools() if self.model_config["supports_functions"] else None
                if tools:
                    self.client = genai.GenerativeModel(
                        model_name=self._model_name,
                        generation_config=self._generation_config,
                        safety_settings=self._safety_settings,
                        tools=tools
                    )
                    self.chat_session = self.client.start_chat(history=self._prepare_history())
            
            # Make the API call
            response = await self._call_google_api(message)
            
            # Process the response
            agent_response = await self._process_google_response(response)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            agent_response.processing_time = processing_time
            
            # Update token usage (Google AI doesn't provide detailed usage info)
            # Estimate based on content length
            estimated_input_tokens = len(message.split()) * 1.3
            estimated_output_tokens = len(agent_response.message.content.split()) * 1.3
            
            agent_response.tokens_used = int(estimated_input_tokens + estimated_output_tokens)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error processing turn in Google agent: {e}")
            
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
    
    def _prepare_tools(self) -> Optional[List[Any]]:
        """Prepare tools for Google AI function calling"""
        if not self.available_tools:
            return None
        
        tools = []
        
        for tool in self.available_tools:
            if not tool.enabled:
                continue
                
            for func in tool.get_functions():
                # Create function declaration for Google AI
                properties = {}
                for p in func.parameters:
                    param_schema = genai.protos.Schema(
                        type=self._convert_type_to_google(p.type),
                        description=p.description
                    )
                    
                    # If parameter is an array, specify items type
                    if p.type == "array":
                        # Default to string items if not specified
                        param_schema.items = genai.protos.Schema(
                            type=genai.protos.Type.STRING
                        )
                    
                    properties[p.name] = param_schema
                
                google_func = genai.protos.FunctionDeclaration(
                    name=f"{tool.name}_{func.name}",  # Use underscore for Google AI
                    description=func.description,
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties=properties,
                        required=[p.name for p in func.parameters if p.required]
                    )
                )
                tools.append(google_func)
        
        return [genai.protos.Tool(function_declarations=tools)] if tools else None
    
    def _convert_type_to_google(self, param_type: str) -> Any:
        """Convert parameter type to Google AI type"""
        type_mapping = {
            "string": genai.protos.Type.STRING,
            "integer": genai.protos.Type.INTEGER,
            "number": genai.protos.Type.NUMBER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT
        }
        return type_mapping.get(param_type, genai.protos.Type.STRING)
    
    def _prepare_history(self) -> List[Dict[str, Any]]:
        """Prepare conversation history for Google AI chat session"""
        history = []
        
        for msg in self.conversation_history:
            if msg.role == MessageRole.SYSTEM:
                # System messages are handled differently in Google AI
                continue
            elif msg.role == MessageRole.USER:
                history.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == MessageRole.ASSISTANT:
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                
                # Add function calls
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append({
                            "function_call": {
                                "name": f"{tc.tool_name}_{tc.function_name}",
                                "args": tc.parameters
                            }
                        })
                
                history.append({
                    "role": "model",
                    "parts": parts
                })
                
                # Add function responses
                if msg.tool_responses:
                    user_parts = []
                    for tr in msg.tool_responses:
                        user_parts.append({
                            "function_response": {
                                "name": f"{tr.call_id}",
                                "response": tr.result if tr.success else {"error": tr.error_message}
                            }
                        })
                    
                    if user_parts:
                        history.append({
                            "role": "user",
                            "parts": user_parts
                        })
        
        return history
    
    async def _call_google_api(self, message: str) -> Any:
        """Make the actual API call to Google AI with automatic retry on rate limits"""
        if not self.chat_session:
            raise RuntimeError("Chat session not initialized")
        
        # Wrap API call in retry logic to handle rate limits automatically
        async def make_api_call():
            # Google's send_message is synchronous, so we run it in executor
            return await asyncio.to_thread(self.chat_session.send_message, message)
        
        response = await retry_with_exponential_backoff(
            make_api_call,
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            provider=f"Google-{self.model}"
        )
        return response
    
    async def _process_google_response(self, response: Any) -> AgentResponse:
        """Process the response from Google AI API"""
        content_text = ""
        tool_calls = []
        tool_responses = []
        
        # Extract content from response
        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                content_text += part.text
            elif hasattr(part, 'function_call'):
                # Parse tool and function name
                full_function_name = part.function_call.name
                if '_' in full_function_name:
                    # Split only on the first underscore to handle tool names with underscores
                    parts = full_function_name.split('_', 1)
                    tool_name = parts[0]
                    function_name = parts[1]
                else:
                    tool_name = "unknown"
                    function_name = full_function_name
                
                # Create tool call
                # Convert protobuf args to regular dict to avoid JSON serialization issues
                parameters = {}
                for key, value in part.function_call.args.items():
                    # Convert protobuf values to Python native types
                    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                        # Handle repeated/list values
                        parameters[key] = list(value)
                    else:
                        parameters[key] = value
                
                tc = ToolCall(
                    call_id=f"call_{int(time.time() * 1000)}",  # Generate unique ID
                    tool_name=tool_name,
                    function_name=function_name,
                    parameters=parameters,
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
        """Test the connection to Google AI API"""
        try:
            # Simple test call
            test_model = genai.GenerativeModel(self.model)
            response = test_model.generate_content("Hello")
            logger.debug("Google AI connection test successful")
            
        except Exception as e:
            logger.error(f"Google AI connection test failed: {e}")
            raise
