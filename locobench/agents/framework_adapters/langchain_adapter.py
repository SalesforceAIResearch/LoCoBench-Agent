"""
LangChain Agent Adapter for LoCoBench-Agent

This adapter integrates LangChain agents into the LoCoBench-Agent
evaluation framework, supporting ReAct patterns, tool usage,
and memory-enabled agents.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.agents.agent_types import AgentType
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import HumanMessage, AIMessage
    from langchain.tools import BaseTool
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    AgentExecutor = None
    create_react_agent = None
    AgentType = None
    ConversationBufferWindowMemory = None
    HumanMessage = None
    AIMessage = None
    BaseTool = None
    PromptTemplate = None
    ChatOpenAI = None
    ChatAnthropic = None

from ..base_agent import (
    BaseAgent, AgentResponse, AgentMessage, MessageRole, 
    ToolCall, ToolResponse, AgentCapabilities, AgentCapability
)
from ...core.tool_registry import Tool

logger = logging.getLogger(__name__)


class LangChainToolWrapper(BaseTool if BaseTool else object):
    """Wrapper to adapt LoCoBench tools to LangChain tools"""
    
    def __init__(self, locobench_tool: Tool):
        self.locobench_tool = locobench_tool
        if BaseTool:
            super().__init__(
                name=locobench_tool.name,
                description=locobench_tool.description
            )
        else:
            self.name = locobench_tool.name
            self.description = locobench_tool.description
    
    def _run(self, tool_input: str) -> str:
        """Run the tool synchronously"""
        try:
            # Simple string input - parse as needed
            result = asyncio.run(self.locobench_tool.call_function(
                "default", {"input": tool_input}
            ))
            return str(result.get("result", ""))
        except Exception as e:
            return f"Tool error: {str(e)}"
    
    async def _arun(self, tool_input: str) -> str:
        """Run the tool asynchronously"""
        try:
            result = await self.locobench_tool.call_function(
                "default", {"input": tool_input}
            )
            return str(result.get("result", ""))
        except Exception as e:
            return f"Tool error: {str(e)}"


class LangChainAdapter(BaseAgent):
    """
    Adapter for LangChain agents
    
    This adapter wraps LangChain's agent framework to work with
    LoCoBench-Agent's evaluation system.
    """
    
    def __init__(
        self,
        name: str = "LangChain Agent",
        langchain_config: Dict[str, Any] = None,
        config: Dict[str, Any] = None
    ):
        if AgentExecutor is None:
            raise ImportError("LangChain packages not installed. Run: pip install langchain langchain-openai langchain-anthropic")
        
        super().__init__(name, config)
        
        self.langchain_config = langchain_config or {}
        self.agent_executor = None
        self.memory = None
        self.llm = None
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.TOOL_USAGE,
                AgentCapability.REASONING,
                AgentCapability.MEMORY,
                AgentCapability.CODE_GENERATION,
                AgentCapability.LONG_CONTEXT
            ],
            max_context_tokens=self.langchain_config.get("max_tokens", 4096),
            supports_function_calling=True,
            supports_tool_usage=True,
        )
        
        logger.info(f"Initialized LangChain adapter: {name}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize LangChain agent session"""
        try:
            # Initialize LLM
            self._initialize_llm()
            
            # Initialize memory
            self._initialize_memory()
            
            # Convert LoCoBench tools to LangChain tools
            langchain_tools = []
            if available_tools:
                for tool in available_tools:
                    if tool.enabled:
                        langchain_tools.append(LangChainToolWrapper(tool))
            
            # Create agent
            self._create_agent(langchain_tools, scenario_context)
            
            logger.info(f"LangChain session initialized with {len(langchain_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain session: {e}")
            return False
    
    async def process_turn(
        self,
        message: str,
        available_tools: List[Tool] = None,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process a turn using LangChain agent"""
        start_time = time.time()
        
        try:
            if not self.agent_executor:
                raise RuntimeError("LangChain agent not initialized")
            
            # Execute the agent
            result = await self.agent_executor.ainvoke({
                "input": message,
                "chat_history": self._get_chat_history()
            })
            
            # Extract response
            output = result.get("output", "")
            
            # Create agent message
            agent_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=output,
                timestamp=datetime.now()
            )
            
            # Add to conversation history
            self.add_message_to_history(agent_message)
            
            # Update memory
            if self.memory:
                self.memory.chat_memory.add_user_message(message)
                self.memory.chat_memory.add_ai_message(output)
            
            # Create agent response
            agent_response = AgentResponse(
                message=agent_message,
                processing_time=time.time() - start_time,
                tool_calls=self._extract_tool_calls(result),
                reasoning=self._extract_reasoning(output),
                confidence=0.8  # Default confidence for LangChain
            )
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in LangChain turn processing: {e}")
            
            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"LangChain error: {str(e)}",
                timestamp=datetime.now()
            )
            
            return AgentResponse(
                message=error_message,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def finalize_session(self) -> Dict[str, Any]:
        """Finalize LangChain session"""
        return {
            "adapter_type": "langchain",
            "agent_type": self.langchain_config.get("agent_type", "react"),
            "total_messages": len(self.conversation_history),
            "memory_enabled": self.memory is not None,
            "capabilities": self.capabilities.to_dict()
        }
    
    def _initialize_llm(self):
        """Initialize the LLM for LangChain with gateway support"""
        import os
        provider = self.langchain_config.get("provider", "openai")
        model_name = self.langchain_config.get("model_name", "gpt-4")
        
        if provider == "openai":
            # Support for Salesforce Research OpenAI Gateway
            api_key = self.langchain_config.get("api_key")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not base_url and self.langchain_config.get("base_url"):
                base_url = self.langchain_config.get("base_url")
            
            llm_kwargs = {
                "model": model_name,
                "temperature": self.langchain_config.get("temperature", 0.1)
            }
            
            if base_url:
                # Gateway requires X-Api-Key header and dummy api_key
                llm_kwargs["api_key"] = "dummy"
                llm_kwargs["base_url"] = base_url
                llm_kwargs["default_headers"] = {"X-Api-Key": api_key}
                logger.info(f"LangChain using OpenAI Gateway: {base_url}")
            else:
                # Direct OpenAI API
                llm_kwargs["api_key"] = api_key
            
            self.llm = ChatOpenAI(**llm_kwargs)
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=self.langchain_config.get("temperature", 0.1),
                api_key=self.langchain_config.get("api_key")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _initialize_memory(self):
        """Initialize memory for the agent"""
        if self.langchain_config.get("enable_memory", True):
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=self.langchain_config.get("memory_window", 10),
                return_messages=True
            )
    
    def _create_agent(self, tools: List[BaseTool], scenario_context: Dict[str, Any]):
        """Create the LangChain agent"""
        
        # Create ReAct prompt template
        react_prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant specialized in software development.
        
        Scenario: {scenario_description}
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought: {agent_scratchpad}
        """)
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=react_prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=self.langchain_config.get("verbose", False),
            max_iterations=self.langchain_config.get("max_iterations", 10),
            early_stopping_method="generate"
        )
    
    def _get_chat_history(self) -> List[Any]:
        """Get chat history in LangChain format"""
        history = []
        
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            if msg.role == MessageRole.USER:
                history.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                history.append(AIMessage(content=msg.content))
        
        return history
    
    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from LangChain result"""
        tool_calls = []
        
        # LangChain doesn't directly expose tool calls in the result
        # This would need to be implemented based on specific agent type
        # For now, return empty list
        
        return tool_calls
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from LangChain response"""
        if not content:
            return None
        
        # Look for ReAct reasoning patterns
        reasoning_patterns = [
            "Thought:", "I need to", "Let me think", "My approach is",
            "I should", "First, I'll", "The plan is"
        ]
        
        lines = content.split('\n')
        reasoning_lines = []
        
        for line in lines:
            line = line.strip()
            if any(pattern in line for pattern in reasoning_patterns):
                reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines) if reasoning_lines else None
