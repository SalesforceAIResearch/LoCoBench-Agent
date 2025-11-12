"""
OpenAI Swarm Agent Adapter for LoCoBench-Agent

This adapter integrates OpenAI Swarm agents into the LoCoBench-Agent
evaluation framework, supporting lightweight multi-agent coordination,
agent handoffs, and function calling orchestration.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

try:
    from swarm import Swarm, Agent as SwarmAgent
except ImportError:
    Swarm = None
    SwarmAgent = None

from ..base_agent import (
    BaseAgent, AgentResponse, AgentMessage, MessageRole, 
    ToolCall, ToolResponse, AgentCapabilities, AgentCapability
)
from ...core.tool_registry import Tool

logger = logging.getLogger(__name__)


class SwarmAdapter(BaseAgent):
    """
    Adapter for OpenAI Swarm agents
    
    This adapter wraps OpenAI Swarm's multi-agent coordination system to work with
    LoCoBench-Agent's evaluation framework.
    """
    
    def __init__(
        self,
        name: str = "Swarm Agent",
        swarm_config: Dict[str, Any] = None,
        config: Dict[str, Any] = None
    ):
        if Swarm is None:
            raise ImportError("OpenAI Swarm package not installed. Run: pip install git+https://github.com/openai/swarm.git")
        
        super().__init__(name, config)
        
        self.swarm_config = swarm_config or {}
        self.swarm_client = Swarm()
        self.swarm_agents = []
        self.current_agent = None
        self.conversation_history = []
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.MULTI_AGENT,
                AgentCapability.FUNCTION_CALLING,
                AgentCapability.AGENT_HANDOFF,
                AgentCapability.COORDINATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.REASONING
            ],
            max_context_tokens=self.swarm_config.get("max_tokens", 4096),
            supports_function_calling=True,
            supports_tool_usage=True,
        )
        
        logger.info(f"Initialized OpenAI Swarm adapter: {name}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize Swarm session"""
        try:
            # Convert LoCoBench tools to Swarm functions
            swarm_functions = []
            if available_tools:
                swarm_functions = self._convert_tools_to_functions(available_tools)
            
            # Create specialized Swarm agents based on scenario
            self._create_swarm_agents(scenario_context, swarm_functions)
            
            # Set initial agent
            if self.swarm_agents:
                self.current_agent = self.swarm_agents[0]
            
            logger.info(f"Swarm session initialized with {len(self.swarm_agents)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Swarm session: {e}")
            return False
    
    async def process_turn(
        self,
        message: str,
        available_tools: List[Tool] = None,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process a turn using Swarm agents"""
        start_time = time.time()
        
        try:
            if not self.current_agent:
                raise RuntimeError("Swarm agent not initialized")
            
            # Prepare messages for Swarm
            messages = self._prepare_messages(message)
            
            # Run Swarm conversation
            response = self.swarm_client.run(
                agent=self.current_agent,
                messages=messages
            )
            
            # Extract response content
            content = ""
            if response.messages:
                content = response.messages[-1].get("content", "")
            
            # Handle agent changes
            if response.agent != self.current_agent:
                self.current_agent = response.agent
                logger.info(f"Agent handoff to: {self.current_agent.name}")
            
            # Create agent message
            agent_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                timestamp=datetime.now()
            )
            
            # Add to conversation history
            self.add_message_to_history(agent_message)
            
            # Extract tool calls from response
            tool_calls = self._extract_tool_calls(response)
            
            # Create agent response
            agent_response = AgentResponse(
                message=agent_message,
                processing_time=time.time() - start_time,
                tool_calls=tool_calls,
                reasoning=self._extract_reasoning(content),
                confidence=0.8  # Default confidence for Swarm
            )
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in Swarm turn processing: {e}")
            
            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"Swarm error: {str(e)}",
                timestamp=datetime.now()
            )
            
            return AgentResponse(
                message=error_message,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def finalize_session(self) -> Dict[str, Any]:
        """Finalize Swarm session"""
        return {
            "adapter_type": "swarm",
            "agents_count": len(self.swarm_agents),
            "current_agent": self.current_agent.name if self.current_agent else None,
            "total_messages": len(self.conversation_history),
            "capabilities": self.capabilities.to_dict()
        }
    
    def _convert_tools_to_functions(self, tools: List[Tool]) -> List[Callable]:
        """Convert LoCoBench tools to Swarm functions"""
        
        swarm_functions = []
        
        for tool in tools:
            if not tool.enabled:
                continue
                
            for func in tool.get_functions():
                # Create a Swarm function
                def create_swarm_function(tool_ref, func_ref):
                    async def swarm_function(**kwargs):
                        """Swarm function wrapper"""
                        try:
                            result = await tool_ref.call_function(func_ref.name, kwargs)
                            return result.get("result", "")
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    # Set function metadata
                    swarm_function.__name__ = f"{tool_ref.name}_{func_ref.name}"
                    swarm_function.__doc__ = func_ref.description
                    
                    return swarm_function
                
                swarm_functions.append(create_swarm_function(tool, func))
        
        return swarm_functions
    
    def _create_swarm_agents(self, scenario_context: Dict[str, Any], functions: List[Callable]):
        """Create specialized Swarm agents based on scenario"""
        
        scenario_type = scenario_context.get("category", "general")
        
        if scenario_type in ["collaborative_development", "team_based_development"]:
            self._create_development_swarm(functions)
        elif scenario_type in ["code_review", "debugging"]:
            self._create_review_swarm(functions)
        else:
            self._create_general_swarm(functions)
    
    def _create_development_swarm(self, functions: List[Callable]):
        """Create a development-focused swarm"""
        
        # Developer Agent
        developer = SwarmAgent(
            name="Developer",
            instructions="""You are a senior software developer. Your role is to:
            - Write clean, efficient, and maintainable code
            - Follow best practices and design patterns
            - Implement features according to requirements
            - Debug and fix issues when they arise
            
            When you need specialized help, transfer to the appropriate agent:
            - For architecture decisions: transfer to Architect
            - For code review: transfer to Reviewer""",
            functions=functions + [self._transfer_to_architect, self._transfer_to_reviewer]
        )
        
        # Architect Agent
        architect = SwarmAgent(
            name="Architect",
            instructions="""You are a software architect. Your role is to:
            - Design system architecture and high-level structure
            - Make technology and framework decisions
            - Ensure scalability and maintainability
            - Define coding standards and patterns
            
            Transfer back to Developer when implementation is needed.""",
            functions=functions + [self._transfer_to_developer]
        )
        
        # Reviewer Agent
        reviewer = SwarmAgent(
            name="Reviewer",
            instructions="""You are a code reviewer. Your role is to:
            - Review code for quality, security, and performance
            - Suggest improvements and optimizations
            - Ensure adherence to coding standards
            - Identify potential bugs and issues
            
            Transfer back to Developer for implementation of fixes.""",
            functions=functions + [self._transfer_to_developer]
        )
        
        self.swarm_agents = [developer, architect, reviewer]
    
    def _create_review_swarm(self, functions: List[Callable]):
        """Create a review-focused swarm"""
        
        # Primary Reviewer
        primary_reviewer = SwarmAgent(
            name="PrimaryReviewer",
            instructions="""You are the primary code reviewer. Your role is to:
            - Perform comprehensive code reviews
            - Check for bugs, security issues, and performance problems
            - Ensure code follows best practices
            - Coordinate with security specialist when needed
            
            Transfer to SecurityReviewer for security-specific concerns.""",
            functions=functions + [self._transfer_to_security]
        )
        
        # Security Reviewer
        security_reviewer = SwarmAgent(
            name="SecurityReviewer",
            instructions="""You are a security specialist. Your role is to:
            - Identify security vulnerabilities
            - Check for OWASP Top 10 issues
            - Ensure secure coding practices
            - Recommend security improvements
            
            Transfer back to PrimaryReviewer for general review tasks.""",
            functions=functions + [self._transfer_to_primary]
        )
        
        self.swarm_agents = [primary_reviewer, security_reviewer]
    
    def _create_general_swarm(self, functions: List[Callable]):
        """Create a general purpose swarm"""
        
        # General Assistant
        assistant = SwarmAgent(
            name="Assistant",
            instructions="""You are a helpful AI assistant specialized in software development.
            You can help with coding, debugging, architecture design, and technical questions.
            You provide clear, accurate, and helpful responses.""",
            functions=functions
        )
        
        self.swarm_agents = [assistant]
    
    def _transfer_to_developer(self):
        """Transfer to Developer agent"""
        return self.swarm_agents[0]  # Developer is first
    
    def _transfer_to_architect(self):
        """Transfer to Architect agent"""
        return self.swarm_agents[1] if len(self.swarm_agents) > 1 else self.swarm_agents[0]
    
    def _transfer_to_reviewer(self):
        """Transfer to Reviewer agent"""
        return self.swarm_agents[2] if len(self.swarm_agents) > 2 else self.swarm_agents[0]
    
    def _transfer_to_primary(self):
        """Transfer to Primary Reviewer agent"""
        return self.swarm_agents[0]  # Primary is first
    
    def _transfer_to_security(self):
        """Transfer to Security Reviewer agent"""
        return self.swarm_agents[1] if len(self.swarm_agents) > 1 else self.swarm_agents[0]
    
    def _prepare_messages(self, current_message: str) -> List[Dict[str, Any]]:
        """Prepare messages for Swarm"""
        
        messages = []
        
        # Add conversation history
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            if msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    def _extract_tool_calls(self, response) -> List[ToolCall]:
        """Extract tool calls from Swarm response"""
        
        tool_calls = []
        
        # Swarm doesn't directly expose tool calls in the response
        # This would need to be implemented based on the response structure
        # For now, return empty list
        
        return tool_calls
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from Swarm response"""
        if not content:
            return None
        
        # Look for reasoning patterns in Swarm responses
        reasoning_patterns = [
            "I need to", "Let me", "I'll", "My approach", "The plan is",
            "First, I", "I should", "Based on", "Given that"
        ]
        
        sentences = content.split('.')
        reasoning_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(pattern.lower() in sentence.lower() for pattern in reasoning_patterns):
                reasoning_sentences.append(sentence)
        
        return '. '.join(reasoning_sentences) if reasoning_sentences else None
