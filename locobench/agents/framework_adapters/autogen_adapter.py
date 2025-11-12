"""
Microsoft AutoGen Agent Adapter for LoCoBench-Agent

This adapter integrates Microsoft AutoGen agents into the LoCoBench-Agent
evaluation framework, enabling evaluation of multi-agent conversations
and group chat capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import autogen
    from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    autogen = None
    ConversableAgent = None
    UserProxyAgent = None
    GroupChat = None
    GroupChatManager = None

from ..base_agent import (
    BaseAgent, AgentResponse, AgentMessage, MessageRole, 
    ToolCall, ToolResponse, AgentCapabilities, AgentCapability
)
from ...core.tool_registry import Tool

logger = logging.getLogger(__name__)


class AutoGenAdapter(BaseAgent):
    """
    Adapter for Microsoft AutoGen agents
    
    This adapter wraps AutoGen's multi-agent system to work with
    LoCoBench-Agent's evaluation framework.
    """
    
    def __init__(
        self,
        name: str = "AutoGen Agent",
        autogen_config: Dict[str, Any] = None,
        config: Dict[str, Any] = None
    ):
        if autogen is None:
            raise ImportError("AutoGen package not installed. Run: pip install pyautogen")
        
        super().__init__(name, config)
        
        self.autogen_config = autogen_config or {}
        self.agents = []
        self.group_chat = None
        self.group_chat_manager = None
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.MULTI_AGENT,
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_EXECUTION,
                AgentCapability.COLLABORATION,
                AgentCapability.REASONING
            ],
            max_context_tokens=self.autogen_config.get("max_tokens", 4096),
            supports_function_calling=True,
            supports_tool_usage=True,
        )
        
        logger.info(f"Initialized AutoGen adapter: {name}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize AutoGen multi-agent session"""
        try:
            # Create AutoGen agents based on configuration
            self._create_autogen_agents(scenario_context, available_tools)
            
            # Set up group chat if multiple agents
            if len(self.agents) > 1:
                self.group_chat = GroupChat(
                    agents=self.agents,
                    messages=[],
                    max_round=self.autogen_config.get("max_rounds", 10)
                )
                
                self.group_chat_manager = GroupChatManager(
                    groupchat=self.group_chat,
                    llm_config=self.autogen_config.get("llm_config", {})
                )
            
            logger.info(f"AutoGen session initialized with {len(self.agents)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen session: {e}")
            return False
    
    async def process_turn(
        self,
        message: str,
        available_tools: List[Tool] = None,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process a turn using AutoGen agents"""
        start_time = time.time()
        
        try:
            if not self.agents:
                raise RuntimeError("AutoGen agents not initialized")
            
            # For single agent, use direct conversation
            if len(self.agents) == 1:
                response_message = await self._single_agent_turn(message)
            else:
                # For multiple agents, use group chat
                response_message = await self._group_chat_turn(message)
            
            # Create agent response
            agent_response = AgentResponse(
                message=response_message,
                processing_time=time.time() - start_time,
                tool_calls=[],  # AutoGen handles tools internally
                reasoning=self._extract_reasoning(response_message.content),
                confidence=0.8  # Default confidence for AutoGen
            )
            
            # Add to conversation history
            self.add_message_to_history(response_message)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in AutoGen turn processing: {e}")
            
            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"AutoGen error: {str(e)}",
                timestamp=datetime.now()
            )
            
            return AgentResponse(
                message=error_message,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def finalize_session(self) -> Dict[str, Any]:
        """Finalize AutoGen session"""
        return {
            "adapter_type": "autogen",
            "agents_count": len(self.agents),
            "total_messages": len(self.conversation_history),
            "group_chat_enabled": self.group_chat is not None,
            "capabilities": self.capabilities.to_dict()
        }
    
    def _create_autogen_agents(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ):
        """Create AutoGen agents based on scenario"""
        
        # Default configuration
        llm_config = self.autogen_config.get("llm_config", {
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": self.autogen_config.get("api_key", ""),
                }
            ],
            "temperature": 0.1,
        })
        
        # Create different agent types based on scenario
        scenario_type = scenario_context.get("category", "general")
        
        if scenario_type in ["collaborative_development", "team_based_development"]:
            # Create a development team
            self._create_development_team(llm_config, available_tools)
        elif scenario_type in ["code_review", "debugging"]:
            # Create reviewer and developer agents
            self._create_review_team(llm_config, available_tools)
        else:
            # Create a general purpose agent
            self._create_general_agent(llm_config, available_tools)
    
    def _create_development_team(self, llm_config: Dict, available_tools: List[Tool]):
        """Create a development team with specialized roles"""
        
        # Software Engineer Agent
        engineer = ConversableAgent(
            name="SoftwareEngineer",
            system_message="""You are a senior software engineer. Your role is to:
            - Write high-quality, well-structured code
            - Follow best practices and design patterns
            - Ensure code is maintainable and scalable
            - Collaborate effectively with team members""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Code Reviewer Agent
        reviewer = ConversableAgent(
            name="CodeReviewer",
            system_message="""You are an expert code reviewer. Your role is to:
            - Review code for quality, security, and performance
            - Suggest improvements and optimizations
            - Ensure adherence to coding standards
            - Provide constructive feedback""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Project Manager Agent
        manager = ConversableAgent(
            name="ProjectManager",
            system_message="""You are a technical project manager. Your role is to:
            - Coordinate between team members
            - Break down complex tasks into manageable parts
            - Ensure project requirements are met
            - Facilitate communication and decision-making""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        self.agents = [engineer, reviewer, manager]
    
    def _create_review_team(self, llm_config: Dict, available_tools: List[Tool]):
        """Create a code review team"""
        
        # Developer Agent
        developer = ConversableAgent(
            name="Developer",
            system_message="""You are a software developer working on a project.
            You write code, fix bugs, and implement features.
            You respond to feedback and make improvements as suggested.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Senior Reviewer Agent
        senior_reviewer = ConversableAgent(
            name="SeniorReviewer",
            system_message="""You are a senior software engineer doing code review.
            You carefully examine code for bugs, performance issues, security problems,
            and adherence to best practices. You provide detailed, actionable feedback.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        self.agents = [developer, senior_reviewer]
    
    def _create_general_agent(self, llm_config: Dict, available_tools: List[Tool]):
        """Create a general purpose agent"""
        
        # General Assistant Agent
        assistant = ConversableAgent(
            name="Assistant",
            system_message="""You are a helpful AI assistant specializing in software development.
            You can help with coding, debugging, architecture design, and technical questions.
            You provide clear, accurate, and helpful responses.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        self.agents = [assistant]
    
    async def _single_agent_turn(self, message: str) -> AgentMessage:
        """Handle a turn with a single AutoGen agent"""
        
        agent = self.agents[0]
        
        # Create a simple conversation
        try:
            # AutoGen's generate_reply method
            reply = agent.generate_reply(
                messages=[{"content": message, "role": "user"}],
                sender=None
            )
            
            content = reply if isinstance(reply, str) else str(reply)
            
            return AgentMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in single agent turn: {e}")
            return AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"Error: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _group_chat_turn(self, message: str) -> AgentMessage:
        """Handle a turn with AutoGen group chat"""
        
        try:
            if not self.group_chat_manager:
                raise RuntimeError("Group chat manager not initialized")
            
            # Add user message to group chat
            user_message = {"content": message, "role": "user", "name": "User"}
            self.group_chat.messages.append(user_message)
            
            # Generate group chat response
            reply = self.group_chat_manager.generate_reply(
                messages=self.group_chat.messages,
                sender=None
            )
            
            content = reply if isinstance(reply, str) else str(reply)
            
            return AgentMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in group chat turn: {e}")
            return AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"Group chat error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from AutoGen response"""
        if not content:
            return None
        
        # Look for reasoning patterns in AutoGen responses
        reasoning_indicators = [
            "Let me think", "First, I'll", "My approach is", "I need to",
            "The solution is", "Here's how", "I'll start by"
        ]
        
        sentences = content.split('.')
        reasoning_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator.lower() in sentence.lower() for indicator in reasoning_indicators):
                reasoning_sentences.append(sentence)
        
        return '. '.join(reasoning_sentences) if reasoning_sentences else None
