"""
CrewAI Agent Adapter for LoCoBench-Agent

This adapter integrates CrewAI agents into the LoCoBench-Agent
evaluation framework, supporting role-based multi-agent systems,
task delegation, and hierarchical agent structures.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool as CrewAIBaseTool
except ImportError:
    Agent = None
    Task = None
    Crew = None
    Process = None
    CrewAIBaseTool = None

from ..base_agent import (
    BaseAgent, AgentResponse, AgentMessage, MessageRole, 
    ToolCall, ToolResponse, AgentCapabilities, AgentCapability
)
from ...core.tool_registry import Tool

logger = logging.getLogger(__name__)


class CrewAIToolWrapper(CrewAIBaseTool if CrewAIBaseTool else object):
    """Wrapper to adapt LoCoBench tools to CrewAI tools"""
    
    def __init__(self, locobench_tool: Tool):
        self.locobench_tool = locobench_tool
        if CrewAIBaseTool:
            super().__init__(
                name=locobench_tool.name,
                description=locobench_tool.description
            )
        else:
            self.name = locobench_tool.name
            self.description = locobench_tool.description
    
    def _run(self, tool_input: str) -> str:
        """Run the tool"""
        try:
            # Simple string input - parse as needed
            result = asyncio.run(self.locobench_tool.call_function(
                "default", {"input": tool_input}
            ))
            return str(result.get("result", ""))
        except Exception as e:
            return f"Tool error: {str(e)}"


class CrewAIAdapter(BaseAgent):
    """
    Adapter for CrewAI agents
    
    This adapter wraps CrewAI's role-based multi-agent system to work with
    LoCoBench-Agent's evaluation framework.
    """
    
    def __init__(
        self,
        name: str = "CrewAI Agent",
        crewai_config: Dict[str, Any] = None,
        config: Dict[str, Any] = None
    ):
        if Agent is None:
            raise ImportError("CrewAI package not installed. Run: pip install crewai")
        
        super().__init__(name, config)
        
        self.crewai_config = crewai_config or {}
        self.crew = None
        self.agents_list = []
        self.current_task = None
        
        # Capabilities
        self.capabilities = AgentCapabilities(
            supported_capabilities=[
                AgentCapability.MULTI_AGENT,
                AgentCapability.ROLE_SPECIALIZATION,
                AgentCapability.TASK_DELEGATION,
                AgentCapability.COLLABORATION,
                AgentCapability.CODE_GENERATION,
                AgentCapability.REASONING
            ],
            max_context_tokens=self.crewai_config.get("max_tokens", 4096),
            supports_function_calling=True,
            supports_tool_usage=True,
        )
        
        logger.info(f"Initialized CrewAI adapter: {name}")
    
    async def initialize_session(
        self,
        scenario_context: Dict[str, Any],
        available_tools: List[Tool] = None
    ) -> bool:
        """Initialize CrewAI session"""
        try:
            # Convert LoCoBench tools to CrewAI tools
            crewai_tools = []
            if available_tools:
                for tool in available_tools:
                    if tool.enabled:
                        crewai_tools.append(CrewAIToolWrapper(tool))
            
            # Create specialized agents based on scenario
            self._create_specialized_agents(scenario_context, crewai_tools)
            
            # Create crew
            self._create_crew(scenario_context)
            
            logger.info(f"CrewAI session initialized with {len(self.agents_list)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CrewAI session: {e}")
            return False
    
    async def process_turn(
        self,
        message: str,
        available_tools: List[Tool] = None,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """Process a turn using CrewAI crew"""
        start_time = time.time()
        
        try:
            if not self.crew:
                raise RuntimeError("CrewAI crew not initialized")
            
            # Create a task from the message
            task = Task(
                description=message,
                expected_output="A comprehensive response addressing the user's request",
                agent=self.agents_list[0]  # Assign to first agent
            )
            
            # Execute the task with the crew
            result = self.crew.kickoff(inputs={"task_description": message})
            
            # Extract response
            output = str(result) if result else "No output generated"
            
            # Create agent message
            agent_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=output,
                timestamp=datetime.now()
            )
            
            # Add to conversation history
            self.add_message_to_history(agent_message)
            
            # Create agent response
            agent_response = AgentResponse(
                message=agent_message,
                processing_time=time.time() - start_time,
                tool_calls=[],  # CrewAI handles tools internally
                reasoning=self._extract_reasoning(output),
                confidence=0.8  # Default confidence for CrewAI
            )
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in CrewAI turn processing: {e}")
            
            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=f"CrewAI error: {str(e)}",
                timestamp=datetime.now()
            )
            
            return AgentResponse(
                message=error_message,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def finalize_session(self) -> Dict[str, Any]:
        """Finalize CrewAI session"""
        return {
            "adapter_type": "crewai",
            "agents_count": len(self.agents_list),
            "total_messages": len(self.conversation_history),
            "process_type": self.crewai_config.get("process", "sequential"),
            "capabilities": self.capabilities.to_dict()
        }
    
    def _create_specialized_agents(self, scenario_context: Dict[str, Any], tools: List):
        """Create specialized agents based on scenario"""
        
        scenario_type = scenario_context.get("category", "general")
        
        if scenario_type in ["collaborative_development", "team_based_development"]:
            self._create_development_crew(tools)
        elif scenario_type in ["code_review", "security_analysis"]:
            self._create_review_crew(tools)
        elif scenario_type in ["debugging", "testing"]:
            self._create_qa_crew(tools)
        else:
            self._create_general_crew(tools)
    
    def _create_development_crew(self, tools: List):
        """Create a software development crew"""
        
        # Senior Developer
        senior_dev = Agent(
            role="Senior Software Developer",
            goal="Write high-quality, maintainable code following best practices",
            backstory="""You are a senior software developer with 10+ years of experience.
            You excel at designing clean architectures, writing efficient code, and mentoring others.
            You always consider scalability, maintainability, and performance in your solutions.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=True,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        # Code Architect
        architect = Agent(
            role="Software Architect",
            goal="Design robust system architectures and ensure technical excellence",
            backstory="""You are a software architect who specializes in designing scalable systems.
            You have deep knowledge of design patterns, system design, and technology trade-offs.
            You guide technical decisions and ensure architectural consistency.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=True,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        # Tech Lead
        tech_lead = Agent(
            role="Technical Lead",
            goal="Coordinate development efforts and ensure project success",
            backstory="""You are a technical lead who bridges the gap between management and development.
            You coordinate team efforts, make technical decisions, and ensure deliverables meet requirements.
            You're skilled at breaking down complex problems and managing technical debt.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=True,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        self.agents_list = [senior_dev, architect, tech_lead]
    
    def _create_review_crew(self, tools: List):
        """Create a code review crew"""
        
        # Code Reviewer
        reviewer = Agent(
            role="Senior Code Reviewer",
            goal="Ensure code quality, security, and adherence to best practices",
            backstory="""You are an expert code reviewer with a keen eye for detail.
            You identify bugs, security vulnerabilities, performance issues, and style violations.
            You provide constructive feedback and help improve code quality.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=False,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        # Security Analyst
        security_analyst = Agent(
            role="Security Analyst",
            goal="Identify and mitigate security vulnerabilities in code",
            backstory="""You are a cybersecurity expert specializing in application security.
            You understand common vulnerabilities like OWASP Top 10 and know how to prevent them.
            You perform security audits and recommend security best practices.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=False,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        self.agents_list = [reviewer, security_analyst]
    
    def _create_qa_crew(self, tools: List):
        """Create a QA and testing crew"""
        
        # QA Engineer
        qa_engineer = Agent(
            role="QA Engineer",
            goal="Ensure software quality through comprehensive testing",
            backstory="""You are a quality assurance engineer with expertise in testing methodologies.
            You design test cases, perform manual and automated testing, and identify defects.
            You understand different testing types and know how to ensure comprehensive coverage.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=False,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        # Test Automation Engineer
        automation_engineer = Agent(
            role="Test Automation Engineer",
            goal="Create and maintain automated test suites",
            backstory="""You are a test automation engineer who specializes in creating robust test frameworks.
            You build automated tests, CI/CD pipelines, and monitoring systems.
            You ensure tests are maintainable, reliable, and provide fast feedback.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=False,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        self.agents_list = [qa_engineer, automation_engineer]
    
    def _create_general_crew(self, tools: List):
        """Create a general purpose crew"""
        
        # General Assistant
        assistant = Agent(
            role="AI Development Assistant",
            goal="Provide helpful assistance with software development tasks",
            backstory="""You are an AI assistant specialized in software development.
            You can help with coding, debugging, architecture design, and technical questions.
            You provide clear, accurate, and helpful responses to development challenges.""",
            verbose=self.crewai_config.get("verbose", False),
            allow_delegation=False,
            tools=tools,
            llm=self.crewai_config.get("llm")
        )
        
        self.agents_list = [assistant]
    
    def _create_crew(self, scenario_context: Dict[str, Any]):
        """Create the CrewAI crew"""
        
        process_type = self.crewai_config.get("process", "sequential")
        
        if process_type == "hierarchical":
            crew_process = Process.hierarchical
        else:
            crew_process = Process.sequential
        
        self.crew = Crew(
            agents=self.agents_list,
            tasks=[],  # Tasks will be created dynamically
            process=crew_process,
            verbose=self.crewai_config.get("verbose", False),
            memory=self.crewai_config.get("enable_memory", False)
        )
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from CrewAI response"""
        if not content:
            return None
        
        # Look for reasoning patterns in CrewAI responses
        reasoning_indicators = [
            "Analysis:", "Approach:", "Strategy:", "Plan:", "Reasoning:",
            "Let me analyze", "My approach", "I'll start by", "The strategy is"
        ]
        
        lines = content.split('\n')
        reasoning_lines = []
        
        for line in lines:
            line = line.strip()
            if any(indicator in line for indicator in reasoning_indicators):
                reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines) if reasoning_lines else None
