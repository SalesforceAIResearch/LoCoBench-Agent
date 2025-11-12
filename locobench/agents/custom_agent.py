"""
Custom Agent Implementation for LoCoBench-Agent

This module provides base classes and utilities for implementing custom agents,
including research agents, specialized agents, and custom integrations.
"""

import asyncio
import json
import logging
from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentResponse, AgentMessage, ToolCall, ToolResponse

logger = logging.getLogger(__name__)


@dataclass
class CustomAgentConfig:
    """Configuration for custom agents"""
    
    # Basic configuration
    name: str
    description: str
    version: str = "1.0.0"
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    supported_languages: List[str] = field(default_factory=lambda: ["python"])
    
    # Behavior configuration
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_memory: bool = True
    enable_learning: bool = False
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Integration settings
    external_apis: Dict[str, str] = field(default_factory=dict)
    plugins: List[str] = field(default_factory=list)


class CustomAgent(BaseAgent):
    """
    Base class for custom agent implementations
    
    This class provides a foundation for implementing custom agents with
    specialized behaviors, external integrations, or research-specific features.
    """
    
    def __init__(self, config: CustomAgentConfig):
        super().__init__(config.name)
        self.config = config
        
        # Custom agent state
        self.memory: Dict[str, Any] = {}
        self.learning_history: List[Dict[str, Any]] = []
        self.external_connections: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            "total_messages": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_response_time": 0.0
        }
        
        logger.info(f"Custom agent '{self.name}' initialized")
    
    async def process_message(
        self,
        message: str,
        tools: List[Any],
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process a message with custom agent logic"""
        
        start_time = datetime.now()
        
        try:
            # Update performance metrics
            self.performance_metrics["total_messages"] += 1
            
            # Store in memory if enabled
            if self.config.enable_memory:
                self._store_in_memory("input_message", message, context)
            
            # Custom preprocessing
            processed_message = await self._preprocess_message(message, context)
            
            # Main processing logic (implemented by subclasses)
            response = await self._process_custom_logic(processed_message, tools, context)
            
            # Custom postprocessing
            final_response = await self._postprocess_response(response, context)
            
            # Update learning if enabled
            if self.config.enable_learning:
                await self._update_learning(message, final_response, context)
            
            # Update performance metrics
            self.performance_metrics["successful_operations"] += 1
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in custom agent processing: {e}")
            self.performance_metrics["failed_operations"] += 1
            
            return AgentResponse(
                content=f"I encountered an error while processing your request: {str(e)}",
                tool_calls=[],
                metadata={"error": str(e), "agent_type": "custom"}
            )
        
        finally:
            # Update response time
            duration = (datetime.now() - start_time).total_seconds()
            current_avg = self.performance_metrics["avg_response_time"]
            total_messages = self.performance_metrics["total_messages"]
            
            self.performance_metrics["avg_response_time"] = (
                (current_avg * (total_messages - 1) + duration) / total_messages
            )
    
    @abstractmethod
    async def _process_custom_logic(
        self,
        message: str,
        tools: List[Any],
        context: Dict[str, Any]
    ) -> AgentResponse:
        """
        Implement custom processing logic
        
        This method must be implemented by subclasses to define the specific
        behavior and capabilities of the custom agent.
        """
        pass
    
    async def _preprocess_message(self, message: str, context: Dict[str, Any]) -> str:
        """Preprocess the input message (can be overridden)"""
        
        # Default preprocessing: basic cleanup
        processed = message.strip()
        
        # Add context-aware preprocessing if needed
        if context.get("scenario_type") == "debugging":
            processed = f"[DEBUG MODE] {processed}"
        
        return processed
    
    async def _postprocess_response(self, response: AgentResponse, context: Dict[str, Any]) -> AgentResponse:
        """Postprocess the response (can be overridden)"""
        
        # Default postprocessing: add metadata
        if not response.metadata:
            response.metadata = {}
        
        response.metadata.update({
            "agent_type": "custom",
            "agent_name": self.name,
            "agent_version": self.config.version,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _store_in_memory(self, key: str, value: Any, context: Dict[str, Any]):
        """Store information in agent memory"""
        
        timestamp = datetime.now().isoformat()
        
        if key not in self.memory:
            self.memory[key] = []
        
        self.memory[key].append({
            "value": value,
            "context": context,
            "timestamp": timestamp
        })
        
        # Limit memory size
        max_memory_items = self.config.custom_parameters.get("max_memory_items", 100)
        if len(self.memory[key]) > max_memory_items:
            self.memory[key] = self.memory[key][-max_memory_items:]
    
    def _retrieve_from_memory(self, key: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve information from agent memory"""
        
        if key not in self.memory:
            return []
        
        return self.memory[key][-limit:]
    
    async def _update_learning(self, input_message: str, response: AgentResponse, context: Dict[str, Any]):
        """Update learning based on interaction"""
        
        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_message,
            "output": response.content,
            "context": context,
            "success": not bool(response.metadata.get("error"))
        }
        
        self.learning_history.append(learning_entry)
        
        # Limit learning history size
        max_learning_items = self.config.custom_parameters.get("max_learning_items", 1000)
        if len(self.learning_history) > max_learning_items:
            self.learning_history = self.learning_history[-max_learning_items:]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_memory_summary(self) -> Dict[str, int]:
        """Get summary of memory contents"""
        return {key: len(values) for key, values in self.memory.items()}
    
    def export_learning_data(self) -> List[Dict[str, Any]]:
        """Export learning history for analysis"""
        return self.learning_history.copy()


class TemplateCustomAgent(CustomAgent):
    """
    Template implementation of a custom agent
    
    This provides a working example that can be used as a starting point
    for implementing specific custom agents.
    """
    
    def __init__(self, config: CustomAgentConfig = None):
        if not config:
            config = CustomAgentConfig(
                name="TemplateAgent",
                description="Template custom agent for demonstration",
                capabilities=["text_processing", "tool_usage", "memory"]
            )
        
        super().__init__(config)
    
    async def _process_custom_logic(
        self,
        message: str,
        tools: List[Any],
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Template implementation of custom processing logic"""
        
        # Example: Simple rule-based processing
        response_content = await self._generate_template_response(message, context)
        
        # Example: Tool usage based on message content
        tool_calls = await self._determine_tool_usage(message, tools, context)
        
        return AgentResponse(
            content=response_content,
            tool_calls=tool_calls,
            metadata={
                "processing_method": "template_rules",
                "memory_items": len(self.memory),
                "capabilities_used": self._get_used_capabilities(message)
            }
        )
    
    async def _generate_template_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate response using template logic"""
        
        message_lower = message.lower()
        
        # Check memory for similar past interactions
        past_inputs = self._retrieve_from_memory("input_message", limit=5)
        similar_past = [
            item for item in past_inputs
            if any(word in item["value"].lower() for word in message_lower.split()[:3])
        ]
        
        # Generate response based on patterns
        if "debug" in message_lower or "error" in message_lower:
            response = "I'll help you debug this issue. Let me analyze the problem systematically."
            
            if similar_past:
                response += " I notice we've worked on similar debugging tasks before."
        
        elif "implement" in message_lower or "create" in message_lower:
            response = "I'll help you implement this feature. Let me break down the requirements and plan the implementation."
        
        elif "test" in message_lower:
            response = "I'll help you with testing. Let me identify what needs to be tested and create appropriate test cases."
        
        elif "review" in message_lower or "analyze" in message_lower:
            response = "I'll review and analyze the code. Let me examine the structure, patterns, and potential improvements."
        
        else:
            response = "I'll help you with this task. Let me understand the requirements and determine the best approach."
        
        # Add context-specific information
        if context.get("phase_name"):
            response += f" We're currently in the {context['phase_name']} phase."
        
        return response
    
    async def _determine_tool_usage(self, message: str, tools: List[Any], context: Dict[str, Any]) -> List[ToolCall]:
        """Determine which tools to use based on message content"""
        
        tool_calls = []
        message_lower = message.lower()
        
        # Map message patterns to tool usage
        if "file" in message_lower or "read" in message_lower:
            # Use file system tool
            for tool in tools:
                if hasattr(tool, 'name') and 'file' in tool.name.lower():
                    tool_calls.append(ToolCall(
                        tool_name=tool.name,
                        function_name="list_files",
                        arguments={"directory": "."},
                        call_id=f"call_{len(tool_calls)+1}"
                    ))
                    break
        
        if "compile" in message_lower or "build" in message_lower:
            # Use compiler tool
            for tool in tools:
                if hasattr(tool, 'name') and 'compiler' in tool.name.lower():
                    tool_calls.append(ToolCall(
                        tool_name=tool.name,
                        function_name="compile_code",
                        arguments={"language": "python", "source_files": []},
                        call_id=f"call_{len(tool_calls)+1}"
                    ))
                    break
        
        if "search" in message_lower or "find" in message_lower:
            # Use search tool
            for tool in tools:
                if hasattr(tool, 'name') and 'search' in tool.name.lower():
                    # Extract search terms from message
                    words = message.split()
                    search_terms = [word for word in words if len(word) > 3][:3]
                    
                    if search_terms:
                        tool_calls.append(ToolCall(
                            tool_name=tool.name,
                            function_name="search_text",
                            arguments={"query": " ".join(search_terms)},
                            call_id=f"call_{len(tool_calls)+1}"
                        ))
                    break
        
        return tool_calls
    
    def _get_used_capabilities(self, message: str) -> List[str]:
        """Determine which capabilities were used for this message"""
        
        used_capabilities = []
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["file", "read", "write"]):
            used_capabilities.append("file_operations")
        
        if any(word in message_lower for word in ["search", "find", "look"]):
            used_capabilities.append("search")
        
        if any(word in message_lower for word in ["analyze", "review", "check"]):
            used_capabilities.append("analysis")
        
        if len(self.memory) > 0:
            used_capabilities.append("memory")
        
        return used_capabilities


class ResearchCustomAgent(CustomAgent):
    """
    Custom agent designed for research scenarios
    
    This agent includes specialized capabilities for research tasks,
    hypothesis testing, and experimental design.
    """
    
    def __init__(self, research_domain: str = "software_engineering"):
        config = CustomAgentConfig(
            name=f"ResearchAgent_{research_domain}",
            description=f"Research-focused agent for {research_domain}",
            capabilities=[
                "hypothesis_generation",
                "experimental_design", 
                "data_analysis",
                "literature_review",
                "result_interpretation"
            ],
            enable_learning=True,
            custom_parameters={
                "research_domain": research_domain,
                "hypothesis_tracking": True,
                "experiment_logging": True
            }
        )
        
        super().__init__(config)
        
        # Research-specific state
        self.hypotheses: List[Dict[str, Any]] = []
        self.experiments: List[Dict[str, Any]] = []
        self.findings: List[Dict[str, Any]] = []
    
    async def _process_custom_logic(
        self,
        message: str,
        tools: List[Any],
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Research-focused processing logic"""
        
        # Identify research task type
        task_type = self._identify_research_task(message)
        
        if task_type == "hypothesis_generation":
            return await self._generate_hypothesis(message, tools, context)
        elif task_type == "experiment_design":
            return await self._design_experiment(message, tools, context)
        elif task_type == "data_analysis":
            return await self._analyze_data(message, tools, context)
        elif task_type == "literature_review":
            return await self._conduct_literature_review(message, tools, context)
        else:
            return await self._general_research_response(message, tools, context)
    
    def _identify_research_task(self, message: str) -> str:
        """Identify the type of research task"""
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hypothesis", "theory", "propose", "assume"]):
            return "hypothesis_generation"
        elif any(word in message_lower for word in ["experiment", "test", "validate", "verify"]):
            return "experiment_design"
        elif any(word in message_lower for word in ["analyze", "data", "results", "statistics"]):
            return "data_analysis"
        elif any(word in message_lower for word in ["literature", "papers", "research", "studies"]):
            return "literature_review"
        else:
            return "general_research"
    
    async def _generate_hypothesis(self, message: str, tools: List[Any], context: Dict[str, Any]) -> AgentResponse:
        """Generate research hypotheses"""
        
        # Extract key concepts from message
        concepts = self._extract_key_concepts(message)
        
        # Generate hypothesis based on concepts and domain knowledge
        hypothesis = {
            "id": f"hyp_{len(self.hypotheses)+1}",
            "statement": f"Based on the concepts {', '.join(concepts)}, I hypothesize that...",
            "concepts": concepts,
            "domain": self.config.custom_parameters["research_domain"],
            "generated_at": datetime.now().isoformat(),
            "status": "proposed"
        }
        
        self.hypotheses.append(hypothesis)
        
        response_content = f"I've generated a research hypothesis: {hypothesis['statement']}\n\n"
        response_content += f"Key concepts: {', '.join(concepts)}\n"
        response_content += f"This hypothesis can be tested through controlled experiments."
        
        return AgentResponse(
            content=response_content,
            tool_calls=[],
            metadata={
                "task_type": "hypothesis_generation",
                "hypothesis_id": hypothesis["id"],
                "concepts": concepts
            }
        )
    
    async def _design_experiment(self, message: str, tools: List[Any], context: Dict[str, Any]) -> AgentResponse:
        """Design research experiments"""
        
        experiment = {
            "id": f"exp_{len(self.experiments)+1}",
            "description": "Experimental design based on research question",
            "variables": self._identify_variables(message),
            "methodology": "Controlled experiment with statistical analysis",
            "designed_at": datetime.now().isoformat(),
            "status": "designed"
        }
        
        self.experiments.append(experiment)
        
        response_content = f"I've designed an experiment (ID: {experiment['id']}):\n\n"
        response_content += f"Variables to test: {', '.join(experiment['variables'])}\n"
        response_content += f"Methodology: {experiment['methodology']}\n"
        response_content += f"This experiment will help validate our hypotheses."
        
        # Suggest tools for data collection
        tool_calls = []
        for tool in tools:
            if hasattr(tool, 'name') and any(keyword in tool.name.lower() for keyword in ['file', 'data', 'analysis']):
                tool_calls.append(ToolCall(
                    tool_name=tool.name,
                    function_name="collect_data",
                    arguments={"experiment_id": experiment["id"]},
                    call_id=f"exp_call_{experiment['id']}"
                ))
                break
        
        return AgentResponse(
            content=response_content,
            tool_calls=tool_calls,
            metadata={
                "task_type": "experiment_design",
                "experiment_id": experiment["id"],
                "variables": experiment["variables"]
            }
        )
    
    async def _analyze_data(self, message: str, tools: List[Any], context: Dict[str, Any]) -> AgentResponse:
        """Analyze research data"""
        
        analysis_methods = ["statistical_analysis", "pattern_recognition", "correlation_analysis"]
        
        finding = {
            "id": f"finding_{len(self.findings)+1}",
            "description": "Data analysis results",
            "methods_used": analysis_methods,
            "confidence_level": 0.85,
            "analyzed_at": datetime.now().isoformat()
        }
        
        self.findings.append(finding)
        
        response_content = f"Data analysis completed (Finding ID: {finding['id']}):\n\n"
        response_content += f"Analysis methods: {', '.join(analysis_methods)}\n"
        response_content += f"Confidence level: {finding['confidence_level']:.2%}\n"
        response_content += f"The results provide insights into the research question."
        
        return AgentResponse(
            content=response_content,
            tool_calls=[],
            metadata={
                "task_type": "data_analysis",
                "finding_id": finding["id"],
                "confidence": finding["confidence_level"]
            }
        )
    
    async def _conduct_literature_review(self, message: str, tools: List[Any], context: Dict[str, Any]) -> AgentResponse:
        """Conduct literature review"""
        
        search_terms = self._extract_key_concepts(message)
        
        response_content = f"Conducting literature review on: {', '.join(search_terms)}\n\n"
        response_content += f"I'll search for relevant papers and studies in {self.config.custom_parameters['research_domain']}.\n"
        response_content += f"This will help establish the current state of knowledge and identify research gaps."
        
        # Use search tools if available
        tool_calls = []
        for tool in tools:
            if hasattr(tool, 'name') and 'search' in tool.name.lower():
                tool_calls.append(ToolCall(
                    tool_name=tool.name,
                    function_name="search_text",
                    arguments={"query": " ".join(search_terms)},
                    call_id=f"lit_search_{len(search_terms)}"
                ))
                break
        
        return AgentResponse(
            content=response_content,
            tool_calls=tool_calls,
            metadata={
                "task_type": "literature_review",
                "search_terms": search_terms
            }
        )
    
    async def _general_research_response(self, message: str, tools: List[Any], context: Dict[str, Any]) -> AgentResponse:
        """General research-oriented response"""
        
        response_content = f"As a research agent in {self.config.custom_parameters['research_domain']}, "
        response_content += f"I'll approach this systematically using scientific methodology.\n\n"
        
        if self.hypotheses:
            response_content += f"Current hypotheses: {len(self.hypotheses)}\n"
        
        if self.experiments:
            response_content += f"Designed experiments: {len(self.experiments)}\n"
        
        if self.findings:
            response_content += f"Research findings: {len(self.findings)}\n"
        
        response_content += f"\nLet me know how you'd like to proceed with the research."
        
        return AgentResponse(
            content=response_content,
            tool_calls=[],
            metadata={
                "task_type": "general_research",
                "research_state": {
                    "hypotheses": len(self.hypotheses),
                    "experiments": len(self.experiments),
                    "findings": len(self.findings)
                }
            }
        )
    
    def _extract_key_concepts(self, message: str) -> List[str]:
        """Extract key concepts from message"""
        
        # Simple keyword extraction (in practice, this could use NLP)
        words = message.lower().split()
        
        # Filter for meaningful terms
        concepts = [
            word for word in words
            if len(word) > 4 and word.isalpha()
        ]
        
        return concepts[:5]  # Return top 5 concepts
    
    def _identify_variables(self, message: str) -> List[str]:
        """Identify experimental variables"""
        
        # Simple variable identification
        message_lower = message.lower()
        
        variables = []
        
        if "performance" in message_lower:
            variables.append("performance_metric")
        
        if "time" in message_lower:
            variables.append("time_measurement")
        
        if "accuracy" in message_lower:
            variables.append("accuracy_score")
        
        if "efficiency" in message_lower:
            variables.append("efficiency_measure")
        
        # Default variables if none identified
        if not variables:
            variables = ["independent_variable", "dependent_variable"]
        
        return variables
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research progress"""
        
        return {
            "domain": self.config.custom_parameters["research_domain"],
            "hypotheses": len(self.hypotheses),
            "experiments": len(self.experiments),
            "findings": len(self.findings),
            "recent_hypotheses": self.hypotheses[-3:] if self.hypotheses else [],
            "recent_experiments": self.experiments[-3:] if self.experiments else [],
            "recent_findings": self.findings[-3:] if self.findings else []
        }


class CustomAgentFactory:
    """Factory for creating custom agents"""
    
    @staticmethod
    def create_template_agent(name: str = None) -> TemplateCustomAgent:
        """Create a template custom agent"""
        
        config = None
        if name:
            config = CustomAgentConfig(
                name=name,
                description=f"Template custom agent: {name}",
                capabilities=["text_processing", "tool_usage", "memory"]
            )
        
        return TemplateCustomAgent(config)
    
    @staticmethod
    def create_research_agent(domain: str = "software_engineering") -> ResearchCustomAgent:
        """Create a research-focused custom agent"""
        
        return ResearchCustomAgent(domain)
    
    @staticmethod
    def create_custom_agent(config: CustomAgentConfig) -> CustomAgent:
        """Create a custom agent with specific configuration"""
        
        # This would be extended to support different custom agent types
        # based on configuration parameters
        
        agent_type = config.custom_parameters.get("agent_type", "template")
        
        if agent_type == "research":
            domain = config.custom_parameters.get("research_domain", "software_engineering")
            return ResearchCustomAgent(domain)
        else:
            return TemplateCustomAgent(config)
