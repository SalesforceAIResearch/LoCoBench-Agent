"""
Agent Factory for LoCoBench-Agent

This module provides factory methods for creating different types of agents
based on configuration specifications.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from .base_agent import BaseAgent, AgentCapabilities, AgentCapability

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents that can be created"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AUTOGEN = "autogen"
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    SWARM = "swarm"
    CUSTOM = "custom"


@dataclass
class AgentConfig:
    """Configuration for creating an agent"""
    agent_type: AgentType
    name: str = "Agent"
    
    # Model configuration
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    
    # Agent-specific settings
    supports_function_calling: bool = True
    supports_streaming: bool = False
    max_context_tokens: int = 128000
    
    # Framework-specific settings (for existing agent frameworks)
    framework_config: Dict[str, Any] = field(default_factory=dict)
    
    # Additional configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "supports_function_calling": self.supports_function_calling,
            "supports_streaming": self.supports_streaming,
            "max_context_tokens": self.max_context_tokens,
            "framework_config": self.framework_config,
            "custom_config": self.custom_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        return cls(
            agent_type=AgentType(data["agent_type"]),
            name=data.get("name", "Agent"),
            model_name=data.get("model_name"),
            temperature=data.get("temperature", 0.1),
            max_tokens=data.get("max_tokens", 4096),
            supports_function_calling=data.get("supports_function_calling", True),
            supports_streaming=data.get("supports_streaming", False),
            max_context_tokens=data.get("max_context_tokens", 128000),
            framework_config=data.get("framework_config", {}),
            custom_config=data.get("custom_config", {})
        )


class AgentFactory:
    """Factory for creating different types of agents"""
    
    # Default model configurations
    DEFAULT_MODELS = {
        AgentType.OPENAI: "gpt-4o",
        AgentType.ANTHROPIC: "claude-sonnet-4",
        AgentType.GOOGLE: "gemini-2.5-pro"
    }
    
    # Environment variable mappings for API keys
    API_KEY_ENV_VARS = {
        AgentType.OPENAI: "OPENAI_API_KEY",
        AgentType.ANTHROPIC: "ANTHROPIC_API_KEY",
        AgentType.GOOGLE: "GOOGLE_API_KEY"
    }
    
    # OpenAI Gateway configuration (Salesforce Research)
    # Set OPENAI_BASE_URL to use the gateway instead of direct OpenAI API
    # Example: https://gateway.salesforceresearch.ai/openai/process/v1
    OPENAI_GATEWAY_ENV_VAR = "OPENAI_BASE_URL"
    
    @staticmethod
    def create_agent(config: AgentConfig) -> BaseAgent:
        """
        Create an agent based on the provided configuration
        
        Args:
            config: Agent configuration specifying type and parameters
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent type is not supported
            ImportError: If required dependencies are not installed
        """
        # Set default model if not specified
        if not config.model_name and config.agent_type in AgentFactory.DEFAULT_MODELS:
            config.model_name = AgentFactory.DEFAULT_MODELS[config.agent_type]
        
        # Set API key from environment if not provided
        if not config.api_key and config.agent_type in AgentFactory.API_KEY_ENV_VARS:
            env_var = AgentFactory.API_KEY_ENV_VARS[config.agent_type]
            config.api_key = os.getenv(env_var)
        
        # Create agent based on type
        if config.agent_type == AgentType.OPENAI:
            return AgentFactory._create_openai_agent(config)
        elif config.agent_type == AgentType.ANTHROPIC:
            return AgentFactory._create_anthropic_agent(config)
        elif config.agent_type == AgentType.GOOGLE:
            return AgentFactory._create_google_agent(config)
        elif config.agent_type == AgentType.AUTOGEN:
            return AgentFactory._create_autogen_agent(config)
        elif config.agent_type == AgentType.LANGCHAIN:
            return AgentFactory._create_langchain_agent(config)
        elif config.agent_type == AgentType.CREWAI:
            return AgentFactory._create_crewai_agent(config)
        elif config.agent_type == AgentType.SWARM:
            return AgentFactory._create_swarm_agent(config)
        elif config.agent_type == AgentType.CUSTOM:
            return AgentFactory._create_custom_agent(config)
        else:
            raise ValueError(f"Unsupported agent type: {config.agent_type}")
    
    @staticmethod
    def _create_openai_agent(config: AgentConfig) -> BaseAgent:
        """Create an OpenAI agent with optional gateway support"""
        try:
            from .openai_agent import OpenAIAgent
            
            # Check for OpenAI gateway base URL from environment or config
            base_url = os.getenv(AgentFactory.OPENAI_GATEWAY_ENV_VAR)
            if not base_url and config.custom_config:
                base_url = config.custom_config.get("base_url")
            
            if base_url:
                logger.info(f"Using OpenAI Gateway: {base_url}")
            
            return OpenAIAgent(
                name=config.name,
                model=config.model_name or "gpt-4o",
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                config=config.custom_config,
                base_url=base_url
            )
        except ImportError as e:
            raise ImportError(f"OpenAI dependencies not available: {e}")
    
    @staticmethod
    def _create_anthropic_agent(config: AgentConfig) -> BaseAgent:
        """Create an Anthropic agent"""
        try:
            from .anthropic_agent import AnthropicAgent
            
            return AnthropicAgent(
                name=config.name,
                model=config.model_name or "claude-sonnet-4",
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                config=config.custom_config
            )
        except ImportError as e:
            raise ImportError(f"Anthropic dependencies not available: {e}")
    
    @staticmethod
    def _create_google_agent(config: AgentConfig) -> BaseAgent:
        """Create a Google agent"""
        try:
            from .google_agent import GoogleAgent
            
            return GoogleAgent(
                name=config.name,
                model=config.model_name or "gemini-2.5-pro",
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                config=config.custom_config
            )
        except ImportError as e:
            raise ImportError(f"Google AI dependencies not available: {e}")
    
    @staticmethod
    def _create_autogen_agent(config: AgentConfig) -> BaseAgent:
        """Create an AutoGen agent adapter"""
        try:
            from .framework_adapters import AutoGenAdapter
            
            autogen_config = config.custom_config or {}
            autogen_config.update({
                "api_key": config.api_key,
                "llm_config": {
                    "config_list": [
                        {
                            "model": config.model_name or "gpt-4",
                            "api_key": config.api_key,
                        }
                    ],
                    "temperature": config.temperature,
                },
                "max_rounds": autogen_config.get("max_rounds", 10)
            })
            
            return AutoGenAdapter(
                name=config.name,
                autogen_config=autogen_config,
                config=config.custom_config
            )
        except ImportError as e:
            raise ImportError(f"AutoGen dependencies not available: {e}")
    
    @staticmethod
    def _create_langchain_agent(config: AgentConfig) -> BaseAgent:
        """Create a LangChain agent adapter"""
        try:
            from .framework_adapters import LangChainAdapter
            
            langchain_config = config.custom_config or {}
            langchain_config.update({
                "provider": "openai",  # Default to OpenAI
                "model_name": config.model_name or "gpt-4",
                "api_key": config.api_key,
                "temperature": config.temperature,
                "enable_memory": langchain_config.get("enable_memory", True),
                "memory_window": langchain_config.get("memory_window", 10),
                "max_iterations": langchain_config.get("max_iterations", 10)
            })
            
            return LangChainAdapter(
                name=config.name,
                langchain_config=langchain_config,
                config=config.custom_config
            )
        except ImportError as e:
            raise ImportError(f"LangChain dependencies not available: {e}")
    
    @staticmethod
    def _create_crewai_agent(config: AgentConfig) -> BaseAgent:
        """Create a CrewAI agent adapter"""
        try:
            from .framework_adapters import CrewAIAdapter
            
            crewai_config = config.custom_config or {}
            crewai_config.update({
                "llm": None,  # Will be set based on model_name
                "process": crewai_config.get("process", "sequential"),
                "enable_memory": crewai_config.get("enable_memory", False),
                "verbose": crewai_config.get("verbose", False)
            })
            
            return CrewAIAdapter(
                name=config.name,
                crewai_config=crewai_config,
                config=config.custom_config
            )
        except ImportError as e:
            raise ImportError(f"CrewAI dependencies not available: {e}")
    
    @staticmethod
    def _create_swarm_agent(config: AgentConfig) -> BaseAgent:
        """Create a Swarm agent adapter"""
        try:
            from .framework_adapters import SwarmAdapter
            
            swarm_config = config.custom_config or {}
            swarm_config.update({
                "model_name": config.model_name or "gpt-4",
                "api_key": config.api_key,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            })
            
            return SwarmAdapter(
                name=config.name,
                swarm_config=swarm_config,
                config=config.custom_config
            )
        except ImportError as e:
            raise ImportError(f"Swarm dependencies not available: {e}")
    
    @staticmethod
    def _create_custom_agent(config: AgentConfig) -> BaseAgent:
        """Create a custom agent"""
        # This would be implemented based on specific custom agent requirements
        raise NotImplementedError("Custom agent creation not yet implemented")
    
    @staticmethod
    def get_available_agent_types() -> List[AgentType]:
        """Get list of available agent types based on installed dependencies"""
        available_types = []
        
        # Check OpenAI
        try:
            import openai
            available_types.append(AgentType.OPENAI)
        except ImportError:
            pass
        
        # Check Anthropic
        try:
            import anthropic
            available_types.append(AgentType.ANTHROPIC)
        except ImportError:
            pass
        
        # Check Google
        try:
            import google.generativeai
            available_types.append(AgentType.GOOGLE)
        except ImportError:
            pass
        
        # Check AutoGen
        try:
            import autogen
            available_types.append(AgentType.AUTOGEN)
        except ImportError:
            pass
        
        # Check LangChain
        try:
            import langchain
            available_types.append(AgentType.LANGCHAIN)
        except ImportError:
            pass
        
        # Check CrewAI
        try:
            import crewai
            available_types.append(AgentType.CREWAI)
        except ImportError:
            pass
        
        # Custom is always available
        available_types.append(AgentType.CUSTOM)
        
        return available_types
    
    @staticmethod
    def create_agent_configs_for_comparison(
        agent_types: List[AgentType] = None,
        models: Dict[AgentType, List[str]] = None
    ) -> List[AgentConfig]:
        """
        Create a set of agent configurations for comparison evaluation
        
        Args:
            agent_types: List of agent types to include (default: all available)
            models: Dictionary mapping agent types to model lists
            
        Returns:
            List of agent configurations for comparison
        """
        if agent_types is None:
            agent_types = AgentFactory.get_available_agent_types()
        
        if models is None:
            models = {
                AgentType.OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                AgentType.ANTHROPIC: ["claude-sonnet-4", "claude-opus-4"],
                AgentType.GOOGLE: ["gemini-2.5-pro", "gemini-2.5-flash"]
            }
        
        configs = []
        
        for agent_type in agent_types:
            if agent_type in models:
                for model in models[agent_type]:
                    config = AgentConfig(
                        agent_type=agent_type,
                        name=f"{agent_type.value.title()} {model}",
                        model_name=model
                    )
                    configs.append(config)
            else:
                # For framework-based agents, create with default settings
                config = AgentConfig(
                    agent_type=agent_type,
                    name=f"{agent_type.value.title()} Agent"
                )
                configs.append(config)
        
        return configs
    
    @staticmethod
    def validate_agent_config(config: AgentConfig) -> List[str]:
        """
        Validate an agent configuration
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check if agent type is available
        available_types = AgentFactory.get_available_agent_types()
        if config.agent_type not in available_types:
            errors.append(f"Agent type {config.agent_type.value} is not available. "
                         f"Available types: {[t.value for t in available_types]}")
        
        # Check API key for LLM-based agents
        if config.agent_type in [AgentType.OPENAI, AgentType.ANTHROPIC, AgentType.GOOGLE]:
            if not config.api_key:
                env_var = AgentFactory.API_KEY_ENV_VARS.get(config.agent_type)
                if not env_var or not os.getenv(env_var):
                    errors.append(f"API key required for {config.agent_type.value}. "
                                 f"Set {env_var} environment variable or provide api_key in config.")
        
        # Validate temperature
        if not 0.0 <= config.temperature <= 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        
        # Validate max_tokens
        if config.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        return errors


# Convenience functions
def create_openai_agent(
    model: str = "gpt-4o",
    name: str = "OpenAI Agent",
    api_key: str = None,
    **kwargs
) -> BaseAgent:
    """Convenience function to create an OpenAI agent"""
    config = AgentConfig(
        agent_type=AgentType.OPENAI,
        name=name,
        model_name=model,
        api_key=api_key,
        **kwargs
    )
    return AgentFactory.create_agent(config)


def create_anthropic_agent(
    model: str = "claude-sonnet-4",
    name: str = "Anthropic Agent",
    api_key: str = None,
    **kwargs
) -> BaseAgent:
    """Convenience function to create an Anthropic agent"""
    config = AgentConfig(
        agent_type=AgentType.ANTHROPIC,
        name=name,
        model_name=model,
        api_key=api_key,
        **kwargs
    )
    return AgentFactory.create_agent(config)


def create_google_agent(
    model: str = "gemini-2.5-pro",
    name: str = "Google Agent",
    api_key: str = None,
    **kwargs
) -> BaseAgent:
    """Convenience function to create a Google agent"""
    config = AgentConfig(
        agent_type=AgentType.GOOGLE,
        name=name,
        model_name=model,
        api_key=api_key,
        **kwargs
    )
    return AgentFactory.create_agent(config)
