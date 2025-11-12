"""
Interactive Scenario Generator for LoCoBench-Agent

This module generates multi-turn, interactive evaluation scenarios for agent evaluation,
extending the original scenario generator with conversation phases, dynamic prompts,
and success criteria for agent-specific tasks.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.task import TaskCategory, DifficultyLevel
from ..core.agent_session import ConversationPhase
from .scenario_generator import ScenarioGenerator

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Types of agent interaction modes"""
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    COLLABORATIVE = "collaborative"
    SUPERVISED = "supervised"


class ToolUsageMode(Enum):
    """Tool usage restriction modes"""
    UNRESTRICTED = "unrestricted"
    LIMITED = "limited"
    PROGRESSIVE = "progressive"
    SPECIFIC = "specific"


@dataclass
class SuccessCriterion:
    """Defines success criteria for a conversation phase or scenario"""
    criterion_id: str
    description: str
    type: str  # "keyword", "action", "outcome", "metric"
    target_value: Any
    weight: float = 1.0
    required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "type": self.type,
            "target_value": self.target_value,
            "weight": self.weight,
            "required": self.required
        }


@dataclass
class DynamicPrompt:
    """Dynamic prompt that adapts based on agent behavior"""
    trigger_condition: str  # Condition that triggers this prompt
    prompt_text: str
    priority: int = 0
    max_uses: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_condition": self.trigger_condition,
            "prompt_text": self.prompt_text,
            "priority": self.priority,
            "max_uses": self.max_uses
        }


@dataclass
class InteractiveScenario:
    """Multi-turn interactive evaluation scenario for agents"""
    scenario_id: str
    title: str
    description: str
    category: TaskCategory
    difficulty: DifficultyLevel
    
    # Context and setup
    initial_context: Dict[str, str]  # Files and project state
    context_files: List[str]
    working_directory: str
    
    # Multi-turn structure
    conversation_phases: List[ConversationPhase]
    global_success_criteria: List[SuccessCriterion]
    available_tools: List[str]
    
    # Agent behavior settings
    interaction_mode: InteractionMode = InteractionMode.AUTONOMOUS
    tool_usage_mode: ToolUsageMode = ToolUsageMode.UNRESTRICTED
    
    # Session constraints
    max_turns: int = 50
    max_duration_minutes: int = 60
    context_window_tokens: int = 1_000_000
    
    # Dynamic adaptation
    dynamic_prompts: List[DynamicPrompt] = field(default_factory=list)
    human_intervention_triggers: List[str] = field(default_factory=list)
    
    # Evaluation metadata
    expected_outcomes: List[str] = field(default_factory=list)
    evaluation_focus: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "initial_context": self.initial_context,
            "context_files": self.context_files,
            "working_directory": self.working_directory,
            "conversation_phases": [phase.to_dict() for phase in self.conversation_phases],
            "global_success_criteria": [sc.to_dict() for sc in self.global_success_criteria],
            "available_tools": self.available_tools,
            "interaction_mode": self.interaction_mode.value,
            "tool_usage_mode": self.tool_usage_mode.value,
            "max_turns": self.max_turns,
            "max_duration_minutes": self.max_duration_minutes,
            "context_window_tokens": self.context_window_tokens,
            "dynamic_prompts": [dp.to_dict() for dp in self.dynamic_prompts],
            "human_intervention_triggers": self.human_intervention_triggers,
            "expected_outcomes": self.expected_outcomes,
            "evaluation_focus": self.evaluation_focus
        }


class InteractiveScenarioGenerator:
    """
    Generator for multi-turn interactive agent evaluation scenarios
    
    Extends the original scenario generator with agent-specific features:
    - Multi-phase conversations
    - Dynamic prompt adaptation
    - Tool usage patterns
    - Success criteria definition
    """
    
    def __init__(self, config, log_file: str = None):
        self.config = config
        self.base_generator = ScenarioGenerator(config, log_file)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
    
    async def generate_interactive_scenario(
        self,
        project_spec: Dict[str, Any],
        project_files: List[Dict[str, str]],
        task_category: TaskCategory,
        target_difficulty: DifficultyLevel,
        scenario_id: str
    ) -> InteractiveScenario:
        """Generate a complete interactive scenario"""
        
        self.logger.info(f"Generating interactive scenario: {scenario_id}")
        
        # Generate base scenario using original generator
        base_scenario = await self.base_generator._generate_single_scenario(
            scenario_id=scenario_id,
            task_category=task_category,
            project_spec=project_spec,
            project_files=project_files,
            project_stats={"files_count": len(project_files)},
            target_difficulty=target_difficulty
        )
        
        # Convert to interactive scenario
        interactive_scenario = await self._convert_to_interactive_scenario(
            base_scenario, project_spec, project_files, task_category, target_difficulty
        )
        
        # Generate conversation phases
        interactive_scenario.conversation_phases = await self._generate_conversation_phases(
            interactive_scenario, project_spec, project_files
        )
        
        # Generate success criteria
        interactive_scenario.global_success_criteria = await self._generate_success_criteria(
            interactive_scenario, task_category
        )
        
        # Generate dynamic prompts
        interactive_scenario.dynamic_prompts = await self._generate_dynamic_prompts(
            interactive_scenario, task_category
        )
        
        # Set tool usage patterns
        interactive_scenario.available_tools = self._determine_available_tools(
            task_category, target_difficulty
        )
        
        self.logger.info(f"Generated interactive scenario with {len(interactive_scenario.conversation_phases)} phases")
        
        return interactive_scenario
    
    async def _convert_to_interactive_scenario(
        self,
        base_scenario: Dict[str, Any],
        project_spec: Dict[str, Any],
        project_files: List[Dict[str, str]],
        task_category: TaskCategory,
        target_difficulty: DifficultyLevel
    ) -> InteractiveScenario:
        """Convert base scenario to interactive scenario"""
        
        return InteractiveScenario(
            scenario_id=base_scenario["id"],
            title=base_scenario["title"],
            description=base_scenario["description"],
            category=task_category,
            difficulty=target_difficulty,
            initial_context=base_scenario.get("context", {}),
            context_files=[f["path"] for f in project_files],
            working_directory=project_spec.get("name", "project"),
            conversation_phases=[],  # Will be generated
            global_success_criteria=[],  # Will be generated
            available_tools=[],  # Will be determined
            interaction_mode=self._determine_interaction_mode(task_category),
            tool_usage_mode=self._determine_tool_usage_mode(task_category, target_difficulty),
            max_turns=self._calculate_max_turns(target_difficulty),
            max_duration_minutes=self._calculate_max_duration(target_difficulty),
            context_window_tokens=self._calculate_context_window(target_difficulty),
            expected_outcomes=base_scenario.get("expected_outcomes", []),
            evaluation_focus=base_scenario.get("evaluation_focus", [])
        )
    
    async def _generate_conversation_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any],
        project_files: List[Dict[str, str]]
    ) -> List[ConversationPhase]:
        """Generate conversation phases for the scenario - supports all 12 extended categories"""
        
        phases = []
        
        # Map categories to phase generation methods
        if scenario.category in [TaskCategory.COLLABORATIVE_FEATURE_DEVELOPMENT, TaskCategory.FEATURE_IMPLEMENTATION]:
            phases = await self._generate_development_phases(scenario, project_spec)
        elif scenario.category in [TaskCategory.INTERACTIVE_DEBUGGING_SESSIONS, TaskCategory.BUG_INVESTIGATION]:
            phases = await self._generate_debugging_phases(scenario, project_files)
        elif scenario.category in [TaskCategory.INTERACTIVE_ARCHITECTURE_EXPLORATION, TaskCategory.ARCHITECTURAL_UNDERSTANDING]:
            phases = await self._generate_architecture_phases(scenario, project_spec)
        elif scenario.category == TaskCategory.HUMAN_AGENT_COLLABORATION:
            phases = await self._generate_collaboration_phases(scenario, project_spec)
        elif scenario.category in [TaskCategory.GUIDED_MULTI_FILE_REFACTORING, TaskCategory.CROSS_FILE_REFACTORING]:
            phases = await self._generate_refactoring_phases(scenario, project_files)
        elif scenario.category in [TaskCategory.EXTENDED_DEVELOPMENT_PROJECTS, TaskCategory.MULTI_SESSION_DEVELOPMENT]:
            phases = await self._generate_extended_project_phases(scenario, project_spec)
        elif scenario.category in [TaskCategory.INTERACTIVE_CODE_EXPLORATION, TaskCategory.CODE_COMPREHENSION]:
            phases = await self._generate_code_exploration_phases(scenario, project_files)
        elif scenario.category in [TaskCategory.TEST_DRIVEN_DEVELOPMENT_SESSIONS, TaskCategory.INTEGRATION_TESTING]:
            phases = await self._generate_tdd_phases(scenario, project_spec)
        elif scenario.category in [TaskCategory.INTERACTIVE_SECURITY_AUDITING, TaskCategory.SECURITY_ANALYSIS]:
            phases = await self._generate_security_audit_phases(scenario, project_files)
        elif scenario.category == TaskCategory.TOOL_MASTERY_EVALUATION:
            phases = await self._generate_tool_mastery_phases(scenario, project_spec)
        elif scenario.category == TaskCategory.DOCUMENTATION_GENERATION:
            phases = await self._generate_documentation_phases(scenario, project_files)
        elif scenario.category == TaskCategory.DEPLOYMENT_DEVOPS:
            phases = await self._generate_devops_phases(scenario, project_spec)
        else:
            # Generate generic phases
            phases = await self._generate_generic_phases(scenario, project_spec)
        
        return phases
    
    async def _generate_development_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for collaborative development scenarios"""
        
        phases = [
            ConversationPhase(
                phase_id="analysis",
                name="Project Analysis",
                initial_prompt=f"""
                Welcome to the collaborative development session! You'll be working on: {scenario.title}
                
                Project: {project_spec.get('name', 'Unknown')}
                Language: {project_spec.get('language', 'Unknown')}
                
                Please start by analyzing the existing codebase:
                1. Examine the project structure and files
                2. Understand the current architecture
                3. Identify the main components and their relationships
                4. Note any existing issues or areas for improvement
                
                Use the available tools to explore the codebase and provide your analysis.
                """,
                expected_actions=["read_file", "list_directory", "search_file"],
                success_conditions=["analysis", "structure", "components", "architecture"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            ),
            
            ConversationPhase(
                phase_id="planning",
                name="Implementation Planning",
                initial_prompt="""
                Based on your analysis, now create a detailed implementation plan:
                1. Break down the task into specific, manageable steps
                2. Identify which files need to be modified or created
                3. Consider dependencies and potential conflicts
                4. Outline the testing strategy
                5. Estimate the complexity and effort required
                
                Be specific about your approach and reasoning.
                """,
                expected_actions=["echo", "calculate"],
                success_conditions=["plan", "steps", "files", "testing", "approach"],
                max_turns_in_phase=6,
                phase_timeout_minutes=10
            ),
            
            ConversationPhase(
                phase_id="implementation",
                name="Code Implementation",
                initial_prompt="""
                Now implement your plan:
                1. Start with the most critical components
                2. Write clean, well-documented code
                3. Follow the project's existing patterns and conventions
                4. Test your changes as you go
                5. Handle edge cases and error conditions
                
                Use the development tools to write, compile, and test your code.
                """,
                expected_actions=["write_file", "compile_code", "run_program"],
                success_conditions=["implementation", "code", "working", "tested"],
                max_turns_in_phase=15,
                phase_timeout_minutes=25
            ),
            
            ConversationPhase(
                phase_id="validation",
                name="Testing and Validation",
                initial_prompt="""
                Validate your implementation:
                1. Run comprehensive tests to ensure functionality
                2. Check for any compilation or runtime errors
                3. Verify that requirements are met
                4. Test edge cases and error handling
                5. Ensure code quality and maintainability
                
                Document any issues found and how you resolved them.
                """,
                expected_actions=["run_tests", "compile_code", "run_program"],
                success_conditions=["tests", "validation", "working", "quality"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            )
        ]
        
        return phases
    
    async def _generate_debugging_phases(
        self,
        scenario: InteractiveScenario,
        project_files: List[Dict[str, str]]
    ) -> List[ConversationPhase]:
        """Generate phases for debugging scenarios"""
        
        phases = [
            ConversationPhase(
                phase_id="problem_identification",
                name="Problem Identification",
                initial_prompt=f"""
                You're tasked with debugging an issue in this codebase: {scenario.title}
                
                Start by identifying and understanding the problem:
                1. Examine the code and identify potential issues
                2. Look for error patterns, logic flaws, or inconsistencies
                3. Understand the expected vs actual behavior
                4. Gather information about the problem scope
                
                Use debugging tools to investigate the issue thoroughly.
                """,
                expected_actions=["read_file", "start_debug_session", "list"],
                success_conditions=["problem", "identified", "issue", "bug"],
                max_turns_in_phase=6,
                phase_timeout_minutes=12
            ),
            
            ConversationPhase(
                phase_id="investigation",
                name="Deep Investigation",
                initial_prompt="""
                Now investigate the problem more deeply:
                1. Set breakpoints at strategic locations
                2. Examine variable values and program state
                3. Trace the execution flow
                4. Identify the root cause of the issue
                5. Understand why the problem occurs
                
                Use the debugger to step through the code and gather evidence.
                """,
                expected_actions=["set_breakpoint", "step", "print", "where"],
                success_conditions=["root cause", "investigation", "evidence", "traced"],
                max_turns_in_phase=10,
                phase_timeout_minutes=18
            ),
            
            ConversationPhase(
                phase_id="solution",
                name="Solution Implementation",
                initial_prompt="""
                Implement a fix for the identified problem:
                1. Design a solution that addresses the root cause
                2. Make minimal, targeted changes
                3. Ensure the fix doesn't introduce new issues
                4. Test the solution thoroughly
                5. Document the changes made
                
                Implement and validate your solution.
                """,
                expected_actions=["write_file", "compile_code", "run_program", "run_tests"],
                success_conditions=["fixed", "solution", "working", "tested"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            )
        ]
        
        return phases
    
    async def _generate_architecture_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for architecture exploration scenarios"""
        
        phases = [
            ConversationPhase(
                phase_id="exploration",
                name="Architecture Exploration",
                initial_prompt=f"""
                Explore and understand the architecture of this system: {scenario.title}
                
                Project: {project_spec.get('name', 'Unknown')}
                
                Your task is to:
                1. Map out the overall system architecture
                2. Identify key components and their responsibilities
                3. Understand data flow and dependencies
                4. Analyze design patterns and architectural decisions
                5. Identify strengths and potential improvements
                
                Use IDE features to navigate and understand the codebase.
                """,
                expected_actions=["open_file", "go_to_definition", "find_references"],
                success_conditions=["architecture", "components", "mapped", "understood"],
                max_turns_in_phase=10,
                phase_timeout_minutes=20
            ),
            
            ConversationPhase(
                phase_id="analysis",
                name="Architectural Analysis",
                initial_prompt="""
                Analyze the architectural quality and design:
                1. Evaluate the separation of concerns
                2. Assess coupling and cohesion
                3. Identify design patterns used
                4. Look for architectural anti-patterns
                5. Consider scalability and maintainability
                
                Provide a comprehensive architectural assessment.
                """,
                expected_actions=["get_code_completion", "refactor_code"],
                success_conditions=["analysis", "quality", "patterns", "assessment"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            ),
            
            ConversationPhase(
                phase_id="recommendations",
                name="Improvement Recommendations",
                initial_prompt="""
                Based on your analysis, provide recommendations:
                1. Suggest architectural improvements
                2. Identify refactoring opportunities
                3. Recommend design pattern applications
                4. Propose scalability enhancements
                5. Prioritize improvements by impact and effort
                
                Create actionable recommendations with reasoning.
                """,
                expected_actions=["echo", "refactor_code"],
                success_conditions=["recommendations", "improvements", "actionable", "prioritized"],
                max_turns_in_phase=6,
                phase_timeout_minutes=12
            ),
            
            ConversationPhase(
                phase_id="implementation",
                name="Architecture Implementation",
                initial_prompt="""
                Now implement your architectural improvements:
                1. Apply the most critical architectural improvements you identified
                2. Implement design pattern improvements or refactoring
                3. Add architectural documentation or comments
                4. Create or modify configuration files if needed
                5. Ensure your changes compile and maintain system integrity
                
                Use write_file to implement your architectural enhancements.
                """,
                expected_actions=["write_file", "refactor_code", "compiler"],
                success_conditions=["improvements_implemented", "code_compiles", "architecture_enhanced"],
                max_turns_in_phase=12,
                phase_timeout_minutes=25
            )
        ]
        
        return phases
    
    async def _generate_collaboration_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for human-agent collaboration scenarios"""
        
        phases = [
            ConversationPhase(
                phase_id="introduction",
                name="Collaboration Setup",
                initial_prompt=f"""
                Welcome to the collaborative session! We'll be working together on: {scenario.title}
                
                Let's start by establishing our collaboration:
                1. Introduce yourself and your capabilities
                2. Understand the task and requirements
                3. Discuss the approach and division of work
                4. Set up communication patterns
                5. Agree on success criteria
                
                This is a collaborative effort - ask questions and seek clarification as needed.
                """,
                expected_actions=["echo"],
                success_conditions=["introduction", "collaboration", "approach", "communication"],
                max_turns_in_phase=5,
                phase_timeout_minutes=10
            ),
            
            ConversationPhase(
                phase_id="collaborative_work",
                name="Collaborative Implementation",
                initial_prompt="""
                Now let's work together on the implementation:
                1. Follow the agreed approach
                2. Communicate your progress and decisions
                3. Ask for feedback and guidance when needed
                4. Adapt based on input and changing requirements
                5. Keep the human collaborator informed
                
                Remember: this is a team effort. Collaborate effectively!
                """,
                expected_actions=["read_file", "write_file", "echo", "compile_code"],
                success_conditions=["collaboration", "communication", "progress", "teamwork"],
                max_turns_in_phase=15,
                phase_timeout_minutes=25
            ),
            
            ConversationPhase(
                phase_id="review",
                name="Collaborative Review",
                initial_prompt="""
                Let's review our work together:
                1. Present what you've accomplished
                2. Explain your decisions and reasoning
                3. Identify areas for improvement
                4. Discuss lessons learned
                5. Plan next steps if applicable
                
                Be open to feedback and ready to iterate.
                """,
                expected_actions=["echo", "run_tests"],
                success_conditions=["review", "explanation", "feedback", "lessons"],
                max_turns_in_phase=6,
                phase_timeout_minutes=12
            )
        ]
        
        return phases
    
    async def _generate_generic_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate generic phases for other scenario types"""
        
        phases = [
            ConversationPhase(
                phase_id="understanding",
                name="Task Understanding",
                initial_prompt=f"""
                Let's start working on: {scenario.title}
                
                Begin by understanding the task:
                1. Analyze the requirements and context
                2. Examine the available resources
                3. Identify the key challenges
                4. Plan your approach
                
                Take time to thoroughly understand what needs to be done.
                """,
                expected_actions=["read_file", "list_directory"],
                success_conditions=["understanding", "requirements", "approach"],
                max_turns_in_phase=5,
                phase_timeout_minutes=10
            ),
            
            ConversationPhase(
                phase_id="execution",
                name="Task Execution",
                initial_prompt="""
                Now execute your plan:
                1. Implement your approach step by step
                2. Use available tools effectively
                3. Monitor your progress
                4. Adapt as needed
                5. Document your work
                
                Focus on producing high-quality results.
                """,
                expected_actions=["write_file", "compile_code", "run_program"],
                success_conditions=["execution", "implementation", "results"],
                max_turns_in_phase=12,
                phase_timeout_minutes=20
            ),
            
            ConversationPhase(
                phase_id="completion",
                name="Task Completion",
                initial_prompt="""
                Complete the task:
                1. Verify that all requirements are met
                2. Test your solution thoroughly
                3. Clean up and finalize your work
                4. Summarize what you accomplished
                
                Ensure everything is working correctly.
                """,
                expected_actions=["run_tests", "echo"],
                success_conditions=["completion", "verified", "working"],
                max_turns_in_phase=6,
                phase_timeout_minutes=12
            )
        ]
        
        return phases
    
    async def _generate_success_criteria(
        self,
        scenario: InteractiveScenario,
        task_category: TaskCategory
    ) -> List[SuccessCriterion]:
        """Generate success criteria for the scenario"""
        
        criteria = []
        
        # Common criteria for all scenarios
        criteria.extend([
            SuccessCriterion(
                criterion_id="task_completion",
                description="Successfully complete the main task",
                type="outcome",
                target_value=True,
                weight=2.0,
                required=True
            ),
            SuccessCriterion(
                criterion_id="tool_usage",
                description="Effectively use available tools",
                type="action",
                target_value="tool_used",
                weight=1.0,
                required=False
            ),
            SuccessCriterion(
                criterion_id="communication_quality",
                description="Provide clear explanations and reasoning",
                type="keyword",
                target_value=["explanation", "reasoning", "because"],
                weight=1.0,
                required=False
            )
        ])
        
        # Category-specific criteria
        if task_category == TaskCategory.COLLABORATIVE_DEVELOPMENT:
            criteria.extend([
                SuccessCriterion(
                    criterion_id="code_quality",
                    description="Produce high-quality, maintainable code",
                    type="outcome",
                    target_value="code_written",
                    weight=1.5,
                    required=True
                ),
                SuccessCriterion(
                    criterion_id="testing",
                    description="Include appropriate testing",
                    type="action",
                    target_value="run_tests",
                    weight=1.0,
                    required=False
                )
            ])
        
        elif task_category == TaskCategory.INTERACTIVE_DEBUGGING:
            criteria.extend([
                SuccessCriterion(
                    criterion_id="bug_identification",
                    description="Correctly identify the bug",
                    type="keyword",
                    target_value=["bug", "issue", "problem", "error"],
                    weight=1.5,
                    required=True
                ),
                SuccessCriterion(
                    criterion_id="debugging_tools",
                    description="Use debugging tools effectively",
                    type="action",
                    target_value="set_breakpoint",
                    weight=1.0,
                    required=False
                )
            ])
        
        elif task_category == TaskCategory.HUMAN_AGENT_COLLABORATION:
            criteria.extend([
                SuccessCriterion(
                    criterion_id="collaboration_quality",
                    description="Demonstrate effective collaboration",
                    type="keyword",
                    target_value=["collaborate", "together", "feedback", "discuss"],
                    weight=1.5,
                    required=True
                ),
                SuccessCriterion(
                    criterion_id="adaptability",
                    description="Adapt based on human feedback",
                    type="outcome",
                    target_value="adapted",
                    weight=1.0,
                    required=False
                )
            ])
        
        return criteria
    
    async def _generate_dynamic_prompts(
        self,
        scenario: InteractiveScenario,
        task_category: TaskCategory
    ) -> List[DynamicPrompt]:
        """Generate dynamic prompts that adapt to agent behavior"""
        
        prompts = []
        
        # Common dynamic prompts
        prompts.extend([
            DynamicPrompt(
                trigger_condition="no_tool_usage_5_turns",
                prompt_text="I notice you haven't used any tools yet. Remember that you have access to various development tools that can help you with this task. Consider using them to be more effective.",
                priority=1,
                max_uses=1
            ),
            DynamicPrompt(
                trigger_condition="stuck_same_action_3_times",
                prompt_text="It seems like you might be stuck or repeating the same approach. Try a different strategy or ask for clarification if needed.",
                priority=2,
                max_uses=2
            ),
            DynamicPrompt(
                trigger_condition="phase_timeout_approaching",
                prompt_text="You're approaching the time limit for this phase. Please focus on the most important aspects to complete this phase successfully.",
                priority=3,
                max_uses=1
            )
        ])
        
        # Category-specific dynamic prompts
        if task_category == TaskCategory.COLLABORATIVE_DEVELOPMENT:
            prompts.extend([
                DynamicPrompt(
                    trigger_condition="no_code_written_10_turns",
                    prompt_text="This is a development task, but I haven't seen any code implementation yet. Consider writing some code to make progress on the task.",
                    priority=1,
                    max_uses=1
                ),
                DynamicPrompt(
                    trigger_condition="compilation_errors_3_times",
                    prompt_text="You've encountered multiple compilation errors. Take a step back and review your code more carefully before proceeding.",
                    priority=2,
                    max_uses=1
                )
            ])
        
        elif task_category == TaskCategory.INTERACTIVE_DEBUGGING:
            prompts.extend([
                DynamicPrompt(
                    trigger_condition="no_debugging_tools_used",
                    prompt_text="This is a debugging task. Consider using debugging tools like breakpoints, variable inspection, and step-through execution to identify the issue.",
                    priority=1,
                    max_uses=1
                ),
                DynamicPrompt(
                    trigger_condition="bug_not_identified_15_turns",
                    prompt_text="You've been working on this for a while without identifying the specific bug. Try a more systematic approach to narrow down the issue.",
                    priority=2,
                    max_uses=1
                )
            ])
        
        return prompts
    
    def _determine_interaction_mode(self, task_category: TaskCategory) -> InteractionMode:
        """Determine the interaction mode based on task category"""
        
        if task_category == TaskCategory.HUMAN_AGENT_COLLABORATION:
            return InteractionMode.COLLABORATIVE
        elif task_category in [TaskCategory.INTERACTIVE_DEBUGGING, TaskCategory.INTERACTIVE_ARCHITECTURE]:
            return InteractionMode.GUIDED
        else:
            return InteractionMode.AUTONOMOUS
    
    def _determine_tool_usage_mode(
        self,
        task_category: TaskCategory,
        difficulty: DifficultyLevel
    ) -> ToolUsageMode:
        """Determine tool usage restrictions based on category and difficulty"""
        
        if difficulty == DifficultyLevel.EASY:
            return ToolUsageMode.LIMITED
        elif difficulty == DifficultyLevel.EXPERT:
            return ToolUsageMode.UNRESTRICTED
        else:
            return ToolUsageMode.PROGRESSIVE
    
    def _determine_available_tools(
        self,
        task_category: TaskCategory,
        difficulty: DifficultyLevel
    ) -> List[str]:
        """Determine which tools should be available for this scenario"""
        
        base_tools = ["echo", "calculator"]
        
        if task_category == TaskCategory.COLLABORATIVE_DEVELOPMENT:
            return base_tools + ["file_system", "compiler", "ide_simulator"]
        elif task_category == TaskCategory.INTERACTIVE_DEBUGGING:
            return base_tools + ["file_system", "debugger", "compiler"]
        elif task_category == TaskCategory.INTERACTIVE_ARCHITECTURE:
            return base_tools + ["file_system", "ide_simulator"]
        else:
            return base_tools + ["file_system"]
    
    def _calculate_max_turns(self, difficulty: DifficultyLevel) -> int:
        """Calculate maximum turns based on difficulty"""
        
        turn_limits = {
            DifficultyLevel.EASY: 20,
            DifficultyLevel.MEDIUM: 35,
            DifficultyLevel.HARD: 50,
            DifficultyLevel.EXPERT: 75
        }
        
        return turn_limits.get(difficulty, 35)
    
    def _calculate_max_duration(self, difficulty: DifficultyLevel) -> int:
        """Calculate maximum duration in minutes based on difficulty"""
        
        duration_limits = {
            DifficultyLevel.EASY: 30,
            DifficultyLevel.MEDIUM: 45,
            DifficultyLevel.HARD: 60,
            DifficultyLevel.EXPERT: 90
        }
        
        return duration_limits.get(difficulty, 45)
    
    def _calculate_context_window(self, difficulty: DifficultyLevel) -> int:
        """Calculate context window size based on difficulty"""
        
        context_sizes = {
            DifficultyLevel.EASY: 100_000,
            DifficultyLevel.MEDIUM: 300_000,
            DifficultyLevel.HARD: 600_000,
            DifficultyLevel.EXPERT: 1_000_000
        }
        
        return context_sizes.get(difficulty, 300_000)
    
    # New phase generation methods for extended categories
    
    async def _generate_refactoring_phases(
        self,
        scenario: InteractiveScenario,
        project_files: List[Dict[str, str]]
    ) -> List[ConversationPhase]:
        """Generate phases for guided multi-file refactoring scenarios"""
        
        return [
            ConversationPhase(
                phase_id="code_analysis",
                name="Code Analysis & Planning",
                initial_prompt=f"""
                You're working on a multi-file refactoring task: {scenario.title}
                
                Start by analyzing the current code structure:
                1. Examine the files that need refactoring
                2. Identify code smells, duplications, and architectural issues
                3. Understand dependencies between files
                4. Plan the refactoring strategy to minimize breaking changes
                
                Use search and analysis tools to understand the codebase thoroughly.
                """,
                expected_actions=["read_file", "search_file", "analyze_dependencies"],
                success_conditions=["analysis", "dependencies", "strategy", "plan"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            ),
            ConversationPhase(
                phase_id="refactoring_execution",
                name="Refactoring Execution",
                initial_prompt="""
                Execute your refactoring plan:
                1. Start with the least risky changes first
                2. Refactor one file at a time, testing as you go
                3. Update imports and dependencies as needed
                4. Ensure all references are updated consistently
                5. Maintain backward compatibility where possible
                """,
                expected_actions=["write_file", "compile_code", "run_tests"],
                success_conditions=["refactored", "working", "consistent", "tested"],
                max_turns_in_phase=15,
                phase_timeout_minutes=30
            )
        ]
    
    async def _generate_extended_project_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for extended development projects"""
        
        return [
            ConversationPhase(
                phase_id="project_setup",
                name="Project Setup & Architecture",
                initial_prompt=f"""
                Welcome to an extended development project: {scenario.title}
                
                This is a multi-session project. Start by setting up the foundation:
                1. Analyze the project requirements and scope
                2. Design the overall architecture
                3. Set up the project structure and build system
                4. Plan the development phases and milestones
                """,
                expected_actions=["create_file", "setup_build", "plan"],
                success_conditions=["setup", "architecture", "structure", "plan"],
                max_turns_in_phase=10,
                phase_timeout_minutes=20
            ),
            ConversationPhase(
                phase_id="iterative_development",
                name="Iterative Development",
                initial_prompt="""
                Begin iterative development:
                1. Implement core functionality first
                2. Build incrementally with regular testing
                3. Maintain clean code and documentation
                4. Handle integration challenges as they arise
                """,
                expected_actions=["write_file", "compile_code", "run_tests", "integrate"],
                success_conditions=["implemented", "tested", "integrated", "documented"],
                max_turns_in_phase=20,
                phase_timeout_minutes=40
            )
        ]
    
    async def _generate_code_exploration_phases(
        self,
        scenario: InteractiveScenario,
        project_files: List[Dict[str, str]]
    ) -> List[ConversationPhase]:
        """Generate phases for interactive code exploration"""
        
        return [
            ConversationPhase(
                phase_id="initial_exploration",
                name="Initial Code Exploration",
                initial_prompt=f"""
                You're exploring a new codebase: {scenario.title}
                
                Start your exploration:
                1. Get an overview of the project structure
                2. Identify the main entry points and key modules
                3. Understand the data flow and control flow
                4. Map out the key relationships between components
                """,
                expected_actions=["list_directory", "read_file", "search_patterns"],
                success_conditions=["overview", "structure", "flow", "relationships"],
                max_turns_in_phase=10,
                phase_timeout_minutes=20
            ),
            ConversationPhase(
                phase_id="deep_analysis",
                name="Deep Code Analysis",
                initial_prompt="""
                Now dive deeper into the codebase:
                1. Analyze complex algorithms and data structures
                2. Understand design patterns and architectural decisions
                3. Identify potential improvements or issues
                4. Document your findings and insights
                """,
                expected_actions=["analyze_code", "trace_execution", "document"],
                success_conditions=["analyzed", "understood", "documented", "insights"],
                max_turns_in_phase=12,
                phase_timeout_minutes=25
            ),
            
            ConversationPhase(
                phase_id="implementation",
                name="Code Enhancement Implementation",
                initial_prompt="""
                Based on your code exploration and analysis, implement improvements:
                1. Apply the improvements or fixes you identified during analysis
                2. Add missing functionality or enhance existing features
                3. Improve code quality, add comments, or refactor problematic areas
                4. Create unit tests or documentation for complex parts
                5. Ensure your changes integrate well with the existing codebase
                
                Use write_file to implement your enhancements and improvements.
                """,
                expected_actions=["write_file", "compiler", "test_runner"],
                success_conditions=["improvements_implemented", "code_enhanced", "quality_improved"],
                max_turns_in_phase=15,
                phase_timeout_minutes=30
            )
        ]
    
    async def _generate_tdd_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for test-driven development sessions"""
        
        return [
            ConversationPhase(
                phase_id="test_planning",
                name="Test Strategy Planning",
                initial_prompt=f"""
                You're working on a TDD project: {scenario.title}
                
                Start with test planning:
                1. Understand the requirements and acceptance criteria
                2. Design a comprehensive test strategy
                3. Plan unit tests, integration tests, and end-to-end tests
                4. Set up the testing framework and tools
                """,
                expected_actions=["plan_tests", "setup_framework", "design_strategy"],
                success_conditions=["strategy", "planned", "framework", "setup"],
                max_turns_in_phase=6,
                phase_timeout_minutes=15
            ),
            ConversationPhase(
                phase_id="red_green_refactor",
                name="Red-Green-Refactor Cycle",
                initial_prompt="""
                Follow the TDD cycle:
                1. Write failing tests first (Red)
                2. Write minimal code to make tests pass (Green)
                3. Refactor code while keeping tests passing (Refactor)
                4. Repeat the cycle for each feature
                """,
                expected_actions=["write_test", "run_tests", "write_code", "refactor"],
                success_conditions=["tests_written", "tests_passing", "code_clean", "cycle"],
                max_turns_in_phase=18,
                phase_timeout_minutes=35
            )
        ]
    
    async def _generate_security_audit_phases(
        self,
        scenario: InteractiveScenario,
        project_files: List[Dict[str, str]]
    ) -> List[ConversationPhase]:
        """Generate phases for interactive security auditing"""
        
        return [
            ConversationPhase(
                phase_id="security_assessment",
                name="Security Assessment",
                initial_prompt=f"""
                You're conducting a security audit: {scenario.title}
                
                Begin the security assessment:
                1. Identify potential security vulnerabilities
                2. Check for common security anti-patterns
                3. Analyze input validation and sanitization
                4. Review authentication and authorization mechanisms
                """,
                expected_actions=["scan_vulnerabilities", "check_patterns", "analyze_auth"],
                success_conditions=["vulnerabilities", "assessed", "patterns", "analysis"],
                max_turns_in_phase=10,
                phase_timeout_minutes=20
            ),
            ConversationPhase(
                phase_id="security_remediation",
                name="Security Remediation",
                initial_prompt="""
                Address the security issues found:
                1. Prioritize vulnerabilities by risk level
                2. Implement security fixes and improvements
                3. Add security testing and validation
                4. Document security measures and recommendations
                """,
                expected_actions=["fix_vulnerabilities", "add_security", "test_security"],
                success_conditions=["fixed", "secured", "tested", "documented"],
                max_turns_in_phase=12,
                phase_timeout_minutes=25
            )
        ]
    
    async def _generate_tool_mastery_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for tool mastery evaluation"""
        
        return [
            ConversationPhase(
                phase_id="tool_exploration",
                name="Tool Exploration & Learning",
                initial_prompt=f"""
                This is a tool mastery evaluation: {scenario.title}
                
                Explore and learn the available tools:
                1. Discover what tools are available to you
                2. Understand the capabilities of each tool
                3. Practice using different tool combinations
                4. Learn efficient workflows and patterns
                """,
                expected_actions=["explore_tools", "test_capabilities", "practice_usage"],
                success_conditions=["explored", "understood", "practiced", "workflows"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            ),
            ConversationPhase(
                phase_id="advanced_tool_usage",
                name="Advanced Tool Usage",
                initial_prompt="""
                Demonstrate advanced tool usage:
                1. Solve complex problems using tool combinations
                2. Optimize your tool usage for efficiency
                3. Handle edge cases and error scenarios
                4. Show creative and effective tool usage patterns
                """,
                expected_actions=["combine_tools", "optimize_usage", "handle_errors"],
                success_conditions=["complex_solved", "optimized", "creative", "effective"],
                max_turns_in_phase=12,
                phase_timeout_minutes=25
            )
        ]
    
    async def _generate_documentation_phases(
        self,
        scenario: InteractiveScenario,
        project_files: List[Dict[str, str]]
    ) -> List[ConversationPhase]:
        """Generate phases for documentation generation"""
        
        return [
            ConversationPhase(
                phase_id="documentation_planning",
                name="Documentation Planning",
                initial_prompt=f"""
                You're creating comprehensive documentation: {scenario.title}
                
                Plan the documentation strategy:
                1. Analyze the codebase to understand what needs documentation
                2. Identify the target audience and their needs
                3. Plan the documentation structure and format
                4. Decide on documentation tools and standards
                """,
                expected_actions=["analyze_code", "plan_structure", "choose_tools"],
                success_conditions=["analyzed", "planned", "structured", "tools_chosen"],
                max_turns_in_phase=6,
                phase_timeout_minutes=12
            ),
            ConversationPhase(
                phase_id="documentation_creation",
                name="Documentation Creation",
                initial_prompt="""
                Create comprehensive documentation:
                1. Write clear and concise API documentation
                2. Create user guides and tutorials
                3. Document architecture and design decisions
                4. Include code examples and usage patterns
                5. Ensure documentation is maintainable and up-to-date
                """,
                expected_actions=["write_docs", "create_examples", "document_api"],
                success_conditions=["documented", "clear", "comprehensive", "examples"],
                max_turns_in_phase=15,
                phase_timeout_minutes=30
            )
        ]
    
    async def _generate_devops_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for deployment & DevOps scenarios"""
        
        return [
            ConversationPhase(
                phase_id="deployment_planning",
                name="Deployment Planning",
                initial_prompt=f"""
                You're working on deployment and DevOps: {scenario.title}
                
                Plan the deployment strategy:
                1. Analyze the application's deployment requirements
                2. Choose appropriate deployment platforms and tools
                3. Design the CI/CD pipeline
                4. Plan monitoring, logging, and alerting
                """,
                expected_actions=["analyze_requirements", "choose_platform", "design_pipeline"],
                success_conditions=["analyzed", "planned", "designed", "strategy"],
                max_turns_in_phase=8,
                phase_timeout_minutes=15
            ),
            ConversationPhase(
                phase_id="deployment_implementation",
                name="Deployment Implementation",
                initial_prompt="""
                Implement the deployment solution:
                1. Set up the deployment environment
                2. Configure CI/CD pipelines
                3. Implement monitoring and logging
                4. Test the deployment process
                5. Document the deployment procedures
                """,
                expected_actions=["setup_environment", "configure_cicd", "implement_monitoring"],
                success_conditions=["deployed", "configured", "monitored", "tested"],
                max_turns_in_phase=15,
                phase_timeout_minutes=30
            )
        ]
    
    async def _generate_collaboration_phases(
        self,
        scenario: InteractiveScenario,
        project_spec: Dict[str, Any]
    ) -> List[ConversationPhase]:
        """Generate phases for human-agent collaboration scenarios"""
        
        return [
            ConversationPhase(
                phase_id="collaboration_setup",
                name="Collaboration Setup",
                initial_prompt=f"""
                Welcome to a human-agent collaboration session: {scenario.title}
                
                Let's establish our collaboration:
                1. Understand the collaborative goals and expectations
                2. Establish communication protocols and preferences
                3. Define roles and responsibilities
                4. Set up shared tools and workflows
                """,
                expected_actions=["establish_goals", "define_roles", "setup_workflow"],
                success_conditions=["goals_clear", "roles_defined", "workflow_setup"],
                max_turns_in_phase=6,
                phase_timeout_minutes=10
            ),
            ConversationPhase(
                phase_id="collaborative_work",
                name="Collaborative Development",
                initial_prompt="""
                Work together on the task:
                1. Share ideas and approaches openly
                2. Ask for feedback and clarification when needed
                3. Adapt your approach based on human input
                4. Provide clear explanations of your reasoning
                5. Maintain effective communication throughout
                """,
                expected_actions=["collaborate", "ask_feedback", "explain_reasoning"],
                success_conditions=["collaborative", "communicative", "adaptive", "effective"],
                max_turns_in_phase=20,
                phase_timeout_minutes=40
            )
        ]
