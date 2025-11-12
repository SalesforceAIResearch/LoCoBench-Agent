"""
Scenario Conversion Utilities for LoCoBench-Agent

This module provides efficient conversion of single-turn scenarios to multi-turn
interactive scenarios, with caching and batch processing capabilities.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..core.config import Config
from ..core.data_loader import DataLoader, ScenarioData
from ..core.task import TaskCategory

logger = logging.getLogger(__name__)


@dataclass
class ConversionStats:
    """Statistics from scenario conversion process"""
    total_scenarios: int = 0
    converted_scenarios: int = 0
    cached_scenarios: int = 0
    failed_conversions: int = 0
    conversion_time_seconds: float = 0.0
    
    def __str__(self):
        return (f"Converted {self.converted_scenarios}/{self.total_scenarios} scenarios "
                f"({self.cached_scenarios} cached, {self.failed_conversions} failed) "
                f"in {self.conversion_time_seconds:.1f}s")


class ScenarioConverter:
    """Converts single-turn scenarios to multi-turn interactive scenarios"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.cache_dir = Path(config.data.output_dir) / "agent_scenarios"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Category mapping from single-turn to multi-turn
        self.category_mapping = {
            "code_comprehension": "interactive_code_exploration",
            "architectural_understanding": "interactive_architecture_exploration", 
            "bug_investigation": "interactive_debugging_sessions",
            "feature_implementation": "collaborative_feature_development",
            "cross_file_refactoring": "guided_multi_file_refactoring",
            "integration_testing": "test_driven_development_sessions",
            "security_analysis": "interactive_security_auditing",
            "multi_session_development": "extended_development_projects"
        }
    
    def is_converted(self, scenario_id: str) -> bool:
        """Check if a scenario has already been converted"""
        cache_file = self.cache_dir / f"{scenario_id}.json"
        return cache_file.exists()
    
    def load_converted_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Load a pre-converted scenario from cache"""
        cache_file = self.cache_dir / f"{scenario_id}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached scenario {scenario_id}: {e}")
            return None
    
    def save_converted_scenario(self, scenario_id: str, converted_scenario: Dict[str, Any]):
        """Save a converted scenario to cache"""
        cache_file = self.cache_dir / f"{scenario_id}.json"
        
        try:
            # Add conversion metadata
            converted_scenario["_conversion_metadata"] = {
                "converted_at": datetime.now().isoformat(),
                "converter_version": "1.0",
                "source": "single_turn_scenario"
            }
            
            with open(cache_file, 'w') as f:
                json.dump(converted_scenario, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving converted scenario {scenario_id}: {e}")
    
    async def convert_scenario(self, scenario_data: ScenarioData) -> Optional[Dict[str, Any]]:
        """Convert a single scenario to multi-turn format"""
        
        try:
            # Extract project information
            scenario_id = scenario_data.scenario_id
            parts = scenario_id.split("_")
            project_name = "_".join(parts[:4]) if len(parts) >= 4 else scenario_id
            
            # Convert task category
            task_category = self.category_mapping.get(
                scenario_data.task_category, 
                "interactive_code_exploration"
            )
            
            # Create conversation phases based on original task
            phases = self._create_conversation_phases(scenario_data)
            
            # Determine tools based on category and difficulty
            available_tools = self._determine_tools(task_category, scenario_data.difficulty)
            
            # Create interactive scenario structure
            converted_scenario = {
                "scenario_id": scenario_id,
                "title": scenario_data.title,
                "description": scenario_data.description,
                "category": task_category,
                "difficulty": scenario_data.difficulty.lower(),
                "context_files": scenario_data.context_files,
                "working_directory": project_name,
                "conversation_phases": phases,
                "available_tools": available_tools,
                "max_turns": self._calculate_max_turns(scenario_data.difficulty),
                "max_duration_minutes": self._calculate_max_duration(scenario_data.difficulty),
                "context_window_tokens": self._calculate_context_window(scenario_data.difficulty),
                
                # Project context (if available)
                "project_files": scenario_data.project_context.files if scenario_data.project_context else [],
                "project_spec": scenario_data.project_context.specification if scenario_data.project_context else {},
                "project_directory": str(scenario_data.project_context.project_dir) if scenario_data.project_context else None,
                "project_name": scenario_data.project_context.project_name if scenario_data.project_context else project_name,
                
                # Original scenario reference
                "original_scenario": scenario_data.raw_data
            }
            
            return converted_scenario
            
        except Exception as e:
            logger.error(f"Error converting scenario {scenario_data.scenario_id}: {e}")
            return None
    
    def _create_conversation_phases(self, scenario_data: ScenarioData) -> List[Dict[str, Any]]:
        """Create conversation phases based on the original scenario"""
        
        category = scenario_data.task_category
        difficulty = scenario_data.difficulty.lower()
        
        # Base phases that work for most scenarios
        phases = [
            {
                "phase_id": "exploration",
                "name": "Code Exploration",
                "initial_prompt": f"Explore and understand the codebase. {scenario_data.description}",
                "expected_actions": ["read_file", "search_code", "list_files"],
                "success_conditions": ["understanding", "analysis", "exploration_complete"],
                "max_turns_in_phase": 8 if difficulty == "expert" else 6,
                "dynamic_prompts": {
                    "file_read": "Based on what you've found, continue exploring other relevant files in the codebase.",
                    "analysis_started": "Please provide more detailed analysis of the code structure and patterns you've identified.",
                    "partial_understanding": "Can you elaborate on your findings and explore additional aspects of the codebase?"
                }
            }
        ]
        
        # Category-specific phases
        if category == "bug_investigation":
            phases.extend([
                {
                    "phase_id": "diagnosis",
                    "name": "Problem Diagnosis", 
                    "initial_prompt": "Based on your exploration, diagnose the root cause of the issue",
                    "expected_actions": ["debugger", "search_code", "trace_execution"],
                    "success_conditions": ["root_cause_identified", "hypothesis_formed"],
                    "max_turns_in_phase": 10 if difficulty == "expert" else 7,
                    "dynamic_prompts": {
                        "debugging_started": "Continue your investigation. What specific aspects of the code are causing the issue?",
                        "hypothesis_forming": "Based on your findings, can you narrow down the root cause further?",
                        "need_more_evidence": "Please gather more evidence to support your diagnosis."
                    }
                },
                {
                    "phase_id": "solution",
                    "name": "Solution Implementation",
                    "initial_prompt": "Implement a fix for the identified issue",
                    "expected_actions": ["write_file", "compiler", "test_runner"],
                    "success_conditions": ["fix_implemented", "tests_passing"],
                    "max_turns_in_phase": 12 if difficulty == "expert" else 8,
                    "dynamic_prompts": {
                        "implementation_started": "Continue implementing the fix by USING write_file tool. Make sure to test your changes.",
                        "testing_needed": "Please run tests using available tools to verify your fix works correctly.",
                        "refinement_needed": "Use write_file tool to refine your solution if needed. Ensure it addresses all aspects of the problem."
                    }
                }
            ])
        
        elif category == "feature_implementation":
            phases.extend([
                {
                    "phase_id": "planning",
                    "name": "Implementation Planning",
                    "initial_prompt": "Plan the implementation approach for the required feature by using read_file and search tools to understand the codebase",
                    "expected_actions": ["search_code", "read_file", "analyze_dependencies"],
                    "success_conditions": ["plan_created", "dependencies_identified"],
                    "max_turns_in_phase": 6,
                    "dynamic_prompts": {
                        "exploration_started": "Continue analyzing the codebase using read_file and search tools to understand how to integrate the new feature.",
                        "dependencies_found": "Based on the dependencies you've identified, how will you structure the implementation?",
                        "planning_progress": "Please elaborate on your implementation plan and consider potential challenges."
                    }
                },
                {
                    "phase_id": "implementation",
                    "name": "Feature Implementation",
                    "initial_prompt": "Implement the planned feature following best practices",
                    "expected_actions": ["write_file", "compiler", "ide_simulator"],
                    "success_conditions": ["feature_implemented", "code_compiles"],
                    "max_turns_in_phase": 15 if difficulty == "expert" else 10,
                    "dynamic_prompts": {
                        "coding_started": "Continue implementing the feature by USING write_file tool to create/modify code files. Make sure to follow the plan you created.",
                        "compilation_issues": "Use write_file tool to fix compilation errors and continue with the implementation.",
                        "feature_progress": "Use write_file tool to continue the implementation. What files have you modified so far?"
                    }
                },
                {
                    "phase_id": "testing",
                    "name": "Testing and Validation",
                    "initial_prompt": "Test the implemented feature and ensure it works correctly",
                    "expected_actions": ["test_runner", "compiler", "debugger"],
                    "success_conditions": ["tests_written", "tests_passing", "feature_validated"],
                    "max_turns_in_phase": 8,
                    "dynamic_prompts": {
                        "testing_started": "Continue writing and running tests for your implemented feature.",
                        "test_failures": "Address any test failures and ensure your feature works as expected.",
                        "validation_needed": "Please validate that your feature meets all the requirements."
                    }
                }
            ])
        
        elif category in ["architectural_understanding", "code_comprehension"]:
            phases.extend([
                {
                    "phase_id": "analysis",
                    "name": "Deep Analysis",
                    "initial_prompt": "Perform a detailed analysis of the architecture and design patterns",
                    "expected_actions": ["search_code", "read_file", "trace_dependencies"],
                    "success_conditions": ["architecture_understood", "patterns_identified"],
                    "max_turns_in_phase": 10 if difficulty == "expert" else 7,
                    "dynamic_prompts": {
                        "analysis_started": "Continue your architectural analysis. What patterns and structures do you see?",
                        "patterns_emerging": "Based on your analysis, can you identify additional architectural patterns?",
                        "deeper_understanding": "Please provide more detailed insights into the system's design."
                    }
                },
                {
                    "phase_id": "implementation",
                    "name": "Analysis Implementation",
                    "initial_prompt": "Based on your analysis, implement improvements, fixes, or enhancements to demonstrate your understanding",
                    "expected_actions": ["write_file", "compiler", "test_runner"],
                    "success_conditions": ["improvements_implemented", "code_compiles", "understanding_demonstrated"],
                    "max_turns_in_phase": 12 if difficulty == "expert" else 8,
                    "dynamic_prompts": {
                        "implementation_started": "Continue implementing improvements based on your analysis. Use write_file to create or modify code.",
                        "compilation_issues": "Address any compilation errors and continue with your implementation.",
                        "demonstration_needed": "How does your implementation demonstrate your understanding of the architecture?"
                    }
                },
                {
                    "phase_id": "documentation",
                    "name": "Documentation Creation",
                    "initial_prompt": "Create comprehensive documentation of your findings and implemented changes",
                    "expected_actions": ["write_file", "create_diagrams"],
                    "success_conditions": ["documentation_created", "insights_documented", "changes_documented"],
                    "max_turns_in_phase": 6,
                    "dynamic_prompts": {
                        "documentation_started": "Continue documenting your architectural findings, insights, and implemented changes.",
                        "more_detail_needed": "Please provide more comprehensive documentation of both the analysis and implementation.",
                        "insights_sharing": "Can you elaborate on the key insights and how your implementation addresses them?"
                    }
                }
            ])
        
        else:
            # Generic implementation phase for other categories
            phases.append({
                "phase_id": "implementation",
                "name": "Implementation",
                "initial_prompt": "Based on your understanding, implement the required solution",
                "expected_actions": ["write_file", "compiler", "test_runner"],
                "success_conditions": ["solution_implemented", "requirements_met"],
                "max_turns_in_phase": 12 if difficulty == "expert" else 8,
                "dynamic_prompts": {
                    "implementation_started": "Continue working on the implementation. How is your progress?",
                    "testing_needed": "Please test your implementation to ensure it meets the requirements.",
                    "refinement_required": "Consider if your solution can be improved or optimized further."
                }
            })
        
        return phases
    
    def _determine_tools(self, category: str, difficulty: str) -> List[str]:
        """Determine appropriate tools based on category and difficulty"""
        
        # Use actual registered tool names to ensure tools are available
        base_tools = ["file_system", "echo", "calculator"]
        
        # Category-specific tools (using actual registered tool names)
        if "debugging" in category:
            base_tools.extend(["debugger", "compiler"])
        elif "development" in category or "implementation" in category:
            base_tools.extend(["ide_simulator", "compiler"])
        elif "architecture" in category or "exploration" in category:
            base_tools.extend(["ide_simulator"])  # Use ide_simulator for architecture exploration
        elif "testing" in category:
            base_tools.extend(["compiler"])  # Use compiler for testing scenarios
        elif "security" in category:
            base_tools.extend(["debugger"])  # Use debugger for security analysis
        
        # Difficulty-based tools (using actual registered tool names)
        if difficulty in ["hard", "expert"]:
            base_tools.extend(["ide_simulator", "debugger"])  # Add more advanced tools for harder scenarios
        
        return list(set(base_tools))  # Remove duplicates
    
    def _calculate_max_turns(self, difficulty: str) -> int:
        """Calculate maximum turns based on difficulty"""
        difficulty_turns = {
            "easy": 20,
            "medium": 30,
            "hard": 40,
            "expert": 50
        }
        return difficulty_turns.get(difficulty.lower(), 30)
    
    def _calculate_max_duration(self, difficulty: str) -> int:
        """Calculate maximum duration in minutes"""
        difficulty_duration = {
            "easy": 30,
            "medium": 45,
            "hard": 60,
            "expert": 90
        }
        return difficulty_duration.get(difficulty.lower(), 45)
    
    def _calculate_context_window(self, difficulty: str) -> int:
        """Calculate context window tokens"""
        difficulty_tokens = {
            "easy": 500_000,
            "medium": 750_000,
            "hard": 1_000_000,
            "expert": 1_500_000
        }
        return difficulty_tokens.get(difficulty.lower(), 1_000_000)
    
    async def convert_all_scenarios(
        self, 
        force_reconvert: bool = False,
        max_concurrent: int = 5,
        limit: Optional[int] = None
    ) -> ConversionStats:
        """Convert all scenarios to multi-turn format with caching"""
        
        start_time = datetime.now()
        stats = ConversionStats()
        
        # Load scenarios
        scenarios = self.data_loader.load_scenarios(
            limit=limit,
            include_project_context=True
        )
        
        stats.total_scenarios = len(scenarios)
        logger.info(f"Starting conversion of {stats.total_scenarios} scenarios")
        
        # Process scenarios with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        
        async def convert_single(scenario_data: ScenarioData):
            async with semaphore:
                return await self._convert_with_cache(scenario_data, force_reconvert)
        
        # Create tasks
        for scenario_data in scenarios:
            task = asyncio.create_task(convert_single(scenario_data))
            tasks.append(task)
        
        # Execute conversions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                stats.failed_conversions += 1
                logger.error(f"Conversion failed: {result}")
            elif result == "cached":
                stats.cached_scenarios += 1
            elif result == "converted":
                stats.converted_scenarios += 1
            elif result == "failed":
                stats.failed_conversions += 1
        
        # Calculate timing
        end_time = datetime.now()
        stats.conversion_time_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Conversion complete: {stats}")
        return stats
    
    async def _convert_with_cache(self, scenario_data: ScenarioData, force_reconvert: bool) -> str:
        """Convert a scenario with caching logic"""
        
        scenario_id = scenario_data.scenario_id
        
        # Check cache first
        if not force_reconvert and self.is_converted(scenario_id):
            return "cached"
        
        # Convert scenario
        converted = await self.convert_scenario(scenario_data)
        
        if converted:
            self.save_converted_scenario(scenario_id, converted)
            return "converted"
        else:
            return "failed"
    
    def load_all_cached_scenarios(self) -> List[Dict[str, Any]]:
        """Load all cached converted scenarios in deterministic order"""
        scenarios = []
        
        if not self.cache_dir.exists():
            return scenarios
        
        # Sort cache files by name for deterministic ordering across runs
        cache_files = sorted(self.cache_dir.glob("*.json"), key=lambda p: p.name)
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    scenario_data = json.load(f)
                    scenarios.append(scenario_data)
            except Exception as e:
                logger.warning(f"Failed to load cached scenario {cache_file}: {e}")
        
        logger.info(f"Loaded {len(scenarios)} cached scenarios in deterministic order")
        return scenarios
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get statistics about converted scenarios"""
        
        if not self.cache_dir.exists():
            return {"total_converted": 0, "cache_size_mb": 0}
        
        converted_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in converted_files)
        
        return {
            "total_converted": len(converted_files),
            "cache_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(self.cache_dir),
            "last_conversion": max((f.stat().st_mtime for f in converted_files), default=0)
        }


def get_scenario_converter(config: Config = None) -> ScenarioConverter:
    """Get a configured scenario converter instance"""
    
    if config is None:
        config = Config()
    
    return ScenarioConverter(config)
