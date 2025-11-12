"""
Scenario Generator for LoCoBench Phase 3
Converts completed projects into long-context evaluation scenarios
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from rich.console import Console
from ..core.task import TaskCategory, DifficultyLevel
from ..core.config import Config
from .synthetic_generator import MultiLLMGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationScenario:
    """Represents a single evaluation scenario"""
    id: str
    task_category: TaskCategory
    difficulty: DifficultyLevel
    title: str
    description: str
    context_files: List[str]
    context_length: int
    task_prompt: str
    expected_approach: str
    ground_truth: str
    evaluation_criteria: List[str]
    metadata: Dict[str, Any]


class ScenarioGenerator:
    """Main scenario generator for Phase 3"""
    
    def __init__(self, config: Config, log_file: str = None):
        self.config = config
        self.llm_generator = MultiLLMGenerator(config, log_file)
        
        # Create output directories
        self.scenarios_dir = Path(config.data.output_dir) / "scenarios"
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_task_scenarios(
        self,
        project_dir: Path,
        project_data: Dict[str, Any],
        task_category: TaskCategory,
        num_instances: int = 1,
        target_difficulty: Optional[DifficultyLevel] = None
    ) -> List[Dict[str, Any]]:
        """Generate evaluation scenarios for a specific task category"""
        
        scenarios = []
        errors = []  # Track errors for better debugging
        
        # Load project data
        spec = project_data['specification']
        generated_stats = project_data['generated_stats']
        project_files = self._load_project_files(project_dir, project_data)
        
        console = Console()
        
        console.print(f"         🔄 Generating {num_instances} {task_category.value} scenarios...")
        
        for i in range(num_instances):
            scenario_id = f"{project_dir.name}_{task_category.value}_{i+1:02d}"
            
            try:
                console.print(f"         📝 Scenario {i+1}/{num_instances}: {scenario_id}", end="")
                
                scenario = await self._generate_single_scenario(
                    scenario_id=scenario_id,
                    task_category=task_category,
                    project_spec=spec,
                    project_files=project_files,
                    project_stats=generated_stats,
                    target_difficulty=target_difficulty
                )
                
                # Show context info
                context_length = scenario.get('context_length', 0)
                files_count = len(scenario.get('context_files', []))
                difficulty = scenario.get('difficulty', 'unknown')
                
                console.print(f" ✅ ({difficulty}, {files_count} files, {context_length:,} chars)")
                
                scenarios.append(scenario)
                logger.info(f"Generated scenario {scenario_id}")
                
            except Exception as e:
                console.print(f" ❌ Error: {str(e)}")
                logger.error(f"Failed to generate scenario {scenario_id}: {e}")
                errors.append(f"Scenario {i+1}: {str(e)}")
                continue
        
        # Check if we generated any scenarios
        if len(scenarios) == 0:
            # All scenarios failed - this should be treated as a task failure
            error_summary = f"All {num_instances} scenarios failed to generate. Errors: {'; '.join(errors)}"
            console.print(f"         ❌ [bold red]TASK FAILED: 0/{num_instances} scenarios generated[/bold red]")
            logger.error(f"Task failure for {project_dir.name}_{task_category.value}: {error_summary}")
            
            # Raise exception to mark task as failed
            from locobench.generation.synthetic_generator import APIError
            raise APIError(
                provider="ScenarioGenerator",
                error_type="GENERATION_FAILED",
                message=error_summary,
                should_retry=True  # Allow retry for this type of failure
            )
        elif len(scenarios) < num_instances:
            # Partial success - log warning but continue
            console.print(f"         ⚠️  [yellow]Partial success: {len(scenarios)}/{num_instances} scenarios generated[/yellow]")
            logger.warning(f"Partial generation for {project_dir.name}_{task_category.value}: {len(scenarios)}/{num_instances} scenarios. Errors: {'; '.join(errors)}")
        else:
            # Full success
            console.print(f"         🎉 [bold green]Completed {len(scenarios)}/{num_instances} scenarios[/bold green]")
        
        return scenarios
    
    def _load_project_files(self, project_dir: Path, project_data: Dict[str, Any]) -> Dict[str, str]:
        """Load all project files into memory"""
        files = {}
        
        # Load files listed in metadata
        for file_info in project_data.get('files', []):
            file_path = project_dir / file_info['path']
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files[file_info['path']] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
        
        return files
    
    async def _generate_single_scenario(
        self,
        scenario_id: str,
        task_category: TaskCategory,
        project_spec: Dict[str, Any],
        project_files: Dict[str, str],
        project_stats: Dict[str, Any],
        target_difficulty: Optional[DifficultyLevel] = None
    ) -> Dict[str, Any]:
        
        # Get coverage requirements
        min_coverage = self.config.phase3.min_information_coverage
        
        # Select files with appropriate strategy
        if target_difficulty is not None:
            # TARGETED GENERATION: Enforce specific difficulty
            logger.info(f"🎯 TARGETED GENERATION: Enforcing {target_difficulty.value} difficulty for {scenario_id}")
            context_files, information_coverage, difficulty = await self._select_files_with_target_difficulty(
                task_category, project_files, project_spec, target_difficulty
            )
        else:
            # NATURAL GENERATION: Let difficulty emerge from coverage
            logger.info(f"🔬 NATURAL GENERATION: Let difficulty emerge naturally for {scenario_id}")
            context_files, information_coverage, difficulty = await self._select_files_with_coverage_retry(
                task_category, project_files, project_spec, min_coverage
            )
        
        context_length = sum(len(content) for content in context_files.values())
        
        # Log coverage and difficulty determination
        if target_difficulty is not None:
            if difficulty == target_difficulty:
                logger.info(f"✅ TARGET ACHIEVED: {scenario_id}: {information_coverage:.2f} coverage → {difficulty.value} difficulty (target: {target_difficulty.value})")
            else:
                logger.warning(f"⚠️  TARGET MISSED: {scenario_id}: {information_coverage:.2f} coverage → {difficulty.value} difficulty (target: {target_difficulty.value})")
        else:
            logger.info(f"📊 NATURAL RESULT: {scenario_id}: {information_coverage:.2f} coverage → {difficulty.value} difficulty")
        
        # Validate minimum coverage requirement
        if information_coverage < min_coverage:
            logger.warning(f"⚠️  Scenario {scenario_id} below minimum coverage ({information_coverage:.2f} < {min_coverage:.2f})")
        else:
            logger.info(f"✅ Scenario {scenario_id} meets minimum coverage ({information_coverage:.2f} >= {min_coverage:.2f})")
        
        # Generate scenario using LLM
        scenario_data = await self._generate_scenario_content(
            task_category=task_category,
            difficulty=difficulty,
            project_spec=project_spec,
            context_files=context_files,
            scenario_id=scenario_id
        )
        
        return {
            "id": scenario_id,
            "task_category": task_category.value,
            "difficulty": difficulty.value,
            "title": scenario_data.get('title', f"{task_category.value.replace('_', ' ').title()} Task"),
            "description": scenario_data.get('description', ''),
            "context_files": list(context_files.keys()),
            "context_length": context_length,
            "task_prompt": scenario_data.get('task_prompt', ''),
            "expected_approach": scenario_data.get('expected_approach', ''),
            "ground_truth": scenario_data.get('ground_truth', ''),
            "evaluation_criteria": scenario_data.get('evaluation_criteria', []),
            "metadata": {
                "context_length": context_length,
                "files_count": len(context_files),
                "information_coverage": information_coverage,
                "coverage_range": self.config.phase3.coverage_ranges.get(difficulty.value.lower(), [0, 1]),
                "generation_timestamp": self._get_timestamp()
            }
        }

    async def _select_files_with_coverage_retry(
        self, 
        task_category: TaskCategory, 
        project_files: Dict[str, str], 
        project_spec: Dict[str, Any],
        min_coverage: float,
        max_retries: int = 3
    ) -> Tuple[Dict[str, str], float, DifficultyLevel]:
        """
        Adaptive file selection to meet coverage requirements.
        Difficulty is determined naturally from achieved coverage.
        
        Returns: (context_files, information_coverage, difficulty)
        """
        
        strategies = [
            "adaptive_smart",      # Primary: Smart adaptive selection
            "adaptive_aggressive", # Secondary: More aggressive file inclusion
            "category_focused",    # Tertiary: Focus on task-category specific files
            "fallback_permissive"  # Final: Accept lower coverage with warning
        ]
        
        for attempt, strategy in enumerate(strategies):
            try:
                logger.debug(f"🔄 Attempt {attempt + 1}/{len(strategies)}: Using {strategy} strategy")
                
                if strategy == "adaptive_smart":
                    context_files = await self._adaptive_file_selection(task_category, project_files, min_coverage)
                elif strategy == "adaptive_aggressive":
                    context_files = await self._adaptive_file_selection(task_category, project_files, min_coverage * 0.9, aggressive=True)
                elif strategy == "category_focused":
                    context_files = self._select_context_files(task_category, project_files)
                    # Add more files if coverage is too low
                    coverage = self._calculate_information_coverage(context_files, project_files)
                    if coverage < min_coverage:
                        context_files = self._expand_file_selection(context_files, project_files, min_coverage)
                else:  # fallback_permissive
                    context_files = self._select_context_files(task_category, project_files)
                
                # Calculate coverage and difficulty
                information_coverage = self._calculate_information_coverage(context_files, project_files)
                context_length = sum(len(content) for content in context_files.values())
                
                # Determine difficulty naturally from achieved coverage
                difficulty = self._determine_difficulty_from_coverage(information_coverage)
                
                # Check if this attempt meets requirements
                if strategy == "fallback_permissive" or information_coverage >= min_coverage:
                    if information_coverage >= min_coverage:
                        logger.info(f"✅ Strategy '{strategy}' achieved coverage {information_coverage:.2f}")
                    else:
                        logger.warning(f"⚠️  Using fallback strategy with coverage {information_coverage:.2f}")
                    return context_files, information_coverage, difficulty
                else:
                    logger.debug(f"❌ Strategy '{strategy}' coverage {information_coverage:.2f} < {min_coverage:.2f}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Strategy '{strategy}' failed: {e}")
                continue
        
        # This should never be reached due to fallback_permissive, but safety fallback
        logger.error("🚨 All file selection strategies failed, using basic selection")
        context_files = self._select_context_files(task_category, project_files)
        information_coverage = self._calculate_information_coverage(context_files, project_files)
        context_length = sum(len(content) for content in context_files.values())
        
        # Determine difficulty naturally from achieved coverage
        difficulty = self._determine_difficulty_from_coverage(information_coverage)
        
        return context_files, information_coverage, difficulty

    async def _select_files_with_target_difficulty(
        self,
        task_category: TaskCategory,
        project_files: Dict[str, str],
        project_spec: Dict[str, Any],
        target_difficulty: DifficultyLevel,
        max_retries: int = 5
    ) -> Tuple[Dict[str, str], float, DifficultyLevel]:
        """
        Select files to achieve a specific target difficulty level.
        Uses progressively more aggressive strategies to hit the target coverage range.
        """
        
        # Get target coverage range for this difficulty
        target_range = self.config.phase3.coverage_ranges.get(target_difficulty.value.lower())
        if not target_range:
            logger.warning(f"No coverage range defined for {target_difficulty}, falling back to natural selection")
            return await self._select_files_with_coverage_retry(task_category, project_files, project_spec, self.config.phase3.min_information_coverage)
        
        min_target_coverage, max_target_coverage = target_range
        optimal_target_coverage = (min_target_coverage + max_target_coverage) / 2  # Aim for middle of range
        
        logger.info(f"🎯 Targeting {target_difficulty.value} difficulty: coverage range [{min_target_coverage:.2f}, {max_target_coverage:.2f}]")
        
        # Progressive strategies: start conservative, get more aggressive
        strategies = [
            ("conservative", optimal_target_coverage),           # Target optimal coverage
            ("slightly_aggressive", min_target_coverage + 0.75 * (max_target_coverage - min_target_coverage)),  # 75% of range
            ("aggressive", min_target_coverage + 0.9 * (max_target_coverage - min_target_coverage)),            # 90% of range  
            ("very_aggressive", max_target_coverage),           # Max coverage for this difficulty
            ("fallback", min_target_coverage)                   # Accept minimum for this difficulty
        ]
        
        for attempt, (strategy_name, target_coverage) in enumerate(strategies):
            try:
                logger.debug(f"🔄 Attempt {attempt + 1}/{len(strategies)}: {strategy_name} targeting coverage {target_coverage:.2f}")
                
                # Use adaptive file selection with target coverage
                context_files = await self._adaptive_file_selection(
                    task_category, 
                    project_files, 
                    target_coverage,
                    aggressive=(strategy_name in ["aggressive", "very_aggressive"])
                )
                
                # Calculate achieved coverage and difficulty
                information_coverage = self._calculate_information_coverage(context_files, project_files)
                achieved_difficulty = self._determine_difficulty_from_coverage(information_coverage)
                
                logger.debug(f"Strategy '{strategy_name}': coverage {information_coverage:.2f} → {achieved_difficulty.value} difficulty")
                
                # Check if we hit the target difficulty
                if achieved_difficulty == target_difficulty:
                    logger.info(f"✅ SUCCESS: {strategy_name} achieved target {target_difficulty.value} (coverage: {information_coverage:.2f})")
                    return context_files, information_coverage, achieved_difficulty
                
                # If we're in the target range but wrong difficulty, log for debugging
                if min_target_coverage <= information_coverage <= max_target_coverage:
                    logger.warning(f"⚠️  Coverage {information_coverage:.2f} in target range but got {achieved_difficulty.value} instead of {target_difficulty.value}")
                
            except Exception as e:
                logger.warning(f"⚠️  Strategy '{strategy_name}' failed: {e}")
                continue
        
        # If all targeted attempts failed, fall back to natural generation
        logger.warning(f"❌ Could not achieve target difficulty {target_difficulty.value}, falling back to natural selection")
        return await self._select_files_with_coverage_retry(task_category, project_files, project_spec, self.config.phase3.min_information_coverage)

    async def _adaptive_file_selection(
        self, 
        task_category: TaskCategory, 
        project_files: Dict[str, str], 
        target_coverage: float,
        aggressive: bool = False
    ) -> Dict[str, str]:
        """
        Intelligently select files to meet target coverage requirements.
        """
        
        # Start with core files for the task category
        selected_files = {}
        core_files = self._get_core_files_for_category(task_category, project_files)
        
        # Add core files first
        for file_path in core_files:
            if file_path in project_files:
                selected_files[file_path] = project_files[file_path]
        
        # Calculate current coverage
        current_coverage = self._calculate_information_coverage(selected_files, project_files)
        
        if current_coverage >= target_coverage:
            return selected_files
        
        # Rank remaining files by relevance
        remaining_files = {k: v for k, v in project_files.items() if k not in selected_files}
        ranked_files = self._rank_files_by_relevance(task_category, remaining_files, aggressive)
        
        # Add files until target coverage is met
        for file_path, _ in ranked_files:
            selected_files[file_path] = project_files[file_path]
            current_coverage = self._calculate_information_coverage(selected_files, project_files)
            
            if current_coverage >= target_coverage:
                break
        
        return selected_files

    def _get_core_files_for_category(self, task_category: TaskCategory, project_files: Dict[str, str]) -> List[str]:
        """Get core files that are essential for each task category."""
        
        core_patterns = {
            TaskCategory.ARCHITECTURAL_UNDERSTANDING: [
                'main.', 'app.', 'index.', 'server.', 'config.', 'settings.',
                'module.', 'package.', '__init__.', 'Cargo.toml', 'package.json',
                'pom.xml', 'build.gradle', 'CMakeLists.txt', 'Makefile'
            ],
            TaskCategory.CROSS_FILE_REFACTORING: [
                'service.', 'manager.', 'handler.', 'controller.', 'model.',
                'util.', 'helper.', 'common.', 'shared.', 'base.'
            ],
            TaskCategory.FEATURE_IMPLEMENTATION: [
                'service.', 'api.', 'controller.', 'handler.', 'model.',
                'entity.', 'repository.', 'dao.', 'interface.'
            ],
            TaskCategory.BUG_INVESTIGATION: [
                'test.', 'spec.', 'error.', 'exception.', 'log.',
                'debug.', 'trace.', 'monitor.'
            ],
            TaskCategory.MULTI_SESSION_DEVELOPMENT: [
                'main.', 'app.', 'core.', 'service.', 'model.',
                'config.', 'util.', 'common.'
            ],
            TaskCategory.CODE_COMPREHENSION: [
                'readme.', 'doc.', 'comment.', 'main.', 'app.',
                'index.', 'core.', 'base.'
            ],
            TaskCategory.INTEGRATION_TESTING: [
                'test.', 'spec.', 'integration.', 'e2e.', 'api.',
                'service.', 'controller.', 'client.'
            ],
            TaskCategory.SECURITY_ANALYSIS: [
                'auth.', 'security.', 'login.', 'password.', 'token.',
                'crypto.', 'hash.', 'encrypt.', 'ssl.', 'cert.'
            ]
        }
        
        patterns = core_patterns.get(task_category, ['main.', 'app.', 'index.'])
        core_files = []
        
        for file_path in project_files.keys():
            file_name = file_path.lower()
            if any(pattern in file_name for pattern in patterns):
                core_files.append(file_path)
        
        return core_files

    def _rank_files_by_relevance(
        self, 
        task_category: TaskCategory, 
        files: Dict[str, str], 
        aggressive: bool = False
    ) -> List[tuple[str, float]]:
        """Rank files by relevance to task category."""
        
        file_scores = []
        
        for file_path, content in files.items():
            score = 0.0
            
            # File name relevance
            file_name = file_path.lower()
            score += self._calculate_filename_relevance(task_category, file_name)
            
            # Content size (larger files often more important)
            content_size = len(content)
            size_score = min(content_size / 10000, 1.0)  # Normalize to 0-1
            score += size_score * 0.3
            
            # Content complexity (more complex = more relevant)
            complexity = self._estimate_file_complexity(content)
            score += complexity * 0.2
            
            # Aggressive mode: prefer larger files more heavily
            if aggressive:
                score += size_score * 0.5
            
            file_scores.append((file_path, score))
        
        # Sort by score (descending)
        return sorted(file_scores, key=lambda x: x[1], reverse=True)

    def _calculate_filename_relevance(self, task_category: TaskCategory, file_name: str) -> float:
        """Calculate relevance score based on filename."""
        
        relevance_keywords = {
            TaskCategory.ARCHITECTURAL_UNDERSTANDING: [
                'architecture', 'design', 'structure', 'pattern', 'module',
                'component', 'service', 'layer', 'interface', 'abstract'
            ],
            TaskCategory.CROSS_FILE_REFACTORING: [
                'refactor', 'restructure', 'service', 'manager', 'handler',
                'controller', 'utility', 'helper', 'common', 'shared'
            ],
            TaskCategory.FEATURE_IMPLEMENTATION: [
                'feature', 'implement', 'new', 'add', 'create', 'build',
                'develop', 'api', 'endpoint', 'functionality'
            ],
            TaskCategory.BUG_INVESTIGATION: [
                'bug', 'error', 'exception', 'fix', 'debug', 'issue',
                'problem', 'fault', 'defect', 'trace'
            ],
            TaskCategory.MULTI_SESSION_DEVELOPMENT: [
                'session', 'state', 'persistent', 'cache', 'storage',
                'memory', 'context', 'history', 'tracking'
            ],
            TaskCategory.CODE_COMPREHENSION: [
                'documentation', 'readme', 'guide', 'tutorial', 'example',
                'demo', 'sample', 'overview', 'explanation'
            ],
            TaskCategory.INTEGRATION_TESTING: [
                'integration', 'test', 'spec', 'e2e', 'api', 'client',
                'mock', 'stub', 'fixture', 'scenario'
            ],
            TaskCategory.SECURITY_ANALYSIS: [
                'security', 'auth', 'authentication', 'authorization', 'login',
                'password', 'token', 'crypto', 'hash', 'encrypt', 'ssl'
            ]
        }
        
        keywords = relevance_keywords.get(task_category, [])
        score = 0.0
        
        for keyword in keywords:
            if keyword in file_name:
                score += 1.0
        
        return min(score, 5.0)  # Cap at 5.0

    def _expand_file_selection(
        self, 
        current_files: Dict[str, str], 
        all_files: Dict[str, str], 
        target_coverage: float
    ) -> Dict[str, str]:
        """Expand file selection to meet target coverage."""
        
        remaining_files = {k: v for k, v in all_files.items() if k not in current_files}
        expanded_files = current_files.copy()
        
        # Sort remaining files by size (larger first)
        sorted_remaining = sorted(
            remaining_files.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        for file_path, content in sorted_remaining:
            expanded_files[file_path] = content
            coverage = self._calculate_information_coverage(expanded_files, all_files)
            
            if coverage >= target_coverage:
                break
        
        return expanded_files

    def _estimate_file_complexity(self, content: str) -> float:
        """Estimate file complexity (0.0 to 1.0)."""
        
        if not content:
            return 0.0
        
        lines = content.split('\n')
        
        # Count complexity indicators
        complexity_indicators = 0
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Function/method definitions
            if any(keyword in line_lower for keyword in ['def ', 'function ', 'class ', 'interface ']):
                complexity_indicators += 2
            
            # Control structures
            if any(keyword in line_lower for keyword in ['if ', 'for ', 'while ', 'switch ', 'case ']):
                complexity_indicators += 1
            
            # Error handling
            if any(keyword in line_lower for keyword in ['try ', 'catch ', 'except ', 'finally ']):
                complexity_indicators += 1
            
            # Imports/includes
            if any(keyword in line_lower for keyword in ['import ', 'include ', 'require ', 'use ']):
                complexity_indicators += 0.5
        
        # Normalize by file length
        normalized_complexity = complexity_indicators / max(len(lines), 1)
        
        return min(normalized_complexity, 1.0)
    
    def _select_context_files(self, task_category: TaskCategory, project_files: Dict[str, str]) -> Dict[str, str]:
        """Select relevant files for the task context"""
        
        if not project_files:
            return {}
        
        # Different strategies based on task category
        if task_category == TaskCategory.ARCHITECTURAL_UNDERSTANDING:
            # Focus on main implementation files
            return self._select_files_by_pattern(project_files, ['src/', 'main.', 'app.', 'server.'])
        
        elif task_category == TaskCategory.CROSS_FILE_REFACTORING:
            # Include multiple related files
            return self._select_random_subset(project_files, min_files=3, max_files=8)
        
        elif task_category == TaskCategory.FEATURE_IMPLEMENTATION:
            # Core files where new features would be added
            return self._select_files_by_pattern(project_files, ['src/', 'lib/', 'core/'])
        
        elif task_category == TaskCategory.BUG_INVESTIGATION:
            # Mix of implementation and test files
            return self._select_files_by_pattern(project_files, ['src/', 'test', 'spec'])
        
        elif task_category == TaskCategory.MULTI_SESSION_DEVELOPMENT:
            # Broader context for multi-session work
            return self._select_random_subset(project_files, min_files=5, max_files=12)
        
        elif task_category == TaskCategory.CODE_COMPREHENSION:
            # Focus on complex implementation files
            return self._select_files_by_complexity(project_files, target_count=4)
        
        elif task_category == TaskCategory.INTEGRATION_TESTING:
            # Test files and integration points
            return self._select_files_by_pattern(project_files, ['test', 'spec', 'integration', 'api/'])
        
        elif task_category == TaskCategory.SECURITY_ANALYSIS:
            # Security-relevant files
            return self._select_files_by_pattern(project_files, ['auth', 'security', 'config', 'env'])
        
        # Default: random selection
        return self._select_random_subset(project_files, min_files=2, max_files=6)
    
    def _select_files_by_pattern(self, project_files: Dict[str, str], patterns: List[str]) -> Dict[str, str]:
        """Select files matching any of the given patterns"""
        selected = {}
        
        for file_path, content in project_files.items():
            if any(pattern.lower() in file_path.lower() for pattern in patterns):
                selected[file_path] = content
        
        # If no matches, return a random subset
        if not selected:
            return self._select_random_subset(project_files, min_files=2, max_files=4)
        
        return selected
    
    def _select_random_subset(self, project_files: Dict[str, str], min_files: int = 2, max_files: int = 6) -> Dict[str, str]:
        """Select a random subset of files"""
        file_list = list(project_files.items())
        count = min(max_files, max(min_files, len(file_list)))
        count = min(count, len(file_list))
        
        selected_items = random.sample(file_list, count)
        return dict(selected_items)
    
    def _select_files_by_complexity(self, project_files: Dict[str, str], target_count: int = 4) -> Dict[str, str]:
        """Select files based on complexity (length, lines, etc.)"""
        # Sort files by length (simple complexity metric)
        sorted_files = sorted(project_files.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Take the most complex files up to target count
        selected_count = min(target_count, len(sorted_files))
        return dict(sorted_files[:selected_count])
    
    def _calculate_information_coverage(self, context_files: Dict[str, str], all_project_files: Dict[str, str]) -> float:
        """Calculate information coverage ratio (context size / total project size)"""
        if not all_project_files:
            return 0.0
        
        # Calculate total project size (characters)
        total_project_size = sum(len(content) for content in all_project_files.values())
        
        # Calculate context size (characters)
        context_size = sum(len(content) for content in context_files.values())
        
        if total_project_size == 0:
            return 0.0
        
        # Information coverage as ratio of context to total project
        coverage = context_size / total_project_size
        
        # Cap at 1.0 (100% coverage)
        return min(coverage, 1.0)
    
    def _determine_difficulty(self, context_length: int, project_complexity: str) -> DifficultyLevel:
        """Determine difficulty level based on context and project complexity"""
        
        # Base difficulty on context length
        if context_length < 20000:
            base_difficulty = DifficultyLevel.EASY
        elif context_length < 60000:
            base_difficulty = DifficultyLevel.MEDIUM
        elif context_length < 150000:
            base_difficulty = DifficultyLevel.HARD
        else:
            base_difficulty = DifficultyLevel.EXPERT
        
        # Adjust based on project complexity
        complexity_adjustment = {
            'easy': -1,
            'medium': 0,
            'hard': 1,
            'expert': 2
        }
        
        difficulty_levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD, DifficultyLevel.EXPERT]
        current_index = difficulty_levels.index(base_difficulty)
        adjustment = complexity_adjustment.get(project_complexity.lower(), 0)
        
        new_index = max(0, min(len(difficulty_levels) - 1, current_index + adjustment))
        return difficulty_levels[new_index]
    
    def _determine_difficulty_from_coverage(self, information_coverage: float) -> DifficultyLevel:
        """Determine difficulty level based on information coverage using range-based approach"""
        
        for difficulty_name, (min_coverage, max_coverage) in self.config.phase3.coverage_ranges.items():
            if min_coverage <= information_coverage <= max_coverage:
                # Convert difficulty_name to the correct enum member
                difficulty_map = {
                    "easy": DifficultyLevel.EASY,
                    "medium": DifficultyLevel.MEDIUM,
                    "hard": DifficultyLevel.HARD,
                    "expert": DifficultyLevel.EXPERT
                }
                return difficulty_map[difficulty_name]
        
        # Fallback: if coverage doesn't fit any range, use the closest one
        logger.warning(f"⚠️  Coverage {information_coverage:.2f} doesn't fit any defined range, using fallback")
        
        if information_coverage < 0.20:
            return DifficultyLevel.EASY
        elif information_coverage >= 0.80:
            return DifficultyLevel.EXPERT
        elif information_coverage >= 0.60:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.MEDIUM
    
    async def _generate_scenario_content(
        self,
        task_category: TaskCategory,
        difficulty: DifficultyLevel,
        project_spec: Dict[str, Any],
        context_files: Dict[str, str],
        scenario_id: str
    ) -> Dict[str, Any]:
        """Generate scenario content using LLM"""
        
        # Import Rich console for debugging output
        from rich.console import Console
        console = Console()
        
        # Create context summary
        files_summary = []
        for file_path, content in context_files.items():
            lines = len(content.splitlines())
            chars = len(content)
            files_summary.append(f"- {file_path}: {lines} lines, {chars} chars")
        
        context_summary = "\n".join(files_summary)
        
        prompt = f"""
        Create a realistic {task_category.value} evaluation scenario for long-context LLMs.
        
        PROJECT CONTEXT:
        - Name: {project_spec.get('name', 'Unknown')}
        - Language: {project_spec.get('language', 'Unknown')}
        - Domain: {project_spec.get('domain', 'Unknown')}
        - Features: {', '.join(project_spec.get('features', [])[:5])}
        - Complexity: {project_spec.get('complexity', 'medium')}
        
        AVAILABLE FILES:
        {context_summary}
        
        TASK REQUIREMENTS:
        - Category: {task_category.value}
        - Difficulty: {difficulty.value}
        - Must be realistic and challenging for long-context LLMs
        - Should require understanding of multiple files
        - Include specific, measurable objectives
        
        Generate a JSON response with these fields:
        {{
            "title": "Clear, descriptive title for the task",
            "description": "Detailed description of the scenario and context",
            "task_prompt": "Specific task instructions for the LLM",
            "expected_approach": "How an expert developer would approach this task",
            "ground_truth": "Expected solution or key insights",
            "evaluation_criteria": ["List of criteria to evaluate long-context model performance"]
        }}
        
        Make the scenario realistic and challenging. Focus on {self._get_category_focus(task_category)}.
        """
        
        system_prompt = f"""You are an expert software engineering instructor creating evaluation scenarios for long-context LLMs. Create realistic, challenging tasks that test {task_category.value} capabilities."""
        
        try:
            console.print(f"           🤖 Calling LLM for {task_category.value}...")
            response = await self.llm_generator.generate_with_model(
                self.llm_generator.generators["scenarios"],
                prompt,
                system_prompt
            )
            
            console.print(f"           📝 LLM response length: {len(response)} chars")
            logger.info(f"Raw LLM response for {scenario_id}: {response[:200]}...")
            
            # Debug: Check response type and content
            logger.debug(f"Response type: {type(response)}, is None: {response is None}")
            if response:
                logger.debug(f"Response starts with: {repr(response[:50])}")
                # Check if response ends properly
                if len(response) > 100:
                    logger.debug(f"Response ends with: {repr(response[-100:])}")
                # Check for JSON structure
                has_opening_brace = '{' in response[:50]
                has_closing_brace = '}' in response[-50:]
                logger.debug(f"Has opening brace in first 50 chars: {has_opening_brace}")
                logger.debug(f"Has closing brace in last 50 chars: {has_closing_brace}")
            
            # Robust JSON extraction with multiple strategies
            def extract_json_from_response(response_text: str) -> dict:
                """Extract JSON from LLM response with multiple fallback strategies"""
                if not response_text or not response_text.strip():
                    raise ValueError("Empty response")
                
                # Strategy 1: Direct JSON parsing
                try:
                    return json.loads(response_text.strip())
                except json.JSONDecodeError:
                    pass
                
                # Strategy 2: Extract from markdown code blocks
                import re
                json_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
                if json_match:
                    extracted_text = json_match.group(1).strip()
                    
                    # Try direct parsing first
                    try:
                        return json.loads(extracted_text)
                    except json.JSONDecodeError:
                        pass
                    
                    # Strategy 3: Find complete JSON object by brace counting (improved)
                    if extracted_text.startswith('{'):
                        try:
                            # Use Python's built-in JSON decoder for better handling
                            from json import JSONDecoder
                            decoder = JSONDecoder()
                            obj, idx = decoder.raw_decode(extracted_text)
                            return obj
                        except (json.JSONDecodeError, ValueError):
                            pass
                        
                        # Fallback: Manual brace counting (last resort)
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        
                        for i, char in enumerate(extracted_text):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        complete_json = extracted_text[:i+1]
                                        try:
                                            return json.loads(complete_json)
                                        except json.JSONDecodeError:
                                            break
                
                # Strategy 4: Look for JSON anywhere in the response (not just code blocks)
                # Find first '{' and try to extract from there
                first_brace = response_text.find('{')
                if first_brace != -1:
                    try:
                        from json import JSONDecoder
                        decoder = JSONDecoder()
                        obj, idx = decoder.raw_decode(response_text[first_brace:])
                        return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                
                raise ValueError("Could not extract valid JSON")
            
            # Try to extract JSON using the robust function
            try:
                parsed_response = extract_json_from_response(response)
                console.print(f"           ✅ JSON extraction successful")
                return parsed_response
            except (ValueError, json.JSONDecodeError) as e:
                console.print(f"           ❌ JSON extraction failed: {str(e)}")
                logger.error(f"JSON extraction failed for {scenario_id}: {e}")
                
                # Enhanced debugging for JSON failures
                if response:
                    logger.error(f"Response length: {len(response)} chars")
                    logger.error(f"Response starts: {response[:300]}")
                    logger.error(f"Response ends: {response[-300:]}")
                    
                    # Check for common JSON issues
                    if '```json' in response:
                        logger.error("Found markdown JSON block")
                        json_start = response.find('```json')
                        json_end = response.find('```', json_start + 7)
                        if json_end == -1:
                            logger.error("❌ JSON markdown block not properly closed")
                        else:
                            logger.error("✅ JSON markdown block appears complete")
                    
                    # Check brace balance
                    open_braces = response.count('{')
                    close_braces = response.count('}')
                    logger.error(f"Brace count: {open_braces} open, {close_braces} close")
                    
                logger.error(f"Raw response: {response}")
                
                # Fail immediately if JSON parsing fails - no fallbacks!
                from locobench.generation.synthetic_generator import APIError
                raise APIError(
                    provider="LLM",
                    error_type="INVALID_JSON", 
                    message=f"LLM failed to generate valid JSON for scenario {scenario_id}. Response was: {response[:200] if response else 'None'}..."
                )
                
        except Exception as e:
            console.print(f"           ❌ LLM generation failed: {str(e)}")
            logger.error(f"Failed to generate scenario content for {scenario_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise the original exception instead of using fallback
            raise e
    
    def _get_category_focus(self, task_category: TaskCategory) -> str:
        """Get the focus description for each task category"""
        focus_map = {
            TaskCategory.ARCHITECTURAL_UNDERSTANDING: "system design patterns, component relationships, and architectural decisions",
            TaskCategory.CROSS_FILE_REFACTORING: "code restructuring across multiple files while maintaining functionality",
            TaskCategory.FEATURE_IMPLEMENTATION: "adding new functionality that integrates well with existing code",
            TaskCategory.BUG_INVESTIGATION: "systematic debugging, root cause analysis, and problem solving",
            TaskCategory.MULTI_SESSION_DEVELOPMENT: "incremental development over multiple sessions with context retention",
            TaskCategory.CODE_COMPREHENSION: "deep understanding of complex code structures and logic",
            TaskCategory.INTEGRATION_TESTING: "testing interactions between components and system validation",
            TaskCategory.SECURITY_ANALYSIS: "identifying security vulnerabilities and implementing security best practices"
        }
        return focus_map.get(task_category, "software development best practices")
    


    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat() 