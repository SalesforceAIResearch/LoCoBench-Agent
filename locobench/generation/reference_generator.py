"""
Reference Solution Generator for Phase 4: Quality Validation and Deployment

This module generates reference solutions using multiple Elite Models and creates
comprehensive evaluation rubrics for LoCoBench scenarios.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.config import Config
from ..core.task import TaskCategory
from .synthetic_generator import MultiLLMGenerator


@dataclass
class ReferenceSolution:
    """A reference solution for a scenario"""
    id: str
    scenario_id: str
    model: str  # Which model generated this solution
    solution_approach: str
    implementation_files: Dict[str, str]  # filename -> content
    explanation: str
    quality_score: float
    generation_time: float
    metadata: Dict[str, Any]


@dataclass
class EvaluationRubric:
    """Evaluation rubric for a task category"""
    task_category: str
    scoring_criteria: List[Dict[str, Any]]
    weight_distribution: Dict[str, float]
    difficulty_multipliers: Dict[str, float]
    max_score: int


class ReferenceGenerator:
    """Generates reference solutions and evaluation rubrics for LoCoBench scenarios"""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.llm_generator = MultiLLMGenerator(config)
        
        # Output directories
        self.output_dir = Path(config.data.output_dir)
        self.scenarios_dir = self.output_dir / "scenarios"
        self.references_dir = self.output_dir / "references"
        self.rubrics_dir = self.output_dir / "rubrics"
        
        # Create directories
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.rubrics_dir.mkdir(parents=True, exist_ok=True)

    async def generate_reference_solutions(self, scenario_file: Path, force_regenerate: bool = False) -> List[ReferenceSolution]:
        """Generate 2-3 reference solutions for a scenario using Elite Models"""
        
        # Load scenario
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
        
        solutions = []
        
        for scenario in scenario_data['scenarios']:
            scenario_id = scenario['id']
            task_category = scenario['task_category']
            
            self.console.print(f"🎯 Generating reference solutions for: {scenario['title'][:60]}...")
            
            # Use our 2 Elite Models for diverse solutions
            models = ['openai', 'google']  # OpenAI o3, Gemini 2.5 Pro
            
            for model in models:
                self.console.print(f"   🤖 Using {model} model...")
                
                start_time = time.time()
                try:
                    solution = await self._generate_single_solution(scenario, model)
                    solution.generation_time = time.time() - start_time
                    solution.model = model
                    solutions.append(solution)
                    
                    self.console.print(f"   ✅ Solution generated in {solution.generation_time:.1f}s")
                    
                except Exception as e:
                    self.console.print(f"   ❌ Failed with {model}: {e}")
                    continue
        
        return solutions

    async def _generate_single_solution(self, scenario: Dict[str, Any], model: str) -> ReferenceSolution:
        """Generate a single reference solution using specified model"""
        
        # Prepare prompt for reference solution generation
        prompt = self._create_solution_prompt(scenario)
        
        # Call LLM to generate solution using specified model
        response = await self.llm_generator.generate_with_model(
            model_type=model,
            prompt=prompt,
            system_prompt="You are an expert software engineer creating reference solutions for LoCoBench scenarios."
        )
        
        # Parse the solution from response
        solution_data = self._parse_solution_response(response)
        
        # Create ReferenceSolution object
        solution = ReferenceSolution(
            id=f"{scenario['id']}_ref_{model}",
            scenario_id=scenario['id'],
            model=model,
            solution_approach=solution_data.get('approach', ''),
            implementation_files=solution_data.get('files', {}),
            explanation=solution_data.get('explanation', ''),
            quality_score=self._calculate_quality_score(solution_data, scenario),
            generation_time=0.0,  # Will be set by caller
            metadata={
                'task_category': scenario['task_category'],
                'difficulty': scenario['difficulty'],
                'context_length': scenario.get('context_length', 0)
            }
        )
        
        return solution

    def _create_solution_prompt(self, scenario: Dict[str, Any]) -> str:
        """Create a prompt for generating reference solutions"""
        
        return f"""You are an expert software engineer tasked with creating a reference solution for an LoCoBench scenario.

**Scenario**: {scenario['title']}

**Description**: {scenario['description']}

**Task**: {scenario['task_prompt']}

**Expected Approach**: {scenario['expected_approach']}

**Context Files Available**: {', '.join(scenario['context_files'])}

Please provide a comprehensive reference solution that includes:

1. **Solution Approach**: A clear explanation of your strategy and reasoning
2. **Implementation Files**: Complete code files with proper structure and error handling
3. **Key Implementation Details**: Critical aspects of your solution
4. **Quality Considerations**: How your solution meets the evaluation criteria

Format your response as JSON:
```json
{{
    "approach": "Your detailed solution approach...",
    "files": {{
        "filename1.go": "complete file content...",
        "filename2.go": "complete file content..."
    }},
    "explanation": "Detailed explanation of implementation decisions...",
    "key_features": ["feature1", "feature2", "feature3"],
    "quality_highlights": ["highlight1", "highlight2"]
}}
```

Ensure your solution:
- Follows the existing code patterns and architecture
- Includes proper error handling and validation
- Is production-ready and well-documented
- Addresses all requirements in the task prompt
- Demonstrates best practices for the given programming language"""

    def _parse_solution_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract solution data"""
        
        # Try to extract JSON from markdown code blocks (similar to scenario generation)
        import re
        
        # First try direct JSON parsing
        try:
            # Parse JSON directly - works perfectly without cleaning
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                # Parse extracted JSON directly
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Fail immediately if JSON parsing fails - no fallbacks!
        self.console.print("❌ JSON parsing failed completely for reference solution")
        from locobench.generation.synthetic_generator import APIError
        raise APIError(
            provider="LLM",
            error_type="INVALID_JSON",
            message=f"LLM failed to generate valid JSON for reference solution. Response was: {response[:200] if response else 'None'}..."
        )

    def _calculate_quality_score(self, solution_data: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Calculate a quality score for the generated solution"""
        
        score = 0.0
        
        # Check completeness (40% of score)
        if solution_data.get('approach'):
            score += 0.15
        if solution_data.get('files'):
            score += 0.15
            # Bonus for multiple files
            if len(solution_data['files']) > 1:
                score += 0.10
        
        # Check explanation quality (30% of score)
        explanation = solution_data.get('explanation', '')
        if len(explanation) > 100:
            score += 0.15
        if len(explanation) > 500:
            score += 0.15
        
        # Check key features (20% of score)
        features = solution_data.get('key_features', [])
        if len(features) >= 3:
            score += 0.20
        elif len(features) >= 1:
            score += 0.10
        
        # Check quality highlights (10% of score)
        highlights = solution_data.get('quality_highlights', [])
        if len(highlights) >= 2:
            score += 0.10
        elif len(highlights) >= 1:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0

    def create_evaluation_rubric(self, task_category: TaskCategory) -> EvaluationRubric:
        """Create evaluation rubric for a specific task category"""
        
        # Task-specific rubrics based on the 6 novel metrics
        rubrics = {
            TaskCategory.ARCHITECTURAL_UNDERSTANDING: {
                "scoring_criteria": [
                    {"name": "Pattern Recognition", "description": "Identifies architectural patterns correctly", "max_points": 20},
                    {"name": "Dependency Analysis", "description": "Maps module dependencies accurately", "max_points": 20},
                    {"name": "Component Relationships", "description": "Understands component interactions", "max_points": 20},
                    {"name": "Design Principles", "description": "Recognizes design principles applied", "max_points": 20},
                    {"name": "Documentation Quality", "description": "Provides clear architectural documentation", "max_points": 20}
                ],
                "weight_distribution": {"ACS": 0.4, "DTA": 0.3, "CFRD": 0.2, "ICU": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.2, "hard": 1.5, "expert": 2.0}
            },
            TaskCategory.CROSS_FILE_REFACTORING: {
                "scoring_criteria": [
                    {"name": "Code Quality", "description": "Improves code structure and maintainability", "max_points": 25},
                    {"name": "Cross-File Coordination", "description": "Manages changes across multiple files", "max_points": 25},
                    {"name": "Backward Compatibility", "description": "Maintains existing functionality", "max_points": 20},
                    {"name": "Best Practices", "description": "Follows language and design best practices", "max_points": 15},
                    {"name": "Testing Integration", "description": "Ensures tests continue to pass", "max_points": 15}
                ],
                "weight_distribution": {"CFRD": 0.4, "ACS": 0.3, "IDC": 0.2, "DTA": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.3, "hard": 1.6, "expert": 2.2}
            },
            TaskCategory.FEATURE_IMPLEMENTATION: {
                "scoring_criteria": [
                    {"name": "Functionality", "description": "Implements required features correctly", "max_points": 30},
                    {"name": "Integration", "description": "Integrates with existing codebase", "max_points": 25},
                    {"name": "Error Handling", "description": "Includes robust error handling", "max_points": 20},
                    {"name": "Performance", "description": "Considers performance implications", "max_points": 15},
                    {"name": "Documentation", "description": "Documents new features appropriately", "max_points": 10}
                ],
                "weight_distribution": {"IDC": 0.4, "ACS": 0.3, "ICU": 0.2, "CFRD": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.4, "hard": 1.8, "expert": 2.5}
            },
            TaskCategory.BUG_INVESTIGATION: {
                "scoring_criteria": [
                    {"name": "Problem Identification", "description": "Correctly identifies the root cause", "max_points": 30},
                    {"name": "Investigation Process", "description": "Uses systematic debugging approach", "max_points": 25},
                    {"name": "Code Analysis", "description": "Analyzes relevant code sections thoroughly", "max_points": 20},
                    {"name": "Solution Quality", "description": "Provides effective solution or fix", "max_points": 15},
                    {"name": "Impact Assessment", "description": "Understands bug impact and implications", "max_points": 10}
                ],
                "weight_distribution": {"DTA": 0.4, "CFRD": 0.3, "ICU": 0.2, "ACS": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.3, "hard": 1.7, "expert": 2.3}
            },
            TaskCategory.MULTI_SESSION_DEVELOPMENT: {
                "scoring_criteria": [
                    {"name": "Session Planning", "description": "Breaks down work into logical sessions", "max_points": 25},
                    {"name": "Context Retention", "description": "Maintains context between sessions", "max_points": 25},
                    {"name": "Incremental Progress", "description": "Makes meaningful progress each session", "max_points": 20},
                    {"name": "Integration Testing", "description": "Tests integration at each milestone", "max_points": 15},
                    {"name": "Documentation Updates", "description": "Keeps documentation current", "max_points": 15}
                ],
                "weight_distribution": {"MMR": 0.5, "IDC": 0.3, "ACS": 0.1, "ICU": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.5, "hard": 2.0, "expert": 3.0}
            },
            TaskCategory.CODE_COMPREHENSION: {
                "scoring_criteria": [
                    {"name": "Code Understanding", "description": "Demonstrates deep code comprehension", "max_points": 30},
                    {"name": "Flow Analysis", "description": "Traces data and control flow accurately", "max_points": 25},
                    {"name": "Pattern Recognition", "description": "Identifies patterns and conventions", "max_points": 20},
                    {"name": "Complexity Assessment", "description": "Assesses code complexity appropriately", "max_points": 15},
                    {"name": "Documentation Quality", "description": "Explains findings clearly", "max_points": 10}
                ],
                "weight_distribution": {"ICU": 0.4, "CFRD": 0.3, "DTA": 0.2, "ACS": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.2, "hard": 1.5, "expert": 2.0}
            },
            TaskCategory.INTEGRATION_TESTING: {
                "scoring_criteria": [
                    {"name": "Test Coverage", "description": "Covers critical integration points", "max_points": 30},
                    {"name": "Test Quality", "description": "Writes effective and maintainable tests", "max_points": 25},
                    {"name": "Edge Cases", "description": "Considers edge cases and error conditions", "max_points": 20},
                    {"name": "Test Organization", "description": "Organizes tests logically", "max_points": 15},
                    {"name": "Performance Testing", "description": "Includes performance considerations", "max_points": 10}
                ],
                "weight_distribution": {"CFRD": 0.4, "ICU": 0.3, "ACS": 0.2, "DTA": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.3, "hard": 1.6, "expert": 2.2}
            },
            TaskCategory.SECURITY_ANALYSIS: {
                "scoring_criteria": [
                    {"name": "Vulnerability Identification", "description": "Identifies security vulnerabilities", "max_points": 35},
                    {"name": "Risk Assessment", "description": "Assesses risk levels accurately", "max_points": 25},
                    {"name": "Remediation", "description": "Provides effective remediation strategies", "max_points": 20},
                    {"name": "Best Practices", "description": "Demonstrates security best practices", "max_points": 15},
                    {"name": "Documentation", "description": "Documents findings professionally", "max_points": 5}
                ],
                "weight_distribution": {"ICU": 0.4, "ACS": 0.3, "DTA": 0.2, "CFRD": 0.1},
                "difficulty_multipliers": {"easy": 1.0, "medium": 1.4, "hard": 1.8, "expert": 2.5}
            }
        }
        
        rubric_data = rubrics.get(task_category, rubrics[TaskCategory.ARCHITECTURAL_UNDERSTANDING])
        
        return EvaluationRubric(
            task_category=task_category.value,
            scoring_criteria=rubric_data["scoring_criteria"],
            weight_distribution=rubric_data["weight_distribution"],
            difficulty_multipliers=rubric_data["difficulty_multipliers"],
            max_score=100
        )

    async def save_enhanced_scenarios(self, scenario_file: Path, solutions: List[ReferenceSolution], rubric: EvaluationRubric):
        """Save scenarios enhanced with reference solutions and evaluation rubrics"""
        
        # Load original scenario
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
        
        # Group solutions by scenario_id
        solutions_by_scenario = {}
        for solution in solutions:
            scenario_id = solution.scenario_id
            if scenario_id not in solutions_by_scenario:
                solutions_by_scenario[scenario_id] = []
            solutions_by_scenario[scenario_id].append(solution)
        
        # Enhance each scenario with reference solutions
        for scenario in scenario_data['scenarios']:
            scenario_id = scenario['id']
            scenario_solutions = solutions_by_scenario.get(scenario_id, [])
            
            # Add reference solutions
            scenario['reference_solutions'] = []
            for solution in scenario_solutions:
                scenario['reference_solutions'].append({
                    'id': solution.id,
                    'model': solution.model,
                    'approach': solution.solution_approach,
                    'implementation_files': solution.implementation_files,
                    'explanation': solution.explanation,
                    'quality_score': solution.quality_score,
                    'generation_time': solution.generation_time,
                    'metadata': solution.metadata
                })
            
            # Add evaluation rubric
            scenario['evaluation_rubric'] = {
                'task_category': rubric.task_category,
                'scoring_criteria': rubric.scoring_criteria,
                'weight_distribution': rubric.weight_distribution,
                'difficulty_multipliers': rubric.difficulty_multipliers,
                'max_score': rubric.max_score
            }
        
        # Add Phase 4 metadata
        scenario_data['phase_4_metadata'] = {
            'reference_solutions_count': len(solutions),
            'elite_models_used': ['openai', 'google'],
            'generation_timestamp': time.time(),
            'quality_validation': True,
            'evaluation_framework': 'LoCoBench-v1'
        }
        
        # Save enhanced scenario
        enhanced_file = self.references_dir / scenario_file.name
        with open(enhanced_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        
        self.console.print(f"✅ Enhanced scenario saved: {enhanced_file.name}") 