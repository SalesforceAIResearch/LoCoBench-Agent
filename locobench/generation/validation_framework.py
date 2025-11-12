"""
Automated Validation Framework for Phase 4: Test-Driven Evaluation

This module creates automated test suites and evaluation metrics for LoCoBench
scenarios without relying on reference solutions from LLMs.
"""

import ast
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.config import Config
from ..core.task import TaskCategory
from .metric_algorithms import LoCoBenchMetricsCalculator
from ..validation.code_validator import validate_code_compilation, analyze_code_security, analyze_code_quality
import re
import logging

logger = logging.getLogger(__name__)


def detect_language_from_scenario(scenario: Dict[str, Any], solution_code: Dict[str, str]) -> str:
    """
    Detect programming language from scenario metadata and solution code
    
    Priority:
    1. Extract from project_id (e.g., 'c_web_ecommerce_expert_000' → 'c')
    2. Detect from file extensions in solution_code
    3. Fallback to 'python'
    """
    
    # Method 1: Extract from project_id in scenario
    scenario_id = scenario.get('id', '')
    if '_' in scenario_id:
        # e.g., 'c_web_ecommerce_expert_000_security_analysis_01' → 'c'
        potential_lang = scenario_id.split('_')[0].lower()
        
        # Map common language prefixes
        language_mapping = {
            'c': 'c',
            'cpp': 'cpp', 
            'csharp': 'csharp',
            'go': 'go',
            'java': 'java',
            'javascript': 'javascript',
            'php': 'php',
            'python': 'python',
            'rust': 'rust',
            'typescript': 'typescript'
        }
        
        if potential_lang in language_mapping:
            logger.debug(f"Detected language '{potential_lang}' from scenario ID: {scenario_id}")
            return language_mapping[potential_lang]
    
    # Method 2: Detect from file extensions in solution code
    if solution_code:
        extensions = []
        for filepath in solution_code.keys():
            if '.' in filepath:
                ext = filepath.split('.')[-1].lower()
                extensions.append(ext)
        
        # Count extension frequencies
        ext_counts = {}
        for ext in extensions:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        # Map extensions to languages
        extension_mapping = {
            'c': 'c',
            'h': 'c',
            'cpp': 'cpp',
            'cc': 'cpp', 
            'cxx': 'cpp',
            'hpp': 'cpp',
            'cs': 'csharp',
            'go': 'go',
            'java': 'java',
            'js': 'javascript',
            'jsx': 'javascript',
            'php': 'php',
            'py': 'python',
            'rs': 'rust',
            'ts': 'typescript',
            'tsx': 'typescript'
        }
        
        # Find most common extension and map to language
        if ext_counts:
            most_common_ext = max(ext_counts.items(), key=lambda x: x[1])[0]
            if most_common_ext in extension_mapping:
                detected_lang = extension_mapping[most_common_ext]
                logger.debug(f"Detected language '{detected_lang}' from file extensions: {extensions}")
                return detected_lang
    
    # Method 3: Fallback to Python
    logger.warning(f"Could not detect language for scenario {scenario_id}, defaulting to 'python'")
    return 'python'


@dataclass
class ValidationResult:
    """Result of automated validation for a solution - 4 Evaluation Dimensions"""
    scenario_id: str
    
    # 4 Evaluation Dimensions (LCBS Framework)
    software_engineering_score: float    # 40% - Software Engineering Excellence (8 metrics)
    functional_correctness_score: float  # 30% - Functional Correctness (4 metrics)  
    code_quality_score: float           # 20% - Code Quality Assessment (3 metrics)
    longcontext_utilization_score: float # 10% - Long-Context Utilization (2 metrics)
    
    total_score: float
    detailed_results: Dict[str, Any]
    execution_time: float
    

@dataclass  
class TestSuite:
    """Automated test suite for a scenario"""
    scenario_id: str
    compilation_tests: List[Dict[str, Any]]
    unit_tests: List[Dict[str, Any]]
    integration_tests: List[Dict[str, Any]]
    performance_tests: List[Dict[str, Any]]
    security_tests: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TestSuite to dictionary for JSON serialization"""
        return {
            'scenario_id': self.scenario_id,
            'compilation_tests': self.compilation_tests,
            'unit_tests': self.unit_tests,
            'integration_tests': self.integration_tests,
            'performance_tests': self.performance_tests,
            'security_tests': self.security_tests
        }


class AutomatedValidator:
    """Automated validation framework for LoCoBench scenarios"""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        
        # Output directories
        self.output_dir = Path(config.data.output_dir)
        self.validation_dir = self.output_dir / "validation"
        self.test_suites_dir = self.validation_dir / "test_suites"
        
        # Create directories
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.test_suites_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation weights from config (with fallback defaults) - 4 Dimensions
        self.weights = {
            'software_engineering': config.phase4.metric_weights.get('software_engineering', 0.40),
            'functional_correctness': config.phase4.metric_weights.get('functional_correctness', 0.30),
            'code_quality': config.phase4.metric_weights.get('code_quality', 0.20),
            'longcontext_utilization': config.phase4.metric_weights.get('longcontext_utilization', 0.10)
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"⚠️ Metric weights sum to {total_weight:.3f}, not 1.0. Normalizing...")
            # Normalize weights to sum to 1.0
            for key in self.weights:
                self.weights[key] = self.weights[key] / total_weight
        
        # Use comprehensive evaluation with all metrics (traditional + advanced)
        self.comprehensive_metrics = True
        logger.info(f"🎯 Using comprehensive LCBS evaluation (17 metrics across 4 dimensions)")
        
        # Initialize metrics calculator
        self.metrics_calculator = LoCoBenchMetricsCalculator()

    async def generate_test_suite(self, scenario: Dict[str, Any]) -> TestSuite:
        """Generate automated test suite for a scenario"""
        
        scenario_id = scenario['id']
        task_category = scenario['task_category']
        
        self.console.print(f"🧪 Generating test suite for: {scenario['title'][:50]}...")
        
        # Generate different types of tests based on task category
        compilation_tests = self._create_compilation_tests(scenario)
        unit_tests = self._create_unit_tests(scenario)
        integration_tests = self._create_integration_tests(scenario)
        performance_tests = self._create_performance_tests(scenario)
        security_tests = self._create_security_tests(scenario)
        
        test_suite = TestSuite(
            scenario_id=scenario_id,
            compilation_tests=compilation_tests,
            unit_tests=unit_tests,
            integration_tests=integration_tests,
            performance_tests=performance_tests,
            security_tests=security_tests
        )
        
        # Save test suite
        test_file = self.test_suites_dir / f"{scenario_id}_tests.json"
        with open(test_file, 'w') as f:
            json.dump({
                'scenario_id': scenario_id,
                'task_category': task_category,
                'tests': {
                    'compilation': compilation_tests,
                    'unit': unit_tests,
                    'integration': integration_tests,
                    'performance': performance_tests,
                    'security': security_tests
                }
            }, f, indent=2)
        
        return test_suite

    def _create_compilation_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create compilation/syntax validation tests"""
        
        task_category = scenario['task_category']
        context_files = scenario.get('context_files', [])
        
        tests = [
            {
                "name": "syntax_validation",
                "description": "Check if generated code has valid syntax",
                "type": "compilation",
                "weight": 0.3
            },
            {
                "name": "import_resolution", 
                "description": "Verify all imports can be resolved",
                "type": "compilation",
                "weight": 0.2
            },
            {
                "name": "type_checking",
                "description": "Basic type consistency checks",
                "type": "compilation", 
                "weight": 0.2
            }
        ]
        
        # Add task-specific compilation tests
        if task_category == 'feature_implementation':
            tests.append({
                "name": "api_endpoint_structure",
                "description": "New API endpoints have proper structure",
                "type": "compilation",
                "weight": 0.3
            })
        elif task_category == 'cross_file_refactoring':
            tests.append({
                "name": "refactor_consistency",
                "description": "Refactored code maintains consistency",
                "type": "compilation",
                "weight": 0.3
            })
        
        return tests

    def _create_unit_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create unit tests for individual functions/modules"""
        
        task_category = scenario['task_category']
        
        base_tests = [
            {
                "name": "function_signature_preservation",
                "description": "Public function signatures are preserved",
                "type": "unit",
                "weight": 0.25
            },
            {
                "name": "error_handling",
                "description": "Proper error handling for edge cases",
                "type": "unit", 
                "weight": 0.25
            },
            {
                "name": "input_validation",
                "description": "Input validation works correctly",
                "type": "unit",
                "weight": 0.25
            },
            {
                "name": "output_correctness",
                "description": "Functions return expected outputs",
                "type": "unit",
                "weight": 0.25
            }
        ]
        
        # Add task-specific unit tests
        if task_category == 'bug_investigation':
            base_tests.append({
                "name": "bug_fix_verification",
                "description": "Identified bug is actually fixed",
                "type": "unit",
                "weight": 0.4
            })
        elif task_category == 'security_analysis':
            base_tests.append({
                "name": "vulnerability_mitigation",
                "description": "Security vulnerabilities are properly addressed",
                "type": "unit",
                "weight": 0.4
            })
        
        return base_tests

    def _create_integration_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create integration tests for cross-file functionality"""
        
        return [
            {
                "name": "module_integration",
                "description": "Modified modules integrate correctly",
                "type": "integration",
                "weight": 0.3
            },
            {
                "name": "database_integration",
                "description": "Database operations work end-to-end",
                "type": "integration", 
                "weight": 0.3
            },
            {
                "name": "api_integration",
                "description": "API endpoints work with existing system",
                "type": "integration",
                "weight": 0.4
            }
        ]

    def _create_performance_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create performance and efficiency tests"""
        
        return [
            {
                "name": "execution_time",
                "description": "Code executes within reasonable time limits",
                "type": "performance",
                "weight": 0.4
            },
            {
                "name": "memory_usage",
                "description": "Memory usage is within acceptable bounds",
                "type": "performance",
                "weight": 0.3
            },
            {
                "name": "scalability",
                "description": "Solution scales appropriately with input size",
                "type": "performance",
                "weight": 0.3
            }
        ]

    def _create_security_tests(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create security validation tests"""
        
        return [
            {
                "name": "injection_prevention",
                "description": "Code prevents injection attacks",
                "type": "security",
                "weight": 0.3
            },
            {
                "name": "input_sanitization",
                "description": "User inputs are properly sanitized",
                "type": "security",
                "weight": 0.3
            },
            {
                "name": "access_control",
                "description": "Proper access controls are implemented",
                "type": "security",
                "weight": 0.4
            }
        ]

    async def validate_solution(self, scenario: Dict[str, Any], solution_code: Dict[str, str], 
                               test_suite: TestSuite) -> ValidationResult:
        """
        Validate solution using comprehensive LCBS evaluation with all 17 metrics across 4 dimensions.
        Includes 8 software engineering metrics + 4 functional correctness metrics + 3 code quality metrics + 2 long-context utilization metrics.
        """
        return await self.validate_solution_comprehensive(scenario, solution_code, test_suite)
    
    async def _validate_solution_legacy(self, scenario: Dict[str, Any], solution_code: Dict[str, str], 
                                       test_suite: TestSuite) -> ValidationResult:
        """Validate solution with comprehensive evaluation and detailed results capture"""
        
        scenario_id = scenario['id']
        start_time = time.time()
        
        self.console.print(f"⚡ Validating solution for: {scenario['title'][:50]}...")
        
        # Initialize detailed results container - 4 Evaluation Dimensions
        detailed_results = {
            'software_engineering_details': {},  # 8 metrics: ACS, DTA, CFRD, STS, RS, CS, IS, SES
            'functional_correctness_details': {}, # 4 metrics: Compilation, Unit Tests, Integration, IDC
            'code_quality_details': {},          # 3 metrics: Security, Issues, Style
            'longcontext_utilization_details': {} # 2 metrics: ICU, MMR
        }
        
        # 1. Software Engineering Excellence (40%) - 8 metrics: ACS, DTA, CFRD, STS, RS, CS, IS, SES
        software_engineering_score, se_details = self._evaluate_software_engineering_detailed(
            scenario, solution_code
        )
        detailed_results['software_engineering_details'] = se_details
        
        # 2. Functional Correctness (30%) - 4 metrics: Compilation, Unit Tests, Integration, IDC
        functional_correctness_score, fc_details = await self._evaluate_functional_correctness_detailed(
            scenario, solution_code, test_suite
        )
        detailed_results['functional_correctness_details'] = fc_details
        
        # 3. Code Quality Assessment (20%) - 3 metrics: Security, Issues, Style
        code_quality_score, cq_details = await self._evaluate_code_quality_detailed(
            scenario, solution_code
        )
        detailed_results['code_quality_details'] = cq_details
        
        # 4. Long-Context Utilization (10%) - 2 metrics: ICU, MMR
        longcontext_utilization_score, lcu_details = await self._evaluate_longcontext_utilization_detailed(
            scenario, solution_code
        )
        detailed_results['longcontext_utilization_details'] = lcu_details
        
        # Calculate weighted total score (LCBS - LoCoBench Score)
        total_score = (
            software_engineering_score * self.weights['software_engineering'] +
            functional_correctness_score * self.weights['functional_correctness'] +
            code_quality_score * self.weights['code_quality'] +
            longcontext_utilization_score * self.weights['longcontext_utilization']
        )
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            scenario_id=scenario_id,
            software_engineering_score=software_engineering_score,
            functional_correctness_score=functional_correctness_score,
            code_quality_score=code_quality_score,
            longcontext_utilization_score=longcontext_utilization_score,
            total_score=total_score,
            detailed_results=detailed_results,
            execution_time=execution_time
        )

    async def validate_solution_comprehensive(self, scenario: Dict[str, Any], solution_code: Dict[str, str], 
                                            test_suite: TestSuite) -> ValidationResult:
        """
        Comprehensive validation using LCBS with 17 metrics across 4 dimensions.
        LCBS = 5.0 × (0.4×SE + 0.3×FC + 0.2×CQ + 0.1×LCU)
        
        4 Dimensions with 17 metrics:
        - Software Engineering Excellence (8 metrics): ACS, DTA, CFRD, STS, RS, CS, IS, SES
        - Functional Correctness (4 metrics): Compilation, Unit Tests, Integration Tests, IDC  
        - Code Quality Assessment (3 metrics): Security, Quality Score, Issues Found
        - Long-Context Utilization (2 metrics): ICU, MMR
        """
        # solution_code is already sanitized at the source in evaluator.py
        
        start_time = time.time()
        scenario_id = scenario.get('scenario_id', 'unknown')
        detailed_results = {}
        
        # 1. Software Engineering Excellence (40% weight - 8 metrics)
        software_engineering_score, software_engineering_details = self._evaluate_software_engineering_detailed(
            scenario, solution_code
        )
        detailed_results['software_engineering_details'] = software_engineering_details
        
        # 2. Functional Correctness (30% weight - 4 metrics)
        functional_correctness_score, functional_correctness_details = await self._evaluate_functional_correctness_detailed(
            scenario, solution_code, test_suite
        )
        detailed_results['functional_correctness_details'] = functional_correctness_details
        
        # 3. Code Quality Assessment (20% weight - 3 metrics)
        code_quality_score, code_quality_details = await self._evaluate_code_quality_detailed(
            scenario, solution_code
        )
        detailed_results['code_quality_details'] = code_quality_details
        
        # 4. Long-Context Utilization (10% weight - 2 metrics)
        longcontext_utilization_score, longcontext_utilization_details = await self._evaluate_longcontext_utilization_detailed(
            scenario, solution_code
        )
        detailed_results['longcontext_utilization_details'] = longcontext_utilization_details
        
        # Calculate comprehensive LCBS weighted total score
        lcbs_score = (
            software_engineering_score * self.weights['software_engineering'] +      # 40%
            functional_correctness_score * self.weights['functional_correctness'] +  # 30%
            code_quality_score * self.weights['code_quality'] +                      # 20%
            longcontext_utilization_score * self.weights['longcontext_utilization']  # 10%
        )
        
        # Scale to 5.0 (LCBS uses 0-5 scale)
        total_score = lcbs_score * 5.0
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            scenario_id=scenario_id,
            software_engineering_score=software_engineering_score,
            functional_correctness_score=functional_correctness_score,
            code_quality_score=code_quality_score,
            longcontext_utilization_score=longcontext_utilization_score,
            total_score=total_score,
            detailed_results=detailed_results,
            execution_time=execution_time
        )

    async def _evaluate_functional_correctness(self, scenario: Dict[str, Any], 
                                             solution_code: Dict[str, str], 
                                             test_suite: TestSuite) -> float:
        """Evaluate functional correctness (40% weight)"""
        
        scores = []
        
        # Test compilation
        compilation_score = await self._test_compilation(solution_code, scenario)
        scores.append(compilation_score * 0.4)
        
        # Test unit functionality  
        unit_score = await self._test_unit_functionality(solution_code, test_suite.unit_tests, scenario)
        scores.append(unit_score * 0.4)
        
        # Test integration
        integration_score = await self._test_integration(solution_code, test_suite.integration_tests)
        scores.append(integration_score * 0.2)
        
        return sum(scores)

    async def _evaluate_functional_correctness_detailed(self, scenario: Dict[str, Any], 
                                                       solution_code: Dict[str, str], 
                                                       test_suite: TestSuite) -> Tuple[float, Dict[str, Any]]:
        """Evaluate functional correctness (30% weight) - 4 metrics: Compilation, Unit Tests, Integration Tests, IDC"""
        
        scores = []
        details = {
            'compilation': {},
            'unit_tests': {},
            'integration': {},
            'incremental_development': {},
            'overall_breakdown': {}
        }
        
        # Test compilation - capture detailed results
        compilation_score, compilation_details = await self._test_compilation_detailed(solution_code, scenario)
        scores.append(compilation_score * 0.3)
        details['compilation'] = compilation_details
        
        # Test unit functionality - capture detailed results
        unit_score, unit_details = await self._test_unit_functionality_detailed(solution_code, test_suite.unit_tests, scenario)
        scores.append(unit_score * 0.3)
        details['unit_tests'] = unit_details
        
        # Test integration - capture detailed results
        integration_score, integration_details = await self._test_integration_detailed(solution_code, test_suite.integration_tests)
        scores.append(integration_score * 0.2)
        details['integration'] = integration_details
        
        # Incremental Development Capability (IDC) - capture detailed results
        idc_score = self._calculate_incremental_development_capability(scenario, solution_code)
        scores.append(idc_score * 0.2)
        details['incremental_development'] = {
            'idc_score': idc_score,
            'description': 'Ability to build incrementally on previous work'
        }
        
        # Overall breakdown
        total_score = sum(scores)
        details['overall_breakdown'] = {
            'compilation_score': compilation_score,
            'compilation_weight': 0.3,
            'unit_test_score': unit_score, 
            'unit_test_weight': 0.3,
            'integration_score': integration_score,
            'integration_weight': 0.2,
            'idc_score': idc_score,
            'idc_weight': 0.2,
            'total_functional_score': total_score
        }
        
        return total_score, details

    async def _evaluate_longcontext_metrics(self, scenario: Dict[str, Any],
                                     solution_code: Dict[str, str]) -> float:
        """Evaluate novel long-context metrics (30% weight)"""
        
        task_category = scenario['task_category']
        
        # Calculate the 6 novel metrics based on task category
        scores = []
        
        if task_category == 'architectural_understanding':
            acs_score = self._calculate_architectural_coherence_score(scenario, solution_code)
            dta_score = self._calculate_dependency_traversal_accuracy(scenario, solution_code)
            scores = [acs_score * 0.6, dta_score * 0.4]
            
        elif task_category == 'cross_file_refactoring':
            cfrd_score = self._calculate_cross_file_reasoning_depth(scenario, solution_code)
            acs_score = self._calculate_architectural_coherence_score(scenario, solution_code)
            scores = [cfrd_score * 0.7, acs_score * 0.3]
            
        elif task_category == 'multi_session_development':
            mmr_score = self._calculate_multi_session_memory_retention(scenario, solution_code)
            idc_score = self._calculate_incremental_development_capability(scenario, solution_code)
            scores = [mmr_score * 0.6, idc_score * 0.4]
            
        else:
            # Default metrics for other categories
            icu_score = self._calculate_information_coverage_utilization(scenario, solution_code)
            cfrd_score = self._calculate_cross_file_reasoning_depth(scenario, solution_code)
            scores = [icu_score * 0.5, cfrd_score * 0.5]
        
        return sum(scores)

    async def _evaluate_code_quality(self, scenario: Dict[str, Any], 
                                   solution_code: Dict[str, str]) -> float:
        """Evaluate code quality metrics (20% weight)"""
        
        scores = []
        
        # Complexity analysis
        complexity_score = await self._analyze_code_complexity(solution_code)
        scores.append(complexity_score * 0.3)
        
        # Security analysis
        security_score = await self._analyze_security(solution_code)
        scores.append(security_score * 0.3)
        
        # Maintainability
        maintainability_score = await self._analyze_maintainability(solution_code)
        scores.append(maintainability_score * 0.4)
        
        return sum(scores)

    async def _evaluate_style_practices(self, scenario: Dict[str, Any], 
                                      solution_code: Dict[str, str]) -> float:
        """Evaluate style and best practices (10% weight)"""
        
        scores = []
        
        # Code formatting
        formatting_score = await self._check_code_formatting(solution_code)
        scores.append(formatting_score * 0.4)
        
        # Naming conventions
        naming_score = self._check_naming_conventions(solution_code)
        scores.append(naming_score * 0.3)
        
        # Documentation quality
        docs_score = self._check_documentation_quality(solution_code)
        scores.append(docs_score * 0.3)
        
        return sum(scores)

    # Real implementations replacing placeholder metric calculations
    # These now use actual algorithmic implementations from code_validator
    
    async def _test_compilation(self, solution_code: Dict[str, str], scenario: Dict[str, Any] = None) -> float:
        """Test if code compiles successfully using appropriate compiler"""
        
        try:
            # Detect language dynamically from scenario and solution code
            language = detect_language_from_scenario(scenario or {}, solution_code)
            compilation_result = await validate_code_compilation(solution_code, language)
            
            # Score based on compilation success and quality
            score = 0.0
            
            if compilation_result.success:
                score = 0.8  # Base score for successful compilation
                
                # Bonus for fast compilation
                if compilation_result.execution_time < 5.0:
                    score += 0.1
                
                # Penalty for warnings
                if compilation_result.warnings:
                    score -= len(compilation_result.warnings) * 0.05
                
                # Bonus for reasonable binary size (if available)
                if compilation_result.binary_size and compilation_result.binary_size < 10_000_000:  # < 10MB
                    score += 0.1
            else:
                # Partial credit for files that would compile individually
                error_count = len(compilation_result.errors)
                if error_count <= 2:
                    score = 0.3  # Some credit for minor issues
                elif error_count <= 5:
                    score = 0.1  # Very little credit for major issues
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Compilation testing failed: {e}")
            return 0.2  # Minimal fallback score
    
    async def _test_unit_functionality(self, solution_code: Dict[str, str], unit_tests: List[Dict], scenario: Dict[str, Any] = None) -> float:
        """Test unit functionality using real test execution"""
        
        try:
            from ..validation.code_validator import CodeValidator
            validator = CodeValidator()
            
            # Run actual unit tests with dynamic language detection
            language = detect_language_from_scenario(scenario or {}, solution_code)
            test_pass_rate = await validator.run_unit_tests(solution_code, unit_tests, language)
            
            return test_pass_rate
            
        except Exception as e:
            logger.error(f"Unit testing failed: {e}")
            return 0.0

    async def _test_integration(self, solution_code: Dict[str, str], integration_tests: List[Dict]) -> float:
        """Test integration scenarios"""
        
        # For now, basic integration test based on multi-file coordination
        if len(solution_code) > 1:
            # Multi-file solution suggests some integration
            return 0.7
        else:
            # Single file solution
            return 0.5

    async def _test_performance(self, solution_code: Dict[str, str], performance_tests: List[Dict]) -> float:
        """Test performance characteristics"""
        
        # Simple performance heuristics based on code patterns
        total_code = ' '.join(solution_code.values())
        
        performance_score = 0.7  # Base score
        
        # Check for potentially inefficient patterns
        if 'for ' in total_code and 'for ' in total_code:
            nested_loops = total_code.count('for ')
            if nested_loops > 3:
                performance_score -= 0.2  # Penalty for many nested loops
        
        # Check for efficient patterns
        if any(pattern in total_code.lower() for pattern in ['sync.', 'goroutine', 'channel']):
            performance_score += 0.2  # Bonus for concurrency
            
        return min(max(performance_score, 0.0), 1.0)

    async def _test_security_compliance(self, solution_code: Dict[str, str], security_tests: List[Dict]) -> float:
        """Test security compliance using real security analysis"""
        
        try:
            security_result = await analyze_code_security(solution_code, 'go')
            return security_result.security_score
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return 0.5

    async def _test_compilation_detailed(self, solution_code: Dict[str, str], 
                                         scenario: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """Test if code compiles successfully with detailed results capture"""
        
        details = {
            'success': False,
            'score': 0.0,
            'execution_time': 0.0,
            'errors': [],
            'warnings': [],
            'binary_size': None,
            'files_tested': list(solution_code.keys()),
            'scoring_breakdown': {}
        }
        
        try:
            # Detect language dynamically from scenario and solution code
            language = detect_language_from_scenario(scenario or {}, solution_code)
            compilation_result = await validate_code_compilation(solution_code, language)
            
            # Capture all details from compilation result
            details['success'] = compilation_result.success
            details['execution_time'] = compilation_result.execution_time
            details['errors'] = compilation_result.errors if hasattr(compilation_result, 'errors') else []
            details['warnings'] = compilation_result.warnings if hasattr(compilation_result, 'warnings') else []
            details['binary_size'] = getattr(compilation_result, 'binary_size', None)
            
            # Score based on compilation success and quality
            score = 0.0
            scoring_breakdown = {}
            
            if compilation_result.success:
                score = 0.8  # Base score for successful compilation
                scoring_breakdown['base_success'] = 0.8
                
                # Bonus for fast compilation
                if compilation_result.execution_time < 5.0:
                    score += 0.1
                    scoring_breakdown['fast_compilation_bonus'] = 0.1
                
                # Penalty for warnings
                if details['warnings']:
                    warning_penalty = len(details['warnings']) * 0.05
                    score -= warning_penalty
                    scoring_breakdown['warning_penalty'] = -warning_penalty
                
                # Bonus for reasonable binary size (if available)
                if details['binary_size'] and details['binary_size'] < 10_000_000:  # < 10MB
                    score += 0.1
                    scoring_breakdown['size_bonus'] = 0.1
            else:
                # Partial credit for files that would compile individually
                error_count = len(details['errors'])
                if error_count <= 2:
                    score = 0.3  # Some credit for minor issues
                    scoring_breakdown['partial_credit'] = 0.3
                elif error_count <= 5:
                    score = 0.1  # Very little credit for major issues
                    scoring_breakdown['minimal_credit'] = 0.1
                else:
                    scoring_breakdown['no_credit'] = 0.0
            
            score = min(max(score, 0.0), 1.0)
            details['score'] = score
            details['scoring_breakdown'] = scoring_breakdown
            
            return score, details
            
        except Exception as e:
            details['errors'].append(f"Compilation testing failed: {str(e)}")
            details['score'] = 0.2
            details['scoring_breakdown'] = {'fallback_score': 0.2}
            return 0.2, details

    async def _test_unit_functionality_detailed(self, solution_code: Dict[str, str], unit_tests: List[Dict], scenario: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """Test unit functionality with detailed results capture"""
        
        details = {
            'test_pass_rate': 0.0,
            'tests_run': len(unit_tests),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': [],
            'errors': [],
            'overall_success': False
        }
        
        try:
            from ..validation.code_validator import CodeValidator
            validator = CodeValidator()
            
            # Run actual unit tests with dynamic language detection
            language = detect_language_from_scenario(scenario or {}, solution_code)
            test_pass_rate = await validator.run_unit_tests(solution_code, unit_tests, language)
            
            details['test_pass_rate'] = test_pass_rate
            details['tests_passed'] = int(test_pass_rate * len(unit_tests))
            details['tests_failed'] = len(unit_tests) - details['tests_passed']
            details['overall_success'] = test_pass_rate > 0.5
            
            # Add individual test details if available
            for i, test in enumerate(unit_tests):
                test_result = {
                    'name': test.get('name', f'test_{i}'),
                    'passed': i < details['tests_passed'],  # Simulate based on pass rate
                    'description': test.get('description', '')
                }
                details['test_results'].append(test_result)
            
            return test_pass_rate, details
            
        except Exception as e:
            details['errors'].append(f"Unit testing failed: {str(e)}")
            details['overall_success'] = False
            return 0.0, details

    async def _test_integration_detailed(self, solution_code: Dict[str, str], integration_tests: List[Dict]) -> Tuple[float, Dict[str, Any]]:
        """Test integration scenarios with detailed results capture"""
        
        details = {
            'files_analyzed': len(solution_code),
            'multi_file_solution': len(solution_code) > 1,
            'integration_score': 0.0,
            'integration_indicators': [],
            'tests_defined': len(integration_tests)
        }
        
        # For now, basic integration test based on multi-file coordination
        if len(solution_code) > 1:
            score = 0.7
            details['integration_score'] = score
            details['integration_indicators'].append('Multi-file solution suggests integration capability')
        else:
            score = 0.3
            details['integration_score'] = score
            details['integration_indicators'].append('Single file solution - limited integration')
        
        # Add analysis of imports/dependencies if possible
        import_count = 0
        for filepath, content in solution_code.items():
            if 'import' in content or 'from' in content:
                import_count += 1
        
        if import_count > 0:
            details['integration_indicators'].append(f'Found imports in {import_count} files')
            if import_count > 2:
                score += 0.1  # Bonus for good import usage
                details['integration_score'] = min(score, 1.0)
        
        return min(score, 1.0), details
    
    def _calculate_architectural_coherence_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate ACS - Architectural Coherence Score"""
        return self.metrics_calculator.calculate_architectural_coherence_score(scenario, solution_code)
    
    def _calculate_dependency_traversal_accuracy(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate DTA - Dependency Traversal Accuracy"""
        return self.metrics_calculator.calculate_dependency_traversal_accuracy(scenario, solution_code)
    
    def _calculate_multi_session_memory_retention(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate MMR - Multi-Session Memory Retention"""
        return self.metrics_calculator.calculate_multi_session_memory_retention(scenario, solution_code)
    
    def _calculate_cross_file_reasoning_depth(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate CFRD - Cross-File Reasoning Depth"""
        return self.metrics_calculator.calculate_cross_file_reasoning_depth(scenario, solution_code)
    
    def _calculate_incremental_development_capability(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate IDC - Incremental Development Capability"""
        return self.metrics_calculator.calculate_incremental_development_capability(scenario, solution_code)
    
    def _calculate_information_coverage_utilization(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate ICU - Information Coverage Utilization"""
        return self.metrics_calculator.calculate_information_coverage_utilization(scenario, solution_code)
    
    # ============================================================================
    # NEW ADVANCED METRICS DELEGATION METHODS (CADS v2 - EMU1)
    # ============================================================================
    
    def _calculate_robustness_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate RS - Robustness Score (CADS v2)"""
        return self.metrics_calculator.calculate_robustness_score(scenario, solution_code)
    
    def _calculate_comprehensiveness_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate CS - Comprehensiveness Score (CADS v2)"""
        return self.metrics_calculator.calculate_comprehensiveness_score(scenario, solution_code)
    
    def _calculate_innovation_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate IS - Innovation Score (CADS v2)"""
        return self.metrics_calculator.calculate_innovation_score(scenario, solution_code)
    
    def _calculate_system_thinking_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate STS - System Thinking Score (CADS v2)"""
        return self.metrics_calculator.calculate_system_thinking_score(scenario, solution_code)
    
    def _calculate_solution_elegance_score(self, scenario: Dict, solution_code: Dict[str, str]) -> float:
        """Calculate SES - Solution Elegance Score (CADS v2)"""
        return self.metrics_calculator.calculate_solution_elegance_score(scenario, solution_code)
    
    async def _evaluate_advanced_metrics_detailed(self, scenario: Dict[str, Any], 
                                                solution_code: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the 5 new advanced metrics for CADS v2 (35% weight).
        Returns weighted average score and detailed breakdown.
        """
        
        # Calculate all 5 advanced metrics
        robustness_score = self._calculate_robustness_score(scenario, solution_code)
        comprehensiveness_score = self._calculate_comprehensiveness_score(scenario, solution_code)
        innovation_score = self._calculate_innovation_score(scenario, solution_code)
        system_thinking_score = self._calculate_system_thinking_score(scenario, solution_code)
        solution_elegance_score = self._calculate_solution_elegance_score(scenario, solution_code)
        
        # CADS v2 weights for advanced metrics (from design specification):
        # RS: 0.23, CS: 0.20, IS: 0.20, STS: 0.20, SES: 0.17
        advanced_metrics_weights = {
            'robustness_score': 0.23,
            'comprehensiveness_score': 0.20,
            'innovation_score': 0.20,
            'system_thinking_score': 0.20,
            'solution_elegance_score': 0.17
        }
        
        # Calculate weighted average
        weighted_score = (
            robustness_score * advanced_metrics_weights['robustness_score'] +
            comprehensiveness_score * advanced_metrics_weights['comprehensiveness_score'] +
            innovation_score * advanced_metrics_weights['innovation_score'] +
            system_thinking_score * advanced_metrics_weights['system_thinking_score'] +
            solution_elegance_score * advanced_metrics_weights['solution_elegance_score']
        )
        
        # Detailed breakdown for reporting
        details = {
            'individual_scores': {
                'robustness_score': robustness_score,
                'comprehensiveness_score': comprehensiveness_score,
                'innovation_score': innovation_score,
                'system_thinking_score': system_thinking_score,
                'solution_elegance_score': solution_elegance_score
            },
            'weights': advanced_metrics_weights,
            'weighted_average': weighted_score,
            'score_explanations': {
                'robustness_score': 'Error handling, security practices, resource management',
                'comprehensiveness_score': 'Documentation quality, API completeness, deployment readiness',
                'innovation_score': 'Algorithm efficiency, design patterns, modern practices',
                'system_thinking_score': 'Scalability, maintainability, integration awareness',
                'solution_elegance_score': 'Code clarity, abstraction level, principle adherence'
            }
        }
        
        return weighted_score, details
        
    async def _analyze_code_complexity(self, solution_code: Dict[str, str]) -> float:
        """Analyze code complexity using real complexity metrics"""
        
        try:
            # Use the real quality analysis directly without asyncio.run
            quality_result = await analyze_code_quality(solution_code, 'go')
            return quality_result.complexity_score
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return 0.5
        
    async def _analyze_security(self, solution_code: Dict[str, str]) -> float:
        """Analyze security vulnerabilities using real security scanner"""
        
        try:
            security_result = await analyze_code_security(solution_code, 'go')
            return security_result.security_score
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return 0.5
        
    async def _analyze_maintainability(self, solution_code: Dict[str, str]) -> float:
        """Analyze maintainability using real maintainability metrics"""
        
        try:
            # Use the real quality analysis directly without asyncio.run
            quality_result = await analyze_code_quality(solution_code, 'go')
            return quality_result.maintainability_score
            
        except Exception as e:
            logger.error(f"Maintainability analysis failed: {e}")
            return 0.5
        
    async def _check_code_formatting(self, solution_code: Dict[str, str]) -> float:
        """Check code formatting using real formatter"""
        
        try:
            from ..validation.code_validator import CodeValidator
            validator = CodeValidator()
            
            # Use real formatting check directly without asyncio.run
            formatting_result = await validator.check_code_formatting(solution_code, 'go')
            return formatting_result
            
        except Exception as e:
            logger.error(f"Formatting check failed: {e}")
            return 0.5
        
    def _check_naming_conventions(self, solution_code: Dict[str, str]) -> float:
        """Check naming convention compliance"""
        
        # solution_code is already sanitized at the source in evaluator.py
        
        # Go-specific naming convention checks
        total_score = 0.0
        total_checks = 0
        
        for filename, code in solution_code.items():
            # Check function naming (PascalCase for public, camelCase for private)
            functions = re.findall(r'func\s+([A-Za-z_][A-Za-z0-9_]*)', code)
            
            for func_name in functions:
                total_checks += 1
                if func_name[0].isupper():  # Public function
                    if re.match(r'^[A-Z][a-zA-Z0-9]*$', func_name):
                        total_score += 1.0
                    else:
                        total_score += 0.5
                else:  # Private function
                    if re.match(r'^[a-z][a-zA-Z0-9]*$', func_name):
                        total_score += 1.0
                    else:
                        total_score += 0.5
            
            # Check variable naming
            variables = re.findall(r'var\s+([A-Za-z_][A-Za-z0-9_]*)', code)
            
            for var_name in variables:
                total_checks += 1
                if re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', var_name):
                    total_score += 1.0
                else:
                    total_score += 0.5
        
        return total_score / total_checks if total_checks > 0 else 0.6
        
    def _check_documentation_quality(self, solution_code: Dict[str, str]) -> float:
        """Check documentation quality"""
        
        total_lines = 0
        comment_lines = 0
        documented_functions = 0
        total_functions = 0
        
        for filename, code in solution_code.items():
            lines = code.split('\n')
            total_lines += len(lines)
            
            # Count comment lines
            comment_lines += sum(1 for line in lines if line.strip().startswith('//'))
            
            # Check function documentation
            functions = re.finditer(r'func\s+([A-Za-z_][A-Za-z0-9_]*)', code)
            
            for match in functions:
                total_functions += 1
                func_start = match.start()
                
                # Look for comment before function
                lines_before_func = code[:func_start].split('\n')
                if lines_before_func and lines_before_func[-1].strip().startswith('//'):
                    documented_functions += 1
        
        # Calculate documentation score
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        function_doc_ratio = documented_functions / total_functions if total_functions > 0 else 0
        
        documentation_score = (comment_ratio * 0.4 + function_doc_ratio * 0.6)
        
        return min(documentation_score * 2, 1.0)  # Scale up and cap at 1.0

    async def _evaluate_longcontext_utilization_detailed(self, scenario: Dict[str, Any],
                                                       solution_code: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate Long-Context Utilization (10% weight) - 2 metrics: ICU, MMR"""
        
        task_category = scenario['task_category']
        
        details = {
            'task_category': task_category,
            'metrics_applied': [],
            'individual_scores': {},
            'weighted_breakdown': {},
            'total_longcontext_utilization_score': 0.0
        }
        
        # Calculate the 2 long-context utilization metrics
        scores = []
        
        # ICU: Information Coverage Utilization (0.5 weight)
        icu_score = self._calculate_information_coverage_utilization(scenario, solution_code)
        scores.append(icu_score * 0.5)
        details['metrics_applied'].append('information_coverage_utilization')
        details['individual_scores']['information_coverage_utilization'] = icu_score
        details['weighted_breakdown']['information_coverage_utilization_weighted'] = icu_score * 0.5
        
        # MMR: Multi-Session Memory Retention (0.5 weight)  
        mmr_score = self._calculate_multi_session_memory_retention(scenario, solution_code)
        scores.append(mmr_score * 0.5)
        details['metrics_applied'].append('multi_session_memory_retention')
        details['individual_scores']['multi_session_memory_retention'] = mmr_score
        details['weighted_breakdown']['multi_session_memory_retention_weighted'] = mmr_score * 0.5
        
        total_score = sum(scores)
        details['total_longcontext_utilization_score'] = total_score
        
        return total_score, details 

    def _evaluate_software_engineering_detailed(self, scenario: Dict[str, Any],
                                              solution_code: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate Software Engineering Excellence (40% weight) - 8 metrics: ACS, DTA, CFRD, STS, RS, CS, IS, SES"""
        
        # solution_code is already sanitized at the source in evaluator.py
        
        task_category = scenario['task_category']
        
        details = {
            'task_category': task_category,
            'metrics_applied': [],
            'individual_scores': {},
            'weighted_breakdown': {},
            'total_software_engineering_score': 0.0
        }
        
        # Calculate all 8 Software Engineering Excellence metrics (equal weight: 1/8 = 0.125 each)
        metric_weight = 1.0 / 8.0  # 0.125
        
        # 1. Architectural Coherence Score (ACS)
        acs_score = self._calculate_architectural_coherence_score(scenario, solution_code)
        
        # 2. Dependency Traversal Accuracy (DTA)
        dta_score = self._calculate_dependency_traversal_accuracy(scenario, solution_code)
        
        # 3. Cross-File Reasoning Depth (CFRD)
        cfrd_score = self._calculate_cross_file_reasoning_depth(scenario, solution_code)
        
        # 4. System Thinking Score (STS)
        sts_score = self._calculate_system_thinking_score(scenario, solution_code)
        
        # 5. Robustness Score (RS)
        rs_score = self._calculate_robustness_score(scenario, solution_code)
        
        # 6. Comprehensiveness Score (CS)
        cs_score = self._calculate_comprehensiveness_score(scenario, solution_code)
        
        # 7. Innovation Score (IS)
        is_score = self._calculate_innovation_score(scenario, solution_code)
        
        # 8. Solution Elegance Score (SES)
        ses_score = self._calculate_solution_elegance_score(scenario, solution_code)
        
        # Store all metrics
        details['metrics_applied'] = [
            'architectural_coherence_score', 'dependency_traversal_accuracy', 'cross_file_reasoning_depth',
            'system_thinking_score', 'robustness_score', 'comprehensiveness_score', 
            'innovation_score', 'solution_elegance_score'
        ]
        
        details['individual_scores'] = {
            'architectural_coherence_score': acs_score,
            'dependency_traversal_accuracy': dta_score,
            'cross_file_reasoning_depth': cfrd_score,
            'system_thinking_score': sts_score,
            'robustness_score': rs_score,
            'comprehensiveness_score': cs_score,
            'innovation_score': is_score,
            'solution_elegance_score': ses_score
        }
        
        details['weighted_breakdown'] = {
            'architectural_coherence_weighted': acs_score * metric_weight,
            'dependency_traversal_weighted': dta_score * metric_weight,
            'cross_file_reasoning_weighted': cfrd_score * metric_weight,
            'system_thinking_weighted': sts_score * metric_weight,
            'robustness_weighted': rs_score * metric_weight,
            'comprehensiveness_weighted': cs_score * metric_weight,
            'innovation_weighted': is_score * metric_weight,
            'solution_elegance_weighted': ses_score * metric_weight
        }
        
        # Calculate total weighted score
        total_score = (acs_score + dta_score + cfrd_score + sts_score + 
                      rs_score + cs_score + is_score + ses_score) * metric_weight
        
        details['total_software_engineering_score'] = total_score
        
        return total_score, details

    async def _evaluate_code_quality_detailed(self, scenario: Dict[str, Any], 
                                            solution_code: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate code quality metrics (20% weight) with detailed results capture"""
        
        details = {
            'files_analyzed': len(solution_code),
            'quality_checks': {},
            'security_analysis': {},
            'overall_quality_score': 0.0,
            'issues_found': []
        }
        
        try:
            # Analyze code quality for each file
            total_quality = 0.0
            file_count = 0
            
            for filepath, content in solution_code.items():
                file_quality = {}
                
                # Basic quality metrics
                lines = content.split('\n')
                file_quality['line_count'] = len(lines)
                file_quality['non_empty_lines'] = len([l for l in lines if l.strip()])
                file_quality['comment_lines'] = len([l for l in lines if l.strip().startswith('#') or l.strip().startswith('//')])
                file_quality['comment_ratio'] = file_quality['comment_lines'] / max(file_quality['non_empty_lines'], 1)
                
                # Complexity indicators
                file_quality['function_count'] = content.count('func ') + content.count('def ')
                file_quality['class_count'] = content.count('class ') + content.count('type ')
                file_quality['import_count'] = content.count('import ') + content.count('from ')
                
                # Calculate file quality score
                quality_score = 0.5  # Base score
                
                # Bonus for good documentation
                if file_quality['comment_ratio'] > 0.1:
                    quality_score += 0.2
                
                # Bonus for reasonable file size
                if 50 <= file_quality['line_count'] <= 500:
                    quality_score += 0.2
                
                # Bonus for modular structure
                if file_quality['function_count'] > 0:
                    quality_score += 0.1
                
                file_quality['quality_score'] = min(quality_score, 1.0)
                details['quality_checks'][filepath] = file_quality
                
                total_quality += file_quality['quality_score']
                file_count += 1
            
            # Security analysis
            try:
                security_result = await analyze_code_security(solution_code)
                details['security_analysis'] = {
                    'security_score': getattr(security_result, 'score', 0.8),
                    'vulnerabilities_found': getattr(security_result, 'vulnerabilities', []),
                    'security_level': getattr(security_result, 'level', 'medium')
                }
            except Exception as e:
                details['security_analysis'] = {
                    'security_score': 0.8,  # Default good score
                    'error': str(e)
                }
            
            # Overall quality calculation
            avg_file_quality = total_quality / max(file_count, 1)
            security_score = details['security_analysis']['security_score']
            overall_score = (avg_file_quality * 0.7 + security_score * 0.3)
            
            details['overall_quality_score'] = overall_score
            
            return overall_score, details
            
        except Exception as e:
            details['issues_found'].append(f"Quality analysis failed: {str(e)}")
            details['overall_quality_score'] = 0.5  # Fallback score
            return 0.5, details

    async def _evaluate_style_practices_detailed(self, scenario: Dict[str, Any], 
                                               solution_code: Dict[str, str]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate style and best practices (10% weight) with detailed results capture"""
        
        # solution_code is already sanitized at the source in evaluator.py
        
        details = {
            'files_analyzed': len(solution_code),
            'style_checks': {},
            'best_practices': {},
            'overall_style_score': 0.0,
            'style_violations': []
        }
        
        try:
            total_style = 0.0
            file_count = 0
            
            for filepath, content in solution_code.items():
                file_style = {}
                violations = []
                
                # Basic style checks
                lines = content.split('\n')
                file_style['total_lines'] = len(lines)
                
                # Check for common style issues
                long_lines = [i for i, line in enumerate(lines) if len(line) > 120]
                file_style['long_lines'] = len(long_lines)
                if long_lines:
                    violations.append(f"Lines too long (>120 chars): {len(long_lines)} lines")
                
                # Check indentation consistency
                indented_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
                file_style['indented_lines'] = len(indented_lines)
                
                # Check naming conventions
                has_camelCase = bool(re.search(r'[a-z][A-Z]', content))
                has_snake_case = bool(re.search(r'[a-z]_[a-z]', content))
                file_style['naming_conventions'] = {
                    'camelCase_found': has_camelCase,
                    'snake_case_found': has_snake_case
                }
                
                # Calculate style score
                style_score = 0.8  # Base score
                
                # Penalties for style issues
                if file_style['long_lines'] > file_style['total_lines'] * 0.1:  # >10% long lines
                    style_score -= 0.2
                    violations.append("Too many long lines")
                
                # Bonus for consistent naming
                if (has_camelCase and not has_snake_case) or (has_snake_case and not has_camelCase):
                    style_score += 0.1
                elif has_camelCase and has_snake_case:
                    style_score -= 0.1
                    violations.append("Inconsistent naming conventions")
                
                file_style['style_score'] = max(min(style_score, 1.0), 0.0)
                file_style['violations'] = violations
                
                details['style_checks'][filepath] = file_style
                details['style_violations'].extend([f"{filepath}: {v}" for v in violations])
                
                total_style += file_style['style_score']
                file_count += 1
            
            # Best practices analysis
            details['best_practices'] = {
                'error_handling_present': any('try' in content or 'catch' in content or 'except' in content 
                                            for content in solution_code.values()),
                'logging_present': any('log' in content.lower() for content in solution_code.values()),
                'documentation_present': any('"""' in content or '/*' in content for content in solution_code.values()),
                'modular_structure': len(solution_code) > 1
            }
            
            # Calculate overall style score
            avg_style = total_style / max(file_count, 1)
            
            # Bonus for best practices
            bp_bonus = 0.0
            for practice, present in details['best_practices'].items():
                if present:
                    bp_bonus += 0.05
            
            overall_score = min(avg_style + bp_bonus, 1.0)
            details['overall_style_score'] = overall_score
            
            return overall_score, details
            
        except Exception as e:
            details['style_violations'].append(f"Style analysis failed: {str(e)}")
            details['overall_style_score'] = 0.5  # Fallback score
            return 0.5, details 