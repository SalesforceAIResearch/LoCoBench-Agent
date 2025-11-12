"""
Execution Testing Framework for LoCoBench Metrics Revision

This module implements the core execution-based testing infrastructure that replaces
keyword matching and LLM-based evaluation with objective, deterministic code execution.

Key Features:
- Functional requirement testing through actual code execution
- Performance profiling and measurement
- Cross-file integration testing
- AST-based structural analysis
- Memory and runtime efficiency measurement

This is the foundation of our bias-free, LCBA-aligned evaluation system.
"""

import ast
import os
import sys
import subprocess
import tempfile
import time
import tracemalloc
import psutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Classification of evaluation scenarios by type"""
    WEB_API = "web_api"
    BUG_FIX = "bug_fix"
    SYSTEM_DESIGN = "system_design"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    UNKNOWN = "unknown"


class TaskComplexity(Enum):
    """Classification of task complexity levels"""
    SIMPLE = "simple"      # 1-2 files, basic functionality
    MODERATE = "moderate"  # 3-5 files, some complexity
    COMPLEX = "complex"    # 6+ files, high complexity
    EXPERT = "expert"      # System-level, performance-critical


@dataclass
class ExecutionResult:
    """Result of code execution testing"""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    memory_usage: float
    timeout: bool = False
    error_message: Optional[str] = None


@dataclass
class FunctionalTest:
    """A functional test case for requirement validation"""
    name: str
    command: List[str]
    expected_output: Optional[str] = None
    expected_return_code: int = 0
    timeout: int = 30
    description: str = ""


@dataclass
class PerformanceProfile:
    """Performance profiling results"""
    peak_memory_mb: float
    execution_time_ms: float
    cpu_usage_percent: float
    io_operations: int = 0
    function_calls: int = 0


class ExecutionTestingFramework:
    """
    Core execution testing framework for bias-free metric evaluation.
    
    This framework provides the foundation for:
    1. Functional requirement testing through actual code execution
    2. Performance measurement and profiling
    3. Cross-file integration testing
    4. AST-based structural analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    # ===== TASK CLASSIFICATION =====
    
    def classify_task_type(self, scenario: Dict[str, Any]) -> TaskType:
        """
        Classify the task type based on scenario description.
        This enables adaptive evaluation based on task characteristics.
        """
        description = scenario.get('description', '').lower()
        task_category = scenario.get('task_category', '').lower()
        
        # Check for specific task indicators
        if any(keyword in description for keyword in ['api', 'endpoint', 'rest', 'http', 'server']):
            return TaskType.WEB_API
        elif any(keyword in description for keyword in ['bug', 'fix', 'error', 'issue']):
            return TaskType.BUG_FIX
        elif any(keyword in description for keyword in ['test', 'testing', 'unit test', 'integration test']):
            return TaskType.TESTING
        elif any(keyword in description for keyword in ['performance', 'optimize', 'speed', 'memory']):
            return TaskType.PERFORMANCE_OPTIMIZATION
        elif any(keyword in description for keyword in ['refactor', 'restructure', 'reorganize']):
            return TaskType.REFACTORING
        elif any(keyword in description for keyword in ['document', 'documentation', 'readme']):
            return TaskType.DOCUMENTATION
        elif any(keyword in description for keyword in ['analyze', 'analysis', 'understand', 'explain']):
            return TaskType.ANALYSIS
        elif any(keyword in description for keyword in ['implement', 'create', 'build', 'develop']):
            return TaskType.IMPLEMENTATION
        elif any(keyword in description for keyword in ['system', 'architecture', 'design']):
            return TaskType.SYSTEM_DESIGN
        else:
            return TaskType.UNKNOWN
    
    def classify_task_complexity(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> TaskComplexity:
        """
        Classify task complexity based on scenario and solution characteristics.
        This enables progressive difficulty scaling in evaluation.
        """
        # Analyze solution characteristics
        total_lines = sum(len(content.split('\n')) for content in solution_code.values())
        num_files = len(solution_code)
        
        # Analyze code complexity using AST
        total_functions = 0
        total_classes = 0
        max_nesting_depth = 0
        
        for filepath, content in solution_code.items():
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        # Calculate nesting depth
                        depth = self._calculate_nesting_depth(node)
                        max_nesting_depth = max(max_nesting_depth, depth)
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        
            except Exception:
                continue
        
        # Complexity scoring
        complexity_score = (
            num_files * 2 +
            total_lines / 50 +
            total_functions * 1.5 +
            total_classes * 3 +
            max_nesting_depth * 2
        )
        
        # Classify based on score
        if complexity_score < 10:
            return TaskComplexity.SIMPLE
        elif complexity_score < 30:
            return TaskComplexity.MODERATE
        elif complexity_score < 60:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth of control structures"""
        max_depth = 0
        current_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(child, ast.FunctionDef) and child != node:
                # Don't count nested functions as nesting depth
                continue
        
        return max_depth
    
    # ===== FUNCTIONAL TESTING =====
    
    def generate_functional_tests(self, scenario: Dict[str, Any], task_type: TaskType) -> List[FunctionalTest]:
        """
        Generate functional tests based on scenario and task type.
        This replaces keyword-based requirement checking with actual execution testing.
        """
        tests = []
        
        if task_type == TaskType.WEB_API:
            tests.extend(self._generate_api_tests(scenario))
        elif task_type == TaskType.BUG_FIX:
            tests.extend(self._generate_bug_fix_tests(scenario))
        elif task_type == TaskType.TESTING:
            tests.extend(self._generate_test_validation_tests(scenario))
        elif task_type == TaskType.PERFORMANCE_OPTIMIZATION:
            tests.extend(self._generate_performance_tests(scenario))
        else:
            # Generic implementation tests
            tests.extend(self._generate_generic_tests(scenario))
        
        return tests
    
    def _generate_api_tests(self, scenario: Dict[str, Any]) -> List[FunctionalTest]:
        """Generate tests for web API scenarios"""
        tests = []
        
        # Test 1: Basic import and syntax check
        tests.append(FunctionalTest(
            name="syntax_check",
            command=[sys.executable, "-m", "py_compile", "main.py"],
            description="Check if main API file compiles without syntax errors"
        ))
        
        # Test 2: Try to run the API server (with timeout)
        tests.append(FunctionalTest(
            name="server_startup",
            command=[sys.executable, "main.py"],
            timeout=5,  # Short timeout for startup test
            description="Check if API server can start without immediate crashes"
        ))
        
        return tests
    
    def _generate_bug_fix_tests(self, scenario: Dict[str, Any]) -> List[FunctionalTest]:
        """Generate tests for bug fix scenarios"""
        tests = []
        
        # Test 1: Run the fixed code
        tests.append(FunctionalTest(
            name="bug_fix_execution",
            command=[sys.executable, "main.py"],
            description="Check if bug fix allows code to run without errors"
        ))
        
        return tests
    
    def _generate_test_validation_tests(self, scenario: Dict[str, Any]) -> List[FunctionalTest]:
        """Generate tests for testing scenarios"""
        tests = []
        
        # Test 1: Run pytest if test files exist
        tests.append(FunctionalTest(
            name="pytest_execution",
            command=[sys.executable, "-m", "pytest", "-v"],
            description="Run pytest on test files"
        ))
        
        # Test 2: Run unittest if no pytest
        tests.append(FunctionalTest(
            name="unittest_execution",
            command=[sys.executable, "-m", "unittest", "discover"],
            description="Run unittest discovery"
        ))
        
        return tests
    
    def _generate_performance_tests(self, scenario: Dict[str, Any]) -> List[FunctionalTest]:
        """Generate tests for performance optimization scenarios"""
        tests = []
        
        # Test 1: Basic execution with profiling
        tests.append(FunctionalTest(
            name="performance_execution",
            command=[sys.executable, "main.py"],
            description="Execute code to measure performance improvements"
        ))
        
        return tests
    
    def _generate_generic_tests(self, scenario: Dict[str, Any]) -> List[FunctionalTest]:
        """Generate generic tests for unknown task types"""
        tests = []
        
        # Test 1: Syntax validation
        tests.append(FunctionalTest(
            name="syntax_validation",
            command=[sys.executable, "-m", "py_compile", "main.py"],
            description="Validate Python syntax"
        ))
        
        # Test 2: Basic execution
        tests.append(FunctionalTest(
            name="basic_execution",
            command=[sys.executable, "main.py"],
            description="Basic code execution test"
        ))
        
        return tests
    
    # ===== CODE EXECUTION =====
    
    def execute_code(self, solution_code: Dict[str, str], test: FunctionalTest) -> ExecutionResult:
        """
        Execute code in a sandboxed environment and measure results.
        This is the core of our execution-first evaluation approach.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write all code files
                for filepath, content in solution_code.items():
                    full_path = os.path.join(temp_dir, filepath)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w') as f:
                        f.write(content)
                
                # Start performance monitoring
                start_time = time.time()
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Execute the test command
                try:
                    result = subprocess.run(
                        test.command,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=test.timeout
                    )
                    
                    # Measure performance
                    execution_time = time.time() - start_time
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage = final_memory - initial_memory
                    
                    # Check success criteria
                    success = (
                        result.returncode == test.expected_return_code and
                        (test.expected_output is None or test.expected_output in result.stdout)
                    )
                    
                    return ExecutionResult(
                        success=success,
                        return_code=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        timeout=False
                    )
                    
                except subprocess.TimeoutExpired:
                    return ExecutionResult(
                        success=False,
                        return_code=-1,
                        stdout="",
                        stderr="Execution timeout",
                        execution_time=test.timeout,
                        memory_usage=0.0,
                        timeout=True,
                        error_message="Execution timed out"
                    )
                    
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                memory_usage=0.0,
                timeout=False,
                error_message=str(e)
            )
    
    # ===== PERFORMANCE PROFILING =====
    
    def profile_performance(self, solution_code: Dict[str, str]) -> PerformanceProfile:
        """
        Profile code performance including memory usage, execution time, and CPU usage.
        This provides objective performance measurements for efficiency metrics.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write all code files
                for filepath, content in solution_code.items():
                    full_path = os.path.join(temp_dir, filepath)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w') as f:
                        f.write(content)
                
                # Create profiling script
                profiler_script = f'''
import psutil
import time
import tracemalloc
import sys
import os

def profile_execution():
    # Start monitoring
    tracemalloc.start()
    process = psutil.Process()
    start_time = time.time()
    start_cpu = process.cpu_percent()
    
    try:
        # Try to import and run main functionality
        sys.path.insert(0, "{temp_dir}")
        
        # Look for main entry points
        main_files = ["main.py", "app.py", "__main__.py"]
        for main_file in main_files:
            if os.path.exists(main_file):
                exec(open(main_file).read())
                break
        
        # Measure results
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = (end_time - start_time) * 1000  # ms
        peak_memory = peak / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()
        
        print(f"EXECUTION_TIME: {{execution_time:.2f}}")
        print(f"PEAK_MEMORY: {{peak_memory:.2f}}")
        print(f"CPU_USAGE: {{cpu_usage:.2f}}")
        
    except Exception as e:
        print(f"PROFILING_ERROR: {{str(e)}}")

if __name__ == "__main__":
    profile_execution()
'''
                
                profiler_path = os.path.join(temp_dir, "profiler.py")
                with open(profiler_path, 'w') as f:
                    f.write(profiler_script)
                
                # Run profiler
                result = subprocess.run(
                    [sys.executable, profiler_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse results
                execution_time = 0.0
                peak_memory = 0.0
                cpu_usage = 0.0
                
                for line in result.stdout.split('\n'):
                    if line.startswith("EXECUTION_TIME:"):
                        execution_time = float(line.split(":")[1].strip())
                    elif line.startswith("PEAK_MEMORY:"):
                        peak_memory = float(line.split(":")[1].strip())
                    elif line.startswith("CPU_USAGE:"):
                        cpu_usage = float(line.split(":")[1].strip())
                
                return PerformanceProfile(
                    peak_memory_mb=peak_memory,
                    execution_time_ms=execution_time,
                    cpu_usage_percent=cpu_usage
                )
                
        except Exception as e:
            self.logger.error(f"Performance profiling failed: {e}")
            return PerformanceProfile(
                peak_memory_mb=0.0,
                execution_time_ms=0.0,
                cpu_usage_percent=0.0
            )
    
    # ===== INTEGRATION TESTING =====
    
    def test_cross_file_integration(self, solution_code: Dict[str, str]) -> float:
        """
        Test cross-file integration by attempting to import and execute files
        that depend on others. This replaces keyword-based dependency analysis.
        """
        if len(solution_code) <= 1:
            return 1.0  # Single file = perfect integration
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write all files
                for filepath, content in solution_code.items():
                    full_path = os.path.join(temp_dir, filepath)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w') as f:
                        f.write(content)
                
                # Test each file's imports
                successful_imports = 0
                total_files_with_imports = 0
                
                for filepath, content in solution_code.items():
                    if 'import' in content or 'from' in content:
                        total_files_with_imports += 1
                        
                        # Test if this file can be imported
                        module_name = os.path.splitext(os.path.basename(filepath))[0]
                        test_script = f'''
import sys
sys.path.insert(0, "{temp_dir}")

try:
    import {module_name}
    print("SUCCESS: {filepath}")
except Exception as e:
    print(f"ERROR: {filepath}: {{str(e)}}")
'''
                        
                        result = subprocess.run(
                            [sys.executable, '-c', test_script],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if "SUCCESS:" in result.stdout:
                            successful_imports += 1
                
                if total_files_with_imports == 0:
                    return 1.0  # No imports to test
                
                return successful_imports / total_files_with_imports
                
        except Exception as e:
            self.logger.error(f"Integration testing failed: {e}")
            return 0.0
    
    # ===== CONTEXTUAL BASELINES =====
    
    def get_contextual_baseline(self, task_type: TaskType, complexity: TaskComplexity) -> Dict[str, Any]:
        """
        Get contextual baselines for adaptive evaluation.
        This enables fair comparison against task-appropriate expectations.
        """
        baselines = {
            TaskType.WEB_API: {
                TaskComplexity.SIMPLE: {"lines": 50, "files": 2, "functions": 5},
                TaskComplexity.MODERATE: {"lines": 150, "files": 4, "functions": 12},
                TaskComplexity.COMPLEX: {"lines": 300, "files": 8, "functions": 25},
                TaskComplexity.EXPERT: {"lines": 500, "files": 15, "functions": 40}
            },
            TaskType.BUG_FIX: {
                TaskComplexity.SIMPLE: {"lines": 20, "files": 1, "functions": 2},
                TaskComplexity.MODERATE: {"lines": 50, "files": 2, "functions": 5},
                TaskComplexity.COMPLEX: {"lines": 100, "files": 3, "functions": 10},
                TaskComplexity.EXPERT: {"lines": 200, "files": 5, "functions": 15}
            },
            TaskType.TESTING: {
                TaskComplexity.SIMPLE: {"lines": 30, "files": 2, "functions": 3},
                TaskComplexity.MODERATE: {"lines": 80, "files": 3, "functions": 8},
                TaskComplexity.COMPLEX: {"lines": 150, "files": 5, "functions": 15},
                TaskComplexity.EXPERT: {"lines": 250, "files": 8, "functions": 25}
            }
        }
        
        # Default baseline for unknown task types
        default_baseline = {
            TaskComplexity.SIMPLE: {"lines": 40, "files": 2, "functions": 4},
            TaskComplexity.MODERATE: {"lines": 100, "files": 4, "functions": 10},
            TaskComplexity.COMPLEX: {"lines": 200, "files": 6, "functions": 20},
            TaskComplexity.EXPERT: {"lines": 350, "files": 10, "functions": 30}
        }
        
        return baselines.get(task_type, default_baseline)[complexity]
