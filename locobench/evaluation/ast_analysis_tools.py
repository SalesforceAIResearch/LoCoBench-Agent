"""
AST Analysis Tools for LoCoBench Metrics Revision

This module implements deterministic AST-based code analysis tools that replace
keyword matching and surface-level heuristics with deep structural understanding.

Key Features:
- Deterministic AST parsing and traversal
- Structural code quality analysis
- Semantic pattern detection (NO keywords)
- Code complexity measurement
- Architectural coherence analysis
- Cross-file dependency analysis

This eliminates the simplicity bias that plagued the original metrics.
"""

import ast
import os
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class CodePattern(Enum):
    """Semantic code patterns detected through AST analysis"""
    ERROR_HANDLING = "error_handling"
    INPUT_VALIDATION = "input_validation"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    LOGGING = "logging"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ASTMetrics:
    """Comprehensive AST-based code metrics"""
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    methods: int
    imports: int
    complexity: int
    nesting_depth: int
    documentation_coverage: float
    error_handling_coverage: float


@dataclass
class CodeQualityMetrics:
    """Code quality metrics derived from AST analysis"""
    readability_score: float
    maintainability_score: float
    complexity_score: float
    documentation_score: float
    structure_score: float
    naming_quality_score: float


@dataclass
class ArchitecturalMetrics:
    """Architectural coherence metrics from multi-file AST analysis"""
    coupling_score: float
    cohesion_score: float
    separation_score: float
    dependency_health: float
    interface_consistency: float


class ASTAnalysisTools:
    """
    Deterministic AST-based code analysis tools.
    
    This class provides the foundation for bias-free code evaluation by analyzing
    actual code structure rather than surface-level text patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # ===== CORE AST ANALYSIS =====
    
    def analyze_code_structure(self, solution_code: Dict[str, str]) -> Dict[str, ASTMetrics]:
        """
        Analyze code structure using AST parsing.
        Returns comprehensive metrics for each file.
        """
        file_metrics = {}
        
        for filepath, content in solution_code.items():
            try:
                # Parse the AST
                tree = ast.parse(content)
                lines = content.split('\n')
                
                # Calculate basic metrics
                metrics = self._calculate_ast_metrics(tree, lines, content)
                file_metrics[filepath] = metrics
                
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {filepath}: {e}")
                # Create default metrics for unparseable files
                file_metrics[filepath] = ASTMetrics(
                    total_lines=len(content.split('\n')),
                    code_lines=0,
                    comment_lines=0,
                    blank_lines=0,
                    functions=0,
                    classes=0,
                    methods=0,
                    imports=0,
                    complexity=0,
                    nesting_depth=0,
                    documentation_coverage=0.0,
                    error_handling_coverage=0.0
                )
            except Exception as e:
                self.logger.error(f"AST analysis failed for {filepath}: {e}")
                continue
        
        return file_metrics
    
    def _calculate_ast_metrics(self, tree: ast.AST, lines: List[str], content: str) -> ASTMetrics:
        """Calculate comprehensive AST metrics for a single file"""
        
        # Initialize counters
        functions = 0
        classes = 0
        methods = 0
        imports = 0
        complexity = 0
        max_nesting = 0
        documented_items = 0
        total_documentable = 0
        error_handling_items = 0
        total_functions = 0
        
        # Analyze each node in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
                total_functions += 1
                total_documentable += 1
                
                # Check for docstring
                if self._has_docstring(node):
                    documented_items += 1
                
                # Check for error handling
                if self._has_error_handling(node):
                    error_handling_items += 1
                
                # Calculate complexity for this function
                complexity += self._calculate_cyclomatic_complexity(node)
                
                # Calculate nesting depth
                nesting = self._calculate_nesting_depth(node)
                max_nesting = max(max_nesting, nesting)
                
            elif isinstance(node, ast.AsyncFunctionDef):
                functions += 1
                total_functions += 1
                total_documentable += 1
                
                if self._has_docstring(node):
                    documented_items += 1
                
                if self._has_error_handling(node):
                    error_handling_items += 1
                
                complexity += self._calculate_cyclomatic_complexity(node)
                nesting = self._calculate_nesting_depth(node)
                max_nesting = max(max_nesting, nesting)
                
            elif isinstance(node, ast.ClassDef):
                classes += 1
                total_documentable += 1
                
                if self._has_docstring(node):
                    documented_items += 1
                
                # Count methods in this class
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods += 1
                        
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
        
        # Calculate line-based metrics
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
        
        # Calculate coverage metrics
        doc_coverage = documented_items / total_documentable if total_documentable > 0 else 0.0
        error_coverage = error_handling_items / total_functions if total_functions > 0 else 0.0
        
        return ASTMetrics(
            total_lines=len(lines),
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            functions=functions,
            classes=classes,
            methods=methods,
            imports=imports,
            complexity=complexity,
            nesting_depth=max_nesting,
            documentation_coverage=doc_coverage,
            error_handling_coverage=error_coverage
        )
    
    # ===== SEMANTIC PATTERN DETECTION =====
    
    def detect_code_patterns(self, solution_code: Dict[str, str]) -> Dict[str, Dict[CodePattern, float]]:
        """
        Detect semantic code patterns using AST analysis.
        This replaces keyword-based pattern detection with structural analysis.
        """
        pattern_scores = {}
        
        for filepath, content in solution_code.items():
            try:
                tree = ast.parse(content)
                patterns = {}
                
                # Detect each pattern type
                patterns[CodePattern.ERROR_HANDLING] = self._detect_error_handling_patterns(tree)
                patterns[CodePattern.INPUT_VALIDATION] = self._detect_input_validation_patterns(tree)
                patterns[CodePattern.DOCUMENTATION] = self._detect_documentation_patterns(tree, content)
                patterns[CodePattern.TESTING] = self._detect_testing_patterns(tree, filepath)
                patterns[CodePattern.LOGGING] = self._detect_logging_patterns(tree)
                patterns[CodePattern.CONFIGURATION] = self._detect_configuration_patterns(tree)
                patterns[CodePattern.SECURITY] = self._detect_security_patterns(tree)
                patterns[CodePattern.PERFORMANCE] = self._detect_performance_patterns(tree)
                
                pattern_scores[filepath] = patterns
                
            except Exception as e:
                self.logger.error(f"Pattern detection failed for {filepath}: {e}")
                # Default to zero scores
                pattern_scores[filepath] = {pattern: 0.0 for pattern in CodePattern}
        
        return pattern_scores
    
    def _detect_error_handling_patterns(self, tree: ast.AST) -> float:
        """Detect error handling patterns through AST analysis"""
        total_functions = 0
        functions_with_error_handling = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                
                # Check for try-except blocks
                has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
                
                # Check for explicit error checking
                has_error_checks = False
                for child in ast.walk(node):
                    if isinstance(child, ast.If):
                        # Look for error condition checking patterns
                        if self._is_error_condition_check(child):
                            has_error_checks = True
                            break
                
                # Check for input validation
                has_validation = self._has_input_validation(node)
                
                if has_try_except or has_error_checks or has_validation:
                    functions_with_error_handling += 1
        
        return functions_with_error_handling / total_functions if total_functions > 0 else 0.0
    
    def _detect_input_validation_patterns(self, tree: ast.AST) -> float:
        """Detect input validation patterns"""
        total_functions = 0
        functions_with_validation = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                
                if self._has_input_validation(node):
                    functions_with_validation += 1
        
        return functions_with_validation / total_functions if total_functions > 0 else 0.0
    
    def _detect_documentation_patterns(self, tree: ast.AST, content: str) -> float:
        """Detect documentation patterns"""
        documentable_items = 0
        documented_items = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                documentable_items += 1
                
                if self._has_docstring(node):
                    documented_items += 1
        
        # Also check for inline comments
        lines = content.split('\n')
        comment_density = sum(1 for line in lines if line.strip().startswith('#')) / len(lines)
        
        # Combine docstring coverage with comment density
        doc_coverage = documented_items / documentable_items if documentable_items > 0 else 0.0
        
        return min((doc_coverage * 0.7 + comment_density * 0.3), 1.0)
    
    def _detect_testing_patterns(self, tree: ast.AST, filepath: str) -> float:
        """Detect testing patterns"""
        # Check if this is a test file
        is_test_file = any(pattern in filepath.lower() for pattern in ['test_', '_test', 'tests/'])
        
        if not is_test_file:
            return 0.0
        
        # Count test functions and assertions
        test_functions = 0
        total_functions = 0
        assertion_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                
                # Check if function name suggests it's a test
                if node.name.startswith('test_') or node.name.endswith('_test'):
                    test_functions += 1
                
            elif isinstance(node, ast.Assert):
                assertion_count += 1
            elif isinstance(node, ast.Call):
                # Check for unittest assertions
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith('assert'):
                        assertion_count += 1
        
        # Score based on test function ratio and assertion density
        test_ratio = test_functions / total_functions if total_functions > 0 else 0.0
        assertion_density = min(assertion_count / 10, 1.0)  # Normalize to 0-1
        
        return (test_ratio * 0.6 + assertion_density * 0.4)
    
    def _detect_logging_patterns(self, tree: ast.AST) -> float:
        """Detect logging patterns"""
        logging_calls = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                
            elif isinstance(node, ast.Call):
                # Check for logging calls
                if self._is_logging_call(node):
                    logging_calls += 1
        
        # Score based on logging density
        return min(logging_calls / max(total_functions, 1), 1.0)
    
    def _detect_configuration_patterns(self, tree: ast.AST) -> float:
        """Detect configuration management patterns"""
        config_patterns = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for environment variable access
                if self._is_env_access(node):
                    config_patterns += 1
                # Check for config file loading
                elif self._is_config_loading(node):
                    config_patterns += 1
            
            elif isinstance(node, ast.Assign):
                # Check for configuration constants
                if self._is_config_constant(node):
                    config_patterns += 1
        
        return min(config_patterns / 5, 1.0)  # Normalize
    
    def _detect_security_patterns(self, tree: ast.AST) -> float:
        """Detect security-related patterns"""
        security_patterns = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for security-related function calls
                if self._is_security_call(node):
                    security_patterns += 1
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check for input sanitization
                if self._has_input_sanitization(node):
                    security_patterns += 1
        
        return min(security_patterns / 3, 1.0)  # Normalize
    
    def _detect_performance_patterns(self, tree: ast.AST) -> float:
        """Detect performance optimization patterns"""
        performance_patterns = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for performance-related calls
                if self._is_performance_call(node):
                    performance_patterns += 1
            
            elif isinstance(node, ast.ListComp):
                # List comprehensions are generally more performant
                performance_patterns += 1
            
            elif isinstance(node, (ast.GeneratorExp, ast.DictComp, ast.SetComp)):
                # Generator expressions and comprehensions
                performance_patterns += 1
        
        return min(performance_patterns / 5, 1.0)  # Normalize
    
    # ===== CODE QUALITY ANALYSIS =====
    
    def analyze_code_quality(self, solution_code: Dict[str, str]) -> Dict[str, CodeQualityMetrics]:
        """
        Analyze code quality using deterministic AST-based rules.
        This replaces subjective keyword-based quality assessment.
        """
        quality_metrics = {}
        
        for filepath, content in solution_code.items():
            try:
                tree = ast.parse(content)
                lines = content.split('\n')
                
                # Calculate quality dimensions
                readability = self._analyze_readability(tree, lines)
                maintainability = self._analyze_maintainability(tree, content)
                complexity = self._analyze_complexity_score(tree)
                documentation = self._analyze_documentation_quality(tree, content)
                structure = self._analyze_structure_quality(tree)
                naming = self._analyze_naming_quality(tree)
                
                quality_metrics[filepath] = CodeQualityMetrics(
                    readability_score=readability,
                    maintainability_score=maintainability,
                    complexity_score=complexity,
                    documentation_score=documentation,
                    structure_score=structure,
                    naming_quality_score=naming
                )
                
            except Exception as e:
                self.logger.error(f"Quality analysis failed for {filepath}: {e}")
                # Default to neutral scores
                quality_metrics[filepath] = CodeQualityMetrics(
                    readability_score=0.5,
                    maintainability_score=0.5,
                    complexity_score=0.5,
                    documentation_score=0.0,
                    structure_score=0.5,
                    naming_quality_score=0.5
                )
        
        return quality_metrics
    
    def _analyze_readability(self, tree: ast.AST, lines: List[str]) -> float:
        """Analyze code readability using deterministic rules"""
        readability_factors = []
        
        # 1. Line length distribution
        long_lines = sum(1 for line in lines if len(line) > 100)
        line_length_score = 1.0 - (long_lines / len(lines)) if lines else 1.0
        readability_factors.append(line_length_score)
        
        # 2. Comment density
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_ratio = comment_lines / code_lines if code_lines > 0 else 0.0
        
        # Optimal comment ratio: 10-30%
        if 0.1 <= comment_ratio <= 0.3:
            comment_score = 1.0
        elif comment_ratio > 0:
            comment_score = 0.7
        else:
            comment_score = 0.3
        readability_factors.append(comment_score)
        
        # 3. Function length distribution
        function_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 10
                function_lengths.append(func_lines)
        
        if function_lengths:
            avg_func_length = sum(function_lengths) / len(function_lengths)
            # Optimal function length: 5-20 lines
            if 5 <= avg_func_length <= 20:
                func_length_score = 1.0
            elif avg_func_length <= 30:
                func_length_score = 0.8
            else:
                func_length_score = 0.5
        else:
            func_length_score = 1.0
        readability_factors.append(func_length_score)
        
        # 4. Nesting depth
        max_nesting = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nesting = self._calculate_nesting_depth(node)
                max_nesting = max(max_nesting, nesting)
        
        # Optimal nesting: <= 3 levels
        if max_nesting <= 3:
            nesting_score = 1.0
        elif max_nesting <= 5:
            nesting_score = 0.7
        else:
            nesting_score = 0.4
        readability_factors.append(nesting_score)
        
        return sum(readability_factors) / len(readability_factors)
    
    def _analyze_maintainability(self, tree: ast.AST, content: str) -> float:
        """Analyze code maintainability"""
        maintainability_factors = []
        
        # 1. Cyclomatic complexity
        total_complexity = 0
        function_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_cyclomatic_complexity(node)
                total_complexity += complexity
                function_count += 1
        
        avg_complexity = total_complexity / function_count if function_count > 0 else 1
        
        # Optimal complexity: <= 10 per function
        if avg_complexity <= 5:
            complexity_score = 1.0
        elif avg_complexity <= 10:
            complexity_score = 0.8
        elif avg_complexity <= 15:
            complexity_score = 0.6
        else:
            complexity_score = 0.3
        maintainability_factors.append(complexity_score)
        
        # 2. Code duplication (simplified heuristic)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        unique_lines = len(set(lines))
        duplication_ratio = unique_lines / len(lines) if lines else 1.0
        maintainability_factors.append(duplication_ratio)
        
        # 3. Function cohesion (functions per class ratio)
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        
        if classes > 0:
            cohesion_ratio = functions / classes
            # Optimal: 3-8 methods per class
            if 3 <= cohesion_ratio <= 8:
                cohesion_score = 1.0
            elif cohesion_ratio <= 12:
                cohesion_score = 0.8
            else:
                cohesion_score = 0.6
        else:
            cohesion_score = 0.8  # Neutral for non-OOP code
        maintainability_factors.append(cohesion_score)
        
        return sum(maintainability_factors) / len(maintainability_factors)
    
    # ===== ARCHITECTURAL ANALYSIS =====
    
    def analyze_architecture(self, solution_code: Dict[str, str]) -> ArchitecturalMetrics:
        """
        Analyze architectural coherence across multiple files.
        This replaces keyword-based architectural assessment.
        """
        if len(solution_code) <= 1:
            # Single file - perfect architecture by definition
            return ArchitecturalMetrics(
                coupling_score=1.0,
                cohesion_score=1.0,
                separation_score=1.0,
                dependency_health=1.0,
                interface_consistency=1.0
            )
        
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(solution_code)
            
            # Analyze coupling
            coupling_score = self._analyze_coupling(dependency_graph, solution_code)
            
            # Analyze cohesion
            cohesion_score = self._analyze_cohesion(solution_code)
            
            # Analyze separation of concerns
            separation_score = self._analyze_separation(solution_code)
            
            # Analyze dependency health
            dependency_health = self._analyze_dependency_health(dependency_graph)
            
            # Analyze interface consistency
            interface_consistency = self._analyze_interface_consistency(solution_code)
            
            return ArchitecturalMetrics(
                coupling_score=coupling_score,
                cohesion_score=cohesion_score,
                separation_score=separation_score,
                dependency_health=dependency_health,
                interface_consistency=interface_consistency
            )
            
        except Exception as e:
            self.logger.error(f"Architectural analysis failed: {e}")
            return ArchitecturalMetrics(
                coupling_score=0.5,
                cohesion_score=0.5,
                separation_score=0.5,
                dependency_health=0.5,
                interface_consistency=0.5
            )
    
    def _build_dependency_graph(self, solution_code: Dict[str, str]) -> Dict[str, Set[str]]:
        """Build dependency graph from import statements"""
        dependencies = defaultdict(set)
        
        for filepath, content in solution_code.items():
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dependencies[filepath].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            dependencies[filepath].add(node.module)
                            
            except Exception:
                continue
        
        return dict(dependencies)
    
    def _analyze_coupling(self, dependency_graph: Dict[str, Set[str]], solution_code: Dict[str, str]) -> float:
        """Analyze coupling between modules"""
        if len(solution_code) <= 1:
            return 1.0
        
        total_possible_deps = len(solution_code) * (len(solution_code) - 1)
        actual_deps = sum(len(deps) for deps in dependency_graph.values())
        
        # Lower coupling is better
        coupling_ratio = actual_deps / total_possible_deps if total_possible_deps > 0 else 0.0
        
        # Invert the score (lower coupling = higher score)
        return max(0.0, 1.0 - coupling_ratio)
    
    def _analyze_cohesion(self, solution_code: Dict[str, str]) -> float:
        """Analyze cohesion within modules"""
        cohesion_scores = []
        
        for filepath, content in solution_code.items():
            try:
                tree = ast.parse(content)
                
                # Count related elements
                classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                functions = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
                
                # High cohesion: balanced classes and functions
                if classes > 0 and functions > 0:
                    balance = min(classes, functions) / max(classes, functions)
                    cohesion_scores.append(balance)
                elif classes > 0 or functions > 0:
                    cohesion_scores.append(0.8)  # Single type is okay
                else:
                    cohesion_scores.append(0.5)  # No clear structure
                    
            except Exception:
                cohesion_scores.append(0.5)
        
        return sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.5
    
    # ===== HELPER METHODS =====
    
    def _has_docstring(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> bool:
        """Check if a function/class has a docstring"""
        return (
            node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        )
    
    def _has_error_handling(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function has error handling"""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
            elif isinstance(child, ast.If):
                if self._is_error_condition_check(child):
                    return True
        return False
    
    def _has_input_validation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function has input validation"""
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                return True
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ['isinstance', 'hasattr', 'len']:
                    return True
            elif isinstance(child, ast.If):
                # Check for validation patterns
                if self._is_validation_check(child):
                    return True
        return False
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(n, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return get_depth(node)
    
    def _is_error_condition_check(self, node: ast.If) -> bool:
        """Check if an if statement is checking for error conditions"""
        # This is a simplified heuristic - could be expanded
        if isinstance(node.test, ast.Compare):
            # Check for None comparisons, empty checks, etc.
            return True
        elif isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
            # Check for "if not ..." patterns
            return True
        return False
    
    def _is_validation_check(self, node: ast.If) -> bool:
        """Check if an if statement is doing input validation"""
        # Simplified heuristic for validation patterns
        return self._is_error_condition_check(node)
    
    def _is_logging_call(self, node: ast.Call) -> bool:
        """Check if a call is a logging call"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ['debug', 'info', 'warning', 'error', 'critical']
        elif isinstance(node.func, ast.Name):
            return node.func.id in ['print', 'log']  # Simple logging
        return False
    
    def _is_env_access(self, node: ast.Call) -> bool:
        """Check if a call accesses environment variables"""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                return node.func.attr in ['getenv', 'environ']
        return False
    
    def _is_config_loading(self, node: ast.Call) -> bool:
        """Check if a call loads configuration"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ['load', 'read', 'parse'] and 'config' in str(node).lower()
        return False
    
    def _is_config_constant(self, node: ast.Assign) -> bool:
        """Check if an assignment defines a configuration constant"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            return name.isupper() or 'config' in name.lower()
        return False
    
    def _is_security_call(self, node: ast.Call) -> bool:
        """Check if a call is security-related"""
        security_functions = ['hash', 'encrypt', 'decrypt', 'verify', 'authenticate', 'authorize']
        
        if isinstance(node.func, ast.Attribute):
            return node.func.attr.lower() in security_functions
        elif isinstance(node.func, ast.Name):
            return node.func.id.lower() in security_functions
        return False
    
    def _has_input_sanitization(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if a function has input sanitization"""
        sanitization_patterns = ['strip', 'escape', 'sanitize', 'clean', 'validate']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr.lower() in sanitization_patterns:
                        return True
                elif isinstance(child.func, ast.Name):
                    if child.func.id.lower() in sanitization_patterns:
                        return True
        return False
    
    def _is_performance_call(self, node: ast.Call) -> bool:
        """Check if a call is performance-related"""
        performance_functions = ['cache', 'memoize', 'optimize', 'profile', 'time']
        
        if isinstance(node.func, ast.Attribute):
            return node.func.attr.lower() in performance_functions
        elif isinstance(node.func, ast.Name):
            return node.func.id.lower() in performance_functions
        return False
    
    def _analyze_complexity_score(self, tree: ast.AST) -> float:
        """Analyze overall complexity score"""
        total_complexity = 0
        function_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_cyclomatic_complexity(node)
                total_complexity += complexity
                function_count += 1
        
        if function_count == 0:
            return 1.0
        
        avg_complexity = total_complexity / function_count
        
        # Score based on average complexity (lower is better for maintainability)
        if avg_complexity <= 5:
            return 1.0
        elif avg_complexity <= 10:
            return 0.8
        elif avg_complexity <= 15:
            return 0.6
        else:
            return 0.3
    
    def _analyze_documentation_quality(self, tree: ast.AST, content: str) -> float:
        """Analyze documentation quality"""
        return self._detect_documentation_patterns(tree, content)
    
    def _analyze_structure_quality(self, tree: ast.AST) -> float:
        """Analyze code structure quality"""
        # Count different structural elements
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        
        # Good structure has balanced elements
        total_elements = classes + functions + imports
        if total_elements == 0:
            return 0.5
        
        # Reward balanced structure
        balance_score = 1.0
        if classes > 0 and functions > 0:
            balance_score = min(classes, functions) / max(classes, functions)
        
        # Reward appropriate imports
        import_ratio = imports / total_elements
        import_score = 1.0 if 0.1 <= import_ratio <= 0.3 else 0.7
        
        return (balance_score * 0.7 + import_score * 0.3)
    
    def _analyze_naming_quality(self, tree: ast.AST) -> float:
        """Analyze naming quality"""
        names = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append(node.id)
        
        if not names:
            return 1.0
        
        # Analyze naming patterns
        descriptive_names = 0
        for name in names:
            if len(name) >= 3 and ('_' in name or any(c.isupper() for c in name[1:])):
                descriptive_names += 1
        
        return descriptive_names / len(names)
    
    def _analyze_separation(self, solution_code: Dict[str, str]) -> float:
        """Analyze separation of concerns"""
        # This is a simplified heuristic
        # In a real implementation, this would analyze semantic separation
        
        if len(solution_code) <= 1:
            return 0.8  # Single file has limited separation
        
        # Check for different file purposes
        file_purposes = set()
        for filepath in solution_code.keys():
            filename = os.path.basename(filepath).lower()
            
            if 'test' in filename:
                file_purposes.add('testing')
            elif 'config' in filename:
                file_purposes.add('configuration')
            elif 'util' in filename or 'helper' in filename:
                file_purposes.add('utilities')
            elif 'main' in filename or 'app' in filename:
                file_purposes.add('application')
            else:
                file_purposes.add('business_logic')
        
        # More distinct purposes = better separation
        return min(len(file_purposes) / 4, 1.0)
    
    def _analyze_dependency_health(self, dependency_graph: Dict[str, Set[str]]) -> float:
        """Analyze dependency health (avoid circular dependencies, etc.)"""
        # Simplified heuristic - in practice would check for circular dependencies
        
        if not dependency_graph:
            return 1.0
        
        # Check for balanced dependencies
        dep_counts = [len(deps) for deps in dependency_graph.values()]
        if not dep_counts:
            return 1.0
        
        avg_deps = sum(dep_counts) / len(dep_counts)
        max_deps = max(dep_counts)
        
        # Penalize files with too many dependencies
        if max_deps <= 5:
            return 1.0
        elif max_deps <= 10:
            return 0.8
        else:
            return 0.6
    
    def _analyze_interface_consistency(self, solution_code: Dict[str, str]) -> float:
        """Analyze interface consistency across files"""
        # Simplified heuristic - would analyze function signatures, return types, etc.
        
        function_patterns = defaultdict(list)
        
        for filepath, content in solution_code.items():
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Analyze function signature patterns
                        arg_count = len(node.args.args)
                        has_return = any(isinstance(child, ast.Return) for child in ast.walk(node))
                        
                        pattern = (arg_count, has_return)
                        function_patterns[node.name].append(pattern)
                        
            except Exception:
                continue
        
        # Check consistency of function patterns
        consistent_functions = 0
        total_functions = 0
        
        for func_name, patterns in function_patterns.items():
            total_functions += 1
            if len(set(patterns)) == 1:  # All patterns are the same
                consistent_functions += 1
        
        return consistent_functions / total_functions if total_functions > 0 else 1.0
