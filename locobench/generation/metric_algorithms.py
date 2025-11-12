"""
Algorithmic implementations of the 17 comprehensive evaluation metrics for LoCoBench

These metrics evaluate software development capabilities across 4 dimensions:
- Software Engineering Excellence (8 metrics)
- Functional Correctness (4 metrics) 
- Code Quality Assessment (3 metrics)
- Long-Context Utilization (2 metrics)
"""

import ast
import re
import os
import subprocess
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
import difflib
import logging

logger = logging.getLogger(__name__)


class LoCoBenchMetricsCalculator:
    """Calculates the 17 comprehensive evaluation metrics across 4 dimensions using code analysis"""
    
    def _sanitize_solution_code(self, solution_code: Any) -> Dict[str, str]:
        """
        Sanitize solution_code to ensure it's a proper Dict[str, str].
        This fixes the systematic issue where nested dictionaries/lists cause type errors.
        """
        if not isinstance(solution_code, dict):
            logger.warning(f"solution_code is not a dict, got {type(solution_code)}. Using empty dict.")
            return {}
        
        sanitized = {}
        for filename, code in solution_code.items():
            if isinstance(code, str):
                sanitized[filename] = code
            elif isinstance(code, dict):
                # Handle nested dictionaries by converting to JSON string
                logger.warning(f"File {filename} contains dict instead of string. Converting to JSON.")
                sanitized[filename] = json.dumps(code, indent=2)
            elif isinstance(code, list):
                # Handle lists by converting to JSON string
                logger.warning(f"File {filename} contains list instead of string. Converting to JSON.")
                sanitized[filename] = json.dumps(code, indent=2)
            else:
                # Convert other types to string
                logger.warning(f"File {filename} contains {type(code)} instead of string. Converting to string.")
                sanitized[filename] = str(code)
        
        return sanitized
    
    def __init__(self):
        self.architectural_patterns = {
            'mvc': ['model', 'view', 'controller', 'models', 'views', 'controllers'],
            'repository': ['repository', 'repo', 'dao', 'data_access'],
            'factory': ['factory', 'builder', 'create'],
            'observer': ['observer', 'listener', 'subscriber', 'event'],
            'strategy': ['strategy', 'algorithm', 'policy'],
            'adapter': ['adapter', 'wrapper', 'bridge']
        }

    def _get_task_prompt_text(self, scenario: Dict[str, Any]) -> str:
        """
        Safely extract task prompt text from scenario, handling both string and dict formats.
        Multi-session scenarios have task_prompt as dict with session_1, session_2, etc.
        """
        task_prompt = scenario.get('task_prompt', '')
        
        # Handle multi-session development scenarios
        if isinstance(task_prompt, dict):
            # Combine all session prompts for analysis
            session_texts = []
            for key in sorted(task_prompt.keys()):
                if isinstance(task_prompt[key], str):
                    session_texts.append(task_prompt[key])
            return ' '.join(session_texts)
        
        # Handle regular scenarios where task_prompt is a string
        return str(task_prompt) if task_prompt else ''

    def calculate_architectural_coherence_score(self, scenario: Dict[str, Any], 
                                             solution_code: Dict[str, str]) -> float:
        """
        ACS: Architectural Coherence Score
        Measures consistency with existing architectural patterns and design principles
        """
        
        # Sanitize solution_code to prevent type errors for multi-session scenarios
        solution_code = self._sanitize_solution_code(solution_code)
        
        context_files = scenario.get('context_files', [])
        task_category = scenario.get('task_category', '')
        
        # 1. Pattern consistency analysis (40%)
        pattern_score = self._analyze_pattern_consistency(solution_code, context_files)
        
        # 2. File organization coherence (30%)
        organization_score = self._analyze_file_organization(solution_code)
        
        # 3. Naming convention consistency (20%)
        naming_score = self._analyze_naming_consistency(solution_code)
        
        # 4. Import/dependency structure (10%)
        dependency_score = self._analyze_dependency_structure(solution_code)
        
        acs_score = (
            pattern_score * 0.4 +
            organization_score * 0.3 +
            naming_score * 0.2 +
            dependency_score * 0.1
        )
        
        return min(max(acs_score, 0.0), 1.0)

    def calculate_dependency_traversal_accuracy(self, scenario: Dict[str, Any], 
                                              solution_code: Dict[str, str]) -> float:
        """
        DTA: Dependency Traversal Accuracy
        Measures how accurately the model navigates complex dependency relationships
        """
        
        # 1. Import resolution accuracy (40%)
        import_score = self._analyze_import_accuracy(solution_code)
        
        # 2. Cross-file reference validity (35%)
        reference_score = self._analyze_cross_file_references(solution_code)
        
        # 3. Dependency order correctness (25%)
        order_score = self._analyze_dependency_order(solution_code)
        
        dta_score = (
            import_score * 0.4 +
            reference_score * 0.35 +
            order_score * 0.25
        )
        
        return min(max(dta_score, 0.0), 1.0)

    def calculate_multi_session_memory_retention(self, scenario: Dict[str, Any], 
                                               solution_code: Dict[str, str]) -> float:
        """
        MMR: Multi-Session Memory Retention
        Measures context persistence and consistency across development sessions
        """
        
        # Sanitize solution_code to prevent type errors for multi-session scenarios
        solution_code = self._sanitize_solution_code(solution_code)
        
        task_category = scenario.get('task_category', '')
        
        # Only applicable for multi-session tasks
        if task_category != 'multi_session_development':
            return self._calculate_context_consistency(scenario, solution_code)
        
        # 1. Variable/function name consistency (40%)
        naming_consistency = self._analyze_naming_consistency_across_sessions(solution_code)
        
        # 2. Approach consistency (35%)
        approach_consistency = self._analyze_approach_consistency(solution_code)
        
        # 3. State management continuity (25%)
        state_consistency = self._analyze_state_management(solution_code)
        
        mmr_score = (
            naming_consistency * 0.4 +
            approach_consistency * 0.35 +
            state_consistency * 0.25
        )
        
        return min(max(mmr_score, 0.0), 1.0)

    def calculate_cross_file_reasoning_depth(self, scenario: Dict[str, Any], 
                                           solution_code: Dict[str, str]) -> float:
        """
        CFRD: Cross-File Reasoning Depth
        Measures understanding of multi-file relationships and coordination
        """
        
        # 1. Interface usage correctness (35%)
        interface_score = self._analyze_interface_usage(solution_code)
        
        # 2. Shared state coordination (30%)
        shared_state_score = self._analyze_shared_state_coordination(solution_code)
        
        # 3. Cross-file modification coordination (25%)
        modification_score = self._analyze_modification_coordination(solution_code, scenario)
        
        # 4. Data flow understanding (10%)
        dataflow_score = self._analyze_data_flow_understanding(solution_code)
        
        cfrd_score = (
            interface_score * 0.35 +
            shared_state_score * 0.30 +
            modification_score * 0.25 +
            dataflow_score * 0.10
        )
        
        return min(max(cfrd_score, 0.0), 1.0)

    def calculate_incremental_development_capability(self, scenario: Dict[str, Any], 
                                                   solution_code: Dict[str, str]) -> float:
        """
        IDC: Incremental Development Capability
        Measures ability to build incrementally on existing work
        """
        
        # 1. Backward compatibility preservation (40%)
        compatibility_score = self._analyze_backward_compatibility(solution_code, scenario)
        
        # 2. Code reuse efficiency (30%)
        reuse_score = self._analyze_code_reuse(solution_code, scenario)
        
        # 3. Extension pattern usage (20%)
        extension_score = self._analyze_extension_patterns(solution_code)
        
        # 4. Minimal disruption principle (10%)
        disruption_score = self._analyze_minimal_disruption(solution_code, scenario)
        
        idc_score = (
            compatibility_score * 0.4 +
            reuse_score * 0.3 +
            extension_score * 0.2 +
            disruption_score * 0.1
        )
        
        return min(max(idc_score, 0.0), 1.0)

    def calculate_information_coverage_utilization(self, scenario: Dict[str, Any], 
                                                 solution_code: Dict[str, str]) -> float:
        """
        ICU: Information Coverage Utilization
        Measures how effectively the model uses available context information
        """
        
        context_files = scenario.get('context_files', [])
        task_prompt = self._get_task_prompt_text(scenario)
        
        # 1. Context file usage efficiency (40%)
        context_usage_score = self._analyze_context_usage(solution_code, context_files)
        
        # 2. Requirement coverage completeness (35%)
        requirement_score = self._analyze_requirement_coverage(solution_code, task_prompt)
        
        # 3. Information extraction accuracy (25%)
        extraction_score = self._analyze_information_extraction(solution_code, scenario)
        
        icu_score = (
            context_usage_score * 0.4 +
            requirement_score * 0.35 +
            extraction_score * 0.25
        )
        
        return min(max(icu_score, 0.0), 1.0)

    # Helper methods for detailed analysis

    def _analyze_pattern_consistency(self, solution_code: Dict[str, str], context_files: List[str]) -> float:
        """Analyze consistency with architectural patterns"""
        
        # solution_code is already sanitized to ensure all values are strings
        
        pattern_scores = []
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Check for architectural pattern indicators
            for pattern, keywords in self.architectural_patterns.items():
                pattern_matches = sum(1 for keyword in keywords if keyword in code_lower)
                if pattern_matches > 0:
                    # Bonus for consistent pattern usage
                    file_score += min(pattern_matches / len(keywords), 0.3)
            
            # Check for proper separation of concerns
            if self._has_clear_separation_of_concerns(code):
                file_score += 0.3
                
            # Check for consistent function/class organization
            if self._has_consistent_organization(code):
                file_score += 0.2
                
            pattern_scores.append(min(file_score, 1.0))
        
        return sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0.0

    def _analyze_file_organization(self, solution_code: Dict[str, str]) -> float:
        """Analyze logical file organization"""
        
        organization_scores = []
        
        for filename, code in solution_code.items():
            score = 0.0
            
            # Check import organization (imports at top)
            if self._has_proper_import_organization(code):
                score += 0.3
                
            # Check function/class ordering
            if self._has_logical_ordering(code):
                score += 0.3
                
            # Check for proper spacing and structure
            if self._has_consistent_spacing(code):
                score += 0.2
                
            # Check for meaningful file structure
            if self._has_meaningful_structure(code):
                score += 0.2
                
            organization_scores.append(score)
        
        return sum(organization_scores) / len(organization_scores) if organization_scores else 0.0

    def _analyze_naming_consistency(self, solution_code: Dict[str, str]) -> float:
        """Analyze naming convention consistency"""
        
        # Sanitize solution_code to prevent type errors
        solution_code = self._sanitize_solution_code(solution_code)
        
        naming_patterns = defaultdict(int)
        total_names = 0
        
        for filename, code in solution_code.items():
            # Extract function and variable names using regex
            function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            variable_names = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)
            
            all_names = function_names + variable_names
            total_names += len(all_names)
            
            for name in all_names:
                # Determine naming pattern
                if '_' in name and name.islower():
                    naming_patterns['snake_case'] += 1
                elif name[0].islower() and any(c.isupper() for c in name[1:]):
                    naming_patterns['camelCase'] += 1
                elif name[0].isupper():
                    naming_patterns['PascalCase'] += 1
                else:
                    naming_patterns['other'] += 1
        
        if total_names == 0:
            return 0.5
            
        # Calculate consistency score based on dominant pattern
        max_pattern_count = max(naming_patterns.values()) if naming_patterns else 0
        consistency_score = max_pattern_count / total_names
        
        return min(consistency_score * 1.2, 1.0)  # Slight bonus for high consistency

    def _analyze_dependency_structure(self, solution_code: Dict[str, str]) -> float:
        """Analyze import and dependency structure"""
        
        dependency_scores = []
        
        for filename, code in solution_code.items():
            score = 0.0
            
            # Check for proper import statements
            import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
            
            if import_lines:
                # Check import organization
                if self._are_imports_organized(import_lines):
                    score += 0.4
                    
                # Check for unused imports (basic check)
                if not self._has_unused_imports(code, import_lines):
                    score += 0.3
                    
                # Check for proper relative vs absolute imports
                if self._has_proper_import_style(import_lines):
                    score += 0.3
            
            dependency_scores.append(score)
        
        return sum(dependency_scores) / len(dependency_scores) if dependency_scores else 0.0

    def _analyze_import_accuracy(self, solution_code: Dict[str, str]) -> float:
        """Analyze accuracy of import statements"""
        
        import_scores = []
        
        for filename, code in solution_code.items():
            score = 0.0
            
            import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
            
            if import_lines:
                # Check for valid import syntax
                valid_imports = sum(1 for imp in import_lines if self._is_valid_import_syntax(imp))
                score += (valid_imports / len(import_lines)) * 0.6
                
                # Check for reasonable import choices
                reasonable_imports = sum(1 for imp in import_lines if self._is_reasonable_import(imp))
                score += (reasonable_imports / len(import_lines)) * 0.4
            
            import_scores.append(score)
        
        return sum(import_scores) / len(import_scores) if import_scores else 0.0

    def _analyze_cross_file_references(self, solution_code: Dict[str, str]) -> float:
        """Analyze validity of cross-file references"""
        
        # Extract all function/class definitions
        definitions = {}
        for filename, code in solution_code.items():
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            definitions[filename] = functions + classes
        
        reference_scores = []
        
        for filename, code in solution_code.items():
            score = 0.0
            
            # Find function/method calls
            calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
            
            if calls:
                # Check how many calls reference defined functions
                valid_references = 0
                for call in calls:
                    # Check if it's defined in any file
                    if any(call in file_definitions for file_definitions in definitions.values()):
                        valid_references += 1
                    # Or if it's a built-in/standard library function
                    elif self._is_builtin_function(call):
                        valid_references += 1
                
                score = valid_references / len(calls) if calls else 0.0
            
            reference_scores.append(score)
        
        return sum(reference_scores) / len(reference_scores) if reference_scores else 0.0

    def _analyze_dependency_order(self, solution_code: Dict[str, str]) -> float:
        """Analyze logical ordering of dependencies"""
        
        order_scores = []
        
        for filename, code in solution_code.items():
            lines = code.split('\n')
            import_section_end = 0
            
            # Find end of import section
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_section_end = i
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # Check if imports are at the top
            imports_at_top = import_section_end < min(10, len(lines) // 4)
            
            order_scores.append(1.0 if imports_at_top else 0.5)
        
        return sum(order_scores) / len(order_scores) if order_scores else 0.0

    # Additional helper methods (simplified implementations)
    
    def _has_clear_separation_of_concerns(self, code: str) -> bool:
        """Check if code has clear separation of concerns"""
        # Simple heuristic: check for multiple classes or clear function grouping
        class_count = len(re.findall(r'class\s+\w+', code))
        function_count = len(re.findall(r'def\s+\w+', code))
        return class_count > 0 or function_count > 2

    def _has_consistent_organization(self, code: str) -> bool:
        """Check for consistent code organization"""
        lines = code.split('\n')
        # Simple check: consistent indentation
        indented_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        return len(indented_lines) > len(lines) * 0.3

    def _has_proper_import_organization(self, code: str) -> bool:
        """Check if imports are properly organized"""
        import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
        if not import_lines:
            return True
        
        # Check if imports are grouped (stdlib, third-party, local)
        return len(import_lines) <= 10  # Simple heuristic

    def _has_logical_ordering(self, code: str) -> bool:
        """Check for logical ordering of functions/classes"""
        # Simple heuristic: classes before functions
        class_positions = [i for i, line in enumerate(code.split('\n')) if line.strip().startswith('class ')]
        function_positions = [i for i, line in enumerate(code.split('\n')) if line.strip().startswith('def ')]
        
        if not class_positions or not function_positions:
            return True
        
        return max(class_positions) < min(function_positions)

    def _has_consistent_spacing(self, code: str) -> bool:
        """Check for consistent spacing"""
        lines = code.split('\n')
        # Simple check: not too many blank lines
        blank_lines = sum(1 for line in lines if not line.strip())
        return blank_lines < len(lines) * 0.3

    def _has_meaningful_structure(self, code: str) -> bool:
        """Check for meaningful code structure"""
        # Simple heuristic: has docstrings or comments
        has_docstrings = '"""' in code or "'''" in code
        has_comments = '#' in code
        return has_docstrings or has_comments

    def _are_imports_organized(self, import_lines: List[str]) -> bool:
        """Check if imports are well organized"""
        # Simple heuristic: standard library imports before others
        stdlib_imports = ['os', 'sys', 'json', 'time', 'datetime', 'collections', 're']
        
        stdlib_found = False
        thirdparty_found = False
        
        for line in import_lines:
            is_stdlib = any(lib in line for lib in stdlib_imports)
            if is_stdlib:
                if thirdparty_found:
                    return False  # stdlib after third-party
                stdlib_found = True
            else:
                thirdparty_found = True
        
        return True

    def _has_unused_imports(self, code: str, import_lines: List[str]) -> bool:
        """Simple check for unused imports"""
        # Extract imported names
        imported_names = []
        for line in import_lines:
            if line.startswith('import '):
                name = line.split()[1].split('.')[0]
                imported_names.append(name)
            elif line.startswith('from '):
                parts = line.split()
                if 'import' in parts:
                    idx = parts.index('import')
                    names = ' '.join(parts[idx+1:]).split(',')
                    imported_names.extend([name.strip() for name in names])
        
        # Check if names are used in code
        for name in imported_names:
            if name not in code:
                return True
        
        return False

    def _has_proper_import_style(self, import_lines: List[str]) -> bool:
        """Check for proper import style"""
        # Prefer absolute imports, avoid wildcard imports
        for line in import_lines:
            if '*' in line:
                return False
        return True

    def _is_valid_import_syntax(self, import_line: str) -> bool:
        """Check if import has valid syntax"""
        try:
            ast.parse(import_line)
            return True
        except SyntaxError:
            return False

    def _is_reasonable_import(self, import_line: str) -> bool:
        """Check if import is reasonable"""
        # Simple heuristics for reasonable imports
        suspicious_patterns = ['__', 'sys.exit', 'eval', 'exec']
        return not any(pattern in import_line for pattern in suspicious_patterns)

    def _is_builtin_function(self, func_name: str) -> bool:
        """Check if function is a built-in"""
        builtins = ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 
                   'range', 'enumerate', 'zip', 'map', 'filter', 'sum', 'max', 'min', 
                   'open', 'abs', 'round', 'sorted', 'reversed']
        return func_name in builtins

    # Real implementations replacing placeholders
    def _calculate_context_consistency(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """Calculate how well solution maintains context consistency"""
        
        # Sanitize solution_code to prevent type errors for multi-session scenarios
        solution_code = self._sanitize_solution_code(solution_code)
        
        task_prompt = self._get_task_prompt_text(scenario)
        context_files = scenario.get('context_files', [])
        
        # 1. Context file reference consistency (40%)
        context_ref_score = self._analyze_context_usage(solution_code, context_files)
        
        # 2. Task requirement alignment (35%)
        requirement_score = self._analyze_requirement_coverage(solution_code, task_prompt)
        
        # 3. Terminology consistency (25%)
        terminology_score = self._analyze_terminology_consistency(solution_code, scenario)
        
        return (
            context_ref_score * 0.4 +
            requirement_score * 0.35 +
            terminology_score * 0.25
        )

    def _analyze_naming_consistency_across_sessions(self, solution_code: Dict[str, str]) -> float:
        """Analyze naming consistency across multiple development sessions"""
        
        all_identifiers = {}
        total_inconsistencies = 0
        total_comparisons = 0
        
        # Extract all identifiers (functions, variables, types)
        for filename, code in solution_code.items():
            identifiers = self._extract_identifiers(code)
            all_identifiers[filename] = identifiers
        
        # Compare naming patterns across files
        file_names = list(all_identifiers.keys())
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                file1_ids = all_identifiers[file_names[i]]
                file2_ids = all_identifiers[file_names[j]]
                
                # Check for similar concepts with different naming
                inconsistencies = self._find_naming_inconsistencies(file1_ids, file2_ids)
                total_inconsistencies += inconsistencies
                total_comparisons += len(file1_ids) + len(file2_ids)
        
        if total_comparisons == 0:
            return 0.5
        
        consistency_ratio = 1.0 - (total_inconsistencies / total_comparisons)
        return max(consistency_ratio, 0.0)

    def _analyze_approach_consistency(self, solution_code: Dict[str, str]) -> float:
        """Analyze consistency of programming approach across files"""
        
        approach_indicators = {
            'error_handling': ['try', 'catch', 'error', 'exception', 'if err'],
            'data_structures': ['map', 'slice', 'array', 'list', 'dict'],
            'patterns': ['interface', 'struct', 'class', 'factory', 'builder'],
            'style': ['func', 'function', 'method', 'procedure']
        }
        
        file_approaches = {}
        
        # Analyze approach in each file
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_approach = {}
            
            for category, indicators in approach_indicators.items():
                usage_count = sum(1 for indicator in indicators if indicator in code_lower)
                file_approach[category] = usage_count
                
            file_approaches[filename] = file_approach
        
        # Calculate consistency across files
        if len(file_approaches) < 2:
            return 0.8  # Single file, assume consistent
        
        consistency_scores = []
        categories = list(approach_indicators.keys())
        
        for category in categories:
            category_values = [approaches.get(category, 0) for approaches in file_approaches.values()]
            category_variance = self._calculate_variance(category_values)
            category_consistency = 1.0 / (1.0 + category_variance)  # Higher variance = lower consistency
            consistency_scores.append(category_consistency)
        
        return sum(consistency_scores) / len(consistency_scores)

    def _analyze_state_management(self, solution_code: Dict[str, str]) -> float:
        """Analyze quality of state management patterns"""
        
        state_indicators = {
            'immutability': ['const', 'readonly', 'immutable', 'copy'],
            'shared_state': ['global', 'static', 'shared', 'singleton'],
            'state_isolation': ['private', 'encapsulated', 'local'],
            'state_validation': ['validate', 'check', 'verify', 'assert']
        }
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Check for good state management patterns
            for pattern, indicators in state_indicators.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    if pattern == 'immutability' or pattern == 'state_isolation':
                        file_score += 0.3  # Bonus for good patterns
                    elif pattern == 'shared_state':
                        file_score -= 0.1  # Penalty for shared state
                    else:
                        file_score += 0.1
            
            # Check for state mutation patterns
            mutation_patterns = ['=', '++', '--', '+=', '-=']
            mutation_count = sum(code.count(pattern) for pattern in mutation_patterns)
            
            # Penalize excessive mutations
            if mutation_count > 10:
                file_score -= 0.2
            
            total_score += max(file_score, 0.0)
        
        return min(total_score / len(solution_code), 1.0)

    def _analyze_interface_usage(self, solution_code: Dict[str, str]) -> float:
        """Analyze proper interface design and usage"""
        
        interface_patterns = {
            'interface_definition': ['interface', 'protocol', 'abstract'],
            'dependency_injection': ['inject', 'provide', 'wire', 'bind'],
            'abstraction': ['implement', 'extend', 'inherit', 'override'],
            'contracts': ['requires', 'ensures', 'contract', 'guarantee']
        }
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Check for interface patterns
            for pattern, indicators in interface_patterns.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    file_score += 0.25
            
            # Check for proper interface naming (ends with -er, -able, or Interface)
            interface_names = re.findall(r'interface\s+([A-Za-z]+)', code)
            proper_interface_names = sum(1 for name in interface_names 
                                       if name.endswith(('er', 'able', 'Interface')))
            
            if interface_names:
                file_score += (proper_interface_names / len(interface_names)) * 0.2
            
            total_score += min(file_score, 1.0)
        
        return total_score / len(solution_code)

    def _analyze_shared_state_coordination(self, solution_code: Dict[str, str]) -> float:
        """Analyze coordination of shared state across files"""
        
        coordination_patterns = {
            'synchronization': ['mutex', 'lock', 'sync', 'atomic', 'synchronized'],
            'messaging': ['channel', 'queue', 'event', 'message', 'signal'],
            'coordination': ['wait', 'notify', 'coordinate', 'barrier'],
            'isolation': ['goroutine', 'thread', 'process', 'worker']
        }
        
        total_coordination_score = 0.0
        shared_state_detected = False
        
        # First, detect if shared state exists
        for filename, code in solution_code.items():
            code_lower = code.lower()
            if any(pattern in code_lower for pattern in ['global', 'static', 'shared']):
                shared_state_detected = True
                break
        
        if not shared_state_detected:
            return 0.8  # No shared state, good isolation
        
        # Analyze coordination mechanisms
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            for pattern, indicators in coordination_patterns.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    file_score += 0.25
            
            total_coordination_score += min(file_score, 1.0)
        
        return total_coordination_score / len(solution_code)

    def _analyze_modification_coordination(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze coordination of modifications across multiple files"""
        
        modification_patterns = {
            'transaction': ['transaction', 'commit', 'rollback', 'begin'],
            'validation': ['validate', 'check', 'verify', 'ensure'],
            'consistency': ['consistent', 'atomic', 'ACID', 'integrity'],
            'error_recovery': ['recover', 'retry', 'fallback', 'compensate']
        }
        
        # Check if this is a modification-heavy task
        task_category = scenario.get('task_category', '')
        task_prompt = self._get_task_prompt_text(scenario).lower()
        
        is_modification_task = any(word in task_prompt for word in 
                                 ['update', 'modify', 'change', 'edit', 'alter', 'refactor'])
        
        if not is_modification_task:
            return 0.6  # Neutral score for non-modification tasks
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            for pattern, indicators in modification_patterns.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    file_score += 0.25
            
            # Check for proper change tracking
            if 'version' in code_lower or 'changelog' in code_lower:
                file_score += 0.1
            
            total_score += min(file_score, 1.0)
        
        return total_score / len(solution_code)

    def _analyze_data_flow_understanding(self, solution_code: Dict[str, str]) -> float:
        """Analyze understanding of data flow patterns"""
        
        data_flow_patterns = {
            'input_validation': ['validate', 'sanitize', 'check', 'verify'],
            'data_transformation': ['transform', 'convert', 'map', 'filter'],
            'output_formatting': ['format', 'serialize', 'marshal', 'encode'],
            'error_propagation': ['error', 'exception', 'fail', 'panic']
        }
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Check for data flow patterns
            for pattern, indicators in data_flow_patterns.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    file_score += 0.2
            
            # Check for proper data flow structure (input -> process -> output)
            has_input = any(word in code_lower for word in ['input', 'request', 'param'])
            has_process = any(word in code_lower for word in ['process', 'handle', 'execute'])
            has_output = any(word in code_lower for word in ['output', 'response', 'return'])
            
            flow_completeness = sum([has_input, has_process, has_output]) / 3.0
            file_score += flow_completeness * 0.4
            
            total_score += min(file_score, 1.0)
        
        return total_score / len(solution_code)

    def _analyze_backward_compatibility(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze maintenance of backward compatibility"""
        
        compatibility_indicators = {
            'versioning': ['version', 'v1', 'v2', 'deprecated', 'legacy'],
            'adaptation': ['adapter', 'wrapper', 'bridge', 'facade'],
            'migration': ['migrate', 'upgrade', 'transition', 'convert'],
            'deprecation': ['deprecated', 'obsolete', 'remove', 'replace']
        }
        
        task_prompt = self._get_task_prompt_text(scenario).lower()
        is_compatibility_relevant = any(word in task_prompt for word in 
                                      ['update', 'upgrade', 'migrate', 'compatibility', 'legacy'])
        
        if not is_compatibility_relevant:
            return 0.7  # Neutral score when compatibility isn't relevant
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            for pattern, indicators in compatibility_indicators.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    if pattern == 'deprecation':
                        file_score += 0.1  # Small bonus for handling deprecation
                    else:
                        file_score += 0.3
            
            # Check for proper API preservation
            if 'api' in code_lower and 'breaking' not in code_lower:
                file_score += 0.2
            
            total_score += min(file_score, 1.0)
        
        return total_score / len(solution_code)

    def _analyze_code_reuse(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze effective code reuse patterns"""
        
        reuse_patterns = {
            'functions': ['func', 'function', 'def', 'method'],
            'modules': ['import', 'include', 'require', 'use'],
            'inheritance': ['extends', 'inherit', 'implement', 'interface'],
            'composition': ['compose', 'mixin', 'trait', 'delegate']
        }
        
        # Detect code duplication
        code_blocks = []
        for filename, code in solution_code.items():
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            code_blocks.extend(lines)
        
        # Simple duplication detection
        unique_lines = set(code_blocks)
        duplication_ratio = 1.0 - (len(unique_lines) / len(code_blocks)) if code_blocks else 0.0
        
        reuse_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Check for reuse patterns
            for pattern, indicators in reuse_patterns.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    file_score += 0.25
            
            reuse_score += min(file_score, 1.0)
        
        # Combine reuse patterns with duplication analysis
        pattern_score = reuse_score / len(solution_code)
        duplication_penalty = duplication_ratio * 0.5  # Penalize duplication
        
        return max(pattern_score - duplication_penalty, 0.0)

    def _analyze_extension_patterns(self, solution_code: Dict[str, str]) -> float:
        """Analyze extensibility and extension patterns"""
        
        extensibility_patterns = {
            'interfaces': ['interface', 'protocol', 'contract'],
            'plugins': ['plugin', 'extension', 'addon', 'module'],
            'hooks': ['hook', 'callback', 'listener', 'event'],
            'factories': ['factory', 'builder', 'creator', 'generator']
        }
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Check for extensibility patterns
            for pattern, indicators in extensibility_patterns.items():
                pattern_usage = sum(1 for indicator in indicators if indicator in code_lower)
                if pattern_usage > 0:
                    file_score += 0.25
            
            # Check for configuration support
            if any(word in code_lower for word in ['config', 'setting', 'option', 'parameter']):
                file_score += 0.15
            
            # Check for modular structure
            if any(word in code_lower for word in ['module', 'component', 'service', 'package']):
                file_score += 0.1
            
            total_score += min(file_score, 1.0)
        
        return total_score / len(solution_code)

    def _analyze_minimal_disruption(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze minimal disruption to existing codebase"""
        
        task_prompt = self._get_task_prompt_text(scenario).lower()
        context_files = scenario.get('context_files', [])
        
        # Check if this is a modification/integration task
        is_modification_task = any(word in task_prompt for word in 
                                 ['add', 'integrate', 'extend', 'enhance', 'modify'])
        
        if not is_modification_task:
            return 0.7  # Neutral score for new implementations
        
        disruption_indicators = {
            'breaking_changes': ['breaking', 'remove', 'delete', 'replace'],
            'non_breaking': ['add', 'extend', 'enhance', 'backward', 'compatible'],
            'isolation': ['separate', 'isolate', 'encapsulate', 'module'],
            'integration': ['integrate', 'connect', 'link', 'bridge']
        }
        
        total_score = 0.0
        
        for filename, code in solution_code.items():
            code_lower = code.lower()
            file_score = 0.0
            
            # Penalize breaking changes
            breaking_count = sum(1 for indicator in disruption_indicators['breaking_changes'] 
                               if indicator in code_lower)
            file_score -= breaking_count * 0.2
            
            # Reward non-breaking approaches
            non_breaking_count = sum(1 for indicator in disruption_indicators['non_breaking'] 
                                   if indicator in code_lower)
            file_score += non_breaking_count * 0.3
            
            # Reward isolation
            isolation_count = sum(1 for indicator in disruption_indicators['isolation'] 
                                if indicator in code_lower)
            file_score += isolation_count * 0.2
            
            total_score += max(file_score, 0.0)
        
        return min(total_score / len(solution_code), 1.0)

    # Helper methods for the new implementations
    def _extract_identifiers(self, code: str) -> List[str]:
        """Extract function and variable identifiers from code"""
        # Go-specific patterns
        go_functions = re.findall(r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        go_variables = re.findall(r'var\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        go_types = re.findall(r'type\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        
        # General patterns
        general_functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        general_variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:?=', code)
        
        return go_functions + go_variables + go_types + general_functions + general_variables

    def _find_naming_inconsistencies(self, identifiers1: List[str], identifiers2: List[str]) -> int:
        """Find naming inconsistencies between two sets of identifiers"""
        inconsistencies = 0
        
        # Simple heuristic: check for similar concepts with different naming styles
        for id1 in identifiers1:
            for id2 in identifiers2:
                # Check if they might represent similar concepts
                if self._are_similar_concepts(id1, id2):
                    # Check if naming styles are different
                    style1 = self._get_naming_style(id1)
                    style2 = self._get_naming_style(id2)
                    if style1 != style2:
                        inconsistencies += 1
        
        return inconsistencies

    def _are_similar_concepts(self, name1: str, name2: str) -> bool:
        """Check if two names might represent similar concepts"""
        # Simple similarity check based on common roots
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check for common prefixes/suffixes
        common_roots = ['get', 'set', 'create', 'update', 'delete', 'handle', 'process']
        
        for root in common_roots:
            if root in name1_lower and root in name2_lower:
                return True
        
        return False

    def _get_naming_style(self, name: str) -> str:
        """Determine the naming style of an identifier"""
        if '_' in name and name.islower():
            return 'snake_case'
        elif name[0].islower() and any(c.isupper() for c in name[1:]):
            return 'camelCase'
        elif name[0].isupper():
            return 'PascalCase'
        else:
            return 'other'

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _analyze_terminology_consistency(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze consistency of domain terminology usage"""
        
        # Extract domain terms from task prompt (use helper to handle multi-session scenarios)
        task_prompt = self._get_task_prompt_text(scenario)
        domain_terms = re.findall(r'\b[A-Z][a-z]+\b', task_prompt)
        
        if not domain_terms:
            return 0.6  # Neutral score if no clear domain terms
        
        total_code = ' '.join(solution_code.values())
        term_usage_consistency = 0.0
        
        for term in set(domain_terms):
            # Check if term is used consistently across solution
            term_variants = [term, term.lower(), term.upper()]
            usage_count = sum(total_code.count(variant) for variant in term_variants)
            
            if usage_count > 0:
                term_usage_consistency += 1.0 / len(set(domain_terms))
        
        return min(term_usage_consistency, 1.0)

    # Real implementations for remaining methods
    def _analyze_architectural_extraction(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze information extraction for architectural understanding tasks"""
        
        description = scenario.get('description', '').lower()
        task_prompt = self._get_task_prompt_text(scenario).lower()
        combined_text = description + ' ' + task_prompt
        total_code = ' '.join(solution_code.values()).lower()
        
        # 1. Architecture pattern recognition (40%)
        architecture_patterns = [
            'middleware', 'handler', 'controller', 'service', 'repository', 'model',
            'mvc', 'rest', 'api', 'router', 'endpoint', 'interface', 'struct',
            'package', 'module', 'component', 'layer', 'tier'
        ]
        
        pattern_extraction_score = 0.0
        patterns_mentioned = [p for p in architecture_patterns if p in combined_text]
        patterns_implemented = [p for p in patterns_mentioned if p in total_code]
        
        if patterns_mentioned:
            pattern_extraction_score = len(patterns_implemented) / len(patterns_mentioned)
        
        # 2. Structural element extraction (35%)
        structural_keywords = ['function', 'method', 'class', 'struct', 'interface', 'package']
        structural_score = 0.0
        
        for keyword in structural_keywords:
            if keyword in combined_text:
                # Check if solution implements this structural element
                if keyword == 'function' and 'func ' in total_code:
                    structural_score += 1
                elif keyword == 'struct' and 'type ' in total_code and 'struct' in total_code:
                    structural_score += 1
                elif keyword == 'interface' and 'interface' in total_code:
                    structural_score += 1
                elif keyword == 'package' and 'package ' in total_code:
                    structural_score += 1
        
        structural_score = min(structural_score / 4.0, 1.0)
        
        # 3. Dependency relationship extraction (25%)
        dependency_keywords = ['import', 'dependency', 'require', 'use', 'call', 'invoke']
        dependency_score = 0.0
        
        deps_mentioned = [d for d in dependency_keywords if d in combined_text]
        deps_implemented = [d for d in deps_mentioned if d in total_code or 'import' in total_code]
        
        if deps_mentioned:
            dependency_score = len(deps_implemented) / len(deps_mentioned)
        
        return (
            pattern_extraction_score * 0.4 +
            structural_score * 0.35 +
            dependency_score * 0.25
        )

    def _analyze_feature_extraction(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze information extraction for feature implementation tasks"""
        
        description = scenario.get('description', '').lower()
        task_prompt = self._get_task_prompt_text(scenario).lower()
        combined_text = description + ' ' + task_prompt
        total_code = ' '.join(solution_code.values()).lower()
        
        # 1. Business logic extraction (45%)
        business_keywords = [
            'user', 'customer', 'order', 'product', 'payment', 'account', 'profile',
            'create', 'update', 'delete', 'get', 'list', 'search', 'filter',
            'validate', 'process', 'calculate', 'generate', 'send', 'receive'
        ]
        
        business_extraction_score = 0.0
        business_mentioned = [k for k in business_keywords if k in combined_text]
        business_implemented = [k for k in business_mentioned if k in total_code]
        
        if business_mentioned:
            business_extraction_score = len(business_implemented) / len(business_mentioned)
        
        # 2. Data flow extraction (30%)
        data_flow_keywords = [
            'input', 'output', 'request', 'response', 'data', 'parameter',
            'return', 'result', 'json', 'xml', 'struct', 'map', 'array', 'slice'
        ]
        
        data_flow_score = 0.0
        data_mentioned = [k for k in data_flow_keywords if k in combined_text]
        data_implemented = [k for k in data_mentioned if k in total_code]
        
        if data_mentioned:
            data_flow_score = len(data_implemented) / len(data_mentioned)
        
        # 3. Error handling extraction (25%)
        error_keywords = ['error', 'exception', 'fail', 'invalid', 'check', 'validate']
        error_score = 0.0
        
        errors_mentioned = [e for e in error_keywords if e in combined_text]
        if errors_mentioned:
            # Check for Go error handling patterns
            has_error_handling = (
                'if err != nil' in total_code or
                'error' in total_code or
                'return' in total_code and 'err' in total_code
            )
            error_score = 1.0 if has_error_handling else 0.5
        
        return (
            business_extraction_score * 0.45 +
            data_flow_score * 0.30 +
            error_score * 0.25
        )

    def _analyze_general_extraction(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze information extraction for general tasks"""
        
        description = scenario.get('description', '').lower()
        task_prompt = self._get_task_prompt_text(scenario).lower()
        combined_text = description + ' ' + task_prompt
        total_code = ' '.join(solution_code.values()).lower()
        
        # 1. Key concept extraction (40%)
        # Extract nouns and important keywords from scenario
        import re
        
        # Simple noun extraction (words that are capitalized or technical terms)
        concept_patterns = re.findall(r'\b[A-Z][a-z]+\b|\b(?:func|struct|interface|package|import)\b', 
                                    description + ' ' + task_prompt)
        concept_score = 0.0
        
        if concept_patterns:
            concepts_in_code = [c for c in concept_patterns if c.lower() in total_code]
            concept_score = len(concepts_in_code) / len(concept_patterns)
        
        # 2. Action extraction (35%)
        action_keywords = [
            'implement', 'create', 'build', 'add', 'update', 'modify', 'delete',
            'handle', 'process', 'manage', 'execute', 'run', 'start', 'stop'
        ]
        
        action_score = 0.0
        actions_mentioned = [a for a in action_keywords if a in combined_text]
        
        if actions_mentioned:
            # Check if solution has function definitions (actions implemented)
            func_count = total_code.count('func ')
            action_score = min(func_count / len(actions_mentioned), 1.0)
        
        # 3. Technical requirement extraction (25%)
        tech_keywords = [
            'http', 'json', 'api', 'rest', 'endpoint', 'server', 'client',
            'database', 'sql', 'query', 'connection', 'session', 'cookie'
        ]
        
        tech_score = 0.0
        tech_mentioned = [t for t in tech_keywords if t in combined_text]
        tech_implemented = [t for t in tech_mentioned if t in total_code]
        
        if tech_mentioned:
            tech_score = len(tech_implemented) / len(tech_mentioned)
        
        return (
            concept_score * 0.4 +
            action_score * 0.35 +
            tech_score * 0.25
        )

    def _analyze_context_usage(self, solution_code: Dict[str, str], context_files: List[str]) -> float:
        """Analyze how well the solution uses provided context"""
        context_usage_score = 0.0
        
        if not context_files:
            return 0.5
        
        # Check if solution references context file patterns
        for filename, code in solution_code.items():
            for context_file in context_files:
                context_name = Path(context_file).stem
                if context_name.lower() in code.lower():
                    context_usage_score += 1.0 / len(context_files)
        
        return min(context_usage_score, 1.0)

    def _analyze_requirement_coverage(self, solution_code: Dict[str, str], task_prompt: str) -> float:
        """Analyze how well solution covers task requirements"""
        if not task_prompt:
            return 0.5
        
        # Extract key requirements from task prompt
        requirement_keywords = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b', task_prompt)
        requirement_keywords.extend(['implement', 'create', 'add', 'update', 'fix', 'analyze'])
        
        coverage_score = 0.0
        total_code = ' '.join(solution_code.values()).lower()
        
        for keyword in set(requirement_keywords):
            if keyword.lower() in total_code:
                coverage_score += 1.0 / len(set(requirement_keywords))
        
        return min(coverage_score, 1.0)

    def _analyze_information_extraction(self, solution_code: Dict[str, str], scenario: Dict[str, Any]) -> float:
        """Analyze quality of information extraction from scenario"""
        task_category = scenario.get('task_category', '')
        
        # Different extraction strategies based on task category
        if task_category == 'architectural_understanding':
            return self._analyze_architectural_extraction(solution_code, scenario)
        elif task_category == 'feature_implementation':
            return self._analyze_feature_extraction(solution_code, scenario)
        else:
            return self._analyze_general_extraction(solution_code, scenario)

    # ============================================================================
    # NEW ADVANCED METRICS FOR CADS v2 (EMU1 Implementation)
    # ============================================================================

    def calculate_robustness_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        RS: Robustness Score
        Measures code resilience, error handling, and security practices
        Components: Error Handling (25%), Input Validation (25%), Security (25%), Resource Management (25%)
        """
        try:
            # Sanitize solution_code to prevent type errors
            solution_code = self._sanitize_solution_code(solution_code)
            # Component 1: Error Handling Coverage (25%)
            error_handling_score = self._analyze_error_handling_coverage(solution_code)
            
            # Component 2: Input Validation (25%)
            input_validation_score = self._analyze_input_validation(solution_code)
            
            # Component 3: Security Practices (25%)
            security_practices_score = self._analyze_security_practices(solution_code)
            
            # Component 4: Resource Management (25%)
            resource_management_score = self._analyze_resource_management(solution_code)
            
            # Calculate weighted average
            rs_score = (
                error_handling_score * 0.25 +
                input_validation_score * 0.25 +
                security_practices_score * 0.25 +
                resource_management_score * 0.25
            )
            
            return min(max(rs_score, 0.0), 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating robustness score: {e}")
            return 0.0

    def calculate_comprehensiveness_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        CS: Comprehensiveness Score
        Measures documentation quality, API completeness, and deployment readiness
        Components: Documentation Quality (30%), API Completeness (25%), Test Indication (25%), Config Management (20%)
        """
        try:
            # Sanitize solution_code to prevent type errors
            solution_code = self._sanitize_solution_code(solution_code)
            # Component 1: Documentation Quality (30%)
            documentation_score = self._analyze_documentation_quality(solution_code)
            
            # Component 2: API Completeness (25%)
            api_completeness_score = self._analyze_api_completeness(solution_code)
            
            # Component 3: Test Indication (25%)
            test_indication_score = self._analyze_test_indication(solution_code)
            
            # Component 4: Configuration Management (20%)
            config_management_score = self._analyze_configuration_management(solution_code)
            
            # Calculate weighted average
            cs_score = (
                documentation_score * 0.30 +
                api_completeness_score * 0.25 +
                test_indication_score * 0.25 +
                config_management_score * 0.20
            )
            
            return min(max(cs_score, 0.0), 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating comprehensiveness score: {e}")
            return 0.0

    def calculate_innovation_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        IS: Innovation Score
        Measures algorithm efficiency, design patterns, and modern practices
        Components: Algorithm Efficiency (30%), Design Patterns (25%), Performance Optimization (25%), Modern Practices (20%)
        """
        try:
            # Sanitize solution_code to prevent type errors
            solution_code = self._sanitize_solution_code(solution_code)
            
            # Component 1: Algorithm Efficiency (30%)
            algorithm_efficiency_score = self._analyze_algorithm_efficiency(solution_code)
            
            # Component 2: Design Patterns (25%)
            design_patterns_score = self._analyze_design_patterns(solution_code)
            
            # Component 3: Performance Optimization (25%)
            performance_optimization_score = self._analyze_performance_optimization(solution_code)
            
            # Component 4: Modern Practices (20%)
            modern_practices_score = self._analyze_modern_practices(solution_code)
            
            # Calculate weighted average
            is_score = (
                algorithm_efficiency_score * 0.30 +
                design_patterns_score * 0.25 +
                performance_optimization_score * 0.25 +
                modern_practices_score * 0.20
            )
            
            return min(max(is_score, 0.0), 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating innovation score: {e}")
            return 0.0

    def calculate_system_thinking_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        STS: System Thinking Score
        Measures scalability, maintainability, and system-level design considerations
        Components: Scalability Design (25%), Maintainability (25%), Integration Awareness (25%), Extensibility (25%)
        """
        try:
            # Sanitize solution_code to prevent type errors
            solution_code = self._sanitize_solution_code(solution_code)
            
            # Component 1: Scalability Design (25%)
            scalability_score = self._analyze_scalability_design(solution_code)
            
            # Component 2: Maintainability (25%)
            maintainability_score = self._analyze_system_maintainability(solution_code)
            
            # Component 3: Integration Awareness (25%)
            integration_awareness_score = self._analyze_integration_awareness(solution_code)
            
            # Component 4: Extensibility (25%)
            extensibility_score = self._analyze_extensibility(solution_code)
            
            # Calculate weighted average
            sts_score = (
                scalability_score * 0.25 +
                maintainability_score * 0.25 +
                integration_awareness_score * 0.25 +
                extensibility_score * 0.25
            )
            
            return min(max(sts_score, 0.0), 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating system thinking score: {e}")
            return 0.0

    def calculate_solution_elegance_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        SES: Solution Elegance Score (DEPRECATED - use solution_quality + solution_conciseness)
        Kept for backwards compatibility
        """
        try:
            # Calculate both new metrics and average them
            quality_score = self.calculate_solution_quality_score(scenario, solution_code)
            conciseness_score = self.calculate_solution_conciseness_score(scenario, solution_code)
            return (quality_score + conciseness_score) / 2.0
        except Exception as e:
            logger.warning(f"Error calculating solution elegance score: {e}")
            return 0.0
    
    def calculate_solution_quality_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        SQS: Solution Quality Score (COMPREHENSION)
        Measures code readability, maintainability, and design quality
        Components: Code Clarity (40%), Abstraction Level (30%), Design Principles (30%)
        """
        try:
            # Sanitize solution_code to prevent type errors
            solution_code = self._sanitize_solution_code(solution_code)
            
            # Component 1: Code Clarity (40%) - Naming, structure, readability
            code_clarity_score = self._analyze_code_clarity(solution_code)
            
            # Component 2: Abstraction Level (30%) - Appropriate abstraction
            abstraction_level_score = self._analyze_abstraction_level(solution_code)
            
            # Component 3: Design Principles (30%) - SRP, modularity (not DRY/KISS)
            design_quality_score = self._analyze_design_quality(solution_code)
            
            # Calculate weighted average
            sqs_score = (
                code_clarity_score * 0.40 +
                abstraction_level_score * 0.30 +
                design_quality_score * 0.30
            )
            
            return min(max(sqs_score, 0.0), 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating solution quality score: {e}")
            return 0.0
    
    def calculate_solution_conciseness_score(self, scenario: Dict[str, Any], solution_code: Dict[str, str]) -> float:
        """
        SCS: Solution Conciseness Score (EFFICIENCY)
        Measures code brevity, simplicity, and avoiding waste
        Components: DRY Adherence (50%), KISS Adherence (50%)
        """
        try:
            # Sanitize solution_code to prevent type errors
            solution_code = self._sanitize_solution_code(solution_code)
            
            # Component 1: DRY Principle (50%) - No code duplication
            dry_score = self._check_dry_principle(solution_code)
            
            # Component 2: KISS Principle (50%) - Low complexity, simplicity
            kiss_score = self._check_kiss_principle(solution_code)
            
            # Calculate weighted average
            scs_score = (
                dry_score * 0.50 +
                kiss_score * 0.50
            )
            
            return min(max(scs_score, 0.0), 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating solution conciseness score: {e}")
            return 0.0

    # ============================================================================
    # ROBUSTNESS SCORE COMPONENT IMPLEMENTATIONS
    # ============================================================================

    def _analyze_error_handling_coverage(self, solution_code: Dict[str, str]) -> float:
        """Analyze error handling coverage in the code"""
        total_code = '\n'.join(solution_code.values())
        
        # Count functions and error handling patterns
        try:
            tree = ast.parse(total_code)
        except SyntaxError:
            return 0.0
        
        function_count = 0
        functions_with_error_handling = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                
                # Check for error handling in function
                has_error_handling = False
                for child in ast.walk(node):
                    if isinstance(child, (ast.Try, ast.Raise)):
                        has_error_handling = True
                        break
                    # Check for defensive programming patterns
                    if isinstance(child, ast.If) and self._is_defensive_check(child):
                        has_error_handling = True
                        break
                
                if has_error_handling:
                    functions_with_error_handling += 1
        
        if function_count == 0:
            return 0.5  # No functions to evaluate
        
        return functions_with_error_handling / function_count

    def _is_defensive_check(self, node: ast.If) -> bool:
        """Check if an if statement represents defensive programming"""
        # Look for common defensive patterns
        if hasattr(node.test, 'comparators'):
            # Check for None checks, empty checks, etc.
            code_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
            defensive_patterns = ['is None', 'is not None', 'len(', 'not ', '== 0', '!= 0']
            return any(pattern in code_str for pattern in defensive_patterns)
        return False

    def _analyze_input_validation(self, solution_code: Dict[str, str]) -> float:
        """Analyze input validation patterns"""
        total_code = '\n'.join(solution_code.values())
        
        # Count validation patterns
        validation_patterns = [
            r'isinstance\(',  # Type checking
            r'assert\s+',     # Assertions
            r'if.*len\(',     # Length validation
            r'if.*range\(',   # Range validation
            r'\.isdigit\(',   # Format validation
            r'\.strip\(',     # Input cleaning
            r're\.match\(',   # Regex validation
        ]
        
        validation_count = 0
        for pattern in validation_patterns:
            validation_count += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        # Normalize based on code size (rough heuristic)
        lines_count = len(total_code.split('\n'))
        if lines_count == 0:
            return 0.0
        
        # Score based on validation density
        validation_density = validation_count / lines_count
        return min(validation_density * 10, 1.0)  # Scale and cap at 1.0

    def _analyze_security_practices(self, solution_code: Dict[str, str]) -> float:
        """Analyze security practices in the code"""
        total_code = '\n'.join(solution_code.values()).lower()
        
        security_score = 0.0
        total_checks = 0
        
        # Check for security-related imports/practices
        security_practices = {
            'crypto': ['hashlib', 'cryptography', 'secrets', 'bcrypt'],
            'input_sanitization': ['escape', 'sanitize', 'clean', 'validate'],
            'secure_random': ['secrets.', 'random.SystemRandom'],
            'sql_injection': ['parameterized', 'prepared', 'placeholder'],
            'authentication': ['auth', 'login', 'password', 'token']
        }
        
        for category, patterns in security_practices.items():
            total_checks += 1
            found_any = any(pattern in total_code for pattern in patterns)
            if found_any:
                security_score += 1
        
        # Check for potential security issues (negative score)
        security_issues = [
            'eval(',           # Code injection
            'exec(',           # Code execution
            'os.system(',      # Command injection
            'subprocess.call(' # Command execution
        ]
        
        issue_count = sum(1 for issue in security_issues if issue in total_code)
        
        if total_checks == 0:
            return 0.5
        
        # Positive practices minus penalties for issues
        base_score = security_score / total_checks
        penalty = min(issue_count * 0.2, 0.5)  # Cap penalty at 0.5
        
        return max(base_score - penalty, 0.0)

    def _analyze_resource_management(self, solution_code: Dict[str, str]) -> float:
        """Analyze resource management patterns"""
        total_code = '\n'.join(solution_code.values())
        
        # Look for resource management patterns
        resource_patterns = [
            r'with\s+open\(',     # File context managers
            r'with\s+.*\(\)\s*:', # General context managers
            r'\.close\(\)',       # Explicit resource closing
            r'finally:',          # Finally blocks
            r'__enter__',         # Context manager implementation
            r'__exit__'           # Context manager implementation
        ]
        
        resource_management_count = 0
        for pattern in resource_patterns:
            resource_management_count += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        # Look for resource acquisition patterns to normalize against
        resource_acquisition = [
            r'open\(',
            r'socket\.',
            r'connect\(',
            r'cursor\(',
            r'thread\.',
            r'process\.'
        ]
        
        acquisition_count = 0
        for pattern in resource_acquisition:
            acquisition_count += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        if acquisition_count == 0:
            # No resources acquired, so perfect score if no management needed
            return 1.0 if resource_management_count == 0 else 0.8
        
        # Score based on ratio of management to acquisition
        return min(resource_management_count / acquisition_count, 1.0)

    # ============================================================================
    # COMPREHENSIVENESS SCORE COMPONENT IMPLEMENTATIONS
    # ============================================================================

    def _analyze_documentation_quality(self, solution_code: Dict[str, str]) -> float:
        """Analyze documentation quality and coverage"""
        total_functions = 0
        documented_functions = 0
        total_comments = 0
        
        for filename, code in solution_code.items():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            
            # Count docstrings and comments
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    
                    # Check for docstring
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Str)):
                        documented_functions += 1
            
            # Count comments (rough approximation)
            comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]
            total_comments += len(comment_lines)
        
        # Calculate documentation coverage
        doc_coverage = documented_functions / total_functions if total_functions > 0 else 0.0
        
        # Calculate comment density
        total_lines = sum(len(code.split('\n')) for code in solution_code.values())
        comment_density = total_comments / total_lines if total_lines > 0 else 0.0
        
        # Combine coverage and density with quality multiplier
        base_score = (doc_coverage * 0.7 + min(comment_density * 5, 1.0) * 0.3)
        
        # Quality multiplier based on README presence and content
        quality_multiplier = 1.0
        if any('readme' in filename.lower() for filename in solution_code.keys()):
            quality_multiplier = 1.2
        
        return min(base_score * quality_multiplier, 1.0)

    def _analyze_api_completeness(self, solution_code: Dict[str, str]) -> float:
        """Analyze API design completeness"""
        total_functions = 0
        functions_with_types = 0
        functions_with_docstrings = 0
        
        for code in solution_code.values():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    
                    # Check for type annotations
                    has_types = (node.returns is not None or 
                               any(arg.annotation is not None for arg in node.args.args))
                    if has_types:
                        functions_with_types += 1
                    
                    # Check for docstrings
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Str)):
                        functions_with_docstrings += 1
        
        if total_functions == 0:
            return 0.5
        
        type_coverage = functions_with_types / total_functions
        doc_coverage = functions_with_docstrings / total_functions
        
        # Simple average for now - could be weighted differently
        return (type_coverage + doc_coverage) / 2

    def _analyze_test_indication(self, solution_code: Dict[str, str]) -> float:
        """Analyze presence and quality of test-related code"""
        score = 0.0
        
        # Check for test files
        test_files = [f for f in solution_code.keys() if 'test' in f.lower()]
        if test_files:
            score += 0.4
        
        # Check for assertion patterns in any file
        total_code = '\n'.join(solution_code.values())
        assertion_patterns = [
            r'assert\s+',
            r'assertEqual\(',
            r'assertTrue\(',
            r'assertFalse\(',
            r'expect\(',
            r'should\.',
        ]
        
        assertion_count = 0
        for pattern in assertion_patterns:
            assertion_count += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        if assertion_count > 0:
            score += 0.4
        
        # Check for test frameworks/imports
        test_imports = ['unittest', 'pytest', 'jest', 'mocha', 'junit']
        for test_import in test_imports:
            if test_import in total_code.lower():
                score += 0.2
                break
        
        return min(score, 1.0)

    def _analyze_configuration_management(self, solution_code: Dict[str, str]) -> float:
        """Analyze configuration and deployment readiness"""
        score = 0.0
        
        # Check for configuration files
        config_files = [
            'config.yaml', 'config.yml', 'config.json', 'config.ini',
            'settings.py', '.env', 'environment', 'docker', 'makefile'
        ]
        
        found_config_files = [f for f in solution_code.keys() 
                            if any(config in f.lower() for config in config_files)]
        if found_config_files:
            score += 0.25
        
        # Check for environment variable usage
        total_code = '\n'.join(solution_code.values())
        env_patterns = ['os.environ', 'getenv', 'env.get', 'process.env']
        if any(pattern in total_code for pattern in env_patterns):
            score += 0.25
        
        # Check for build/deployment scripts
        build_patterns = ['setup.py', 'requirements.txt', 'package.json', 'build', 'install']
        if any(pattern in filename.lower() for filename in solution_code.keys() 
               for pattern in build_patterns):
            score += 0.25
        
        # Check for documentation about deployment
        doc_patterns = ['deploy', 'install', 'setup', 'run', 'start']
        if any(pattern in total_code.lower() for pattern in doc_patterns):
            score += 0.25
        
        return min(score, 1.0)

    # ============================================================================
    # INNOVATION SCORE COMPONENT IMPLEMENTATIONS
    # ============================================================================

    def _analyze_algorithm_efficiency(self, solution_code: Dict[str, str]) -> float:
        """Analyze algorithm efficiency and optimization"""
        total_code = '\n'.join(solution_code.values())
        
        # Look for efficient algorithms and data structures
        efficient_patterns = [
            r'collections\.deque',       # Efficient queue
            r'collections\.defaultdict', # Efficient dictionary
            r'set\(',                    # Set operations
            r'bisect\.',                 # Binary search
            r'heapq\.',                  # Heap operations
            r'itertools\.',              # Efficient iterators
            r'enumerate\(',              # Efficient iteration
            r'zip\(',                    # Parallel iteration
        ]
        
        efficiency_score = 0
        for pattern in efficient_patterns:
            if re.search(pattern, total_code):
                efficiency_score += 1
        
        # Penalize inefficient patterns
        inefficient_patterns = [
            r'\.append\(.*in.*for',  # List comprehension as append in loop
            r'string.*\+.*in.*for',  # String concatenation in loop
            r'list.*dict\.keys\(\)', # Converting dict keys to list unnecessarily
        ]
        
        inefficiency_penalty = 0
        for pattern in inefficient_patterns:
            inefficiency_penalty += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        # Normalize score
        base_score = min(efficiency_score / 8, 1.0)  # Scale by number of patterns
        penalty = min(inefficiency_penalty * 0.1, 0.5)  # Cap penalty
        
        return max(base_score - penalty, 0.0)

    def _analyze_design_patterns(self, solution_code: Dict[str, str]) -> float:
        """Analyze usage of design patterns"""
        total_code = '\n'.join(solution_code.values()).lower()
        
        # Common design patterns
        pattern_indicators = {
            'factory': ['factory', 'create', 'builder'],
            'singleton': ['singleton', '_instance', '__new__'],
            'observer': ['observer', 'listener', 'notify', 'subscribe'],
            'strategy': ['strategy', 'algorithm', 'policy'],
            'decorator': ['decorator', '@', 'wrapper'],
            'adapter': ['adapter', 'wrapper', 'bridge'],
            'mvc': ['model', 'view', 'controller'],
            'repository': ['repository', 'repo', 'dao']
        }
        
        patterns_found = 0
        for pattern_name, indicators in pattern_indicators.items():
            if any(indicator in total_code for indicator in indicators):
                patterns_found += 1
        
        # Score based on pattern usage appropriateness
        return min(patterns_found / len(pattern_indicators), 1.0)

    def _analyze_performance_optimization(self, solution_code: Dict[str, str]) -> float:
        """Analyze performance optimization techniques"""
        total_code = '\n'.join(solution_code.values())
        
        # Performance optimization indicators
        optimization_patterns = [
            r'@lru_cache',        # Memoization
            r'@cache',            # Caching decorator
            r'async\s+def',       # Async programming
            r'await\s+',          # Await calls
            r'threading\.',       # Threading
            r'multiprocessing\.', # Multiprocessing
            r'concurrent\.',      # Concurrent execution
            r'asyncio\.',         # Async I/O
            r'pool\.',            # Thread/process pools
        ]
        
        optimization_count = 0
        for pattern in optimization_patterns:
            optimization_count += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        # Normalize based on code complexity
        lines_count = len(total_code.split('\n'))
        if lines_count == 0:
            return 0.0
        
        optimization_density = optimization_count / lines_count
        return min(optimization_density * 20, 1.0)  # Scale and cap

    def _analyze_modern_practices(self, solution_code: Dict[str, str]) -> float:
        """Analyze usage of modern frameworks and practices"""
        total_code = '\n'.join(solution_code.values()).lower()
        
        # Modern practice indicators
        modern_indicators = [
            'fastapi', 'flask', 'django',  # Modern web frameworks
            'pydantic', 'dataclass',       # Modern data validation
            'typing', 'type:', '->',       # Type annotations
            'f"', "f'",                    # F-strings
            'pathlib', 'path',             # Modern path handling
            'json', 'requests',            # Modern I/O
            'logging',                     # Proper logging
            'with open',                   # Context managers
        ]
        
        modern_count = sum(1 for indicator in modern_indicators if indicator in total_code)
        
        # Score based on modern practice adoption
        return min(modern_count / len(modern_indicators), 1.0)

    # ============================================================================
    # SYSTEM THINKING SCORE COMPONENT IMPLEMENTATIONS
    # ============================================================================

    def _analyze_scalability_design(self, solution_code: Dict[str, str]) -> float:
        """Analyze scalability considerations"""
        total_code = '\n'.join(solution_code.values()).lower()
        
        scalability_patterns = [
            'queue', 'pagination', 'batch', 'chunk',     # Load handling
            'pool', 'cache', 'redis', 'memcache',        # Resource scaling
            'async', 'thread', 'concurrent', 'parallel', # Concurrency
            'database', 'connection', 'session',         # Database handling
            'rate_limit', 'throttle', 'backoff',         # Rate limiting
        ]
        
        scalability_count = sum(1 for pattern in scalability_patterns if pattern in total_code)
        
        # Normalize and score
        return min(scalability_count / len(scalability_patterns), 1.0)

    def _analyze_system_maintainability(self, solution_code: Dict[str, str]) -> float:
        """Analyze maintainability indicators"""
        total_functions = 0
        total_complexity = 0
        
        for code in solution_code.values():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    # Simple complexity measure (nested depth)
                    complexity = self._calculate_function_complexity(node)
                    total_complexity += complexity
        
        if total_functions == 0:
            return 0.5
        
        avg_complexity = total_complexity / total_functions
        
        # Lower complexity is better (inverted score)
        complexity_score = max(1.0 - (avg_complexity - 1) / 10, 0.0)
        
        # Check for modular organization
        modularity_score = len(solution_code) / max(len(solution_code), 3)  # Prefer multiple files
        
        return (complexity_score + modularity_score) / 2

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate simple complexity metric for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
        
        return complexity

    def _analyze_integration_awareness(self, solution_code: Dict[str, str]) -> float:
        """Analyze integration and API design awareness"""
        total_code = '\n'.join(solution_code.values()).lower()
        
        integration_patterns = [
            'api', 'endpoint', 'route', 'rest',     # API design
            'json', 'xml', 'serialize', 'marshal',  # Data interchange
            'http', 'request', 'response',          # Web integration
            'client', 'server', 'service',          # Service design
            'interface', 'contract', 'schema',      # Interface definition
        ]
        
        integration_count = sum(1 for pattern in integration_patterns if pattern in total_code)
        
        return min(integration_count / len(integration_patterns), 1.0)

    def _analyze_extensibility(self, solution_code: Dict[str, str]) -> float:
        """Analyze extensibility and flexibility"""
        total_code = '\n'.join(solution_code.values())
        
        extensibility_patterns = [
            r'class.*\(.*\):',    # Inheritance
            r'abc\.ABC',          # Abstract base classes
            r'@abstractmethod',   # Abstract methods
            r'plugin', 'hook',    # Plugin patterns
            r'config', 'setting', # Configuration
            r'registry', 'factory' # Extensible factories
        ]
        
        extensibility_count = 0
        for pattern in extensibility_patterns:
            extensibility_count += len(re.findall(pattern, total_code, re.IGNORECASE))
        
        # Normalize based on code size
        lines_count = len(total_code.split('\n'))
        if lines_count == 0:
            return 0.0
        
        extensibility_density = extensibility_count / lines_count
        return min(extensibility_density * 10, 1.0)

    # ============================================================================
    # SOLUTION ELEGANCE SCORE COMPONENT IMPLEMENTATIONS
    # ============================================================================

    def _analyze_code_clarity(self, solution_code: Dict[str, str]) -> float:
        """Analyze code clarity and readability"""
        # Sanitize solution_code to prevent type errors
        solution_code = self._sanitize_solution_code(solution_code)
        
        total_score = 0
        total_files = 0
        
        for filename, code in solution_code.items():
            total_files += 1
            file_score = 0
            
            # Analyze naming quality
            naming_score = self._analyze_naming_quality(code)
            file_score += naming_score * 0.4
            
            # Analyze structure clarity
            structure_score = self._analyze_structure_clarity(code)
            file_score += structure_score * 0.3
            
            # Analyze line length and complexity
            readability_score = self._analyze_readability_metrics(code)
            file_score += readability_score * 0.3
            
            total_score += file_score
        
        return total_score / total_files if total_files > 0 else 0.0

    def _analyze_naming_quality(self, code: str) -> float:
        """Analyze quality of variable and function names"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
        
        if not names:
            return 0.5
        
        # Analyze naming patterns
        good_names = 0
        for name in names:
            if len(name) > 2 and not name.startswith('_') and name.islower():
                good_names += 1
        
        return good_names / len(names)

    def _analyze_structure_clarity(self, code: str) -> float:
        """Analyze code structure clarity"""
        lines = code.split('\n')
        
        # Check for consistent indentation
        indentation_consistent = True
        current_indent = 0
        
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # Non-empty line
                line_indent = len(line) - len(stripped)
                if line_indent % 4 != 0:  # Assume 4-space indentation
                    indentation_consistent = False
                    break
        
        # Check for reasonable function length
        function_lengths = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 10
                    function_lengths.append(func_length)
        except SyntaxError:
            pass
        
        avg_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 10
        length_score = max(1.0 - (avg_function_length - 10) / 50, 0.0)  # Prefer shorter functions
        
        consistency_score = 1.0 if indentation_consistent else 0.5
        
        return (consistency_score + length_score) / 2

    def _analyze_readability_metrics(self, code: str) -> float:
        """Analyze readability metrics"""
        lines = [line for line in code.split('\n') if line.strip()]
        
        if not lines:
            return 0.5
        
        # Check average line length
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        line_length_score = max(1.0 - (avg_line_length - 80) / 80, 0.0)  # Prefer lines under 80 chars
        
        # Check comment ratio
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        comment_ratio = len(comment_lines) / len(lines)
        comment_score = min(comment_ratio * 5, 1.0)  # Reward commenting
        
        return (line_length_score + comment_score) / 2

    def _analyze_abstraction_level(self, solution_code: Dict[str, str]) -> float:
        """Analyze appropriateness of abstraction level"""
        total_code = '\n'.join(solution_code.values())
        
        # Measure complexity indicators
        try:
            tree = ast.parse(total_code)
        except SyntaxError:
            return 0.5
        
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        total_lines = len(total_code.split('\n'))
        
        if total_lines == 0:
            return 0.5
        
        # Calculate abstraction density
        abstraction_density = (class_count + function_count) / total_lines
        
        # Optimal range: not too abstract, not too concrete
        if 0.05 <= abstraction_density <= 0.2:
            return 1.0
        elif abstraction_density < 0.05:
            return abstraction_density / 0.05  # Under-abstracted
        else:
            return max(0.4 - (abstraction_density - 0.2) / 0.1, 0.0)  # Over-abstracted

    def _analyze_principle_adherence(self, solution_code: Dict[str, str]) -> float:
        """Analyze adherence to SOLID and other principles"""
        total_score = 0
        principle_count = 0
        
        # DRY Principle (Don't Repeat Yourself)
        dry_score = self._check_dry_principle(solution_code)
        total_score += dry_score
        principle_count += 1
        
        # KISS Principle (Keep It Simple, Stupid)
        kiss_score = self._check_kiss_principle(solution_code)
        total_score += kiss_score
        principle_count += 1
        
        # Single Responsibility (basic check)
        srp_score = self._check_single_responsibility(solution_code)
        total_score += srp_score
        principle_count += 1
        
        return total_score / principle_count if principle_count > 0 else 0.0

    def _check_dry_principle(self, solution_code: Dict[str, str]) -> float:
        """Check for code duplication (DRY principle)"""
        all_lines = []
        for code in solution_code.values():
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            all_lines.extend(lines)
        
        if len(all_lines) == 0:
            return 1.0
        
        # Count duplicate lines
        line_counts = Counter(all_lines)
        duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        duplication_ratio = duplicate_lines / len(all_lines)
        return max(1.0 - duplication_ratio * 2, 0.0)  # Penalize duplication

    def _check_kiss_principle(self, solution_code: Dict[str, str]) -> float:
        """Check for unnecessary complexity (KISS principle)"""
        complexity_score = 0
        file_count = 0
        
        for code in solution_code.values():
            file_count += 1
            try:
                tree = ast.parse(code)
                
                # Count nested structures
                max_nesting = 0
                current_nesting = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                        current_nesting += 1
                        max_nesting = max(max_nesting, current_nesting)
                    # Note: This is a simplified nesting calculation
                
                # Score based on nesting depth (simpler is better)
                file_complexity = max(1.0 - (max_nesting - 2) / 5, 0.0)
                complexity_score += file_complexity
                
            except SyntaxError:
                complexity_score += 0.5
        
        return complexity_score / file_count if file_count > 0 else 0.0

    def _check_single_responsibility(self, solution_code: Dict[str, str]) -> float:
        """Check for single responsibility principle (basic heuristic)"""
        total_score = 0
        total_classes = 0
        
        for code in solution_code.values():
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        total_classes += 1
                        
                        # Count methods to estimate responsibility
                        method_count = sum(1 for child in node.body 
                                         if isinstance(child, ast.FunctionDef))
                        
                        # Prefer classes with moderate number of methods
                        if 1 <= method_count <= 10:
                            total_score += 1.0
                        elif method_count <= 15:
                            total_score += 0.7
                        else:
                            total_score += 0.3
                            
            except SyntaxError:
                continue
        
        return total_score / total_classes if total_classes > 0 else 0.8
    
    def _analyze_design_quality(self, solution_code: Dict[str, str]) -> float:
        """
        Analyze design quality (for solution_quality metric - COMPREHENSION)
        Focuses on SRP and modularity principles, excludes DRY/KISS (those are in solution_conciseness)
        """
        total_score = 0
        principle_count = 0
        
        # Single Responsibility Principle
        srp_score = self._check_single_responsibility(solution_code)
        total_score += srp_score
        principle_count += 1
        
        # Check for modularity (appropriate function decomposition)
        modularity_score = self._check_modularity(solution_code)
        total_score += modularity_score
        principle_count += 1
        
        return total_score / principle_count if principle_count > 0 else 0.0
    
    def _check_modularity(self, solution_code: Dict[str, str]) -> float:
        """Check for appropriate function decomposition"""
        total_score = 0
        file_count = 0
        
        for code in solution_code.values():
            file_count += 1
            try:
                tree = ast.parse(code)
                
                # Count functions and their average length
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                
                if len(functions) == 0:
                    # No functions - procedural code (neutral score)
                    total_score += 0.5
                    continue
                
                # Analyze function lengths
                total_lines = len(code.split('\n'))
                avg_function_length = total_lines / len(functions) if functions else 0
                
                # Prefer moderate function lengths (10-50 lines)
                if 10 <= avg_function_length <= 50:
                    total_score += 1.0
                elif 5 <= avg_function_length <= 100:
                    total_score += 0.7
                else:
                    total_score += 0.4
                    
            except SyntaxError:
                total_score += 0.5
        
        return total_score / file_count if file_count > 0 else 0.0

    def _analyze_style_consistency(self, solution_code: Dict[str, str]) -> float:
        """Analyze style consistency across files"""
        if not solution_code:
            return 0.0
        
        # Check naming conventions consistency
        naming_styles = []
        for code in solution_code.values():
            style = self._detect_naming_style(code)
            naming_styles.append(style)
        
        # Check indentation consistency
        indentation_styles = []
        for code in solution_code.values():
            style = self._detect_indentation_style(code)
            indentation_styles.append(style)
        
        # Calculate consistency scores
        naming_consistency = len(set(naming_styles)) == 1 if naming_styles else True
        indentation_consistency = len(set(indentation_styles)) == 1 if indentation_styles else True
        
        consistency_score = (naming_consistency + indentation_consistency) / 2
        
        return consistency_score

    def _detect_naming_style(self, code: str) -> str:
        """Detect naming style (snake_case, camelCase, etc.)"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "unknown"
        
        snake_case_count = 0
        camel_case_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Name)):
                name = getattr(node, 'name', getattr(node, 'id', ''))
                if '_' in name:
                    snake_case_count += 1
                elif any(c.isupper() for c in name[1:]):
                    camel_case_count += 1
        
        return "snake_case" if snake_case_count > camel_case_count else "camelCase"

    def _detect_indentation_style(self, code: str) -> str:
        """Detect indentation style (spaces vs tabs, size)"""
        lines = code.split('\n')
        
        for line in lines:
            if line.startswith('    '):
                return "4_spaces"
            elif line.startswith('  '):
                return "2_spaces"
            elif line.startswith('\t'):
                return "tabs"
        
        return "unknown" 