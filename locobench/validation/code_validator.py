"""
Real Code Validation Components for LoCoBench

This module implements actual compilation testing, security scanning,
and code quality analysis for the functional correctness evaluation.
"""

import ast
import subprocess
import tempfile
import shutil
import json
import re
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Result of code compilation attempt"""
    success: bool
    errors: List[str]
    warnings: List[str]
    execution_time: float
    binary_size: Optional[int] = None
    

@dataclass
class SecurityAnalysisResult:
    """Result of security analysis"""
    security_score: float
    vulnerabilities: List[Dict[str, Any]]
    risk_level: str
    

@dataclass
class QualityAnalysisResult:
    """Result of code quality analysis"""
    complexity_score: float
    maintainability_score: float
    test_coverage: float
    code_smells: List[Dict[str, Any]]


class CodeValidator:
    """Real code validation with compilation, security, and quality analysis"""
    
    def __init__(self):
        self.temp_dir = None
        self.supported_languages = {
            'go': {
                'compile_cmd': ['go', 'build'],
                'run_cmd': ['go', 'run'],
                'test_cmd': ['go', 'test'],
                'fmt_cmd': ['go', 'fmt'],
                'vet_cmd': ['go', 'vet'],
                'extension': '.go'
            },
            'python': {
                'compile_cmd': ['python3', '-m', 'py_compile'],
                'run_cmd': ['python3'],
                'test_cmd': ['python3', '-m', 'pytest'],
                'fmt_cmd': ['black', '--check'],
                'lint_cmd': ['pylint'],
                'extension': '.py'
            },
            'javascript': {
                'compile_cmd': ['node', '--check'],
                'run_cmd': ['node'],
                'test_cmd': ['npm', 'test'],
                'fmt_cmd': ['prettier', '--check'],
                'lint_cmd': ['eslint'],
                'extension': '.js'
            },
            'java': {
                'compile_cmd': ['javac'],
                'run_cmd': ['java'],
                'test_cmd': ['java', '-cp', '.', 'org.junit.runner.JUnitCore'],
                'fmt_cmd': ['google-java-format', '--dry-run'],
                'extension': '.java'
            },
            'c': {
                'compile_cmd': ['gcc', '-o', 'main'],
                'run_cmd': ['./main'],
                'test_cmd': ['gcc', '-o', 'test'],
                'fmt_cmd': ['clang-format', '--dry-run'],
                'extension': '.c'
            },
            'cpp': {
                'compile_cmd': ['g++', '-o', 'main'],
                'run_cmd': ['./main'],
                'test_cmd': ['g++', '-o', 'test'],
                'fmt_cmd': ['clang-format', '--dry-run'],
                'extension': '.cpp'
            },
            'csharp': {
                'compile_cmd': ['dotnet', 'build'],
                'run_cmd': ['dotnet', 'run'],
                'test_cmd': ['dotnet', 'test'],
                'fmt_cmd': ['dotnet', 'format', '--verify-no-changes'],
                'extension': '.cs'
            },
            'php': {
                'compile_cmd': ['php', '-l'],  # Syntax check
                'run_cmd': ['php'],
                'test_cmd': ['php', 'vendor/bin/phpunit'],
                'fmt_cmd': ['php-cs-fixer', 'fix', '--dry-run'],
                'extension': '.php'
            },
            'rust': {
                'compile_cmd': ['rustc'],
                'run_cmd': ['./main'],
                'test_cmd': ['cargo', 'test'],
                'fmt_cmd': ['rustfmt', '--check'],
                'extension': '.rs'
            },
            'typescript': {
                'compile_cmd': ['tsc', '--noEmit'],  # Type check only
                'run_cmd': ['node'],
                'test_cmd': ['npm', 'test'],
                'fmt_cmd': ['prettier', '--check'],
                'extension': '.ts'
            }
        }

    async def validate_compilation(self, solution_code: Dict[str, str], 
                                 language: str = 'go') -> CompilationResult:
        """Test if code compiles successfully"""
        
        if language not in self.supported_languages:
            return CompilationResult(
                success=False,
                errors=[f"Unsupported language: {language}"],
                warnings=[],
                execution_time=0.0
            )
        
        lang_config = self.supported_languages[language]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Write code files to temporary directory
                await self._write_code_files(solution_code, temp_dir, lang_config['extension'])
                
                # Initialize Go module if needed
                if language == 'go':
                    await self._init_go_module(temp_dir)
                
                # Attempt compilation
                start_time = time.time()
                result = await self._compile_code(temp_dir, lang_config['compile_cmd'])
                compilation_time = time.time() - start_time
                
                # Check for binary output
                binary_size = self._get_binary_size(temp_dir, language)
                
                return CompilationResult(
                    success=result['success'],
                    errors=result['errors'],
                    warnings=result['warnings'],
                    execution_time=compilation_time,
                    binary_size=binary_size
                )
                
            except Exception as e:
                logger.error(f"Compilation validation failed: {e}")
                return CompilationResult(
                    success=False,
                    errors=[f"Validation error: {str(e)}"],
                    warnings=[],
                    execution_time=0.0
                )

    async def analyze_security(self, solution_code: Dict[str, str], 
                             language: str = 'go') -> SecurityAnalysisResult:
        """Analyze code for security vulnerabilities"""
        
        # solution_code is already sanitized at the source in evaluator.py
        
        vulnerabilities = []
        total_score = 1.0
        
        # Security pattern analysis
        security_patterns = self._get_security_patterns(language)
        
        for filename, code in solution_code.items():
            file_vulns = self._scan_security_patterns(code, security_patterns, filename)
            vulnerabilities.extend(file_vulns)
        
        # Calculate security score based on vulnerabilities
        if vulnerabilities:
            # Deduct points based on severity
            severity_weights = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
            total_deduction = sum(severity_weights.get(vuln['severity'], 0.1) 
                                for vuln in vulnerabilities)
            total_score = max(0.0, 1.0 - total_deduction)
        
        # Determine risk level
        risk_level = self._calculate_risk_level(total_score, len(vulnerabilities))
        
        return SecurityAnalysisResult(
            security_score=total_score,
            vulnerabilities=vulnerabilities,
            risk_level=risk_level
        )

    async def analyze_code_quality(self, solution_code: Dict[str, str], 
                                 language: str = 'go') -> QualityAnalysisResult:
        """Analyze code quality metrics"""
        
        # solution_code is already sanitized at the source in evaluator.py
        
        complexity_scores = []
        maintainability_scores = []
        code_smells = []
        
        for filename, code in solution_code.items():
            # Complexity analysis
            complexity = self._calculate_complexity(code, language)
            complexity_scores.append(complexity)
            
            # Maintainability analysis
            maintainability = self._calculate_maintainability(code, language)
            maintainability_scores.append(maintainability)
            
            # Code smell detection
            smells = self._detect_code_smells(code, language, filename)
            code_smells.extend(smells)
        
        # Calculate averages
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
        avg_maintainability = sum(maintainability_scores) / len(maintainability_scores) if maintainability_scores else 0.0
        
        # Mock test coverage for now (would integrate with actual coverage tools)
        test_coverage = self._estimate_test_coverage(solution_code, language)
        
        return QualityAnalysisResult(
            complexity_score=avg_complexity,
            maintainability_score=avg_maintainability,
            test_coverage=test_coverage,
            code_smells=code_smells
        )

    async def run_unit_tests(self, solution_code: Dict[str, str], 
                           test_definitions: List[Dict], 
                           language: str = 'go') -> float:
        """Run unit tests against the solution"""
        
        if not test_definitions:
            return 0.5  # No tests provided
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Write solution code
                lang_config = self.supported_languages[language]
                await self._write_code_files(solution_code, temp_dir, lang_config['extension'])
                
                # Generate and write test files
                test_files = self._generate_test_files(test_definitions, language)
                await self._write_code_files(test_files, temp_dir, f"_test{lang_config['extension']}")
                
                # Initialize project if needed
                if language == 'go':
                    await self._init_go_module(temp_dir)
                
                # Run tests
                test_result = await self._run_tests(temp_dir, lang_config['test_cmd'])
                
                return test_result['pass_rate']
                
            except Exception as e:
                logger.error(f"Unit test execution failed: {e}")
                return 0.0

    async def check_code_formatting(self, solution_code: Dict[str, str], 
                                  language: str = 'go') -> float:
        """Check code formatting compliance"""
        
        if language not in self.supported_languages:
            return 0.5
        
        lang_config = self.supported_languages[language]
        
        if 'fmt_cmd' not in lang_config:
            return 0.5  # No formatter available
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Write code files
                await self._write_code_files(solution_code, temp_dir, lang_config['extension'])
                
                # Run formatter check
                fmt_result = await self._check_formatting(temp_dir, lang_config['fmt_cmd'])
                
                return fmt_result['compliance_score']
                
            except Exception as e:
                logger.error(f"Formatting check failed: {e}")
                return 0.5

    # Helper methods
    
    async def _write_code_files(self, code_files: Dict[str, str], 
                              target_dir: str, extension: str):
        """Write code files to target directory with proper directory structure"""
        
        for filename, code in code_files.items():
            # Normalize the file path by:
            # 1. Remove any leading/trailing slashes
            # 2. Replace multiple slashes with single slashes  
            # 3. Convert to proper Path object
            normalized_filename = filename.strip('/').replace('//', '/').replace('\\\\', '\\')
            
            # Ensure proper extension if not already present
            if not normalized_filename.endswith(extension):
                # Keep the directory structure but fix the extension
                path_parts = normalized_filename.split('/')
                if len(path_parts) > 1:
                    # Has directory structure - fix only the filename
                    path_parts[-1] = f"{Path(path_parts[-1]).stem}{extension}"
                    normalized_filename = '/'.join(path_parts)
                else:
                    # Simple filename - fix directly
                    normalized_filename = f"{Path(normalized_filename).stem}{extension}"
            
            # Create full file path
            file_path = Path(target_dir) / normalized_filename
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file content
            try:
                file_path.write_text(code, encoding='utf-8')
                logger.debug(f"Successfully wrote file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to write file {file_path}: {e}")
                # Create a fallback simple filename
                fallback_name = f"fallback_{Path(normalized_filename).name}"
                fallback_path = Path(target_dir) / fallback_name
                fallback_path.write_text(code, encoding='utf-8')
                logger.info(f"Created fallback file: {fallback_path}")

    async def _init_go_module(self, project_dir: str):
        """Initialize Go module in project directory"""
        
        try:
            result = subprocess.run(
                ['go', 'mod', 'init', 'testproject'],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _compile_code(self, project_dir: str, compile_cmd: List[str]) -> Dict[str, Any]:
        """Compile code and return results"""
        
        try:
            # Get all source files in the project directory
            source_files = []
            file_extensions = {
                'python3': ['.py'],
                'py_compile': ['.py'], 
                'gcc': ['.c'],
                'clang': ['.c'],
                'g++': ['.cpp', '.cc', '.cxx'],
                'clang++': ['.cpp', '.cc', '.cxx'],
                'javac': ['.java'],
                'rustc': ['.rs'],
                'tsc': ['.ts'],
                'node': ['.js'],
                'php': ['.php']
            }
            
            # Determine which compiler we're using
            compiler = compile_cmd[0] if compile_cmd else None
            
            # For each known compiler, find matching source files
            if compiler in file_extensions:
                extensions = file_extensions[compiler]
                for root, dirs, files in os.walk(project_dir):
                    for file in files:
                        if any(file.endswith(ext) for ext in extensions):
                            rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                            source_files.append(rel_path)
            
            # Special handling for specific languages
            if compile_cmd == ['php', '-l']:
                # PHP syntax checking - check each file individually
                if not source_files:
                    return {
                        'success': False,
                        'errors': ['No PHP files found to validate'],
                        'warnings': []
                    }
                
                all_errors = []
                all_warnings = []
                success = True
                
                for php_file in source_files:
                    file_result = subprocess.run(
                        ['php', '-l', php_file],
                        cwd=project_dir,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if file_result.returncode != 0:
                        success = False
                        if file_result.stderr:
                            all_errors.extend(file_result.stderr.split('\n'))
                    
                    if file_result.stderr:
                        all_warnings.extend(self._extract_warnings(file_result.stderr))
                
                return {
                    'success': success,
                    'errors': [e for e in all_errors if e.strip()],
                    'warnings': all_warnings,
                    'stdout': f"Checked {len(source_files)} PHP files"
                }
            
            elif compile_cmd[0] == 'python3' and '-m' in compile_cmd and 'py_compile' in compile_cmd:
                # Python compilation - compile each .py file
                if not source_files:
                    return {
                        'success': False,
                        'errors': ['No Python files found to compile'],
                        'warnings': []
                    }
                
                all_errors = []
                all_warnings = []
                success = True
                
                for py_file in source_files:
                    file_result = subprocess.run(
                        ['python3', '-m', 'py_compile', py_file],
                        cwd=project_dir,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if file_result.returncode != 0:
                        success = False
                        if file_result.stderr:
                            all_errors.extend(file_result.stderr.split('\n'))
                
                return {
                    'success': success,
                    'errors': [e for e in all_errors if e.strip()],
                    'warnings': all_warnings,
                    'stdout': f"Compiled {len(source_files)} Python files"
                }
            
            elif compile_cmd[0] in ['gcc', 'clang', 'g++', 'clang++']:
                # C/C++ compilation - add source files to command
                if not source_files:
                    return {
                        'success': False,
                        'errors': [f'No {compiler} source files found to compile'],
                        'warnings': []
                    }
                
                # Add source files to compile command
                full_cmd = compile_cmd + source_files
                
                result = subprocess.run(
                    full_cmd,
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                return {
                    'success': result.returncode == 0,
                    'errors': result.stderr.split('\n') if result.stderr else [],
                    'warnings': self._extract_warnings(result.stderr or ''),
                    'stdout': result.stdout
                }
            
            elif compile_cmd[0] == 'javac':
                # Java compilation - add .java files to command
                if not source_files:
                    return {
                        'success': False,
                        'errors': ['No Java files found to compile'],
                        'warnings': []
                    }
                
                # Add source files to compile command
                full_cmd = compile_cmd + source_files
                
                result = subprocess.run(
                    full_cmd,
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                return {
                    'success': result.returncode == 0,
                    'errors': result.stderr.split('\n') if result.stderr else [],
                    'warnings': self._extract_warnings(result.stderr or ''),
                    'stdout': result.stdout
                }
            
            elif compile_cmd[0] == 'rustc':
                # Rust compilation - add .rs files to command
                if not source_files:
                    return {
                        'success': False,
                        'errors': ['No Rust files found to compile'],
                        'warnings': []
                    }
                
                # For Rust, usually compile the main file
                main_file = next((f for f in source_files if 'main.rs' in f), source_files[0])
                full_cmd = compile_cmd + [main_file]
                
                result = subprocess.run(
                    full_cmd,
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                return {
                    'success': result.returncode == 0,
                    'errors': result.stderr.split('\n') if result.stderr else [],
                    'warnings': self._extract_warnings(result.stderr or ''),
                    'stdout': result.stdout
                }
            
            elif compile_cmd[0] == 'tsc':
                # TypeScript compilation - add .ts files or use tsconfig
                if not source_files:
                    return {
                        'success': False,
                        'errors': ['No TypeScript files found to compile'],
                        'warnings': []
                    }
                
                # Check if tsconfig.json exists
                tsconfig_path = os.path.join(project_dir, 'tsconfig.json')
                if os.path.exists(tsconfig_path):
                    full_cmd = compile_cmd  # Use existing tsconfig
                else:
                    full_cmd = compile_cmd + source_files
                
                result = subprocess.run(
                    full_cmd,
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                return {
                    'success': result.returncode == 0,
                    'errors': result.stderr.split('\n') if result.stderr else [],
                    'warnings': self._extract_warnings(result.stderr or ''),
                    'stdout': result.stdout
                }
            
            elif compile_cmd[0] == 'node' and '--check' in compile_cmd:
                # Node.js syntax checking - check each .js file
                if not source_files:
                    return {
                        'success': False,
                        'errors': ['No JavaScript files found to check'],
                        'warnings': []
                    }
                
                all_errors = []
                all_warnings = []
                success = True
                
                for js_file in source_files:
                    file_result = subprocess.run(
                        ['node', '--check', js_file],
                        cwd=project_dir,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if file_result.returncode != 0:
                        success = False
                        if file_result.stderr:
                            all_errors.extend(file_result.stderr.split('\n'))
                
                return {
                    'success': success,
                    'errors': [e for e in all_errors if e.strip()],
                    'warnings': all_warnings,
                    'stdout': f"Checked {len(source_files)} JavaScript files"
                }
            
            # For other commands (go build, dotnet build, etc.), use as-is
            else:
                result = subprocess.run(
                    compile_cmd,
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                return {
                    'success': result.returncode == 0,
                    'errors': result.stderr.split('\n') if result.stderr else [],
                    'warnings': self._extract_warnings(result.stderr or ''),
                    'stdout': result.stdout
                }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'errors': ['Compilation timeout'],
                'warnings': []
            }
        except Exception as e:
            return {
                'success': False,
                'errors': [f'Compilation error: {str(e)}'],
                'warnings': []
            }

    def _get_binary_size(self, project_dir: str, language: str) -> Optional[int]:
        """Get size of compiled binary if it exists"""
        
        if language == 'go':
            # Look for compiled binary
            for file in Path(project_dir).iterdir():
                if file.is_file() and not file.suffix:
                    return file.stat().st_size
        
        return None

    def _get_security_patterns(self, language: str) -> Dict[str, List[Dict]]:
        """Get security vulnerability patterns for language"""
        
        patterns = {
            'go': [
                {
                    'pattern': r'exec\.Command\(',
                    'severity': 'high',
                    'description': 'Command execution vulnerability'
                },
                {
                    'pattern': r'http\.Get\(',
                    'severity': 'medium',
                    'description': 'Unvalidated HTTP request'
                },
                {
                    'pattern': r'sql\.Query\(',
                    'severity': 'high',
                    'description': 'Potential SQL injection'
                },
                {
                    'pattern': r'fmt\.Sprintf.*%s',
                    'severity': 'low',
                    'description': 'String formatting without validation'
                }
            ],
            'python': [
                {
                    'pattern': r'eval\(',
                    'severity': 'high',
                    'description': 'Code injection vulnerability'
                },
                {
                    'pattern': r'exec\(',
                    'severity': 'high',
                    'description': 'Code execution vulnerability'
                },
                {
                    'pattern': r'pickle\.loads?\(',
                    'severity': 'high',
                    'description': 'Unsafe deserialization'
                }
            ]
        }
        
        return patterns.get(language, [])

    def _scan_security_patterns(self, code: str, patterns: List[Dict], 
                              filename: str) -> List[Dict]:
        """Scan code for security vulnerability patterns"""
        
        vulnerabilities = []
        
        for pattern_info in patterns:
            matches = re.finditer(pattern_info['pattern'], code, re.MULTILINE)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                
                vulnerabilities.append({
                    'type': 'security_pattern',
                    'severity': pattern_info['severity'],
                    'description': pattern_info['description'],
                    'file': filename,
                    'line': line_num,
                    'code_snippet': self._get_code_snippet(code, match.start())
                })
        
        return vulnerabilities

    def _calculate_risk_level(self, security_score: float, vuln_count: int) -> str:
        """Calculate overall risk level"""
        
        if security_score >= 0.8 and vuln_count == 0:
            return 'low'
        elif security_score >= 0.6 and vuln_count <= 2:
            return 'medium'
        else:
            return 'high'

    def _calculate_complexity(self, code: str, language: str) -> float:
        """Calculate cyclomatic complexity"""
        
        # Simplified complexity calculation
        complexity_indicators = {
            'go': ['if ', 'for ', 'switch ', 'case ', 'func '],
            'python': ['if ', 'for ', 'while ', 'def ', 'class '],
            'javascript': ['if ', 'for ', 'while ', 'function ', 'switch ']
        }
        
        indicators = complexity_indicators.get(language, [])
        complexity_count = sum(code.count(indicator) for indicator in indicators)
        
        # Normalize to 0-1 scale (assuming 20+ indicators = complex)
        normalized_complexity = min(complexity_count / 20.0, 1.0)
        
        # Return inverse (lower complexity = higher score)
        return 1.0 - normalized_complexity

    def _calculate_maintainability(self, code: str, language: str) -> float:
        """Calculate maintainability index"""
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Simple maintainability factors
        factors = {
            'line_count': len(non_empty_lines),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0,
            'comment_ratio': sum(1 for line in lines if line.strip().startswith(('//','#'))) / len(lines) if lines else 0
        }
        
        # Simple scoring (could be much more sophisticated)
        score = 1.0
        
        # Penalize very long files
        if factors['line_count'] > 100:
            score *= 0.8
        
        # Penalize very long lines
        if factors['avg_line_length'] > 80:
            score *= 0.9
        
        # Reward comments
        score += factors['comment_ratio'] * 0.2
        
        return min(score, 1.0)

    def _detect_code_smells(self, code: str, language: str, filename: str) -> List[Dict]:
        """Detect code smells and anti-patterns"""
        
        smells = []
        lines = code.split('\n')
        
        # Long method detection
        current_function_length = 0
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ['func ', 'def ', 'function ']):
                current_function_length = 1
            elif current_function_length > 0:
                if line.strip():
                    current_function_length += 1
                if line.strip() == '}' or (language == 'python' and not line.startswith(' ')):
                    if current_function_length > 20:
                        smells.append({
                            'type': 'long_method',
                            'severity': 'medium',
                            'description': f'Function is {current_function_length} lines long',
                            'file': filename,
                            'line': i + 1
                        })
                    current_function_length = 0
        
        # Duplicate code detection (simplified)
        line_counts = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10:  # Only check meaningful lines
                if stripped in line_counts:
                    line_counts[stripped].append(i + 1)
                else:
                    line_counts[stripped] = [i + 1]
        
        for line_content, line_numbers in line_counts.items():
            if len(line_numbers) > 2:
                smells.append({
                    'type': 'duplicate_code',
                    'severity': 'low',
                    'description': f'Line repeated {len(line_numbers)} times',
                    'file': filename,
                    'lines': line_numbers
                })
        
        return smells

    def _estimate_test_coverage(self, solution_code: Dict[str, str], language: str) -> float:
        """Estimate test coverage (simplified)"""
        
        # Look for test files
        test_files = [f for f in solution_code.keys() if 'test' in f.lower()]
        
        if not test_files:
            return 0.0
        
        # Simple heuristic: ratio of test lines to code lines
        test_lines = sum(len(solution_code[f].split('\n')) for f in test_files)
        code_lines = sum(len(code.split('\n')) for f, code in solution_code.items() if f not in test_files)
        
        if code_lines == 0:
            return 0.0
        
        coverage_ratio = test_lines / code_lines
        return min(coverage_ratio, 1.0)

    def _generate_test_files(self, test_definitions: List[Dict], language: str) -> Dict[str, str]:
        """Generate test files from test definitions"""
        
        # This is a simplified implementation
        # In practice, would generate more sophisticated tests
        
        if language == 'go':
            test_code = """package main

import "testing"

func TestBasicFunctionality(t *testing.T) {
    // Basic test generated from definitions
    result := main()
    if result == nil {
        t.Error("Expected non-nil result")
    }
}
"""
            return {"main_test.go": test_code}
        
        return {}

    async def _run_tests(self, project_dir: str, test_cmd: List[str]) -> Dict[str, Any]:
        """Run tests and return results"""
        
        try:
            result = subprocess.run(
                test_cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse test output to calculate pass rate
            pass_rate = self._parse_test_results(result.stdout, result.stderr)
            
            return {
                'success': result.returncode == 0,
                'pass_rate': pass_rate,
                'output': result.stdout,
                'errors': result.stderr
            }
            
        except Exception as e:
            return {
                'success': False,
                'pass_rate': 0.0,
                'output': '',
                'errors': str(e)
            }

    def _parse_test_results(self, stdout: str, stderr: str) -> float:
        """Parse test output to determine pass rate"""
        
        # Simplified test result parsing
        # In practice, would parse specific test framework outputs
        
        if 'PASS' in stdout or 'ok' in stdout:
            return 1.0
        elif 'FAIL' in stdout or 'error' in stderr:
            return 0.0
        else:
            return 0.5

    async def _check_formatting(self, project_dir: str, fmt_cmd: List[str]) -> Dict[str, Any]:
        """Check code formatting compliance"""
        
        try:
            result = subprocess.run(
                fmt_cmd,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Calculate compliance score based on formatter output
            compliance_score = 1.0 if result.returncode == 0 else 0.7
            
            return {
                'compliance_score': compliance_score,
                'issues': result.stderr.split('\n') if result.stderr else []
            }
            
        except Exception as e:
            return {
                'compliance_score': 0.5,
                'issues': [str(e)]
            }

    def _extract_warnings(self, stderr: str) -> List[str]:
        """Extract warnings from compiler output"""
        
        warnings = []
        for line in stderr.split('\n'):
            if 'warning' in line.lower():
                warnings.append(line.strip())
        
        return warnings

    def _get_code_snippet(self, code: str, position: int, context_lines: int = 2) -> str:
        """Get code snippet around a specific position"""
        
        lines = code.split('\n')
        line_num = code[:position].count('\n')
        
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines + 1)
        
        snippet_lines = lines[start:end]
        return '\n'.join(snippet_lines)


# Convenience functions
async def validate_code_compilation(solution_code: Dict[str, str], language: str = 'go') -> CompilationResult:
    """Validate code compilation"""
    validator = CodeValidator()
    return await validator.validate_compilation(solution_code, language)

async def analyze_code_security(solution_code: Dict[str, str], language: str = 'go') -> SecurityAnalysisResult:
    """Analyze code security"""
    validator = CodeValidator()
    return await validator.analyze_security(solution_code, language)

async def analyze_code_quality(solution_code: Dict[str, str], language: str = 'go') -> QualityAnalysisResult:
    """Analyze code quality"""
    validator = CodeValidator()
    return await validator.analyze_code_quality(solution_code, language) 