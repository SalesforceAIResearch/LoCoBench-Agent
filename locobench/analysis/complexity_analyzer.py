"""
Code complexity analysis utilities for LoCoBench

This module provides comprehensive code complexity analysis using
multiple metrics and tools including radon, lizard, and custom metrics.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import tempfile
import os

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit, h_visit
    from radon.raw import analyze
    import lizard
    ANALYSIS_TOOLS_AVAILABLE = True
except ImportError:
    ANALYSIS_TOOLS_AVAILABLE = False
    logging.warning("Analysis tools (radon/lizard) not available. Using simplified metrics.")

logger = logging.getLogger(__name__)


class ComplexityAnalyzer:
    """Analyzes code complexity using multiple metrics"""
    
    def __init__(self):
        self.tools_available = ANALYSIS_TOOLS_AVAILABLE
        
    def analyze_complexity_from_content(self, content: str, language: str = "python") -> Dict[str, Any]:
        """Analyze complexity directly from code content"""
        if not content or not content.strip():
            return self._empty_result()
        
        try:
            if self.tools_available:
                return self._analyze_with_tools(content, language)
            else:
                return self._analyze_simple(content, language)
                
        except Exception as e:
            logger.debug(f"Complexity analysis failed: {e}")
            return self._fallback_analysis(content)
    
    def analyze_complexity(self, file_path: str) -> Dict[str, Any]:
        """Analyze complexity from file path"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return self._empty_result()
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = self._detect_language(file_path)
            return self.analyze_complexity_from_content(content, language)
            
        except Exception as e:
            logger.debug(f"Failed to analyze file {file_path}: {e}")
            return self._empty_result()
    
    def _analyze_with_tools(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze using radon and lizard tools"""
        results = {
            'cyclomatic_complexity': 0.0,
            'maintainability_index': 0.0,
            'halstead_difficulty': 0.0,
            'lines_of_code': 0,
            'logical_lines': 0,
            'comment_lines': 0,
            'function_count': 0,
            'class_count': 0,
            'nloc': 0,
            'language': language
        }
        
        # Only use radon for Python code
        if language in ['python', 'py']:
            try:
                # Cyclomatic Complexity
                cc_results = cc_visit(content)
                if cc_results:
                    avg_complexity = sum(item.complexity for item in cc_results) / len(cc_results)
                    results['cyclomatic_complexity'] = avg_complexity
                    results['function_count'] = len(cc_results)
                
                # Maintainability Index
                mi_results = mi_visit(content, multi=True)
                if mi_results:
                    results['maintainability_index'] = mi_results
                
                # Halstead metrics
                h_results = h_visit(content)
                if h_results:
                    results['halstead_difficulty'] = getattr(h_results, 'difficulty', 0.0)
                
                # Raw metrics
                raw_results = analyze(content)
                if raw_results:
                    results['lines_of_code'] = raw_results.loc
                    results['logical_lines'] = raw_results.lloc
                    results['comment_lines'] = raw_results.comments
                    results['nloc'] = raw_results.lloc
                    
            except Exception as e:
                logger.debug(f"Radon analysis failed: {e}")
        
        # Use lizard for all languages
        try:
            # Create temporary file for lizard analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                lizard_results = lizard.analyze_file(tmp_path)
                
                if lizard_results:
                    # Average complexity across functions
                    if lizard_results.function_list:
                        avg_complexity = sum(func.cyclomatic_complexity for func in lizard_results.function_list) / len(lizard_results.function_list)
                        results['cyclomatic_complexity'] = max(results['cyclomatic_complexity'], avg_complexity)
                        results['function_count'] = max(results['function_count'], len(lizard_results.function_list))
                    
                    results['nloc'] = max(results['nloc'], lizard_results.nloc)
                    results['lines_of_code'] = max(results['lines_of_code'], lizard_results.nloc)
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Lizard analysis failed: {e}")
        
        # Fall back to simple analysis if tools failed
        if results['cyclomatic_complexity'] == 0.0:
            simple_results = self._analyze_simple(content, language)
            results.update(simple_results)
        
        return results
    
    def _analyze_simple(self, content: str, language: str) -> Dict[str, Any]:
        """Simple complexity analysis without external tools"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Count various complexity indicators
        complexity_indicators = self._count_complexity_indicators(content, language)
        
        # Estimate cyclomatic complexity
        cyclomatic_complexity = 1 + complexity_indicators['control_structures']
        
        # Simple maintainability estimate
        lines_factor = min(len(non_empty_lines) / 100.0, 2.0)  # Penalize very long files
        complexity_factor = min(cyclomatic_complexity / 10.0, 2.0)  # Penalize high complexity
        maintainability = max(0.0, 100.0 - (lines_factor + complexity_factor) * 20)
        
        return {
            'cyclomatic_complexity': cyclomatic_complexity,
            'maintainability_index': maintainability,
            'halstead_difficulty': complexity_indicators['operators'] + complexity_indicators['operands'],
            'lines_of_code': len(lines),
            'logical_lines': len(non_empty_lines),
            'comment_lines': complexity_indicators['comments'],
            'function_count': complexity_indicators['functions'],
            'class_count': complexity_indicators['classes'],
            'nloc': len(non_empty_lines),
            'language': language
        }
    
    def _count_complexity_indicators(self, content: str, language: str) -> Dict[str, int]:
        """Count various complexity indicators in code"""
        indicators = {
            'control_structures': 0,
            'functions': 0,
            'classes': 0,
            'comments': 0,
            'operators': 0,
            'operands': 0
        }
        
        # Language-specific patterns
        patterns = self._get_language_patterns(language)
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Count control structures
            for pattern in patterns['control_structures']:
                indicators['control_structures'] += len(re.findall(pattern, line, re.IGNORECASE))
            
            # Count functions
            for pattern in patterns['functions']:
                indicators['functions'] += len(re.findall(pattern, line, re.IGNORECASE))
            
            # Count classes
            for pattern in patterns['classes']:
                indicators['classes'] += len(re.findall(pattern, line, re.IGNORECASE))
            
            # Count comments
            for pattern in patterns['comments']:
                if re.search(pattern, line):
                    indicators['comments'] += 1
                    break
            
            # Count operators and operands (simplified)
            operators = re.findall(r'[+\-*/%=<>!&|^~]', line)
            indicators['operators'] += len(operators)
            
            # Count identifiers as operands
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
            indicators['operands'] += len(identifiers)
        
        return indicators
    
    def _get_language_patterns(self, language: str) -> Dict[str, List[str]]:
        """Get regex patterns for different language constructs"""
        
        # Python patterns
        python_patterns = {
            'control_structures': [
                r'\bif\b', r'\belif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
                r'\btry\b', r'\bexcept\b', r'\bfinally\b', r'\bwith\b',
                r'\band\b', r'\bor\b', r'\bnot\b'
            ],
            'functions': [r'\bdef\s+\w+', r'\blambda\b'],
            'classes': [r'\bclass\s+\w+'],
            'comments': [r'^\s*#', r'""".*"""', r"'''.*'''"]
        }
        
        # JavaScript/TypeScript patterns
        js_patterns = {
            'control_structures': [
                r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b',
                r'\bcase\b', r'\btry\b', r'\bcatch\b', r'\bfinally\b',
                r'&&', r'\|\|', r'!', r'\?.*:'
            ],
            'functions': [r'\bfunction\s+\w+', r'=>', r'\w+\s*:\s*function'],
            'classes': [r'\bclass\s+\w+'],
            'comments': [r'^\s*//', r'/\*.*\*/']
        }
        
        # Java patterns
        java_patterns = {
            'control_structures': [
                r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b',
                r'\bcase\b', r'\btry\b', r'\bcatch\b', r'\bfinally\b',
                r'&&', r'\|\|', r'!', r'\?.*:'
            ],
            'functions': [r'\b\w+\s+\w+\s*\(', r'\bpublic\s+\w+', r'\bprivate\s+\w+'],
            'classes': [r'\bclass\s+\w+', r'\binterface\s+\w+'],
            'comments': [r'^\s*//', r'/\*.*\*/']
        }
        
        # C++ patterns
        cpp_patterns = {
            'control_structures': [
                r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b',
                r'\bcase\b', r'\btry\b', r'\bcatch\b',
                r'&&', r'\|\|', r'!', r'\?.*:'
            ],
            'functions': [r'\b\w+\s+\w+\s*\(', r'\b\w+::\w+'],
            'classes': [r'\bclass\s+\w+', r'\bstruct\s+\w+'],
            'comments': [r'^\s*//', r'/\*.*\*/']
        }
        
        # Go patterns
        go_patterns = {
            'control_structures': [
                r'\bif\b', r'\belse\b', r'\bfor\b', r'\bswitch\b',
                r'\bcase\b', r'\bselect\b',
                r'&&', r'\|\|', r'!'
            ],
            'functions': [r'\bfunc\s+\w+', r'\bfunc\s*\('],
            'classes': [r'\btype\s+\w+\s+struct', r'\binterface\s*{'],
            'comments': [r'^\s*//', r'/\*.*\*/']
        }
        
        # Map languages to patterns
        pattern_map = {
            'python': python_patterns,
            'py': python_patterns,
            'javascript': js_patterns,
            'js': js_patterns,
            'typescript': js_patterns,
            'ts': js_patterns,
            'java': java_patterns,
            'cpp': cpp_patterns,
            'c++': cpp_patterns,
            'cc': cpp_patterns,
            'go': go_patterns
        }
        
        return pattern_map.get(language.lower(), python_patterns)  # Default to Python
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.go': 'go'
        }
        
        extension = file_path.suffix.lower()
        return extension_map.get(extension, 'unknown')
    
    def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback analysis when everything else fails"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Very simple complexity estimation
        complexity = 1 + content.count('if') + content.count('for') + content.count('while')
        
        return {
            'cyclomatic_complexity': float(complexity),
            'maintainability_index': max(0.0, 100.0 - len(non_empty_lines) / 10.0),
            'halstead_difficulty': len(non_empty_lines) / 10.0,
            'lines_of_code': len(lines),
            'logical_lines': len(non_empty_lines),
            'comment_lines': content.count('#') + content.count('//'),
            'function_count': content.count('def ') + content.count('function '),
            'class_count': content.count('class '),
            'nloc': len(non_empty_lines),
            'language': 'unknown'
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty/default result"""
        return {
            'cyclomatic_complexity': 0.0,
            'maintainability_index': 0.0,
            'halstead_difficulty': 0.0,
            'lines_of_code': 0,
            'logical_lines': 0,
            'comment_lines': 0,
            'function_count': 0,
            'class_count': 0,
            'nloc': 0,
            'language': 'unknown'
        }
    
    def calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate normalized complexity score (0-1 scale)"""
        if not metrics:
            return 0.0
        
        # Normalize different metrics to 0-1 scale
        cyclomatic = min(metrics.get('cyclomatic_complexity', 1.0) / 20.0, 1.0)
        maintainability = max(0.0, 1.0 - metrics.get('maintainability_index', 100.0) / 100.0)
        halstead = min(metrics.get('halstead_difficulty', 1.0) / 50.0, 1.0)
        size_factor = min(metrics.get('nloc', 1) / 1000.0, 1.0)
        
        # Weighted combination
        complexity_score = (
            cyclomatic * 0.4 +
            maintainability * 0.3 +
            halstead * 0.2 +
            size_factor * 0.1
        )
        
        return float(complexity_score)
    
    def analyze_repository_complexity(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze complexity across an entire repository"""
        all_metrics = []
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        for file_info in files:
            content = file_info.get('content', '')
            if not content:
                continue
            
            file_metrics = self.analyze_complexity_from_content(content)
            all_metrics.append(file_metrics)
            
            total_lines += file_metrics.get('nloc', 0)
            total_functions += file_metrics.get('function_count', 0)
            total_classes += file_metrics.get('class_count', 0)
        
        if not all_metrics:
            return self._empty_result()
        
        # Calculate aggregated metrics
        avg_complexity = sum(m.get('cyclomatic_complexity', 0) for m in all_metrics) / len(all_metrics)
        avg_maintainability = sum(m.get('maintainability_index', 0) for m in all_metrics) / len(all_metrics)
        avg_halstead = sum(m.get('halstead_difficulty', 0) for m in all_metrics) / len(all_metrics)
        
        repository_metrics = {
            'average_cyclomatic_complexity': avg_complexity,
            'average_maintainability_index': avg_maintainability,
            'average_halstead_difficulty': avg_halstead,
            'total_lines_of_code': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'file_count': len(all_metrics),
            'complexity_distribution': {
                'low': sum(1 for m in all_metrics if m.get('cyclomatic_complexity', 0) < 5),
                'medium': sum(1 for m in all_metrics if 5 <= m.get('cyclomatic_complexity', 0) < 15),
                'high': sum(1 for m in all_metrics if m.get('cyclomatic_complexity', 0) >= 15)
            }
        }
        
        # Calculate overall repository complexity score
        repository_metrics['complexity_score'] = self.calculate_complexity_score({
            'cyclomatic_complexity': avg_complexity,
            'maintainability_index': avg_maintainability,
            'halstead_difficulty': avg_halstead,
            'nloc': total_lines
        })
        
        return repository_metrics 