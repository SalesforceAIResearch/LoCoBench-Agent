"""
AST analysis utilities for LoCoBench

This module provides AST parsing and analysis capabilities for code quality metrics
and symbol extraction across different programming languages.
"""

import ast
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ASTAnalyzer:
    """AST analyzer with support for multiple languages"""
    
    def __init__(self, language: str = "python"):
        self.language = language.lower()
        self.supported_languages = {"python", "go", "javascript", "typescript"}
        
        if self.language not in self.supported_languages:
            logger.warning(f"Language '{language}' not fully supported. Using basic analysis.")
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a file and return comprehensive AST information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return self._empty_ast_result()
    
    def parse_content(self, content: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Parse content string and return AST information"""
        if not content.strip():
            return self._empty_ast_result()
        
        if self.language == "python":
            return self._parse_python(content, file_path)
        elif self.language == "go":
            return self._parse_go(content, file_path)
        elif self.language in ["javascript", "typescript"]:
            return self._parse_javascript(content, file_path)
        else:
            return self._parse_generic(content, file_path)
    
    def extract_symbols(self, file_path: str) -> List[str]:
        """Extract all symbols (functions, classes, variables) from a file"""
        ast_info = self.parse_file(file_path)
        symbols = []
        
        # Collect all symbol types
        symbols.extend(ast_info.get("functions", []))
        symbols.extend(ast_info.get("classes", []))
        symbols.extend(ast_info.get("variables", []))
        symbols.extend(ast_info.get("types", []))
        
        return list(set(symbols))  # Remove duplicates
    
    def calculate_complexity(self, file_path: str) -> float:
        """Calculate cyclomatic complexity of a file"""
        ast_info = self.parse_file(file_path)
        return ast_info.get("complexity", 1.0)
    
    def extract_dependencies(self, file_path: str) -> List[str]:
        """Extract import/dependency information"""
        ast_info = self.parse_file(file_path)
        return ast_info.get("imports", [])
    
    def get_function_signatures(self, file_path: str) -> List[Dict[str, Any]]:
        """Get detailed function signature information"""
        ast_info = self.parse_file(file_path)
        return ast_info.get("function_details", [])
    
    # Language-specific parsers
    
    def _parse_python(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse Python code using ast module"""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            variables = []
            function_details = []
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    function_details.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line_number": node.lineno,
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    })
                    
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}" if module else alias.name)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)
                
                # Calculate complexity (simplified McCabe)
                elif isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    complexity += 1
                elif isinstance(node, (ast.And, ast.Or)):
                    complexity += 1
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "variables": variables,
                "function_details": function_details,
                "complexity": complexity,
                "total_lines": len(content.split('\n')),
                "language": "python"
            }
            
        except SyntaxError as e:
            logger.error(f"Python syntax error in {file_path}: {e}")
            return self._empty_ast_result()
        except Exception as e:
            logger.error(f"Error parsing Python code in {file_path}: {e}")
            return self._empty_ast_result()
    
    def _parse_go(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse Go code using regex patterns"""
        functions = []
        types = []
        imports = []
        variables = []
        function_details = []
        complexity = 1
        
        lines = content.split('\n')
        
        # Extract functions
        func_pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)'
        for line_num, line in enumerate(lines, 1):
            if match := re.search(func_pattern, line):
                func_name = match.group(1)
                functions.append(func_name)
                
                # Extract parameters
                param_match = re.search(r'func\s+(?:\([^)]*\)\s+)?\w+\s*\(([^)]*)\)', line)
                params = []
                if param_match and param_match.group(1).strip():
                    param_str = param_match.group(1)
                    # Simple parameter parsing
                    params = [p.strip().split()[-1] for p in param_str.split(',') if p.strip()]
                
                function_details.append({
                    "name": func_name,
                    "args": params,
                    "line_number": line_num,
                    "is_method": "(" in line and ")" in line and line.index("(") < line.index("func")
                })
        
        # Extract types (structs, interfaces)
        type_pattern = r'type\s+(\w+)\s+(?:struct|interface)'
        for line in lines:
            if match := re.search(type_pattern, line):
                types.append(match.group(1))
        
        # Extract imports
        import_pattern = r'import\s+(?:"([^"]+)"|`([^`]+)`)'
        for line in lines:
            if match := re.search(import_pattern, line):
                import_path = match.group(1) or match.group(2)
                imports.append(import_path)
        
        # Extract variables (simplified)
        var_pattern = r'(?:var|:=)\s+(\w+)'
        for line in lines:
            if match := re.search(var_pattern, line):
                variables.append(match.group(1))
        
        # Calculate complexity
        complexity_keywords = ['if', 'for', 'switch', 'case', 'select']
        for line in lines:
            for keyword in complexity_keywords:
                if re.search(rf'\b{keyword}\b', line):
                    complexity += 1
        
        return {
            "functions": functions,
            "types": types,
            "imports": imports,
            "variables": variables,
            "function_details": function_details,
            "complexity": complexity,
            "total_lines": len(lines),
            "language": "go"
        }
    
    def _parse_javascript(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse JavaScript/TypeScript code using regex patterns"""
        functions = []
        classes = []
        imports = []
        variables = []
        function_details = []
        complexity = 1
        
        lines = content.split('\n')
        
        # Extract functions (including arrow functions)
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
            r'(\w+)\s*:\s*(?:async\s+)?function\s*\(',
            r'(\w+)\s*\([^)]*\)\s*\{'
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern in func_patterns:
                if match := re.search(pattern, line):
                    func_name = match.group(1)
                    functions.append(func_name)
                    function_details.append({
                        "name": func_name,
                        "line_number": line_num,
                        "is_async": "async" in line
                    })
                    break
        
        # Extract classes
        class_pattern = r'class\s+(\w+)'
        for line in lines:
            if match := re.search(class_pattern, line):
                classes.append(match.group(1))
        
        # Extract imports/requires
        import_patterns = [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        for line in lines:
            for pattern in import_patterns:
                if match := re.search(pattern, line):
                    imports.append(match.group(1))
        
        # Extract variables
        var_pattern = r'(?:const|let|var)\s+(\w+)'
        for line in lines:
            if match := re.search(var_pattern, line):
                variables.append(match.group(1))
        
        # Calculate complexity
        complexity_keywords = ['if', 'while', 'for', 'switch', 'case', 'catch']
        for line in lines:
            for keyword in complexity_keywords:
                if re.search(rf'\b{keyword}\b', line):
                    complexity += 1
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "variables": variables,
            "function_details": function_details,
            "complexity": complexity,
            "total_lines": len(lines),
            "language": self.language
        }
    
    def _parse_generic(self, content: str, file_path: str) -> Dict[str, Any]:
        """Generic parsing for unsupported languages"""
        lines = content.split('\n')
        
        # Very basic symbol extraction
        symbols = []
        word_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        for line in lines:
            words = re.findall(word_pattern, line)
            symbols.extend(words)
        
        # Remove common keywords and duplicates
        common_keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'class', 'function', 'def', 'return', 'import', 'from', 'as',
            'true', 'false', 'null', 'undefined', 'this', 'self'
        }
        
        unique_symbols = list(set(symbols) - common_keywords)
        
        return {
            "symbols": unique_symbols[:50],  # Limit to first 50 unique symbols
            "total_lines": len(lines),
            "complexity": max(1, len(lines) // 10),  # Rough complexity estimate
            "language": self.language
        }
    
    def _empty_ast_result(self) -> Dict[str, Any]:
        """Return empty AST result structure"""
        return {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "function_details": [],
            "complexity": 1.0,
            "total_lines": 0,
            "language": self.language
        }


def analyze_code_structure(file_path: str, language: str = None) -> Dict[str, Any]:
    """Convenience function to analyze code structure"""
    if language is None:
        # Detect language from file extension
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.go': 'go',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript'
        }
        language = language_map.get(ext, 'generic')
    
    analyzer = ASTAnalyzer(language)
    return analyzer.parse_file(file_path) 