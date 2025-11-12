"""
Dependency analysis utilities for LoCoBench

This module provides dependency analysis capabilities for understanding
code relationships, import patterns, and dependency graphs.
"""

import re
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analyzes dependencies between files and modules"""
    
    def __init__(self, language: str = "auto"):
        self.language = language.lower() if language != "auto" else None
        self.supported_languages = {"python", "go", "javascript", "typescript"}
    
    def analyze_dependencies(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Analyze dependencies between files"""
        dependencies = {}
        
        for file_path in file_paths:
            deps = self._extract_file_dependencies(file_path)
            dependencies[file_path] = deps
        
        # Resolve internal dependencies (files depending on each other)
        resolved_deps = self._resolve_internal_dependencies(dependencies, file_paths)
        
        return resolved_deps
    
    def build_dependency_graph(self, dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build dependency graph with nodes and edges"""
        nodes = set()
        edges = []
        
        # Collect all nodes
        for file_path, deps in dependencies.items():
            nodes.add(file_path)
            nodes.update(deps)
        
        # Create edges
        for file_path, deps in dependencies.items():
            for dep in deps:
                edges.append({"from": file_path, "to": dep, "type": "imports"})
        
        return {
            "nodes": [{"id": node, "label": Path(node).name} for node in nodes],
            "edges": edges,
            "metrics": self._calculate_graph_metrics(dependencies)
        }
    
    def find_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Find circular dependencies in the dependency graph"""
        def dfs(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        visited = set()
        cycles = []
        
        for node in dependencies:
            if node not in visited:
                cycle = dfs(node, visited, set(), [])
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def analyze_dependency_depth(self, dependencies: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate dependency depth for each file"""
        depths = {}
        
        def calculate_depth(file_path, visited):
            if file_path in visited:
                return float('inf')  # Circular dependency
            if file_path in depths:
                return depths[file_path]
            
            visited.add(file_path)
            
            deps = dependencies.get(file_path, [])
            if not deps:
                depths[file_path] = 0
            else:
                max_depth = 0
                for dep in deps:
                    depth = calculate_depth(dep, visited.copy())
                    max_depth = max(max_depth, depth)
                depths[file_path] = max_depth + 1
            
            return depths[file_path]
        
        for file_path in dependencies:
            calculate_depth(file_path, set())
        
        return depths
    
    def _extract_file_dependencies(self, file_path: str) -> List[str]:
        """Extract dependencies from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Detect language if auto
        language = self._detect_language(file_path) if self.language is None else self.language
        
        if language == "python":
            return self._extract_python_dependencies(content)
        elif language == "go":
            return self._extract_go_dependencies(content)
        elif language in ["javascript", "typescript"]:
            return self._extract_javascript_dependencies(content)
        else:
            return self._extract_generic_dependencies(content)
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.go': 'go',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript'
        }
        return language_map.get(ext, 'generic')
    
    def _extract_python_dependencies(self, content: str) -> List[str]:
        """Extract Python import dependencies"""
        dependencies = []
        
        # Regular import patterns
        import_patterns = [
            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import',
            r'^\s*from\s+\.+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in import_patterns:
                match = re.search(pattern, line)
                if match:
                    module = match.group(1)
                    dependencies.append(module)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_go_dependencies(self, content: str) -> List[str]:
        """Extract Go import dependencies"""
        dependencies = []
        
        # Extract import blocks
        import_patterns = [
            r'import\s+"([^"]+)"',
            r'import\s+`([^`]+)`',
            r'import\s+\(\s*([^)]+)\s*\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if '(' in pattern:  # Multi-line import block
                    # Parse multi-line imports
                    import_lines = match.split('\n')
                    for line in import_lines:
                        import_match = re.search(r'["`]([^"`]+)["`]', line)
                        if import_match:
                            dependencies.append(import_match.group(1))
                else:
                    dependencies.append(match)
        
        return list(set(dependencies))
    
    def _extract_javascript_dependencies(self, content: str) -> List[str]:
        """Extract JavaScript/TypeScript dependencies"""
        dependencies = []
        
        # Import patterns for ES6, CommonJS, and dynamic imports
        import_patterns = [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'import\s+[\'"]([^\'"]+)[\'"]'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _extract_generic_dependencies(self, content: str) -> List[str]:
        """Extract dependencies using generic patterns"""
        dependencies = []
        
        # Look for common import-like patterns
        generic_patterns = [
            r'#include\s*[<"]([^>"]+)[>"]',  # C/C++
            r'using\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',  # C#
            r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',  # Generic import
        ]
        
        for pattern in generic_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _resolve_internal_dependencies(self, dependencies: Dict[str, List[str]], 
                                     file_paths: List[str]) -> Dict[str, List[str]]:
        """Resolve dependencies to internal files where possible"""
        resolved = {}
        
        # Create mapping of module names to file paths
        file_map = {}
        for file_path in file_paths:
            path_obj = Path(file_path)
            # Map by filename (without extension)
            module_name = path_obj.stem
            file_map[module_name] = file_path
            
            # Map by relative path
            rel_path = str(path_obj.with_suffix(''))
            file_map[rel_path] = file_path
        
        for file_path, deps in dependencies.items():
            resolved_deps = []
            for dep in deps:
                # Try to resolve to internal files
                if dep in file_map:
                    resolved_deps.append(file_map[dep])
                else:
                    # Check if it's a relative path reference
                    dep_path = Path(dep)
                    if dep_path.stem in file_map:
                        resolved_deps.append(file_map[dep_path.stem])
                    else:
                        # Keep as external dependency
                        resolved_deps.append(dep)
            
            resolved[file_path] = resolved_deps
        
        return resolved
    
    def _calculate_graph_metrics(self, dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate metrics for the dependency graph"""
        total_files = len(dependencies)
        total_dependencies = sum(len(deps) for deps in dependencies.values())
        
        # Calculate fan-out (number of dependencies per file)
        fan_out = {file_path: len(deps) for file_path, deps in dependencies.items()}
        
        # Calculate fan-in (number of files depending on each file)
        fan_in = defaultdict(int)
        for file_path, deps in dependencies.items():
            for dep in deps:
                fan_in[dep] += 1
        
        # Find files with highest fan-in/fan-out
        max_fan_out = max(fan_out.values()) if fan_out else 0
        max_fan_in = max(fan_in.values()) if fan_in else 0
        
        return {
            "total_files": total_files,
            "total_dependencies": total_dependencies,
            "average_dependencies_per_file": total_dependencies / total_files if total_files > 0 else 0,
            "max_fan_out": max_fan_out,
            "max_fan_in": max_fan_in,
            "fan_out_distribution": dict(fan_out),
            "fan_in_distribution": dict(fan_in),
            "circular_dependencies": len(self.find_circular_dependencies(dependencies))
        }


def analyze_project_dependencies(project_dir: str, language: str = "auto") -> Dict[str, Any]:
    """Convenience function to analyze dependencies for an entire project"""
    
    # Find all source files
    project_path = Path(project_dir)
    
    if language == "auto":
        # Auto-detect based on files present
        extensions = {'.py', '.go', '.js', '.ts', '.jsx', '.tsx'}
    else:
        ext_map = {
            'python': {'.py'},
            'go': {'.go'},
            'javascript': {'.js', '.jsx'},
            'typescript': {'.ts', '.tsx'}
        }
        extensions = ext_map.get(language, {'.py'})
    
    file_paths = []
    for ext in extensions:
        file_paths.extend(str(p) for p in project_path.rglob(f"*{ext}"))
    
    if not file_paths:
        return {"error": "No source files found"}
    
    # Analyze dependencies
    analyzer = DependencyAnalyzer(language)
    dependencies = analyzer.analyze_dependencies(file_paths)
    graph = analyzer.build_dependency_graph(dependencies)
    cycles = analyzer.find_circular_dependencies(dependencies)
    depths = analyzer.analyze_dependency_depth(dependencies)
    
    return {
        "dependencies": dependencies,
        "graph": graph,
        "circular_dependencies": cycles,
        "dependency_depths": depths,
        "analysis_summary": {
            "total_files": len(file_paths),
            "files_with_dependencies": len([f for f, deps in dependencies.items() if deps]),
            "circular_dependency_count": len(cycles),
            "max_dependency_depth": max(depths.values()) if depths else 0
        }
    } 