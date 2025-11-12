"""
Search Tools for LoCoBench-Agent

This module provides advanced code search and navigation capabilities
for agents working with large codebases, including semantic search,
symbol lookup, and dependency analysis.
"""

import ast
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from ..core.tool_registry import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of code search operations"""
    TEXT_SEARCH = "text_search"
    REGEX_SEARCH = "regex_search"
    SYMBOL_SEARCH = "symbol_search"
    SEMANTIC_SEARCH = "semantic_search"
    DEPENDENCY_SEARCH = "dependency_search"
    USAGE_SEARCH = "usage_search"


@dataclass
class SearchResult:
    """Represents a search result"""
    
    file_path: str
    line_number: int
    column: int
    match_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    
    # Additional metadata
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    match_type: Optional[str] = None
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "match_text": self.match_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "match_type": self.match_type,
            "confidence_score": self.confidence_score
        }


@dataclass
class SymbolInfo:
    """Information about a code symbol"""
    
    name: str
    symbol_type: str  # "function", "class", "variable", "import", etc.
    file_path: str
    line_number: int
    column: int
    
    # Context information
    parent_class: Optional[str] = None
    parent_function: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    
    # Usage information
    definitions: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol_type": self.symbol_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "parent_class": self.parent_class,
            "parent_function": self.parent_function,
            "signature": self.signature,
            "docstring": self.docstring,
            "definitions": self.definitions,
            "references": self.references
        }


class CodeSearchTool(Tool):
    """
    Advanced code search and navigation tool
    
    Provides comprehensive search capabilities including text search,
    regex search, symbol lookup, and dependency analysis.
    """
    
    # Supported file extensions for code search
    CODE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    def __init__(
        self,
        name: str = "code_search",
        allowed_directories: List[str] = None,
        max_results: int = 100,
        context_lines: int = 3
    ):
        super().__init__(
            name=name,
            description="Advanced code search and navigation tool",
            category=ToolCategory.ANALYSIS
        )
        
        self.allowed_directories = allowed_directories or ["."]
        self.max_results = max_results
        self.context_lines = context_lines
        
        # Convert to Path objects and resolve
        self.allowed_paths = [Path(d).resolve() for d in self.allowed_directories]
        
        # Symbol index cache
        self.symbol_index: Dict[str, List[SymbolInfo]] = {}
        self.index_timestamp = 0
        
        logger.info(f"CodeSearchTool initialized with {len(self.allowed_directories)} allowed directories")
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories"""
        try:
            resolved_path = Path(path).resolve()
            
            for allowed_path in self.allowed_paths:
                try:
                    resolved_path.relative_to(allowed_path)
                    return True
                except ValueError:
                    continue
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking path permissions for {path}: {e}")
            return False
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if a file is a code file based on extension"""
        return Path(file_path).suffix.lower() in self.CODE_EXTENSIONS
    
    def _get_file_language(self, file_path: str) -> Optional[str]:
        """Get the programming language for a file"""
        return self.CODE_EXTENSIONS.get(Path(file_path).suffix.lower())
    
    @tool_function(
        description="Search for text patterns in code files",
        parameters=[
            ToolParameter("query", "string", "Text to search for", required=True),
            ToolParameter("search_type", "string", "Type of search", required=False, default="text_search",
                         enum_values=["text_search", "regex_search"]),
            ToolParameter("file_pattern", "string", "File pattern to search in (glob)", required=False, default="*"),
            ToolParameter("case_sensitive", "boolean", "Case sensitive search", required=False, default=False),
            ToolParameter("whole_word", "boolean", "Match whole words only", required=False, default=False),
            ToolParameter("max_results", "integer", "Maximum number of results", required=False, default=50)
        ],
        returns="List of search results with file locations and context",
        category=ToolCategory.ANALYSIS
    )
    async def search_text(
        self,
        query: str,
        search_type: str = "text_search",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        whole_word: bool = False,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """Search for text patterns in code files"""
        
        try:
            results = []
            files_searched = 0
            
            # Build search pattern
            if search_type == "regex_search":
                if case_sensitive:
                    pattern = re.compile(query)
                else:
                    pattern = re.compile(query, re.IGNORECASE)
            else:
                # Text search
                if whole_word:
                    query_pattern = r'\b' + re.escape(query) + r'\b'
                else:
                    query_pattern = re.escape(query)
                
                if case_sensitive:
                    pattern = re.compile(query_pattern)
                else:
                    pattern = re.compile(query_pattern, re.IGNORECASE)
            
            # Search through allowed directories
            for allowed_path in self.allowed_paths:
                if file_pattern == "*":
                    file_iterator = allowed_path.rglob("*")
                else:
                    file_iterator = allowed_path.rglob(file_pattern)
                
                for file_path in file_iterator:
                    if not file_path.is_file() or not self._is_code_file(str(file_path)):
                        continue
                    
                    if not self._is_path_allowed(str(file_path)):
                        continue
                    
                    files_searched += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                        
                        for line_num, line in enumerate(lines, 1):
                            matches = pattern.finditer(line)
                            
                            for match in matches:
                                if len(results) >= max_results:
                                    break
                                
                                # Get context
                                context_before = []
                                context_after = []
                                
                                start_idx = max(0, line_num - self.context_lines - 1)
                                end_idx = min(len(lines), line_num + self.context_lines)
                                
                                for i in range(start_idx, line_num - 1):
                                    context_before.append(lines[i].rstrip())
                                
                                for i in range(line_num, end_idx):
                                    context_after.append(lines[i].rstrip())
                                
                                # Determine context (function/class)
                                function_name, class_name = self._get_context_info(
                                    lines, line_num - 1, str(file_path)
                                )
                                
                                result = SearchResult(
                                    file_path=str(file_path.relative_to(allowed_path)),
                                    line_number=line_num,
                                    column=match.start() + 1,
                                    match_text=match.group(),
                                    context_before=context_before,
                                    context_after=context_after,
                                    function_name=function_name,
                                    class_name=class_name,
                                    match_type=search_type
                                )
                                
                                results.append(result)
                            
                            if len(results) >= max_results:
                                break
                    
                    except Exception as e:
                        logger.warning(f"Error searching file {file_path}: {e}")
                        continue
                    
                    if len(results) >= max_results:
                        break
                
                if len(results) >= max_results:
                    break
            
            logger.info(f"Text search completed: {len(results)} results in {files_searched} files")
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "results_count": len(results),
                "files_searched": files_searched,
                "results": [result.to_dict() for result in results]
            }
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Search for symbol definitions and references",
        parameters=[
            ToolParameter("symbol_name", "string", "Name of the symbol to search for", required=True),
            ToolParameter("symbol_type", "string", "Type of symbol", required=False,
                         enum_values=["function", "class", "variable", "import", "any"]),
            ToolParameter("include_references", "boolean", "Include symbol references", required=False, default=True),
            ToolParameter("language", "string", "Programming language filter", required=False)
        ],
        returns="Symbol information including definitions and references",
        category=ToolCategory.ANALYSIS
    )
    async def search_symbol(
        self,
        symbol_name: str,
        symbol_type: str = "any",
        include_references: bool = True,
        language: str = None
    ) -> Dict[str, Any]:
        """Search for symbol definitions and references"""
        
        try:
            # Ensure symbol index is up to date
            await self._update_symbol_index()
            
            results = []
            
            # Search for symbols
            for file_path, symbols in self.symbol_index.items():
                # Filter by language if specified
                if language and self._get_file_language(file_path) != language:
                    continue
                
                for symbol in symbols:
                    if symbol.name == symbol_name:
                        if symbol_type == "any" or symbol.symbol_type == symbol_type:
                            # Found matching symbol
                            symbol_info = symbol.to_dict()
                            
                            if include_references:
                                # Find references to this symbol
                                references = await self._find_symbol_references(symbol_name, file_path)
                                symbol_info["references"] = references
                            
                            results.append(symbol_info)
            
            logger.info(f"Symbol search completed: {len(results)} results for '{symbol_name}'")
            
            return {
                "success": True,
                "symbol_name": symbol_name,
                "symbol_type": symbol_type,
                "results_count": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in symbol search: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Find dependencies and imports for a file",
        parameters=[
            ToolParameter("file_path", "string", "Path to the file", required=True),
            ToolParameter("include_reverse_deps", "boolean", "Include files that depend on this file", required=False, default=False)
        ],
        returns="List of dependencies and reverse dependencies",
        category=ToolCategory.ANALYSIS
    )
    async def analyze_dependencies(
        self,
        file_path: str,
        include_reverse_deps: bool = False
    ) -> Dict[str, Any]:
        """Analyze dependencies for a file"""
        
        try:
            if not self._is_path_allowed(file_path):
                return {"success": False, "error": "Access denied to file"}
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {"success": False, "error": "File not found"}
            
            language = self._get_file_language(file_path)
            if not language:
                return {"success": False, "error": "Unsupported file type"}
            
            dependencies = []
            reverse_dependencies = []
            
            # Analyze imports/dependencies based on language
            if language == "python":
                dependencies = self._analyze_python_dependencies(file_path)
            elif language in ["javascript", "typescript"]:
                dependencies = self._analyze_js_dependencies(file_path)
            elif language == "java":
                dependencies = self._analyze_java_dependencies(file_path)
            
            # Find reverse dependencies if requested
            if include_reverse_deps:
                reverse_dependencies = await self._find_reverse_dependencies(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "language": language,
                "dependencies": dependencies,
                "reverse_dependencies": reverse_dependencies if include_reverse_deps else None,
                "dependency_count": len(dependencies),
                "reverse_dependency_count": len(reverse_dependencies) if include_reverse_deps else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Navigate to symbol definition",
        parameters=[
            ToolParameter("symbol_name", "string", "Name of the symbol", required=True),
            ToolParameter("current_file", "string", "Current file context", required=False),
            ToolParameter("line_number", "integer", "Current line number", required=False)
        ],
        returns="Location of symbol definition",
        category=ToolCategory.ANALYSIS
    )
    async def go_to_definition(
        self,
        symbol_name: str,
        current_file: str = None,
        line_number: int = None
    ) -> Dict[str, Any]:
        """Navigate to symbol definition"""
        
        try:
            # Search for symbol definition
            symbol_results = await self.search_symbol(
                symbol_name=symbol_name,
                include_references=False
            )
            
            if not symbol_results["success"] or not symbol_results["results"]:
                return {
                    "success": False,
                    "error": f"Definition not found for symbol: {symbol_name}"
                }
            
            # Find the best definition match
            definitions = symbol_results["results"]
            
            # If we have current file context, prefer definitions in the same file
            if current_file:
                same_file_defs = [d for d in definitions if current_file in d["file_path"]]
                if same_file_defs:
                    definitions = same_file_defs
            
            # Return the first/best definition
            best_definition = definitions[0]
            
            return {
                "success": True,
                "symbol_name": symbol_name,
                "definition": {
                    "file_path": best_definition["file_path"],
                    "line_number": best_definition["line_number"],
                    "column": best_definition["column"],
                    "symbol_type": best_definition["symbol_type"],
                    "signature": best_definition.get("signature"),
                    "docstring": best_definition.get("docstring")
                },
                "alternative_definitions": len(definitions) - 1
            }
            
        except Exception as e:
            logger.error(f"Error finding definition: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_symbol_index(self):
        """Update the symbol index if needed"""
        
        # Simple timestamp-based cache invalidation
        current_time = os.path.getmtime(self.allowed_paths[0]) if self.allowed_paths else 0
        
        if current_time > self.index_timestamp:
            logger.info("Updating symbol index...")
            self.symbol_index = {}
            
            for allowed_path in self.allowed_paths:
                for file_path in allowed_path.rglob("*"):
                    if file_path.is_file() and self._is_code_file(str(file_path)):
                        if self._is_path_allowed(str(file_path)):
                            symbols = self._extract_symbols(str(file_path))
                            if symbols:
                                self.symbol_index[str(file_path)] = symbols
            
            self.index_timestamp = current_time
            logger.info(f"Symbol index updated: {len(self.symbol_index)} files indexed")
    
    def _extract_symbols(self, file_path: str) -> List[SymbolInfo]:
        """Extract symbols from a code file"""
        
        language = self._get_file_language(file_path)
        
        if language == "python":
            return self._extract_python_symbols(file_path)
        elif language in ["javascript", "typescript"]:
            return self._extract_js_symbols(file_path)
        else:
            # Fallback to simple regex-based extraction
            return self._extract_generic_symbols(file_path)
    
    def _extract_python_symbols(self, file_path: str) -> List[SymbolInfo]:
        """Extract symbols from Python file using AST"""
        
        symbols = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append(SymbolInfo(
                        name=node.name,
                        symbol_type="function",
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        signature=f"def {node.name}({', '.join(arg.arg for arg in node.args.args)})",
                        docstring=ast.get_docstring(node)
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    symbols.append(SymbolInfo(
                        name=node.name,
                        symbol_type="class",
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        signature=f"class {node.name}",
                        docstring=ast.get_docstring(node)
                    ))
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbols.append(SymbolInfo(
                            name=alias.name,
                            symbol_type="import",
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        symbols.append(SymbolInfo(
                            name=alias.name,
                            symbol_type="import",
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset
                        ))
            
        except Exception as e:
            logger.warning(f"Error extracting Python symbols from {file_path}: {e}")
        
        return symbols
    
    def _extract_js_symbols(self, file_path: str) -> List[SymbolInfo]:
        """Extract symbols from JavaScript/TypeScript file using regex"""
        
        symbols = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Regex patterns for JS/TS symbols
            function_pattern = re.compile(r'^\s*(export\s+)?(async\s+)?function\s+(\w+)\s*\(', re.MULTILINE)
            class_pattern = re.compile(r'^\s*(export\s+)?(abstract\s+)?class\s+(\w+)', re.MULTILINE)
            const_pattern = re.compile(r'^\s*(export\s+)?const\s+(\w+)', re.MULTILINE)
            import_pattern = re.compile(r'^\s*import\s+.*from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE)
            
            content = ''.join(lines)
            
            # Find functions
            for match in function_pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                symbols.append(SymbolInfo(
                    name=match.group(3),
                    symbol_type="function",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start() - content.rfind('\n', 0, match.start())
                ))
            
            # Find classes
            for match in class_pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                symbols.append(SymbolInfo(
                    name=match.group(3),
                    symbol_type="class",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start() - content.rfind('\n', 0, match.start())
                ))
            
            # Find constants
            for match in const_pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                symbols.append(SymbolInfo(
                    name=match.group(2),
                    symbol_type="variable",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start() - content.rfind('\n', 0, match.start())
                ))
            
            # Find imports
            for match in import_pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                symbols.append(SymbolInfo(
                    name=match.group(1),
                    symbol_type="import",
                    file_path=file_path,
                    line_number=line_num,
                    column=match.start() - content.rfind('\n', 0, match.start())
                ))
            
        except Exception as e:
            logger.warning(f"Error extracting JS symbols from {file_path}: {e}")
        
        return symbols
    
    def _extract_generic_symbols(self, file_path: str) -> List[SymbolInfo]:
        """Generic symbol extraction using simple patterns"""
        
        symbols = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Simple patterns that work across languages
            for line_num, line in enumerate(lines, 1):
                # Function-like patterns
                func_match = re.search(r'\b(function|def|fn)\s+(\w+)', line)
                if func_match:
                    symbols.append(SymbolInfo(
                        name=func_match.group(2),
                        symbol_type="function",
                        file_path=file_path,
                        line_number=line_num,
                        column=func_match.start(2)
                    ))
                
                # Class-like patterns
                class_match = re.search(r'\b(class|struct|interface)\s+(\w+)', line)
                if class_match:
                    symbols.append(SymbolInfo(
                        name=class_match.group(2),
                        symbol_type="class",
                        file_path=file_path,
                        line_number=line_num,
                        column=class_match.start(2)
                    ))
        
        except Exception as e:
            logger.warning(f"Error extracting generic symbols from {file_path}: {e}")
        
        return symbols
    
    def _get_context_info(self, lines: List[str], line_index: int, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Get function and class context for a line"""
        
        function_name = None
        class_name = None
        
        # Look backwards to find containing function/class
        for i in range(line_index, -1, -1):
            line = lines[i].strip()
            
            # Python function
            func_match = re.match(r'def\s+(\w+)', line)
            if func_match and not function_name:
                function_name = func_match.group(1)
            
            # Python class
            class_match = re.match(r'class\s+(\w+)', line)
            if class_match and not class_name:
                class_name = class_match.group(1)
            
            # JavaScript function
            js_func_match = re.match(r'function\s+(\w+)', line)
            if js_func_match and not function_name:
                function_name = js_func_match.group(1)
            
            # Stop if we have both
            if function_name and class_name:
                break
        
        return function_name, class_name
    
    def _analyze_python_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze Python imports and dependencies"""
        
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append({
                            "name": alias.name,
                            "type": "import",
                            "line": node.lineno,
                            "alias": alias.asname
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        dependencies.append({
                            "name": f"{module}.{alias.name}" if module else alias.name,
                            "type": "from_import",
                            "module": module,
                            "imported_name": alias.name,
                            "line": node.lineno,
                            "alias": alias.asname
                        })
        
        except Exception as e:
            logger.warning(f"Error analyzing Python dependencies: {e}")
        
        return dependencies
    
    def _analyze_js_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze JavaScript/TypeScript imports"""
        
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            import_pattern = re.compile(r'import\s+(.+)\s+from\s+[\'"]([^\'"]+)[\'"]')
            require_pattern = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)')
            
            for line_num, line in enumerate(lines, 1):
                # ES6 imports
                import_match = import_pattern.search(line)
                if import_match:
                    dependencies.append({
                        "name": import_match.group(2),
                        "type": "import",
                        "imported": import_match.group(1).strip(),
                        "line": line_num
                    })
                
                # CommonJS requires
                require_match = require_pattern.search(line)
                if require_match:
                    dependencies.append({
                        "name": require_match.group(1),
                        "type": "require",
                        "line": line_num
                    })
        
        except Exception as e:
            logger.warning(f"Error analyzing JS dependencies: {e}")
        
        return dependencies
    
    def _analyze_java_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze Java imports"""
        
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            import_pattern = re.compile(r'import\s+(static\s+)?([^;]+);')
            
            for line_num, line in enumerate(lines, 1):
                import_match = import_pattern.search(line)
                if import_match:
                    dependencies.append({
                        "name": import_match.group(2).strip(),
                        "type": "import",
                        "static": bool(import_match.group(1)),
                        "line": line_num
                    })
        
        except Exception as e:
            logger.warning(f"Error analyzing Java dependencies: {e}")
        
        return dependencies
    
    async def _find_symbol_references(self, symbol_name: str, definition_file: str) -> List[Dict[str, Any]]:
        """Find all references to a symbol"""
        
        references = []
        
        # Use text search to find references
        search_result = await self.search_text(
            query=symbol_name,
            search_type="text_search",
            whole_word=True,
            max_results=100
        )
        
        if search_result["success"]:
            for result in search_result["results"]:
                # Skip the definition itself
                if result["file_path"] == definition_file:
                    continue
                
                references.append({
                    "file_path": result["file_path"],
                    "line_number": result["line_number"],
                    "column": result["column"],
                    "context": result["match_text"],
                    "function_name": result.get("function_name"),
                    "class_name": result.get("class_name")
                })
        
        return references
    
    async def _find_reverse_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        """Find files that depend on the given file"""
        
        reverse_deps = []
        
        # Extract the module name from file path
        module_name = Path(file_path).stem
        
        # Search for imports of this module
        search_result = await self.search_text(
            query=module_name,
            search_type="regex_search",
            max_results=100
        )
        
        if search_result["success"]:
            for result in search_result["results"]:
                # Check if this looks like an import statement
                context = result["match_text"].lower()
                if any(keyword in context for keyword in ["import", "require", "from", "include"]):
                    reverse_deps.append({
                        "file_path": result["file_path"],
                        "line_number": result["line_number"],
                        "import_statement": result["match_text"]
                    })
        
        return reverse_deps
