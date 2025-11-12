"""
IDE Simulator Tool for LoCoBench-Agent

This tool simulates IDE functionality including project navigation, code completion,
refactoring operations, and integrated development environment features.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.tool_registry import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class RefactoringType(Enum):
    """Types of refactoring operations"""
    RENAME = "rename"
    EXTRACT_METHOD = "extract_method"
    EXTRACT_VARIABLE = "extract_variable"
    INLINE = "inline"
    MOVE_METHOD = "move_method"
    EXTRACT_CLASS = "extract_class"


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, variable, etc.)"""
    name: str
    type: str  # "function", "class", "variable", "import", etc.
    file_path: str
    line_number: int
    column: int
    scope: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    references: List[Tuple[str, int]] = field(default_factory=list)  # (file_path, line_number)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "scope": self.scope,
            "signature": self.signature,
            "docstring": self.docstring,
            "references": self.references
        }


@dataclass
class CodeCompletion:
    """Represents a code completion suggestion"""
    text: str
    type: str  # "function", "variable", "keyword", "snippet"
    description: str
    insert_text: str
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "description": self.description,
            "insert_text": self.insert_text,
            "priority": self.priority
        }


@dataclass
class DiagnosticMessage:
    """Represents a diagnostic message (error, warning, info)"""
    severity: str  # "error", "warning", "info", "hint"
    message: str
    file_path: str
    line_number: int
    column: int
    code: Optional[str] = None
    source: str = "ide_simulator"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "code": self.code,
            "source": self.source
        }


class IDESimulatorTool(Tool):
    """
    Tool that simulates IDE functionality for code development and analysis
    """
    
    def __init__(
        self,
        name: str = "ide_simulator",
        allowed_directories: List[str] = None,
        enable_ai_completion: bool = True
    ):
        super().__init__(
            name=name,
            description="Tool that simulates IDE functionality including navigation, completion, and refactoring",
            category=ToolCategory.ANALYSIS
        )
        
        self.allowed_directories = allowed_directories or ["."]
        self.enable_ai_completion = enable_ai_completion
        
        # Convert to Path objects and resolve
        self.allowed_paths = [Path(d).resolve() for d in self.allowed_directories]
        
        # Project state
        self.project_symbols: Dict[str, List[CodeSymbol]] = {}  # file_path -> symbols
        self.project_diagnostics: List[DiagnosticMessage] = []
        self.open_files: Dict[str, str] = {}  # file_path -> content
        
        logger.info(f"IDESimulatorTool initialized")
    
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
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin"
        }
        
        return extension_map.get(file_ext)
    
    @tool_function(
        description="Open a file in the IDE simulator",
        parameters=[
            ToolParameter("file_path", "string", "Path to the file to open", required=True)
        ],
        returns="File content and basic analysis",
        category=ToolCategory.ANALYSIS
    )
    async def open_file(self, file_path: str) -> Dict[str, Any]:
        """Open a file in the IDE simulator"""
        
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access denied to file: {file_path}")
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"success": False, "error": f"Path is not a file: {file_path}"}
            
            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # CRITICAL: Limit content size to prevent 20MB message errors
            max_content_size = 8_000_000  # 8MB safety limit (OpenAI limit is 10MB)
            original_content = content
            if len(content) > max_content_size:
                logger.warning(f"IDE file {file_path} too large ({len(content)} chars), truncating to {max_content_size}")
                content = content[:max_content_size] + f"\n\n[Content truncated - file was {len(original_content)} characters, showing first {max_content_size}]"
            
            # Store in open files
            self.open_files[file_path] = content
            
            # Analyze file
            language = self._detect_language(file_path)
            symbols = await self._analyze_file_symbols(file_path, content, language)
            diagnostics = await self._analyze_file_diagnostics(file_path, content, language)
            
            # Store symbols
            self.project_symbols[file_path] = symbols
            
            # Update diagnostics
            self.project_diagnostics = [d for d in self.project_diagnostics if d.file_path != file_path]
            self.project_diagnostics.extend(diagnostics)
            
            logger.info(f"Opened file: {file_path} ({len(symbols)} symbols, {len(diagnostics)} diagnostics)")
            
            return {
                "success": True,
                "file_path": file_path,
                "content": content,
                "language": language,
                "line_count": len(content.split('\n')),
                "symbols": [s.to_dict() for s in symbols],
                "diagnostics": [d.to_dict() for d in diagnostics]
            }
            
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Get code completion suggestions at a specific position",
        parameters=[
            ToolParameter("file_path", "string", "Path to the file", required=True),
            ToolParameter("line_number", "integer", "Line number (1-based)", required=True),
            ToolParameter("column", "integer", "Column position (0-based)", required=True),
            ToolParameter("trigger_character", "string", "Character that triggered completion", required=False)
        ],
        returns="List of code completion suggestions",
        category=ToolCategory.ANALYSIS
    )
    async def get_code_completion(
        self,
        file_path: str,
        line_number: int,
        column: int,
        trigger_character: str = None
    ) -> Dict[str, Any]:
        """Get code completion suggestions"""
        
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access denied to file: {file_path}")
        
        try:
            # Get file content
            if file_path not in self.open_files:
                await self.open_file(file_path)
            
            content = self.open_files.get(file_path, "")
            language = self._detect_language(file_path)
            
            # Get completion suggestions
            completions = await self._generate_completions(
                file_path, content, line_number, column, language, trigger_character
            )
            
            logger.debug(f"Generated {len(completions)} completions for {file_path}:{line_number}:{column}")
            
            return {
                "success": True,
                "file_path": file_path,
                "position": {"line": line_number, "column": column},
                "completions": [c.to_dict() for c in completions]
            }
            
        except Exception as e:
            logger.error(f"Error getting code completion: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Find all references to a symbol",
        parameters=[
            ToolParameter("symbol_name", "string", "Name of the symbol to find", required=True),
            ToolParameter("file_path", "string", "File containing the symbol", required=False),
            ToolParameter("line_number", "integer", "Line number of the symbol", required=False)
        ],
        returns="List of all references to the symbol",
        category=ToolCategory.ANALYSIS
    )
    async def find_references(
        self,
        symbol_name: str,
        file_path: str = None,
        line_number: int = None
    ) -> Dict[str, Any]:
        """Find all references to a symbol"""
        
        try:
            references = []
            
            # Search through all analyzed files
            for fp, symbols in self.project_symbols.items():
                for symbol in symbols:
                    if symbol.name == symbol_name:
                        # Add the definition
                        references.append({
                            "file_path": symbol.file_path,
                            "line_number": symbol.line_number,
                            "column": symbol.column,
                            "type": "definition",
                            "context": f"{symbol.type} {symbol.name}"
                        })
                        
                        # Add all references
                        for ref_file, ref_line in symbol.references:
                            references.append({
                                "file_path": ref_file,
                                "line_number": ref_line,
                                "column": 0,  # Would need more detailed analysis
                                "type": "reference",
                                "context": f"Reference to {symbol.name}"
                            })
            
            # If no specific file/line provided, search in all files
            if not file_path:
                for fp in self.open_files:
                    content = self.open_files[fp]
                    refs = await self._find_symbol_references(fp, content, symbol_name)
                    references.extend(refs)
            
            logger.info(f"Found {len(references)} references to symbol '{symbol_name}'")
            
            return {
                "success": True,
                "symbol_name": symbol_name,
                "references": references,
                "total_references": len(references)
            }
            
        except Exception as e:
            logger.error(f"Error finding references: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Navigate to symbol definition",
        parameters=[
            ToolParameter("symbol_name", "string", "Name of the symbol", required=True),
            ToolParameter("current_file", "string", "Current file path", required=False)
        ],
        returns="Location of the symbol definition",
        category=ToolCategory.ANALYSIS
    )
    async def go_to_definition(
        self,
        symbol_name: str,
        current_file: str = None
    ) -> Dict[str, Any]:
        """Navigate to symbol definition"""
        
        try:
            # Search for symbol definition
            for file_path, symbols in self.project_symbols.items():
                for symbol in symbols:
                    if symbol.name == symbol_name:
                        logger.info(f"Found definition of '{symbol_name}' at {file_path}:{symbol.line_number}")
                        
                        return {
                            "success": True,
                            "symbol_name": symbol_name,
                            "definition": {
                                "file_path": symbol.file_path,
                                "line_number": symbol.line_number,
                                "column": symbol.column,
                                "type": symbol.type,
                                "signature": symbol.signature,
                                "docstring": symbol.docstring
                            }
                        }
            
            return {
                "success": False,
                "error": f"Definition not found for symbol: {symbol_name}"
            }
            
        except Exception as e:
            logger.error(f"Error finding definition: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Perform a refactoring operation",
        parameters=[
            ToolParameter("refactoring_type", "string", "Type of refactoring", required=True,
                         enum_values=["rename", "extract_method", "extract_variable", "inline", "move_method", "extract_class"]),
            ToolParameter("file_path", "string", "File to refactor", required=True),
            ToolParameter("old_name", "string", "Current name (for rename operations)", required=False),
            ToolParameter("new_name", "string", "New name (for rename operations)", required=True),
            ToolParameter("start_line", "integer", "Start line for extraction", required=False),
            ToolParameter("end_line", "integer", "End line for extraction", required=False)
        ],
        returns="Refactoring result and changes",
        category=ToolCategory.ANALYSIS
    )
    async def refactor_code(
        self,
        refactoring_type: str,
        file_path: str,
        new_name: str,
        old_name: str = None,
        start_line: int = None,
        end_line: int = None
    ) -> Dict[str, Any]:
        """Perform a refactoring operation"""
        
        if not self._is_path_allowed(file_path):
            raise PermissionError(f"Access denied to file: {file_path}")
        
        try:
            refactor_type = RefactoringType(refactoring_type)
            
            # Get file content
            if file_path not in self.open_files:
                await self.open_file(file_path)
            
            content = self.open_files[file_path]
            lines = content.split('\n')
            
            changes = []
            
            if refactor_type == RefactoringType.RENAME:
                if not old_name:
                    return {"success": False, "error": "old_name required for rename operation"}
                
                # Simple rename implementation
                new_content = content.replace(old_name, new_name)
                changes.append({
                    "type": "replace_all",
                    "old_text": old_name,
                    "new_text": new_name,
                    "occurrences": content.count(old_name)
                })
                
            elif refactor_type == RefactoringType.EXTRACT_METHOD:
                if start_line is None or end_line is None:
                    return {"success": False, "error": "start_line and end_line required for extract_method"}
                
                # Extract method implementation
                extracted_lines = lines[start_line-1:end_line]
                extracted_code = '\n'.join(extracted_lines)
                
                # Create method
                method_code = f"\ndef {new_name}():\n"
                for line in extracted_lines:
                    method_code += f"    {line}\n"
                
                # Replace extracted code with method call
                new_lines = lines[:start_line-1] + [f"    {new_name}()"] + lines[end_line:]
                new_content = '\n'.join(new_lines)
                
                # Insert method at appropriate location
                # (Simplified: insert at the end of file)
                new_content += f"\n{method_code}"
                
                changes.append({
                    "type": "extract_method",
                    "method_name": new_name,
                    "extracted_lines": f"{start_line}-{end_line}",
                    "method_code": method_code
                })
                
            else:
                return {
                    "success": False,
                    "error": f"Refactoring type '{refactoring_type}' not yet implemented"
                }
            
            # Update file content
            self.open_files[file_path] = new_content
            
            logger.info(f"Performed {refactoring_type} refactoring in {file_path}")
            
            return {
                "success": True,
                "refactoring_type": refactoring_type,
                "file_path": file_path,
                "changes": changes,
                "new_content": new_content
            }
            
        except Exception as e:
            logger.error(f"Error performing refactoring: {e}")
            return {"success": False, "error": str(e)}
    
    @tool_function(
        description="Get project overview and statistics",
        parameters=[
            ToolParameter("directory", "string", "Project directory to analyze", required=False, default=".")
        ],
        returns="Project overview with file counts, languages, and structure",
        category=ToolCategory.ANALYSIS
    )
    async def get_project_overview(self, directory: str = ".") -> Dict[str, Any]:
        """Get project overview and statistics"""
        
        if not self._is_path_allowed(directory):
            raise PermissionError(f"Access denied to directory: {directory}")
        
        try:
            project_path = Path(directory)
            
            # Analyze project structure
            files_by_language = {}
            total_files = 0
            total_lines = 0
            
            for file_path in project_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    language = self._detect_language(str(file_path))
                    if language:
                        if language not in files_by_language:
                            files_by_language[language] = {"count": 0, "lines": 0, "files": []}
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                line_count = len(content.split('\n'))
                                
                                files_by_language[language]["count"] += 1
                                files_by_language[language]["lines"] += line_count
                                files_by_language[language]["files"].append(str(file_path.relative_to(project_path)))
                                
                                total_files += 1
                                total_lines += line_count
                        except Exception:
                            # Skip files that can't be read
                            pass
            
            # Get symbol statistics
            total_symbols = sum(len(symbols) for symbols in self.project_symbols.values())
            symbol_types = {}
            
            for symbols in self.project_symbols.values():
                for symbol in symbols:
                    symbol_types[symbol.type] = symbol_types.get(symbol.type, 0) + 1
            
            # Get diagnostic statistics
            diagnostic_stats = {"error": 0, "warning": 0, "info": 0, "hint": 0}
            for diagnostic in self.project_diagnostics:
                diagnostic_stats[diagnostic.severity] = diagnostic_stats.get(diagnostic.severity, 0) + 1
            
            logger.info(f"Analyzed project: {total_files} files, {total_lines} lines")
            
            return {
                "success": True,
                "project_directory": directory,
                "total_files": total_files,
                "total_lines": total_lines,
                "languages": files_by_language,
                "symbols": {
                    "total": total_symbols,
                    "by_type": symbol_types
                },
                "diagnostics": diagnostic_stats,
                "open_files": list(self.open_files.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting project overview: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_file_symbols(
        self,
        file_path: str,
        content: str,
        language: str
    ) -> List[CodeSymbol]:
        """Analyze file and extract symbols"""
        
        symbols = []
        
        if not language:
            return symbols
        
        lines = content.split('\n')
        
        if language == "python":
            # Simple Python symbol extraction
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Functions
                if line.startswith('def '):
                    match = re.match(r'def\s+(\w+)\s*\((.*?)\):', line)
                    if match:
                        name, params = match.groups()
                        symbols.append(CodeSymbol(
                            name=name,
                            type="function",
                            file_path=file_path,
                            line_number=i + 1,
                            column=0,
                            scope="module",
                            signature=f"def {name}({params})"
                        ))
                
                # Classes
                elif line.startswith('class '):
                    match = re.match(r'class\s+(\w+)(?:\([^)]*\))?:', line)
                    if match:
                        name = match.group(1)
                        symbols.append(CodeSymbol(
                            name=name,
                            type="class",
                            file_path=file_path,
                            line_number=i + 1,
                            column=0,
                            scope="module",
                            signature=line
                        ))
                
                # Variables (simple assignment)
                elif '=' in line and not line.startswith((' ', '\t')):
                    match = re.match(r'(\w+)\s*=', line)
                    if match:
                        name = match.group(1)
                        symbols.append(CodeSymbol(
                            name=name,
                            type="variable",
                            file_path=file_path,
                            line_number=i + 1,
                            column=0,
                            scope="module"
                        ))
        
        elif language == "javascript" or language == "typescript":
            # Simple JavaScript/TypeScript symbol extraction
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Functions
                if 'function ' in line:
                    match = re.search(r'function\s+(\w+)\s*\(([^)]*)\)', line)
                    if match:
                        name, params = match.groups()
                        symbols.append(CodeSymbol(
                            name=name,
                            type="function",
                            file_path=file_path,
                            line_number=i + 1,
                            column=0,
                            scope="module",
                            signature=f"function {name}({params})"
                        ))
                
                # Classes
                elif line.startswith('class '):
                    match = re.match(r'class\s+(\w+)', line)
                    if match:
                        name = match.group(1)
                        symbols.append(CodeSymbol(
                            name=name,
                            type="class",
                            file_path=file_path,
                            line_number=i + 1,
                            column=0,
                            scope="module",
                            signature=line
                        ))
        
        return symbols
    
    async def _analyze_file_diagnostics(
        self,
        file_path: str,
        content: str,
        language: str
    ) -> List[DiagnosticMessage]:
        """Analyze file and generate diagnostics"""
        
        diagnostics = []
        
        if not language:
            return diagnostics
        
        lines = content.split('\n')
        
        # Simple diagnostic rules
        for i, line in enumerate(lines):
            # Long lines
            if len(line) > 120:
                diagnostics.append(DiagnosticMessage(
                    severity="warning",
                    message="Line too long (>120 characters)",
                    file_path=file_path,
                    line_number=i + 1,
                    column=120,
                    code="line_length"
                ))
            
            # TODO comments
            if "TODO" in line:
                diagnostics.append(DiagnosticMessage(
                    severity="info",
                    message="TODO comment found",
                    file_path=file_path,
                    line_number=i + 1,
                    column=line.find("TODO"),
                    code="todo_comment"
                ))
            
            # Empty catch blocks (simple check)
            if language in ["java", "javascript", "typescript", "csharp"] and "catch" in line:
                if i + 1 < len(lines) and lines[i + 1].strip() == "}":
                    diagnostics.append(DiagnosticMessage(
                        severity="warning",
                        message="Empty catch block",
                        file_path=file_path,
                        line_number=i + 1,
                        column=0,
                        code="empty_catch"
                    ))
        
        return diagnostics
    
    async def _generate_completions(
        self,
        file_path: str,
        content: str,
        line_number: int,
        column: int,
        language: str,
        trigger_character: str
    ) -> List[CodeCompletion]:
        """Generate code completion suggestions"""
        
        completions = []
        
        if not language:
            return completions
        
        # Get current line and context
        lines = content.split('\n')
        if line_number <= len(lines):
            current_line = lines[line_number - 1]
            prefix = current_line[:column]
        else:
            current_line = ""
            prefix = ""
        
        # Language-specific completions
        if language == "python":
            # Python keywords
            python_keywords = [
                "def", "class", "if", "elif", "else", "for", "while", "try", "except", "finally",
                "import", "from", "return", "yield", "break", "continue", "pass", "with", "as"
            ]
            
            for keyword in python_keywords:
                if keyword.startswith(prefix.split()[-1] if prefix.split() else ""):
                    completions.append(CodeCompletion(
                        text=keyword,
                        type="keyword",
                        description=f"Python keyword: {keyword}",
                        insert_text=keyword,
                        priority=10
                    ))
            
            # Common Python functions
            if trigger_character == ".":
                common_methods = [
                    ("append", "list.append(item)", "Add item to list"),
                    ("split", "str.split(sep)", "Split string by separator"),
                    ("join", "str.join(iterable)", "Join iterable with string"),
                    ("format", "str.format(*args)", "Format string")
                ]
                
                for method, signature, desc in common_methods:
                    completions.append(CodeCompletion(
                        text=method,
                        type="method",
                        description=desc,
                        insert_text=method,
                        priority=8
                    ))
        
        elif language in ["javascript", "typescript"]:
            # JavaScript keywords
            js_keywords = [
                "function", "var", "let", "const", "if", "else", "for", "while", "do",
                "switch", "case", "default", "try", "catch", "finally", "return", "break", "continue"
            ]
            
            for keyword in js_keywords:
                if keyword.startswith(prefix.split()[-1] if prefix.split() else ""):
                    completions.append(CodeCompletion(
                        text=keyword,
                        type="keyword",
                        description=f"JavaScript keyword: {keyword}",
                        insert_text=keyword,
                        priority=10
                    ))
        
        # Add symbols from current project
        for symbols in self.project_symbols.values():
            for symbol in symbols:
                if symbol.name.startswith(prefix.split()[-1] if prefix.split() else ""):
                    completions.append(CodeCompletion(
                        text=symbol.name,
                        type=symbol.type,
                        description=f"{symbol.type}: {symbol.signature or symbol.name}",
                        insert_text=symbol.name,
                        priority=5
                    ))
        
        # Sort by priority and relevance
        completions.sort(key=lambda x: (-x.priority, x.text))
        
        return completions[:20]  # Limit to top 20 suggestions
    
    async def _find_symbol_references(
        self,
        file_path: str,
        content: str,
        symbol_name: str
    ) -> List[Dict[str, Any]]:
        """Find references to a symbol in file content"""
        
        references = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if symbol_name in line:
                # Simple word boundary check
                if re.search(rf'\b{re.escape(symbol_name)}\b', line):
                    references.append({
                        "file_path": file_path,
                        "line_number": i + 1,
                        "column": line.find(symbol_name),
                        "type": "reference",
                        "context": line.strip()
                    })
        
        return references
