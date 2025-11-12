"""
Debugger Interface Tool for LoCoBench-Agent

This tool provides debugging capabilities including breakpoint management,
variable inspection, and step-by-step execution for multiple programming languages.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.tool_registry import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class DebuggerState(Enum):
    """State of the debugger session"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class Breakpoint:
    """Represents a breakpoint"""
    id: int
    file_path: str
    line_number: int
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "condition": self.condition,
            "enabled": self.enabled,
            "hit_count": self.hit_count
        }


@dataclass
class StackFrame:
    """Represents a stack frame"""
    level: int
    function_name: str
    file_path: str
    line_number: int
    variables: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "function_name": self.function_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "variables": self.variables
        }


@dataclass
class DebugSession:
    """Represents a debugging session"""
    session_id: str
    language: str
    program_path: str
    state: DebuggerState
    current_line: Optional[int] = None
    current_file: Optional[str] = None
    breakpoints: List[Breakpoint] = None
    stack_frames: List[StackFrame] = None
    
    def __post_init__(self):
        if self.breakpoints is None:
            self.breakpoints = []
        if self.stack_frames is None:
            self.stack_frames = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "program_path": self.program_path,
            "state": self.state.value,
            "current_line": self.current_line,
            "current_file": self.current_file,
            "breakpoints": [bp.to_dict() for bp in self.breakpoints],
            "stack_frames": [sf.to_dict() for sf in self.stack_frames]
        }


class DebuggerTool(Tool):
    """
    Tool for debugging programs across multiple programming languages
    """
    
    # Debugger configurations for different languages
    DEBUGGER_CONFIGS = {
        "python": {
            "debugger": "pdb",
            "start_command": ["python", "-m", "pdb", "{program}"],
            "commands": {
                "continue": "c",
                "step": "s",
                "next": "n",
                "break": "b {file}:{line}",
                "list": "l",
                "print": "p {expression}",
                "where": "w",
                "quit": "q"
            }
        },
        "javascript": {
            "debugger": "node",
            "start_command": ["node", "inspect", "{program}"],
            "commands": {
                "continue": "cont",
                "step": "step",
                "next": "next",
                "break": "setBreakpoint('{file}', {line})",
                "list": "list(5)",
                "print": "exec {expression}",
                "where": "backtrace",
                "quit": ".exit"
            }
        },
        "java": {
            "debugger": "jdb",
            "start_command": ["jdb", "-classpath", ".", "{class}"],
            "commands": {
                "continue": "cont",
                "step": "step",
                "next": "next",
                "break": "stop at {class}:{line}",
                "list": "list",
                "print": "print {expression}",
                "where": "where",
                "quit": "quit"
            }
        },
        "cpp": {
            "debugger": "gdb",
            "start_command": ["gdb", "{program}"],
            "commands": {
                "continue": "continue",
                "step": "step",
                "next": "next",
                "break": "break {file}:{line}",
                "list": "list",
                "print": "print {expression}",
                "where": "backtrace",
                "quit": "quit"
            }
        },
        "c": {
            "debugger": "gdb",
            "start_command": ["gdb", "{program}"],
            "commands": {
                "continue": "continue",
                "step": "step",
                "next": "next",
                "break": "break {file}:{line}",
                "list": "list",
                "print": "print {expression}",
                "where": "backtrace",
                "quit": "quit"
            }
        },
        "go": {
            "debugger": "dlv",
            "start_command": ["dlv", "debug", "{program}"],
            "commands": {
                "continue": "continue",
                "step": "step",
                "next": "next",
                "break": "break {file}:{line}",
                "list": "list",
                "print": "print {expression}",
                "where": "stack",
                "quit": "quit"
            }
        }
    }
    
    def __init__(
        self,
        name: str = "debugger",
        allowed_directories: List[str] = None,
        timeout_seconds: int = 300,
        max_sessions: int = 5
    ):
        super().__init__(
            name=name,
            description="Tool for debugging programs with breakpoints, variable inspection, and step execution",
            category=ToolCategory.DEBUGGER
        )
        
        self.allowed_directories = allowed_directories or ["."]
        self.timeout_seconds = timeout_seconds
        self.max_sessions = max_sessions
        
        # Convert to Path objects and resolve
        self.allowed_paths = [Path(d).resolve() for d in self.allowed_directories]
        
        # Active debugging sessions
        self.active_sessions: Dict[str, DebugSession] = {}
        self.session_processes: Dict[str, subprocess.Popen] = {}
        self.next_breakpoint_id = 1
        
        logger.info(f"DebuggerTool initialized with max {max_sessions} sessions")
    
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
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust"
        }
        
        return extension_map.get(file_ext)
    
    @tool_function(
        description="Start a debugging session for a program",
        parameters=[
            ToolParameter("program_path", "string", "Path to the program to debug", required=True),
            ToolParameter("language", "string", "Programming language (auto-detected if not specified)", required=False),
            ToolParameter("arguments", "array", "Command line arguments for the program", required=False, default=[]),
            ToolParameter("working_directory", "string", "Working directory", required=False, default=".")
        ],
        returns="Debug session information",
        category=ToolCategory.DEBUGGER
    )
    async def start_debug_session(
        self,
        program_path: str,
        language: str = None,
        arguments: List[str] = None,
        working_directory: str = "."
    ) -> Dict[str, Any]:
        """Start a debugging session"""
        
        # Check permissions
        if not self._is_path_allowed(working_directory):
            raise PermissionError(f"Access denied to directory: {working_directory}")
        
        full_program_path = Path(working_directory) / program_path
        if not self._is_path_allowed(str(full_program_path)):
            raise PermissionError(f"Access denied to program: {program_path}")
        
        # Check session limit
        if len(self.active_sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum number of debug sessions ({self.max_sessions}) reached")
        
        try:
            # Detect language if not specified
            if not language:
                language = self._detect_language(program_path)
            
            if not language:
                raise ValueError("Could not detect programming language")
            
            if language not in self.DEBUGGER_CONFIGS:
                raise ValueError(f"Debugging not supported for language: {language}")
            
            # Generate session ID
            session_id = f"debug_{int(time.time() * 1000)}"
            
            # Create debug session
            session = DebugSession(
                session_id=session_id,
                language=language,
                program_path=program_path,
                state=DebuggerState.IDLE
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Started debug session {session_id} for {language} program: {program_path}")
            
            return {
                "success": True,
                "session_id": session_id,
                "session": session.to_dict(),
                "supported_commands": list(self.DEBUGGER_CONFIGS[language]["commands"].keys())
            }
            
        except Exception as e:
            logger.error(f"Error starting debug session: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool_function(
        description="Set a breakpoint in the debugging session",
        parameters=[
            ToolParameter("session_id", "string", "Debug session ID", required=True),
            ToolParameter("file_path", "string", "File path for the breakpoint", required=True),
            ToolParameter("line_number", "integer", "Line number for the breakpoint", required=True),
            ToolParameter("condition", "string", "Optional condition for the breakpoint", required=False)
        ],
        returns="Breakpoint information",
        category=ToolCategory.DEBUGGER
    )
    async def set_breakpoint(
        self,
        session_id: str,
        file_path: str,
        line_number: int,
        condition: str = None
    ) -> Dict[str, Any]:
        """Set a breakpoint in the debugging session"""
        
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Debug session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        
        try:
            # Create breakpoint
            breakpoint = Breakpoint(
                id=self.next_breakpoint_id,
                file_path=file_path,
                line_number=line_number,
                condition=condition
            )
            
            self.next_breakpoint_id += 1
            session.breakpoints.append(breakpoint)
            
            logger.info(f"Set breakpoint {breakpoint.id} at {file_path}:{line_number}")
            
            return {
                "success": True,
                "breakpoint": breakpoint.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error setting breakpoint: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool_function(
        description="Execute a debugger command",
        parameters=[
            ToolParameter("session_id", "string", "Debug session ID", required=True),
            ToolParameter("command", "string", "Debugger command", required=True,
                         enum_values=["continue", "step", "next", "list", "print", "where", "quit"]),
            ToolParameter("parameters", "string", "Command parameters (e.g., variable name for print)", required=False)
        ],
        returns="Command execution result",
        category=ToolCategory.DEBUGGER
    )
    async def execute_debug_command(
        self,
        session_id: str,
        command: str,
        parameters: str = ""
    ) -> Dict[str, Any]:
        """Execute a debugger command"""
        
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Debug session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        
        try:
            config = self.DEBUGGER_CONFIGS[session.language]
            
            if command not in config["commands"]:
                return {"success": False, "error": f"Unknown command: {command}"}
            
            # Format command
            cmd_template = config["commands"][command]
            if parameters:
                if command == "print":
                    formatted_cmd = cmd_template.replace("{expression}", parameters)
                else:
                    formatted_cmd = f"{cmd_template} {parameters}"
            else:
                formatted_cmd = cmd_template
            
            # Simulate command execution (in a real implementation, this would interact with actual debugger)
            result = await self._simulate_debug_command(session, command, parameters)
            
            logger.info(f"Executed debug command '{command}' in session {session_id}")
            
            return {
                "success": True,
                "command": command,
                "output": result.get("output", ""),
                "session_state": session.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error executing debug command: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool_function(
        description="Get the current state of a debugging session",
        parameters=[
            ToolParameter("session_id", "string", "Debug session ID", required=True)
        ],
        returns="Current session state and information",
        category=ToolCategory.DEBUGGER
    )
    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get the current state of a debugging session"""
        
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Debug session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        
        return {
            "success": True,
            "session": session.to_dict()
        }
    
    @tool_function(
        description="List all active debugging sessions",
        parameters=[],
        returns="List of active debugging sessions",
        category=ToolCategory.DEBUGGER
    )
    async def list_debug_sessions(self) -> Dict[str, Any]:
        """List all active debugging sessions"""
        
        sessions = []
        for session_id, session in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "language": session.language,
                "program_path": session.program_path,
                "state": session.state.value,
                "breakpoints_count": len(session.breakpoints)
            })
        
        return {
            "success": True,
            "active_sessions": sessions,
            "total_sessions": len(sessions)
        }
    
    @tool_function(
        description="Stop and cleanup a debugging session",
        parameters=[
            ToolParameter("session_id", "string", "Debug session ID", required=True)
        ],
        returns="Session cleanup result",
        category=ToolCategory.DEBUGGER
    )
    async def stop_debug_session(self, session_id: str) -> Dict[str, Any]:
        """Stop and cleanup a debugging session"""
        
        if session_id not in self.active_sessions:
            return {"success": False, "error": f"Debug session {session_id} not found"}
        
        try:
            # Cleanup process if exists
            if session_id in self.session_processes:
                process = self.session_processes[session_id]
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                del self.session_processes[session_id]
            
            # Remove session
            session = self.active_sessions[session_id]
            session.state = DebuggerState.FINISHED
            del self.active_sessions[session_id]
            
            logger.info(f"Stopped debug session {session_id}")
            
            return {
                "success": True,
                "message": f"Debug session {session_id} stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Error stopping debug session: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_debug_command(self, session: DebugSession, command: str, parameters: str) -> Dict[str, Any]:
        """Simulate debugger command execution (placeholder implementation)"""
        
        # This is a simplified simulation. In a real implementation, this would:
        # 1. Start the actual debugger process
        # 2. Send commands to the debugger
        # 3. Parse debugger output
        # 4. Update session state accordingly
        
        if command == "continue":
            session.state = DebuggerState.RUNNING
            return {"output": "Continuing execution..."}
        
        elif command == "step":
            session.state = DebuggerState.PAUSED
            session.current_line = (session.current_line or 1) + 1
            return {"output": f"Stepped to line {session.current_line}"}
        
        elif command == "next":
            session.state = DebuggerState.PAUSED
            session.current_line = (session.current_line or 1) + 1
            return {"output": f"Next line: {session.current_line}"}
        
        elif command == "list":
            # Simulate code listing
            current_line = session.current_line or 1
            return {
                "output": f"""
    {current_line - 2}: # Previous line
    {current_line - 1}: # Previous line
--> {current_line}: # Current line (simulated)
    {current_line + 1}: # Next line
    {current_line + 2}: # Next line
                """.strip()
            }
        
        elif command == "print":
            # Simulate variable inspection
            if parameters:
                return {"output": f"{parameters} = <simulated_value>"}
            else:
                return {"output": "No variable specified"}
        
        elif command == "where":
            # Simulate stack trace
            frame = StackFrame(
                level=0,
                function_name="main",
                file_path=session.program_path,
                line_number=session.current_line or 1,
                variables={"x": 42, "y": "hello"}
            )
            session.stack_frames = [frame]
            
            return {
                "output": f"#0  main() at {session.program_path}:{session.current_line or 1}"
            }
        
        elif command == "quit":
            session.state = DebuggerState.FINISHED
            return {"output": "Debugger session ended"}
        
        else:
            return {"output": f"Command '{command}' executed (simulated)"}
    
    def cleanup_all_sessions(self) -> None:
        """Cleanup all active debugging sessions"""
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            try:
                asyncio.create_task(self.stop_debug_session(session_id))
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        logger.info(f"Cleaned up {len(session_ids)} debug sessions")
