"""
Compiler Interface Tool for LoCoBench-Agent

This tool provides compilation and testing capabilities for multiple programming
languages, enabling agents to compile code and run tests during evaluation.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..core.tool_registry import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Result of a compilation operation"""
    success: bool
    output: str
    error_output: str
    warnings: List[str]
    execution_time: float
    exit_code: int
    artifacts: List[str]  # Generated files (executables, bytecode, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error_output": self.error_output,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "exit_code": self.exit_code,
            "artifacts": self.artifacts
        }


@dataclass
class TestResult:
    """Result of a test execution"""
    success: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    test_output: str
    execution_time: float
    coverage_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "test_output": self.test_output,
            "execution_time": self.execution_time,
            "coverage_info": self.coverage_info
        }


class CompilerTool(Tool):
    """
    Tool for compiling code and running tests across multiple programming languages
    """
    
    # Language configurations
    LANGUAGE_CONFIGS = {
        "python": {
            "extensions": [".py"],
            "compile_command": None,  # Python is interpreted
            "run_command": ["python", "{file}"],
            "test_command": ["python", "-m", "pytest", "{test_dir}", "-v"],
            "test_patterns": ["test_*.py", "*_test.py"],
            "package_manager": "pip"
        },
        "javascript": {
            "extensions": [".js"],
            "compile_command": None,  # JavaScript is interpreted
            "run_command": ["node", "{file}"],
            "test_command": ["npm", "test"],
            "test_patterns": ["*.test.js", "*.spec.js"],
            "package_manager": "npm"
        },
        "typescript": {
            "extensions": [".ts"],
            "compile_command": ["tsc", "{file}"],
            "run_command": ["node", "{compiled_file}"],
            "test_command": ["npm", "test"],
            "test_patterns": ["*.test.ts", "*.spec.ts"],
            "package_manager": "npm"
        },
        "java": {
            "extensions": [".java"],
            "compile_command": ["javac", "{file}"],
            "run_command": ["java", "{class_name}"],
            "test_command": ["mvn", "test"],
            "test_patterns": ["*Test.java"],
            "package_manager": "maven"
        },
        "cpp": {
            "extensions": [".cpp", ".cc", ".cxx"],
            "compile_command": ["g++", "-o", "{output}", "{file}"],
            "run_command": ["./{executable}"],
            "test_command": ["make", "test"],
            "test_patterns": ["*_test.cpp", "test_*.cpp"],
            "package_manager": None
        },
        "c": {
            "extensions": [".c"],
            "compile_command": ["gcc", "-o", "{output}", "{file}"],
            "run_command": ["./{executable}"],
            "test_command": ["make", "test"],
            "test_patterns": ["*_test.c", "test_*.c"],
            "package_manager": None
        },
        "go": {
            "extensions": [".go"],
            "compile_command": ["go", "build", "{file}"],
            "run_command": ["go", "run", "{file}"],
            "test_command": ["go", "test", "./..."],
            "test_patterns": ["*_test.go"],
            "package_manager": "go"
        },
        "rust": {
            "extensions": [".rs"],
            "compile_command": ["rustc", "{file}"],
            "run_command": ["./{executable}"],
            "test_command": ["cargo", "test"],
            "test_patterns": ["*.rs"],  # Tests are typically in the same files
            "package_manager": "cargo"
        },
        "csharp": {
            "extensions": [".cs"],
            "compile_command": ["csc", "{file}"],
            "run_command": ["mono", "{executable}"],
            "test_command": ["dotnet", "test"],
            "test_patterns": ["*Test.cs", "*Tests.cs"],
            "package_manager": "dotnet"
        },
        "php": {
            "extensions": [".php"],
            "compile_command": None,  # PHP is interpreted
            "run_command": ["php", "{file}"],
            "test_command": ["phpunit", "{test_dir}"],
            "test_patterns": ["*Test.php"],
            "package_manager": "composer"
        }
    }
    
    def __init__(
        self,
        name: str = "compiler",
        allowed_directories: List[str] = None,
        timeout_seconds: int = 300,
        max_output_size: int = 1024 * 1024,  # 1MB
        enable_network: bool = False
    ):
        super().__init__(
            name=name,
            description="Tool for compiling code, running programs, and executing tests",
            category=ToolCategory.COMPILER
        )
        
        self.allowed_directories = allowed_directories or ["."]
        self.timeout_seconds = timeout_seconds
        self.max_output_size = max_output_size
        self.enable_network = enable_network
        
        # Convert to Path objects and resolve
        self.allowed_paths = [Path(d).resolve() for d in self.allowed_directories]
        
        # Context for scenario files (will be set by the session)
        self.scenario_context = {}
        
        logger.info(f"CompilerTool initialized with timeout {timeout_seconds}s")
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the scenario context for this tool"""
        self.scenario_context = context or {}
    
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
        
        for language, config in self.LANGUAGE_CONFIGS.items():
            if file_ext in config["extensions"]:
                return language
        
        return None
    
    async def _run_command(
        self,
        command: List[str],
        working_directory: str = None,
        env: Dict[str, str] = None
    ) -> Tuple[int, str, str, float]:
        """Run a command and return exit code, stdout, stderr, and execution time"""
        start_time = time.time()
        
        try:
            # Set up environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Disable network if not enabled
            if not self.enable_network:
                process_env["http_proxy"] = "127.0.0.1:1"
                process_env["https_proxy"] = "127.0.0.1:1"
            
            # Run the command
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_directory,
                env=process_env
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                execution_time = time.time() - start_time
                return -1, "", f"Command timed out after {self.timeout_seconds}s", execution_time
            
            execution_time = time.time() - start_time
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')[:self.max_output_size]
            stderr_str = stderr.decode('utf-8', errors='replace')[:self.max_output_size]
            
            return process.returncode, stdout_str, stderr_str, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error running command {command}: {e}")
            return -1, "", str(e), execution_time
    
    @tool_function(
        description="Compile source code files",
        parameters=[
            ToolParameter("files", "array", "List of source files to compile", required=True),
            ToolParameter("language", "string", "Programming language (auto-detected if not specified)", required=False),
            ToolParameter("output_name", "string", "Name of output executable/artifact", required=False),
            ToolParameter("compiler_flags", "array", "Additional compiler flags", required=False, default=[]),
            ToolParameter("working_directory", "string", "Working directory for compilation", required=False, default=".")
        ],
        returns="Compilation result with success status, output, and artifacts",
        category=ToolCategory.COMPILER
    )
    async def compile_code(
        self,
        files: List[str],
        language: str = None,
        output_name: str = None,
        compiler_flags: List[str] = None,
        working_directory: str = "."
    ) -> Dict[str, Any]:
        """Compile source code files"""
        
        # CRITICAL: Use temp workspace if available (on-demand mode)
        # This ensures files are accessible when using empty/minimal context modes
        if self.scenario_context.get("_temp_workspace"):
            working_directory = self.scenario_context["_temp_workspace"]
            logger.debug(f"Using temp workspace for compilation: {working_directory}")
        elif self.scenario_context.get("working_directory"):
            working_directory = self.scenario_context["working_directory"]
        
        # Check permissions
        if not self._is_path_allowed(working_directory):
            raise PermissionError(f"Access denied to directory: {working_directory}")
        
        # For files in scenario context, check if they exist in project_files
        project_files = self.scenario_context.get("project_files", {})
        for file_path in files:
            # Check if file exists in project context
            full_path = Path(working_directory) / file_path
            if project_files:
                # Look for the file in project context using comprehensive path normalization
                # Don't double the working directory if file_path already contains it
                if str(file_path).startswith(working_directory):
                    project_key = str(file_path)
                else:
                    project_key = f"{working_directory}/{file_path}".replace("//", "/")
                
                # Apply the same normalization logic as FileSystemTool
                normalizations_to_try = [
                    str(file_path),  # Try the file path as-is first
                    project_key,     # Then try with working directory
                ]
                
                # CRITICAL FIX: Direct single-to-double slash conversion
                for path_to_normalize in [str(file_path), project_key]:
                    if "/" in path_to_normalize and "//" not in path_to_normalize:
                        double_slash_version = path_to_normalize.replace("/", "//")
                        normalizations_to_try.append(double_slash_version)
                
                # Also try double-to-single conversion
                if "//" in project_key:
                    single_slash_version = project_key.replace("//", "/")
                    normalizations_to_try.append(single_slash_version)
                
                # Remove duplicates
                normalizations_to_try = list(dict.fromkeys(normalizations_to_try))
                
                # Check if any normalization matches
                found = any(norm in project_files for norm in normalizations_to_try)
                
                if not found:
                    # In scenario context, missing files are expected - agents may try to compile
                    # files that aren't part of the selected scenario files. Log as debug instead of warning.
                    logger.debug(f"File {file_path} not found in scenario context (this is expected - scenario contains subset of project files)")
            
            if not self._is_path_allowed(str(full_path)):
                raise PermissionError(f"Access denied to file: {file_path}")
        
        # Create project files in a temporary directory if needed
        temp_dir = None
        actual_working_dir = working_directory
        
        if project_files and not Path(working_directory).exists():
            # Use local temp directory instead of system temp
            import uuid
            temp_base = Path("temp")
            temp_base.mkdir(exist_ok=True)
            
            session_id = str(uuid.uuid4())[:8]
            temp_dir = temp_base / f"compile_workspace_{session_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            actual_working_dir = str(temp_dir)
            
            # Create project files in temp directory
            for file_path, content in project_files.items():
                # Handle different path formats in project files
                relative_path = None
                
                if file_path.startswith(working_directory + "/"):
                    # Remove the working directory prefix with single slash
                    relative_path = file_path.replace(f"{working_directory}/", "")
                elif file_path.startswith(working_directory + "//"):
                    # Remove the working directory prefix with double slash
                    relative_path = file_path.replace(f"{working_directory}//", "")
                elif "//" in file_path and file_path.split("//")[0] == working_directory:
                    # Handle double slash format
                    relative_path = "//".join(file_path.split("//")[1:])
                else:
                    # For files that don't match the working directory pattern,
                    # use the file path as-is (this handles cases where the working
                    # directory doesn't match the actual project structure)
                    relative_path = file_path
                
                if relative_path:
                    # Normalize path separators and ensure it's not an absolute path
                    relative_path = relative_path.replace("//", "/")
                    # Remove leading slash if present to ensure relative path
                    if relative_path.startswith("/"):
                        relative_path = relative_path[1:]
                    
                    temp_file_path = Path(temp_dir) / relative_path
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        temp_file_path.write_text(content, encoding='utf-8')
                        logger.debug(f"Created temp file: {temp_file_path}")
                    except Exception as e:
                        logger.warning(f"Could not create temp file {temp_file_path}: {e}")
        
        start_time = time.time()
        compiler_flags = compiler_flags or []
        
        try:
            # Detect language if not specified
            if not language and files:
                language = self._detect_language(files[0])
            
            if not language:
                raise ValueError("Could not detect programming language")
            
            if language not in self.LANGUAGE_CONFIGS:
                raise ValueError(f"Unsupported language: {language}")
            
            config = self.LANGUAGE_CONFIGS[language]
            
            # Check if compilation is needed
            if config["compile_command"] is None:
                return CompilationResult(
                    success=True,
                    output=f"No compilation needed for {language}",
                    error_output="",
                    warnings=[],
                    execution_time=time.time() - start_time,
                    exit_code=0,
                    artifacts=files
                ).to_dict()
            
            # Prepare compilation command
            compile_cmd = config["compile_command"].copy()
            
            # Replace placeholders
            for i, arg in enumerate(compile_cmd):
                if "{file}" in arg:
                    if len(files) == 1:
                        compile_cmd[i] = arg.replace("{file}", files[0])
                    else:
                        # For multiple files, replace with all files
                        compile_cmd = compile_cmd[:i] + files + compile_cmd[i+1:]
                        break
                elif "{output}" in arg:
                    if not output_name:
                        # Generate output name
                        base_name = Path(files[0]).stem
                        output_name = f"{base_name}.out" if language in ["cpp", "c"] else base_name
                    compile_cmd[i] = arg.replace("{output}", output_name)
            
            # Add compiler flags
            compile_cmd.extend(compiler_flags)
            
            logger.info(f"Compiling {language} files: {files}")
            logger.debug(f"Compilation command: {compile_cmd}")
            
            # Run compilation
            exit_code, stdout, stderr, execution_time = await self._run_command(
                compile_cmd,
                working_directory=actual_working_dir
            )
            
            # Parse output for warnings
            warnings = self._parse_warnings(stderr, language)
            
            # Determine artifacts
            artifacts = []
            if exit_code == 0:
                if output_name:
                    artifacts.append(output_name)
                # Look for other generated files
                artifacts.extend(self._find_generated_files(working_directory, files, language))
            
            result = CompilationResult(
                success=(exit_code == 0),
                output=stdout,
                error_output=stderr,
                warnings=warnings,
                execution_time=execution_time,
                exit_code=exit_code,
                artifacts=artifacts
            )
            
            logger.info(f"Compilation {'succeeded' if result.success else 'failed'} in {execution_time:.2f}s")
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return CompilationResult(
                success=False,
                output="",
                error_output=str(e),
                warnings=[],
                execution_time=time.time() - start_time,
                exit_code=-1,
                artifacts=[]
            ).to_dict()
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")
    
    @tool_function(
        description="Run a compiled program or script",
        parameters=[
            ToolParameter("file_or_command", "string", "Executable file or command to run", required=True),
            ToolParameter("arguments", "array", "Command line arguments", required=False, default=[]),
            ToolParameter("working_directory", "string", "Working directory", required=False, default="."),
            ToolParameter("input_data", "string", "Data to pass to stdin", required=False, default="")
        ],
        returns="Execution result with output and exit code",
        category=ToolCategory.COMPILER
    )
    async def run_program(
        self,
        file_or_command: str,
        arguments: List[str] = None,
        working_directory: str = ".",
        input_data: str = ""
    ) -> Dict[str, Any]:
        """Run a compiled program or script"""
        
        # Check permissions
        if not self._is_path_allowed(working_directory):
            raise PermissionError(f"Access denied to directory: {working_directory}")
        
        arguments = arguments or []
        start_time = time.time()
        
        try:
            # Check if file_or_command exists and is executable
            full_path = os.path.join(working_directory, file_or_command) if not os.path.isabs(file_or_command) else file_or_command
            
            if os.path.isdir(file_or_command) or os.path.isdir(full_path):
                logger.debug(f"Cannot execute directory '{file_or_command}' (this is expected - agents sometimes try to run project directories)")
                return {
                    "success": False,
                    "exit_code": -1,
                    "output": "",
                    "error_output": f"Cannot execute directory '{file_or_command}'. Please specify an executable file or command.",
                    "execution_time": 0.0
                }
            
            # Prepare command
            command = [file_or_command] + arguments
            
            logger.info(f"Running program: {command}")
            
            # Run the program
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                cwd=working_directory
            )
            
            # Send input and wait for completion
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode() if input_data else None),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                execution_time = time.time() - start_time
                return {
                    "success": False,
                    "exit_code": -1,
                    "output": "",
                    "error_output": f"Program timed out after {self.timeout_seconds}s",
                    "execution_time": execution_time
                }
            
            execution_time = time.time() - start_time
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')[:self.max_output_size]
            stderr_str = stderr.decode('utf-8', errors='replace')[:self.max_output_size]
            
            result = {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "output": stdout_str,
                "error_output": stderr_str,
                "execution_time": execution_time
            }
            
            logger.info(f"Program {'succeeded' if result['success'] else 'failed'} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle common expected errors as debug messages
            error_msg = str(e)
            if "No such file or directory" in error_msg and any(dir_name in error_msg for dir_name in ["EduGate_ScholarLink", "pulsestream", "spotlight", "synapsecanvas", "edugate", "rate_limiter"]):
                logger.debug(f"Program execution failed (expected - agent tried to run project directory): {e}")
            else:
                logger.error(f"Program execution error: {e}")
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "error_output": str(e),
                "execution_time": time.time() - start_time
            }
    
    @tool_function(
        description="Run tests for a project",
        parameters=[
            ToolParameter("language", "string", "Programming language", required=True),
            ToolParameter("test_directory", "string", "Directory containing tests", required=False, default="tests"),
            ToolParameter("working_directory", "string", "Project root directory", required=False, default="."),
            ToolParameter("test_pattern", "string", "Pattern for test files", required=False),
            ToolParameter("coverage", "boolean", "Generate coverage report", required=False, default=False)
        ],
        returns="Test execution results with pass/fail counts and output",
        category=ToolCategory.COMPILER
    )
    async def run_tests(
        self,
        language: str,
        test_directory: str = "tests",
        working_directory: str = ".",
        test_pattern: str = None,
        coverage: bool = False
    ) -> Dict[str, Any]:
        """Run tests for a project"""
        
        # CRITICAL: Use temp workspace if available (on-demand mode)
        if self.scenario_context.get("_temp_workspace"):
            working_directory = self.scenario_context["_temp_workspace"]
            logger.debug(f"Using temp workspace for tests: {working_directory}")
        elif self.scenario_context.get("working_directory"):
            working_directory = self.scenario_context["working_directory"]
        
        # Check permissions
        if not self._is_path_allowed(working_directory):
            raise PermissionError(f"Access denied to directory: {working_directory}")
        
        # Create project files in a temporary directory if needed
        temp_dir = None
        actual_working_dir = working_directory
        project_files = self.scenario_context.get("project_files", {})
        
        if project_files and not Path(working_directory).exists():
            # Use local temp directory instead of system temp
            import uuid
            temp_base = Path("temp")
            temp_base.mkdir(exist_ok=True)
            
            session_id = str(uuid.uuid4())[:8]
            temp_dir = temp_base / f"test_workspace_{session_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            actual_working_dir = str(temp_dir)
            
            # Create project files in temp directory
            for file_path, content in project_files.items():
                # Handle different path formats in project files
                relative_path = None
                
                if file_path.startswith(working_directory + "/"):
                    # Remove the working directory prefix with single slash
                    relative_path = file_path.replace(f"{working_directory}/", "")
                elif file_path.startswith(working_directory + "//"):
                    # Remove the working directory prefix with double slash
                    relative_path = file_path.replace(f"{working_directory}//", "")
                elif "//" in file_path and file_path.split("//")[0] == working_directory:
                    # Handle double slash format
                    relative_path = "//".join(file_path.split("//")[1:])
                else:
                    # For files that don't match the working directory pattern,
                    # use the file path as-is (this handles cases where the working
                    # directory doesn't match the actual project structure)
                    relative_path = file_path
                
                if relative_path:
                    # Normalize path separators and ensure it's not an absolute path
                    relative_path = relative_path.replace("//", "/")
                    # Remove leading slash if present to ensure relative path
                    if relative_path.startswith("/"):
                        relative_path = relative_path[1:]
                    
                    temp_file_path = Path(temp_dir) / relative_path
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        temp_file_path.write_text(content, encoding='utf-8')
                        logger.debug(f"Created temp file: {temp_file_path}")
                    except Exception as e:
                        logger.warning(f"Could not create temp file {temp_file_path}: {e}")
        
        start_time = time.time()
        
        try:
            if language not in self.LANGUAGE_CONFIGS:
                raise ValueError(f"Unsupported language: {language}")
            
            config = self.LANGUAGE_CONFIGS[language]
            
            # Prepare test command with Maven fallback for Java
            test_cmd = config["test_command"].copy()
            
            # Special handling for Java - check if Maven is available
            if language == "java" and test_cmd[0] == "mvn":
                # Check if Maven is available
                try:
                    process = await asyncio.create_subprocess_exec(
                        "mvn", "--version",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await process.wait()
                    if process.returncode != 0:
                        raise FileNotFoundError("Maven not available")
                except (FileNotFoundError, OSError):
                    # Maven not available, skip tests gracefully
                    logger.info("Maven not found, skipping Java tests (this is expected in environments without Maven)")
                    return TestResult(
                        success=True,  # Don't fail the evaluation
                        tests_run=0,
                        tests_passed=0,
                        tests_failed=0,
                        test_output="Maven not available - tests skipped",
                        execution_time=time.time() - start_time,
                        coverage_info=None
                    ).to_dict()
            
            # Replace placeholders
            for i, arg in enumerate(test_cmd):
                if "{test_dir}" in arg:
                    test_cmd[i] = arg.replace("{test_dir}", test_directory)
            
            # Add coverage flags if requested
            if coverage:
                if language == "python":
                    test_cmd = ["python", "-m", "pytest", "--cov=.", test_directory, "-v"]
                elif language == "javascript" or language == "typescript":
                    test_cmd = ["npm", "run", "test:coverage"]
                elif language == "go":
                    test_cmd = ["go", "test", "-cover", "./..."]
            
            logger.info(f"Running {language} tests in {test_directory}")
            logger.debug(f"Test command: {test_cmd}")
            
            # Run tests
            exit_code, stdout, stderr, execution_time = await self._run_command(
                test_cmd,
                working_directory=actual_working_dir
            )
            
            # Parse test results
            test_results = self._parse_test_results(stdout, stderr, language)
            
            result = TestResult(
                success=(exit_code == 0),
                tests_run=test_results.get("tests_run", 0),
                tests_passed=test_results.get("tests_passed", 0),
                tests_failed=test_results.get("tests_failed", 0),
                test_output=stdout,
                execution_time=execution_time,
                coverage_info=test_results.get("coverage") if coverage else None
            )
            
            logger.info(f"Tests {'passed' if result.success else 'failed'}: {result.tests_passed}/{result.tests_run}")
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            return TestResult(
                success=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                test_output=str(e),
                execution_time=time.time() - start_time
            ).to_dict()
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")
    
    def _parse_warnings(self, stderr: str, language: str) -> List[str]:
        """Parse compiler warnings from stderr"""
        warnings = []
        
        warning_patterns = {
            "cpp": [r"warning:", r"note:"],
            "c": [r"warning:", r"note:"],
            "java": [r"warning:", r"Note:"],
            "csharp": [r"warning CS\d+:"],
            "go": [r"warning:"],
            "rust": [r"warning:"]
        }
        
        patterns = warning_patterns.get(language, [r"warning:"])
        
        for line in stderr.split('\n'):
            if any(pattern in line.lower() for pattern in patterns):
                warnings.append(line.strip())
        
        return warnings
    
    def _find_generated_files(self, working_directory: str, source_files: List[str], language: str) -> List[str]:
        """Find files generated during compilation"""
        artifacts = []
        work_dir = Path(working_directory)
        
        # Common artifact patterns by language
        patterns = {
            "java": ["*.class"],
            "typescript": ["*.js", "*.js.map"],
            "csharp": ["*.exe", "*.dll"],
            "go": [f"{Path(f).stem}" for f in source_files],  # Go binaries
            "rust": ["target/debug/*", "target/release/*"]
        }
        
        if language in patterns:
            for pattern in patterns[language]:
                for artifact in work_dir.glob(pattern):
                    if artifact.is_file():
                        artifacts.append(str(artifact.relative_to(work_dir)))
        
        return artifacts
    
    def _parse_test_results(self, stdout: str, stderr: str, language: str) -> Dict[str, Any]:
        """Parse test results from output"""
        results = {"tests_run": 0, "tests_passed": 0, "tests_failed": 0}
        
        output = stdout + stderr
        
        if language == "python":
            # Parse pytest output
            import re
            match = re.search(r"(\d+) passed", output)
            if match:
                results["tests_passed"] = int(match.group(1))
            
            match = re.search(r"(\d+) failed", output)
            if match:
                results["tests_failed"] = int(match.group(1))
            
            results["tests_run"] = results["tests_passed"] + results["tests_failed"]
            
        elif language == "go":
            # Parse go test output
            import re
            if "PASS" in output:
                # Count test functions
                test_count = len(re.findall(r"--- PASS:", output))
                results["tests_passed"] = test_count
                results["tests_run"] = test_count
            elif "FAIL" in output:
                pass_count = len(re.findall(r"--- PASS:", output))
                fail_count = len(re.findall(r"--- FAIL:", output))
                results["tests_passed"] = pass_count
                results["tests_failed"] = fail_count
                results["tests_run"] = pass_count + fail_count
        
        elif language in ["javascript", "typescript"]:
            # Parse Jest/npm test output
            import re
            match = re.search(r"Tests:\s+(\d+) passed", output)
            if match:
                results["tests_passed"] = int(match.group(1))
            
            match = re.search(r"(\d+) failed", output)
            if match:
                results["tests_failed"] = int(match.group(1))
            
            results["tests_run"] = results["tests_passed"] + results["tests_failed"]
        
        return results
