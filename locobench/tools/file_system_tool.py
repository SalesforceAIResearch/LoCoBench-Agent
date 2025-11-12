"""
File System Tool for LoCoBench-Agent

This tool provides file system operations that agents can use to read,
write, and navigate files during evaluation sessions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..core.tool_registry import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class FileSystemTool(Tool):
    """
    Tool for file system operations
    
    Provides safe file system access with configurable restrictions
    to prevent agents from accessing unauthorized areas.
    """
    
    def __init__(
        self,
        name: str = "file_system",
        allowed_directories: List[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        readonly_mode: bool = False
    ):
        super().__init__(
            name=name,
            description="Tool for reading, writing, and navigating files and directories",
            category=ToolCategory.FILE_SYSTEM
        )
        
        self.allowed_directories = allowed_directories or ["."]
        self.max_file_size = max_file_size
        self.readonly_mode = readonly_mode
        
        # Convert to Path objects and resolve
        self.allowed_paths = [Path(d).resolve() for d in self.allowed_directories]
        
        # Context for scenario files (will be set by the session)
        self.scenario_context = {}
        
        # Temporary directory for agent-created files (to prevent polluting main codebase)
        self.temp_workspace = None
        
        logger.info(f"FileSystemTool initialized with allowed directories: {self.allowed_directories}")
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the scenario context for this tool"""
        self.scenario_context = context or {}
        
        # Set up temporary workspace for agent file operations
        if not self.temp_workspace:
            # Use local temp directory instead of system temp
            import uuid
            temp_base = Path("temp")
            temp_base.mkdir(exist_ok=True)
            
            session_id = str(uuid.uuid4())[:8]  # Short unique ID
            self.temp_workspace = temp_base / f"agent_workspace_{session_id}"
            self.temp_workspace.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created temporary workspace: {self.temp_workspace}")
            
            # Store temp workspace path in scenario_context for other tools (e.g., compiler)
            self.scenario_context["_temp_workspace"] = str(self.temp_workspace)
            
            # Copy project files to temp workspace if available
            # Use _all_files_for_tools if available (on-demand mode), otherwise use project_files
            files_to_copy = self.scenario_context.get("_all_files_for_tools") or self.scenario_context.get("project_files", {})
            
            if files_to_copy:
                # Handle both dict {path: content} and list [{path, content}, ...] formats
                file_count = 0
                if isinstance(files_to_copy, dict):
                    items_to_copy = files_to_copy.items()
                elif isinstance(files_to_copy, list):
                    items_to_copy = [(item['path'], item['content']) for item in files_to_copy if isinstance(item, dict) and 'path' in item and 'content' in item]
                else:
                    items_to_copy = []
                
                for file_path, content in items_to_copy:
                    # Normalize path to remove double slashes and make it relative
                    normalized_path = file_path.replace("//", "/")
                    if normalized_path.startswith("/"):
                        normalized_path = normalized_path[1:]
                    
                    temp_file_path = self.temp_workspace / normalized_path
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_file_path.write_text(content, encoding='utf-8')
                    file_count += 1
                
                logger.info(f"✅ Copied {file_count} project files to temp workspace: {self.temp_workspace}")
    
    def cleanup_temp_workspace(self) -> None:
        """Clean up the temporary workspace"""
        if self.temp_workspace and self.temp_workspace.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_workspace)
                logger.info(f"Cleaned up temporary workspace: {self.temp_workspace}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary workspace {self.temp_workspace}: {e}")
            finally:
                self.temp_workspace = None
    
    def _list_directory_from_context(self, path: str, project_files: Dict[str, str], include_hidden: bool, recursive: bool) -> List[Dict[str, Any]]:
        """List directory contents from project context files"""
        # Normalize the path
        path = path.rstrip('/')
        if path == '.':
            path = ''
        
        # Try different path normalizations
        normalizations_to_try = [
            path,
            path.replace('//', '/'),
            path.replace('/', '//'),
        ]
        
        # Remove duplicates while preserving order
        normalizations_to_try = list(dict.fromkeys(normalizations_to_try))
        
        items = []
        directories = set()
        
        for file_path in project_files.keys():
            for normalized_path in normalizations_to_try:
                if normalized_path == '':
                    # Root directory - include all files
                    parts = file_path.split('/')
                    if len(parts) > 1:
                        # This is a file in a subdirectory
                        dir_name = parts[0]
                        if dir_name and (include_hidden or not dir_name.startswith('.')):
                            directories.add(dir_name)
                    else:
                        # This is a file in the root
                        if include_hidden or not parts[0].startswith('.'):
                            file_name = parts[0]
                            items.append({
                                "name": file_name,
                                "path": file_name,
                                "absolute_path": file_path,
                                "type": "file",
                                "size": len(project_files[file_path]),
                                "modified": 0,  # Not available in context
                                "extension": Path(file_name).suffix if '.' in file_name else ""
                            })
                elif file_path.startswith(normalized_path + '/') or file_path.startswith(normalized_path + '//'):
                    # This file is in the requested directory
                    relative_path = file_path[len(normalized_path):].lstrip('/')
                    if '/' in relative_path and not recursive:
                        # This is in a subdirectory, add the directory name
                        dir_name = relative_path.split('/')[0]
                        if dir_name and (include_hidden or not dir_name.startswith('.')):
                            directories.add(dir_name)
                    else:
                        # This is a direct file in the directory
                        if include_hidden or not relative_path.startswith('.'):
                            items.append({
                                "name": relative_path,
                                "path": relative_path,
                                "absolute_path": file_path,
                                "type": "file",
                                "size": len(project_files[file_path]),
                                "modified": 0,  # Not available in context
                                "extension": Path(relative_path).suffix if '.' in relative_path else ""
                            })
                    break
        
        # Add directories
        for dir_name in directories:
            items.append({
                "name": dir_name,
                "path": dir_name,
                "absolute_path": f"{path}/{dir_name}" if path else dir_name,
                "type": "directory",
                "size": None,
                "modified": 0,
                "permissions": "755"
            })
        
        logger.debug(f"Listed {len(items)} items from project context for path: {path}")
        return items
    
    def _search_files_from_context(self, pattern: str, directory: str, project_files: Dict[str, str], recursive: bool) -> List[Dict[str, Any]]:
        """Search for files matching a pattern in project context"""
        import fnmatch
        
        # Normalize the directory path
        directory = directory.rstrip('/')
        if directory == '.':
            directory = ''
        
        # Try different path normalizations for the directory
        dir_normalizations = [
            directory,
            directory.replace('//', '/'),
            directory.replace('/', '//'),
        ]
        
        # Remove duplicates while preserving order
        dir_normalizations = list(dict.fromkeys(dir_normalizations))
        
        matches = []
        
        for file_path in project_files.keys():
            file_name = Path(file_path).name
            
            # Check if the file matches the pattern
            if not fnmatch.fnmatch(file_name, pattern):
                continue
            
            # Check if the file is in the requested directory
            file_in_directory = False
            relative_path = ""
            
            for normalized_dir in dir_normalizations:
                if normalized_dir == '':
                    # Root directory - all files are candidates
                    if recursive or '/' not in file_path.strip('/'):
                        file_in_directory = True
                        relative_path = file_path
                        break
                elif file_path.startswith(normalized_dir + '/') or file_path.startswith(normalized_dir + '//'):
                    # File is in the requested directory
                    relative_path = file_path[len(normalized_dir):].lstrip('/')
                    if recursive or '/' not in relative_path:
                        file_in_directory = True
                        break
            
            if file_in_directory:
                match_info = {
                    "name": file_name,
                    "path": relative_path,
                    "absolute_path": file_path,
                    "type": "file",
                    "size": len(project_files[file_path]),
                    "modified": 0,  # Not available in context
                    "extension": Path(file_name).suffix if '.' in file_name else ""
                }
                matches.append(match_info)
        
        logger.debug(f"Found {len(matches)} matches for pattern '{pattern}' in directory '{directory}' from project context")
        return matches
    
    def _path_structures_match(self, requested_path: str, available_path: str) -> bool:
        """Check if two paths represent the same file with different slash formats"""
        # Normalize both paths to compare structure
        req_normalized = requested_path.replace("//", "/").replace("\\", "/").strip("/")
        avail_normalized = available_path.replace("//", "/").replace("\\", "/").strip("/")
        
        # Split into components
        req_parts = [p for p in req_normalized.split("/") if p]
        avail_parts = [p for p in avail_normalized.split("/") if p]
        
        # If the requested path is a suffix of the available path, it's a match
        if len(req_parts) <= len(avail_parts):
            return req_parts == avail_parts[-len(req_parts):]
        
        return False
    
    def _is_path_allowed(self, path: Union[str, Path]) -> bool:
        """Check if a path is within allowed directories"""
        try:
            resolved_path = Path(path).resolve()
            
            # Special case: prevent access to root directory and system paths
            if str(resolved_path) in ['/', '/root', '/etc', '/usr', '/var', '/sys', '/proc']:
                logger.debug(f"Blocked access to system directory: {path}")  # Reduce log level
                return False
            
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
    
    def _safe_path_operation(self, path: str, operation: str) -> Dict[str, Any]:
        """Safely perform a path operation with permission checks"""
        if not self._is_path_allowed(path):
            # Provide more helpful error messages
            if path in ['/', '/root', '/etc', '/usr', '/var', '/sys', '/proc']:
                # Get project name from scenario context for better guidance
                project_name = "the project directory"
                if self.scenario_context.get("project_files"):
                    # Try to extract project name from first file path
                    first_file = next(iter(self.scenario_context["project_files"].keys()), "")
                    if "//" in first_file:
                        project_name = first_file.split("//")[0]
                
                return {
                    "success": False,
                    "error": f"Access denied: Cannot access system directory '{path}'. For security reasons, you can only access project files. Try: list_directory('{project_name}') or list_directory('.') to explore the current project instead."
                }
            else:
                return {
                    "success": False,
                    "error": f"Access denied: Path '{path}' is outside allowed project directories. Available directories: {[str(p) for p in self.allowed_paths]}"
                }
        
        if self.readonly_mode and operation in ["write", "create", "delete"]:
            return {
                "success": False,
                "error": f"Operation '{operation}' not allowed in readonly mode"
            }
        
        return {"success": True}
    
    @tool_function(
        description="Read the contents of a text file",
        parameters=[
            ToolParameter("path", "string", "Path to the file to read", required=True),
            ToolParameter("encoding", "string", "File encoding (default: utf-8)", required=False, default="utf-8")
        ],
        returns="Content of the file as a string",
        category=ToolCategory.FILE_SYSTEM
    )
    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read a text file and return its contents"""
        permission_check = self._safe_path_operation(path, "read")
        if not permission_check["success"]:
            raise PermissionError(permission_check["error"])
        
        try:
            # First, check if the file exists in the scenario context
            # Use _all_files_for_tools if available (on-demand mode), otherwise use project_files
            project_files_raw = self.scenario_context.get("_all_files_for_tools") or self.scenario_context.get("project_files", {})
            
            # Normalize to dict format for easier access
            if isinstance(project_files_raw, list):
                project_files = {item['path']: item['content'] for item in project_files_raw if isinstance(item, dict) and 'path' in item and 'content' in item}
            elif isinstance(project_files_raw, dict):
                project_files = project_files_raw
            else:
                project_files = {}
            
            logger.debug(f"FileSystemTool.read_file: Requested path: {path}")
            logger.debug(f"FileSystemTool.read_file: Available project files count: {len(project_files)}")
            if len(project_files) > 0:
                logger.debug(f"FileSystemTool.read_file: Sample available paths: {list(project_files.keys())[:5]}")
            
            if path in project_files:
                content = project_files[path]
                logger.debug(f"Read file from context (exact match): {path} ({len(content)} characters)")
                
                # CRITICAL: Limit content size to prevent 20MB message errors
                max_content_size = 8_000_000  # 8MB safety limit (OpenAI limit is 10MB)
                if len(content) > max_content_size:
                    logger.warning(f"File {path} too large ({len(content)} chars), truncating to {max_content_size}")
                    content = content[:max_content_size] + f"\n\n[Content truncated - file was {len(project_files[path])} characters, showing first {max_content_size}]"
                
                return content
            
            # ENHANCED PATH NORMALIZATION - Handle single/double slash variations more effectively
            normalizations_to_try = [
                path,  # Original path
            ]
            
            # CRITICAL FIX: Direct single-to-double slash conversion (highest priority)
            if "/" in path and "//" not in path:
                # Convert all single slashes to double slashes
                double_slash_version = path.replace("/", "//")
                normalizations_to_try.append(double_slash_version)
            
            # Also try double-to-single conversion
            if "//" in path:
                single_slash_version = path.replace("//", "/")
                normalizations_to_try.append(single_slash_version)
            
            # Additional normalizations
            normalizations_to_try.extend([
                path.replace("\\", "/"),  # Backslash to forward slash
                path.replace("\\", "//"),  # Backslash to double slash
                # Try partial matches - sometimes files have different base paths
                "/" + path if not path.startswith("/") else path,
                "//" + path if not path.startswith("//") else path,
            ])
            
            # Also try to find exact matches by looking for any file that ends with our path
            # This handles cases where the prefix might be different
            path_suffix = path
            if "/" in path:
                # Try progressively shorter suffixes
                path_parts = path.split("/")
                for i in range(len(path_parts)):
                    suffix = "/".join(path_parts[i:])
                    suffix_double = "//".join(path_parts[i:])
                    normalizations_to_try.extend([
                        suffix,
                        suffix_double,
                        "//" + suffix,
                        "/" + suffix
                    ])
            
            # Also try fuzzy matching - look for files that end with the same path
            path_parts = path.split('/')
            if len(path_parts) > 1:
                # Try matching by the last few components
                for i in range(1, min(4, len(path_parts))):
                    suffix = '/'.join(path_parts[-i:])
                    suffix_double = '//'.join(path_parts[-i:])
                    normalizations_to_try.extend([
                        suffix,
                        suffix_double,
                        "//" + suffix,
                        "/" + suffix
                    ])
            
            # Remove duplicates while preserving order
            normalizations_to_try = list(dict.fromkeys(normalizations_to_try))
            
            for normalized_path in normalizations_to_try:
                if normalized_path in project_files:
                    content = project_files[normalized_path]
                    logger.debug(f"Read file from context (normalized {path} -> {normalized_path}): ({len(content)} characters)")
                    
                    # CRITICAL: Limit content size to prevent 20MB message errors
                    max_content_size = 8_000_000  # 8MB safety limit (OpenAI limit is 10MB)
                    if len(content) > max_content_size:
                        logger.warning(f"File {normalized_path} too large ({len(content)} chars), truncating to {max_content_size}")
                        content = content[:max_content_size] + f"\n\n[Content truncated - file was {len(project_files[normalized_path])} characters, showing first {max_content_size}]"
                    
                    return content
                else:
                    logger.debug(f"Normalization attempt failed: {normalized_path} not in project_files")
            
            # Log normalization failure for debugging (reduced verbosity)
            logger.debug(f"Path normalization failed for: {path} (tried {len(normalizations_to_try)} variations)")
            
            # Check if any of our normalizations should have worked
            logger.debug(f"All normalization attempts failed for: {path}")
            logger.debug(f"Normalizations tried: {normalizations_to_try[:3]}...")  # Show first 3
            
            # If all normalizations fail, try smart fuzzy matching
            # Look for files that end with the same path structure
            filename = Path(path).name
            if filename:
                logger.debug(f"Trying fuzzy matching for filename: {filename}")
                # Find files with the same filename
                for file_path, content in project_files.items():
                    if file_path.endswith(filename) or file_path.endswith("/" + filename) or file_path.endswith("//" + filename):
                        logger.debug(f"Found file with matching name: {file_path}")
                        # Check if the path structure matches (allowing for different slash formats)
                        if self._path_structures_match(path, file_path):
                            logger.debug(f"Read file from context (fuzzy match {path} -> {file_path}): ({len(content)} characters)")
                            
                            # CRITICAL: Limit content size to prevent 20MB message errors
                            max_content_size = 8_000_000  # 8MB safety limit (OpenAI limit is 10MB)
                            if len(content) > max_content_size:
                                logger.warning(f"File {file_path} too large ({len(content)} chars), truncating to {max_content_size}")
                                content = content[:max_content_size] + f"\n\n[Content truncated - file was {len(content)} characters, showing first {max_content_size}]"
                            
                            return content
                        else:
                            logger.debug(f"Path structure doesn't match: {path} vs {file_path}")
            
            # Only log debug message if file is truly not found after all normalization attempts
            logger.debug(f"File not found in project context after normalization: {path}")
            prefix = path.split('/')[0] if '/' in path else path.split('//')[0]
            matching_files = [k for k in project_files.keys() if k.startswith(prefix) or k.startswith(prefix + "//")]
            
            # Try to find similar files
            similar_files = []
            filename = Path(path).name
            if filename:
                similar_files = [k for k in project_files.keys() if filename in k]
            
            # Show a diverse sample of available files (not just first 10 alphabetically)
            sample_files = []
            if matching_files:
                # Include some from the beginning
                sample_files.extend(matching_files[:3])
                # Include some source files if they exist
                src_files = [f for f in matching_files if '/src/' in f or '//src//' in f][:4]
                sample_files.extend(src_files)
                # Include some from the end if space allows
                if len(sample_files) < 8:
                    sample_files.extend(matching_files[-(8-len(sample_files)):])
                # Remove duplicates while preserving order
                seen = set()
                sample_files = [f for f in sample_files if not (f in seen or seen.add(f))]
            
            logger.warning(f"Available files starting with prefix '{prefix}': {sample_files[:10]}")
            if similar_files:
                logger.warning(f"Files with similar names containing '{filename}': {similar_files[:5]}")
            logger.warning(f"Tried normalizations: {normalizations_to_try[:5]}...")  # Limit output
            
            # Create a helpful error message for the agent with EXACT COMPLETE paths
            all_files = list(project_files.keys())
            source_files = [f for f in all_files if any(pattern in f for pattern in ['/src/', '//src//', '.c', '.cpp', '.h', '.hpp', '.py', '.js', '.ts', '.java', '.rs'])]
            config_files = [f for f in all_files if any(pattern in f for pattern in ['config', 'package.json', 'pom.xml', 'Cargo.toml', 'CMakeLists.txt'])]
            
            helpful_error = f"❌ FILE NOT FOUND: '{path}'\n\n"
            helpful_error += f"📋 This file was not included in this evaluation scenario.\n"
            helpful_error += f"The scenario contains {len(all_files)} files. Here are the EXACT paths available:\n\n"
            
            if source_files:
                helpful_error += f"📄 SOURCE FILES ({len(source_files)} total) - Use EXACT paths:\n"
                for f in source_files[:8]:  # Show more examples with complete paths
                    helpful_error += f"   • {f}\n"
                if len(source_files) > 8:
                    helpful_error += f"   ... and {len(source_files) - 8} more source files\n"
                helpful_error += "\n"
            
            if config_files:
                helpful_error += f"⚙️  CONFIG FILES ({len(config_files)} total) - Use EXACT paths:\n"
                for f in config_files[:6]:
                    helpful_error += f"   • {f}\n"
                if len(config_files) > 6:
                    helpful_error += f"   ... and {len(config_files) - 6} more config files\n"
                helpful_error += "\n"
            
            if matching_files and len(matching_files) <= 15:
                # If there are few matching files, show ALL of them with complete paths
                helpful_error += f"🎯 FILES WITH PREFIX '{prefix}' ({len(matching_files)} total):\n"
                for f in matching_files:
                    helpful_error += f"   • {f}\n"
                helpful_error += "\n"
            
            if similar_files and len(similar_files) <= 10:
                helpful_error += f"🔍 FILES CONTAINING '{filename}':\n"
                for f in similar_files:
                    helpful_error += f"   • {f}\n"
                helpful_error += "\n"
            
            helpful_error += f"⚠️  CRITICAL: Use EXACT paths shown above. Do NOT modify or guess paths.\n"
            helpful_error += f"💡 Use list_directory() to explore directory contents, or search_files() to find specific files."
            
            logger.info(f"Generated helpful error message for agent about missing file: {path}")
            
            # Return helpful error instead of falling back to file system
            # This prevents agents from accessing files outside the scenario context
            raise FileNotFoundError(helpful_error)
            
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise
    
    @tool_function(
        description="Write content to a text file",
        parameters=[
            ToolParameter("path", "string", "Path to the file to write", required=True),
            ToolParameter("content", "string", "Content to write to the file", required=True),
            ToolParameter("encoding", "string", "File encoding (default: utf-8)", required=False, default="utf-8"),
            ToolParameter("create_directories", "boolean", "Create parent directories if they don't exist", required=False, default=True)
        ],
        returns="Success message with file information",
        category=ToolCategory.FILE_SYSTEM
    )
    def write_file(self, path: str, content: str, encoding: str = "utf-8", create_directories: bool = True) -> str:
        """Write content to a text file"""
        # CRITICAL FIX: Use temporary workspace to prevent polluting main codebase
        if self.temp_workspace:
            # Write to temporary workspace instead of main directory
            normalized_path = path.replace("//", "/")
            if normalized_path.startswith("/"):
                normalized_path = normalized_path[1:]
            
            file_path = self.temp_workspace / normalized_path
            
            # Check content size
            content_size = len(content.encode(encoding))
            if content_size > self.max_file_size:
                raise ValueError(f"Content too large: {content_size} bytes (max: {self.max_file_size})")
            
            try:
                # Create parent directories if needed
                if create_directories:
                    # CRITICAL FIX: Handle conflicts when creating parent directories
                    # If any part of the path exists as a file (not directory), use alternative naming
                    parent_path = file_path.parent
                    path_parts = list(parent_path.parts)
                    
                    # Check each part of the path for conflicts
                    current_path = Path(path_parts[0]) if path_parts else Path(".")
                    for i, part in enumerate(path_parts[1:], 1):
                        next_path = current_path / part
                        
                        # If a file exists with this name, we need to use alternative naming
                        if next_path.exists() and next_path.is_file():
                            logger.warning(f"File exists at {next_path}, using alternative directory structure")
                            # Create alternative path by adding suffix to conflicting part
                            alt_part = f"{part}_dir"
                            counter = 1
                            while (current_path / alt_part).exists():
                                alt_part = f"{part}_dir_{counter}"
                                counter += 1
                            
                            # Update the file path to use the alternative structure
                            remaining_parts = path_parts[i:]
                            new_parent = current_path / alt_part / Path(*remaining_parts)
                            file_path = new_parent / file_path.name
                            break
                        
                        current_path = next_path
                    
                    # Now create the (possibly modified) parent directories
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle case where the target file path is a directory
                if file_path.exists() and file_path.is_dir():
                    logger.warning(f"Directory exists with same name as target file {file_path}")
                    # Use alternative filename
                    alt_name = f"{file_path.stem}_file{file_path.suffix}"
                    counter = 1
                    while (file_path.parent / alt_name).exists():
                        alt_name = f"{file_path.stem}_file_{counter}{file_path.suffix}"
                        counter += 1
                    file_path = file_path.parent / alt_name
                    logger.info(f"Using alternative filename: {file_path}")
                
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                logger.info(f"Wrote file to temp workspace: {normalized_path} ({len(content)} characters)")
                return f"Successfully wrote {len(content)} characters to {path}"
                
            except Exception as e:
                logger.error(f"Error writing file to temp workspace {file_path}: {e}")
                raise
        else:
            # Fallback to original behavior if no temp workspace (shouldn't happen in agent mode)
            permission_check = self._safe_path_operation(path, "write")
            if not permission_check["success"]:
                raise PermissionError(permission_check["error"])
            
            try:
                file_path = Path(path)
                
                # Check content size
                content_size = len(content.encode(encoding))
                if content_size > self.max_file_size:
                    raise ValueError(f"Content too large: {content_size} bytes (max: {self.max_file_size})")
                
                # Create parent directories if needed
                if create_directories:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                logger.info(f"Wrote file: {path} ({len(content)} characters)")
                return f"Successfully wrote {len(content)} characters to {path}"
                
            except Exception as e:
                logger.error(f"Error writing file {path}: {e}")
                raise
    
    @tool_function(
        description="List files and directories in a directory",
        parameters=[
            ToolParameter("path", "string", "Path to the directory to list", required=True),
            ToolParameter("include_hidden", "boolean", "Include hidden files (starting with .)", required=False, default=False),
            ToolParameter("recursive", "boolean", "List files recursively", required=False, default=False)
        ],
        returns="List of files and directories with metadata",
        category=ToolCategory.FILE_SYSTEM
    )
    def list_directory(self, path: str, include_hidden: bool = False, recursive: bool = False) -> List[Dict[str, Any]]:
        """List contents of a directory"""
        permission_check = self._safe_path_operation(path, "read")
        if not permission_check["success"]:
            raise PermissionError(permission_check["error"])
        
        try:
            # First, try to list from project context
            # Use _all_files_for_tools if available (on-demand mode), otherwise use project_files
            project_files_raw = self.scenario_context.get("_all_files_for_tools") or self.scenario_context.get("project_files", {})
            
            # Normalize to dict format for easier access
            if isinstance(project_files_raw, list):
                project_files = {item['path']: item['content'] for item in project_files_raw if isinstance(item, dict) and 'path' in item and 'content' in item}
            elif isinstance(project_files_raw, dict):
                project_files = project_files_raw
            else:
                project_files = {}
            
            if project_files:
                logger.debug(f"Listing directory from project context: {path}")
                return self._list_directory_from_context(path, project_files, include_hidden, recursive)
            
            # Fall back to filesystem if no project context
            dir_path = Path(path)
            
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            if not dir_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            
            items = []
            
            if recursive:
                pattern = "**/*"
                iterator = dir_path.rglob(pattern)
            else:
                iterator = dir_path.iterdir()
            
            for item in iterator:
                # Skip hidden files if not requested
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                # Check if item is within allowed directories
                if not self._is_path_allowed(item):
                    continue
                
                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "path": str(item.relative_to(dir_path)),
                        "absolute_path": str(item),
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime,
                        "permissions": oct(stat.st_mode)[-3:]
                    }
                    
                    # Add file extension for files
                    if item.is_file():
                        item_info["extension"] = item.suffix
                    
                    items.append(item_info)
                    
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not access {item}: {e}")
                    continue
            
            logger.debug(f"Listed directory: {path} ({len(items)} items)")
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            raise
    
    @tool_function(
        description="Check if a file or directory exists",
        parameters=[
            ToolParameter("path", "string", "Path to check", required=True)
        ],
        returns="Boolean indicating whether the path exists and metadata if it exists",
        category=ToolCategory.FILE_SYSTEM
    )
    def path_exists(self, path: str) -> Dict[str, Any]:
        """Check if a path exists and return metadata"""
        permission_check = self._safe_path_operation(path, "read")
        if not permission_check["success"]:
            return {
                "exists": False,
                "error": permission_check["error"]
            }
        
        try:
            file_path = Path(path)
            exists = file_path.exists()
            
            result = {"exists": exists}
            
            if exists:
                stat = file_path.stat()
                result.update({
                    "type": "directory" if file_path.is_dir() else "file",
                    "size": stat.st_size if file_path.is_file() else None,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:]
                })
                
                if file_path.is_file():
                    result["extension"] = file_path.suffix
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking path {path}: {e}")
            return {
                "exists": False,
                "error": str(e)
            }
    
    @tool_function(
        description="Search for files matching a pattern",
        parameters=[
            ToolParameter("pattern", "string", "Glob pattern to search for", required=True),
            ToolParameter("directory", "string", "Directory to search in (default: current directory)", required=False, default="."),
            ToolParameter("recursive", "boolean", "Search recursively", required=False, default=True)
        ],
        returns="List of matching files with metadata",
        category=ToolCategory.FILE_SYSTEM
    )
    def search_files(self, pattern: str, directory: str = ".", recursive: bool = True) -> List[Dict[str, Any]]:
        """Search for files matching a pattern"""
        permission_check = self._safe_path_operation(directory, "read")
        if not permission_check["success"]:
            raise PermissionError(permission_check["error"])
        
        try:
            # First, try to search from project context
            # Use _all_files_for_tools if available (on-demand mode), otherwise use project_files
            project_files_raw = self.scenario_context.get("_all_files_for_tools") or self.scenario_context.get("project_files", {})
            
            # Normalize to dict format for easier access
            if isinstance(project_files_raw, list):
                project_files = {item['path']: item['content'] for item in project_files_raw if isinstance(item, dict) and 'path' in item and 'content' in item}
            elif isinstance(project_files_raw, dict):
                project_files = project_files_raw
            else:
                project_files = {}
            
            if project_files:
                logger.debug(f"Searching files from project context: pattern='{pattern}', directory='{directory}'")
                return self._search_files_from_context(pattern, directory, project_files, recursive)
            
            # Fall back to filesystem if no project context
            search_path = Path(directory)
            
            if not search_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            if not search_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {directory}")
            
            matches = []
            
            if recursive:
                iterator = search_path.rglob(pattern)
            else:
                iterator = search_path.glob(pattern)
            
            for match in iterator:
                # Check if match is within allowed directories
                if not self._is_path_allowed(match):
                    continue
                
                try:
                    stat = match.stat()
                    match_info = {
                        "name": match.name,
                        "path": str(match.relative_to(search_path)),
                        "absolute_path": str(match),
                        "type": "directory" if match.is_dir() else "file",
                        "size": stat.st_size if match.is_file() else None,
                        "modified": stat.st_mtime
                    }
                    
                    if match.is_file():
                        match_info["extension"] = match.suffix
                    
                    matches.append(match_info)
                    
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not access {match}: {e}")
                    continue
            
            logger.debug(f"Found {len(matches)} matches for pattern '{pattern}' in {directory}")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching for pattern {pattern} in {directory}: {e}")
            raise
    
    @tool_function(
        description="Get the current working directory",
        parameters=[],
        returns="Current working directory path",
        category=ToolCategory.FILE_SYSTEM
    )
    def get_current_directory(self) -> str:
        """Get the current working directory"""
        try:
            current_dir = Path.cwd()
            
            # Check if current directory is allowed
            if not self._is_path_allowed(current_dir):
                raise PermissionError("Current directory is outside allowed directories")
            
            return str(current_dir)
            
        except Exception as e:
            logger.error(f"Error getting current directory: {e}")
            raise
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "readonly_mode": self.readonly_mode,
            "allowed_directories": self.allowed_directories,
            "max_file_size": self.max_file_size,
            "functions": len(self.functions)
        }
