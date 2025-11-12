"""
Agent Session Management for LoCoBench-Agent

This module manages multi-turn conversations between agents and the evaluation
environment, handling context management, tool usage, and session state.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..agents.base_agent import BaseAgent, AgentMessage, AgentResponse, MessageRole
from ..core.task import TaskCategory, DifficultyLevel
from .context_management import (
    BaseContextManager, ContextManagementConfig, ContextManagementStrategy,
    create_context_manager, ContextState, ConversationTurn
)

# Forward declaration for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tools.base_tool import Tool

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Status of an agent session"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ConversationPhase:
    """A phase in the multi-turn conversation"""
    phase_id: str
    name: str
    initial_prompt: str
    expected_actions: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    max_turns_in_phase: int = 10
    dynamic_prompts: Dict[str, str] = field(default_factory=dict)
    human_intervention_triggers: List[str] = field(default_factory=list)
    phase_timeout_minutes: int = 15  # Add missing field
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "initial_prompt": self.initial_prompt,
            "expected_actions": self.expected_actions,
            "success_conditions": self.success_conditions,
            "max_turns_in_phase": self.max_turns_in_phase,
            "dynamic_prompts": self.dynamic_prompts,
            "human_intervention_triggers": self.human_intervention_triggers
        }


@dataclass
class SessionConfig:
    """Configuration for an agent session"""
    max_turns: int = 50
    max_context_tokens: int = 128_000  # Default to GPT-4o limit, will be adjusted per model
    timeout_seconds: int = 3600  # 1 hour
    memory_compression_threshold: float = 0.5  # Compress when 50% of context is used (very aggressive)
    enable_human_intervention: bool = False
    save_checkpoints: bool = False  # Disabled: pipeline-level checkpointing is used instead
    checkpoint_interval: int = 5  # Save checkpoint every 5 turns
    
    # Context management settings
    context_management_strategy: str = "adaptive"
    context_early_warning_threshold: float = 0.4  # Start compression at 40% (very aggressive)
    context_critical_threshold: float = 0.6       # Critical at 60% (very aggressive)
    preserve_initial_turns: int = 2
    preserve_recent_turns: int = 3
    enable_conversation_summary: bool = True
    enable_file_compression: bool = True
    enable_architectural_summary: bool = True
    response_buffer_tokens: int = 4096
    model_name: str = "gpt-4"
    
    # NEW: Cursor-aligned features (ENABLED BY DEFAULT)
    enable_semantic_search: bool = True         # Enable RAG/semantic code search
    enable_enhanced_summarization: bool = True  # Enable LLM-based summarization
    
    # NEW: Initial context loading strategy (Cursor-aligned architecture)
    # "full" = Load all files upfront (OLD behavior, causes context overflow)
    # "minimal" = Load only README + entry points (RECOMMENDED)
    # "empty" = Load nothing, agent discovers everything (MOST REALISTIC)
    initial_context_mode: str = "minimal"  # Default to minimal for Cursor-like behavior
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "max_context_tokens": self.max_context_tokens,
            "timeout_seconds": self.timeout_seconds,
            "memory_compression_threshold": self.memory_compression_threshold,
            "enable_human_intervention": self.enable_human_intervention,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_interval": self.checkpoint_interval
        }


@dataclass
class SessionCheckpoint:
    """Checkpoint data for resuming sessions"""
    session_id: str
    turn_number: int
    phase_id: str
    conversation_history: List[Dict[str, Any]]
    session_state: Dict[str, Any]
    timestamp: datetime
    agent_statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "phase_id": self.phase_id,
            "conversation_history": self.conversation_history,
            "session_state": self.session_state,
            "timestamp": self.timestamp.isoformat(),
            "agent_statistics": self.agent_statistics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionCheckpoint':
        return cls(
            session_id=data["session_id"],
            turn_number=data["turn_number"],
            phase_id=data["phase_id"],
            conversation_history=data["conversation_history"],
            session_state=data["session_state"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_statistics=data["agent_statistics"]
        )


class AgentSession:
    """
    Manages a multi-turn conversation session between an agent and the evaluation environment
    """
    
    def __init__(
        self,
        session_id: str,
        agent: BaseAgent,
        scenario_context: Dict[str, Any],
        conversation_phases: List[ConversationPhase],
        available_tools: List['Tool'] = None,
        config: SessionConfig = None
    ):
        self.session_id = session_id
        self.agent = agent
        self.scenario_context = scenario_context
        self.conversation_phases = conversation_phases
        self.available_tools = available_tools or []
        self.config = config or SessionConfig()
        
        # CRITICAL FIX: Use agent's actual context limits if available
        if hasattr(agent, 'capabilities') and hasattr(agent.capabilities, 'max_context_tokens'):
            self.config.max_context_tokens = agent.capabilities.max_context_tokens
            logger.info(f"Using agent's context limit: {agent.capabilities.max_context_tokens} tokens")
        
        # Also update model name for proper token counting
        if hasattr(agent, 'model'):
            self.config.model_name = agent.model
            logger.info(f"Using agent's model for token counting: {agent.model}")
        
        # Session state
        self.status = SessionStatus.INITIALIZING
        self.current_phase_index = 0
        self.current_turn = 0
        self.session_start_time = datetime.now()
        self.session_end_time: Optional[datetime] = None
        
        # Tracking
        self.phase_history: List[Dict[str, Any]] = []
        self.tool_usage_log: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []
        self.human_interventions: List[Dict[str, Any]] = []
        self.modified_files: Dict[str, str] = {}  # Track files written by agent (path -> content)
        
        # Context management
        self.context_manager = self._initialize_context_manager()
        self.context_state = ContextState()
        self._populate_initial_context()
        
        # Semantic search (NEW - Cursor alignment)
        self.semantic_retriever = None
        self.semantic_search_enabled = self.config.enable_semantic_search if hasattr(self.config, 'enable_semantic_search') else False
        if self.semantic_search_enabled:
            self._initialize_semantic_search()
        
        # Enhanced summarization (NEW - Cursor alignment)
        self.llm_summarizer = None
        self.hierarchical_memory = None
        self.enhanced_summarization_enabled = self.config.enable_enhanced_summarization if hasattr(self.config, 'enable_enhanced_summarization') else False
        if self.enhanced_summarization_enabled:
            self._initialize_enhanced_summarization()
        
        # Checkpointing (disabled - using pipeline-level checkpointing instead)
        self.checkpoint_dir = None
        
        logger.info(f"Created agent session {session_id} with {len(conversation_phases)} phases")
        if self.semantic_search_enabled:
            logger.info("✨ Semantic search enabled (Cursor @codebase equivalent)")
        if self.enhanced_summarization_enabled:
            logger.info("✨ Enhanced LLM summarization enabled")
    
    def _initialize_context_manager(self) -> BaseContextManager:
        """Initialize the context manager based on session configuration"""
        context_config = ContextManagementConfig(
            strategy=ContextManagementStrategy(self.config.context_management_strategy),
            early_warning_threshold=self.config.context_early_warning_threshold,
            critical_threshold=self.config.context_critical_threshold,
            preserve_initial_turns=self.config.preserve_initial_turns,
            preserve_recent_turns=self.config.preserve_recent_turns,
            enable_conversation_summary=self.config.enable_conversation_summary,
            enable_file_compression=self.config.enable_file_compression,
            enable_architectural_summary=self.config.enable_architectural_summary,
            model_name=self.config.model_name,
            max_context_tokens=self.config.max_context_tokens,
            response_buffer_tokens=self.config.response_buffer_tokens
        )
        return create_context_manager(context_config)
    
    def _normalize_project_files(self, files_data):
        """Normalize project files to dict format {path: content}
        
        Handles two formats:
        1. Dict: {path: content, ...}
        2. List: [{path: ..., content: ...}, ...]
        """
        if isinstance(files_data, dict):
            return files_data
        elif isinstance(files_data, list):
            # Convert list of dicts to path->content dict
            normalized = {}
            for item in files_data:
                if isinstance(item, dict) and 'path' in item and 'content' in item:
                    normalized[item['path']] = item['content']
            return normalized
        else:
            return {}
    
    def _populate_initial_context(self):
        """Populate initial context state - Cursor-aligned on-demand architecture"""
        
        mode = self.config.initial_context_mode
        # Check both field names for compatibility
        # Handle nested structure: initial_context.project_files or direct project_files
        initial_ctx = self.scenario_context.get("initial_context", {})
        
        if isinstance(initial_ctx, dict) and "project_files" in initial_ctx:
            all_files_raw = initial_ctx["project_files"]
        else:
            all_files_raw = self.scenario_context.get("project_files", {})
        
        # CRITICAL: Store original files IMMEDIATELY for tool access
        # Store raw format (could be list or dict) for tools
        if isinstance(all_files_raw, (dict, list)):
            self._all_project_files = all_files_raw.copy() if isinstance(all_files_raw, list) else all_files_raw.copy()
            self.scenario_context["_all_files_for_tools"] = all_files_raw.copy() if isinstance(all_files_raw, list) else all_files_raw.copy()
        else:
            self._all_project_files = {}
            self.scenario_context["_all_files_for_tools"] = {}
        
        # Normalize to dict format for population methods
        all_files = self._normalize_project_files(all_files_raw)
        
        logger.info(f"🎯 Initial context mode: {mode} ({len(all_files)} files available)")
        logger.info(f"📦 Stored {len(self.scenario_context.get('_all_files_for_tools', {}))} files for tool access")
        
        if mode == "full":
            # OLD BEHAVIOR: Load everything upfront (causes context overflow!)
            logger.warning("⚠️  Using 'full' mode - may cause context overflow on large codebases")
            self._populate_initial_context_full(all_files)
            
        elif mode == "minimal":
            # RECOMMENDED: Load README + entry points (Cursor-like)
            logger.info("✅ Using 'minimal' mode - Cursor-aligned architecture")
            self._populate_initial_context_minimal(all_files)
            
        elif mode == "empty":
            # MOST REALISTIC: Agent discovers everything via tools
            logger.info("✅ Using 'empty' mode - Full on-demand discovery")
            self._populate_initial_context_empty(all_files)
            
        else:
            logger.error(f"Unknown initial_context_mode: {mode}, defaulting to 'minimal'")
            self._populate_initial_context_minimal(all_files)
        
        # Update scenario_context to reflect what's actually in LLM context
        if mode in ["minimal", "empty"]:
            # Replace the nested structure with only what's in context_state
            if isinstance(initial_ctx, dict) and "project_files" in initial_ctx:
                # Update the nested structure
                self.scenario_context["initial_context"]["project_files"] = self.context_state.project_files.copy()
            else:
                # Update direct structure
                self.scenario_context["project_files"] = self.context_state.project_files.copy()
            
            logger.info(f"📝 Updated scenario_context: {len(self.context_state.project_files)} files in LLM context (down from {len(all_files)})")
        
        # Report final context size
        initial_tokens, initial_usage = self.context_manager.calculate_context_usage(self.context_state)
        logger.info(f"📊 Initial context: {len(self.context_state.project_files)} files, "
                   f"{initial_tokens} tokens ({initial_usage:.1%} of {self.config.max_context_tokens} limit)")
    
    def _populate_initial_context_full(self, all_files: Dict[str, str]):
        """LEGACY: Load all files upfront (OLD behavior - causes overflow!)"""
        for filename, content in all_files.items():
            self.context_state.project_files[filename] = content
        self.context_state.active_files = list(self.context_state.project_files.keys())
        
        # Try to compress if too large
        initial_tokens, initial_usage = self.context_manager.calculate_context_usage(self.context_state)
        
        if initial_usage > 1.0:
            logger.error(f"⚠️  CRITICAL: Initial context EXCEEDS model limit ({initial_tokens} tokens)")
            logger.warning(f"   Applying AGGRESSIVE compression...")
            self.context_state = self._compress_initial_context_aggressive(self.context_state)
            final_tokens, final_usage = self.context_manager.calculate_context_usage(self.context_state)
            logger.info(f"   ✅ Compressed: {initial_tokens} -> {final_tokens} tokens ({final_usage:.1%})")
        elif initial_usage > 0.7:
            logger.warning(f"Initial context large ({initial_tokens} tokens), applying compression")
            self.context_state = self._compress_initial_context(self.context_state)
            final_tokens, final_usage = self.context_manager.calculate_context_usage(self.context_state)
            logger.info(f"Compressed: {initial_tokens} -> {final_tokens} tokens ({final_usage:.1%})")
    
    def _populate_initial_context_minimal(self, all_files: Dict[str, str]):
        """RECOMMENDED: Load only README + entry points (Cursor-like)"""
        
        # Priority files to include (if they exist)
        priority_patterns = [
            # Documentation
            "README.md", "README.txt", "README.rst", "readme.md",
            # Entry points by language
            "main.py", "app.py", "__main__.py", "run.py",  # Python
            "main.c", "main.cpp", "main.java", "Main.java",  # C/C++/Java
            "index.js", "index.ts", "main.js", "main.ts", "app.js", "app.ts",  # JavaScript/TypeScript
            "index.html", "main.html",  # HTML
            "main.go",  # Go
            "main.rs", "lib.rs",  # Rust
            "Program.cs", "Main.cs",  # C#
            # Package/config files (metadata only, usually small)
            "package.json", "setup.py", "requirements.txt", "Cargo.toml", "go.mod",
            "tsconfig.json", "pyproject.toml", "Makefile", "CMakeLists.txt"
        ]
        
        loaded_count = 0
        for pattern in priority_patterns:
            # Match both exact names and paths ending with pattern
            for filename, content in all_files.items():
                if filename == pattern or filename.endswith(f"/{pattern}") or filename.endswith(f"\\{pattern}"):
                    self.context_state.project_files[filename] = content
                    self.context_state.active_files.append(filename)
                    loaded_count += 1
                    logger.debug(f"  📄 Loaded priority file: {filename}")
                    break  # Only load first match for each pattern
        
        logger.info(f"✅ Loaded {loaded_count} priority files (README, entry points, configs)")
        logger.info(f"📂 {len(all_files) - loaded_count} files available via tools/semantic search")
    
    def _populate_initial_context_empty(self, all_files: Dict[str, str]):
        """MOST REALISTIC: Load nothing, but provide file structure map (Cursor-like)"""
        
        # Start with completely empty context (no file content)
        self.context_state.project_files = {}
        self.context_state.active_files = []
        
        # CURSOR-ALIGNED: Provide a lightweight file structure map
        # This gives the agent a "map" without loading content (minimal tokens)
        file_tree = self._generate_file_tree(all_files)
        self.context_state.metadata["file_structure"] = file_tree
        
        logger.info(f"📂 {len(all_files)} files available for discovery via tools")
        logger.info(f"🗺️  Provided file structure map ({len(file_tree.split(chr(10)))} lines) to guide exploration")
        logger.info("🔍 Agent will use file_system tools to read file content on-demand")
    
    def _generate_file_tree(self, all_files: Dict[str, str]) -> str:
        """Generate a lightweight file tree structure (paths only, no content)
        
        CRITICAL: This must show EXACT file paths that the agent can use to read files.
        Ambiguous paths cause LLM hallucination where the agent guesses file names.
        """
        if not all_files:
            return "No files in project"
        
        # Build tree structure showing COMPLETE file paths
        tree_lines = ["Project File Structure (exact paths for read_file()):", ""]
        
        # Sort files for consistent display
        sorted_files = sorted(all_files.keys())
        
        # Group by directory
        from collections import defaultdict
        dirs = defaultdict(list)
        root_files = []
        
        for filepath in sorted_files:
            # Normalize path separators for display (// or /)
            normalized = filepath.replace('//', '/')
            
            if '/' in normalized:
                # Extract directory
                dir_path = normalized.rsplit('/', 1)[0]
                dirs[dir_path].append(filepath)  # Keep original path for reading
            else:
                root_files.append(filepath)
        
        # Add root files with FULL PATHS
        if root_files:
            tree_lines.append("📄 Root Files:")
            for f in root_files[:15]:  # Show more root files
                tree_lines.append(f"   {f}")
            if len(root_files) > 15:
                tree_lines.append(f"   ... and {len(root_files) - 15} more")
            tree_lines.append("")
        
        # Add directories and their files with FULL PATHS
        tree_lines.append("📂 Directory Structure:")
        for dir_path in sorted(dirs.keys())[:30]:  # Show more directories
            tree_lines.append(f"   {dir_path}/ ({len(dirs[dir_path])} files)")
            for f in dirs[dir_path][:8]:  # Show more files per directory
                # Show COMPLETE path so agent knows EXACTLY what to pass to read_file()
                tree_lines.append(f"      {f}")
            if len(dirs[dir_path]) > 8:
                remaining = dirs[dir_path][8:]
                tree_lines.append(f"      ... and {len(remaining)} more: {', '.join([fp.rsplit('/', 1)[-1].rsplit('//', 1)[-1] for fp in remaining[:3]])}")
        
        if len(dirs) > 30:
            tree_lines.append(f"   ... and {len(dirs) - 30} more directories")
        
        tree_lines.append("")
        tree_lines.append(f"📊 Total: {len(sorted_files)} files in {len(dirs)} directories")
        tree_lines.append("")
        tree_lines.append("⚠️  CRITICAL: Use EXACT paths shown above (including '//' or '/').")
        tree_lines.append("    Do NOT guess or modify paths - use what you see here or discover via list_directory().")
        
        return "\n".join(tree_lines)
    
    def _initialize_semantic_search(self):
        """Initialize semantic search system (Cursor @codebase equivalent)"""
        try:
            import os
            from pathlib import Path
            from .semantic_search import SemanticCodeRetriever
            
            # Check if OpenAI API is available
            use_openai = bool(os.getenv("OPENAI_API_KEY"))
            if use_openai:
                logger.info("🔹 OpenAI API detected - using OpenAI embeddings for semantic search")
            
            # Use project-local cache directory instead of home directory
            # This keeps all evaluation data together and makes cleanup easier
            project_cache_dir = Path.cwd() / ".locobench" / "semantic_cache"
            self.semantic_retriever = SemanticCodeRetriever(
                cache_dir=project_cache_dir,
                use_openai=use_openai
            )
            
            # Index project files if available (use _all_files_for_tools if in on-demand mode)
            files_to_index = self.scenario_context.get("_all_files_for_tools")
            if not files_to_index:
                initial_ctx = self.scenario_context.get("initial_context", {})
                if isinstance(initial_ctx, dict) and "project_files" in initial_ctx:
                    files_to_index = initial_ctx["project_files"]
                else:
                    files_to_index = self.scenario_context.get("project_files", {})
            
            if files_to_index:
                project_name = self.scenario_context.get("project_name", self.session_id)
                logger.info(f"Indexing {len(files_to_index)} files for semantic search...")
                
                self.semantic_retriever.index_project(
                    project_files=files_to_index,
                    project_name=project_name,
                    force_reindex=False  # Use cache if available
                )
                
                logger.info(f"✅ Semantic search ready with {len(files_to_index)} files indexed")
                
                # Update semantic search tool if it exists in available_tools
                for tool in self.available_tools:
                    if hasattr(tool, 'name') and tool.name == 'semantic_search':
                        tool.semantic_retriever = self.semantic_retriever
                        logger.info("✅ SemanticSearchTool linked to retriever")
                        break
        except ImportError as e:
            logger.warning(f"Semantic search dependencies not available: {e}")
            logger.warning("Install with: pip install sentence-transformers numpy")
            self.semantic_search_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}", exc_info=True)
            self.semantic_search_enabled = False
    
    def _initialize_enhanced_summarization(self):
        """Initialize enhanced LLM summarization system"""
        try:
            from .enhanced_summarization import LLMSummarizer, HierarchicalMemoryManager
            
            # Use the agent's LLM client for summarization
            # Use a cheaper/smaller model for summarization (e.g., gpt-4o-mini instead of gpt-4o)
            summarization_model = self._get_summarization_model()
            
            logger.info(f"Initializing LLM summarization with model: {summarization_model}")
            
            self.llm_summarizer = LLMSummarizer(
                llm_client=self.agent.client if hasattr(self.agent, 'client') else None,
                model_name=summarization_model
            )
            
            self.hierarchical_memory = HierarchicalMemoryManager(
                working_memory_turns=5,  # Last 5 turns in full detail
                short_term_turns=20,     # 6-20 turns as summaries
                summarizer=self.llm_summarizer
            )
            
            logger.info("✅ Enhanced summarization initialized (3-tier memory system)")
        except ImportError as e:
            logger.warning(f"Enhanced summarization dependencies not available: {e}")
            self.enhanced_summarization_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize enhanced summarization: {e}", exc_info=True)
            self.enhanced_summarization_enabled = False
    
    def _get_summarization_model(self) -> str:
        """Get appropriate model for summarization (cheaper than main model)"""
        agent_model = self.agent.model if hasattr(self.agent, 'model') else "gpt-4o"
        
        # Map to cheaper summarization models
        model_mapping = {
            "gpt-4o": "gpt-4o-mini",
            "gpt-4": "gpt-4o-mini",
            "gpt-4-turbo": "gpt-4o-mini",
            "claude-3-opus": "claude-3-haiku",
            "claude-3-sonnet": "claude-3-haiku",
            "claude-sonnet-4": "claude-3-haiku",
            "gemini-pro": "gemini-flash",
        }
        
        return model_mapping.get(agent_model, agent_model)
    
    def _compress_initial_context_aggressive(self, state: ContextState) -> ContextState:
        """AGGRESSIVE compression for scenarios that exceed model limits (>100%)"""
        logger.warning("⚠️  Applying AGGRESSIVE initial context compression (scenario >100% of model limit)")
        
        # For extremely large scenarios, use only 30% of tokens for files
        # Reserve 70% for conversation, tool outputs, and response buffer
        target_tokens = int(self.config.max_context_tokens * 0.30)
        
        # Prioritize files by importance
        prioritized_files = self._prioritize_files_by_importance(state.project_files)
        
        selected_files = {}
        current_tokens = 0
        
        # Only include the most important files that fit in budget
        for filename, content in prioritized_files:
            file_tokens = self.context_manager.count_tokens(content)
            
            # For very large files, truncate them
            if file_tokens > 10000:  # Files >10K tokens get truncated
                truncated_content = content[:int(len(content) * 0.3)]  # Keep first 30%
                truncated_content += f"\n\n... [File truncated - original size: {file_tokens} tokens] ..."
                file_tokens = self.context_manager.count_tokens(truncated_content)
                content = truncated_content
                logger.debug(f"Truncated large file {filename} to ~{file_tokens} tokens")
            
            if current_tokens + file_tokens <= target_tokens:
                selected_files[filename] = content
                current_tokens += file_tokens
            else:
                # Stop adding files once we hit budget
                break
        
        # Update context state
        state.project_files = selected_files
        state.active_files = list(selected_files.keys())
        
        # Create summary of ALL excluded content
        excluded_files = {k: v for k, v in dict(prioritized_files).items() if k not in selected_files}
        if excluded_files:
            state.architectural_summary = self._generate_excluded_files_summary(excluded_files)
            logger.warning(f"   📊 Kept {len(selected_files)}/{len(dict(prioritized_files))} files")
            logger.warning(f"   📊 Excluded {len(excluded_files)} files (architectural summary created)")
            logger.warning(f"   📊 Final file tokens: {current_tokens}/{target_tokens} (target)")
        
        return state
    
    def _compress_initial_context(self, state: ContextState) -> ContextState:
        """Compress initial context when scenario is too large for model"""
        logger.info("Applying intelligent initial context compression for large scenario")
        
        # Strategy 1: Prioritize files by importance and relevance
        prioritized_files = self._prioritize_files_by_importance(state.project_files)
        
        # Strategy 2: Progressive file inclusion until we hit token budget
        target_tokens = int(self.config.max_context_tokens * 0.6)  # Use 60% for initial context
        selected_files = {}
        current_tokens = 0
        
        for filename, content in prioritized_files:
            file_tokens = self.context_manager.count_tokens(content)
            if current_tokens + file_tokens <= target_tokens:
                selected_files[filename] = content
                current_tokens += file_tokens
                logger.debug(f"Included {filename} ({file_tokens} tokens)")
            else:
                logger.debug(f"Skipped {filename} ({file_tokens} tokens) - would exceed budget")
        
        # Update context state with selected files
        state.project_files = selected_files
        state.active_files = list(selected_files.keys())
        
        # Create architectural summary of excluded files
        excluded_files = {k: v for k, v in dict(prioritized_files).items() if k not in selected_files}
        if excluded_files:
            state.architectural_summary = self._generate_excluded_files_summary(excluded_files)
            logger.info(f"Created architectural summary for {len(excluded_files)} excluded files")
        
        logger.info(f"Initial compression: kept {len(selected_files)}/{len(dict(prioritized_files))} files ({current_tokens} tokens)")
        return state
    
    def _prioritize_files_by_importance(self, project_files: Dict[str, str]) -> List[Tuple[str, str]]:
        """Prioritize files by importance for initial context"""
        files_with_scores = []
        
        for filename, content in project_files.items():
            score = self._calculate_file_importance_score(filename, content)
            files_with_scores.append((filename, content, score))
        
        # Sort by importance score (descending)
        files_with_scores.sort(key=lambda x: x[2], reverse=True)
        return [(filename, content) for filename, content, _ in files_with_scores]
    
    def _calculate_file_importance_score(self, filename: str, content: str) -> float:
        """Calculate importance score for a file (higher = more important)"""
        score = 0.0
        filename_lower = filename.lower()
        
        # Core source files (highest priority)
        if any(pattern in filename_lower for pattern in ['/src/', '//src//', '/source/', '//source//']):
            score += 100
        
        # Main entry points
        if any(name in filename_lower for name in ['main.', 'index.', 'app.', 'server.', 'client.']):
            score += 80
        
        # Configuration files (important for understanding setup)
        if any(ext in filename_lower for ext in ['.json', '.yaml', '.yml', '.toml', '.ini']):
            score += 60
        
        # Source code files by language
        source_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.php']
        if any(filename_lower.endswith(ext) for ext in source_extensions):
            score += 70
        
        # Test files (lower priority but still important)
        if any(pattern in filename_lower for pattern in ['test', 'spec', '__test__']):
            score += 40
        
        # Documentation (lowest priority for initial context)
        if any(ext in filename_lower for ext in ['.md', '.txt', '.rst']):
            score += 10
        
        # Penalize very large files (they might be generated or less important)
        if len(content) > 50000:  # >50KB
            score -= 20
        
        # Boost files with important keywords in content
        important_keywords = ['class ', 'function ', 'def ', 'import ', 'from ', 'export ', 'public ']
        keyword_count = sum(content.count(keyword) for keyword in important_keywords)
        score += min(keyword_count * 0.5, 30)  # Cap at 30 points
        
        return score
    
    def _generate_excluded_files_summary(self, excluded_files: Dict[str, str]) -> str:
        """Generate summary of files excluded from initial context"""
        summary_parts = ["EXCLUDED FILES SUMMARY (available via file system tools):"]
        
        # Group files by type
        file_groups = {
            'Source Files': [],
            'Configuration': [],
            'Documentation': [],
            'Tests': [],
            'Other': []
        }
        
        for filename in excluded_files.keys():
            filename_lower = filename.lower()
            if any(ext in filename_lower for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go']):
                file_groups['Source Files'].append(filename)
            elif any(ext in filename_lower for ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']):
                file_groups['Configuration'].append(filename)
            elif any(ext in filename_lower for ext in ['.md', '.txt', '.rst']):
                file_groups['Documentation'].append(filename)
            elif any(pattern in filename_lower for pattern in ['test', 'spec', '__test__']):
                file_groups['Tests'].append(filename)
            else:
                file_groups['Other'].append(filename)
        
        for group_name, files in file_groups.items():
            if files:
                summary_parts.append(f"\n{group_name} ({len(files)}):")
                # Show first few files in each category
                for filename in files[:5]:
                    summary_parts.append(f"  - {filename}")
                if len(files) > 5:
                    summary_parts.append(f"  ... and {len(files) - 5} more")
        
        summary_parts.append(f"\nTotal excluded: {len(excluded_files)} files")
        summary_parts.append("Note: These files can be accessed using file system tools when needed.")
        
        return "\n".join(summary_parts)
    
    def _add_conversation_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a turn to the conversation and manage context"""
        turn = ConversationTurn(
            turn_number=len(self.context_state.conversation_turns) + 1,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Count tokens for the turn
        turn.token_count = self.context_manager.count_tokens(content)
        
        # Add to context state
        self.context_state.conversation_turns.append(turn)
        
        # Check if context compression is needed
        if self.context_manager.should_compress(self.context_state):
            logger.info(f"Context compression triggered at turn {turn.turn_number}")
            self.context_state = self.context_manager.compress_context(self.context_state)
            logger.info(f"Context compressed: {len(self.context_state.conversation_turns)} turns remaining")
    
    def get_context_usage(self) -> Tuple[int, float]:
        """Get current context usage in tokens and percentage"""
        # Calculate tokens from agent's actual conversation history
        total_tokens = sum(msg.context_tokens for msg in self.agent.conversation_history)
        max_tokens = self.config.max_context_tokens
        usage_percentage = total_tokens / max_tokens if max_tokens > 0 else 0.0
        return total_tokens, usage_percentage
    
    def _get_managed_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the managed conversation history for the agent"""
        managed_history = []
        
        # Add conversation summary if available
        if self.context_state.conversation_summary:
            managed_history.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self.context_state.conversation_summary}",
                "metadata": {"type": "conversation_summary"}
            })
        
        # Add current conversation turns
        for turn in self.context_state.conversation_turns:
            managed_history.append({
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat(),
                "turn_number": turn.turn_number,
                "metadata": turn.metadata
            })
        
        return managed_history
    
    @property
    def current_phase(self) -> Optional[ConversationPhase]:
        """Get the current conversation phase"""
        if 0 <= self.current_phase_index < len(self.conversation_phases):
            return self.conversation_phases[self.current_phase_index]
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if the session is currently active"""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def is_completed(self) -> bool:
        """Check if the session is completed"""
        return self.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT]
    
    async def initialize(self) -> bool:
        """Initialize the agent session"""
        try:
            self.status = SessionStatus.INITIALIZING
            logger.info(f"Initializing session {self.session_id}")
            
            # Initialize the agent
            success = await self.agent.initialize_session(
                self.scenario_context,
                self.available_tools
            )
            
            if not success:
                self.status = SessionStatus.FAILED
                logger.error(f"Failed to initialize agent for session {self.session_id}")
                return False
            
            # Add initial system message with scenario context
            initial_message = AgentMessage(
                role=MessageRole.SYSTEM,
                content=self._create_initial_system_prompt(),
                context_tokens=len(self._create_initial_system_prompt().split()) * 1.3  # Rough estimate
            )
            self.agent.add_message_to_history(initial_message)
            
            self.status = SessionStatus.ACTIVE
            logger.info(f"Successfully initialized session {self.session_id}")
            return True
            
        except Exception as e:
            self.status = SessionStatus.FAILED
            import traceback
            logger.error(f"Error initializing session {self.session_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def execute_conversation(self) -> Dict[str, Any]:
        """Execute the full multi-turn conversation"""
        if not await self.initialize():
            return self._create_session_result()
        
        try:
            # Execute each phase
            for phase_index, phase in enumerate(self.conversation_phases):
                self.current_phase_index = phase_index
                logger.info(f"Starting phase {phase_index + 1}/{len(self.conversation_phases)}: {phase.name}")
                
                phase_result = await self._execute_phase(phase)
                self.phase_history.append(phase_result)
                
                # Check if we should stop early
                if not phase_result.get("success", False) and phase_result.get("critical_failure", False):
                    logger.warning(f"Critical failure in phase {phase.name}, stopping session")
                    break
                
                # Check session limits
                if self._should_terminate_session():
                    break
            
            # Finalize session
            await self._finalize_session()
            
        except asyncio.TimeoutError:
            self.status = SessionStatus.TIMEOUT
            logger.warning(f"Session {self.session_id} timed out")
        except Exception as e:
            self.status = SessionStatus.FAILED
            logger.error(f"Error in session {self.session_id}: {e}")
            self.error_log.append({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "turn": self.current_turn,
                "phase": self.current_phase.phase_id if self.current_phase else None
            })
        
        return self._create_session_result()
    
    async def _execute_phase(self, phase: ConversationPhase) -> Dict[str, Any]:
        """Execute a single conversation phase"""
        phase_start_time = datetime.now()
        phase_turns = 0
        phase_success = False
        
        logger.info(f"Executing phase: {phase.name}")
        
        # Send initial phase prompt
        user_message = AgentMessage(
            role=MessageRole.USER,
            content=phase.initial_prompt,
            context_tokens=len(phase.initial_prompt.split()) * 1.3
        )
        self.agent.add_message_to_history(user_message)
        
        # Only track conversation for context management if needed
        # (Disabled by default to prevent conversation contamination)
        
        # Execute turns in this phase
        current_user_message = user_message  # Start with the initial prompt
        
        while phase_turns < phase.max_turns_in_phase and not self._should_terminate_session():
            turn_start_time = time.time()
            self.current_turn += 1
            phase_turns += 1
            
            logger.debug(f"Executing turn {self.current_turn} in phase {phase.name}")
            
            # Only proceed if we have a user message to process
            if not current_user_message or not current_user_message.content.strip():
                logger.debug(f"No user message for turn {phase_turns}, ending phase {phase.name}")
                break
            
            try:
                # Check if we need context management (only when approaching limits)
                # CRITICAL FIX: Use context manager's calculation which includes project files
                current_tokens, usage_pct = self.context_manager.calculate_context_usage(self.context_state)
                use_context_management = usage_pct > self.config.context_early_warning_threshold
                
                if use_context_management:
                    # Use managed context only when needed
                    managed_context = self.scenario_context.copy()
                    managed_context['managed_conversation'] = self._get_managed_conversation_history()
                    managed_context['context_summary'] = self.context_state.conversation_summary
                    managed_context['architectural_summary'] = self.context_state.architectural_summary
                    
                    # Add managed project files (only active files to save context)
                    managed_context['project_files'] = {
                        filename: content 
                        for filename, content in self.context_state.project_files.items()
                        if filename in self.context_state.active_files
                    }
                    
                    logger.info(f"Using context management due to high usage: {usage_pct:.1%}")
                    context_to_pass = managed_context
                else:
                    # Use normal context - let agent handle its own conversation history
                    context_to_pass = self.scenario_context
                
                # Process the current user message
                agent_response = await self.agent.process_turn(
                    message=current_user_message.content,
                    available_tools=self.available_tools,
                    context=context_to_pass
                )
                
                # Note: Agent handles adding its own response to history
                
                # Only track for context management if actually being used
                # (Disabled by default to prevent conversation contamination)
                
                # Log tool usage and track file modifications
                if agent_response.tool_calls:
                    for tool_call in agent_response.tool_calls:
                        self.tool_usage_log.append({
                            "turn": self.current_turn,
                            "phase": phase.phase_id,
                            "tool_call": tool_call.to_dict(),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Track files written by the agent
                        if 'write_file' in tool_call.function_name.lower() or 'write' in tool_call.function_name.lower():
                            # Extract file path and content from parameters
                            params = tool_call.parameters or {}
                            file_path = params.get('path') or params.get('file_path') or params.get('filepath')
                            file_content = params.get('content') or params.get('contents') or ''
                            if file_path:
                                self.modified_files[file_path] = file_content
                                logger.debug(f"Tracked file modification: {file_path} ({len(file_content)} chars)")
                
                # CRITICAL FIX: Progress monitoring to prevent analysis loops
                # Check if agent is making progress in implementation tasks
                if self._is_implementation_phase(phase) and phase_turns >= 5:
                    # Count recent write_file calls in last 5 turns
                    recent_write_calls = self._count_recent_write_calls(phase_turns, phase_start_turn=self.current_turn - phase_turns + 1)
                    
                    if recent_write_calls == 0:
                        # No code written in last 5 turns - inject reminder
                        logger.warning(f"Progress monitoring: No code written in {phase_turns} turns of implementation phase")
                        
                        if phase_turns >= 10:
                            # After 10 turns without code - force end with penalty
                            logger.error(f"Phase {phase.name} terminated: No code written after 10 turns")
                            phase_success = False
                            self.human_interventions.append({
                                "turn": self.current_turn,
                                "phase": phase.phase_id,
                                "reason": "terminated_no_code_progress",
                                "message": "Phase terminated after 10 turns without writing code"
                            })
                            break
                        elif phase_turns >= 5:
                            # After 5 turns - inject strong reminder
                            reminder = """
⚠️ CRITICAL REMINDER: You have been analyzing for 5+ turns without writing any code!

For implementation tasks, you MUST:
1. USE the write_file tool to create or modify code
2. Stop analyzing and START IMPLEMENTING
3. Write the actual code changes NOW

Your next response MUST include a write_file tool call, or this phase will fail."""
                            
                            current_user_message = AgentMessage(
                                role=MessageRole.USER,
                                content=reminder,
                                context_tokens=len(reminder.split()) * 1.3
                            )
                            self.agent.add_message_to_history(current_user_message)
                            self.human_interventions.append({
                                "turn": self.current_turn,
                                "phase": phase.phase_id,
                                "reason": "progress_monitoring_reminder",
                                "message": "Injected reminder to write code"
                            })
                            continue  # Skip success check, go to next turn
                
                # Check success conditions
                if self._check_phase_success_conditions(phase, agent_response):
                    phase_success = True
                    logger.info(f"Phase {phase.name} completed successfully")
                    break
                
                # Check for dynamic prompts for the next turn
                dynamic_prompt = self._get_dynamic_prompt(phase, agent_response)
                if dynamic_prompt:
                    current_user_message = AgentMessage(
                        role=MessageRole.USER,
                        content=dynamic_prompt,
                        context_tokens=len(dynamic_prompt.split()) * 1.3
                    )
                    self.agent.add_message_to_history(current_user_message)
                    
                    # Only track for context management if actually being used
                    # (Disabled by default to prevent conversation contamination)
                else:
                    # No dynamic prompt generated, end this phase
                    logger.debug(f"No dynamic prompt generated, ending phase {phase.name}")
                    current_user_message = None
                
                # Save checkpoint if needed
                # Checkpoint saving disabled - using pipeline-level checkpointing instead
                # if self.config.save_checkpoints and self.current_turn % self.config.checkpoint_interval == 0:
                #     await self._save_checkpoint()
                
                # Context management is handled automatically in _add_conversation_turn()
                # Log current context usage for monitoring
                tokens, usage_pct = self.get_context_usage()
                if usage_pct > 0.7:  # Log when approaching limits
                    logger.info(f"Context usage: {tokens:,} tokens ({usage_pct:.1%})")
                    if usage_pct > 0.9:
                        logger.warning(f"High context usage: {tokens:,} tokens ({usage_pct:.1%})")
                
                turn_time = time.time() - turn_start_time
                logger.debug(f"Turn {self.current_turn} completed in {turn_time:.2f}s")
                
            except Exception as e:
                # THROTTLING-RESILIENT EVALUATION: Distinguish between model failures and API infrastructure issues
                error_str = str(e).lower()
                is_throttling_error = any(keyword in error_str for keyword in [
                    'throttling', 'rate limit', 'too many tokens', 'quota exceeded', 
                    'rate_limit_exceeded', 'throttlingexception', 'too many requests'
                ])
                
                if is_throttling_error:
                    # API throttling - NOT a model performance issue
                    logger.warning(f"API throttling in turn {self.current_turn}: {e}")
                    self.error_log.append({
                        "error": str(e),
                        "error_type": "api_throttling",
                        "infrastructure_issue": True,
                        "affects_model_evaluation": False,
                        "timestamp": datetime.now().isoformat(),
                        "turn": self.current_turn,
                        "phase": phase.phase_id
                    })
                    
                    # For throttling: Continue phase but mark as throttling-affected
                    # Don't break - let the phase continue with what was accomplished
                    logger.info(f"Continuing phase {phase.name} despite throttling (infrastructure issue)")
                    
                    # Set current_user_message to None to end the phase gracefully
                    current_user_message = None
                    
                else:
                    # Genuine model/logic error - this should affect evaluation
                    logger.error(f"Model error in turn {self.current_turn}: {e}")
                    self.error_log.append({
                        "error": str(e),
                        "error_type": "model_error", 
                        "infrastructure_issue": False,
                        "affects_model_evaluation": True,
                        "timestamp": datetime.now().isoformat(),
                        "turn": self.current_turn,
                        "phase": phase.phase_id
                    })
                    break
        
        phase_duration = (datetime.now() - phase_start_time).total_seconds()
        
        # Check if phase was affected by throttling (infrastructure issues)
        throttling_errors = [err for err in self.error_log if err.get("error_type") == "api_throttling" and err.get("phase") == phase.phase_id]
        model_errors = [err for err in self.error_log if err.get("error_type") == "model_error" and err.get("phase") == phase.phase_id]
        
        return {
            "phase_id": phase.phase_id,
            "phase_name": phase.name,
            "success": phase_success,
            "turns_executed": phase_turns,
            "duration_seconds": phase_duration,
            "critical_failure": len(model_errors) > 0,  # Only model errors are critical failures
            "throttling_affected": len(throttling_errors) > 0,  # Track infrastructure issues separately
            "infrastructure_issues": len(throttling_errors),
            "model_errors": len(model_errors),
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_initial_system_prompt(self) -> str:
        """Create the initial system prompt with scenario context"""
        scenario_title = self.scenario_context.get("title", "Development Task")
        scenario_description = self.scenario_context.get("description", "")
        available_files = self.scenario_context.get("context_files", [])
        
        # Create detailed tool descriptions
        tool_descriptions = []
        for tool in self.available_tools:
            # Get tool functions
            functions = []
            if hasattr(tool, 'functions'):
                # tool.functions could be a list of function objects or strings
                for f in tool.functions:
                    if hasattr(f, 'name'):
                        functions.append(f.name)
                    elif isinstance(f, str):
                        functions.append(f)
            elif hasattr(tool, 'function_schemas'):
                functions = [f['function']['name'] for f in tool.function_schemas]
            
            if functions:
                tool_desc = f"{tool.name}: {', '.join(functions)}"
            else:
                tool_desc = tool.name
            tool_descriptions.append(tool_desc)
        
        prompt = f"""You are an expert software engineer working on a development task. 

**TASK**: {scenario_title}

**DESCRIPTION**: {scenario_description}

**AVAILABLE CONTEXT FILES**: {', '.join(available_files)}

**AVAILABLE TOOLS AND FUNCTIONS**:
{chr(10).join(f'  • {desc}' for desc in tool_descriptions)}

**CRITICAL INSTRUCTIONS**:
1. You MUST use the available tools to complete tasks
2. To read files: Use file_system_read_file or ide_simulator_read_file
3. To write/modify files: Use file_system_write_file (THIS IS REQUIRED FOR IMPLEMENTATION TASKS!)
4. To explore structure: Use ide_simulator_get_project_overview or file_system_list_directory
5. DO NOT just explain what you would do - ACTUALLY USE THE TOOLS!
6. When asked to "implement", you MUST call write_file tools, not just describe the implementation"""

        # Add file structure map if available (empty mode)
        if "file_structure" in self.context_state.metadata:
            prompt += f"\n\n**PROJECT STRUCTURE**:\n```\n{self.context_state.metadata['file_structure']}\n```"
            prompt += """

**CRITICAL FILE ACCESS RULES**: 
1. The PROJECT STRUCTURE above lists ALL files with their EXACT paths
2. To read a file, copy-paste the EXACT path from the structure (e.g., "ProjectName//src//file.c")
3. DO NOT modify, guess, or extrapolate file paths - use them exactly as shown
4. DO NOT assume files exist based on directory names (e.g., if you see "src/", don't guess "src/main.c" unless it's listed)
5. If a path looks like a directory, use list_directory() to see what's inside
6. File paths may use '//' or '/' - use them EXACTLY as shown in the structure

**EXAMPLES OF CORRECT vs INCORRECT**:
✅ CORRECT: read_file("EduGate//src//main.c")  [exact path from structure]
❌ WRONG: read_file("EduGate/src/main.c")      [changed // to /]
❌ WRONG: read_file("src/auth.c")              [guessed a file that might exist]
❌ WRONG: read_file("edugateway-c/src/files/lesson_cache.c")  [extrapolated from directory]"""
        
        prompt += """

**INSTRUCTIONS**:
- You will work through this task in multiple phases
- Use the available tools to explore, analyze, and modify the codebase
- Provide clear explanations of your reasoning and actions
- Ask for clarification when needed
- Work systematically and document your progress

This is a multi-turn conversation. I will guide you through different phases of the development process."""

        return prompt
    
    def _check_phase_success_conditions(self, phase: ConversationPhase, agent_response: AgentResponse) -> bool:
        """Check if the phase success conditions have been met"""
        if not phase.success_conditions:
            return False
        
        # Basic implementation - can be enhanced with more sophisticated condition checking
        response_content = agent_response.message.content.lower()
        
        for condition in phase.success_conditions:
            condition_lower = condition.lower()
            if condition_lower in response_content:
                return True
        
        # Check if expected actions were performed
        if phase.expected_actions and agent_response.tool_calls:
            performed_actions = {tc.function_name.lower() for tc in agent_response.tool_calls}
            expected_actions = {action.lower() for action in phase.expected_actions}
            
            if expected_actions.intersection(performed_actions):
                return True
        
        return False
    
    def _get_dynamic_prompt(self, phase: ConversationPhase, agent_response: AgentResponse) -> Optional[str]:
        """Get a dynamic prompt based on agent actions and phase configuration"""
        if not phase.dynamic_prompts:
            return None
        
        response_content = agent_response.message.content.lower()
        
        # Check for tool usage patterns
        if agent_response.tool_calls:
            tool_names = [tc.tool_name.lower() for tc in agent_response.tool_calls]
            function_names = [tc.function_name.lower() for tc in agent_response.tool_calls]
            
            # Map tool/function usage to dynamic prompt conditions
            if any("read" in fn or "list" in fn for fn in function_names):
                if "file_read" in phase.dynamic_prompts:
                    return phase.dynamic_prompts["file_read"]
            
            if any("search" in fn for fn in function_names):
                if "analysis_started" in phase.dynamic_prompts:
                    return phase.dynamic_prompts["analysis_started"]
        
        # Check for content-based triggers
        analysis_keywords = ["analysis", "understand", "structure", "pattern", "architecture"]
        if any(keyword in response_content for keyword in analysis_keywords):
            if "analysis_started" in phase.dynamic_prompts:
                return phase.dynamic_prompts["analysis_started"]
        
        documentation_keywords = ["document", "findings", "insights", "summary"]
        if any(keyword in response_content for keyword in documentation_keywords):
            if "documentation_started" in phase.dynamic_prompts:
                return phase.dynamic_prompts["documentation_started"]
        
        # Default fallback - return the first available dynamic prompt
        if phase.dynamic_prompts:
            return next(iter(phase.dynamic_prompts.values()))
        
        return None
    
    def _should_terminate_session(self) -> bool:
        """Check if the session should be terminated"""
        # Check turn limit
        if self.current_turn >= self.config.max_turns:
            logger.info(f"Session {self.session_id} reached max turns ({self.config.max_turns})")
            return True
        
        # Check timeout
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        if session_duration >= self.config.timeout_seconds:
            logger.info(f"Session {self.session_id} reached timeout ({self.config.timeout_seconds}s)")
            self.status = SessionStatus.TIMEOUT
            return True
        
        # Check context limit
        if self.agent.total_tokens_used >= self.config.max_context_tokens:
            logger.info(f"Session {self.session_id} reached context limit ({self.config.max_context_tokens} tokens)")
            return True
        
        return False
    
    def _is_implementation_phase(self, phase: ConversationPhase) -> bool:
        """
        Check if a phase requires code implementation.
        CRITICAL FIX: Helper for progress monitoring.
        """
        # Check phase name/ID for implementation keywords
        phase_indicators = ['implement', 'bug', 'refactor', 'feature', 'fix', 'code']
        phase_name_lower = phase.name.lower()
        phase_id_lower = phase.phase_id.lower()
        
        return any(indicator in phase_name_lower or indicator in phase_id_lower 
                  for indicator in phase_indicators)
    
    def _count_recent_write_calls(self, current_phase_turns: int, phase_start_turn: int) -> int:
        """
        Count write_file tool calls in recent turns of current phase.
        CRITICAL FIX: Helper for progress monitoring.
        """
        count = 0
        # Check tool usage log for write calls in current phase
        for log_entry in self.tool_usage_log:
            turn = log_entry.get('turn', 0)
            # Only count turns in current phase
            if turn >= phase_start_turn and turn <= self.current_turn:
                tool_call = log_entry.get('tool_call', {})
                function_name = tool_call.get('function_name', '').lower()
                if 'write' in function_name or 'write_file' in function_name:
                    count += 1
        
        return count
    
    def _should_compress_context(self) -> bool:
        """Check if context should be compressed"""
        token_usage_ratio = self.agent.total_tokens_used / self.config.max_context_tokens
        return token_usage_ratio >= self.config.memory_compression_threshold
    
    async def _save_checkpoint(self) -> None:
        """Save a session checkpoint"""
        try:
            checkpoint = SessionCheckpoint(
                session_id=self.session_id,
                turn_number=self.current_turn,
                phase_id=self.current_phase.phase_id if self.current_phase else "",
                conversation_history=[msg.to_dict() for msg in self.agent.conversation_history],
                session_state={
                    "current_phase_index": self.current_phase_index,
                    "phase_history": self.phase_history,
                    "tool_usage_log": self.tool_usage_log,
                    "error_log": self.error_log
                },
                timestamp=datetime.now(),
                agent_statistics=self.agent.get_session_statistics()
            )
            
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.current_turn:03d}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            
            logger.debug(f"Saved checkpoint for turn {self.current_turn}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    async def _finalize_session(self) -> None:
        """Finalize the session"""
        self.session_end_time = datetime.now()
        
        if self.status == SessionStatus.ACTIVE:
            self.status = SessionStatus.COMPLETED
        
        # Finalize agent and capture throttling metadata
        try:
            self.agent_finalize_data = await self.agent.finalize_session()
        except Exception as e:
            logger.error(f"Error finalizing agent: {e}")
            self.agent_finalize_data = {}
        
        # CRITICAL: Clean up temporary workspace to free disk space
        # Each scenario creates a temp workspace that can be several MB
        # Without cleanup, thousands of scenarios will fill the disk
        for tool in self.available_tools:
            if hasattr(tool, 'cleanup_temp_workspace'):
                try:
                    tool.cleanup_temp_workspace()
                    logger.debug(f"✅ Cleaned up temp workspace for session {self.session_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp workspace: {e}")
        
        # CRITICAL: Clean up semantic search index to free disk space
        # Each scenario creates a ~3.3MB semantic index file
        # Without cleanup, 8,000 scenarios × multiple runs = 300GB+ disk usage
        if hasattr(self, 'semantic_retriever') and self.semantic_retriever is not None:
            try:
                self.semantic_retriever.cleanup_current_index()
                logger.debug(f"✅ Cleaned up semantic index for session {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup semantic index: {e}")
        
        logger.info(f"Session {self.session_id} finalized with status: {self.status.value}")
    
    def _create_session_result(self) -> Dict[str, Any]:
        """Create the final session result"""
        session_duration = (
            (self.session_end_time or datetime.now()) - self.session_start_time
        ).total_seconds()
        
        # THROTTLING-RESILIENT EVALUATION: Separate infrastructure issues from model performance
        throttling_errors = [err for err in self.error_log if err.get("error_type") == "api_throttling"]
        model_errors = [err for err in self.error_log if err.get("error_type") == "model_error"]
        
        # Get agent-level throttling metadata (from finalize_session)
        agent_throttling_count = getattr(self, 'agent_finalize_data', {}).get('throttling_error_count', 0)
        agent_infrastructure_count = getattr(self, 'agent_finalize_data', {}).get('infrastructure_error_count', 0)
        agent_model_errors = getattr(self, 'agent_finalize_data', {}).get('model_error_count', 0)
        
        # Determine if session was affected by throttling (from error log OR agent metadata)
        throttling_affected = len(throttling_errors) > 0 or agent_throttling_count > 0
        total_infrastructure_issues = len(throttling_errors) + agent_throttling_count + agent_infrastructure_count
        total_model_errors = len(model_errors) + agent_model_errors
        
        # Adjust status for fair evaluation: If session only failed due to throttling, mark as completed
        fair_status = self.status.value
        if self.status == SessionStatus.FAILED and total_model_errors == 0 and total_infrastructure_issues > 0:
            fair_status = "completed_with_throttling"
            logger.info(f"Session {self.session_id} marked as completed despite throttling (infrastructure issue)")
        
        return {
            "session_id": self.session_id,
            "agent_name": self.agent.name,
            "status": fair_status,  # Use throttling-adjusted status
            "original_status": self.status.value,  # Keep original for debugging
            "total_turns": self.current_turn,
            "total_phases": len(self.conversation_phases),
            "completed_phases": len(self.phase_history),
            "session_duration_seconds": session_duration,
            "total_tokens_used": self.agent.total_tokens_used,
            "total_cost": self.agent.total_cost,
            "conversation_history": [msg.to_dict() for msg in self.agent.conversation_history],
            "phase_history": self.phase_history,
            "tool_usage_log": self.tool_usage_log,
            "error_log": self.error_log,
            "human_interventions": self.human_interventions,
            "modified_files": self.modified_files,  # Track files written by agent (dict: path -> content)
            "agent_statistics": self.agent.get_session_statistics(),
            "scenario_context": self.scenario_context,
            "session_config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            
            # THROTTLING-RESILIENT EVALUATION METADATA (Enhanced with agent-level tracking)
            "throttling_affected": throttling_affected,
            "infrastructure_issues": total_infrastructure_issues,  # Session + Agent throttling
            "model_errors": total_model_errors,  # Session + Agent model errors
            "agent_throttling_errors": agent_throttling_count,  # NEW: Agent-level throttling
            "agent_infrastructure_errors": agent_infrastructure_count,  # NEW: Agent-level infrastructure errors
            "agent_model_errors": agent_model_errors,  # NEW: Agent-level model errors
            "evaluation_validity": "valid" if agent_throttling_count < 5 else "throttling_impacted",
            "fair_comparison": agent_throttling_count < 5,  # True = can be fairly compared with other models
            "throttling_severity": "none" if agent_throttling_count == 0 else ("low" if agent_throttling_count < 5 else ("moderate" if agent_throttling_count < 15 else "severe"))
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the session"""
        return {
            "session_id": self.session_id,
            "agent_name": self.agent.name,
            "status": self.status.value,
            "current_turn": self.current_turn,
            "current_phase": self.current_phase.name if self.current_phase else None,
            "total_tokens_used": self.agent.total_tokens_used,
            "session_duration": (
                datetime.now() - self.session_start_time
            ).total_seconds(),
            "tools_used": len(self.tool_usage_log),
            "errors": len(self.error_log)
        }
