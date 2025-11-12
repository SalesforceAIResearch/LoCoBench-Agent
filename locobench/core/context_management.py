"""
Context Management System for LoCoBench-Agent

Provides intelligent context window management for multi-turn agent conversations
to handle model context limits gracefully while preserving evaluation integrity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextManagementStrategy(Enum):
    """Available context management strategies"""
    NONE = "none"           # No management - may fail on long contexts
    BASIC = "basic"         # Simple turn deletion strategy
    ADAPTIVE = "adaptive"   # Intelligent compression and summarization


@dataclass
class ContextManagementConfig:
    """Configuration for context management"""
    strategy: ContextManagementStrategy = ContextManagementStrategy.ADAPTIVE
    
    # Thresholds (as percentage of max context) - MUCH more aggressive
    early_warning_threshold: float = 0.4    # 40% (very aggressive)
    critical_threshold: float = 0.6         # 60% (very aggressive)
    
    # Basic strategy settings
    preserve_initial_turns: int = 2          # Always keep first N turns
    
    # Adaptive strategy settings
    preserve_recent_turns: int = 3           # Keep last N turns in detail
    enable_conversation_summary: bool = True # Generate summaries of removed turns
    enable_file_compression: bool = True     # Compress inactive project files
    enable_architectural_summary: bool = True # Create architectural summaries
    
    # Token counting settings
    model_name: str = "gpt-4"               # For tiktoken encoding
    max_context_tokens: int = 128000        # Model's context limit
    response_buffer_tokens: int = 4096      # Reserve tokens for response


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    turn_number: int
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata
        }


@dataclass
class ContextState:
    """Current state of the conversation context"""
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    project_files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    active_files: List[str] = field(default_factory=list)       # currently active files
    conversation_summary: str = ""                               # summary of compressed turns
    architectural_summary: str = ""                             # summary of compressed files
    metadata: Dict[str, Any] = field(default_factory=dict)      # additional metadata (e.g., file_structure)
    
    total_tokens: int = 0
    last_compression_turn: int = 0
    compression_history: List[Dict[str, Any]] = field(default_factory=list)


class BaseContextManager(ABC):
    """Abstract base class for context management strategies"""
    
    def __init__(self, config: ContextManagementConfig):
        self.config = config
        # Try to get encoding for model, with fallback for unknown models
        try:
            self.encoding = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            # For unknown/new models, use intelligent fallback based on model name
            # Encoding guide:
            # - o200k_base: o1, o3, o4 series (reasoning models)
            # - cl100k_base: GPT-4, GPT-4o, GPT-4.1, GPT-5 series (standard models)
            fallback_encoding = "cl100k_base"  # Default for most modern models
            
            model_lower = config.model_name.lower()
            
            # Check for o-series reasoning models (use o200k_base)
            if any(prefix in model_lower for prefix in ["o1", "o3", "o4"]):
                fallback_encoding = "o200k_base"
            # Check for GPT-5, GPT-4.x, GPT-4o series (use cl100k_base)
            elif any(prefix in model_lower for prefix in ["gpt-5", "gpt-4.1", "gpt-4o", "gpt-4"]):
                fallback_encoding = "cl100k_base"
            
            logger.info(f"Unknown model '{config.model_name}', using {fallback_encoding} encoding")
            self.encoding = tiktoken.get_encoding(fallback_encoding)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using character approximation")
            return len(text) // 4  # Rough approximation: 4 chars per token
    
    def calculate_context_usage(self, state: ContextState) -> Tuple[int, float]:
        """Calculate current token usage and percentage"""
        total_tokens = 0
        
        # Count conversation tokens
        for turn in state.conversation_turns:
            if turn.token_count == 0:
                turn.token_count = self.count_tokens(turn.content)
            total_tokens += turn.token_count
        
        # Count project file tokens
        for filename, content in state.project_files.items():
            total_tokens += self.count_tokens(content)
        
        # Count summary tokens
        if state.conversation_summary:
            total_tokens += self.count_tokens(state.conversation_summary)
        if state.architectural_summary:
            total_tokens += self.count_tokens(state.architectural_summary)
        
        state.total_tokens = total_tokens
        usage_percentage = total_tokens / self.config.max_context_tokens
        
        return total_tokens, usage_percentage
    
    @abstractmethod
    def should_compress(self, state: ContextState) -> bool:
        """Check if context compression is needed"""
        pass
    
    @abstractmethod
    def compress_context(self, state: ContextState) -> ContextState:
        """Compress the context to fit within limits"""
        pass
    
    def log_compression(self, state: ContextState, compression_type: str, details: Dict[str, Any]):
        """Log compression event for analysis"""
        compression_event = {
            "timestamp": datetime.now().isoformat(),
            "compression_type": compression_type,
            "turn_number": len(state.conversation_turns),
            "tokens_before": details.get("tokens_before", 0),
            "tokens_after": details.get("tokens_after", 0),
            "details": details
        }
        state.compression_history.append(compression_event)
        
        logger.info(f"Context compression: {compression_type} at turn {len(state.conversation_turns)}")
        logger.info(f"Tokens: {details.get('tokens_before', 0)} → {details.get('tokens_after', 0)}")


class NoContextManager(BaseContextManager):
    """No context management - allows natural overflow (may cause failures)"""
    
    def should_compress(self, state: ContextState) -> bool:
        return False
    
    def compress_context(self, state: ContextState) -> ContextState:
        return state  # No compression


class BasicContextManager(BaseContextManager):
    """Basic context management using simple turn deletion"""
    
    def should_compress(self, state: ContextState) -> bool:
        _, usage_percentage = self.calculate_context_usage(state)
        return usage_percentage >= self.config.early_warning_threshold
    
    def compress_context(self, state: ContextState) -> ContextState:
        """Compress by deleting turns (preserve initial + recent turns)"""
        tokens_before, usage_before = self.calculate_context_usage(state)
        
        if not self.should_compress(state):
            return state
        
        # Find turns to delete (skip initial turns and recent turns)
        turns = state.conversation_turns
        preserve_initial = self.config.preserve_initial_turns
        
        # Find the oldest deletable turn (after preserved initial turns)
        deletable_turns = []
        for i, turn in enumerate(turns):
            if i >= preserve_initial:  # Skip initial preserved turns
                deletable_turns.append(i)
        
        if not deletable_turns:
            logger.warning("No deletable turns found - cannot compress further")
            return state
        
        # Delete the oldest deletable turn
        turn_to_delete = deletable_turns[0]
        deleted_turn = turns.pop(turn_to_delete)
        
        # Update turn numbers
        for i, turn in enumerate(turns):
            turn.turn_number = i + 1
        
        # Recalculate tokens
        tokens_after, usage_after = self.calculate_context_usage(state)
        
        # Log compression
        self.log_compression(state, "basic_turn_deletion", {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "deleted_turn": turn_to_delete + 1,
            "deleted_content_preview": deleted_turn.content[:100] + "..."
        })
        
        return state


class AdaptiveContextManager(BaseContextManager):
    """Adaptive context management with intelligent compression"""
    
    def should_compress(self, state: ContextState) -> bool:
        _, usage_percentage = self.calculate_context_usage(state)
        return usage_percentage >= self.config.early_warning_threshold
    
    def compress_context(self, state: ContextState) -> ContextState:
        """Intelligent context compression with multiple strategies"""
        
        # CRITICAL: First enforce maximum message size limits to prevent API errors
        max_message_size = 8_000_000  # 8MB safety margin (OpenAI limit is 10MB)
        for turn in state.conversation_turns:
            if len(turn.content) > max_message_size:
                logger.warning(f"Turn {turn.turn_number} too large ({len(turn.content)} chars), truncating")
                turn.content = turn.content[:max_message_size] + "\n\n[Content truncated due to size limit]"
        
        tokens_before, usage_before = self.calculate_context_usage(state)
        
        if usage_before < self.config.early_warning_threshold:
            return state
        
        compression_actions = []
        
        # Strategy 1: Compress conversation history (80% threshold)
        if usage_before >= self.config.early_warning_threshold:
            state = self._compress_conversation_history(state)
            compression_actions.append("conversation_history")
        
        # Strategy 2: Compress inactive files (still above threshold)
        tokens_mid, usage_mid = self.calculate_context_usage(state)
        if usage_mid >= self.config.early_warning_threshold and self.config.enable_file_compression:
            state = self._compress_inactive_files(state)
            compression_actions.append("inactive_files")
        
        # Strategy 3: Aggressive truncation (95% threshold)
        tokens_after, usage_after = self.calculate_context_usage(state)
        if usage_after >= self.config.critical_threshold:
            state = self._aggressive_truncation(state)
            compression_actions.append("aggressive_truncation")
        
        # Final token count
        tokens_final, usage_final = self.calculate_context_usage(state)
        
        # Log compression
        self.log_compression(state, "adaptive_compression", {
            "tokens_before": tokens_before,
            "tokens_after": tokens_final,
            "usage_before": usage_before,
            "usage_after": usage_final,
            "actions": compression_actions
        })
        
        return state
    
    def _compress_conversation_history(self, state: ContextState) -> ContextState:
        """Compress conversation by summarizing old turns"""
        turns = state.conversation_turns
        preserve_recent = self.config.preserve_recent_turns
        
        if len(turns) <= preserve_recent + 2:  # Not enough turns to compress
            return state
        
        # Identify turns to summarize (all except recent ones)
        turns_to_summarize = turns[:-preserve_recent] if preserve_recent > 0 else turns[:-1]
        recent_turns = turns[-preserve_recent:] if preserve_recent > 0 else [turns[-1]]
        
        # Generate summary of old turns
        if self.config.enable_conversation_summary:
            summary_content = self._generate_conversation_summary(turns_to_summarize)
            state.conversation_summary = summary_content
        
        # Keep only recent turns
        state.conversation_turns = recent_turns
        
        # Update turn numbers
        for i, turn in enumerate(state.conversation_turns):
            turn.turn_number = i + 1
        
        return state
    
    def _compress_inactive_files(self, state: ContextState) -> ContextState:
        """Compress inactive project files to architectural summaries"""
        if not self.config.enable_file_compression:
            return state
        
        # Identify active files (mentioned in recent turns)
        active_files = set(state.active_files)
        recent_turns = state.conversation_turns[-3:]  # Check last 3 turns
        
        for turn in recent_turns:
            # Simple heuristic: look for file extensions in content
            import re
            file_mentions = re.findall(r'[\w/]+\.\w+', turn.content)
            for mention in file_mentions:
                if mention in state.project_files:
                    active_files.add(mention)
        
        # CRITICAL FIX: Always preserve source files (files in src/ directories)
        # and other important files that agents commonly need to access
        protected_patterns = [
            '/src/', '//src//', '/source/', '//source//',
            '.c', '.cpp', '.h', '.hpp', '.py', '.js', '.ts', '.java', '.rs',
            '.go', '.php', '.rb', '.cs', '.swift', '.kt', '.scala'
        ]
        
        protected_files = set()
        for filename in state.project_files.keys():
            # Protect source files and files with common source extensions
            if any(pattern in filename for pattern in protected_patterns):
                protected_files.add(filename)
        
        # Only compress non-protected, truly inactive files (documentation, configs, etc.)
        inactive_files = set(state.project_files.keys()) - active_files - protected_files
        
        # Only compress files that are clearly documentation/config files
        compressible_patterns = [
            'README', 'LICENSE', 'CONTRIBUTING', '.md', '.txt', '.yml', '.yaml',
            '.json', '.xml', '.toml', '.ini', '.cfg', '.conf', '.properties'
        ]
        
        truly_inactive_files = set()
        for filename in inactive_files:
            if any(pattern in filename for pattern in compressible_patterns):
                truly_inactive_files.add(filename)
        
        if truly_inactive_files and self.config.enable_architectural_summary:
            architectural_summary = self._generate_architectural_summary(
                {f: state.project_files[f] for f in truly_inactive_files}
            )
            state.architectural_summary = architectural_summary
            
            # Only remove truly inactive documentation/config files
            for filename in truly_inactive_files:
                del state.project_files[filename]
            
            logger.debug(f"Context compression: protected {len(protected_files)} source files, "
                        f"compressed {len(truly_inactive_files)} documentation files")
        
        return state
    
    def _aggressive_truncation(self, state: ContextState) -> ContextState:
        """Aggressive truncation when approaching critical limits"""
        # Keep only last 2 turns + current project state
        if len(state.conversation_turns) > 2:
            state.conversation_turns = state.conversation_turns[-2:]
            
            # Update turn numbers
            for i, turn in enumerate(state.conversation_turns):
                turn.turn_number = i + 1
        
        # Further compress files if needed
        if len(state.project_files) > 3:
            # Keep only 3 most recently mentioned files
            files_to_keep = list(state.project_files.keys())[:3]
            state.project_files = {f: state.project_files[f] for f in files_to_keep}
        
        return state
    
    def _generate_conversation_summary(self, turns: List[ConversationTurn]) -> str:
        """Generate a summary of conversation turns"""
        if not turns:
            return ""
        
        # Simple extractive summary (in production, could use LLM summarization)
        key_points = []
        for turn in turns:
            # Extract first sentence or key phrases
            sentences = turn.content.split('.')
            if sentences:
                key_points.append(f"Turn {turn.turn_number}: {sentences[0][:100]}...")
        
        summary = "CONVERSATION SUMMARY:\n" + "\n".join(key_points[:5])  # Max 5 key points
        return summary
    
    def _generate_architectural_summary(self, files: Dict[str, str]) -> str:
        """Generate architectural summary of files"""
        if not files:
            return ""
        
        summary_parts = ["ARCHITECTURAL SUMMARY:"]
        
        for filename, content in files.items():
            # Extract key information (classes, functions, imports)
            lines = content.split('\n')
            key_lines = []
            
            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if (line.startswith('class ') or 
                    line.startswith('def ') or 
                    line.startswith('import ') or 
                    line.startswith('from ')):
                    key_lines.append(line)
            
            if key_lines:
                summary_parts.append(f"{filename}: {', '.join(key_lines[:3])}")
        
        return "\n".join(summary_parts)


def create_context_manager(config: ContextManagementConfig) -> BaseContextManager:
    """Factory function to create appropriate context manager"""
    if config.strategy == ContextManagementStrategy.NONE:
        return NoContextManager(config)
    elif config.strategy == ContextManagementStrategy.BASIC:
        return BasicContextManager(config)
    elif config.strategy == ContextManagementStrategy.ADAPTIVE:
        return AdaptiveContextManager(config)
    else:
        raise ValueError(f"Unknown context management strategy: {config.strategy}")
