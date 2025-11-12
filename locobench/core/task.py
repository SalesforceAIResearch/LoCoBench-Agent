"""
Task definition and management for LoCoBench
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass


class TaskCategory(Enum):
    """Task categories for LoCoBench-Agent - Extended to 12 categories"""
    
    # Original 8 categories (evolved for agent scenarios)
    INTERACTIVE_ARCHITECTURE_EXPLORATION = "interactive_architecture_exploration"  # evolved from architectural_understanding
    GUIDED_MULTI_FILE_REFACTORING = "guided_multi_file_refactoring"  # evolved from cross_file_refactoring
    COLLABORATIVE_FEATURE_DEVELOPMENT = "collaborative_feature_development"  # evolved from feature_implementation
    INTERACTIVE_DEBUGGING_SESSIONS = "interactive_debugging_sessions"  # evolved from bug_investigation
    EXTENDED_DEVELOPMENT_PROJECTS = "extended_development_projects"  # evolved from multi_session_development
    INTERACTIVE_CODE_EXPLORATION = "interactive_code_exploration"  # evolved from code_comprehension
    TEST_DRIVEN_DEVELOPMENT_SESSIONS = "test_driven_development_sessions"  # evolved from integration_testing
    INTERACTIVE_SECURITY_AUDITING = "interactive_security_auditing"  # evolved from security_analysis
    
    # 4 NEW agent-specific categories
    HUMAN_AGENT_COLLABORATION = "human_agent_collaboration"  # NEW
    TOOL_MASTERY_EVALUATION = "tool_mastery_evaluation"  # NEW
    DOCUMENTATION_GENERATION = "documentation_generation"  # NEW
    DEPLOYMENT_DEVOPS = "deployment_devops"  # NEW
    
    # Legacy compatibility (deprecated but maintained for backward compatibility)
    ARCHITECTURAL_UNDERSTANDING = "architectural_understanding"  # deprecated -> use INTERACTIVE_ARCHITECTURE_EXPLORATION
    CROSS_FILE_REFACTORING = "cross_file_refactoring"  # deprecated -> use GUIDED_MULTI_FILE_REFACTORING
    FEATURE_IMPLEMENTATION = "feature_implementation"  # deprecated -> use COLLABORATIVE_FEATURE_DEVELOPMENT
    BUG_INVESTIGATION = "bug_investigation"  # deprecated -> use INTERACTIVE_DEBUGGING_SESSIONS
    MULTI_SESSION_DEVELOPMENT = "multi_session_development"  # deprecated -> use EXTENDED_DEVELOPMENT_PROJECTS
    CODE_COMPREHENSION = "code_comprehension"  # deprecated -> use INTERACTIVE_CODE_EXPLORATION
    INTEGRATION_TESTING = "integration_testing"  # deprecated -> use TEST_DRIVEN_DEVELOPMENT_SESSIONS
    SECURITY_ANALYSIS = "security_analysis"  # deprecated -> use INTERACTIVE_SECURITY_AUDITING


class DifficultyLevel(Enum):
    """Difficulty levels for tasks"""
    EASY = "easy"
    MEDIUM = "medium" 
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Task:
    """Represents a single evaluation task"""
    id: str
    category: TaskCategory
    difficulty: DifficultyLevel
    description: str
    context_files: List[str]
    context_length: int
    information_coverage: float
    session_count: int = 1
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"Task(id='{self.id}', category={self.category.value}, difficulty={self.difficulty.value})" 