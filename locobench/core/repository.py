"""
Repository utilities for LoCoBench synthetic projects
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path


@dataclass
class SyntheticRepository:
    """Represents a synthetically generated repository for evaluation"""
    name: str
    language: str
    file_count: int
    total_tokens: int
    complexity_score: float
    domain: str
    files: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'language': self.language,
            'file_count': self.file_count,
            'total_tokens': self.total_tokens,
            'complexity_score': self.complexity_score,
            'domain': self.domain,
            'files': self.files
        }


# Legacy compatibility - alias to new synthetic class
Repository = SyntheticRepository 