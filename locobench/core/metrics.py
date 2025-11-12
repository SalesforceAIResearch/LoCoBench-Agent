"""
Evaluation metrics for LoCoBench
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for LoCoBench evaluation metrics across 4 dimensions"""
    
    # Software Engineering Excellence (8 metrics)
    architectural_coherence: float = 0.0  # ACS - Software Engineering Excellence
    dependency_traversal: float = 0.0     # DTA - Software Engineering Excellence  
    cross_file_reasoning: float = 0.0     # CFRD - Software Engineering Excellence
    system_thinking: float = 0.0          # STS - Already here
    robustness: float = 0.0               # RS - Already here
    comprehensiveness: float = 0.0        # CS - Already here
    innovation: float = 0.0               # IS - Already here
    solution_elegance: float = 0.0        # SES - Already here
    
    # Functional Correctness (4 metrics)
    compilation_success: float = 0.0      # Already here
    unit_test_performance: float = 0.0    # Already here
    integration_performance: float = 0.0  # Already here
    incremental_development: float = 0.0  # IDC - Functional Correctness
    
    # Code Quality Assessment (3 metrics)
    security_analysis: float = 0.0        # Already here
    average_issues: float = 0.0           # Already here (lower is better)
    code_style: float = 0.0               # Already here
    
    # Long-Context Utilization (2 metrics)
    information_coverage: float = 0.0     # ICU - Long-Context Utilization
    multi_session_memory: float = 0.0     # MMR - Long-Context Utilization
    
    # Composite score (0-5 scale)
    composite_score: Optional[float] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate LCBS (LoCoBench Score) across 4 dimensions"""
        if weights is None:
            weights = {
                # Software Engineering Excellence (40% - 8 metrics)
                "architectural_coherence": 0.05,
                "dependency_traversal": 0.05,
                "cross_file_reasoning": 0.05,
                "system_thinking": 0.05,
                "robustness": 0.05,
                "comprehensiveness": 0.05,
                "innovation": 0.05,
                "solution_elegance": 0.05,
                
                # Functional Correctness (30% - 4 metrics)
                "compilation_success": 0.075,
                "unit_test_performance": 0.075,
                "integration_performance": 0.075,
                "incremental_development": 0.075,
                
                # Code Quality Assessment (20% - 3 metrics)
                "security_analysis": 0.067,
                "average_issues": 0.067,  # Note: inverted - lower is better
                "code_style": 0.066,
                
                # Long-Context Utilization (10% - 2 metrics)
                "information_coverage": 0.05,
                "multi_session_memory": 0.05
            }
        
        score = (
            # Software Engineering Excellence
            self.architectural_coherence * weights["architectural_coherence"] +
            self.dependency_traversal * weights["dependency_traversal"] +
            self.cross_file_reasoning * weights["cross_file_reasoning"] +
            self.system_thinking * weights["system_thinking"] +
            self.robustness * weights["robustness"] +
            self.comprehensiveness * weights["comprehensiveness"] +
            self.innovation * weights["innovation"] +
            self.solution_elegance * weights["solution_elegance"] +
            
            # Functional Correctness
            self.compilation_success * weights["compilation_success"] +
            self.unit_test_performance * weights["unit_test_performance"] +
            self.integration_performance * weights["integration_performance"] +
            self.incremental_development * weights["incremental_development"] +
            
            # Code Quality Assessment
            self.security_analysis * weights["security_analysis"] +
            (1.0 - self.average_issues) * weights["average_issues"] +  # Inverted
            self.code_style * weights["code_style"] +
            
            # Long-Context Utilization
            self.information_coverage * weights["information_coverage"] +
            self.multi_session_memory * weights["multi_session_memory"]
        )
        
        # Scale to 0-5 range
        self.composite_score = score * 5.0
        return self.composite_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "architectural_coherence": self.architectural_coherence,
            "dependency_traversal": self.dependency_traversal,
            "multi_session_memory": self.multi_session_memory,
            "cross_file_reasoning": self.cross_file_reasoning,
            "incremental_development": self.incremental_development,
            "information_coverage": self.information_coverage,
            "composite_score": self.composite_score,
            "metadata": self.metadata
        } 