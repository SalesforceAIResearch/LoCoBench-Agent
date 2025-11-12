"""
Agent-Specific Evaluation Metrics for LoCoBench-Agent

This module implements the comprehensive 25-metric evaluation system for agent evaluation,
extending the original 17 metrics with 8 new agent-specific metrics across 5 dimensions.

Metric Framework (25 Total Metrics):
- 🤖 Agent Interaction Excellence (25% - 8 NEW metrics)
- 🏗️ Software Engineering Excellence (30% - 8 metrics) 
- ⚙️ Functional Correctness (25% - 4 metrics)
- 🔍 Code Quality Assessment (15% - 3 metrics)
- 🧠 Long-Context Utilization (5% - 2 metrics)
"""

import asyncio
import json
import logging
import math
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.metrics import EvaluationMetrics
from ..core.agent_session import AgentSession
from ..generation.metric_algorithms import LoCoBenchMetricsCalculator

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of evaluation metrics"""
    AGENT_INTERACTION = "agent_interaction"
    SOFTWARE_ENGINEERING = "software_engineering"
    FUNCTIONAL_CORRECTNESS = "functional_correctness"
    CODE_QUALITY = "code_quality"
    LONG_CONTEXT_UTILIZATION = "long_context_utilization"


@dataclass
class AgentMetricResult:
    """Result of a single metric evaluation"""
    metric_name: str
    category: MetricCategory
    score: float  # 0.0 to 5.0
    max_score: float = 5.0
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "category": self.category.value,
            "score": self.score,
            "max_score": self.max_score,
            "weight": self.weight,
            "details": self.details,
            "explanation": self.explanation,
            "normalized_score": self.score / self.max_score if self.max_score > 0 else 0.0
        }


@dataclass
class AgentEvaluationResults:
    """Complete evaluation results for an agent"""
    agent_name: str
    scenario_id: str
    session_id: str
    
    # Individual metric results
    metric_results: List[AgentMetricResult] = field(default_factory=list)
    
    # Category scores
    category_scores: Dict[MetricCategory, float] = field(default_factory=dict)
    
    # Overall scores
    overall_score: float = 0.0  # LoCoBench Agent Score (LCAS) - Legacy, for backward compatibility
    max_overall_score: float = 5.0
    
    # LCBA Final Scores (Primary evaluation metrics)
    lcba_comprehension: float = 0.0  # LCBA-Comprehension: Quality, depth, correctness (23 metrics)
    lcba_efficiency: float = 0.0  # LCBA-Efficiency: Speed, conciseness, resource optimization (8 metrics)
    
    # Session metadata
    total_turns: int = 0
    session_duration: float = 0.0
    
    # BUGFIX: Add missing session data fields
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_usage_log: List[Dict[str, Any]] = field(default_factory=list)
    modified_files: Dict[str, str] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    session_status: str = "unknown"
    completed_phases: int = 0
    total_phases: int = 0
    error_rate: float = 0.0
    phases_completed: List[str] = field(default_factory=list)
    
    # BUGFIX: Add scenario context to preserve conversation phases
    scenario_context: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    evaluator_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        # Handle metric_results - could be AgentMetricResult objects or dicts
        metric_results_dict = []
        for mr in self.metric_results:
            if hasattr(mr, 'to_dict'):
                metric_results_dict.append(mr.to_dict())
            elif isinstance(mr, dict):
                metric_results_dict.append(mr)
            else:
                # Fallback for other types
                metric_results_dict.append(str(mr))
        
        return {
            "agent_name": self.agent_name,
            "scenario_id": self.scenario_id,
            "session_id": self.session_id,
            "metric_results": metric_results_dict,
            "category_scores": {cat.value: score for cat, score in self.category_scores.items()},
            "overall_score": self.overall_score,
            "max_overall_score": self.max_overall_score,
            "normalized_overall_score": self.overall_score / self.max_overall_score if self.max_overall_score > 0 else 0.0,
            # LCBA Final Scores (Primary evaluation metrics)
            "lcba_comprehension": self.lcba_comprehension,
            "lcba_efficiency": self.lcba_efficiency,
            "total_turns": self.total_turns,
            "session_duration": self.session_duration,
            # BUGFIX: Include session data in output
            "conversation_history": self.conversation_history,
            "tool_usage_log": self.tool_usage_log,
            "modified_files": self.modified_files,
            "error_log": self.error_log,
            "session_status": self.session_status,
            "completed_phases": self.completed_phases,
            "total_phases": self.total_phases,
            "error_rate": self.error_rate,
            "phases_completed": self.phases_completed,
            # BUGFIX: Include scenario context to preserve conversation phases
            "scenario_context": self.scenario_context,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat() if hasattr(self.evaluation_timestamp, 'isoformat') else str(self.evaluation_timestamp),
            "evaluator_version": self.evaluator_version
        }


class AgentMetricsCalculator:
    """
    Calculator for agent-specific evaluation metrics
    
    Implements all 31 metrics across 5 dimensions with proper weighting
    and normalization for fair comparison across different agent types.
    """
    
    # Metric weights by category (must sum to 1.0)
    CATEGORY_WEIGHTS = {
        MetricCategory.AGENT_INTERACTION: 0.25,          # 8 metrics - Agent-specific
        MetricCategory.SOFTWARE_ENGINEERING: 0.30,      # 10 metrics - Now includes efficiency!
        MetricCategory.FUNCTIONAL_CORRECTNESS: 0.25,    # 4 metrics - Maintained
        MetricCategory.CODE_QUALITY: 0.15,              # 3 metrics - Reduced from 20%
        MetricCategory.LONG_CONTEXT_UTILIZATION: 0.05   # 2 metrics - Reduced from 10%
    }
    
    # Agent Interaction Excellence metric weights (within 25% category)
    # Updated: Split context_management_score, tool_usage_efficiency, turn_optimization_score
    # Removed 2 metrics, redistributed weights: 9 metrics × 0.111 = 1.0
    AGENT_INTERACTION_WEIGHTS = {
        "tool_efficiency": 0.111,                # Minimizing tool calls (EFFICIENCY)
        "tool_quality": 0.111,                   # Tool mastery and correct usage (COMPREHENSION)
        # "collaboration_intelligence": 0.091,   # REMOVED: Keyword-based heuristic bias
        "context_efficiency": 0.111,             # Token usage efficiency (EFFICIENCY)
        "context_quality": 0.111,                # Context consistency + retention (COMPREHENSION)
        "adaptive_learning_rate": 0.111,         # Learning from feedback within session
        "turn_efficiency": 0.111,                # Completing tasks in fewer turns (EFFICIENCY)
        "turn_effectiveness": 0.111,             # Phase completion and turn quality (COMPREHENSION)
        "error_recovery_capability": 0.112,      # Handling mistakes and corrections
        "exploration_exploitation_balance": 0.111, # Information gathering strategy
        # "communication_clarity_score": 0.091   # REMOVED: Keyword-based heuristic bias
    }
    
    # Software Engineering Excellence weights (within 30% category)
    # Updated: Split solution_elegance into solution_quality + solution_conciseness
    SOFTWARE_ENGINEERING_WEIGHTS = {
        "architectural_coherence": 0.091,     # 10/11 ≈ 0.091
        "dependency_traversal": 0.091,        # 10/11 ≈ 0.091
        "cross_file_reasoning": 0.091,        # 10/11 ≈ 0.091
        "system_thinking": 0.091,             # 10/11 ≈ 0.091
        "robustness": 0.091,                  # 10/11 ≈ 0.091
        "comprehensiveness": 0.091,           # 10/11 ≈ 0.091
        "innovation": 0.091,                  # 10/11 ≈ 0.091
        "solution_quality": 0.091,            # NEW: Readability, maintainability (COMPREHENSION)
        "solution_conciseness": 0.091,        # NEW: Brevity, simplicity (EFFICIENCY)
        "runtime_efficiency": 0.091,          # Time complexity and execution speed
        "memory_efficiency": 0.091            # Space complexity and memory usage
    }
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validate weights sum to 1.0
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that all weights sum to 1.0"""
        
        category_sum = sum(self.CATEGORY_WEIGHTS.values())
        if abs(category_sum - 1.0) > 0.01:
            raise ValueError(f"Category weights must sum to 1.0, got {category_sum}")
        
        interaction_sum = sum(self.AGENT_INTERACTION_WEIGHTS.values())
        if abs(interaction_sum - 1.0) > 0.01:
            raise ValueError(f"Agent interaction weights must sum to 1.0, got {interaction_sum}")
        
        se_sum = sum(self.SOFTWARE_ENGINEERING_WEIGHTS.values())
        if abs(se_sum - 1.0) > 0.01:
            raise ValueError(f"Software engineering weights must sum to 1.0, got {se_sum}")
    
    async def evaluate_agent_session(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any],
        agent_name: str
    ) -> AgentEvaluationResults:
        """Evaluate a complete agent session across all metrics"""
        
        self.logger.info(f"Evaluating agent session: {session_result.get('session_id', 'unknown')}")
        
        # Initialize results with BUGFIX: populate session data fields
        results = AgentEvaluationResults(
            agent_name=agent_name,
            scenario_id=scenario_context.get("scenario_id", "unknown"),
            session_id=session_result.get("session_id", "unknown"),
            total_turns=session_result.get("total_turns", 0),
            session_duration=session_result.get("session_duration_seconds", 0.0),
            # BUGFIX: Extract session data from session_result
            conversation_history=session_result.get("conversation_history", []),
            tool_usage_log=session_result.get("tool_usage_log", []),
            modified_files=session_result.get("modified_files", {}),
            error_log=session_result.get("error_log", []),
            session_status=session_result.get("status", "unknown"),
            completed_phases=session_result.get("completed_phases", 0),
            total_phases=session_result.get("total_phases", 0),
            error_rate=session_result.get("error_rate", 0.0),
            phases_completed=session_result.get("phases_completed", []),
            # BUGFIX: Include scenario context to preserve conversation phases
            scenario_context=scenario_context
        )
        
        # Calculate all metrics
        metric_results = []
        
        # 1. Agent Interaction Excellence (8 metrics)
        interaction_metrics = await self._calculate_agent_interaction_metrics(
            session_result, scenario_context
        )
        metric_results.extend(interaction_metrics)
        
        # 2. Software Engineering Excellence (8 metrics) 
        se_metrics = await self._calculate_software_engineering_metrics(
            session_result, scenario_context
        )
        metric_results.extend(se_metrics)
        
        # 3. Functional Correctness (4 metrics)
        fc_metrics = await self._calculate_functional_correctness_metrics(
            session_result, scenario_context
        )
        metric_results.extend(fc_metrics)
        
        # 4. Code Quality Assessment (3 metrics)
        cq_metrics = await self._calculate_code_quality_metrics(
            session_result, scenario_context
        )
        metric_results.extend(cq_metrics)
        
        # 5. Long-Context Utilization (2 metrics)
        lcu_metrics = await self._calculate_long_context_metrics(
            session_result, scenario_context
        )
        metric_results.extend(lcu_metrics)
        
        # Store metric results
        results.metric_results = metric_results
        
        # Calculate category scores
        results.category_scores = self._calculate_category_scores(metric_results)
        
        # Calculate overall score (LCAS - LoCoBench Agent Score) - Legacy
        results.overall_score = self._calculate_overall_score(results.category_scores)
        
        # Calculate LCBA Final Scores (Primary evaluation metrics)
        results.lcba_comprehension = self._calculate_lcba_comprehension(metric_results)
        results.lcba_efficiency = self._calculate_lcba_efficiency(metric_results)
        
        self.logger.info(f"Evaluation complete. Overall: {results.overall_score:.2f}/5.0, "
                        f"LCBA-Comp: {results.lcba_comprehension:.2f}/5.0, "
                        f"LCBA-Eff: {results.lcba_efficiency:.2f}/5.0")
        
        return results
    
    async def _calculate_agent_interaction_metrics(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> List[AgentMetricResult]:
        """Calculate the 11 Agent Interaction Excellence metrics"""
        
        metrics = []
        
        # 1a. Tool Efficiency (TES) - NEW: Split from tool_usage_efficiency
        tes_score = await self._calculate_tool_efficiency(session_result)
        metrics.append(AgentMetricResult(
            metric_name="tool_efficiency",
            category=MetricCategory.AGENT_INTERACTION,
            score=tes_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["tool_efficiency"],
            explanation="Measures minimizing tool calls - fewer calls to complete tasks (EFFICIENCY)"
        ))
        
        # 1b. Tool Quality (TQS) - NEW: Split from tool_usage_efficiency
        tqs_score = await self._calculate_tool_quality(session_result)
        metrics.append(AgentMetricResult(
            metric_name="tool_quality",
            category=MetricCategory.AGENT_INTERACTION,
            score=tqs_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["tool_quality"],
            explanation="Measures tool mastery - correct selection and successful execution (COMPREHENSION)"
        ))
        
        # 2. Collaboration Intelligence (CI)
        # REMOVED: collaboration_intelligence (keyword-based heuristic bias)
        # ci_score = await self._calculate_collaboration_intelligence(session_result)
        # metrics.append(AgentMetricResult(
        #     metric_name="collaboration_intelligence",
        #     category=MetricCategory.AGENT_INTERACTION,
        #     score=ci_score,
        #     weight=self.AGENT_INTERACTION_WEIGHTS["collaboration_intelligence"],
        #     explanation="Evaluates the quality of human-agent interaction and collaboration"
        # ))
        
        # 3a. Context Efficiency (CES) - NEW: Split from context_management_score
        ces_score = await self._calculate_context_efficiency(session_result)
        metrics.append(AgentMetricResult(
            metric_name="context_efficiency",
            category=MetricCategory.AGENT_INTERACTION,
            score=ces_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["context_efficiency"],
            explanation="Measures token usage efficiency - rewards low token usage (EFFICIENCY)"
        ))
        
        # 3b. Context Quality (CQS) - NEW: Split from context_management_score
        cqs_score = await self._calculate_context_quality(session_result)
        metrics.append(AgentMetricResult(
            metric_name="context_quality",
            category=MetricCategory.AGENT_INTERACTION,
            score=cqs_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["context_quality"],
            explanation="Measures context consistency and information retention (COMPREHENSION)"
        ))
        
        # 4. Adaptive Learning Rate (ALR)
        alr_score = await self._calculate_adaptive_learning_rate(session_result)
        metrics.append(AgentMetricResult(
            metric_name="adaptive_learning_rate",
            category=MetricCategory.AGENT_INTERACTION,
            score=alr_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["adaptive_learning_rate"],
            explanation="Measures how well the agent learns from feedback within the session"
        ))
        
        # 5a. Turn Efficiency (TTES) - NEW: Split from turn_optimization_score
        ttes_score = await self._calculate_turn_efficiency(session_result)
        metrics.append(AgentMetricResult(
            metric_name="turn_efficiency",
            category=MetricCategory.AGENT_INTERACTION,
            score=ttes_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["turn_efficiency"],
            explanation="Measures completing tasks in fewer turns (EFFICIENCY)"
        ))
        
        # 5b. Turn Effectiveness (TTEF) - NEW: Split from turn_optimization_score
        ttef_score = await self._calculate_turn_effectiveness(session_result)
        metrics.append(AgentMetricResult(
            metric_name="turn_effectiveness",
            category=MetricCategory.AGENT_INTERACTION,
            score=ttef_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["turn_effectiveness"],
            explanation="Measures phase completion and turn quality (COMPREHENSION)"
        ))
        
        # 6. Error Recovery Capability (ERC)
        erc_score = await self._calculate_error_recovery_capability(session_result)
        metrics.append(AgentMetricResult(
            metric_name="error_recovery_capability",
            category=MetricCategory.AGENT_INTERACTION,
            score=erc_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["error_recovery_capability"],
            explanation="Assesses the agent's ability to handle and recover from mistakes"
        ))
        
        # 7. Exploration vs Exploitation Balance (EEB)
        eeb_score = await self._calculate_exploration_exploitation_balance(session_result)
        metrics.append(AgentMetricResult(
            metric_name="exploration_exploitation_balance",
            category=MetricCategory.AGENT_INTERACTION,
            score=eeb_score,
            weight=self.AGENT_INTERACTION_WEIGHTS["exploration_exploitation_balance"],
            explanation="Evaluates strategic balance between information gathering and task execution"
        ))
        
        # 8. Communication Clarity Score (CCS)
        # REMOVED: communication_clarity_score (keyword-based heuristic bias)
        # ccs_score = await self._calculate_communication_clarity_score(session_result)
        # metrics.append(AgentMetricResult(
        #     metric_name="communication_clarity_score",
        #     category=MetricCategory.AGENT_INTERACTION,
        #     score=ccs_score,
        #     weight=self.AGENT_INTERACTION_WEIGHTS["communication_clarity_score"],
        #     explanation="Measures the quality of explanations and reasoning provided by the agent"
        # ))
        
        return metrics
    
    async def _calculate_software_engineering_metrics(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> List[AgentMetricResult]:
        """Calculate the 10 Software Engineering Excellence metrics"""
        
        metrics = []
        
        # Use existing metrics calculation from original LoCoBench
        # These would be adapted for agent evaluation context
        
        # Software Engineering metrics (11 total after splitting solution_elegance)
        se_metrics = [
            ("architectural_coherence", "Evaluates architectural understanding and coherence"),
            ("dependency_traversal", "Measures ability to navigate and understand dependencies"),
            ("cross_file_reasoning", "Assesses reasoning across multiple files"),
            ("system_thinking", "Evaluates holistic system understanding"),
            ("robustness", "Measures solution robustness and error handling"),
            ("comprehensiveness", "Assesses completeness of the solution"),
            ("innovation", "Evaluates creative and innovative approaches"),
            ("solution_quality", "Measures code readability and design quality (COMPREHENSION)"),
            ("solution_conciseness", "Measures code brevity and simplicity (EFFICIENCY)"),
            ("runtime_efficiency", "Evaluates time complexity and execution performance"),
            ("memory_efficiency", "Evaluates space complexity and memory usage")
        ]
        
        for metric_name, description in se_metrics:
            # Use specialized calculators for efficiency metrics
            if metric_name == "runtime_efficiency":
                score = await self._calculate_runtime_efficiency(session_result, scenario_context)
            elif metric_name == "memory_efficiency":
                score = await self._calculate_memory_efficiency(session_result, scenario_context)
            else:
                # Use actual metric algorithms from LoCoBenchMetricsCalculator
                score = await self._calculate_software_engineering_metric_with_algorithm(
                    session_result, scenario_context, metric_name
                )
            
            metrics.append(AgentMetricResult(
                metric_name=metric_name,
                category=MetricCategory.SOFTWARE_ENGINEERING,
                score=score,
                weight=self.SOFTWARE_ENGINEERING_WEIGHTS[metric_name],
                explanation=description
            ))
        
        return metrics
    
    async def _calculate_functional_correctness_metrics(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> List[AgentMetricResult]:
        """Calculate the 4 Functional Correctness metrics"""
        
        metrics = []
        
        # 1. Code Compilation Success
        compilation_score = await self._calculate_compilation_success(session_result)
        metrics.append(AgentMetricResult(
            metric_name="code_compilation_success",
            category=MetricCategory.FUNCTIONAL_CORRECTNESS,
            score=compilation_score,
            weight=0.25,
            explanation="Measures whether generated code compiles successfully"
        ))
        
        # 2. Unit Test Performance
        unit_test_score = await self._calculate_unit_test_performance(session_result)
        metrics.append(AgentMetricResult(
            metric_name="unit_test_performance",
            category=MetricCategory.FUNCTIONAL_CORRECTNESS,
            score=unit_test_score,
            weight=0.25,
            explanation="Evaluates performance on unit tests"
        ))
        
        # 3. Integration Test Performance
        integration_test_score = await self._calculate_integration_test_performance(session_result)
        metrics.append(AgentMetricResult(
            metric_name="integration_test_performance",
            category=MetricCategory.FUNCTIONAL_CORRECTNESS,
            score=integration_test_score,
            weight=0.25,
            explanation="Assesses performance on integration tests"
        ))
        
        # 4. Incremental Development Capability (IDC)
        idc_score = await self._calculate_incremental_development_capability(session_result)
        metrics.append(AgentMetricResult(
            metric_name="incremental_development_capability",
            category=MetricCategory.FUNCTIONAL_CORRECTNESS,
            score=idc_score,
            weight=0.25,
            explanation="Measures ability to develop incrementally and iteratively"
        ))
        
        return metrics
    
    async def _calculate_code_quality_metrics(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> List[AgentMetricResult]:
        """Calculate the 3 Code Quality Assessment metrics"""
        
        metrics = []
        
        # 1. Security Analysis Score
        security_score = await self._calculate_security_analysis_score(session_result)
        metrics.append(AgentMetricResult(
            metric_name="security_analysis_score",
            category=MetricCategory.CODE_QUALITY,
            score=security_score,
            weight=0.4,
            explanation="Evaluates security considerations and vulnerability avoidance"
        ))
        
        # 2. Average Issues Found (inverted)
        issues_score = await self._calculate_issues_found_score(session_result)
        metrics.append(AgentMetricResult(
            metric_name="average_issues_found",
            category=MetricCategory.CODE_QUALITY,
            score=issues_score,
            weight=0.3,
            explanation="Measures code quality by issues found (inverted - fewer issues = higher score)"
        ))
        
        # 3. Code Style Adherence
        style_score = await self._calculate_code_style_adherence(session_result)
        metrics.append(AgentMetricResult(
            metric_name="code_style_adherence",
            category=MetricCategory.CODE_QUALITY,
            score=style_score,
            weight=0.3,
            explanation="Assesses adherence to coding standards and style guidelines"
        ))
        
        return metrics
    
    async def _calculate_long_context_metrics(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> List[AgentMetricResult]:
        """Calculate the 1 Long-Context Utilization metric"""
        
        metrics = []
        
        # 1. Multi-Session Memory Retention (MMR)
        mmr_score = await self._calculate_multi_session_memory_retention(session_result)
        metrics.append(AgentMetricResult(
            metric_name="multi_session_memory_retention",
            category=MetricCategory.LONG_CONTEXT_UTILIZATION,
            score=mmr_score,
            weight=1.0,  # Now the only metric, so weight = 1.0
            explanation="Evaluates retention and utilization of information across conversation turns"
        ))
        
        return metrics
    
    # Individual metric calculation methods
    
    def _is_failed_session(self, session_result: Dict[str, Any]) -> bool:
        """Helper method to detect if a session failed before meaningful work"""
        status = session_result.get("status", "unknown")
        total_turns = session_result.get("total_turns", 0)
        error_log = session_result.get("error_log", [])
        
        # Get completed_phases - handle both int and list formats
        completed_phases_raw = session_result.get("completed_phases", 0)
        if isinstance(completed_phases_raw, list):
            completed_phases = len(completed_phases_raw)
        else:
            completed_phases = completed_phases_raw if isinstance(completed_phases_raw, int) else 0
        
        # Failed/error status with no completed work
        if status in ["failed", "error"] and completed_phases == 0:
            return True
        
        # High error rate (>50% of turns are errors)
        if total_turns > 0 and len(error_log) / total_turns > 0.5:
            return True
        
        # Session never really started
        if total_turns == 0 and status in ["failed", "error"]:
            return True
        
        return False
    
    async def _calculate_tool_usage_efficiency(self, session_result: Dict[str, Any]) -> float:
        """
        Calculate Tool Usage Efficiency score (DEPRECATED - use tool_efficiency + tool_quality)
        Kept for backwards compatibility
        """
        try:
            # Calculate both new metrics and average them
            efficiency_score = await self._calculate_tool_efficiency(session_result)
            quality_score = await self._calculate_tool_quality(session_result)
            # Equal weight for backwards compatibility
            return (efficiency_score + quality_score) / 2.0
        except Exception as e:
            self.logger.warning(f"Error calculating tool usage efficiency: {e}")
            return 0.0
    
    async def _calculate_tool_efficiency(self, session_result: Dict[str, Any]) -> float:
        """
        TES: Tool Efficiency Score (EFFICIENCY)
        Measures minimizing tool calls - rewards FEWER tool calls to complete tasks
        Formula: Penalizes excessive or redundant tool usage
        """
        # BUGFIX: Check if session failed
        if self._is_failed_session(session_result):
            self.logger.warning("Tool efficiency = 0.0 (session failed)")
            return 0.0
        
        tool_usage_log = session_result.get("tool_usage_log", [])
        completed_phases_raw = session_result.get("completed_phases", 0)
        
        if isinstance(completed_phases_raw, list):
            completed_phases = len(completed_phases_raw)
        else:
            completed_phases = completed_phases_raw if isinstance(completed_phases_raw, int) else 0
        
        # If no phases completed, efficiency is 0
        if completed_phases == 0:
            return 0.0
        
        # If no tools used but phases completed, that's very efficient!
        if not tool_usage_log:
            return 5.0
        
        total_tool_calls = len(tool_usage_log)
        
        # Calculate tool call efficiency: fewer calls per completed phase is better
        # Ideal: 2-5 tool calls per phase
        calls_per_phase = total_tool_calls / completed_phases if completed_phases > 0 else total_tool_calls
        
        if calls_per_phase <= 3:
            efficiency_ratio = 1.0  # Excellent efficiency
        elif calls_per_phase <= 5:
            efficiency_ratio = 0.9  # Good efficiency
        elif calls_per_phase <= 10:
            efficiency_ratio = 0.7  # Moderate efficiency
        elif calls_per_phase <= 15:
            efficiency_ratio = 0.5  # Low efficiency
        else:
            efficiency_ratio = 0.3  # Poor efficiency
        
        # Check for redundant calls (same function called multiple times in short succession)
        function_names = []
        for usage in tool_usage_log:
            tool_call = usage.get("tool_call", {})
            function_name = tool_call.get("function_name", "")
            function_names.append(function_name)
        
        # Penalize redundancy
        unique_calls = len(set(function_names))
        redundancy_ratio = unique_calls / total_tool_calls if total_tool_calls > 0 else 1.0
        
        # Combine: efficiency ratio (70%) + redundancy penalty (30%)
        final_score = (efficiency_ratio * 0.7 + redundancy_ratio * 0.3) * 5.0
        
        return min(final_score, 5.0)
    
    async def _calculate_tool_quality(self, session_result: Dict[str, Any]) -> float:
        """
        TQS: Tool Quality Score (COMPREHENSION)
        Measures tool mastery - rewards CORRECT tool selection and successful execution
        Formula: success_rate (40%) + diversity (30%) + appropriateness (30%)
        """
        # BUGFIX: Check if session failed
        if self._is_failed_session(session_result):
            self.logger.warning("Tool quality = 0.0 (session failed)")
            return 0.0
        
        tool_usage_log = session_result.get("tool_usage_log", [])
        
        if not tool_usage_log:
            # BUGFIX: If no tools used and session completed phases, that might be okay (score 3.0)
            # But if no tools AND failed, score 0.0
            completed_phases_raw = session_result.get("completed_phases", 0)
            if isinstance(completed_phases_raw, list):
                completed_phases = len(completed_phases_raw)
            else:
                completed_phases = completed_phases_raw if isinstance(completed_phases_raw, int) else 0
            
            status = session_result.get("status", "unknown")
            if completed_phases == 0 and status in ["failed", "error"]:
                return 0.0
            return 3.0  # Neutral score for no tool usage in successful session
        
        # Metrics for tool quality
        # Extract function names from nested structure (more specific than tool_name)
        function_names = []
        successful_tools = 0
        for usage in tool_usage_log:
            tool_call = usage.get("tool_call", {})
            function_name = tool_call.get("function_name", "")
            function_names.append(function_name)
            # Check success from tool_call or top-level
            if usage.get("success", tool_call.get("success", False)):
                successful_tools += 1
        
        tool_success_rate = successful_tools / len(tool_usage_log) if tool_usage_log else 0.0
        tool_diversity = len(set(name for name in function_names if name))  # Count unique non-empty names
        tool_appropriateness = self._assess_tool_appropriateness(tool_usage_log, session_result)
        
        # Combine metrics
        quality_score = (
            tool_success_rate * 0.4 +
            min(tool_diversity / 3.0, 1.0) * 0.3 +  # Normalize diversity
            tool_appropriateness * 0.3
        )
        
        return min(quality_score * 5.0, 5.0)
    
    async def _calculate_collaboration_intelligence(self, session_result: Dict[str, Any]) -> float:
        """Calculate Collaboration Intelligence score"""
        
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Collaboration intelligence = 0.0 (session failed)")
            return 0.0
        
        conversation_log = session_result.get("conversation_history", [])
        
        # Analyze conversation for collaboration indicators
        collaboration_indicators = [
            "please", "thank you", "help", "assist", "collaborate", "together",
            "feedback", "suggestion", "clarification", "understand", "explain"
        ]
        
        total_messages = len(conversation_log)
        if total_messages == 0:
            return 3.0  # Default score
        
        collaborative_messages = 0
        for message in conversation_log:
            content = message.get("content", "").lower()
            if any(indicator in content for indicator in collaboration_indicators):
                collaborative_messages += 1
        
        collaboration_ratio = collaborative_messages / total_messages
        
        # Also consider response appropriateness and clarity
        clarity_score = self._assess_communication_clarity(conversation_log)
        
        final_score = (collaboration_ratio * 0.6 + clarity_score * 0.4) * 5.0
        return min(final_score, 5.0)
    
    async def _calculate_context_management_score(self, session_result: Dict[str, Any]) -> float:
        """
        Calculate Context Management score (DEPRECATED - use context_efficiency + context_quality)
        Kept for backwards compatibility
        """
        try:
            # Calculate both new metrics and average them
            efficiency_score = await self._calculate_context_efficiency(session_result)
            quality_score = await self._calculate_context_quality(session_result)
            # Weighted average: 30% efficiency, 70% quality (original formula)
            return efficiency_score * 0.3 + quality_score * 0.7
        except Exception as e:
            self.logger.warning(f"Error calculating context management score: {e}")
            return 0.0
    
    async def _calculate_context_efficiency(self, session_result: Dict[str, Any]) -> float:
        """
        CES: Context Efficiency Score (EFFICIENCY)
        Measures token usage efficiency - rewards LOW token usage
        Formula: (1 - tokens_used / max_tokens) * 5.0
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Context efficiency = 0.0 (session failed)")
            return 0.0
        
        context_tokens_used = session_result.get("total_tokens_used", 0)
        max_context_tokens = session_result.get("max_context_tokens", 100000)
        
        # Efficiency of context usage (lower usage = higher score)
        context_efficiency_ratio = min(context_tokens_used / max_context_tokens, 1.0)
        
        # Invert: reward LOW usage (efficient on-demand retrieval)
        efficiency_score = (1.0 - context_efficiency_ratio) * 5.0
        
        return min(efficiency_score, 5.0)
    
    async def _calculate_context_quality(self, session_result: Dict[str, Any]) -> float:
        """
        CQS: Context Quality Score (COMPREHENSION)
        Measures context consistency and information retention
        Formula: consistency (60%) + retention (40%)
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Context quality = 0.0 (session failed)")
            return 0.0
        
        conversation_log = session_result.get("conversation_history", [])
        
        # Consistency across turns (measure how well context is maintained)
        consistency_score = self._assess_context_consistency(conversation_log)
        
        # Information retention across turns
        retention_score = self._assess_information_retention(conversation_log)
        
        # Weighted combination (adjusted from original 40/30 to 60/40 for clarity)
        quality_score = (
            consistency_score * 0.6 +
            retention_score * 0.4
        ) * 5.0
        
        return min(quality_score, 5.0)
    
    async def _calculate_adaptive_learning_rate(self, session_result: Dict[str, Any]) -> float:
        """Calculate Adaptive Learning Rate score"""
        
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Adaptive learning rate = 0.0 (session failed)")
            return 0.0
        
        conversation_log = session_result.get("conversation_history", [])
        
        # Look for evidence of learning and adaptation
        learning_indicators = [
            "i understand", "i see", "i learned", "now i know", "based on your feedback",
            "you're right", "i'll adjust", "let me correct", "i'll try a different approach"
        ]
        
        improvement_over_time = self._assess_improvement_over_time(conversation_log)
        error_correction_rate = self._assess_error_correction(conversation_log)
        feedback_incorporation = self._assess_feedback_incorporation(conversation_log, learning_indicators)
        
        final_score = (
            improvement_over_time * 0.4 +
            error_correction_rate * 0.3 +
            feedback_incorporation * 0.3
        ) * 5.0
        
        return min(final_score, 5.0)
    
    async def _calculate_turn_optimization_score(self, session_result: Dict[str, Any]) -> float:
        """
        Calculate Turn Optimization score (DEPRECATED - use turn_efficiency + turn_effectiveness)
        Kept for backwards compatibility
        """
        try:
            # Calculate both new metrics and weighted average
            efficiency_score = await self._calculate_turn_efficiency(session_result)
            effectiveness_score = await self._calculate_turn_effectiveness(session_result)
            # Weighted average: 30% efficiency, 70% effectiveness (original formula)
            return efficiency_score * 0.3 + effectiveness_score * 0.7
        except Exception as e:
            self.logger.warning(f"Error calculating turn optimization score: {e}")
            return 0.0
    
    async def _calculate_turn_efficiency(self, session_result: Dict[str, Any]) -> float:
        """
        TTES: Turn Efficiency Score (EFFICIENCY)
        Measures completing tasks in FEWER turns
        Formula: Rewards optimal turn count (10-15 turns), penalizes excessive turns
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Turn efficiency = 0.0 (session failed)")
            return 0.0
        
        total_turns = session_result.get("total_turns", 1)
        completed_phases_raw = session_result.get("completed_phases", 0)
        
        if isinstance(completed_phases_raw, list):
            completed_phases = len(completed_phases_raw)
        else:
            completed_phases = completed_phases_raw if isinstance(completed_phases_raw, int) else 0
        
        # If no phases completed, efficiency is 0
        if completed_phases == 0:
            return 0.0
        
        # Calculate turn efficiency: fewer turns is better
        # CRITICAL FIX: Relaxed optimal range to avoid penalizing thorough models
        # Previous: 10-15 optimal (too harsh, created 13.4% bias)
        # New: 10-25 optimal (allows thorough analysis without penalty)
        # Only penalize truly excessive turn counts (>30)
        if total_turns <= 10:
            efficiency_score = 1.0  # Excellent - very efficient
        elif total_turns <= 20:
            efficiency_score = 0.95  # Good - efficient and thorough
        elif total_turns <= 25:
            efficiency_score = 0.90  # Good - within optimal range
        elif total_turns <= 30:
            efficiency_score = 0.75  # Acceptable - getting verbose
        elif total_turns <= 40:
            efficiency_score = 0.55  # Moderate - verbose but tolerable
        elif total_turns <= 50:
            efficiency_score = 0.35  # Low - too many turns
        else:
            efficiency_score = 0.15  # Very poor - excessive
        
        return efficiency_score * 5.0
    
    async def _calculate_turn_effectiveness(self, session_result: Dict[str, Any]) -> float:
        """
        TTEF: Turn Effectiveness Score (COMPREHENSION)
        Measures how WELL agent completes phases and quality of each turn
        Formula: phase_completion (60%) + turn_quality (40%)
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Turn effectiveness = 0.0 (session failed)")
            return 0.0
        
        completed_phases_raw = session_result.get("completed_phases", 0)
        total_phases = session_result.get("total_phases", 1)
        
        if isinstance(completed_phases_raw, list):
            completed_phases = len(completed_phases_raw)
        else:
            completed_phases = completed_phases_raw if isinstance(completed_phases_raw, int) else 0
        
        # Phase completion rate (how many phases were completed)
        phase_completion_rate = completed_phases / total_phases if total_phases > 0 else 0.0
        
        # Quality of turns (meaningful progress per turn)
        conversation_log = session_result.get("conversation_history", [])
        turn_quality = self._assess_turn_quality(conversation_log)
        
        # Combine: phase completion (60%) + turn quality (40%)
        # Adjusted from original 40/30 to 60/40 for split
        effectiveness_score = (
            phase_completion_rate * 0.6 +
            turn_quality * 0.4
        ) * 5.0
        
        return min(effectiveness_score, 5.0)
    
    async def _calculate_error_recovery_capability(self, session_result: Dict[str, Any]) -> float:
        """Calculate Error Recovery Capability score"""
        
        # BUGFIX: Check for failed sessions - they failed to recover!
        if self._is_failed_session(session_result):
            self.logger.warning("Error recovery capability = 0.0 (session failed)")
            return 0.0
        
        tool_usage_log = session_result.get("tool_usage_log", [])
        conversation_log = session_result.get("conversation_history", [])
        
        # Count errors and recoveries
        errors = [usage for usage in tool_usage_log if not usage.get("success", True)]
        recoveries = self._identify_error_recoveries(conversation_log, errors)
        
        if len(errors) == 0:
            return 5.0  # No errors to recover from
        
        recovery_rate = len(recoveries) / len(errors)
        recovery_speed = self._assess_recovery_speed(recoveries)
        
        final_score = (recovery_rate * 0.7 + recovery_speed * 0.3) * 5.0
        return min(final_score, 5.0)
    
    async def _calculate_exploration_exploitation_balance(self, session_result: Dict[str, Any]) -> float:
        """Calculate Exploration vs Exploitation Balance score"""
        
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Exploration/exploitation balance = 0.0 (session failed)")
            return 0.0
        
        tool_usage_log = session_result.get("tool_usage_log", [])
        conversation_log = session_result.get("conversation_history", [])
        
        # Exploration: trying different approaches, asking questions, investigating
        exploration_actions = self._count_exploration_actions(tool_usage_log, conversation_log)
        
        # Exploitation: focused work, implementation, using known solutions
        exploitation_actions = self._count_exploitation_actions(tool_usage_log, conversation_log)
        
        total_actions = exploration_actions + exploitation_actions
        if total_actions == 0:
            return 3.0
        
        # Ideal balance is roughly 30% exploration, 70% exploitation
        exploration_ratio = exploration_actions / total_actions
        ideal_exploration = 0.3
        
        # Calculate balance score and clamp to [0, 1] to prevent negative scores
        balance_score = 1.0 - abs(exploration_ratio - ideal_exploration) / ideal_exploration
        balance_score = max(0.0, balance_score)  # BUGFIX: Prevent negative scores
        
        return balance_score * 5.0
    
    async def _calculate_communication_clarity_score(self, session_result: Dict[str, Any]) -> float:
        """Calculate Communication Clarity score"""
        
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Communication clarity score = 0.0 (session failed)")
            return 0.0
        
        conversation_log = session_result.get("conversation_history", [])
        
        clarity_score = self._assess_communication_clarity(conversation_log)
        reasoning_quality = self._assess_reasoning_quality(conversation_log)
        explanation_completeness = self._assess_explanation_completeness(conversation_log)
        
        final_score = (
            clarity_score * 0.4 +
            reasoning_quality * 0.4 +
            explanation_completeness * 0.2
        ) * 5.0
        
        return min(final_score, 5.0)
    
    # Helper methods for metric calculations
    
    def _assess_tool_appropriateness(self, tool_usage_log: List[Dict], session_result: Dict) -> float:
        """Assess whether tools were used appropriately for the context"""
        if not tool_usage_log:
            return 1.0
        
        # Simple heuristic: variety and success rate
        unique_tools = set(usage.get("tool_name", "") for usage in tool_usage_log)
        success_rate = sum(1 for usage in tool_usage_log if usage.get("success", False)) / len(tool_usage_log)
        
        return min(len(unique_tools) / 3.0, 1.0) * 0.5 + success_rate * 0.5
    
    def _assess_communication_clarity(self, conversation_log: List[Dict]) -> float:
        """Assess the clarity of communication"""
        if not conversation_log:
            return 0.5
        
        clarity_indicators = [
            "because", "therefore", "however", "first", "second", "finally",
            "in order to", "as a result", "for example", "specifically"
        ]
        
        total_messages = len(conversation_log)
        clear_messages = 0
        
        for message in conversation_log:
            content = message.get("content", "").lower()
            if any(indicator in content for indicator in clarity_indicators):
                clear_messages += 1
        
        return clear_messages / total_messages if total_messages > 0 else 0.5
    
    def _assess_context_consistency(self, conversation_log: List[Dict]) -> float:
        """Assess consistency of context usage across conversation"""
        if not conversation_log:
            return 0.5
        
        # CRITICAL FIX: Analyze actual context references
        context_references = 0
        total_turns = len(conversation_log)
        
        for i, turn in enumerate(conversation_log):
            if i == 0:
                continue  # Skip first turn
            
            content = str(turn.get("content", "")).lower()
            # Check for references to previous context
            if any(word in content for word in ["earlier", "previous", "before", "mentioned", "said", "as i"]):
                context_references += 1
        
        if total_turns <= 1:
            return 0.7
        
        consistency = min(context_references / (total_turns - 1), 1.0)
        return 0.5 + consistency * 0.5  # Scale to 0.5-1.0
    
    def _assess_information_retention(self, conversation_log: List[Dict]) -> float:
        """Assess how well information is retained across turns"""
        if not conversation_log or len(conversation_log) < 3:
            return 0.6
        
        # CRITICAL FIX: Check for information reuse from earlier turns
        retention_score = 0.5  # Base score
        
        for i in range(2, len(conversation_log)):
            current_turn = str(conversation_log[i].get("content", "")).lower()
            
            # Check if current turn references information from > 2 turns ago
            for j in range(max(0, i-5), i-1):
                earlier_turn = str(conversation_log[j].get("content", "")).lower()
                
                # Extract key terms from earlier turn (words > 4 chars)
                earlier_words = set(word for word in earlier_turn.split() if len(word) > 4)
                current_words = set(word for word in current_turn.split() if len(word) > 4)
                
                # Check for term overlap indicating retention
                overlap = earlier_words & current_words
                if len(overlap) > 2:
                    retention_score += 0.05
        
        return min(retention_score, 1.0)
    
    async def _calculate_software_engineering_metric_with_algorithm(
        self, 
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any],
        metric_name: str
    ) -> float:
        """
        Calculate software engineering metrics using actual LoCoBench metric algorithms.
        
        CRITICAL FIX: Replaces placeholder that returned constant values.
        Now uses actual code analysis from metric_algorithms.py.
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning(f"{metric_name} = 0.0 (session failed)")
            return 0.0
        
        # Get modified files (the code the agent wrote)
        modified_files = session_result.get("modified_files", {})
        
        if not modified_files:
            # No code written - check if this task requires code implementation
            task_category = scenario_context.get('task_category', '').lower()
            task_prompt = scenario_context.get('task_prompt', '').lower()
            scenario_id = scenario_context.get('scenario_id', '').lower()
            
            # Tasks that DON'T require code modification (analysis/comprehension only)
            analysis_tasks = [
                'code_comprehension',
                'architectural_understanding',
                'code_analysis',
                'analysis',
                'understanding'
            ]
            
            # Check if this is an analysis-only task
            is_analysis_only = (
                task_category in analysis_tasks or
                any(keyword in scenario_id for keyword in ['comprehension', 'understanding', 'analysis']) or
                any(keyword in task_prompt for keyword in ['analyze', 'explain', 'describe', 'understand', 'review'])
            )
            
            if is_analysis_only:
                # Correct behavior: analysis tasks don't require code
                self.logger.warning(f"{metric_name} = 1.0 (no code modified, analysis task)")
                return 1.0
            else:
                # CRITICAL FIX: Task requires code but agent skipped it - penalize with 0 score
                self.logger.warning(f"{metric_name} = 0.0 (no code modified, but implementation required)")
                return 0.0
        
        # Initialize metric calculator
        calculator = LoCoBenchMetricsCalculator()
        
        # Prepare scenario data for metric calculation
        scenario_for_metric = {
            'task_prompt': scenario_context.get('task_prompt', ''),
            'task_category': scenario_context.get('task_category', ''),
            'context_files': scenario_context.get('initial_context', {}).get('project_files', []),
            'difficulty': scenario_context.get('difficulty', 'medium')
        }
        
        # modified_files is now Dict[str, str] (file path -> content)
        # from agent_session.py, no conversion needed
        solution_code_dict = modified_files if isinstance(modified_files, dict) else {}
        
        try:
            # Call the appropriate metric calculation method
            if metric_name == "architectural_coherence":
                raw_score = calculator.calculate_architectural_coherence_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "dependency_traversal":
                raw_score = calculator.calculate_dependency_traversal_accuracy(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "cross_file_reasoning":
                raw_score = calculator.calculate_cross_file_reasoning_depth(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "system_thinking":
                raw_score = calculator.calculate_system_thinking_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "robustness":
                raw_score = calculator.calculate_robustness_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "comprehensiveness":
                raw_score = calculator.calculate_comprehensiveness_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "innovation":
                raw_score = calculator.calculate_innovation_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "solution_quality":
                raw_score = calculator.calculate_solution_quality_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "solution_conciseness":
                raw_score = calculator.calculate_solution_conciseness_score(
                    scenario_for_metric, solution_code_dict
                )
            elif metric_name == "solution_elegance":
                # DEPRECATED: kept for backwards compatibility
                raw_score = calculator.calculate_solution_elegance_score(
                    scenario_for_metric, solution_code_dict
                )
            else:
                # Fallback for unknown metrics
                self.logger.warning(f"Unknown SE metric: {metric_name}, using neutral score")
                return 3.0
            
            # LoCoBench metrics return 0-1, we need 0-5
            score = raw_score * 5.0
            
            self.logger.info(f"{metric_name} = {score:.2f} (calculated from actual code)")
            return min(max(score, 0.0), 5.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating {metric_name}: {e}")
            # Return neutral score on error, not 0 (failed) or 5 (perfect)
            return 3.0
    
    async def _calculate_runtime_efficiency(
        self, 
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> float:
        """
        Calculate Runtime Efficiency Score
        
        Evaluates time complexity and execution performance of the implemented solution.
        
        Components:
        - Time complexity analysis (Big O notation) - 40%
        - Algorithm appropriateness - 30%
        - Actual execution time (if available) - 30%
        
        Score: 0.0-5.0
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Runtime efficiency = 0.0 (session failed)")
            return 0.0
        
        # Get modified files from session
        modified_files = session_result.get("modified_files", {})
        if not modified_files:
            # Check if this task requires code implementation
            task_category = scenario_context.get('task_category', '').lower()
            scenario_id = scenario_context.get('scenario_id', '').lower()
            task_prompt = scenario_context.get('task_prompt', '').lower()
            
            # Analysis-only tasks don't require code
            is_analysis_only = (
                task_category in ['code_comprehension', 'architectural_understanding', 'code_analysis', 'analysis', 'understanding'] or
                any(keyword in scenario_id for keyword in ['comprehension', 'understanding', 'analysis']) or
                any(keyword in task_prompt for keyword in ['analyze', 'explain', 'describe', 'understand', 'review'])
            )
            
            if is_analysis_only:
                self.logger.warning("No modified files found for runtime efficiency analysis (analysis task, neutral score)")
                return 3.0  # Neutral score for analysis tasks
            else:
                self.logger.warning("No modified files found for runtime efficiency analysis (implementation required, 0 score)")
                return 0.0  # Penalize for skipping required implementation
        
        # Component 1: Time Complexity Analysis (40%)
        complexity_score = self._analyze_time_complexity(modified_files)
        
        # Component 2: Algorithm Appropriateness (30%)
        algorithm_score = self._analyze_algorithm_choices(modified_files)
        
        # Component 3: Execution Performance (30%)
        # Check if we have benchmark/test results
        execution_score = self._analyze_execution_performance(session_result)
        
        # Weighted combination
        final_score = (
            complexity_score * 0.40 +
            algorithm_score * 0.30 +
            execution_score * 0.30
        )
        
        return min(max(final_score, 0.0), 5.0)
    
    async def _calculate_memory_efficiency(
        self, 
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any]
    ) -> float:
        """
        Calculate Memory Efficiency Score
        
        Evaluates space complexity and memory usage of the implemented solution.
        
        Components:
        - Space complexity analysis (Big O notation) - 40%
        - Memory usage patterns - 30%
        - Resource management - 30%
        
        Score: 0.0-5.0
        """
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Memory efficiency = 0.0 (session failed)")
            return 0.0
        
        # Get modified files from session
        modified_files = session_result.get("modified_files", {})
        if not modified_files:
            # Check if this task requires code implementation
            task_category = scenario_context.get('task_category', '').lower()
            scenario_id = scenario_context.get('scenario_id', '').lower()
            task_prompt = scenario_context.get('task_prompt', '').lower()
            
            # Analysis-only tasks don't require code
            is_analysis_only = (
                task_category in ['code_comprehension', 'architectural_understanding', 'code_analysis', 'analysis', 'understanding'] or
                any(keyword in scenario_id for keyword in ['comprehension', 'understanding', 'analysis']) or
                any(keyword in task_prompt for keyword in ['analyze', 'explain', 'describe', 'understand', 'review'])
            )
            
            if is_analysis_only:
                self.logger.warning("No modified files found for memory efficiency analysis (analysis task, neutral score)")
                return 3.0  # Neutral score for analysis tasks
            else:
                self.logger.warning("No modified files found for memory efficiency analysis (implementation required, 0 score)")
                return 0.0  # Penalize for skipping required implementation
        
        # Component 1: Space Complexity Analysis (40%)
        complexity_score = self._analyze_space_complexity(modified_files)
        
        # Component 2: Memory Usage Patterns (30%)
        pattern_score = self._analyze_memory_patterns(modified_files)
        
        # Component 3: Resource Management (30%)
        resource_score = self._analyze_resource_management(modified_files)
        
        # Weighted combination
        final_score = (
            complexity_score * 0.40 +
            pattern_score * 0.30 +
            resource_score * 0.30
        )
        
        return min(max(final_score, 0.0), 5.0)
    
    # Helper methods for runtime efficiency analysis
    
    def _analyze_time_complexity(self, modified_files: Dict[str, str]) -> float:
        """Analyze time complexity patterns in code"""
        # Heuristic-based analysis of common complexity patterns
        # This is a simplified version - would integrate with AST analysis
        
        total_score = 0.0
        file_count = 0
        
        for filepath, content in modified_files.items():
            if not content or not isinstance(content, str):
                continue
            
            file_count += 1
            file_score = 5.0  # Start optimistic
            
            # Check for nested loops (O(n²) or worse)
            nested_loop_count = self._count_nested_loops(content)
            if nested_loop_count >= 3:
                file_score = 1.5  # O(n³) or worse
            elif nested_loop_count == 2:
                file_score = 2.5  # O(n²)
            elif nested_loop_count == 1:
                file_score = 4.0  # O(n)
            
            # Bonus for efficient patterns (hash tables, binary search, etc.)
            if self._uses_efficient_data_structures(content):
                file_score = min(file_score + 0.5, 5.0)
            
            # Penalty for inefficient patterns (repeated searches, etc.)
            if self._has_inefficient_patterns(content):
                file_score = max(file_score - 1.0, 1.0)
            
            total_score += file_score
        
        return total_score / file_count if file_count > 0 else 3.0
    
    def _analyze_algorithm_choices(self, modified_files: Dict[str, str]) -> float:
        """Analyze whether appropriate algorithms are used"""
        total_score = 0.0
        file_count = 0
        
        for filepath, content in modified_files.items():
            if not content or not isinstance(content, str):
                continue
            
            file_count += 1
            file_score = 3.0  # Default neutral
            
            # Look for good algorithm choices
            good_patterns = [
                'hash', 'dict', 'set', 'map',  # Fast lookups
                'binary_search', 'bisect',  # Efficient search
                'sorted', 'heap',  # Efficient sorting/priority
            ]
            
            poor_patterns = [
                'list.*in list',  # Linear search in list
                'for.*for.*in',  # Nested iteration
            ]
            
            content_lower = content.lower()
            
            good_count = sum(1 for p in good_patterns if p in content_lower)
            poor_count = sum(1 for p in poor_patterns if p in content_lower)
            
            file_score = 3.0 + (good_count * 0.5) - (poor_count * 0.5)
            total_score += max(min(file_score, 5.0), 1.0)
        
        return total_score / file_count if file_count > 0 else 3.0
    
    def _analyze_execution_performance(self, session_result: Dict[str, Any]) -> float:
        """Analyze actual execution performance if benchmarks are available"""
        # Check for test execution times
        tool_usage = session_result.get("tool_usage_log", [])
        
        # Look for test/compile execution times
        execution_times = []
        for tool_use in tool_usage:
            tool_call = tool_use.get("tool_call", {})
            function_name = tool_call.get("function_name", "").lower()
            # Check for test or compile functions
            if any(keyword in function_name for keyword in ["test", "compile"]):
                # In future, we could track actual execution times
                if tool_use.get("success", tool_call.get("success", False)):
                    execution_times.append(1.0)  # Successful execution
        
        if not execution_times:
            return 3.0  # Neutral if no execution data
        
        # For now, just check if tests ran successfully
        success_rate = sum(execution_times) / len(execution_times)
        return 3.0 + (success_rate * 2.0)  # 3.0-5.0 range
    
    # Helper methods for memory efficiency analysis
    
    def _analyze_space_complexity(self, modified_files: Dict[str, str]) -> float:
        """Analyze space complexity patterns in code"""
        total_score = 0.0
        file_count = 0
        
        for filepath, content in modified_files.items():
            if not content or not isinstance(content, str):
                continue
            
            file_count += 1
            file_score = 5.0  # Start optimistic
            
            # Check for memory-heavy patterns
            if 'readlines()' in content or 'read()' in content:
                # Reading entire files into memory
                file_score = min(file_score, 3.5)
            
            if '[' in content and 'for' in content and 'in' in content:
                # List comprehensions - check if creating large structures
                list_comp_count = content.count('[') + content.count('(')
                if list_comp_count > 5:
                    file_score = min(file_score, 4.0)
            
            # Bonus for streaming/generator patterns
            if 'yield' in content or 'generator' in content.lower():
                file_score = min(file_score + 0.5, 5.0)
            
            total_score += file_score
        
        return total_score / file_count if file_count > 0 else 3.0
    
    def _analyze_memory_patterns(self, modified_files: Dict[str, str]) -> float:
        """Analyze memory usage patterns"""
        total_score = 0.0
        file_count = 0
        
        for filepath, content in modified_files.items():
            if not content or not isinstance(content, str):
                continue
            
            file_count += 1
            file_score = 3.0  # Default neutral
            
            # Good patterns
            if 'with open' in content:
                file_score += 0.5  # Proper file handling
            
            if any(p in content for p in ['itertools', 'generator', 'yield']):
                file_score += 0.5  # Memory-efficient iteration
            
            # Poor patterns
            if '.copy()' in content:
                copy_count = content.count('.copy()')
                file_score -= min(copy_count * 0.2, 1.0)  # Unnecessary copies
            
            if 'global' in content.lower():
                file_score -= 0.3  # Global variables (memory leaks)
            
            total_score += max(min(file_score, 5.0), 1.0)
        
        return total_score / file_count if file_count > 0 else 3.0
    
    def _analyze_resource_management(self, modified_files: Dict[str, str]) -> float:
        """Analyze resource management (cleanup, etc.)"""
        total_score = 0.0
        file_count = 0
        
        for filepath, content in modified_files.items():
            if not content or not isinstance(content, str):
                continue
            
            file_count += 1
            file_score = 3.0  # Default neutral
            
            # Good patterns
            if 'with' in content:
                file_score += 1.0  # Context managers for cleanup
            
            if 'finally' in content or 'close()' in content:
                file_score += 0.5  # Explicit cleanup
            
            if 'del' in content and '[' not in content.split('del')[1].split('\n')[0]:
                file_score += 0.3  # Explicit memory release
            
            # Check for potential leaks (opening files without closing)
            open_count = content.count('open(')
            with_count = content.count('with')
            close_count = content.count('close()')
            
            if open_count > (with_count + close_count):
                file_score -= 1.0  # Potential resource leak
            
            total_score += max(min(file_score, 5.0), 1.0)
        
        return total_score / file_count if file_count > 0 else 3.0
    
    # Helper methods for pattern detection
    
    def _count_nested_loops(self, content: str) -> int:
        """Count maximum nesting level of loops"""
        # Simple heuristic: count indented for/while statements
        lines = content.split('\n')
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(('for ', 'while ')):
                indent_level = (len(line) - len(stripped)) // 4
                current_nesting = indent_level + 1
                max_nesting = max(max_nesting, current_nesting)
        
        return max_nesting
    
    def _uses_efficient_data_structures(self, content: str) -> bool:
        """Check if code uses efficient data structures"""
        efficient_patterns = [
            'dict(', '{', 'set(', 'defaultdict', 'Counter',
            'heapq', 'deque', 'bisect'
        ]
        return any(pattern in content for pattern in efficient_patterns)
    
    def _has_inefficient_patterns(self, content: str) -> bool:
        """Check for common inefficient patterns"""
        inefficient_patterns = [
            'in list',  # Linear search in list
            'list.*index(',  # .index() on list
            '.count(' if 'list' in content else None,  # .count() on list
        ]
        return any(pattern and pattern in content for pattern in inefficient_patterns)
    
    # Additional helper methods would be implemented here...
    # (For brevity, showing the structure rather than all implementations)
    
    def _calculate_category_scores(self, metric_results: List[AgentMetricResult]) -> Dict[MetricCategory, float]:
        """Calculate weighted scores for each category"""
        
        category_scores = {}
        
        for category in MetricCategory:
            category_metrics = [m for m in metric_results if m.category == category]
            
            if not category_metrics:
                category_scores[category] = 0.0
                continue
            
            # Calculate weighted average within category
            total_weighted_score = sum(m.score * m.weight for m in category_metrics)
            total_weight = sum(m.weight for m in category_metrics)
            
            category_scores[category] = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        return category_scores
    
    def _calculate_overall_score(self, category_scores: Dict[MetricCategory, float]) -> float:
        """Calculate the overall LoCoBench Agent Score (LCAS) - Legacy, for backward compatibility"""
        
        overall_score = 0.0
        
        for category, score in category_scores.items():
            weight = self.CATEGORY_WEIGHTS.get(category, 0.0)
            overall_score += score * weight
        
        return min(overall_score, 5.0)
    
    def _calculate_lcba_comprehension(self, metric_results: List[AgentMetricResult]) -> float:
        """
        Calculate LCBA-Comprehension score: Quality, depth, correctness (21 metrics)
        
        Comprehension metrics (21 total):
        - Agent Interaction (4): tool_quality, adaptive_learning_rate,
                                 error_recovery_capability, context_quality
        - Software Engineering (8): architectural_coherence, dependency_traversal, cross_file_reasoning,
                                    system_thinking, robustness, comprehensiveness, innovation, solution_quality
        - Functional Correctness (4): compilation_success, unit_test_performance,
                                       integration_test_performance, incremental_development_capability
        - Code Quality (3): security_analysis_score, issues_found_score, code_style_adherence
        - Long-Context (1): multi_session_memory_retention
        - Turn Effectiveness (1): turn_effectiveness
        
        NOTE: Removed collaboration_intelligence and communication_clarity_score due to 
        keyword-based heuristic bias that favored verbose models over quality.
        """
        
        comprehension_metrics = [
            # Agent Interaction (4) - REMOVED: collaboration_intelligence, communication_clarity_score
            'tool_quality',
            'adaptive_learning_rate',
            'error_recovery_capability',
            'context_quality',
            'turn_effectiveness',
            
            # Software Engineering (8)
            'architectural_coherence',
            'dependency_traversal',
            'cross_file_reasoning',
            'system_thinking',
            'robustness',
            'comprehensiveness',
            'innovation',
            'solution_quality',
            
            # Functional Correctness (4)
            'compilation_success',
            'unit_test_performance',
            'integration_test_performance',
            'incremental_development_capability',
            
            # Code Quality (3)
            'security_analysis_score',
            'issues_found_score',
            'code_style_adherence',
            
            # Long-Context (1)
            'multi_session_memory_retention'
        ]
        
        # Extract scores for comprehension metrics
        comp_scores = []
        for metric in metric_results:
            if metric.metric_name in comprehension_metrics:
                comp_scores.append(metric.score)
        
        # Calculate average
        if comp_scores:
            return min(sum(comp_scores) / len(comp_scores), 5.0)
        return 0.0
    
    def _calculate_lcba_efficiency(self, metric_results: List[AgentMetricResult]) -> float:
        """
        Calculate LCBA-Efficiency score: Speed, conciseness, resource optimization (8 metrics)
        
        Efficiency metrics (8 total):
        - Agent Interaction (5): tool_efficiency, context_efficiency, turn_efficiency,
                                 exploration_exploitation_balance
        - Software Engineering (3): solution_conciseness, runtime_efficiency, memory_efficiency
        """
        
        efficiency_metrics = [
            # Agent Interaction (5)
            'tool_efficiency',
            'context_efficiency',
            'turn_efficiency',
            'exploration_exploitation_balance',
            
            # Software Engineering (3)
            'solution_conciseness',
            'runtime_efficiency',
            'memory_efficiency'
        ]
        
        # Extract scores for efficiency metrics
        eff_scores = []
        for metric in metric_results:
            if metric.metric_name in efficiency_metrics:
                eff_scores.append(metric.score)
        
        # Calculate average
        if eff_scores:
            return min(sum(eff_scores) / len(eff_scores), 5.0)
        return 0.0
    
    # CRITICAL FIX: Actual implementations for helper methods
    def _assess_improvement_over_time(self, conversation_log: List[Dict]) -> float:
        if not conversation_log or len(conversation_log) < 3:
            return 0.6
        
        # Check if later turns have better quality than earlier turns
        first_half = conversation_log[:len(conversation_log)//2]
        second_half = conversation_log[len(conversation_log)//2:]
        
        # Measure quality by checking for detailed/thoughtful responses
        def quality_score(turns):
            avg_length = sum(len(str(t.get("content", ""))) for t in turns) / len(turns) if turns else 0
            return min(avg_length / 500.0, 1.0)  # Normalize by expected length
        
        first_quality = quality_score(first_half)
        second_quality = quality_score(second_half)
        
        if second_quality > first_quality:
            return 0.7 + (second_quality - first_quality) * 0.3
        return 0.6
    
    def _assess_error_correction(self, conversation_log: List[Dict]) -> float:
        if not conversation_log:
            return 0.7
        
        # Check for error correction patterns
        corrections = 0
        for turn in conversation_log:
            content = str(turn.get("content", "")).lower()
            if any(word in content for word in ["sorry", "mistake", "correct", "fix", "error", "wrong"]):
                corrections += 1
        
        # Some corrections are good (shows self-awareness), too many is bad
        if corrections == 0:
            return 0.8  # No errors to correct
        elif corrections <= 2:
            return 0.9  # Good error correction
        elif corrections <= 4:
            return 0.7  # Moderate
        else:
            return 0.5  # Too many errors
    
    def _assess_feedback_incorporation(self, conversation_log: List[Dict], indicators: List[str]) -> float:
        if not conversation_log:
            return 0.5
        
        # Check how often feedback indicators appear
        feedback_turns = sum(1 for turn in conversation_log 
                           if any(indicator in str(turn.get("content", "")).lower() 
                                 for indicator in indicators))
        
        incorporation_rate = min(feedback_turns / max(len(conversation_log), 1), 1.0)
        return 0.4 + incorporation_rate * 0.6  # Scale to 0.4-1.0
    
    def _assess_turn_quality(self, conversation_log: List[Dict]) -> float:
        """Assess the quality of individual turns in conversation
        
        Analyzes:
        - Message length (too short = low quality, too long = verbose)
        - Content richness (questions, explanations, actions)
        - Response appropriateness
        """
        if not conversation_log:
            return 0.0
        
        quality_scores = []
        
        for turn in conversation_log:
            content = turn.get("content", "")
            role = turn.get("role", "")
            
            # Skip system messages
            if role == "system":
                continue
            
            # Analyze content length (optimal: 50-500 chars)
            length = len(content)
            if length < 20:
                length_score = 0.3  # Too short
            elif length < 50:
                length_score = 0.6
            elif length <= 500:
                length_score = 1.0  # Optimal
            elif length <= 1000:
                length_score = 0.8  # A bit verbose
            else:
                length_score = 0.6  # Too verbose
            
            # Analyze content richness
            richness_score = 0.0
            content_lower = content.lower()
            
            # Check for questions (exploration)
            if any(q in content_lower for q in ["?", "what", "how", "why", "when", "where"]):
                richness_score += 0.3
            
            # Check for explanations
            if any(e in content_lower for e in ["because", "since", "therefore", "this means", "in order to"]):
                richness_score += 0.3
            
            # Check for actions/tools
            if turn.get("tool_calls") or turn.get("tool_responses"):
                richness_score += 0.4
            
            # Combine scores
            turn_quality = (length_score * 0.4) + (min(richness_score, 1.0) * 0.6)
            quality_scores.append(turn_quality)
        
        if not quality_scores:
            return 0.5
        
        return sum(quality_scores) / len(quality_scores)
    
    def _identify_error_recoveries(self, conversation_log: List[Dict], errors: List[Dict]) -> List[Dict]:
        """Identify error recovery patterns in conversation
        
        Looks for:
        - Error followed by corrective action
        - Failed tool call followed by retry with different approach
        - Error acknowledgment followed by fix
        """
        if not errors or not conversation_log:
            return []
        
        recoveries = []
        
        for error in errors:
            error_turn = error.get("turn_number", -1)
            error_type = error.get("type", "")
            
            # Look for recovery in subsequent turns (within 5 turns)
            for i, turn in enumerate(conversation_log):
                turn_idx = i + 1  # 1-indexed
                
                if turn_idx <= error_turn or turn_idx > error_turn + 5:
                    continue
                
                content = turn.get("content", "").lower()
                tool_calls = turn.get("tool_calls", [])
                
                # Check for recovery indicators
                recovery_indicators = [
                    "fix", "correct", "retry", "try again", "instead", "alternatively",
                    "let me", "sorry", "mistake", "error", "issue", "problem"
                ]
                
                has_recovery_language = any(indicator in content for indicator in recovery_indicators)
                has_tool_retry = len(tool_calls) > 0  # Agent trying again
                
                if has_recovery_language or has_tool_retry:
                    recoveries.append({
                        "error_turn": error_turn,
                        "recovery_turn": turn_idx,
                        "turns_to_recover": turn_idx - error_turn,
                        "error_type": error_type,
                        "recovery_approach": "language" if has_recovery_language else "tool_retry"
                    })
                    break  # Found recovery for this error
        
        return recoveries
    
    def _assess_recovery_speed(self, recoveries: List[Dict]) -> float:
        """Assess how quickly the agent recovers from errors
        
        Scoring:
        - 1 turn to recover: 1.0 (immediate)
        - 2 turns: 0.9 (fast)
        - 3 turns: 0.7 (moderate)
        - 4-5 turns: 0.5 (slow)
        - No recovery: 0.0
        """
        if not recoveries:
            return 0.0  # No recoveries to assess
        
        speed_scores = []
        
        for recovery in recoveries:
            turns_to_recover = recovery.get("turns_to_recover", 999)
            
            if turns_to_recover == 1:
                speed_scores.append(1.0)  # Immediate recovery
            elif turns_to_recover == 2:
                speed_scores.append(0.9)  # Fast
            elif turns_to_recover == 3:
                speed_scores.append(0.7)  # Moderate
            elif turns_to_recover <= 5:
                speed_scores.append(0.5)  # Slow but recovered
            else:
                speed_scores.append(0.2)  # Very slow
        
        return sum(speed_scores) / len(speed_scores)
    
    def _count_exploration_actions(self, tool_usage_log: List[Dict], conversation_log: List[Dict]) -> int:
        """Count exploration actions (searching, discovering, investigating)
        
        Exploration tools:
        - read_file (discovering content)
        - list_directory (exploring structure)
        - search_files (finding relevant files)
        - grep/search tools (investigating codebase)
        
        Exploration conversation patterns:
        - Questions about the codebase
        - Investigation language
        """
        if not tool_usage_log:
            return 0
        
        exploration_count = 0
        
        # Classify tool calls
        exploration_tools = ["read_file", "list_directory", "search_files", "search", "grep", "find", "explore"]
        
        for tool_use in tool_usage_log:
            # Extract function name from nested structure (more specific than tool_name)
            tool_call = tool_use.get("tool_call", {})
            function_name = tool_call.get("function_name", "").lower()
            
            # Check if it's an exploration tool
            if any(exp_tool in function_name for exp_tool in exploration_tools):
                exploration_count += 1
        
        # Also count exploration patterns in conversation
        if conversation_log:
            exploration_keywords = ["what is", "what does", "how does", "where is", "can you show", 
                                   "let me check", "let me see", "let me look", "investigating"]
            
            for turn in conversation_log:
                content = turn.get("content", "").lower()
                if any(keyword in content for keyword in exploration_keywords):
                    exploration_count += 0.5  # Partial credit for exploration language
        
        return int(exploration_count)
    
    def _count_exploitation_actions(self, tool_usage_log: List[Dict], conversation_log: List[Dict]) -> int:
        """Count exploitation actions (using knowledge, implementing, modifying)
        
        Exploitation tools:
        - write_file (implementing solution)
        - modify_file (applying changes)
        - compile (testing implementation)
        - run_tests (validating solution)
        
        Exploitation conversation patterns:
        - Implementation language
        - Solution application
        """
        if not tool_usage_log:
            return 0
        
        exploitation_count = 0
        
        # Classify tool calls
        exploitation_tools = ["write_file", "modify_file", "compile", "run_tests", "execute", 
                             "run", "test", "build", "deploy"]
        
        for tool_use in tool_usage_log:
            # Extract function name from nested structure (more specific than tool_name)
            tool_call = tool_use.get("tool_call", {})
            function_name = tool_call.get("function_name", "").lower()
            
            # Check if it's an exploitation tool
            if any(exp_tool in function_name for exp_tool in exploitation_tools):
                exploitation_count += 1
        
        # Also count exploitation patterns in conversation
        if conversation_log:
            exploitation_keywords = ["i will", "let me write", "let me modify", "let me implement",
                                    "let me fix", "let me add", "let me change", "implementing",
                                    "writing", "modifying", "fixing", "adding"]
            
            for turn in conversation_log:
                content = turn.get("content", "").lower()
                if any(keyword in content for keyword in exploitation_keywords):
                    exploitation_count += 0.5  # Partial credit for exploitation language
        
        return int(exploitation_count)
    
    def _assess_reasoning_quality(self, conversation_log: List[Dict]) -> float:
        """Assess the quality of reasoning in conversation
        
        Analyzes:
        - Logical connectors (because, therefore, since)
        - Causal reasoning (if-then, cause-effect)
        - Analytical thinking (analyze, consider, evaluate)
        - Problem-solving language (solution, approach, strategy)
        """
        if not conversation_log:
            return 0.0
        
        reasoning_indicators = {
            # Logical connectors (0.2 points each)
            "logical": ["because", "therefore", "thus", "hence", "since", "as a result", "consequently"],
            # Causal reasoning (0.25 points each)
            "causal": ["if", "then", "causes", "leads to", "results in", "due to", "this means"],
            # Analytical thinking (0.3 points each)
            "analytical": ["analyze", "consider", "evaluate", "assess", "examine", "investigate", "determine"],
            # Problem-solving (0.25 points each)
            "problem_solving": ["solution", "approach", "strategy", "method", "technique", "plan", "implement"]
        }
        
        reasoning_scores = []
        
        for turn in conversation_log:
            content = turn.get("content", "").lower()
            role = turn.get("role", "")
            
            # Only analyze assistant messages
            if role != "assistant":
                continue
            
            turn_score = 0.0
            
            # Check for logical connectors
            logical_count = sum(1 for indicator in reasoning_indicators["logical"] if indicator in content)
            turn_score += min(logical_count * 0.2, 0.4)  # Max 0.4
            
            # Check for causal reasoning
            causal_count = sum(1 for indicator in reasoning_indicators["causal"] if indicator in content)
            turn_score += min(causal_count * 0.25, 0.3)  # Max 0.3
            
            # Check for analytical thinking
            analytical_count = sum(1 for indicator in reasoning_indicators["analytical"] if indicator in content)
            turn_score += min(analytical_count * 0.3, 0.6)  # Max 0.6
            
            # Check for problem-solving language
            problem_solving_count = sum(1 for indicator in reasoning_indicators["problem_solving"] if indicator in content)
            turn_score += min(problem_solving_count * 0.25, 0.5)  # Max 0.5
            
            # Normalize to 0-1
            turn_score = min(turn_score, 1.0)
            reasoning_scores.append(turn_score)
        
        if not reasoning_scores:
            return 0.3  # No assistant messages to analyze
        
        return sum(reasoning_scores) / len(reasoning_scores)
    
    def _assess_explanation_completeness(self, conversation_log: List[Dict]) -> float:
        """Assess the completeness of explanations provided
        
        Analyzes:
        - Presence of what, why, how explanations
        - Use of examples and illustrations
        - Step-by-step breakdowns
        - Context and background information
        """
        if not conversation_log:
            return 0.0
        
        completeness_scores = []
        
        for turn in conversation_log:
            content = turn.get("content", "").lower()
            role = turn.get("role", "")
            
            # Only analyze assistant messages
            if role != "assistant":
                continue
            
            # Skip very short messages (likely just tool calls)
            if len(content) < 30:
                continue
            
            turn_score = 0.0
            
            # Check for "what" explanations (describes what is happening)
            what_indicators = ["this is", "this means", "it is", "we are", "the code", "the function"]
            if any(indicator in content for indicator in what_indicators):
                turn_score += 0.2
            
            # Check for "why" explanations (explains reasoning)
            why_indicators = ["because", "since", "reason", "purpose", "in order to", "so that"]
            if any(indicator in content for indicator in why_indicators):
                turn_score += 0.25
            
            # Check for "how" explanations (describes process)
            how_indicators = ["by", "using", "through", "first", "then", "next", "step"]
            if any(indicator in content for indicator in how_indicators):
                turn_score += 0.25
            
            # Check for examples and illustrations
            example_indicators = ["example", "for instance", "such as", "like", "e.g.", "i.e."]
            if any(indicator in content for indicator in example_indicators):
                turn_score += 0.15
            
            # Check for step-by-step breakdown
            step_indicators = ["first", "second", "third", "step 1", "step 2", "1.", "2.", "3."]
            if any(indicator in content for indicator in step_indicators):
                turn_score += 0.15
            
            # Normalize to 0-1
            turn_score = min(turn_score, 1.0)
            completeness_scores.append(turn_score)
        
        if not completeness_scores:
            return 0.3  # No substantial assistant messages
        
        return sum(completeness_scores) / len(completeness_scores)
    
    # Functional correctness metric implementations
    async def _calculate_compilation_success(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Compilation success = 0.0 (session failed)")
            return 0.0
        
        tool_usage_log = session_result.get("tool_usage_log", [])
        
        # Check for compilation attempts using function_name from nested structure
        compile_attempts = []
        for usage in tool_usage_log:
            tool_call = usage.get("tool_call", {})
            function_name = tool_call.get("function_name", "").lower()
            if "compile" in function_name:
                compile_attempts.append(usage)
        
        if not compile_attempts:
            return 3.0  # Neutral if no compilation attempted
        
        # Check success from tool responses or top-level
        successful_compiles = 0
        for attempt in compile_attempts:
            tool_call = attempt.get("tool_call", {})
            # Check success from various possible locations
            if attempt.get("success", tool_call.get("success", False)):
                successful_compiles += 1
        
        return (successful_compiles / len(compile_attempts)) * 5.0
    
    async def _calculate_unit_test_performance(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Unit test performance = 0.0 (session failed)")
            return 0.0
        
        # CRITICAL FIX: Check actual test execution from tool_usage_log
        tool_usage_log = session_result.get("tool_usage_log", [])
        
        test_attempts = []
        for tool_call in tool_usage_log:
            tool_name = tool_call.get("tool_name", "")
            function_name = tool_call.get("function_name", "")
            
            # Check for test execution
            if "test" in tool_name.lower() or "test" in function_name.lower():
                result = tool_call.get("result", {})
                if isinstance(result, dict):
                    # Check for pass rate or success indicators
                    pass_rate = result.get("pass_rate", result.get("success_rate", None))
                    if pass_rate is not None:
                        test_attempts.append(float(pass_rate))
                    elif result.get("success", False):
                        test_attempts.append(1.0)
                    elif "passed" in str(result).lower():
                        test_attempts.append(0.8)
                    else:
                        test_attempts.append(0.0)
        
        if test_attempts:
            avg_performance = sum(test_attempts) / len(test_attempts)
            score = avg_performance * 5.0
            self.logger.info(f"Unit test performance = {score:.2f} (from {len(test_attempts)} test runs)")
            return score
        
        # No tests run - check if code was modified
        modified_files = session_result.get("modified_files", {})
        if modified_files:
            self.logger.warning("Unit test performance = 2.5 (code written but no tests run)")
            return 2.5
        else:
            self.logger.warning("Unit test performance = 1.0 (no code written)")
            return 1.0
    
    async def _calculate_integration_test_performance(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Integration test performance = 0.0 (session failed)")
            return 0.0
        
        # CRITICAL FIX: Check for integration testing attempts
        tool_usage_log = session_result.get("tool_usage_log", [])
        conversation_history = session_result.get("conversation_history", [])
        
        integration_indicators = 0
        integration_successes = 0
        
        # Check tool usage for integration tests
        for tool_call in tool_usage_log:
            tool_name = tool_call.get("tool_name", "")
            function_name = tool_call.get("function_name", "")
            
            if "integration" in tool_name.lower() or "integration" in function_name.lower():
                integration_indicators += 1
                result = tool_call.get("result", {})
                if isinstance(result, dict) and result.get("success", False):
                    integration_successes += 1
        
        # Check conversation for integration discussion
        for turn in conversation_history:
            content = str(turn.get("content", "")).lower()
            if "integration" in content and ("test" in content or "testing" in content):
                integration_indicators += 0.5  # Lighter weight for discussion
        
        if integration_indicators > 0:
            success_rate = integration_successes / max(integration_indicators, 1)
            score = success_rate * 5.0
            self.logger.info(f"Integration test performance = {score:.2f} ({integration_successes}/{integration_indicators} successful)")
            return min(score, 5.0)
        
        # No integration testing - check if multiple files modified
        modified_files = session_result.get("modified_files", {})
        if len(modified_files) > 1:
            # Multiple files modified but no integration testing
            self.logger.warning("Integration test performance = 2.0 (multi-file changes but no integration testing)")
            return 2.0
        else:
            self.logger.warning("Integration test performance = 1.5 (single file or no changes)")
            return 1.5
    
    async def _calculate_incremental_development_capability(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Incremental development capability = 0.0 (session failed)")
            return 0.0
        
        # CRITICAL FIX: Assess actual incremental development patterns
        conversation_history = session_result.get("conversation_history", [])
        modified_files = session_result.get("modified_files", {})
        phases_completed = session_result.get("completed_phases", 0)
        total_phases = session_result.get("total_phases", 1)
        
        # Check for incremental patterns
        incremental_score = 0.0
        
        # 1. Phase completion (40%)
        phase_completion = (phases_completed / total_phases) * 0.4
        incremental_score += phase_completion
        
        # 2. Multiple file modifications suggesting iterative development (30%)
        if len(modified_files) > 0:
            # More files = more likely to be incremental
            file_diversity = min(len(modified_files) / 5.0, 1.0) * 0.3
            incremental_score += file_diversity
        
        # 3. Evidence of testing/validation between steps (30%)
        test_validation_turns = 0
        for i, turn in enumerate(conversation_history):
            content = str(turn.get("content", "")).lower()
            if any(word in content for word in ["test", "validate", "check", "verify", "compile"]):
                test_validation_turns += 1
        
        if len(conversation_history) > 0:
            validation_ratio = min(test_validation_turns / len(conversation_history), 1.0) * 0.3
            incremental_score += validation_ratio
        
        score = incremental_score * 5.0
        self.logger.info(f"Incremental development capability = {score:.2f} (phases: {phases_completed}/{total_phases}, files: {len(modified_files)}, validations: {test_validation_turns})")
        return min(score, 5.0)
    
    # Code quality metric implementations
    async def _calculate_security_analysis_score(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Security analysis score = 0.0 (session failed)")
            return 0.0
        
        # CRITICAL FIX: Analyze actual code for security issues
        modified_files = session_result.get("modified_files", {})
        
        if not modified_files:
            return 2.0
        
        security_score = 3.0  # Start from neutral
        
        for filepath, content in modified_files.items():
            if not isinstance(content, str):
                continue
            content_lower = content.lower()
            
            # Security vulnerabilities (penalties)
            if 'eval(' in content or 'exec(' in content:
                security_score -= 0.5
            if any(word in content_lower for word in ['password', 'secret']) and '=' in content:
                security_score -= 0.3
            
            # Good practices (bonuses)
            if 'try:' in content and 'except' in content:
                security_score += 0.3
            if any(word in content_lower for word in ['validate', 'sanitize']):
                security_score += 0.2
        
        return min(max(security_score, 0.0), 5.0)
    
    async def _calculate_issues_found_score(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Issues found score = 0.0 (session failed)")
            return 0.0
        
        # CRITICAL FIX: Count actual issues in code
        modified_files = session_result.get("modified_files", {})
        error_log = session_result.get("error_log", [])
        
        if not modified_files:
            return 2.0
        
        issue_count = len(error_log)
        
        # Analyze code for common issues
        for filepath, content in modified_files.items():
            if not isinstance(content, str):
                continue
            
            # Simple heuristics for code issues
            if '# TODO' in content or '# FIXME' in content:
                issue_count += content.count('# TODO') + content.count('# FIXME')
            if 'pass  # placeholder' in content.lower():
                issue_count += 1
        
        # Inverted score: fewer issues = higher score
        if issue_count == 0:
            return 5.0
        elif issue_count <= 2:
            return 4.0
        elif issue_count <= 5:
            return 3.0
        elif issue_count <= 10:
            return 2.0
        else:
            return 1.0
    
    async def _calculate_code_style_adherence(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Code style adherence = 0.0 (session failed)")
            return 0.0
        
        # CRITICAL FIX: Analyze actual code style
        modified_files = session_result.get("modified_files", {})
        
        if not modified_files:
            return 2.0
        
        style_score = 3.0  # Start from neutral
        
        for filepath, content in modified_files.items():
            if not isinstance(content, str):
                continue
            
            lines = content.split('\n')
            
            # Check for good style practices
            has_docstrings = '"""' in content or "'''" in content
            has_type_hints = ':' in content and '->' in content
            reasonable_line_length = all(len(line) <= 120 for line in lines if line.strip())
            has_blank_lines = '\n\n' in content
            
            if has_docstrings:
                style_score += 0.3
            if has_type_hints:
                style_score += 0.2
            if reasonable_line_length:
                style_score += 0.3
            if has_blank_lines:
                style_score += 0.2
        
        return min(max(style_score, 0.0), 5.0)
    
    # Long-context metric implementations
    async def _calculate_multi_session_memory_retention(self, session_result: Dict[str, Any]) -> float:
        # BUGFIX: Check for failed sessions
        if self._is_failed_session(session_result):
            self.logger.warning("Multi-session memory retention = 0.0 (session failed)")
            return 0.0
        
        # Assess memory retention across conversation turns
        total_turns = session_result.get("total_turns", 1)
        
        # Longer sessions with maintained context get higher scores
        if total_turns > 20:
            return 4.5
        elif total_turns > 10:
            return 4.0
        else:
            return 3.5
