"""
Session Evaluator for LoCoBench-Agent

This module provides specialized evaluation capabilities for multi-turn agent sessions,
focusing on conversation flow, turn efficiency, and session-level metrics.
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .agent_metrics import MetricCategory, AgentMetricResult
from ..core.agent_session import AgentSession, SessionStatus

logger = logging.getLogger(__name__)


class SessionQuality(Enum):
    """Overall session quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    FAILED = "failed"


class ConversationPattern(Enum):
    """Types of conversation patterns observed"""
    LINEAR_PROGRESSION = "linear_progression"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    EXPLORATORY_DISCOVERY = "exploratory_discovery"
    PROBLEM_SOLVING_SPIRAL = "problem_solving_spiral"
    ERROR_RECOVERY_CYCLE = "error_recovery_cycle"
    COLLABORATIVE_DIALOG = "collaborative_dialog"


@dataclass
class TurnAnalysis:
    """Analysis of a single conversation turn"""
    
    turn_number: int
    message_content: str
    message_length: int
    
    # Tool usage in this turn
    tools_used: List[str] = field(default_factory=list)
    tool_calls_count: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    
    # Turn quality metrics
    clarity_score: float = 0.0
    relevance_score: float = 0.0
    progress_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Context information
    phase_name: Optional[str] = None
    addresses_previous_feedback: bool = False
    introduces_new_concepts: bool = False
    
    # Timing
    response_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "message_length": self.message_length,
            "tools_used": self.tools_used,
            "tool_calls_count": self.tool_calls_count,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "clarity_score": self.clarity_score,
            "relevance_score": self.relevance_score,
            "progress_score": self.progress_score,
            "efficiency_score": self.efficiency_score,
            "phase_name": self.phase_name,
            "addresses_previous_feedback": self.addresses_previous_feedback,
            "introduces_new_concepts": self.introduces_new_concepts,
            "response_time_seconds": self.response_time_seconds
        }


@dataclass
class SessionEvaluationResult:
    """Complete evaluation result for a multi-turn session"""
    
    session_id: str
    agent_name: str
    evaluation_timestamp: datetime
    
    # Session overview
    total_turns: int
    session_duration_seconds: float
    session_status: SessionStatus
    session_quality: SessionQuality
    
    # Conversation analysis
    conversation_pattern: ConversationPattern
    turn_analyses: List[TurnAnalysis] = field(default_factory=list)
    
    # Session-level metrics
    overall_coherence_score: float = 0.0
    goal_achievement_score: float = 0.0
    conversation_efficiency_score: float = 0.0
    context_utilization_score: float = 0.0
    
    # Tool usage analysis
    total_tool_calls: int = 0
    unique_tools_used: int = 0
    tool_success_rate: float = 0.0
    tool_efficiency_score: float = 0.0
    
    # Conversation flow metrics
    average_turn_length: float = 0.0
    turn_length_consistency: float = 0.0
    response_time_consistency: float = 0.0
    
    # Learning and adaptation
    learning_evidence_score: float = 0.0
    error_correction_score: float = 0.0
    feedback_incorporation_score: float = 0.0
    
    # Phase-specific analysis
    phase_transition_quality: float = 0.0
    phase_completion_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "total_turns": self.total_turns,
            "session_duration_seconds": self.session_duration_seconds,
            "session_status": self.session_status.value,
            "session_quality": self.session_quality.value,
            "conversation_pattern": self.conversation_pattern.value,
            "turn_analyses": [turn.to_dict() for turn in self.turn_analyses],
            "overall_coherence_score": self.overall_coherence_score,
            "goal_achievement_score": self.goal_achievement_score,
            "conversation_efficiency_score": self.conversation_efficiency_score,
            "context_utilization_score": self.context_utilization_score,
            "total_tool_calls": self.total_tool_calls,
            "unique_tools_used": self.unique_tools_used,
            "tool_success_rate": self.tool_success_rate,
            "tool_efficiency_score": self.tool_efficiency_score,
            "average_turn_length": self.average_turn_length,
            "turn_length_consistency": self.turn_length_consistency,
            "response_time_consistency": self.response_time_consistency,
            "learning_evidence_score": self.learning_evidence_score,
            "error_correction_score": self.error_correction_score,
            "feedback_incorporation_score": self.feedback_incorporation_score,
            "phase_transition_quality": self.phase_transition_quality,
            "phase_completion_rate": self.phase_completion_rate
        }


class SessionEvaluator:
    """
    Evaluator for multi-turn agent sessions
    
    This evaluator focuses on session-level metrics including conversation flow,
    turn efficiency, learning patterns, and overall session quality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Evaluation thresholds
        self.quality_thresholds = {
            SessionQuality.EXCELLENT: 4.5,
            SessionQuality.GOOD: 3.5,
            SessionQuality.AVERAGE: 2.5,
            SessionQuality.POOR: 1.5
        }
        
        logger.info("SessionEvaluator initialized")
    
    async def evaluate_session(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any] = None
    ) -> SessionEvaluationResult:
        """
        Evaluate a complete multi-turn session
        
        Args:
            session_result: Results from AgentSession execution
            scenario_context: Context about the scenario/task
            
        Returns:
            Complete session evaluation result
        """
        
        logger.info(f"Evaluating session: {session_result.get('session_id', 'unknown')}")
        
        # Initialize result
        result = SessionEvaluationResult(
            session_id=session_result.get("session_id", "unknown"),
            agent_name=session_result.get("agent_name", "unknown"),
            evaluation_timestamp=datetime.now(),
            total_turns=len(session_result.get("conversation_history", [])),
            session_duration_seconds=session_result.get("duration_seconds", 0),
            session_status=session_result.get("status", "unknown")
        )
        
        # Analyze individual turns
        result.turn_analyses = await self._analyze_turns(session_result)
        
        # Analyze conversation pattern
        result.conversation_pattern = self._identify_conversation_pattern(result.turn_analyses)
        
        # Calculate session-level metrics
        result.overall_coherence_score = self._calculate_coherence_score(result.turn_analyses)
        result.goal_achievement_score = self._calculate_goal_achievement(session_result, scenario_context)
        result.conversation_efficiency_score = self._calculate_conversation_efficiency(result.turn_analyses)
        result.context_utilization_score = self._calculate_context_utilization(session_result)
        
        # Tool usage analysis
        result.total_tool_calls = sum(turn.tool_calls_count for turn in result.turn_analyses)
        result.unique_tools_used = len(set().union(*[turn.tools_used for turn in result.turn_analyses]))
        result.tool_success_rate = self._calculate_tool_success_rate(result.turn_analyses)
        result.tool_efficiency_score = self._calculate_tool_efficiency(result.turn_analyses)
        
        # Conversation flow metrics
        turn_lengths = [turn.message_length for turn in result.turn_analyses]
        response_times = [turn.response_time_seconds for turn in result.turn_analyses if turn.response_time_seconds > 0]
        
        result.average_turn_length = statistics.mean(turn_lengths) if turn_lengths else 0
        result.turn_length_consistency = 1.0 - (statistics.stdev(turn_lengths) / statistics.mean(turn_lengths)) if len(turn_lengths) > 1 and statistics.mean(turn_lengths) > 0 else 1.0
        result.response_time_consistency = 1.0 - (statistics.stdev(response_times) / statistics.mean(response_times)) if len(response_times) > 1 and statistics.mean(response_times) > 0 else 1.0
        
        # Learning and adaptation metrics
        result.learning_evidence_score = self._calculate_learning_evidence(result.turn_analyses)
        result.error_correction_score = self._calculate_error_correction(result.turn_analyses)
        result.feedback_incorporation_score = self._calculate_feedback_incorporation(result.turn_analyses)
        
        # Phase analysis
        result.phase_transition_quality = self._calculate_phase_transition_quality(session_result)
        result.phase_completion_rate = self._calculate_phase_completion_rate(session_result)
        
        # Determine overall session quality
        result.session_quality = self._determine_session_quality(result)
        
        logger.info(f"Session evaluation completed. Quality: {result.session_quality.value}")
        
        return result
    
    async def _analyze_turns(self, session_result: Dict[str, Any]) -> List[TurnAnalysis]:
        """Analyze individual conversation turns"""
        
        turn_analyses = []
        conversation_history = session_result.get("conversation_history", [])
        tool_usage_log = session_result.get("tool_usage_log", [])
        error_log = session_result.get("error_log", [])
        
        for i, turn in enumerate(conversation_history):
            if turn.get("role") != "assistant":
                continue
            
            turn_analysis = TurnAnalysis(
                turn_number=i + 1,
                message_content=turn.get("content", ""),
                message_length=len(turn.get("content", ""))
            )
            
            # BUGFIX: Detect error turns (e.g., context_length_exceeded, API errors)
            is_error_turn = self._is_error_turn(turn, error_log, i + 1)
            
            # Analyze tool usage in this turn
            turn_tools = [
                tool for tool in tool_usage_log 
                if tool.get("turn_number") == i + 1
            ]
            
            turn_analysis.tools_used = list(set(tool.get("tool_name") for tool in turn_tools))
            turn_analysis.tool_calls_count = len(turn_tools)
            turn_analysis.successful_tool_calls = len([t for t in turn_tools if t.get("success", False)])
            turn_analysis.failed_tool_calls = turn_analysis.tool_calls_count - turn_analysis.successful_tool_calls
            
            # BUGFIX: If this is an error turn, set quality scores to 0
            if is_error_turn:
                turn_analysis.clarity_score = 0.0
                turn_analysis.relevance_score = 0.0
                turn_analysis.progress_score = 0.0
                turn_analysis.efficiency_score = 0.0
            else:
                # Calculate turn quality scores
                turn_analysis.clarity_score = self._calculate_turn_clarity(turn_analysis)
                turn_analysis.relevance_score = self._calculate_turn_relevance(turn_analysis, session_result)
                turn_analysis.progress_score = self._calculate_turn_progress(turn_analysis, i, conversation_history)
                turn_analysis.efficiency_score = self._calculate_turn_efficiency(turn_analysis)
            
            # Context analysis
            turn_analysis.phase_name = self._get_turn_phase(turn, session_result)
            turn_analysis.addresses_previous_feedback = self._addresses_feedback(turn, conversation_history[:i])
            turn_analysis.introduces_new_concepts = self._introduces_concepts(turn, conversation_history[:i])
            
            # Timing
            turn_analysis.response_time_seconds = turn.get("response_time", 0)
            
            turn_analyses.append(turn_analysis)
        
        return turn_analyses
    
    def _is_error_turn(self, turn: Dict[str, Any], error_log: List[Dict[str, Any]], turn_number: int) -> bool:
        """Detect if a turn is an error turn (API failure, context overflow, etc.)"""
        
        # Check if there's an error logged for this turn
        turn_errors = [e for e in error_log if e.get("turn") == turn_number]
        if turn_errors:
            return True
        
        # Check for error messages in the content
        content = turn.get("content", "").lower()
        error_indicators = [
            "i encountered an error",
            "error processing",
            "context_length_exceeded",
            "context length exceeded",
            "maximum context length",
            "rate limit exceeded",
            "api error",
            "failed to process",
            "unable to complete"
        ]
        
        for indicator in error_indicators:
            if indicator in content:
                return True
        
        # Check metadata for error flags
        metadata = turn.get("metadata", {})
        if metadata.get("error") or metadata.get("is_error"):
            return True
        
        return False
    
    def _identify_conversation_pattern(self, turn_analyses: List[TurnAnalysis]) -> ConversationPattern:
        """Identify the overall conversation pattern"""
        
        if not turn_analyses:
            return ConversationPattern.LINEAR_PROGRESSION
        
        # Analyze progress scores over time
        progress_scores = [turn.progress_score for turn in turn_analyses]
        
        # Calculate trend
        if len(progress_scores) < 3:
            return ConversationPattern.LINEAR_PROGRESSION
        
        # Check for different patterns
        increasing_trend = all(progress_scores[i] <= progress_scores[i+1] for i in range(len(progress_scores)-1))
        has_cycles = self._has_iterative_cycles(progress_scores)
        has_exploration = self._has_exploration_pattern(turn_analyses)
        has_error_recovery = any(turn.failed_tool_calls > 0 for turn in turn_analyses)
        
        if has_error_recovery and self._has_recovery_pattern(turn_analyses):
            return ConversationPattern.ERROR_RECOVERY_CYCLE
        elif has_exploration:
            return ConversationPattern.EXPLORATORY_DISCOVERY
        elif has_cycles:
            return ConversationPattern.ITERATIVE_REFINEMENT
        elif increasing_trend:
            return ConversationPattern.LINEAR_PROGRESSION
        else:
            return ConversationPattern.PROBLEM_SOLVING_SPIRAL
    
    def _calculate_coherence_score(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate overall conversation coherence"""
        
        if not turn_analyses:
            return 0.0
        
        # Coherence based on relevance consistency
        relevance_scores = [turn.relevance_score for turn in turn_analyses]
        
        # High coherence = consistent high relevance + smooth transitions
        avg_relevance = statistics.mean(relevance_scores)
        relevance_consistency = 1.0 - (statistics.stdev(relevance_scores) / avg_relevance) if avg_relevance > 0 and len(relevance_scores) > 1 else 1.0
        
        # Penalize for abrupt topic changes
        topic_continuity = self._calculate_topic_continuity(turn_analyses)
        
        coherence_score = (avg_relevance * 0.5 + relevance_consistency * 0.3 + topic_continuity * 0.2)
        
        return min(5.0, max(0.0, coherence_score))
    
    def _calculate_goal_achievement(self, session_result: Dict[str, Any], scenario_context: Dict[str, Any] = None) -> float:
        """Calculate how well the session achieved its goals"""
        
        # Base score from session status
        status = session_result.get("status", "unknown")
        base_score = {
            "completed": 4.0,
            "partial_completion": 3.0,
            "in_progress": 2.0,
            "failed": 0.5,  # BUGFIX: Failed sessions should score very low
            "error": 0.0    # BUGFIX: Error sessions should score 0
        }.get(status, 2.0)
        
        # BUGFIX: Check if session failed before meaningful work began
        error_log = session_result.get("error_log", [])
        total_turns = session_result.get("total_turns", 0)
        
        # If session has many errors relative to turns, it's a failed session
        if total_turns > 0 and len(error_log) / total_turns > 0.5:
            logger.warning(f"Session has {len(error_log)} errors in {total_turns} turns - marking as failed")
            return 0.0
        
        # If session failed/errored and completed no phases, return minimal score
        phases_completed = len(session_result.get("completed_phases", []))
        if status in ["failed", "error"] and phases_completed == 0:
            logger.warning(f"Session status={status} with 0 phases completed - minimal score")
            return 0.0
        
        # Adjust based on scenario success criteria if available
        if scenario_context and "success_criteria" in scenario_context:
            success_criteria = scenario_context["success_criteria"]
            met_criteria = session_result.get("success_criteria_met", [])
            
            if success_criteria:
                criteria_score = len(met_criteria) / len(success_criteria)
                base_score = (base_score + criteria_score * 5.0) / 2.0
        
        # Adjust based on phase completion
        total_phases = len(session_result.get("conversation_phases", []))
        
        if total_phases > 0:
            phase_score = (phases_completed / total_phases) * 5.0
            base_score = (base_score + phase_score) / 2.0
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_conversation_efficiency(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate conversation efficiency"""
        
        if not turn_analyses:
            return 0.0
        
        # Efficiency based on progress per turn and tool usage effectiveness
        avg_progress = statistics.mean([turn.progress_score for turn in turn_analyses])
        avg_efficiency = statistics.mean([turn.efficiency_score for turn in turn_analyses])
        
        # Penalize for excessive turns without progress
        progress_per_turn = avg_progress / len(turn_analyses) if turn_analyses else 0
        
        efficiency_score = (avg_efficiency * 0.4 + avg_progress * 0.4 + progress_per_turn * 0.2)
        
        return min(5.0, max(0.0, efficiency_score))
    
    def _calculate_context_utilization(self, session_result: Dict[str, Any]) -> float:
        """Calculate how well the agent utilized available context"""
        
        # Base on tool usage and information access patterns
        tool_usage_log = session_result.get("tool_usage_log", [])
        
        if not tool_usage_log:
            return 2.0  # Neutral score if no tools used
        
        # Analyze diversity of tools used
        unique_tools = len(set(tool.get("tool_name") for tool in tool_usage_log))
        total_available = len(session_result.get("available_tools", []))
        
        tool_diversity = unique_tools / total_available if total_available > 0 else 0
        
        # Analyze information gathering patterns
        info_tools = ["file_system", "search", "ide_simulator"]
        info_tool_usage = len([t for t in tool_usage_log if t.get("tool_name") in info_tools])
        
        info_gathering_score = min(1.0, info_tool_usage / max(1, len(tool_usage_log) * 0.3))
        
        context_score = (tool_diversity * 3.0 + info_gathering_score * 2.0)
        
        return min(5.0, max(0.0, context_score))
    
    def _calculate_tool_success_rate(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate overall tool usage success rate"""
        
        total_calls = sum(turn.tool_calls_count for turn in turn_analyses)
        successful_calls = sum(turn.successful_tool_calls for turn in turn_analyses)
        
        return successful_calls / total_calls if total_calls > 0 else 1.0
    
    def _calculate_tool_efficiency(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate tool usage efficiency"""
        
        if not turn_analyses:
            return 0.0
        
        # Efficiency = successful tool calls / total turns with meaningful progress
        meaningful_turns = [turn for turn in turn_analyses if turn.progress_score > 2.0]
        tool_turns = [turn for turn in turn_analyses if turn.tool_calls_count > 0]
        
        if not tool_turns:
            return 3.0  # Neutral score if no tools used
        
        avg_success_rate = statistics.mean([
            turn.successful_tool_calls / turn.tool_calls_count 
            for turn in tool_turns 
            if turn.tool_calls_count > 0
        ])
        
        efficiency_ratio = len(meaningful_turns) / len(turn_analyses) if turn_analyses else 0
        
        efficiency_score = (avg_success_rate * 3.0 + efficiency_ratio * 2.0)
        
        return min(5.0, max(0.0, efficiency_score))
    
    def _calculate_learning_evidence(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate evidence of learning/adaptation during the session"""
        
        if len(turn_analyses) < 3:
            return 0.0
        
        # Look for improvement patterns
        early_turns = turn_analyses[:len(turn_analyses)//3]
        late_turns = turn_analyses[-len(turn_analyses)//3:]
        
        early_avg_efficiency = statistics.mean([turn.efficiency_score for turn in early_turns])
        late_avg_efficiency = statistics.mean([turn.efficiency_score for turn in late_turns])
        
        early_avg_success = statistics.mean([
            turn.successful_tool_calls / max(1, turn.tool_calls_count) for turn in early_turns
        ])
        late_avg_success = statistics.mean([
            turn.successful_tool_calls / max(1, turn.tool_calls_count) for turn in late_turns
        ])
        
        efficiency_improvement = late_avg_efficiency - early_avg_efficiency
        success_improvement = late_avg_success - early_avg_success
        
        # Evidence of addressing feedback
        feedback_responses = sum(1 for turn in turn_analyses if turn.addresses_previous_feedback)
        feedback_score = feedback_responses / len(turn_analyses)
        
        learning_score = (efficiency_improvement + success_improvement + feedback_score * 2.0)
        
        return min(5.0, max(0.0, learning_score))
    
    def _calculate_error_correction(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate error correction capability"""
        
        error_turns = [turn for turn in turn_analyses if turn.failed_tool_calls > 0]
        
        if not error_turns:
            return 5.0  # Perfect score if no errors
        
        # Look for recovery after errors
        recovery_count = 0
        
        for i, turn in enumerate(turn_analyses):
            if turn.failed_tool_calls > 0 and i < len(turn_analyses) - 1:
                next_turn = turn_analyses[i + 1]
                # Check if next turn shows improvement
                if next_turn.progress_score > turn.progress_score:
                    recovery_count += 1
        
        recovery_rate = recovery_count / len(error_turns) if error_turns else 1.0
        
        return recovery_rate * 5.0
    
    def _calculate_feedback_incorporation(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate how well the agent incorporates feedback"""
        
        feedback_turns = [turn for turn in turn_analyses if turn.addresses_previous_feedback]
        
        if not feedback_turns:
            return 3.0  # Neutral score if no feedback detected
        
        feedback_rate = len(feedback_turns) / len(turn_analyses)
        
        # Quality of feedback incorporation
        avg_progress_after_feedback = statistics.mean([
            turn.progress_score for turn in feedback_turns
        ])
        
        incorporation_score = (feedback_rate * 2.5 + avg_progress_after_feedback / 5.0 * 2.5)
        
        return min(5.0, max(0.0, incorporation_score))
    
    def _calculate_phase_transition_quality(self, session_result: Dict[str, Any]) -> float:
        """Calculate quality of phase transitions"""
        
        phase_history = session_result.get("phase_history", [])
        
        if len(phase_history) < 2:
            return 5.0  # Perfect if only one phase
        
        # Analyze transition smoothness
        smooth_transitions = 0
        
        for i in range(len(phase_history) - 1):
            current_phase = phase_history[i]
            next_phase = phase_history[i + 1]
            
            # Check if phase was completed before transition
            if current_phase.get("completion_status") == "completed":
                smooth_transitions += 1
        
        transition_quality = smooth_transitions / (len(phase_history) - 1)
        
        return transition_quality * 5.0
    
    def _calculate_phase_completion_rate(self, session_result: Dict[str, Any]) -> float:
        """Calculate phase completion rate"""
        
        total_phases = len(session_result.get("conversation_phases", []))
        completed_phases = len(session_result.get("completed_phases", []))
        
        if total_phases == 0:
            return 5.0
        
        completion_rate = completed_phases / total_phases
        
        return completion_rate * 5.0
    
    def _determine_session_quality(self, result: SessionEvaluationResult) -> SessionQuality:
        """Determine overall session quality"""
        
        # BUGFIX: Check for failed sessions first
        if result.session_status in [SessionStatus.FAILED, SessionStatus.ERROR]:
            # If session failed and achieved nothing, mark as FAILED
            if result.goal_achievement_score < 0.5:
                logger.warning(f"Session marked as FAILED due to status={result.session_status.value} and low goal achievement")
                return SessionQuality.FAILED
        
        # BUGFIX: Check if session had too many errors to be considered successful
        error_rate = 0.0
        if result.total_turns > 0 and hasattr(result, 'error_count'):
            error_rate = result.error_count / result.total_turns
        
        if error_rate > 0.5:
            logger.warning(f"Session marked as FAILED due to high error rate: {error_rate:.1%}")
            return SessionQuality.FAILED
        
        # Composite score from key metrics
        composite_score = (
            result.overall_coherence_score * 0.25 +
            result.goal_achievement_score * 0.25 +
            result.conversation_efficiency_score * 0.20 +
            result.context_utilization_score * 0.15 +
            result.learning_evidence_score * 0.15
        )
        
        # Map to quality levels
        for quality, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if composite_score >= threshold:
                return quality
        
        return SessionQuality.FAILED
    
    # Helper methods for pattern analysis
    
    def _calculate_turn_clarity(self, turn_analysis: TurnAnalysis) -> float:
        """Calculate clarity score for a turn"""
        
        # Simple heuristics based on message structure
        content = turn_analysis.message_content
        
        # Penalize very short or very long messages
        length_score = 1.0
        if len(content) < 10:
            length_score = 0.3
        elif len(content) > 2000:
            length_score = 0.7
        
        # Reward structured content
        structure_indicators = ['. ', '\n', '1.', '2.', '3.', '- ', '* ']
        structure_score = min(1.0, sum(content.count(indicator) for indicator in structure_indicators) / 10.0)
        
        clarity_score = (length_score + structure_score) * 2.5
        
        return min(5.0, max(0.0, clarity_score))
    
    def _calculate_turn_relevance(self, turn_analysis: TurnAnalysis, session_result: Dict[str, Any]) -> float:
        """Calculate relevance score for a turn"""
        
        # Base relevance on tool usage appropriateness and content
        base_score = 3.0
        
        # Reward tool usage
        if turn_analysis.tool_calls_count > 0:
            base_score += 1.0
            
            # Reward successful tool usage
            if turn_analysis.tool_success_rate > 0.8:
                base_score += 0.5
        
        # Penalize excessive failed tool calls
        if turn_analysis.failed_tool_calls > 2:
            base_score -= 1.0
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_turn_progress(self, turn_analysis: TurnAnalysis, turn_index: int, conversation_history: List[Dict]) -> float:
        """Calculate progress score for a turn"""
        
        # Base progress on successful actions and meaningful content
        base_score = 2.0
        
        # Reward successful tool usage
        if turn_analysis.successful_tool_calls > 0:
            base_score += min(2.0, turn_analysis.successful_tool_calls * 0.5)
        
        # Reward substantive content
        if turn_analysis.message_length > 100:
            base_score += 0.5
        
        # Reward introducing new concepts (exploration)
        if turn_analysis.introduces_new_concepts:
            base_score += 0.5
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_turn_efficiency(self, turn_analysis: TurnAnalysis) -> float:
        """Calculate efficiency score for a turn"""
        
        if turn_analysis.tool_calls_count == 0:
            # Base efficiency for non-tool turns
            return 3.0
        
        # Efficiency = successful calls / total calls, adjusted for response time
        success_rate = turn_analysis.successful_tool_calls / turn_analysis.tool_calls_count
        
        # Penalize excessive tool calls in one turn
        call_efficiency = 1.0
        if turn_analysis.tool_calls_count > 5:
            call_efficiency = 0.7
        elif turn_analysis.tool_calls_count > 3:
            call_efficiency = 0.9
        
        efficiency_score = success_rate * call_efficiency * 5.0
        
        return min(5.0, max(0.0, efficiency_score))
    
    def _get_turn_phase(self, turn: Dict[str, Any], session_result: Dict[str, Any]) -> Optional[str]:
        """Get the phase name for a turn"""
        
        phase_history = session_result.get("phase_history", [])
        
        # Simple heuristic to match turn to phase
        # In a real implementation, this would use turn timestamps
        if phase_history:
            return phase_history[0].get("phase_name")
        
        return None
    
    def _addresses_feedback(self, turn: Dict[str, Any], previous_turns: List[Dict]) -> bool:
        """Check if turn addresses previous feedback"""
        
        # Simple heuristic looking for feedback-related keywords
        content = turn.get("content", "").lower()
        feedback_keywords = ["corrected", "fixed", "updated", "revised", "changed", "addressed"]
        
        return any(keyword in content for keyword in feedback_keywords)
    
    def _introduces_concepts(self, turn: Dict[str, Any], previous_turns: List[Dict]) -> bool:
        """Check if turn introduces new concepts"""
        
        # Simple heuristic based on content novelty
        current_content = turn.get("content", "").lower()
        previous_content = " ".join([t.get("content", "") for t in previous_turns]).lower()
        
        # Look for new technical terms or concepts
        current_words = set(current_content.split())
        previous_words = set(previous_content.split())
        
        new_words = current_words - previous_words
        
        # Filter for potentially meaningful new words (length > 4)
        meaningful_new = [word for word in new_words if len(word) > 4]
        
        return len(meaningful_new) > 2
    
    def _has_iterative_cycles(self, progress_scores: List[float]) -> bool:
        """Check if progress scores show iterative refinement cycles"""
        
        # Look for patterns of progress followed by slight regression followed by higher progress
        cycles = 0
        
        for i in range(2, len(progress_scores)):
            if (progress_scores[i-2] < progress_scores[i-1] > progress_scores[i] and 
                progress_scores[i] < progress_scores[i-1]):
                cycles += 1
        
        return cycles >= 2
    
    def _has_exploration_pattern(self, turn_analyses: List[TurnAnalysis]) -> bool:
        """Check for exploratory discovery pattern"""
        
        # High tool diversity and concept introduction
        tool_diversity = len(set().union(*[turn.tools_used for turn in turn_analyses]))
        concept_introductions = sum(1 for turn in turn_analyses if turn.introduces_new_concepts)
        
        return tool_diversity > 3 and concept_introductions > len(turn_analyses) * 0.3
    
    def _has_recovery_pattern(self, turn_analyses: List[TurnAnalysis]) -> bool:
        """Check for error recovery pattern"""
        
        # Look for improvement after errors
        recovery_instances = 0
        
        for i in range(len(turn_analyses) - 1):
            if (turn_analyses[i].failed_tool_calls > 0 and 
                turn_analyses[i+1].progress_score > turn_analyses[i].progress_score):
                recovery_instances += 1
        
        return recovery_instances >= 2
    
    def _calculate_topic_continuity(self, turn_analyses: List[TurnAnalysis]) -> float:
        """Calculate topic continuity across turns"""
        
        # Simple heuristic based on tool usage consistency and phase transitions
        if len(turn_analyses) < 2:
            return 1.0
        
        continuity_score = 0.0
        
        for i in range(len(turn_analyses) - 1):
            current_turn = turn_analyses[i]
            next_turn = turn_analyses[i + 1]
            
            # Check tool usage continuity
            common_tools = set(current_turn.tools_used) & set(next_turn.tools_used)
            tool_continuity = len(common_tools) / max(1, len(set(current_turn.tools_used) | set(next_turn.tools_used)))
            
            # Check phase continuity
            phase_continuity = 1.0 if current_turn.phase_name == next_turn.phase_name else 0.5
            
            continuity_score += (tool_continuity * 0.6 + phase_continuity * 0.4)
        
        return continuity_score / (len(turn_analyses) - 1)
    
    async def save_session_evaluation(
        self,
        evaluation_result: SessionEvaluationResult,
        output_directory: Path
    ) -> Path:
        """Save session evaluation results"""
        
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Save detailed evaluation
        eval_file = output_directory / f"{evaluation_result.session_id}_session_evaluation.json"
        
        with open(eval_file, 'w') as f:
            json.dump(evaluation_result.to_dict(), f, indent=2)
        
        logger.info(f"Session evaluation saved to: {eval_file}")
        
        return eval_file
