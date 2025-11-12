"""
Collaboration Metrics for LoCoBench-Agent

This module provides specialized metrics for evaluating agent-human interaction
and collaboration patterns in software development scenarios.
"""

import json
import logging
import re
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .agent_metrics import MetricCategory, AgentMetricResult

logger = logging.getLogger(__name__)


class CollaborationMode(Enum):
    """Types of collaboration modes"""
    AUTONOMOUS = "autonomous"
    GUIDED = "guided"
    COLLABORATIVE = "collaborative"
    SUPERVISED = "supervised"
    INTERACTIVE = "interactive"


class InteractionType(Enum):
    """Types of human-agent interactions"""
    CLARIFICATION_REQUEST = "clarification_request"
    FEEDBACK_PROVISION = "feedback_provision"
    CORRECTION = "correction"
    GUIDANCE = "guidance"
    APPROVAL_REQUEST = "approval_request"
    STATUS_UPDATE = "status_update"
    QUESTION = "question"
    SUGGESTION = "suggestion"


class CommunicationQuality(Enum):
    """Quality levels for communication"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    UNCLEAR = "unclear"


@dataclass
class InteractionEvent:
    """Represents a single interaction event"""
    
    timestamp: datetime
    interaction_type: InteractionType
    initiator: str  # "agent" or "human"
    content: str
    
    # Context
    turn_number: int
    phase_name: Optional[str] = None
    
    # Analysis
    clarity_score: float = 0.0
    relevance_score: float = 0.0
    timeliness_score: float = 0.0
    
    # Response tracking
    response_received: bool = False
    response_time_seconds: Optional[float] = None
    response_quality: Optional[CommunicationQuality] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "interaction_type": self.interaction_type.value,
            "initiator": self.initiator,
            "content": self.content,
            "turn_number": self.turn_number,
            "phase_name": self.phase_name,
            "clarity_score": self.clarity_score,
            "relevance_score": self.relevance_score,
            "timeliness_score": self.timeliness_score,
            "response_received": self.response_received,
            "response_time_seconds": self.response_time_seconds,
            "response_quality": self.response_quality.value if self.response_quality else None
        }


@dataclass
class CollaborationPattern:
    """Represents a collaboration pattern observed in the session"""
    
    pattern_name: str
    frequency: int
    effectiveness_score: float
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "frequency": self.frequency,
            "effectiveness_score": self.effectiveness_score,
            "examples": self.examples
        }


@dataclass
class CollaborationAnalysisResult:
    """Complete collaboration analysis result"""
    
    session_id: str
    collaboration_mode: CollaborationMode
    analysis_timestamp: datetime
    
    # Interaction analysis
    total_interactions: int = 0
    agent_initiated: int = 0
    human_initiated: int = 0
    interaction_events: List[InteractionEvent] = field(default_factory=list)
    
    # Communication quality
    average_clarity_score: float = 0.0
    average_relevance_score: float = 0.0
    average_response_time: float = 0.0
    communication_effectiveness: float = 0.0
    
    # Collaboration patterns
    identified_patterns: List[CollaborationPattern] = field(default_factory=list)
    
    # Specific metrics
    collaboration_intelligence_score: float = 0.0
    proactivity_score: float = 0.0
    responsiveness_score: float = 0.0
    adaptability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "collaboration_mode": self.collaboration_mode.value,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_interactions": self.total_interactions,
            "agent_initiated": self.agent_initiated,
            "human_initiated": self.human_initiated,
            "interaction_events": [event.to_dict() for event in self.interaction_events],
            "average_clarity_score": self.average_clarity_score,
            "average_relevance_score": self.average_relevance_score,
            "average_response_time": self.average_response_time,
            "communication_effectiveness": self.communication_effectiveness,
            "identified_patterns": [pattern.to_dict() for pattern in self.identified_patterns],
            "collaboration_intelligence_score": self.collaboration_intelligence_score,
            "proactivity_score": self.proactivity_score,
            "responsiveness_score": self.responsiveness_score,
            "adaptability_score": self.adaptability_score
        }


class CollaborationMetricsCalculator:
    """
    Calculator for collaboration and human-agent interaction metrics
    
    This class analyzes conversation patterns to identify collaboration
    effectiveness, communication quality, and interaction patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Patterns for identifying interaction types
        self.interaction_patterns = {
            InteractionType.CLARIFICATION_REQUEST: [
                r"what do you mean", r"could you clarify", r"i don't understand",
                r"can you explain", r"what exactly", r"unclear about"
            ],
            InteractionType.FEEDBACK_PROVISION: [
                r"that looks good", r"i think", r"consider", r"suggestion",
                r"feedback", r"improvement", r"better if"
            ],
            InteractionType.CORRECTION: [
                r"that's wrong", r"incorrect", r"actually", r"should be",
                r"error", r"mistake", r"fix"
            ],
            InteractionType.APPROVAL_REQUEST: [
                r"is this okay", r"does this look right", r"should i proceed",
                r"do you approve", r"permission", r"go ahead"
            ],
            InteractionType.QUESTION: [
                r"how should", r"what would", r"which way", r"do you want",
                r"preference", r"question"
            ]
        }
        
        logger.info("CollaborationMetricsCalculator initialized")
    
    async def analyze_collaboration(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any] = None
    ) -> CollaborationAnalysisResult:
        """
        Analyze collaboration patterns in a session
        
        Args:
            session_result: Results from agent session
            scenario_context: Context about the scenario
            
        Returns:
            Complete collaboration analysis result
        """
        
        logger.info(f"Analyzing collaboration for session: {session_result.get('session_id', 'unknown')}")
        
        # Initialize result
        result = CollaborationAnalysisResult(
            session_id=session_result.get("session_id", "unknown"),
            collaboration_mode=self._determine_collaboration_mode(session_result, scenario_context),
            analysis_timestamp=datetime.now()
        )
        
        # Extract interaction events
        result.interaction_events = self._extract_interaction_events(session_result)
        result.total_interactions = len(result.interaction_events)
        
        # Count initiators
        result.agent_initiated = len([e for e in result.interaction_events if e.initiator == "agent"])
        result.human_initiated = len([e for e in result.interaction_events if e.initiator == "human"])
        
        # Analyze communication quality
        if result.interaction_events:
            result.average_clarity_score = statistics.mean([e.clarity_score for e in result.interaction_events])
            result.average_relevance_score = statistics.mean([e.relevance_score for e in result.interaction_events])
            
            response_times = [e.response_time_seconds for e in result.interaction_events if e.response_time_seconds is not None]
            result.average_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # Calculate communication effectiveness
        result.communication_effectiveness = self._calculate_communication_effectiveness(result.interaction_events)
        
        # Identify collaboration patterns
        result.identified_patterns = self._identify_collaboration_patterns(result.interaction_events)
        
        # Calculate specific collaboration metrics
        result.collaboration_intelligence_score = await self._calculate_collaboration_intelligence(session_result, result)
        result.proactivity_score = self._calculate_proactivity_score(result.interaction_events)
        result.responsiveness_score = self._calculate_responsiveness_score(result.interaction_events)
        result.adaptability_score = self._calculate_adaptability_score(session_result, result.interaction_events)
        
        logger.info(f"Collaboration analysis completed. Intelligence score: {result.collaboration_intelligence_score:.2f}")
        
        return result
    
    def _determine_collaboration_mode(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any] = None
    ) -> CollaborationMode:
        """Determine the collaboration mode from session data"""
        
        # Check scenario context first
        if scenario_context and "collaboration_mode" in scenario_context:
            mode_str = scenario_context["collaboration_mode"]
            try:
                return CollaborationMode(mode_str)
            except ValueError:
                pass
        
        # Infer from session data
        human_interventions = session_result.get("human_interventions", [])
        conversation_history = session_result.get("conversation_history", [])
        
        human_messages = len([msg for msg in conversation_history if msg.get("role") == "user"])
        total_messages = len(conversation_history)
        
        if human_messages == 0:
            return CollaborationMode.AUTONOMOUS
        elif human_messages / total_messages > 0.4:
            return CollaborationMode.COLLABORATIVE
        elif len(human_interventions) > 0:
            return CollaborationMode.GUIDED
        else:
            return CollaborationMode.SUPERVISED
    
    def _extract_interaction_events(self, session_result: Dict[str, Any]) -> List[InteractionEvent]:
        """Extract interaction events from session data"""
        
        events = []
        conversation_history = session_result.get("conversation_history", [])
        human_interventions = session_result.get("human_interventions", [])
        
        # Process conversation history
        for i, message in enumerate(conversation_history):
            role = message.get("role", "")
            content = message.get("content", "")
            timestamp = datetime.fromisoformat(message.get("timestamp", datetime.now().isoformat()))
            
            if role in ["user", "assistant"]:
                interaction_type = self._classify_interaction_type(content)
                initiator = "human" if role == "user" else "agent"
                
                event = InteractionEvent(
                    timestamp=timestamp,
                    interaction_type=interaction_type,
                    initiator=initiator,
                    content=content,
                    turn_number=i + 1
                )
                
                # Analyze event quality
                event.clarity_score = self._analyze_clarity(content)
                event.relevance_score = self._analyze_relevance(content, conversation_history[:i])
                event.timeliness_score = self._analyze_timeliness(event, conversation_history[:i])
                
                events.append(event)
        
        # Process human interventions
        for intervention in human_interventions:
            timestamp = datetime.fromisoformat(intervention.get("timestamp", datetime.now().isoformat()))
            content = intervention.get("message", "")
            
            event = InteractionEvent(
                timestamp=timestamp,
                interaction_type=InteractionType.GUIDANCE,
                initiator="human",
                content=content,
                turn_number=intervention.get("turn_number", 0)
            )
            
            event.clarity_score = self._analyze_clarity(content)
            event.relevance_score = 4.0  # Interventions are typically relevant
            event.timeliness_score = 4.0  # Interventions are typically timely
            
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        # Analyze response patterns
        self._analyze_response_patterns(events)
        
        return events
    
    def _classify_interaction_type(self, content: str) -> InteractionType:
        """Classify the type of interaction based on content"""
        
        content_lower = content.lower()
        
        # Check patterns for each interaction type
        for interaction_type, patterns in self.interaction_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return interaction_type
        
        # Default classification based on content characteristics
        if "?" in content:
            return InteractionType.QUESTION
        elif any(word in content_lower for word in ["update", "status", "progress"]):
            return InteractionType.STATUS_UPDATE
        else:
            return InteractionType.FEEDBACK_PROVISION
    
    def _analyze_clarity(self, content: str) -> float:
        """Analyze clarity of communication"""
        
        # Simple heuristics for clarity
        clarity_score = 3.0  # Base score
        
        # Length appropriateness
        if 20 <= len(content) <= 500:
            clarity_score += 1.0
        elif len(content) < 10 or len(content) > 1000:
            clarity_score -= 1.0
        
        # Structure indicators
        structure_indicators = ['. ', '\n', ':', '1.', '2.', '-', '*']
        structure_count = sum(content.count(indicator) for indicator in structure_indicators)
        
        if structure_count > 2:
            clarity_score += 0.5
        
        # Question marks for questions
        if content.count('?') > 0 and content.count('?') <= 3:
            clarity_score += 0.5
        
        return min(5.0, max(0.0, clarity_score))
    
    def _analyze_relevance(self, content: str, previous_messages: List[Dict]) -> float:
        """Analyze relevance of communication to context"""
        
        if not previous_messages:
            return 4.0  # First message is typically relevant
        
        relevance_score = 3.0  # Base score
        
        # Check for references to previous content
        recent_content = " ".join([msg.get("content", "") for msg in previous_messages[-3:]])
        
        # Simple word overlap analysis
        current_words = set(content.lower().split())
        recent_words = set(recent_content.lower().split())
        
        overlap = len(current_words & recent_words)
        total_words = len(current_words | recent_words)
        
        if total_words > 0:
            overlap_ratio = overlap / total_words
            relevance_score += overlap_ratio * 2.0
        
        return min(5.0, max(0.0, relevance_score))
    
    def _analyze_timeliness(self, event: InteractionEvent, previous_messages: List[Dict]) -> float:
        """Analyze timeliness of communication"""
        
        if not previous_messages:
            return 5.0  # First message is always timely
        
        timeliness_score = 4.0  # Base score
        
        # For questions and requests, timeliness is critical
        if event.interaction_type in [InteractionType.CLARIFICATION_REQUEST, InteractionType.QUESTION]:
            # Should be asked early when confusion arises
            if event.turn_number <= 3:
                timeliness_score += 1.0
        
        # For corrections, earlier is better
        elif event.interaction_type == InteractionType.CORRECTION:
            if event.turn_number <= 5:
                timeliness_score += 0.5
        
        return min(5.0, max(0.0, timeliness_score))
    
    def _analyze_response_patterns(self, events: List[InteractionEvent]):
        """Analyze response patterns between events"""
        
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # Check if next event is a response
            if (current_event.initiator != next_event.initiator and
                (next_event.timestamp - current_event.timestamp).total_seconds() < 300):  # 5 minutes
                
                current_event.response_received = True
                current_event.response_time_seconds = (next_event.timestamp - current_event.timestamp).total_seconds()
                current_event.response_quality = self._assess_response_quality(current_event, next_event)
    
    def _assess_response_quality(self, request_event: InteractionEvent, response_event: InteractionEvent) -> CommunicationQuality:
        """Assess the quality of a response"""
        
        # Simple heuristics based on response characteristics
        response_length = len(response_event.content)
        
        if response_event.clarity_score >= 4.0 and response_event.relevance_score >= 4.0:
            return CommunicationQuality.EXCELLENT
        elif response_event.clarity_score >= 3.5 and response_event.relevance_score >= 3.5:
            return CommunicationQuality.GOOD
        elif response_event.clarity_score >= 2.5 and response_event.relevance_score >= 2.5:
            return CommunicationQuality.ADEQUATE
        elif response_event.clarity_score >= 1.5:
            return CommunicationQuality.POOR
        else:
            return CommunicationQuality.UNCLEAR
    
    def _calculate_communication_effectiveness(self, events: List[InteractionEvent]) -> float:
        """Calculate overall communication effectiveness"""
        
        if not events:
            return 0.0
        
        # Average quality scores
        avg_clarity = statistics.mean([e.clarity_score for e in events])
        avg_relevance = statistics.mean([e.relevance_score for e in events])
        
        # Response rate
        responded_events = [e for e in events if e.response_received]
        response_rate = len(responded_events) / len(events) if events else 0
        
        # Response quality
        quality_scores = []
        for event in responded_events:
            if event.response_quality:
                quality_map = {
                    CommunicationQuality.EXCELLENT: 5.0,
                    CommunicationQuality.GOOD: 4.0,
                    CommunicationQuality.ADEQUATE: 3.0,
                    CommunicationQuality.POOR: 2.0,
                    CommunicationQuality.UNCLEAR: 1.0
                }
                quality_scores.append(quality_map[event.response_quality])
        
        avg_response_quality = statistics.mean(quality_scores) if quality_scores else 3.0
        
        # Combine metrics
        effectiveness = (avg_clarity * 0.3 + avg_relevance * 0.3 + response_rate * 2.5 + avg_response_quality * 0.2)
        
        return min(5.0, max(0.0, effectiveness))
    
    def _identify_collaboration_patterns(self, events: List[InteractionEvent]) -> List[CollaborationPattern]:
        """Identify collaboration patterns in the interaction events"""
        
        patterns = []
        
        # Pattern 1: Proactive Communication
        proactive_events = [e for e in events if e.initiator == "agent" and 
                           e.interaction_type in [InteractionType.STATUS_UPDATE, InteractionType.QUESTION]]
        
        if proactive_events:
            effectiveness = statistics.mean([e.relevance_score for e in proactive_events])
            patterns.append(CollaborationPattern(
                pattern_name="Proactive Communication",
                frequency=len(proactive_events),
                effectiveness_score=effectiveness,
                examples=[e.content[:100] + "..." if len(e.content) > 100 else e.content 
                         for e in proactive_events[:3]]
            ))
        
        # Pattern 2: Clarification Seeking
        clarification_events = [e for e in events if e.interaction_type == InteractionType.CLARIFICATION_REQUEST]
        
        if clarification_events:
            effectiveness = statistics.mean([e.timeliness_score for e in clarification_events])
            patterns.append(CollaborationPattern(
                pattern_name="Clarification Seeking",
                frequency=len(clarification_events),
                effectiveness_score=effectiveness,
                examples=[e.content[:100] + "..." if len(e.content) > 100 else e.content 
                         for e in clarification_events[:3]]
            ))
        
        # Pattern 3: Feedback Integration
        feedback_events = [e for e in events if e.initiator == "human" and 
                          e.interaction_type == InteractionType.FEEDBACK_PROVISION]
        
        if feedback_events:
            # Check for responses to feedback
            responses = []
            for event in feedback_events:
                if event.response_received:
                    responses.append(event)
            
            effectiveness = len(responses) / len(feedback_events) * 5.0 if feedback_events else 0
            patterns.append(CollaborationPattern(
                pattern_name="Feedback Integration",
                frequency=len(feedback_events),
                effectiveness_score=effectiveness,
                examples=[e.content[:100] + "..." if len(e.content) > 100 else e.content 
                         for e in feedback_events[:3]]
            ))
        
        # Pattern 4: Error Recovery Collaboration
        correction_events = [e for e in events if e.interaction_type == InteractionType.CORRECTION]
        
        if correction_events:
            # Check for appropriate responses to corrections
            good_responses = [e for e in correction_events if 
                            e.response_received and 
                            e.response_quality in [CommunicationQuality.GOOD, CommunicationQuality.EXCELLENT]]
            
            effectiveness = len(good_responses) / len(correction_events) * 5.0 if correction_events else 0
            patterns.append(CollaborationPattern(
                pattern_name="Error Recovery Collaboration",
                frequency=len(correction_events),
                effectiveness_score=effectiveness,
                examples=[e.content[:100] + "..." if len(e.content) > 100 else e.content 
                         for e in correction_events[:3]]
            ))
        
        return patterns
    
    async def _calculate_collaboration_intelligence(
        self,
        session_result: Dict[str, Any],
        collaboration_result: CollaborationAnalysisResult
    ) -> float:
        """Calculate overall collaboration intelligence score"""
        
        # Base score from communication effectiveness
        base_score = collaboration_result.communication_effectiveness
        
        # Adjust for interaction balance
        if collaboration_result.total_interactions > 0:
            agent_ratio = collaboration_result.agent_initiated / collaboration_result.total_interactions
            # Optimal ratio is around 0.6-0.7 (agent slightly more proactive)
            balance_score = 1.0 - abs(agent_ratio - 0.65) * 2
            base_score = (base_score + balance_score * 5.0) / 2.0
        
        # Adjust for pattern effectiveness
        if collaboration_result.identified_patterns:
            pattern_scores = [p.effectiveness_score for p in collaboration_result.identified_patterns]
            avg_pattern_effectiveness = statistics.mean(pattern_scores)
            base_score = (base_score + avg_pattern_effectiveness) / 2.0
        
        # Adjust for adaptability (based on session progression)
        conversation_history = session_result.get("conversation_history", [])
        if len(conversation_history) > 5:
            # Look for improvement in communication over time
            early_events = collaboration_result.interaction_events[:len(collaboration_result.interaction_events)//3]
            late_events = collaboration_result.interaction_events[-len(collaboration_result.interaction_events)//3:]
            
            if early_events and late_events:
                early_avg = statistics.mean([e.clarity_score for e in early_events])
                late_avg = statistics.mean([e.clarity_score for e in late_events])
                
                if late_avg > early_avg:
                    base_score += 0.5  # Bonus for improvement
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_proactivity_score(self, events: List[InteractionEvent]) -> float:
        """Calculate proactivity score"""
        
        if not events:
            return 0.0
        
        # Count proactive agent behaviors
        proactive_types = [
            InteractionType.STATUS_UPDATE,
            InteractionType.QUESTION,
            InteractionType.SUGGESTION,
            InteractionType.CLARIFICATION_REQUEST
        ]
        
        agent_events = [e for e in events if e.initiator == "agent"]
        proactive_events = [e for e in agent_events if e.interaction_type in proactive_types]
        
        if not agent_events:
            return 0.0
        
        proactivity_ratio = len(proactive_events) / len(agent_events)
        
        # Quality of proactive communications
        if proactive_events:
            avg_quality = statistics.mean([
                (e.clarity_score + e.relevance_score + e.timeliness_score) / 3.0 
                for e in proactive_events
            ])
        else:
            avg_quality = 0.0
        
        proactivity_score = (proactivity_ratio * 2.5 + avg_quality * 0.5)
        
        return min(5.0, max(0.0, proactivity_score))
    
    def _calculate_responsiveness_score(self, events: List[InteractionEvent]) -> float:
        """Calculate responsiveness score"""
        
        if not events:
            return 0.0
        
        # Find events that should have responses
        response_requiring_events = [
            e for e in events 
            if e.interaction_type in [
                InteractionType.QUESTION,
                InteractionType.CLARIFICATION_REQUEST,
                InteractionType.APPROVAL_REQUEST,
                InteractionType.CORRECTION
            ]
        ]
        
        if not response_requiring_events:
            return 5.0  # Perfect if no responses needed
        
        # Calculate response rate
        responded_events = [e for e in response_requiring_events if e.response_received]
        response_rate = len(responded_events) / len(response_requiring_events)
        
        # Calculate average response time (lower is better)
        response_times = [e.response_time_seconds for e in responded_events if e.response_time_seconds is not None]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            # Convert to score (60 seconds = perfect, 300 seconds = poor)
            time_score = max(0.0, 1.0 - (avg_response_time - 60) / 240)
        else:
            time_score = 0.0
        
        # Calculate response quality
        quality_scores = []
        for event in responded_events:
            if event.response_quality:
                quality_map = {
                    CommunicationQuality.EXCELLENT: 1.0,
                    CommunicationQuality.GOOD: 0.8,
                    CommunicationQuality.ADEQUATE: 0.6,
                    CommunicationQuality.POOR: 0.4,
                    CommunicationQuality.UNCLEAR: 0.2
                }
                quality_scores.append(quality_map[event.response_quality])
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        responsiveness_score = (response_rate * 2.0 + time_score * 1.5 + avg_quality * 1.5)
        
        return min(5.0, max(0.0, responsiveness_score))
    
    def _calculate_adaptability_score(self, session_result: Dict[str, Any], events: List[InteractionEvent]) -> float:
        """Calculate adaptability score"""
        
        if len(events) < 5:
            return 3.0  # Neutral score for short sessions
        
        # Look for adaptation patterns
        
        # 1. Improvement in communication quality over time
        early_events = events[:len(events)//3]
        late_events = events[-len(events)//3:]
        
        early_avg_clarity = statistics.mean([e.clarity_score for e in early_events if e.initiator == "agent"])
        late_avg_clarity = statistics.mean([e.clarity_score for e in late_events if e.initiator == "agent"])
        
        clarity_improvement = late_avg_clarity - early_avg_clarity
        
        # 2. Adaptation to feedback
        feedback_events = [e for e in events if e.interaction_type == InteractionType.FEEDBACK_PROVISION]
        corrections = [e for e in events if e.interaction_type == InteractionType.CORRECTION]
        
        adaptation_evidence = 0.0
        
        # Look for changes after feedback
        for feedback in feedback_events + corrections:
            # Find subsequent agent messages
            later_events = [e for e in events if e.timestamp > feedback.timestamp and e.initiator == "agent"][:3]
            
            if later_events:
                avg_quality_after = statistics.mean([
                    (e.clarity_score + e.relevance_score) / 2.0 for e in later_events
                ])
                
                if avg_quality_after > 3.5:  # Good quality after feedback
                    adaptation_evidence += 1.0
        
        feedback_adaptation = adaptation_evidence / max(1, len(feedback_events) + len(corrections))
        
        # 3. Strategy changes (tool usage patterns, communication styles)
        tool_usage_log = session_result.get("tool_usage_log", [])
        
        strategy_adaptation = 0.0
        if len(tool_usage_log) > 5:
            # Simple heuristic: diversity of tools used indicates adaptation
            early_tools = set([t.get("tool_name") for t in tool_usage_log[:len(tool_usage_log)//3]])
            late_tools = set([t.get("tool_name") for t in tool_usage_log[-len(tool_usage_log)//3:]])
            
            if late_tools - early_tools:  # New tools introduced
                strategy_adaptation = 1.0
        
        # Combine adaptation indicators
        adaptability_score = (
            clarity_improvement * 1.0 +
            feedback_adaptation * 2.0 +
            strategy_adaptation * 2.0
        )
        
        return min(5.0, max(0.0, adaptability_score))
    
    async def save_collaboration_analysis(
        self,
        analysis_result: CollaborationAnalysisResult,
        output_directory: str
    ) -> str:
        """Save collaboration analysis results"""
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed analysis
        analysis_file = output_path / f"{analysis_result.session_id}_collaboration_analysis.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)
        
        logger.info(f"Collaboration analysis saved to: {analysis_file}")
        
        return str(analysis_file)
