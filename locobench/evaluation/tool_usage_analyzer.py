"""
Tool Usage Analyzer for LoCoBench-Agent

This module provides comprehensive analysis of tool usage patterns,
effectiveness metrics, and optimization recommendations for agent tool usage.
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools available to agents"""
    FILE_SYSTEM = "file_system"
    COMPILER = "compiler"
    DEBUGGER = "debugger"
    IDE_SIMULATOR = "ide_simulator"
    SEARCH = "search"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    UTILITY = "utility"


class UsagePattern(Enum):
    """Patterns of tool usage"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ITERATIVE = "iterative"
    EXPLORATORY = "exploratory"
    FOCUSED = "focused"
    SCATTERED = "scattered"


class EfficiencyLevel(Enum):
    """Tool usage efficiency levels"""
    OPTIMAL = "optimal"
    EFFICIENT = "efficient"
    ADEQUATE = "adequate"
    INEFFICIENT = "inefficient"
    WASTEFUL = "wasteful"


@dataclass
class ToolUsageEvent:
    """Represents a single tool usage event"""
    
    timestamp: datetime
    tool_name: str
    tool_category: ToolCategory
    
    # Call details
    function_called: str
    parameters: Dict[str, Any]
    
    # Results
    success: bool
    execution_time_ms: float
    result_size: int  # Size of result data
    
    # Context
    turn_number: int
    phase_name: Optional[str] = None
    
    # Analysis
    necessity_score: float = 0.0  # How necessary was this call
    effectiveness_score: float = 0.0  # How effective was this call
    efficiency_score: float = 0.0  # How efficient was this call
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "tool_category": self.tool_category.value,
            "function_called": self.function_called,
            "parameters": self.parameters,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "result_size": self.result_size,
            "turn_number": self.turn_number,
            "phase_name": self.phase_name,
            "necessity_score": self.necessity_score,
            "effectiveness_score": self.effectiveness_score,
            "efficiency_score": self.efficiency_score
        }


@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for a specific tool"""
    
    tool_name: str
    tool_category: ToolCategory
    
    # Usage statistics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Performance statistics
    avg_execution_time: float = 0.0
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0
    
    # Effectiveness metrics
    avg_necessity_score: float = 0.0
    avg_effectiveness_score: float = 0.0
    avg_efficiency_score: float = 0.0
    
    # Usage patterns
    usage_frequency: float = 0.0
    usage_phases: List[str] = field(default_factory=list)
    common_functions: List[Tuple[str, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_category": self.tool_category.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(1, self.total_calls),
            "avg_execution_time": self.avg_execution_time,
            "min_execution_time": self.min_execution_time,
            "max_execution_time": self.max_execution_time,
            "avg_necessity_score": self.avg_necessity_score,
            "avg_effectiveness_score": self.avg_effectiveness_score,
            "avg_efficiency_score": self.avg_efficiency_score,
            "usage_frequency": self.usage_frequency,
            "usage_phases": self.usage_phases,
            "common_functions": self.common_functions
        }


@dataclass
class ToolUsageAnalysisResult:
    """Complete tool usage analysis result"""
    
    session_id: str
    analysis_timestamp: datetime
    
    # Overall statistics
    total_tool_calls: int = 0
    unique_tools_used: int = 0
    total_execution_time: float = 0.0
    overall_success_rate: float = 0.0
    
    # Usage events
    usage_events: List[ToolUsageEvent] = field(default_factory=list)
    
    # Tool-specific metrics
    tool_metrics: Dict[str, ToolPerformanceMetrics] = field(default_factory=dict)
    
    # Pattern analysis
    usage_pattern: UsagePattern = UsagePattern.SEQUENTIAL
    efficiency_level: EfficiencyLevel = EfficiencyLevel.ADEQUATE
    
    # Category analysis
    category_usage: Dict[ToolCategory, int] = field(default_factory=dict)
    category_effectiveness: Dict[ToolCategory, float] = field(default_factory=dict)
    
    # Optimization insights
    optimization_recommendations: List[str] = field(default_factory=list)
    underused_tools: List[str] = field(default_factory=list)
    overused_tools: List[str] = field(default_factory=list)
    
    # Efficiency metrics
    tool_usage_efficiency_score: float = 0.0
    tool_selection_accuracy: float = 0.0
    redundancy_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_tool_calls": self.total_tool_calls,
            "unique_tools_used": self.unique_tools_used,
            "total_execution_time": self.total_execution_time,
            "overall_success_rate": self.overall_success_rate,
            "usage_events": [event.to_dict() for event in self.usage_events],
            "tool_metrics": {k: v.to_dict() for k, v in self.tool_metrics.items()},
            "usage_pattern": self.usage_pattern.value,
            "efficiency_level": self.efficiency_level.value,
            "category_usage": {k.value: v for k, v in self.category_usage.items()},
            "category_effectiveness": {k.value: v for k, v in self.category_effectiveness.items()},
            "optimization_recommendations": self.optimization_recommendations,
            "underused_tools": self.underused_tools,
            "overused_tools": self.overused_tools,
            "tool_usage_efficiency_score": self.tool_usage_efficiency_score,
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "redundancy_score": self.redundancy_score
        }


class ToolUsageAnalyzer:
    """
    Analyzer for tool usage patterns and effectiveness
    
    This analyzer examines how agents use available tools, identifies
    patterns, measures effectiveness, and provides optimization recommendations.
    """
    
    # Tool category mappings
    TOOL_CATEGORIES = {
        "file_system": ToolCategory.FILE_SYSTEM,
        "compiler": ToolCategory.COMPILER,
        "debugger": ToolCategory.DEBUGGER,
        "ide_simulator": ToolCategory.IDE_SIMULATOR,
        "code_search": ToolCategory.SEARCH,
        "search_tools": ToolCategory.SEARCH,
        "calculator": ToolCategory.UTILITY,
        "echo": ToolCategory.UTILITY
    }
    
    # Expected tool usage patterns for different phases
    PHASE_TOOL_EXPECTATIONS = {
        "analysis": [ToolCategory.FILE_SYSTEM, ToolCategory.SEARCH, ToolCategory.IDE_SIMULATOR],
        "planning": [ToolCategory.SEARCH, ToolCategory.ANALYSIS],
        "implementation": [ToolCategory.FILE_SYSTEM, ToolCategory.COMPILER, ToolCategory.IDE_SIMULATOR],
        "testing": [ToolCategory.COMPILER, ToolCategory.DEBUGGER],
        "debugging": [ToolCategory.DEBUGGER, ToolCategory.SEARCH, ToolCategory.IDE_SIMULATOR],
        "documentation": [ToolCategory.FILE_SYSTEM, ToolCategory.SEARCH]
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Analysis thresholds
        self.efficiency_thresholds = {
            EfficiencyLevel.OPTIMAL: 4.5,
            EfficiencyLevel.EFFICIENT: 3.5,
            EfficiencyLevel.ADEQUATE: 2.5,
            EfficiencyLevel.INEFFICIENT: 1.5
        }
        
        logger.info("ToolUsageAnalyzer initialized")
    
    async def analyze_tool_usage(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any] = None
    ) -> ToolUsageAnalysisResult:
        """
        Analyze tool usage patterns in a session
        
        Args:
            session_result: Results from agent session
            scenario_context: Context about the scenario
            
        Returns:
            Complete tool usage analysis result
        """
        
        logger.info(f"Analyzing tool usage for session: {session_result.get('session_id', 'unknown')}")
        
        # Initialize result
        result = ToolUsageAnalysisResult(
            session_id=session_result.get("session_id", "unknown"),
            analysis_timestamp=datetime.now()
        )
        
        # Extract tool usage events
        result.usage_events = self._extract_usage_events(session_result)
        result.total_tool_calls = len(result.usage_events)
        
        if result.usage_events:
            # Calculate basic statistics
            result.unique_tools_used = len(set(event.tool_name for event in result.usage_events))
            result.total_execution_time = sum(event.execution_time_ms for event in result.usage_events)
            result.overall_success_rate = sum(1 for event in result.usage_events if event.success) / len(result.usage_events)
            
            # Analyze individual events
            await self._analyze_usage_events(result.usage_events, session_result)
            
            # Calculate tool-specific metrics
            result.tool_metrics = self._calculate_tool_metrics(result.usage_events)
            
            # Analyze usage patterns
            result.usage_pattern = self._identify_usage_pattern(result.usage_events)
            result.efficiency_level = self._assess_efficiency_level(result.usage_events)
            
            # Analyze by category
            result.category_usage, result.category_effectiveness = self._analyze_by_category(result.usage_events)
            
            # Calculate efficiency metrics
            result.tool_usage_efficiency_score = self._calculate_usage_efficiency(result.usage_events)
            result.tool_selection_accuracy = self._calculate_selection_accuracy(result.usage_events, scenario_context)
            result.redundancy_score = self._calculate_redundancy_score(result.usage_events)
            
            # Generate recommendations
            result.optimization_recommendations = self._generate_optimization_recommendations(result)
            result.underused_tools, result.overused_tools = self._identify_usage_imbalances(result, scenario_context)
        
        logger.info(f"Tool usage analysis completed. Efficiency score: {result.tool_usage_efficiency_score:.2f}")
        
        return result
    
    def _extract_usage_events(self, session_result: Dict[str, Any]) -> List[ToolUsageEvent]:
        """Extract tool usage events from session data"""
        
        events = []
        tool_usage_log = session_result.get("tool_usage_log", [])
        
        for entry in tool_usage_log:
            try:
                timestamp = datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat()))
                tool_name = entry.get("tool_name", "unknown")
                
                # Determine tool category
                tool_category = self.TOOL_CATEGORIES.get(tool_name, ToolCategory.UTILITY)
                
                event = ToolUsageEvent(
                    timestamp=timestamp,
                    tool_name=tool_name,
                    tool_category=tool_category,
                    function_called=entry.get("function_name", "unknown"),
                    parameters=entry.get("parameters", {}),
                    success=entry.get("success", False),
                    execution_time_ms=entry.get("execution_time_ms", 0),
                    result_size=len(str(entry.get("result", ""))),
                    turn_number=entry.get("turn_number", 0),
                    phase_name=entry.get("phase_name")
                )
                
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Error processing tool usage entry: {e}")
                continue
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    async def _analyze_usage_events(self, events: List[ToolUsageEvent], session_result: Dict[str, Any]):
        """Analyze individual usage events for necessity, effectiveness, and efficiency"""
        
        for i, event in enumerate(events):
            # Analyze necessity
            event.necessity_score = self._calculate_necessity_score(event, events[:i], session_result)
            
            # Analyze effectiveness
            event.effectiveness_score = self._calculate_effectiveness_score(event, events[i+1:i+6])
            
            # Analyze efficiency
            event.efficiency_score = self._calculate_efficiency_score(event)
    
    def _calculate_necessity_score(
        self,
        event: ToolUsageEvent,
        previous_events: List[ToolUsageEvent],
        session_result: Dict[str, Any]
    ) -> float:
        """Calculate how necessary a tool call was"""
        
        base_score = 3.0
        
        # Check if similar call was made recently (redundancy penalty)
        recent_similar = [
            e for e in previous_events[-5:]  # Last 5 events
            if e.tool_name == event.tool_name and e.function_called == event.function_called
        ]
        
        if recent_similar:
            # Penalize redundant calls
            base_score -= min(2.0, len(recent_similar) * 0.5)
        
        # Reward tool calls that are appropriate for the phase
        if event.phase_name:
            expected_categories = self.PHASE_TOOL_EXPECTATIONS.get(event.phase_name, [])
            if event.tool_category in expected_categories:
                base_score += 1.0
        
        # Reward successful information gathering
        if event.success and event.result_size > 0:
            base_score += 0.5
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_effectiveness_score(
        self,
        event: ToolUsageEvent,
        subsequent_events: List[ToolUsageEvent]
    ) -> float:
        """Calculate how effective a tool call was"""
        
        if not event.success:
            return 1.0  # Failed calls are minimally effective
        
        base_score = 3.0
        
        # Reward if result was substantial
        if event.result_size > 100:
            base_score += 1.0
        elif event.result_size > 10:
            base_score += 0.5
        
        # Check if this call enabled subsequent successful actions
        enabled_actions = 0
        for subsequent in subsequent_events:
            if subsequent.success:
                enabled_actions += 1
        
        if enabled_actions > 0:
            base_score += min(1.0, enabled_actions * 0.2)
        
        # Specific effectiveness for different tool types
        if event.tool_category == ToolCategory.FILE_SYSTEM:
            # File operations should lead to meaningful changes
            if "write" in event.function_called.lower() or "create" in event.function_called.lower():
                base_score += 0.5
        
        elif event.tool_category == ToolCategory.COMPILER:
            # Compilation should be successful for effectiveness
            if event.success:
                base_score += 1.0
        
        elif event.tool_category == ToolCategory.DEBUGGER:
            # Debugging should identify issues
            if event.result_size > 50:  # Meaningful debug output
                base_score += 0.5
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_efficiency_score(self, event: ToolUsageEvent) -> float:
        """Calculate how efficient a tool call was"""
        
        base_score = 3.0
        
        # Penalize very slow operations
        if event.execution_time_ms > 5000:  # > 5 seconds
            base_score -= 2.0
        elif event.execution_time_ms > 2000:  # > 2 seconds
            base_score -= 1.0
        elif event.execution_time_ms < 100:  # < 100ms
            base_score += 1.0
        
        # Reward successful calls
        if event.success:
            base_score += 1.0
        else:
            base_score -= 1.0
        
        # Consider result size vs execution time ratio
        if event.execution_time_ms > 0 and event.result_size > 0:
            efficiency_ratio = event.result_size / event.execution_time_ms
            if efficiency_ratio > 1.0:  # Good data per ms
                base_score += 0.5
        
        return min(5.0, max(0.0, base_score))
    
    def _calculate_tool_metrics(self, events: List[ToolUsageEvent]) -> Dict[str, ToolPerformanceMetrics]:
        """Calculate performance metrics for each tool"""
        
        tool_metrics = {}
        
        # Group events by tool
        tool_events = defaultdict(list)
        for event in events:
            tool_events[event.tool_name].append(event)
        
        for tool_name, tool_event_list in tool_events.items():
            if not tool_event_list:
                continue
            
            first_event = tool_event_list[0]
            metrics = ToolPerformanceMetrics(
                tool_name=tool_name,
                tool_category=first_event.tool_category
            )
            
            # Basic statistics
            metrics.total_calls = len(tool_event_list)
            metrics.successful_calls = sum(1 for e in tool_event_list if e.success)
            metrics.failed_calls = metrics.total_calls - metrics.successful_calls
            
            # Performance statistics
            execution_times = [e.execution_time_ms for e in tool_event_list]
            metrics.avg_execution_time = statistics.mean(execution_times)
            metrics.min_execution_time = min(execution_times)
            metrics.max_execution_time = max(execution_times)
            
            # Effectiveness metrics
            metrics.avg_necessity_score = statistics.mean([e.necessity_score for e in tool_event_list])
            metrics.avg_effectiveness_score = statistics.mean([e.effectiveness_score for e in tool_event_list])
            metrics.avg_efficiency_score = statistics.mean([e.efficiency_score for e in tool_event_list])
            
            # Usage patterns
            metrics.usage_frequency = metrics.total_calls / len(events)
            metrics.usage_phases = list(set(e.phase_name for e in tool_event_list if e.phase_name))
            
            # Common functions
            function_counts = Counter(e.function_called for e in tool_event_list)
            metrics.common_functions = function_counts.most_common(5)
            
            tool_metrics[tool_name] = metrics
        
        return tool_metrics
    
    def _identify_usage_pattern(self, events: List[ToolUsageEvent]) -> UsagePattern:
        """Identify the overall tool usage pattern"""
        
        if len(events) < 3:
            return UsagePattern.SEQUENTIAL
        
        # Analyze tool switching patterns
        tool_switches = 0
        for i in range(1, len(events)):
            if events[i].tool_name != events[i-1].tool_name:
                tool_switches += 1
        
        switch_rate = tool_switches / (len(events) - 1)
        
        # Analyze tool diversity
        unique_tools = len(set(e.tool_name for e in events))
        diversity_ratio = unique_tools / len(events)
        
        # Analyze repetition patterns
        tool_counts = Counter(e.tool_name for e in events)
        max_usage = max(tool_counts.values())
        repetition_ratio = max_usage / len(events)
        
        # Classify pattern
        if switch_rate > 0.7:
            return UsagePattern.SCATTERED
        elif switch_rate > 0.4 and diversity_ratio > 0.3:
            return UsagePattern.EXPLORATORY
        elif repetition_ratio > 0.6:
            return UsagePattern.FOCUSED
        elif self._has_iterative_cycles(events):
            return UsagePattern.ITERATIVE
        elif switch_rate < 0.2:
            return UsagePattern.SEQUENTIAL
        else:
            return UsagePattern.PARALLEL
    
    def _has_iterative_cycles(self, events: List[ToolUsageEvent]) -> bool:
        """Check if events show iterative cycles"""
        
        # Look for repeated sequences of tools
        if len(events) < 6:
            return False
        
        # Simple pattern detection
        tool_sequence = [e.tool_name for e in events]
        
        # Look for repeating subsequences
        for length in range(2, len(tool_sequence) // 3):
            for start in range(len(tool_sequence) - 2 * length):
                subseq = tool_sequence[start:start + length]
                next_subseq = tool_sequence[start + length:start + 2 * length]
                
                if subseq == next_subseq:
                    return True
        
        return False
    
    def _assess_efficiency_level(self, events: List[ToolUsageEvent]) -> EfficiencyLevel:
        """Assess overall efficiency level"""
        
        if not events:
            return EfficiencyLevel.ADEQUATE
        
        # Calculate composite efficiency score
        avg_efficiency = statistics.mean([e.efficiency_score for e in events])
        avg_necessity = statistics.mean([e.necessity_score for e in events])
        avg_effectiveness = statistics.mean([e.effectiveness_score for e in events])
        
        composite_score = (avg_efficiency * 0.4 + avg_necessity * 0.3 + avg_effectiveness * 0.3)
        
        # Map to efficiency levels
        for level, threshold in sorted(self.efficiency_thresholds.items(), key=lambda x: x[1], reverse=True):
            if composite_score >= threshold:
                return level
        
        return EfficiencyLevel.WASTEFUL
    
    def _analyze_by_category(self, events: List[ToolUsageEvent]) -> Tuple[Dict[ToolCategory, int], Dict[ToolCategory, float]]:
        """Analyze tool usage by category"""
        
        category_usage = Counter()
        category_effectiveness = defaultdict(list)
        
        for event in events:
            category_usage[event.tool_category] += 1
            category_effectiveness[event.tool_category].append(event.effectiveness_score)
        
        # Calculate average effectiveness by category
        avg_effectiveness = {}
        for category, scores in category_effectiveness.items():
            avg_effectiveness[category] = statistics.mean(scores)
        
        return dict(category_usage), avg_effectiveness
    
    def _calculate_usage_efficiency(self, events: List[ToolUsageEvent]) -> float:
        """Calculate overall tool usage efficiency score"""
        
        if not events:
            return 0.0
        
        # Base efficiency from individual scores
        avg_efficiency = statistics.mean([e.efficiency_score for e in events])
        
        # Adjust for success rate
        success_rate = sum(1 for e in events if e.success) / len(events)
        
        # Adjust for redundancy
        redundancy_penalty = self._calculate_redundancy_score(events) / 5.0
        
        # Adjust for tool diversity (good) vs scattered usage (bad)
        unique_tools = len(set(e.tool_name for e in events))
        diversity_ratio = unique_tools / len(events)
        
        if 0.2 <= diversity_ratio <= 0.6:
            diversity_bonus = 0.5
        else:
            diversity_bonus = -0.5
        
        efficiency_score = (avg_efficiency + success_rate * 2.0 - redundancy_penalty + diversity_bonus)
        
        return min(5.0, max(0.0, efficiency_score))
    
    def _calculate_selection_accuracy(self, events: List[ToolUsageEvent], scenario_context: Dict[str, Any] = None) -> float:
        """Calculate tool selection accuracy"""
        
        if not events:
            return 0.0
        
        # Base accuracy on necessity scores
        avg_necessity = statistics.mean([e.necessity_score for e in events])
        
        # Adjust for phase-appropriate tool usage
        phase_appropriate = 0
        total_with_phase = 0
        
        for event in events:
            if event.phase_name:
                total_with_phase += 1
                expected_categories = self.PHASE_TOOL_EXPECTATIONS.get(event.phase_name, [])
                if event.tool_category in expected_categories:
                    phase_appropriate += 1
        
        phase_accuracy = phase_appropriate / max(1, total_with_phase)
        
        selection_accuracy = (avg_necessity + phase_accuracy * 5.0) / 2.0
        
        return min(5.0, max(0.0, selection_accuracy))
    
    def _calculate_redundancy_score(self, events: List[ToolUsageEvent]) -> float:
        """Calculate redundancy in tool usage (higher = more redundant)"""
        
        if len(events) < 2:
            return 0.0
        
        redundant_calls = 0
        
        for i, event in enumerate(events):
            # Look for similar calls in recent history
            recent_window = events[max(0, i-5):i]  # Last 5 calls
            
            similar_calls = [
                e for e in recent_window
                if (e.tool_name == event.tool_name and 
                    e.function_called == event.function_called and
                    e.success)  # Only count successful similar calls
            ]
            
            if similar_calls:
                redundant_calls += 1
        
        redundancy_ratio = redundant_calls / len(events)
        
        return redundancy_ratio * 5.0
    
    def _generate_optimization_recommendations(self, result: ToolUsageAnalysisResult) -> List[str]:
        """Generate optimization recommendations based on analysis"""
        
        recommendations = []
        
        # Success rate recommendations
        if result.overall_success_rate < 0.8:
            recommendations.append(
                f"Improve tool call success rate (currently {result.overall_success_rate:.1%}). "
                "Consider validating parameters before tool calls."
            )
        
        # Efficiency recommendations
        if result.efficiency_level in [EfficiencyLevel.INEFFICIENT, EfficiencyLevel.WASTEFUL]:
            recommendations.append(
                "Tool usage efficiency is below optimal. Consider reducing redundant calls "
                "and using more appropriate tools for each task phase."
            )
        
        # Pattern recommendations
        if result.usage_pattern == UsagePattern.SCATTERED:
            recommendations.append(
                "Tool usage appears scattered. Consider following a more systematic approach "
                "with focused tool usage per phase."
            )
        elif result.usage_pattern == UsagePattern.FOCUSED and result.unique_tools_used < 3:
            recommendations.append(
                "Tool usage is too focused on few tools. Consider exploring other available "
                "tools that might be more suitable for different tasks."
            )
        
        # Redundancy recommendations
        if result.redundancy_score > 2.0:
            recommendations.append(
                "High redundancy detected in tool calls. Avoid repeating similar operations "
                "within short time windows."
            )
        
        # Category balance recommendations
        total_calls = sum(result.category_usage.values())
        if total_calls > 0:
            file_system_ratio = result.category_usage.get(ToolCategory.FILE_SYSTEM, 0) / total_calls
            
            if file_system_ratio > 0.6:
                recommendations.append(
                    "Heavy reliance on file system operations. Consider using IDE simulator "
                    "or search tools for more efficient code exploration."
                )
            elif file_system_ratio < 0.2:
                recommendations.append(
                    "Limited file system interaction. Ensure you're adequately exploring "
                    "and modifying the codebase as needed."
                )
        
        # Tool-specific recommendations
        for tool_name, metrics in result.tool_metrics.items():
            if metrics.total_calls > 10 and metrics.successful_calls / metrics.total_calls < 0.7:
                recommendations.append(
                    f"Low success rate for {tool_name} ({metrics.successful_calls / metrics.total_calls:.1%}). "
                    "Review parameter usage and error handling."
                )
        
        return recommendations
    
    def _identify_usage_imbalances(
        self,
        result: ToolUsageAnalysisResult,
        scenario_context: Dict[str, Any] = None
    ) -> Tuple[List[str], List[str]]:
        """Identify underused and overused tools"""
        
        available_tools = []
        if scenario_context and "available_tools" in scenario_context:
            available_tools = scenario_context["available_tools"]
        
        used_tools = set(result.tool_metrics.keys())
        total_calls = result.total_tool_calls
        
        underused_tools = []
        overused_tools = []
        
        # Identify underused tools
        if available_tools:
            unused_tools = set(available_tools) - used_tools
            underused_tools.extend(unused_tools)
        
        # Tools used very infrequently
        for tool_name, metrics in result.tool_metrics.items():
            if metrics.usage_frequency < 0.05 and total_calls > 20:  # Less than 5% usage
                underused_tools.append(tool_name)
        
        # Identify overused tools
        for tool_name, metrics in result.tool_metrics.items():
            if metrics.usage_frequency > 0.5:  # More than 50% of all calls
                overused_tools.append(tool_name)
        
        return underused_tools, overused_tools
    
    async def save_tool_usage_analysis(
        self,
        analysis_result: ToolUsageAnalysisResult,
        output_directory: str
    ) -> str:
        """Save tool usage analysis results"""
        
        from pathlib import Path
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed analysis
        analysis_file = output_path / f"{analysis_result.session_id}_tool_usage_analysis.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)
        
        # Save summary report
        summary_file = output_path / f"{analysis_result.session_id}_tool_usage_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Tool Usage Analysis Summary\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Session ID: {analysis_result.session_id}\n")
            f.write(f"Total Tool Calls: {analysis_result.total_tool_calls}\n")
            f.write(f"Unique Tools Used: {analysis_result.unique_tools_used}\n")
            f.write(f"Overall Success Rate: {analysis_result.overall_success_rate:.1%}\n")
            f.write(f"Usage Pattern: {analysis_result.usage_pattern.value}\n")
            f.write(f"Efficiency Level: {analysis_result.efficiency_level.value}\n")
            f.write(f"Efficiency Score: {analysis_result.tool_usage_efficiency_score:.2f}/5.0\n\n")
            
            f.write("Tool Performance:\n")
            f.write("-" * 20 + "\n")
            for tool_name, metrics in analysis_result.tool_metrics.items():
                f.write(f"{tool_name}: {metrics.successful_calls}/{metrics.total_calls} success "
                       f"({metrics.successful_calls/max(1,metrics.total_calls):.1%})\n")
            
            f.write("\nOptimization Recommendations:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(analysis_result.optimization_recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Tool usage analysis saved to: {analysis_file}")
        
        return str(analysis_file)
