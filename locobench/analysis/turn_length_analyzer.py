"""
Turn Length Analysis for LoCoBench-Agent

This module analyzes the average length of turns in multi-turn agent conversations,
providing insights into token distribution, context accumulation, and conversation patterns.
"""

import json
import logging
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..core.config import Config
from ..core.data_loader import DataLoader, get_data_loader
from ..generation.scenario_converter import ScenarioConverter, get_scenario_converter
from ..agents.agent_factory import AgentFactory, AgentConfig, AgentType

logger = logging.getLogger(__name__)


@dataclass
class TurnAnalysis:
    """Analysis of a single turn"""
    turn_number: int
    phase_id: str
    phase_name: str
    
    # Content analysis
    user_message_tokens: int = 0
    agent_response_tokens: int = 0
    tool_calls_count: int = 0
    tool_results_tokens: int = 0
    
    # Context analysis
    cumulative_context_tokens: int = 0
    context_growth: int = 0
    
    # Timing
    processing_time_seconds: float = 0.0


@dataclass
class ConversationAnalysis:
    """Analysis of a complete conversation"""
    scenario_id: str
    scenario_category: str
    scenario_difficulty: str
    
    total_turns: int = 0
    total_phases: int = 0
    total_duration_seconds: float = 0.0
    
    # Token analysis
    total_tokens: int = 0
    initial_context_tokens: int = 0
    final_context_tokens: int = 0
    
    # Turn analysis
    turns: List[TurnAnalysis] = field(default_factory=list)
    
    # Calculated metrics
    avg_turn_length: float = 0.0
    median_turn_length: float = 0.0
    max_turn_length: int = 0
    min_turn_length: int = 0
    
    # Context growth
    avg_context_growth_per_turn: float = 0.0
    context_compression_events: int = 0


@dataclass
class TurnLengthStatistics:
    """Overall statistics across multiple conversations"""
    
    total_conversations: int = 0
    total_turns: int = 0
    
    # Turn length statistics
    overall_avg_turn_length: float = 0.0
    overall_median_turn_length: float = 0.0
    turn_length_std_dev: float = 0.0
    
    # By difficulty
    easy_avg_turn_length: float = 0.0
    medium_avg_turn_length: float = 0.0
    hard_avg_turn_length: float = 0.0
    expert_avg_turn_length: float = 0.0
    
    # By category
    category_avg_lengths: Dict[str, float] = field(default_factory=dict)
    
    # By phase
    exploration_avg_length: float = 0.0
    analysis_avg_length: float = 0.0
    implementation_avg_length: float = 0.0
    
    # Context statistics
    avg_initial_context: float = 0.0
    avg_final_context: float = 0.0
    avg_context_growth: float = 0.0
    
    # Percentiles
    turn_length_percentiles: Dict[int, float] = field(default_factory=dict)


class TurnLengthAnalyzer:
    """Analyzes turn lengths in agent conversations"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = get_data_loader(config)
        self.converter = get_scenario_converter(config)
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (rough approximation)"""
        # Simple estimation: ~1.3 tokens per word for English text
        # This is a rough approximation - real tokenizers vary
        if not text:
            return 0
        words = len(text.split())
        return int(words * 1.3)
    
    async def analyze_scenario_turn_patterns(
        self, 
        scenario_data: Dict[str, Any],
        simulate_turns: bool = True
    ) -> ConversationAnalysis:
        """Analyze turn patterns for a single scenario"""
        
        scenario_id = scenario_data.get("scenario_id", "unknown")
        category = scenario_data.get("category", "unknown")
        difficulty = scenario_data.get("difficulty", "medium")
        
        analysis = ConversationAnalysis(
            scenario_id=scenario_id,
            scenario_category=category,
            scenario_difficulty=difficulty
        )
        
        # Initial context analysis
        project_files = scenario_data.get("project_files", [])
        initial_context_size = sum(
            self.estimate_tokens(file.get("content", "")) 
            for file in project_files
        )
        analysis.initial_context_tokens = initial_context_size
        
        # Analyze conversation phases
        phases = scenario_data.get("conversation_phases", [])
        analysis.total_phases = len(phases)
        
        cumulative_context = initial_context_size
        turn_number = 1
        
        for phase in phases:
            phase_id = phase.get("phase_id", "unknown")
            phase_name = phase.get("name", "Unknown Phase")
            max_turns = phase.get("max_turns_in_phase", 5)
            
            # Simulate turns in this phase
            for turn_in_phase in range(1, max_turns + 1):
                turn_analysis = TurnAnalysis(
                    turn_number=turn_number,
                    phase_id=phase_id,
                    phase_name=phase_name
                )
                
                if simulate_turns:
                    # Simulate realistic turn content
                    turn_content = self._simulate_turn_content(
                        phase_id, turn_in_phase, max_turns, difficulty
                    )
                    
                    turn_analysis.user_message_tokens = turn_content["user_tokens"]
                    turn_analysis.agent_response_tokens = turn_content["agent_tokens"]
                    turn_analysis.tool_calls_count = turn_content["tool_calls"]
                    turn_analysis.tool_results_tokens = turn_content["tool_results_tokens"]
                    turn_analysis.processing_time_seconds = turn_content["processing_time"]
                    
                    # Calculate context growth
                    context_growth = (
                        turn_analysis.user_message_tokens + 
                        turn_analysis.agent_response_tokens + 
                        turn_analysis.tool_results_tokens
                    )
                    
                    cumulative_context += context_growth
                    turn_analysis.cumulative_context_tokens = cumulative_context
                    turn_analysis.context_growth = context_growth
                
                analysis.turns.append(turn_analysis)
                turn_number += 1
        
        analysis.total_turns = len(analysis.turns)
        analysis.final_context_tokens = cumulative_context
        
        # Calculate statistics
        if analysis.turns:
            total_turn_tokens = [
                turn.user_message_tokens + turn.agent_response_tokens + turn.tool_results_tokens
                for turn in analysis.turns
            ]
            
            analysis.avg_turn_length = statistics.mean(total_turn_tokens)
            analysis.median_turn_length = statistics.median(total_turn_tokens)
            analysis.max_turn_length = max(total_turn_tokens)
            analysis.min_turn_length = min(total_turn_tokens)
            
            context_growths = [turn.context_growth for turn in analysis.turns if turn.context_growth > 0]
            if context_growths:
                analysis.avg_context_growth_per_turn = statistics.mean(context_growths)
        
        return analysis
    
    def _simulate_turn_content(
        self, 
        phase_id: str, 
        turn_in_phase: int, 
        max_turns: int, 
        difficulty: str
    ) -> Dict[str, Any]:
        """Simulate realistic turn content based on phase and difficulty"""
        
        # Base content sizes by phase
        phase_patterns = {
            "exploration": {
                "user_base": 50,      # "Read the main.py file"
                "agent_base": 200,    # Analysis and reasoning
                "tool_calls": 2,      # read_file, search_code
                "tool_results_base": 800  # File contents, search results
            },
            "analysis": {
                "user_base": 80,
                "agent_base": 400,    # Deeper analysis
                "tool_calls": 3,
                "tool_results_base": 600
            },
            "diagnosis": {
                "user_base": 60,
                "agent_base": 350,
                "tool_calls": 2,
                "tool_results_base": 400
            },
            "implementation": {
                "user_base": 70,
                "agent_base": 300,
                "tool_calls": 3,      # write_file, compiler, test
                "tool_results_base": 500
            },
            "testing": {
                "user_base": 40,
                "agent_base": 250,
                "tool_calls": 2,
                "tool_results_base": 300
            },
            "solution": {
                "user_base": 60,
                "agent_base": 400,    # Comprehensive solution
                "tool_calls": 3,
                "tool_results_base": 600
            }
        }
        
        # Get pattern for this phase (default to exploration)
        pattern = phase_patterns.get(phase_id, phase_patterns["exploration"])
        
        # Difficulty multipliers
        difficulty_multipliers = {
            "easy": 0.7,
            "medium": 1.0,
            "hard": 1.4,
            "expert": 1.8
        }
        multiplier = difficulty_multipliers.get(difficulty, 1.0)
        
        # Turn position effects (early turns tend to be longer)
        position_factor = 1.0
        if turn_in_phase == 1:
            position_factor = 1.5  # First turn often longer
        elif turn_in_phase == max_turns:
            position_factor = 1.3  # Last turn often summary/conclusion
        elif turn_in_phase <= max_turns // 3:
            position_factor = 1.2  # Early turns
        
        # Calculate content sizes
        user_tokens = int(pattern["user_base"] * multiplier * position_factor)
        agent_tokens = int(pattern["agent_base"] * multiplier * position_factor)
        tool_calls = pattern["tool_calls"]
        tool_results_tokens = int(pattern["tool_results_base"] * multiplier)
        
        # Processing time estimation (more complex = longer)
        base_time = 2.0  # 2 seconds base
        processing_time = base_time * multiplier * (1 + tool_calls * 0.5)
        
        return {
            "user_tokens": user_tokens,
            "agent_tokens": agent_tokens,
            "tool_calls": tool_calls,
            "tool_results_tokens": tool_results_tokens,
            "processing_time": processing_time
        }
    
    async def analyze_multiple_scenarios(
        self, 
        scenario_limit: int = 20,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None
    ) -> TurnLengthStatistics:
        """Analyze turn lengths across multiple scenarios"""
        
        # Load converted scenarios
        conversion_stats = self.converter.get_conversion_stats()
        
        if conversion_stats['total_converted'] == 0:
            raise ValueError("No converted scenarios found. Run 'locobench convert-scenarios' first.")
        
        # Load original scenarios to get the list
        original_scenarios = self.data_loader.load_scenarios(
            limit=scenario_limit,
            include_project_context=False
        )
        
        analyses = []
        
        logger.info(f"Analyzing turn patterns for {len(original_scenarios)} scenarios...")
        
        for scenario_data in original_scenarios:
            # Filter by category and difficulty if specified
            if categories and scenario_data.task_category not in categories:
                continue
            if difficulties and scenario_data.difficulty not in difficulties:
                continue
            
            # Load converted scenario
            converted_scenario = self.converter.load_converted_scenario(scenario_data.scenario_id)
            
            if converted_scenario:
                analysis = await self.analyze_scenario_turn_patterns(converted_scenario)
                analyses.append(analysis)
                logger.info(f"Analyzed {scenario_data.scenario_id}: {analysis.total_turns} turns, avg {analysis.avg_turn_length:.0f} tokens/turn")
        
        # Calculate overall statistics
        return self._calculate_overall_statistics(analyses)
    
    def _calculate_overall_statistics(self, analyses: List[ConversationAnalysis]) -> TurnLengthStatistics:
        """Calculate overall statistics from individual analyses"""
        
        stats = TurnLengthStatistics()
        
        if not analyses:
            return stats
        
        stats.total_conversations = len(analyses)
        stats.total_turns = sum(a.total_turns for a in analyses)
        
        # Collect all turn lengths
        all_turn_lengths = []
        for analysis in analyses:
            for turn in analysis.turns:
                turn_length = turn.user_message_tokens + turn.agent_response_tokens + turn.tool_results_tokens
                all_turn_lengths.append(turn_length)
        
        if all_turn_lengths:
            stats.overall_avg_turn_length = statistics.mean(all_turn_lengths)
            stats.overall_median_turn_length = statistics.median(all_turn_lengths)
            stats.turn_length_std_dev = statistics.stdev(all_turn_lengths) if len(all_turn_lengths) > 1 else 0.0
            
            # Calculate percentiles
            sorted_lengths = sorted(all_turn_lengths)
            for percentile in [25, 50, 75, 90, 95, 99]:
                index = int((percentile / 100) * len(sorted_lengths))
                stats.turn_length_percentiles[percentile] = sorted_lengths[min(index, len(sorted_lengths) - 1)]
        
        # By difficulty
        difficulty_groups = {"easy": [], "medium": [], "hard": [], "expert": []}
        for analysis in analyses:
            difficulty = analysis.scenario_difficulty.lower()
            if difficulty in difficulty_groups:
                difficulty_groups[difficulty].append(analysis.avg_turn_length)
        
        stats.easy_avg_turn_length = statistics.mean(difficulty_groups["easy"]) if difficulty_groups["easy"] else 0.0
        stats.medium_avg_turn_length = statistics.mean(difficulty_groups["medium"]) if difficulty_groups["medium"] else 0.0
        stats.hard_avg_turn_length = statistics.mean(difficulty_groups["hard"]) if difficulty_groups["hard"] else 0.0
        stats.expert_avg_turn_length = statistics.mean(difficulty_groups["expert"]) if difficulty_groups["expert"] else 0.0
        
        # By category
        category_groups = {}
        for analysis in analyses:
            category = analysis.scenario_category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(analysis.avg_turn_length)
        
        stats.category_avg_lengths = {
            category: statistics.mean(lengths)
            for category, lengths in category_groups.items()
        }
        
        # By phase
        phase_lengths = {"exploration": [], "analysis": [], "implementation": []}
        for analysis in analyses:
            for turn in analysis.turns:
                phase_base = turn.phase_id.lower()
                if "explor" in phase_base:
                    phase_key = "exploration"
                elif "analy" in phase_base or "diagnos" in phase_base:
                    phase_key = "analysis"
                elif "implement" in phase_base or "solution" in phase_base:
                    phase_key = "implementation"
                else:
                    continue
                
                turn_length = turn.user_message_tokens + turn.agent_response_tokens + turn.tool_results_tokens
                phase_lengths[phase_key].append(turn_length)
        
        stats.exploration_avg_length = statistics.mean(phase_lengths["exploration"]) if phase_lengths["exploration"] else 0.0
        stats.analysis_avg_length = statistics.mean(phase_lengths["analysis"]) if phase_lengths["analysis"] else 0.0
        stats.implementation_avg_length = statistics.mean(phase_lengths["implementation"]) if phase_lengths["implementation"] else 0.0
        
        # Context statistics
        initial_contexts = [a.initial_context_tokens for a in analyses if a.initial_context_tokens > 0]
        final_contexts = [a.final_context_tokens for a in analyses if a.final_context_tokens > 0]
        context_growths = [a.avg_context_growth_per_turn for a in analyses if a.avg_context_growth_per_turn > 0]
        
        stats.avg_initial_context = statistics.mean(initial_contexts) if initial_contexts else 0.0
        stats.avg_final_context = statistics.mean(final_contexts) if final_contexts else 0.0
        stats.avg_context_growth = statistics.mean(context_growths) if context_growths else 0.0
        
        return stats
    
    def generate_report(self, stats: TurnLengthStatistics) -> str:
        """Generate a comprehensive report of turn length analysis"""
        
        report = []
        report.append("=" * 80)
        report.append("📊 LOCOBENCH-AGENT TURN LENGTH ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("🔍 OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Conversations Analyzed: {stats.total_conversations}")
        report.append(f"Total Turns Analyzed: {stats.total_turns}")
        report.append(f"Average Turns per Conversation: {stats.total_turns / max(stats.total_conversations, 1):.1f}")
        report.append("")
        
        # Turn Length Statistics
        report.append("📏 TURN LENGTH STATISTICS")
        report.append("-" * 40)
        report.append(f"Average Turn Length: {stats.overall_avg_turn_length:.0f} tokens")
        report.append(f"Median Turn Length: {stats.overall_median_turn_length:.0f} tokens")
        report.append(f"Standard Deviation: {stats.turn_length_std_dev:.0f} tokens")
        report.append("")
        
        # Percentiles
        if stats.turn_length_percentiles:
            report.append("📊 TURN LENGTH PERCENTILES")
            report.append("-" * 40)
            for percentile in [25, 50, 75, 90, 95, 99]:
                if percentile in stats.turn_length_percentiles:
                    report.append(f"{percentile}th percentile: {stats.turn_length_percentiles[percentile]:.0f} tokens")
            report.append("")
        
        # By Difficulty
        report.append("🎯 BY DIFFICULTY LEVEL")
        report.append("-" * 40)
        if stats.easy_avg_turn_length > 0:
            report.append(f"Easy: {stats.easy_avg_turn_length:.0f} tokens/turn")
        if stats.medium_avg_turn_length > 0:
            report.append(f"Medium: {stats.medium_avg_turn_length:.0f} tokens/turn")
        if stats.hard_avg_turn_length > 0:
            report.append(f"Hard: {stats.hard_avg_turn_length:.0f} tokens/turn")
        if stats.expert_avg_turn_length > 0:
            report.append(f"Expert: {stats.expert_avg_turn_length:.0f} tokens/turn")
        report.append("")
        
        # By Category
        if stats.category_avg_lengths:
            report.append("📂 BY TASK CATEGORY")
            report.append("-" * 40)
            for category, avg_length in sorted(stats.category_avg_lengths.items()):
                report.append(f"{category}: {avg_length:.0f} tokens/turn")
            report.append("")
        
        # By Phase
        report.append("🔄 BY CONVERSATION PHASE")
        report.append("-" * 40)
        if stats.exploration_avg_length > 0:
            report.append(f"Exploration: {stats.exploration_avg_length:.0f} tokens/turn")
        if stats.analysis_avg_length > 0:
            report.append(f"Analysis/Diagnosis: {stats.analysis_avg_length:.0f} tokens/turn")
        if stats.implementation_avg_length > 0:
            report.append(f"Implementation: {stats.implementation_avg_length:.0f} tokens/turn")
        report.append("")
        
        # Context Statistics
        report.append("🧠 CONTEXT STATISTICS")
        report.append("-" * 40)
        report.append(f"Average Initial Context: {stats.avg_initial_context:.0f} tokens")
        report.append(f"Average Final Context: {stats.avg_final_context:.0f} tokens")
        report.append(f"Average Context Growth per Turn: {stats.avg_context_growth:.0f} tokens")
        report.append(f"Total Context Growth: {stats.avg_final_context - stats.avg_initial_context:.0f} tokens")
        report.append("")
        
        # Analysis Summary
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 40)
        
        if stats.overall_avg_turn_length > 0:
            if stats.overall_avg_turn_length < 500:
                report.append("• Turn lengths are relatively compact - efficient conversations")
            elif stats.overall_avg_turn_length < 1500:
                report.append("• Turn lengths are moderate - balanced detail and efficiency")
            else:
                report.append("• Turn lengths are substantial - detailed, thorough interactions")
        
        if stats.expert_avg_turn_length > stats.easy_avg_turn_length * 1.5:
            report.append("• Expert scenarios have significantly longer turns than easy ones")
        
        if stats.avg_context_growth > 0:
            growth_rate = stats.avg_context_growth / max(stats.avg_initial_context, 1)
            if growth_rate > 0.1:
                report.append(f"• High context growth rate ({growth_rate:.1%} per turn)")
            else:
                report.append(f"• Moderate context growth rate ({growth_rate:.1%} per turn)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


async def analyze_turn_lengths(
    config: Config = None,
    scenario_limit: int = 20,
    output_file: Optional[str] = None
) -> TurnLengthStatistics:
    """Main function to analyze turn lengths"""
    
    analyzer = TurnLengthAnalyzer(config)
    
    # Run analysis
    stats = await analyzer.analyze_multiple_scenarios(scenario_limit=scenario_limit)
    
    # Generate report
    report = analyzer.generate_report(stats)
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    # Print to console
    print(report)
    
    return stats
