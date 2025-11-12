"""
Agent Comparison and Analysis Framework for LoCoBench-Agent

This module provides comprehensive comparison and analysis capabilities
for evaluating multiple LLM agents across different scenarios and metrics.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..evaluation.agent_metrics import AgentEvaluationResults, MetricCategory

logger = logging.getLogger(__name__)


class ComparisonMode(Enum):
    """Different modes for agent comparison"""
    HEAD_TO_HEAD = "head_to_head"  # Direct pairwise comparison
    LEADERBOARD = "leaderboard"    # Ranking all agents
    CATEGORY_ANALYSIS = "category_analysis"  # Analysis by metric categories
    SCENARIO_ANALYSIS = "scenario_analysis"  # Analysis by scenarios
    TREND_ANALYSIS = "trend_analysis"  # Performance trends over time


class StatisticalTest(Enum):
    """Statistical tests for significance analysis"""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"


@dataclass
class ComparisonConfig:
    """Configuration for agent comparison analysis"""
    
    # Comparison modes
    modes: List[ComparisonMode] = field(default_factory=lambda: [ComparisonMode.LEADERBOARD])
    
    # Statistical analysis
    enable_statistical_tests: bool = True
    significance_level: float = 0.05
    preferred_tests: List[StatisticalTest] = field(default_factory=lambda: [StatisticalTest.T_TEST, StatisticalTest.WILCOXON])
    
    # Visualization
    generate_charts: bool = True
    chart_formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    
    # Output settings
    save_detailed_comparisons: bool = True
    save_summary_tables: bool = True
    generate_html_report: bool = True
    
    # Analysis depth
    include_metric_breakdown: bool = True
    include_scenario_breakdown: bool = True
    include_cost_analysis: bool = True
    include_performance_analysis: bool = True


@dataclass
class AgentComparisonResult:
    """Results from comparing two agents"""
    
    agent_a: str
    agent_b: str
    
    # Overall comparison
    overall_winner: str
    overall_score_diff: float
    overall_p_value: Optional[float] = None
    
    # Category comparisons
    category_winners: Dict[str, str] = field(default_factory=dict)
    category_score_diffs: Dict[str, float] = field(default_factory=dict)
    category_p_values: Dict[str, Optional[float]] = field(default_factory=dict)
    
    # Detailed metrics
    metric_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    cost_comparison: Dict[str, float] = field(default_factory=dict)
    time_comparison: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "overall_winner": self.overall_winner,
            "overall_score_diff": self.overall_score_diff,
            "overall_p_value": self.overall_p_value,
            "category_winners": self.category_winners,
            "category_score_diffs": self.category_score_diffs,
            "category_p_values": self.category_p_values,
            "metric_comparisons": self.metric_comparisons,
            "cost_comparison": self.cost_comparison,
            "time_comparison": self.time_comparison
        }


@dataclass
class LeaderboardEntry:
    """Entry in the agent leaderboard"""
    
    rank: int
    agent_name: str
    overall_score: float
    category_scores: Dict[str, float]
    total_evaluations: int
    avg_cost: float
    avg_duration: float
    win_rate: float  # Against other agents
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "agent_name": self.agent_name,
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "total_evaluations": self.total_evaluations,
            "avg_cost": self.avg_cost,
            "avg_duration": self.avg_duration,
            "win_rate": self.win_rate
        }


@dataclass
class ComparisonAnalysisResult:
    """Complete analysis results from agent comparison"""
    
    analysis_id: str
    timestamp: datetime
    config: ComparisonConfig
    
    # Input data
    agents_analyzed: List[str]
    total_evaluations: int
    
    # Comparison results
    pairwise_comparisons: List[AgentComparisonResult] = field(default_factory=list)
    leaderboard: List[LeaderboardEntry] = field(default_factory=list)
    
    # Analysis results
    category_analysis: Dict[str, Any] = field(default_factory=dict)
    scenario_analysis: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance insights
    cost_analysis: Dict[str, Any] = field(default_factory=dict)
    efficiency_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "agents_analyzed": self.agents_analyzed,
            "total_evaluations": self.total_evaluations,
            "pairwise_comparisons": [comp.to_dict() for comp in self.pairwise_comparisons],
            "leaderboard": [entry.to_dict() for entry in self.leaderboard],
            "category_analysis": self.category_analysis,
            "scenario_analysis": self.scenario_analysis,
            "statistical_analysis": self.statistical_analysis,
            "cost_analysis": self.cost_analysis,
            "efficiency_analysis": self.efficiency_analysis
        }


class AgentComparisonFramework:
    """
    Comprehensive framework for comparing and analyzing LLM agents
    
    This framework provides:
    1. Pairwise agent comparisons with statistical testing
    2. Multi-agent leaderboards and rankings
    3. Category-based performance analysis
    4. Scenario-based performance analysis
    5. Cost and efficiency analysis
    6. Statistical significance testing
    7. Visualization and reporting
    """
    
    def __init__(self, config: ComparisonConfig = None):
        self.config = config or ComparisonConfig()
        
        # Analysis results storage
        self.comparison_results: List[ComparisonAnalysisResult] = []
        
        logger.info("AgentComparisonFramework initialized")
    
    async def compare_agents(
        self,
        evaluation_results: List[AgentEvaluationResults],
        analysis_id: str = None
    ) -> ComparisonAnalysisResult:
        """
        Perform comprehensive comparison analysis of agents
        
        Args:
            evaluation_results: List of agent evaluation results
            analysis_id: Optional analysis ID
            
        Returns:
            Complete comparison analysis results
        """
        
        if not analysis_id:
            analysis_id = f"comparison_{int(datetime.now().timestamp())}"
        
        logger.info(f"Starting agent comparison analysis: {analysis_id}")
        
        # Initialize analysis result
        analysis_result = ComparisonAnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            config=self.config,
            agents_analyzed=list(set(r.agent_name for r in evaluation_results)),
            total_evaluations=len(evaluation_results)
        )
        
        # Group results by agent
        agent_results = self._group_results_by_agent(evaluation_results)
        
        # Perform different types of analysis based on configuration
        if ComparisonMode.HEAD_TO_HEAD in self.config.modes:
            analysis_result.pairwise_comparisons = await self._perform_pairwise_comparisons(
                agent_results
            )
        
        if ComparisonMode.LEADERBOARD in self.config.modes:
            analysis_result.leaderboard = await self._create_leaderboard(
                agent_results, analysis_result.pairwise_comparisons
            )
        
        if ComparisonMode.CATEGORY_ANALYSIS in self.config.modes:
            analysis_result.category_analysis = await self._perform_category_analysis(
                agent_results
            )
        
        if ComparisonMode.SCENARIO_ANALYSIS in self.config.modes:
            analysis_result.scenario_analysis = await self._perform_scenario_analysis(
                evaluation_results
            )
        
        # Statistical analysis
        if self.config.enable_statistical_tests:
            analysis_result.statistical_analysis = await self._perform_statistical_analysis(
                agent_results
            )
        
        # Performance analysis
        if self.config.include_cost_analysis:
            analysis_result.cost_analysis = await self._perform_cost_analysis(agent_results)
        
        if self.config.include_performance_analysis:
            analysis_result.efficiency_analysis = await self._perform_efficiency_analysis(
                agent_results
            )
        
        # Store results
        self.comparison_results.append(analysis_result)
        
        logger.info(f"Agent comparison analysis completed: {analysis_id}")
        
        return analysis_result
    
    def _group_results_by_agent(
        self,
        evaluation_results: List[AgentEvaluationResults]
    ) -> Dict[str, List[AgentEvaluationResults]]:
        """Group evaluation results by agent name"""
        
        agent_results = {}
        
        for result in evaluation_results:
            if result.agent_name not in agent_results:
                agent_results[result.agent_name] = []
            agent_results[result.agent_name].append(result)
        
        return agent_results
    
    async def _perform_pairwise_comparisons(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> List[AgentComparisonResult]:
        """Perform pairwise comparisons between all agents"""
        
        logger.info("Performing pairwise agent comparisons")
        
        comparisons = []
        agent_names = list(agent_results.keys())
        
        for i, agent_a in enumerate(agent_names):
            for agent_b in agent_names[i + 1:]:
                comparison = await self._compare_two_agents(
                    agent_a, agent_results[agent_a],
                    agent_b, agent_results[agent_b]
                )
                comparisons.append(comparison)
        
        logger.info(f"Completed {len(comparisons)} pairwise comparisons")
        
        return comparisons
    
    async def _compare_two_agents(
        self,
        agent_a: str,
        results_a: List[AgentEvaluationResults],
        agent_b: str,
        results_b: List[AgentEvaluationResults]
    ) -> AgentComparisonResult:
        """Compare two specific agents"""
        
        # Calculate average scores
        scores_a = [r.overall_score for r in results_a]
        scores_b = [r.overall_score for r in results_b]
        
        avg_score_a = statistics.mean(scores_a)
        avg_score_b = statistics.mean(scores_b)
        
        # Determine winner
        overall_winner = agent_a if avg_score_a > avg_score_b else agent_b
        overall_score_diff = abs(avg_score_a - avg_score_b)
        
        # Statistical test
        overall_p_value = None
        if self.config.enable_statistical_tests and len(scores_a) > 1 and len(scores_b) > 1:
            overall_p_value = self._perform_statistical_test(scores_a, scores_b)
        
        # Category comparisons
        category_winners = {}
        category_score_diffs = {}
        category_p_values = {}
        
        for category in MetricCategory:
            cat_scores_a = [
                r.category_scores.get(category, 0) for r in results_a
                if category in r.category_scores
            ]
            cat_scores_b = [
                r.category_scores.get(category, 0) for r in results_b
                if category in r.category_scores
            ]
            
            if cat_scores_a and cat_scores_b:
                avg_cat_a = statistics.mean(cat_scores_a)
                avg_cat_b = statistics.mean(cat_scores_b)
                
                category_winners[category.value] = agent_a if avg_cat_a > avg_cat_b else agent_b
                category_score_diffs[category.value] = abs(avg_cat_a - avg_cat_b)
                
                if self.config.enable_statistical_tests and len(cat_scores_a) > 1 and len(cat_scores_b) > 1:
                    category_p_values[category.value] = self._perform_statistical_test(
                        cat_scores_a, cat_scores_b
                    )
        
        # Cost and time comparisons
        costs_a = [r.total_cost for r in results_a]
        costs_b = [r.total_cost for r in results_b]
        times_a = [r.session_duration for r in results_a]
        times_b = [r.session_duration for r in results_b]
        
        cost_comparison = {
            "agent_a_avg": statistics.mean(costs_a),
            "agent_b_avg": statistics.mean(costs_b),
            "cost_efficient_agent": agent_a if statistics.mean(costs_a) < statistics.mean(costs_b) else agent_b
        }
        
        time_comparison = {
            "agent_a_avg": statistics.mean(times_a),
            "agent_b_avg": statistics.mean(times_b),
            "faster_agent": agent_a if statistics.mean(times_a) < statistics.mean(times_b) else agent_b
        }
        
        return AgentComparisonResult(
            agent_a=agent_a,
            agent_b=agent_b,
            overall_winner=overall_winner,
            overall_score_diff=overall_score_diff,
            overall_p_value=overall_p_value,
            category_winners=category_winners,
            category_score_diffs=category_score_diffs,
            category_p_values=category_p_values,
            cost_comparison=cost_comparison,
            time_comparison=time_comparison
        )
    
    def _perform_statistical_test(self, scores_a: List[float], scores_b: List[float]) -> float:
        """Perform statistical test between two score distributions"""
        
        try:
            # Use scipy if available, otherwise return None
            from scipy import stats
            
            # Default to t-test
            if StatisticalTest.T_TEST in self.config.preferred_tests:
                _, p_value = stats.ttest_ind(scores_a, scores_b)
                return p_value
            elif StatisticalTest.MANN_WHITNEY in self.config.preferred_tests:
                _, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
                return p_value
            else:
                # Fallback to t-test
                _, p_value = stats.ttest_ind(scores_a, scores_b)
                return p_value
        
        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return None
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return None
    
    async def _create_leaderboard(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]],
        pairwise_comparisons: List[AgentComparisonResult]
    ) -> List[LeaderboardEntry]:
        """Create a comprehensive leaderboard"""
        
        logger.info("Creating agent leaderboard")
        
        leaderboard_entries = []
        
        for agent_name, results in agent_results.items():
            # Calculate averages
            overall_scores = [r.overall_score for r in results]
            costs = [r.total_cost for r in results]
            durations = [r.session_duration for r in results]
            
            # Category averages
            category_scores = {}
            for category in MetricCategory:
                cat_scores = [
                    r.category_scores.get(category, 0) for r in results
                    if category in r.category_scores
                ]
                if cat_scores:
                    category_scores[category.value] = statistics.mean(cat_scores)
            
            # Calculate win rate from pairwise comparisons
            wins = 0
            total_comparisons = 0
            
            for comparison in pairwise_comparisons:
                if comparison.agent_a == agent_name or comparison.agent_b == agent_name:
                    total_comparisons += 1
                    if comparison.overall_winner == agent_name:
                        wins += 1
            
            win_rate = wins / total_comparisons if total_comparisons > 0 else 0.0
            
            entry = LeaderboardEntry(
                rank=0,  # Will be set after sorting
                agent_name=agent_name,
                overall_score=statistics.mean(overall_scores),
                category_scores=category_scores,
                total_evaluations=len(results),
                avg_cost=statistics.mean(costs),
                avg_duration=statistics.mean(durations),
                win_rate=win_rate
            )
            
            leaderboard_entries.append(entry)
        
        # Sort by overall score (descending)
        leaderboard_entries.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard_entries):
            entry.rank = i + 1
        
        logger.info(f"Leaderboard created with {len(leaderboard_entries)} agents")
        
        return leaderboard_entries
    
    async def _perform_category_analysis(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Any]:
        """Perform analysis by metric categories"""
        
        logger.info("Performing category-based analysis")
        
        category_analysis = {}
        
        for category in MetricCategory:
            category_data = {}
            
            # Collect scores for each agent in this category
            agent_category_scores = {}
            
            for agent_name, results in agent_results.items():
                cat_scores = [
                    r.category_scores.get(category, 0) for r in results
                    if category in r.category_scores
                ]
                
                if cat_scores:
                    agent_category_scores[agent_name] = {
                        "average": statistics.mean(cat_scores),
                        "min": min(cat_scores),
                        "max": max(cat_scores),
                        "std": statistics.stdev(cat_scores) if len(cat_scores) > 1 else 0
                    }
            
            # Rank agents in this category
            if agent_category_scores:
                ranked_agents = sorted(
                    agent_category_scores.items(),
                    key=lambda x: x[1]["average"],
                    reverse=True
                )
                
                category_data["rankings"] = [
                    {
                        "rank": i + 1,
                        "agent_name": agent_name,
                        "average_score": data["average"]
                    }
                    for i, (agent_name, data) in enumerate(ranked_agents)
                ]
                
                category_data["best_agent"] = ranked_agents[0][0]
                category_data["agent_scores"] = agent_category_scores
                category_data["overall_stats"] = {
                    "category_average": statistics.mean([
                        data["average"] for data in agent_category_scores.values()
                    ]),
                    "score_range": max([
                        data["average"] for data in agent_category_scores.values()
                    ]) - min([
                        data["average"] for data in agent_category_scores.values()
                    ])
                }
            
            category_analysis[category.value] = category_data
        
        return category_analysis
    
    async def _perform_scenario_analysis(
        self,
        evaluation_results: List[AgentEvaluationResults]
    ) -> Dict[str, Any]:
        """Perform analysis by scenarios"""
        
        logger.info("Performing scenario-based analysis")
        
        # Group results by scenario
        scenario_results = {}
        
        for result in evaluation_results:
            if result.scenario_id not in scenario_results:
                scenario_results[result.scenario_id] = []
            scenario_results[result.scenario_id].append(result)
        
        scenario_analysis = {}
        
        for scenario_id, results in scenario_results.items():
            # Agent performance on this scenario
            agent_scores = {}
            
            for result in results:
                agent_scores[result.agent_name] = result.overall_score
            
            # Rank agents for this scenario
            ranked_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            
            scenario_analysis[scenario_id] = {
                "total_agents": len(agent_scores),
                "average_score": statistics.mean(agent_scores.values()),
                "best_agent": ranked_agents[0][0] if ranked_agents else None,
                "worst_agent": ranked_agents[-1][0] if ranked_agents else None,
                "score_range": max(agent_scores.values()) - min(agent_scores.values()) if agent_scores else 0,
                "agent_rankings": [
                    {
                        "rank": i + 1,
                        "agent_name": agent_name,
                        "score": score
                    }
                    for i, (agent_name, score) in enumerate(ranked_agents)
                ]
            }
        
        return scenario_analysis
    
    async def _perform_statistical_analysis(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        
        logger.info("Performing statistical analysis")
        
        try:
            from scipy import stats
            
            # Collect all scores for ANOVA
            all_scores = []
            agent_labels = []
            
            for agent_name, results in agent_results.items():
                scores = [r.overall_score for r in results]
                all_scores.extend(scores)
                agent_labels.extend([agent_name] * len(scores))
            
            # Convert to numpy arrays if possible
            try:
                import numpy as np
                score_groups = [
                    [r.overall_score for r in results] 
                    for results in agent_results.values()
                ]
                
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*score_groups)
                
                statistical_analysis = {
                    "anova_test": {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < self.config.significance_level
                    },
                    "significance_level": self.config.significance_level,
                    "interpretation": {
                        "significant_differences": p_value < self.config.significance_level,
                        "recommendation": "Agents show significantly different performance" if p_value < self.config.significance_level else "No significant differences between agents"
                    }
                }
                
            except ImportError:
                statistical_analysis = {
                    "message": "Advanced statistical analysis requires numpy and scipy"
                }
            
        except ImportError:
            statistical_analysis = {
                "message": "Statistical analysis requires scipy package"
            }
        
        return statistical_analysis
    
    async def _perform_cost_analysis(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Any]:
        """Perform cost analysis across agents"""
        
        logger.info("Performing cost analysis")
        
        cost_analysis = {}
        
        for agent_name, results in agent_results.items():
            costs = [r.total_cost for r in results]
            scores = [r.overall_score for r in results]
            
            cost_analysis[agent_name] = {
                "average_cost": statistics.mean(costs),
                "total_cost": sum(costs),
                "cost_per_point": statistics.mean(costs) / statistics.mean(scores) if statistics.mean(scores) > 0 else float('inf'),
                "cost_efficiency_rank": 0  # Will be calculated after all agents
            }
        
        # Rank by cost efficiency (cost per point)
        ranked_efficiency = sorted(
            cost_analysis.items(),
            key=lambda x: x[1]["cost_per_point"]
        )
        
        for i, (agent_name, data) in enumerate(ranked_efficiency):
            cost_analysis[agent_name]["cost_efficiency_rank"] = i + 1
        
        # Overall cost statistics
        all_costs = [data["average_cost"] for data in cost_analysis.values()]
        
        cost_analysis["overall_stats"] = {
            "most_cost_efficient": ranked_efficiency[0][0],
            "least_cost_efficient": ranked_efficiency[-1][0],
            "average_cost_across_agents": statistics.mean(all_costs),
            "cost_range": max(all_costs) - min(all_costs)
        }
        
        return cost_analysis
    
    async def _perform_efficiency_analysis(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Any]:
        """Perform efficiency analysis (score per time/cost)"""
        
        logger.info("Performing efficiency analysis")
        
        efficiency_analysis = {}
        
        for agent_name, results in agent_results.items():
            scores = [r.overall_score for r in results]
            durations = [r.session_duration for r in results]
            costs = [r.total_cost for r in results]
            
            avg_score = statistics.mean(scores)
            avg_duration = statistics.mean(durations)
            avg_cost = statistics.mean(costs)
            
            efficiency_analysis[agent_name] = {
                "score_per_second": avg_score / avg_duration if avg_duration > 0 else 0,
                "score_per_dollar": avg_score / avg_cost if avg_cost > 0 else 0,
                "time_efficiency_rank": 0,
                "cost_efficiency_rank": 0,
                "overall_efficiency_score": 0  # Composite score
            }
        
        # Rank by time efficiency
        time_ranked = sorted(
            efficiency_analysis.items(),
            key=lambda x: x[1]["score_per_second"],
            reverse=True
        )
        
        for i, (agent_name, data) in enumerate(time_ranked):
            efficiency_analysis[agent_name]["time_efficiency_rank"] = i + 1
        
        # Rank by cost efficiency
        cost_ranked = sorted(
            efficiency_analysis.items(),
            key=lambda x: x[1]["score_per_dollar"],
            reverse=True
        )
        
        for i, (agent_name, data) in enumerate(cost_ranked):
            efficiency_analysis[agent_name]["cost_efficiency_rank"] = i + 1
        
        # Calculate composite efficiency score
        num_agents = len(efficiency_analysis)
        
        for agent_name, data in efficiency_analysis.items():
            # Lower rank is better, so invert
            time_score = (num_agents - data["time_efficiency_rank"] + 1) / num_agents
            cost_score = (num_agents - data["cost_efficiency_rank"] + 1) / num_agents
            
            data["overall_efficiency_score"] = (time_score + cost_score) / 2
        
        # Overall efficiency statistics
        efficiency_scores = [data["overall_efficiency_score"] for data in efficiency_analysis.values()]
        
        efficiency_analysis["overall_stats"] = {
            "most_efficient_agent": max(efficiency_analysis.items(), key=lambda x: x[1]["overall_efficiency_score"])[0],
            "average_efficiency": statistics.mean(efficiency_scores),
            "efficiency_distribution": {
                "high_efficiency": len([s for s in efficiency_scores if s >= 0.7]),
                "medium_efficiency": len([s for s in efficiency_scores if 0.3 <= s < 0.7]),
                "low_efficiency": len([s for s in efficiency_scores if s < 0.3])
            }
        }
        
        return efficiency_analysis
    
    async def save_comparison_results(
        self,
        analysis_result: ComparisonAnalysisResult,
        output_directory: Path
    ) -> Dict[str, Path]:
        """Save comparison results to files"""
        
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save main analysis result
        main_file = output_directory / f"{analysis_result.analysis_id}_comparison.json"
        with open(main_file, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2)
        
        saved_files["main_analysis"] = main_file
        
        # Save leaderboard
        if analysis_result.leaderboard:
            leaderboard_file = output_directory / f"{analysis_result.analysis_id}_leaderboard.json"
            with open(leaderboard_file, 'w') as f:
                json.dump([entry.to_dict() for entry in analysis_result.leaderboard], f, indent=2)
            
            saved_files["leaderboard"] = leaderboard_file
        
        # Save pairwise comparisons
        if analysis_result.pairwise_comparisons:
            pairwise_file = output_directory / f"{analysis_result.analysis_id}_pairwise.json"
            with open(pairwise_file, 'w') as f:
                json.dump([comp.to_dict() for comp in analysis_result.pairwise_comparisons], f, indent=2)
            
            saved_files["pairwise_comparisons"] = pairwise_file
        
        # Generate HTML report if enabled
        if self.config.generate_html_report:
            html_file = output_directory / f"{analysis_result.analysis_id}_report.html"
            html_content = await self._generate_html_comparison_report(analysis_result)
            
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            saved_files["html_report"] = html_file
        
        logger.info(f"Comparison results saved to: {output_directory}")
        
        return saved_files
    
    async def _generate_html_comparison_report(
        self,
        analysis_result: ComparisonAnalysisResult
    ) -> str:
        """Generate HTML report for comparison results"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Comparison Report - {analysis_result.analysis_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .leaderboard {{ width: 100%; border-collapse: collapse; }}
                .leaderboard th, .leaderboard td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .leaderboard th {{ background-color: #f2f2f2; }}
                .rank-1 {{ background-color: #ffd700; }}
                .rank-2 {{ background-color: #c0c0c0; }}
                .rank-3 {{ background-color: #cd7f32; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Agent Comparison Report</h1>
                <p><strong>Analysis ID:</strong> {analysis_result.analysis_id}</p>
                <p><strong>Timestamp:</strong> {analysis_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Agents Analyzed:</strong> {len(analysis_result.agents_analyzed)}</p>
                <p><strong>Total Evaluations:</strong> {analysis_result.total_evaluations}</p>
            </div>
        """
        
        # Leaderboard section
        if analysis_result.leaderboard:
            html_template += """
            <div class="section">
                <h2>Agent Leaderboard</h2>
                <table class="leaderboard">
                    <tr>
                        <th>Rank</th>
                        <th>Agent</th>
                        <th>Overall Score</th>
                        <th>Win Rate</th>
                        <th>Avg Cost</th>
                        <th>Avg Duration</th>
                    </tr>
            """
            
            for entry in analysis_result.leaderboard:
                rank_class = ""
                if entry.rank == 1:
                    rank_class = "rank-1"
                elif entry.rank == 2:
                    rank_class = "rank-2"
                elif entry.rank == 3:
                    rank_class = "rank-3"
                
                html_template += f"""
                    <tr class="{rank_class}">
                        <td>{entry.rank}</td>
                        <td>{entry.agent_name}</td>
                        <td>{entry.overall_score:.2f}</td>
                        <td>{entry.win_rate:.1%}</td>
                        <td>${entry.avg_cost:.4f}</td>
                        <td>{entry.avg_duration:.1f}s</td>
                    </tr>
                """
            
            html_template += """
                </table>
            </div>
            """
        
        # Statistical analysis section
        if analysis_result.statistical_analysis:
            html_template += f"""
            <div class="section">
                <h2>Statistical Analysis</h2>
                <p>Statistical analysis results and significance testing.</p>
                <pre>{json.dumps(analysis_result.statistical_analysis, indent=2)}</pre>
            </div>
            """
        
        html_template += """
        </body>
        </html>
        """
        
        return html_template
    
    def get_comparison_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a specific comparison analysis"""
        
        for result in self.comparison_results:
            if result.analysis_id == analysis_id:
                return {
                    "analysis_id": analysis_id,
                    "agents_analyzed": result.agents_analyzed,
                    "total_evaluations": result.total_evaluations,
                    "leaderboard_winner": result.leaderboard[0].agent_name if result.leaderboard else None,
                    "analysis_timestamp": result.timestamp.isoformat()
                }
        
        return None
    
    def list_comparison_results(self) -> List[Dict[str, Any]]:
        """List all comparison analysis results"""
        
        return [
            {
                "analysis_id": result.analysis_id,
                "timestamp": result.timestamp.isoformat(),
                "agents_analyzed": result.agents_analyzed,
                "total_evaluations": result.total_evaluations
            }
            for result in self.comparison_results
        ]
