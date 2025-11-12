"""
Agent Evaluator for LoCoBench-Agent

This module provides comprehensive evaluation pipeline for agent sessions,
integrating multi-turn conversation analysis, tool usage assessment,
and the complete 25-metric evaluation framework.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .agent_metrics import AgentMetricsCalculator, AgentEvaluationResults, MetricCategory
from .bias_free_evaluator import BiasFreEvaluator, BiasFreEvaluationResult
from ..core.agent_session import AgentSession
from ..generation.interactive_scenario_generator import InteractiveScenario
from ..agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for agent evaluation"""
    
    # Evaluation scope
    evaluate_all_metrics: bool = True
    metric_categories: List[MetricCategory] = field(default_factory=lambda: list(MetricCategory))
    
    # Performance settings
    max_concurrent_evaluations: int = 3
    evaluation_timeout_seconds: int = 3600  # 1 hour
    
    # Output settings
    save_detailed_logs: bool = True
    save_conversation_transcripts: bool = True
    save_tool_usage_reports: bool = True
    
    # Quality assurance
    require_minimum_turns: int = 5
    require_tool_usage: bool = False
    validate_session_completion: bool = True
    
    # NEW: Cursor-aligned features (ENABLED BY DEFAULT)
    enable_semantic_search: bool = True
    enable_enhanced_summarization: bool = True
    initial_context_mode: str = "minimal"  # "full", "minimal", or "empty"
    
    # NEW: Bias-free evaluation (ENABLED BY DEFAULT)
    use_bias_free_evaluator: bool = True
    enable_human_validation: bool = False


@dataclass
class AgentEvaluationSession:
    """Represents a complete agent evaluation session"""
    
    session_id: str
    agent_name: str
    scenario: InteractiveScenario
    
    # Session execution
    agent_session: Optional[AgentSession] = None
    session_result: Optional[Dict[str, Any]] = None
    
    # Evaluation results
    evaluation_results: Optional[AgentEvaluationResults] = None
    
    # Metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "scenario_id": self.scenario.scenario_id,
            "scenario_title": self.scenario.title,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "evaluation_results": self.evaluation_results.to_dict() if self.evaluation_results else None,
            "error_message": self.error_message
        }


class AgentEvaluator:
    """
    Comprehensive evaluator for LLM agents in multi-turn software development scenarios
    
    This evaluator orchestrates the complete evaluation pipeline:
    1. Agent session execution
    2. Multi-turn conversation analysis
    3. Tool usage assessment
    4. 25-metric evaluation
    5. Comparative analysis
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.metrics_calculator = AgentMetricsCalculator()
        
        # NEW: Bias-free evaluator
        if self.config.use_bias_free_evaluator:
            self.bias_free_evaluator = BiasFreEvaluator(
                enable_human_validation=self.config.enable_human_validation
            )
            logger.info("🎯 Bias-free evaluator enabled - using revolutionary metric system")
        else:
            self.bias_free_evaluator = None
            logger.info("⚠️ Using legacy biased metrics - consider enabling bias_free_evaluator")
        
        # Active evaluations
        self.active_evaluations: Dict[str, AgentEvaluationSession] = {}
        
        # Results storage
        self.evaluation_results: List[AgentEvaluationResults] = []
        
        # Session data storage for conversation saving
        self.stored_session_data: Dict[str, Dict[str, Any]] = {}
        
        logger.info("AgentEvaluator initialized")
    
    async def evaluate_agent(
        self,
        agent: BaseAgent,
        scenario: InteractiveScenario,
        session_id: str = None
    ) -> AgentEvaluationResults:
        """
        Evaluate a single agent on a single scenario
        
        Args:
            agent: The agent to evaluate
            scenario: The interactive scenario to run
            session_id: Optional session ID (auto-generated if not provided)
            
        Returns:
            Complete evaluation results
        """
        
        if not session_id:
            session_id = f"eval_{agent.name}_{scenario.scenario_id}_{int(time.time())}"
        
        logger.info(f"Starting agent evaluation: {session_id}")
        
        # Create evaluation session
        eval_session = AgentEvaluationSession(
            session_id=session_id,
            agent_name=agent.name,
            scenario=scenario,
            start_time=datetime.now()
        )
        
        self.active_evaluations[session_id] = eval_session
        
        try:
            eval_session.status = "running"
            
            # Execute agent session
            eval_session.agent_session = await self._create_agent_session(
                agent, scenario, session_id
            )
            
            eval_session.session_result = await self._execute_agent_session(
                eval_session.agent_session
            )
            
            # Evaluate session results
            eval_session.evaluation_results = await self._evaluate_session_results(
                eval_session.session_result,
                scenario.to_dict(),
                agent.name
            )
            
            eval_session.status = "completed"
            eval_session.end_time = datetime.now()
            
            logger.info(f"Agent evaluation completed: {session_id}")
            
            # Store results
            self.evaluation_results.append(eval_session.evaluation_results)
            
            # Store session data for conversation saving
            self._store_session_data(eval_session)
            
            return eval_session.evaluation_results
            
        except Exception as e:
            logger.error(f"Agent evaluation failed: {session_id} - {e}")
            
            eval_session.status = "failed"
            eval_session.error_message = str(e)
            eval_session.end_time = datetime.now()
            
            raise
        
        finally:
            # Clean up active evaluation
            if session_id in self.active_evaluations:
                del self.active_evaluations[session_id]
    
    async def evaluate_multiple_agents(
        self,
        agents: List[BaseAgent],
        scenarios: List[InteractiveScenario],
        comparison_mode: bool = True
    ) -> List[AgentEvaluationResults]:
        """
        Evaluate multiple agents across multiple scenarios
        
        Args:
            agents: List of agents to evaluate
            scenarios: List of scenarios to run
            comparison_mode: Whether to enable cross-agent comparison
            
        Returns:
            List of evaluation results for all agent-scenario combinations
        """
        
        logger.info(f"Starting multi-agent evaluation: {len(agents)} agents × {len(scenarios)} scenarios")
        
        # Create all evaluation tasks
        evaluation_tasks = []
        
        for agent in agents:
            for scenario in scenarios:
                session_id = f"multi_eval_{agent.name}_{scenario.scenario_id}_{int(time.time())}"
                
                task = self.evaluate_agent(agent, scenario, session_id)
                evaluation_tasks.append(task)
        
        # Execute evaluations with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)
        
        async def evaluate_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Run all evaluations
        results = await asyncio.gather(
            *[evaluate_with_semaphore(task) for task in evaluation_tasks],
            return_exceptions=True
        )
        
        # Filter out exceptions and collect successful results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed: {result}")
                failed_count += 1
            else:
                successful_results.append(result)
        
        logger.info(f"Multi-agent evaluation completed: {len(successful_results)} successful, {failed_count} failed")
        
        return successful_results
    
    async def _create_agent_session(
        self,
        agent: BaseAgent,
        scenario: InteractiveScenario,
        session_id: str
    ) -> AgentSession:
        """Create an agent session for evaluation"""
        
        from ..core.agent_session import SessionConfig
        from ..core.tool_registry import get_tool_registry
        
        # Get available tools based on scenario
        tool_registry = get_tool_registry()
        available_tools = []
        
        for tool_name in scenario.available_tools:
            tool = tool_registry.get_tool(tool_name)
            if tool:
                # Create a copy of the tool to prevent cross-session contamination
                tool_copy = self._create_tool_copy(tool)
                # NOTE: Don't set context yet - will do after AgentSession initializes
                # so tools get the enriched context with _all_files_for_tools
                available_tools.append(tool_copy)
        
        # Create session configuration
        session_config = SessionConfig(
            max_turns=scenario.max_turns,
            timeout_seconds=scenario.max_duration_minutes * 60,
            save_checkpoints=True,
            checkpoint_interval=5,
            enable_semantic_search=self.config.enable_semantic_search,
            enable_enhanced_summarization=self.config.enable_enhanced_summarization,
            initial_context_mode=self.config.initial_context_mode
        )
        
        # Create agent session (this will populate scenario_context with _all_files_for_tools)
        agent_session = AgentSession(
            session_id=session_id,
            agent=agent,
            scenario_context=scenario.to_dict(),
            conversation_phases=scenario.conversation_phases,
            available_tools=available_tools,
            config=session_config
        )
        
        # CRITICAL: Now set tool context AFTER AgentSession has enriched scenario_context
        # This ensures tools get _all_files_for_tools for on-demand retrieval
        for tool in available_tools:
            if hasattr(tool, 'set_context'):
                tool.set_context(agent_session.scenario_context)
                logger.debug(f"Set enriched context on tool: {tool.name}")
        
        return agent_session
    
    def _create_tool_copy(self, tool):
        """Create a copy of a tool to prevent cross-session contamination"""
        import copy
        try:
            # Create a deep copy of the tool to ensure complete isolation
            tool_copy = copy.deepcopy(tool)
            # Give the copy a unique name to help with debugging
            tool_copy.name = f"{tool.name}_copy_{id(tool_copy)}"
            logger.debug(f"Created tool copy: {tool_copy.name} from {tool.name}")
            return tool_copy
        except Exception as e:
            logger.warning(f"Could not create deep copy of tool {tool.name}, using shallow copy: {e}")
            # Fallback to shallow copy if deep copy fails
            tool_copy = copy.copy(tool)
            tool_copy.name = f"{tool.name}_copy_{id(tool_copy)}"
            return tool_copy
    
    async def _execute_agent_session(self, agent_session: AgentSession) -> Dict[str, Any]:
        """Execute the agent session and return results"""
        
        logger.info(f"Executing agent session: {agent_session.session_id}")
        
        # Execute the conversation
        session_result = await agent_session.execute_conversation()
        
        logger.info(f"Session completed: {agent_session.session_id} - Status: {session_result['status']}")
        
        return session_result
    
    async def _evaluate_session_results(
        self,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any],
        agent_name: str
    ) -> AgentEvaluationResults:
        """Evaluate session results using the comprehensive metrics"""
        
        logger.info(f"Evaluating session results for agent: {agent_name}")
        
        # NEW: Use bias-free evaluator if enabled
        if self.bias_free_evaluator:
            logger.info("🎯 Using bias-free evaluation system")
            
            # Extract solution code from session result
            solution_code = session_result.get("modified_files", {})
            
            # Run bias-free evaluation
            bias_free_result = await self.bias_free_evaluator.evaluate_agent_performance(
                scenario=scenario_context,
                solution_code=solution_code,
                session_result=session_result
            )
            
            # Convert bias-free result to legacy format for compatibility
            evaluation_results = self._convert_bias_free_to_legacy_format(
                bias_free_result, session_result, scenario_context, agent_name
            )
            
            logger.info(f"🎯 Bias-free evaluation completed. LCBA-Comp: {bias_free_result.lcba_scores.comprehension_score:.2f}, LCBA-Eff: {bias_free_result.lcba_scores.efficiency_score:.2f}")
        else:
            logger.info("⚠️ Using legacy biased metrics")
            
            # Use the legacy metrics calculator to compute all 28 metrics
            evaluation_results = await self.metrics_calculator.evaluate_agent_session(
                session_result=session_result,
                scenario_context=scenario_context,
                agent_name=agent_name
            )
            
            logger.info(f"Legacy evaluation completed. Overall score: {evaluation_results.overall_score:.2f}/5.0")
        
        return evaluation_results
    
    def _convert_bias_free_to_legacy_format(
        self,
        bias_free_result: BiasFreEvaluationResult,
        session_result: Dict[str, Any],
        scenario_context: Dict[str, Any],
        agent_name: str
    ) -> AgentEvaluationResults:
        """Convert bias-free evaluation result to legacy format for compatibility"""
        
        # Create legacy metric results from bias-free results
        from .agent_metrics import AgentMetricResult
        
        legacy_metrics = []
        for metric_name, metric_result in bias_free_result.metric_results.items():
            # Determine category based on LCBA alignment
            lcba_alignment = metric_result.details.get("lcba_alignment", "other")
            if lcba_alignment == "comprehension":
                category = MetricCategory.SOFTWARE_ENGINEERING  # Most comprehension metrics are SE
            elif lcba_alignment == "efficiency":
                category = MetricCategory.AGENT_INTERACTION  # Most efficiency metrics are AI
            elif lcba_alignment == "functional_correctness":
                category = MetricCategory.FUNCTIONAL_CORRECTNESS
            elif lcba_alignment == "code_quality":
                category = MetricCategory.CODE_QUALITY
            else:
                category = MetricCategory.SOFTWARE_ENGINEERING  # Default
            
            legacy_metric = AgentMetricResult(
                metric_name=metric_name,
                category=category,
                score=metric_result.score,
                max_score=5.0,
                weight=1.0,  # Will be normalized later
                details=metric_result.details,
                explanation=f"Bias-free metric - {lcba_alignment.upper()} aligned"
            )
            legacy_metrics.append(legacy_metric)
        
        # Create legacy evaluation results
        evaluation_results = AgentEvaluationResults(
            agent_name=agent_name,
            scenario_id=scenario_context.get("scenario_id", scenario_context.get("id", "unknown")),
            session_id=session_result.get("session_id", "unknown"),
            metric_results=legacy_metrics,
            category_scores={
                MetricCategory.AGENT_INTERACTION: bias_free_result.lcba_scores.efficiency_score,
                MetricCategory.SOFTWARE_ENGINEERING: bias_free_result.lcba_scores.comprehension_score,
                MetricCategory.FUNCTIONAL_CORRECTNESS: bias_free_result.lcba_scores.comprehension_score,
                MetricCategory.CODE_QUALITY: bias_free_result.lcba_scores.comprehension_score,
                MetricCategory.LONG_CONTEXT_UTILIZATION: bias_free_result.lcba_scores.comprehension_score
            },
            overall_score=bias_free_result.lcba_scores.overall_score,
            max_overall_score=5.0,
            
            # NEW: LCBA scores
            lcba_comprehension=bias_free_result.lcba_scores.comprehension_score,
            lcba_efficiency=bias_free_result.lcba_scores.efficiency_score,
            
            # Session metadata
            total_turns=session_result.get("total_turns", 0),
            session_duration=session_result.get("session_duration", 0.0),
            conversation_history=session_result.get("conversation_history", []),
            tool_usage_log=session_result.get("tool_usage_log", []),
            modified_files=session_result.get("modified_files", {}),
            error_log=session_result.get("error_log", []),
            session_status=session_result.get("status", "unknown"),
            completed_phases=session_result.get("completed_phases", 0),
            total_phases=session_result.get("total_phases", 1),
            error_rate=len(session_result.get("error_log", [])) / max(session_result.get("total_turns", 1), 1),
            phases_completed=session_result.get("completed_phases", 0),
            # BUGFIX: Include scenario context to preserve conversation phases
            scenario_context=scenario_context,
            evaluation_timestamp=datetime.now().isoformat(),
            evaluator_version="2.0_bias_free"
        )
        
        return evaluation_results
    
    async def save_evaluation_results(
        self,
        results: List[AgentEvaluationResults],
        output_directory: Path
    ) -> Dict[str, Path]:
        """Save evaluation results to files"""
        
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save individual results
        for result in results:
            result_file = output_directory / f"{result.agent_name}_{result.scenario_id}_results.json"
            
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            saved_files[f"{result.agent_name}_{result.scenario_id}"] = result_file
        
        # Save summary report
        summary_file = output_directory / "evaluation_summary.json"
        summary_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "agents_evaluated": list(set(r.agent_name for r in results)),
            "scenarios_evaluated": list(set(r.scenario_id for r in results)),
            "average_scores": self._calculate_average_scores(results),
            "results": [r.to_dict() for r in results]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        saved_files["summary"] = summary_file
        
        logger.info(f"Evaluation results saved to: {output_directory}")
        
        return saved_files
    
    def _calculate_average_scores(self, results: List[AgentEvaluationResults]) -> Dict[str, float]:
        """Calculate average scores across all evaluations"""
        
        if not results:
            return {}
        
        # Overall scores
        overall_scores = [r.overall_score for r in results]
        
        # Category scores
        category_averages = {}
        for category in MetricCategory:
            category_scores = []
            for result in results:
                if category in result.category_scores:
                    category_scores.append(result.category_scores[category])
            
            if category_scores:
                category_averages[category.value] = sum(category_scores) / len(category_scores)
        
        return {
            "overall_average": sum(overall_scores) / len(overall_scores),
            "category_averages": category_averages,
            "score_distribution": {
                "min": min(overall_scores),
                "max": max(overall_scores),
                "median": sorted(overall_scores)[len(overall_scores) // 2]
            }
        }
    
    def get_evaluation_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific evaluation"""
        
        if session_id in self.active_evaluations:
            return self.active_evaluations[session_id].to_dict()
        
        return None
    
    def list_active_evaluations(self) -> List[Dict[str, Any]]:
        """List all active evaluations"""
        
        return [eval_session.to_dict() for eval_session in self.active_evaluations.values()]
    
    async def cancel_evaluation(self, session_id: str) -> bool:
        """Cancel an active evaluation"""
        
        if session_id in self.active_evaluations:
            eval_session = self.active_evaluations[session_id]
            
            # Cancel the agent session if it's running
            if eval_session.agent_session:
                # This would need to be implemented in AgentSession
                # await eval_session.agent_session.cancel()
                pass
            
            eval_session.status = "cancelled"
            eval_session.end_time = datetime.now()
            
            del self.active_evaluations[session_id]
            
            logger.info(f"Evaluation cancelled: {session_id}")
            return True
        
        return False
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about completed evaluations"""
        
        if not self.evaluation_results:
            return {"message": "No evaluations completed yet"}
        
        total_evaluations = len(self.evaluation_results)
        agents_evaluated = set(r.agent_name for r in self.evaluation_results)
        scenarios_evaluated = set(r.scenario_id for r in self.evaluation_results)
        
        # Score statistics
        overall_scores = [r.overall_score for r in self.evaluation_results]
        
        # Performance statistics
        session_durations = [r.session_duration for r in self.evaluation_results]
        total_costs = [r.total_cost for r in self.evaluation_results]
        turn_counts = [r.total_turns for r in self.evaluation_results]
        
        return {
            "total_evaluations": total_evaluations,
            "unique_agents": len(agents_evaluated),
            "unique_scenarios": len(scenarios_evaluated),
            "score_statistics": {
                "average": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
                "median": sorted(overall_scores)[len(overall_scores) // 2]
            },
            "performance_statistics": {
                "average_duration": sum(session_durations) / len(session_durations),
                "average_cost": sum(total_costs) / len(total_costs),
                "average_turns": sum(turn_counts) / len(turn_counts)
            },
            "agents_evaluated": list(agents_evaluated),
            "scenarios_evaluated": list(scenarios_evaluated)
        }
    
    def _store_session_data(self, eval_session: AgentEvaluationSession):
        """Store session data for conversation saving"""
        
        if not eval_session.agent_session or not eval_session.session_result:
            return
        
        session_id = eval_session.session_id
        agent_session = eval_session.agent_session
        session_result = eval_session.session_result
        
        # Extract conversation data from the agent session
        conversation_data = {
            "session_id": session_id,
            "conversation_history": session_result.get("conversation_history", []),
            "phase_history": getattr(agent_session, 'phase_history', []),
            "tool_usage_log": getattr(agent_session, 'tool_usage_log', []),
            "error_log": getattr(agent_session, 'error_log', []),
            "human_interventions": getattr(agent_session, 'human_interventions', []),
            "session_metadata": {
                "status": session_result.get("status", "unknown"),
                "total_turns": session_result.get("total_turns", 0),
                "duration_seconds": session_result.get("duration_seconds", 0),
                "start_time": agent_session.session_start_time.isoformat() if hasattr(agent_session, 'session_start_time') else None,
                "end_time": agent_session.session_end_time.isoformat() if hasattr(agent_session, 'session_end_time') and agent_session.session_end_time else None
            }
        }
        
        # Store the session data
        self.stored_session_data[session_id] = conversation_data
        
        logger.info(f"Stored session data for: {session_id}")
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get stored session data for conversation saving"""
        return self.stored_session_data.get(session_id)
    
    def get_all_session_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored session data"""
        return self.stored_session_data.copy()
