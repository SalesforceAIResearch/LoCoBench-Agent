"""
Multi-Turn Evaluation Pipeline for LoCoBench-Agent

This module provides the core pipeline for executing and evaluating
multi-turn agent conversations with comprehensive analysis and reporting.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from ..agents.base_agent import BaseAgent
from ..generation.interactive_scenario_generator import InteractiveScenario, ConversationPhase
from ..evaluation.agent_evaluator import AgentEvaluator, EvaluationConfig, AgentEvaluationResults
from ..core.agent_session import AgentSession, SessionConfig
from ..core.tool_registry import get_tool_registry

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages of the multi-turn evaluation pipeline"""
    INITIALIZATION = "initialization"
    SCENARIO_LOADING = "scenario_loading"
    AGENT_SETUP = "agent_setup"
    SESSION_EXECUTION = "session_execution"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the multi-turn evaluation pipeline"""
    
    # Execution settings
    max_concurrent_sessions: int = 5
    session_timeout_minutes: int = 60
    enable_checkpointing: bool = True
    
    # Evaluation settings
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Analysis settings
    enable_comparative_analysis: bool = True
    enable_statistical_analysis: bool = True
    enable_trend_analysis: bool = True
    
    # Output settings
    output_directory: Optional[Path] = None
    save_raw_conversations: bool = True
    save_tool_usage_logs: bool = True
    save_performance_metrics: bool = True
    generate_html_report: bool = True
    
    # Quality control
    minimum_conversation_turns: int = 3
    require_successful_completion: bool = False
    validate_tool_usage: bool = True


@dataclass
class PipelineResult:
    """Results from a complete pipeline execution"""
    
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Configuration
    config: PipelineConfig = None
    
    # Execution status
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    status: str = "running"  # running, completed, failed, cancelled
    error_message: Optional[str] = None
    
    # Results
    agent_results: List[AgentEvaluationResults] = field(default_factory=list)
    comparative_analysis: Optional[Dict[str, Any]] = None
    statistical_analysis: Optional[Dict[str, Any]] = None
    
    # Session data for conversation saving
    session_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    total_duration_seconds: float = 0.0
    
    # Output files
    output_files: Dict[str, Path] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_stage": self.current_stage.value,
            "status": self.status,
            "error_message": self.error_message,
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "failed_sessions": self.failed_sessions,
            "total_duration_seconds": self.total_duration_seconds,
            "output_files": {k: str(v) for k, v in self.output_files.items()},
            "agent_results_summary": [
                {
                    "agent_name": result.agent_name,
                    "scenario_id": result.scenario_id,
                    "overall_score": result.overall_score,
                    "total_turns": result.total_turns
                }
                for result in self.agent_results
            ] if self.agent_results else []
        }


class MultiTurnEvaluationPipeline:
    """
    Comprehensive pipeline for multi-turn agent evaluation
    
    This pipeline orchestrates the complete evaluation process:
    1. Scenario loading and validation
    2. Agent initialization and configuration
    3. Multi-turn conversation execution
    4. Comprehensive evaluation using all 25 metrics
    5. Comparative and statistical analysis
    6. Report generation
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Core components
        self.agent_evaluator = AgentEvaluator(self.config.evaluation_config)
        
        # Pipeline state
        self.active_pipelines: Dict[str, PipelineResult] = {}
        self.completed_pipelines: List[PipelineResult] = []
        
        logger.info("MultiTurnEvaluationPipeline initialized")
    
    async def run_evaluation_pipeline(
        self,
        agents: List[BaseAgent],
        scenarios: List[InteractiveScenario],
        pipeline_id: str = None
    ) -> PipelineResult:
        """
        Run the complete multi-turn evaluation pipeline
        
        Args:
            agents: List of agents to evaluate
            scenarios: List of scenarios to run
            pipeline_id: Optional pipeline ID (auto-generated if not provided)
            
        Returns:
            Complete pipeline results
        """
        
        if not pipeline_id:
            pipeline_id = f"pipeline_{int(time.time())}"
        
        logger.info(f"Starting multi-turn evaluation pipeline: {pipeline_id}")
        
        # Initialize pipeline result
        pipeline_result = PipelineResult(
            pipeline_id=pipeline_id,
            start_time=datetime.now(),
            config=self.config,
            total_sessions=len(agents) * len(scenarios)
        )
        
        self.active_pipelines[pipeline_id] = pipeline_result
        
        try:
            # Stage 1: Initialization
            await self._stage_initialization(pipeline_result)
            
            # Stage 2: Scenario Loading
            await self._stage_scenario_loading(pipeline_result, scenarios)
            
            # Stage 3: Agent Setup
            await self._stage_agent_setup(pipeline_result, agents)
            
            # Stage 4: Session Execution
            await self._stage_session_execution(pipeline_result, agents, scenarios)
            
            # Stage 5: Evaluation
            await self._stage_evaluation(pipeline_result)
            
            # Stage 6: Analysis
            await self._stage_analysis(pipeline_result)
            
            # Stage 7: Reporting
            await self._stage_reporting(pipeline_result)
            
            # Complete pipeline
            pipeline_result.current_stage = PipelineStage.COMPLETED
            pipeline_result.status = "completed"
            pipeline_result.end_time = datetime.now()
            pipeline_result.total_duration_seconds = (
                pipeline_result.end_time - pipeline_result.start_time
            ).total_seconds()
            
            logger.info(f"Pipeline completed successfully: {pipeline_id}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {pipeline_id} - {e}")
            
            pipeline_result.current_stage = PipelineStage.FAILED
            pipeline_result.status = "failed"
            pipeline_result.error_message = str(e)
            pipeline_result.end_time = datetime.now()
            
            raise
        
        finally:
            # Move to completed pipelines
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
            
            self.completed_pipelines.append(pipeline_result)
        
        return pipeline_result
    
    async def _stage_initialization(self, pipeline_result: PipelineResult):
        """Initialize pipeline components and setup"""
        
        logger.info(f"Pipeline stage: Initialization - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.INITIALIZATION
        
        # Setup output directory
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
        else:
            output_dir = Path(f"output/pipeline_{pipeline_result.pipeline_id}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline_result.output_files["base_directory"] = output_dir
        
        # Create subdirectories (removed redundant analysis/ and reports/ directories)
        (output_dir / "sessions").mkdir(exist_ok=True)
        (output_dir / "evaluations").mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized with output directory: {output_dir}")
    
    async def _stage_scenario_loading(
        self,
        pipeline_result: PipelineResult,
        scenarios: List[InteractiveScenario]
    ):
        """Load and validate scenarios"""
        
        logger.info(f"Pipeline stage: Scenario Loading - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.SCENARIO_LOADING
        
        # Validate scenarios
        valid_scenarios = []
        
        for scenario in scenarios:
            if self._validate_scenario(scenario):
                valid_scenarios.append(scenario)
            else:
                logger.warning(f"Invalid scenario skipped: {scenario.scenario_id}")
        
        if not valid_scenarios:
            raise ValueError("No valid scenarios found")
        
        # Save scenario metadata
        scenario_metadata = {
            "total_scenarios": len(valid_scenarios),
            "scenario_details": [
                {
                    "scenario_id": s.scenario_id,
                    "title": s.title,
                    "difficulty": s.difficulty.value,
                    "category": s.category.value,
                    "max_turns": s.max_turns,
                    "available_tools": s.available_tools
                }
                for s in valid_scenarios
            ]
        }
        
        metadata_file = pipeline_result.output_files["base_directory"] / "scenario_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(scenario_metadata, f, indent=2)
        
        pipeline_result.output_files["scenario_metadata"] = metadata_file
        
        logger.info(f"Loaded {len(valid_scenarios)} valid scenarios")
    
    async def _stage_agent_setup(
        self,
        pipeline_result: PipelineResult,
        agents: List[BaseAgent]
    ):
        """Setup and validate agents"""
        
        logger.info(f"Pipeline stage: Agent Setup - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.AGENT_SETUP
        
        # Validate agents
        valid_agents = []
        
        for agent in agents:
            if await self._validate_agent(agent):
                valid_agents.append(agent)
            else:
                logger.warning(f"Invalid agent skipped: {agent.name}")
        
        if not valid_agents:
            raise ValueError("No valid agents found")
        
        # Save agent metadata
        agent_metadata = {
            "total_agents": len(valid_agents),
            "agent_details": [
                {
                    "name": agent.name,
                    "model": getattr(agent, 'model', 'unknown'),
                    "capabilities": agent.capabilities.to_dict() if hasattr(agent, 'capabilities') and agent.capabilities else {}
                }
                for agent in valid_agents
            ]
        }
        
        metadata_file = pipeline_result.output_files["base_directory"] / "agent_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(agent_metadata, f, indent=2)
        
        pipeline_result.output_files["agent_metadata"] = metadata_file
        
        logger.info(f"Setup {len(valid_agents)} valid agents")
    
    async def _stage_session_execution(
        self,
        pipeline_result: PipelineResult,
        agents: List[BaseAgent],
        scenarios: List[InteractiveScenario]
    ):
        """Execute all agent-scenario combinations"""
        
        logger.info(f"Pipeline stage: Session Execution - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.SESSION_EXECUTION
        
        # Execute evaluations using the agent evaluator
        evaluation_results = await self.agent_evaluator.evaluate_multiple_agents(
            agents=agents,
            scenarios=scenarios,
            comparison_mode=self.config.enable_comparative_analysis
        )
        
        # Store results
        pipeline_result.agent_results = evaluation_results
        pipeline_result.successful_sessions = len(evaluation_results)
        pipeline_result.failed_sessions = pipeline_result.total_sessions - pipeline_result.successful_sessions
        
        # Store session data for conversation saving
        pipeline_result.session_data = self.agent_evaluator.get_all_session_data()
        
        logger.info(f"Session execution completed: {pipeline_result.successful_sessions} successful, {pipeline_result.failed_sessions} failed")
        logger.info(f"Stored session data for {len(pipeline_result.session_data)} sessions")
    
    async def _stage_evaluation(self, pipeline_result: PipelineResult):
        """Process evaluation results"""
        
        logger.info(f"Pipeline stage: Evaluation - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.EVALUATION
        
        # Save detailed evaluation results
        eval_dir = pipeline_result.output_files["base_directory"] / "evaluations"
        
        saved_files = await self.agent_evaluator.save_evaluation_results(
            results=pipeline_result.agent_results,
            output_directory=eval_dir
        )
        
        pipeline_result.output_files.update(saved_files)
        
        logger.info(f"Evaluation results saved: {len(saved_files)} files")
    
    async def _stage_analysis(self, pipeline_result: PipelineResult):
        """Perform comparative and statistical analysis"""
        
        logger.info(f"Pipeline stage: Analysis - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.ANALYSIS
        
        analysis_dir = pipeline_result.output_files["base_directory"] / "analysis"
        
        if self.config.enable_comparative_analysis:
            pipeline_result.comparative_analysis = await self._perform_comparative_analysis(
                pipeline_result.agent_results
            )
            
            # Save comparative analysis
            comp_file = analysis_dir / "comparative_analysis.json"
            with open(comp_file, 'w') as f:
                json.dump(pipeline_result.comparative_analysis, f, indent=2)
            
            pipeline_result.output_files["comparative_analysis"] = comp_file
        
        if self.config.enable_statistical_analysis:
            pipeline_result.statistical_analysis = await self._perform_statistical_analysis(
                pipeline_result.agent_results
            )
            
            # Save statistical analysis
            stat_file = analysis_dir / "statistical_analysis.json"
            with open(stat_file, 'w') as f:
                json.dump(pipeline_result.statistical_analysis, f, indent=2)
            
            pipeline_result.output_files["statistical_analysis"] = stat_file
        
        logger.info("Analysis completed")
    
    async def _stage_reporting(self, pipeline_result: PipelineResult):
        """Generate comprehensive reports"""
        
        logger.info(f"Pipeline stage: Reporting - {pipeline_result.pipeline_id}")
        pipeline_result.current_stage = PipelineStage.REPORTING
        
        reports_dir = pipeline_result.output_files["base_directory"] / "reports"
        
        # Generate summary report
        summary_report = await self._generate_summary_report(pipeline_result)
        
        summary_file = reports_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        pipeline_result.output_files["summary_report"] = summary_file
        
        # Generate HTML report if enabled
        if self.config.generate_html_report:
            html_report = await self._generate_html_report(pipeline_result)
            
            html_file = reports_dir / "pipeline_report.html"
            with open(html_file, 'w') as f:
                f.write(html_report)
            
            pipeline_result.output_files["html_report"] = html_file
        
        logger.info("Reporting completed")
    
    def _validate_scenario(self, scenario: InteractiveScenario) -> bool:
        """Validate a scenario for pipeline execution"""
        
        # Check required fields
        if not scenario.scenario_id or not scenario.title:
            return False
        
        # Check conversation phases
        if not scenario.conversation_phases:
            return False
        
        # Check minimum turns
        if scenario.max_turns < self.config.minimum_conversation_turns:
            return False
        
        return True
    
    async def _validate_agent(self, agent: BaseAgent) -> bool:
        """Validate an agent for pipeline execution"""
        
        try:
            # Check basic properties
            if not agent.name:
                return False
            
            # Test basic functionality (simple ping)
            test_message = "Test message for validation"
            response = await agent.process_turn(test_message, [], {})
            
            return response is not None
        
        except Exception as e:
            logger.warning(f"Agent validation failed: {agent.name} - {e}")
            return False
    
    async def _perform_comparative_analysis(
        self,
        results: List[AgentEvaluationResults]
    ) -> Dict[str, Any]:
        """Perform comparative analysis across agents"""
        
        if len(results) < 2:
            return {"message": "Insufficient data for comparative analysis"}
        
        # Group results by agent
        agents_results = {}
        for result in results:
            if result.agent_name not in agents_results:
                agents_results[result.agent_name] = []
            agents_results[result.agent_name].append(result)
        
        # Calculate agent averages
        agent_averages = {}
        for agent_name, agent_results in agents_results.items():
            overall_scores = [r.overall_score for r in agent_results]
            agent_averages[agent_name] = {
                "average_score": sum(overall_scores) / len(overall_scores),
                "min_score": min(overall_scores),
                "max_score": max(overall_scores),
                "total_sessions": len(agent_results)
            }
        
        # Rank agents
        ranked_agents = sorted(
            agent_averages.items(),
            key=lambda x: x[1]["average_score"],
            reverse=True
        )
        
        return {
            "agent_averages": agent_averages,
            "agent_rankings": [
                {
                    "rank": i + 1,
                    "agent_name": agent_name,
                    "average_score": data["average_score"]
                }
                for i, (agent_name, data) in enumerate(ranked_agents)
            ],
            "best_agent": ranked_agents[0][0] if ranked_agents else None,
            "score_spread": max(a["average_score"] for a in agent_averages.values()) - 
                           min(a["average_score"] for a in agent_averages.values()) if agent_averages else 0
        }
    
    async def _perform_statistical_analysis(
        self,
        results: List[AgentEvaluationResults]
    ) -> Dict[str, Any]:
        """Perform statistical analysis of results"""
        
        if not results:
            return {"message": "No data for statistical analysis"}
        
        import statistics
        
        # Overall score statistics
        overall_scores = [r.overall_score for r in results]
        
        # Performance statistics
        session_durations = [r.session_duration for r in results]
        turn_counts = [r.total_turns for r in results]
        costs = [r.total_cost for r in results]
        
        return {
            "overall_score_stats": {
                "mean": statistics.mean(overall_scores),
                "median": statistics.median(overall_scores),
                "stdev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                "min": min(overall_scores),
                "max": max(overall_scores)
            },
            "performance_stats": {
                "average_duration": statistics.mean(session_durations),
                "average_turns": statistics.mean(turn_counts),
                "average_cost": statistics.mean(costs),
                "total_cost": sum(costs)
            },
            "distribution": {
                "high_performers": len([s for s in overall_scores if s >= 4.0]),
                "medium_performers": len([s for s in overall_scores if 3.0 <= s < 4.0]),
                "low_performers": len([s for s in overall_scores if s < 3.0])
            }
        }
    
    async def _generate_summary_report(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        
        return {
            "pipeline_info": pipeline_result.to_dict(),
            "execution_summary": {
                "total_evaluations": len(pipeline_result.agent_results),
                "successful_evaluations": pipeline_result.successful_sessions,
                "failed_evaluations": pipeline_result.failed_sessions,
                "success_rate": pipeline_result.successful_sessions / pipeline_result.total_sessions if pipeline_result.total_sessions > 0 else 0,
                "total_duration_hours": pipeline_result.total_duration_seconds / 3600
            },
            "performance_overview": {
                "average_score": sum(r.overall_score for r in pipeline_result.agent_results) / len(pipeline_result.agent_results) if pipeline_result.agent_results else 0,
                "best_score": max(r.overall_score for r in pipeline_result.agent_results) if pipeline_result.agent_results else 0,
                "worst_score": min(r.overall_score for r in pipeline_result.agent_results) if pipeline_result.agent_results else 0
            },
            "comparative_analysis": pipeline_result.comparative_analysis,
            "statistical_analysis": pipeline_result.statistical_analysis
        }
    
    async def _generate_html_report(self, pipeline_result: PipelineResult) -> str:
        """Generate an HTML report"""
        
        summary = await self._generate_summary_report(pipeline_result)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LoCoBench-Agent Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LoCoBench-Agent Evaluation Report</h1>
                <p><strong>Pipeline ID:</strong> {pipeline_result.pipeline_id}</p>
                <p><strong>Execution Time:</strong> {pipeline_result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Duration:</strong> {pipeline_result.total_duration_seconds / 3600:.2f} hours</p>
            </div>
            
            <div class="section">
                <h2>Execution Summary</h2>
                <div class="metric">
                    <strong>Total Evaluations:</strong> {summary['execution_summary']['total_evaluations']}
                </div>
                <div class="metric">
                    <strong>Success Rate:</strong> {summary['execution_summary']['success_rate']:.1%}
                </div>
                <div class="metric">
                    <strong>Average Score:</strong> {summary['performance_overview']['average_score']:.2f}/5.0
                </div>
            </div>
            
            <div class="section">
                <h2>Agent Results</h2>
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Scenario</th>
                        <th>Score</th>
                        <th>Turns</th>
                        <th>Duration</th>
                    </tr>
        """
        
        for result in pipeline_result.agent_results:
            html_template += f"""
                    <tr>
                        <td>{result.agent_name}</td>
                        <td>{result.scenario_id}</td>
                        <td>{result.overall_score:.2f}</td>
                        <td>{result.total_turns}</td>
                        <td>{result.session_duration:.1f}s</td>
                    </tr>
            """
        
        html_template += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific pipeline"""
        
        if pipeline_id in self.active_pipelines:
            return self.active_pipelines[pipeline_id].to_dict()
        
        # Check completed pipelines
        for pipeline in self.completed_pipelines:
            if pipeline.pipeline_id == pipeline_id:
                return pipeline.to_dict()
        
        return None
    
    def list_active_pipelines(self) -> List[Dict[str, Any]]:
        """List all active pipelines"""
        
        return [pipeline.to_dict() for pipeline in self.active_pipelines.values()]
    
    def list_completed_pipelines(self) -> List[Dict[str, Any]]:
        """List all completed pipelines"""
        
        return [pipeline.to_dict() for pipeline in self.completed_pipelines]
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel an active pipeline"""
        
        if pipeline_id in self.active_pipelines:
            pipeline = self.active_pipelines[pipeline_id]
            pipeline.status = "cancelled"
            pipeline.end_time = datetime.now()
            
            # Move to completed
            del self.active_pipelines[pipeline_id]
            self.completed_pipelines.append(pipeline)
            
            logger.info(f"Pipeline cancelled: {pipeline_id}")
            return True
        
        return False
