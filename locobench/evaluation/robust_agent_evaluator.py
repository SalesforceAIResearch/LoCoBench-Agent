"""
Robust Agent Evaluator - Based on Original LoCoBench Logic

This module provides a robust, crash-safe agent evaluation system that adopts
the proven checkpoint and incremental saving strategies from the original LoCoBench.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import signal
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID, TimeElapsedColumn

from .agent_evaluator import AgentEvaluator, AgentEvaluationResults
from .agent_metrics import AgentMetricsCalculator
from ..core.config import Config
from ..agents.base_agent import BaseAgent
from ..generation.interactive_scenario_generator import InteractiveScenario
from ..core.task import DifficultyLevel

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class AgentEvaluationCheckpoint:
    """Checkpoint state for resumable agent evaluations"""
    started_at: str
    checkpoint_version: str = "2.0"
    agents: List[str] = None  # agent names
    scenarios: List[str] = None  # scenario IDs
    task_categories: Optional[List[str]] = None
    difficulty_levels: Optional[List[str]] = None
    completed_evaluations: Dict[str, List[str]] = None  # agent -> [scenario_ids]
    total_evaluations: int = 0
    completed_count: int = 0
    last_updated: str = ""
    output_file: Optional[str] = None
    
    # Retry tracking for failed scenarios
    failed_attempts: Dict[str, Dict[str, int]] = None  # agent -> {scenario_id: attempt_count}
    max_retry_limit: int = 3  # Lower limit for agent evaluations (more expensive)
    
    # Agent-specific metadata
    conversation_stats: Dict[str, Dict[str, Any]] = None  # agent -> stats
    
    def __post_init__(self):
        if self.completed_evaluations is None:
            self.completed_evaluations = {}
        if self.failed_attempts is None:
            self.failed_attempts = {}
        if self.conversation_stats is None:
            self.conversation_stats = {}


@dataclass
class AgentEvaluationSummary:
    """Summary of agent evaluation results"""
    agent_name: str
    total_scenarios: int
    completed_scenarios: int
    failed_scenarios: int
    
    # Agent-specific metrics
    avg_overall_score: float  # Legacy, for backward compatibility
    avg_conversation_turns: float
    avg_session_duration: float
    
    # Category breakdowns
    category_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    
    # Success rates
    parsing_success_rate: float
    tool_usage_success_rate: float
    phase_completion_rate: float
    
    # LCBA Final Scores (Primary evaluation metrics) - with defaults for backward compatibility
    avg_lcba_comprehension: float = 0.0  # LCBA-Comprehension: Quality, depth, correctness
    avg_lcba_efficiency: float = 0.0  # LCBA-Efficiency: Speed, conciseness, resource optimization


class RobustAgentEvaluator:
    """
    Robust agent evaluator with crash-safe checkpointing and incremental saving
    
    Based on the proven architecture of the original LoCoBench evaluator,
    adapted for multi-turn agent evaluation scenarios.
    """
    
    def __init__(self, config: Config, agent_name: str = None, enable_semantic_search: bool = False, enable_enhanced_summarization: bool = False, initial_context_mode: str = "minimal"):
        self.config = config
        self.initial_context_mode = initial_context_mode
        
        # Configure base evaluator with Cursor-aligned features
        from .agent_evaluator import EvaluationConfig
        eval_config = EvaluationConfig(
            enable_semantic_search=enable_semantic_search,
            enable_enhanced_summarization=enable_enhanced_summarization,
            initial_context_mode=initial_context_mode
        )
        self.base_evaluator = AgentEvaluator(config=eval_config)
        
        self.results: List[AgentEvaluationResults] = []
        self.checkpoint: Optional[AgentEvaluationCheckpoint] = None
        
        # Create intermediate_agent_results directory
        intermediate_dir = Path("intermediate_agent_results")
        intermediate_dir.mkdir(exist_ok=True)
        
        # Agent-specific checkpoint files to avoid conflicts
        if agent_name:
            safe_agent_name = agent_name.replace('-', '_').replace('.', '_').lower()
            self.checkpoint_file = intermediate_dir / f"agent_evaluation_checkpoint_{safe_agent_name}.json"
            self.incremental_file = intermediate_dir / f"agent_incremental_results_{safe_agent_name}.jsonl"
        else:
            self.checkpoint_file = intermediate_dir / "agent_evaluation_checkpoint.json"
            self.incremental_file = intermediate_dir / "agent_incremental_results.jsonl"
        
        self.current_agent = agent_name
        self._interrupted = False
        self._start_time = None
        self._scenario_times = []
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle graceful shutdown on interrupt"""
        console.print("\n🛑 Evaluation interrupted by user", style="bold yellow")
        self._interrupted = True
        
        # Save current state
        if self.checkpoint:
            self._save_checkpoint()
            console.print("💾 Checkpoint saved successfully!", style="green")
        
        # Clean exit message
        console.print("\n✅ Agent evaluation safely interrupted. Resume later with:", style="bold green")
        console.print("   locobench evaluate --mode agent --resume [other-options]", style="cyan")
        
        sys.exit(0)
    
    def _save_checkpoint(self):
        """Save current evaluation state to checkpoint file"""
        if self.checkpoint:
            self.checkpoint.last_updated = datetime.now().isoformat()
            self.checkpoint.completed_count = sum(len(scenarios) for scenarios in self.checkpoint.completed_evaluations.values())
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(asdict(self.checkpoint), f, indent=2)
            logger.info(f"💾 Agent checkpoint saved: {self.checkpoint.completed_count}/{self.checkpoint.total_evaluations} completed")
    
    def _load_checkpoint(self) -> Optional[AgentEvaluationCheckpoint]:
        """Load checkpoint from file if it exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                checkpoint = AgentEvaluationCheckpoint(**data)
                
                # Check if this looks like a crash recovery situation
                if self._detect_crash_recovery(checkpoint):
                    console.print("🔄 Detected potential agent evaluation crash recovery", style="yellow")
                    console.print(f"📊 Last checkpoint: {checkpoint.completed_count}/{checkpoint.total_evaluations} completed")
                    console.print("💡 Tip: Use --resume to continue from where you left off")
                
                logger.info(f"📂 Loaded agent checkpoint: {checkpoint.completed_count}/{checkpoint.total_evaluations} completed")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load agent checkpoint: {e}")
        return None
    
    def _detect_crash_recovery(self, checkpoint: AgentEvaluationCheckpoint) -> bool:
        """Detect if this appears to be a crash recovery situation"""
        try:
            last_update = datetime.fromisoformat(checkpoint.last_updated) if checkpoint.last_updated else None
            if last_update:
                hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
                is_recent = hours_since_update < 24
                is_incomplete = checkpoint.completed_count < checkpoint.total_evaluations
                return is_recent and is_incomplete
        except:
            pass
        return False
    
    def _save_incremental_result(self, result: AgentEvaluationResults):
        """Save individual agent result to incremental file (crash-safe JSONL)"""
        if not result or not result.scenario_id:
            logger.error("Attempted to save invalid agent result - skipping")
            return
        
        try:
            # Convert result to JSON and append as new line
            result_dict = result.to_dict()
            result_json = json.dumps(result_dict)
            
            # Append to JSONL file (atomic operation)
            with open(self.incremental_file, 'a', encoding='utf-8') as f:
                f.write(result_json + '\n')
                f.flush()
            
            logger.debug(f"💾 Saved incremental agent result: {result.agent_name} on {result.scenario_id}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to save incremental agent result for {result.scenario_id}: {e}")
    
    def _load_incremental_results(self) -> List[AgentEvaluationResults]:
        """Load all previously saved incremental results from JSONL format"""
        if not self.incremental_file.exists():
            return []
        
        results = []
        try:
            with open(self.incremental_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        result_data = json.loads(line)
                        
                        # Remove fields that don't exist in the dataclass
                        if 'normalized_overall_score' in result_data:
                            del result_data['normalized_overall_score']
                        
                        # Convert timestamp string back to datetime if needed
                        if 'evaluation_timestamp' in result_data and isinstance(result_data['evaluation_timestamp'], str):
                            result_data['evaluation_timestamp'] = datetime.fromisoformat(result_data['evaluation_timestamp'])
                        
                        # Convert category_scores keys back to MetricCategory enums
                        if 'category_scores' in result_data and isinstance(result_data['category_scores'], dict):
                            from .agent_metrics import MetricCategory
                            category_scores = {}
                            for key, value in result_data['category_scores'].items():
                                try:
                                    category_scores[MetricCategory(key)] = value
                                except ValueError:
                                    # Skip invalid categories
                                    continue
                            result_data['category_scores'] = category_scores
                        
                        # Convert back to AgentEvaluationResults object
                        result = AgentEvaluationResults(**result_data)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Skipping invalid result on line {line_num}: {e}")
            
            logger.info(f"📂 Loaded {len(results)} incremental agent results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load incremental agent results: {e}")
            return []
    
    def _update_checkpoint_completion(self, agent_name: str, scenario_id: str):
        """Mark an agent-scenario combination as completed in checkpoint"""
        if self.checkpoint:
            if agent_name not in self.checkpoint.completed_evaluations:
                self.checkpoint.completed_evaluations[agent_name] = []
            if scenario_id not in self.checkpoint.completed_evaluations[agent_name]:
                self.checkpoint.completed_evaluations[agent_name].append(scenario_id)
            self._save_checkpoint()
    
    def _is_evaluation_completed(self, agent_name: str, scenario_id: str) -> bool:
        """Check if an agent-scenario combination has already been completed"""
        if not self.checkpoint:
            return False
        agent_completed = self.checkpoint.completed_evaluations.get(agent_name, [])
        return scenario_id in agent_completed
    
    def _increment_failure_count(self, agent_name: str, scenario_id: str):
        """Increment the failure count for an agent-scenario combination"""
        if not self.checkpoint:
            return
        
        if not self.checkpoint.failed_attempts:
            self.checkpoint.failed_attempts = {}
        
        if agent_name not in self.checkpoint.failed_attempts:
            self.checkpoint.failed_attempts[agent_name] = {}
        
        current_count = self.checkpoint.failed_attempts[agent_name].get(scenario_id, 0)
        self.checkpoint.failed_attempts[agent_name][scenario_id] = current_count + 1
        
        self._save_checkpoint()
        logger.debug(f"Agent failure count for {agent_name} on {scenario_id}: {current_count + 1}/{self.checkpoint.max_retry_limit}")
    
    def _has_exceeded_retry_limit(self, agent_name: str, scenario_id: str) -> bool:
        """Check if a scenario has exceeded the retry limit for an agent"""
        if not self.checkpoint or not self.checkpoint.failed_attempts:
            return False
        
        agent_failures = self.checkpoint.failed_attempts.get(agent_name, {})
        attempt_count = agent_failures.get(scenario_id, 0)
        return attempt_count >= self.checkpoint.max_retry_limit
    
    async def evaluate_agents(
        self,
        agents: List[BaseAgent],
        scenarios: List[InteractiveScenario],
        resume: bool = True,
        max_concurrent_scenarios: int = 1
    ) -> Dict[str, List[AgentEvaluationResults]]:
        """
        Evaluate multiple agents on multiple scenarios with robust checkpointing
        """
        
        # Initialize or load checkpoint
        if resume:
            self.checkpoint = self._load_checkpoint()
            if self.checkpoint:
                console.print(f"🔄 Resuming agent evaluation from checkpoint...")
                console.print(f"📊 Progress: {self.checkpoint.completed_count}/{self.checkpoint.total_evaluations} completed")
                
                # CRITICAL FIX: Validate that the checkpoint matches current evaluation parameters
                current_agents = [agent.name for agent in agents]
                current_scenarios = [scenario.scenario_id for scenario in scenarios]
                
                # Check if agents and scenarios are compatible (more lenient validation)
                agents_match = self.checkpoint.agents == current_agents
                scenario_count_match = len(self.checkpoint.scenarios) == len(current_scenarios)
                total_evaluations_match = self.checkpoint.total_evaluations == len(agents) * len(scenarios)
                
                # More lenient scenario matching - check if most scenarios overlap
                if scenario_count_match:
                    scenario_overlap = len(set(self.checkpoint.scenarios) & set(current_scenarios))
                    scenario_overlap_ratio = scenario_overlap / len(current_scenarios)
                    scenarios_compatible = scenario_overlap_ratio >= 0.95  # Allow 5% difference
                else:
                    scenarios_compatible = False
                
                if agents_match and scenarios_compatible and total_evaluations_match:
                    console.print("✅ Checkpoint matches current evaluation parameters", style="green")
                    if scenario_overlap_ratio < 1.0:
                        console.print(f"   📊 Scenario overlap: {scenario_overlap_ratio:.1%} ({scenario_overlap}/{len(current_scenarios)})", style="blue")
                else:
                    console.print("⚠️  Checkpoint parameters don't match current evaluation, starting fresh", style="yellow")
                    console.print(f"   Agents match: {agents_match} ({self.checkpoint.agents} vs {current_agents})")
                    console.print(f"   Scenario count match: {scenario_count_match} ({len(self.checkpoint.scenarios)} vs {len(current_scenarios)})")
                    console.print(f"   Total evaluations match: {total_evaluations_match} ({self.checkpoint.total_evaluations} vs {len(agents) * len(scenarios)})")
                    if scenario_count_match:
                        console.print(f"   Scenario overlap: {scenario_overlap_ratio:.1%} ({scenario_overlap}/{len(current_scenarios)})")
                    self.checkpoint = None
                    resume = False
            else:
                console.print("⚠️  No agent checkpoint found, starting fresh evaluation")
                resume = False
        
        if not resume or not self.checkpoint:
            # Create new checkpoint
            console.print("🆕 Creating new checkpoint for fresh evaluation", style="blue")
            self.checkpoint = AgentEvaluationCheckpoint(
                started_at=datetime.now().isoformat(),
                agents=[agent.name for agent in agents],
                scenarios=[scenario.scenario_id for scenario in scenarios],
                total_evaluations=len(agents) * len(scenarios)
            )
            
            # CRITICAL FIX: Try to recover progress from incremental results if they exist
            if self.incremental_file.exists():
                console.print("🔄 Attempting to recover progress from incremental results...", style="yellow")
                incremental_results = self._load_incremental_results()
                if incremental_results:
                    # Rebuild checkpoint from incremental results
                    for result in incremental_results:
                        if result.agent_name not in self.checkpoint.completed_evaluations:
                            self.checkpoint.completed_evaluations[result.agent_name] = []
                        if result.scenario_id not in self.checkpoint.completed_evaluations[result.agent_name]:
                            self.checkpoint.completed_evaluations[result.agent_name].append(result.scenario_id)
                    
                    # Update completed count
                    self.checkpoint.completed_count = sum(len(scenarios) for scenarios in self.checkpoint.completed_evaluations.values())
                    console.print(f"✅ Recovered {self.checkpoint.completed_count} completed evaluations from incremental results", style="green")
            
            self._save_checkpoint()
        
        # Get remaining work
        remaining_work = self._get_remaining_work(agents, scenarios)
        console.print(f"🎯 Total agent work: {len(remaining_work)} evaluations")
        
        if resume and self.checkpoint:
            skipped_count = (len(agents) * len(scenarios)) - len(remaining_work)
            console.print(f"⏭️  Skipping {skipped_count} completed agent evaluations")
        
        # Load existing incremental results
        all_results = self._load_incremental_results()
        results = {}
        
        # Organize existing results by agent
        for result in all_results:
            if result.agent_name not in results:
                results[result.agent_name] = []
            results[result.agent_name].append(result)
        
        # Process remaining work with parallel execution
        if max_concurrent_scenarios > 1:
            console.print(f"⚡ Running up to {max_concurrent_scenarios} scenarios concurrently", style="blue")
            results = await self._evaluate_agents_parallel(remaining_work, results, max_concurrent_scenarios)
        else:
            console.print(f"🔄 Running scenarios sequentially", style="blue")
            results = await self._evaluate_agents_sequential(remaining_work, results)
        
        return results
    
    def _get_remaining_work(self, agents: List[BaseAgent], scenarios: List[InteractiveScenario]) -> List[Tuple[BaseAgent, InteractiveScenario]]:
        """Get list of (agent, scenario) pairs that still need to be evaluated"""
        remaining = []
        
        for agent in agents:
            for scenario in scenarios:
                if not self._is_evaluation_completed(agent.name, scenario.scenario_id):
                    if not self._has_exceeded_retry_limit(agent.name, scenario.scenario_id):
                        remaining.append((agent, scenario))
        
        return remaining
    
    async def _evaluate_agents_sequential(
        self, 
        remaining_work: List[Tuple[BaseAgent, InteractiveScenario]], 
        results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, List[AgentEvaluationResults]]:
        """Evaluate agents sequentially with progress tracking"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Evaluating agents...", total=len(remaining_work))
            
            for agent, scenario in remaining_work:
                if self._interrupted:
                    break
                
                progress.update(task, description=f"Agent: {agent.name} | Scenario: {scenario.scenario_id[:50]}...")
                
                # Evaluate agent on scenario
                result = await self._evaluate_agent_on_scenario(agent, scenario)
                
                if result:
                    # Save incrementally (crash-safe)
                    self._save_incremental_result(result)
                    
                    # Update checkpoint
                    self._update_checkpoint_completion(agent.name, scenario.scenario_id)
                    
                    # Add to results
                    if agent.name not in results:
                        results[agent.name] = []
                    results[agent.name].append(result)
                
                progress.advance(task)
        
        return results
    
    async def _evaluate_agents_parallel(
        self, 
        remaining_work: List[Tuple[BaseAgent, InteractiveScenario]], 
        results: Dict[str, List[AgentEvaluationResults]],
        max_concurrent_scenarios: int
    ) -> Dict[str, List[AgentEvaluationResults]]:
        """Evaluate agents in parallel with concurrency control"""
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent_scenarios)
        
        async def evaluate_with_semaphore(agent: BaseAgent, scenario: InteractiveScenario):
            """Wrapper to evaluate with concurrency control and crash-safe saving"""
            async with semaphore:
                if self._interrupted:
                    return None
                
                # CRITICAL FIX: Create a separate agent instance for this evaluation
                # to prevent conversation history contamination across parallel evaluations
                agent_copy = self._create_agent_copy(agent)
                
                # Evaluate the copied agent on scenario
                result = await self._evaluate_agent_on_scenario(agent_copy, scenario)
                
                if result:
                    # Save incrementally (crash-safe)
                    self._save_incremental_result(result)
                    
                    # Update checkpoint
                    self._update_checkpoint_completion(agent.name, scenario.scenario_id)
                
                return result
        
        # Create all evaluation tasks
        evaluation_tasks = [
            evaluate_with_semaphore(agent, scenario) 
            for agent, scenario in remaining_work
        ]
        
        # Execute with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Evaluating agents in parallel...", total=len(evaluation_tasks))
            
            # Process in batches to avoid overwhelming the system
            batch_size = max_concurrent_scenarios
            for i in range(0, len(evaluation_tasks), batch_size):
                if self._interrupted:
                    break
                
                batch = evaluation_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Process batch results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Parallel evaluation failed: {result}")
                    elif result:
                        agent_name = remaining_work[i + j][0].name
                        if agent_name not in results:
                            results[agent_name] = []
                        results[agent_name].append(result)
                    
                    progress.advance(task)
        
        return results
    
    async def _evaluate_agent_on_scenario(self, agent: BaseAgent, scenario: InteractiveScenario) -> Optional[AgentEvaluationResults]:
        """Evaluate a single agent on a single scenario with retry logic"""
        
        try:
            # CRITICAL: Clear agent's conversation history to prevent contamination across evaluations
            agent.clear_history()
            logger.debug(f"🧹 Cleared conversation history for agent {agent.name}")
            
            # Use the base evaluator to perform the actual evaluation
            result = await self.base_evaluator.evaluate_agent(agent, scenario)
            
            if result and result.overall_score > 0:  # Consider successful if score > 0
                logger.info(f"✅ Agent {agent.name} completed scenario {scenario.scenario_id}: {result.overall_score:.2f}")
                
                # Save conversation transcript
                await self._save_conversation_transcript(agent, scenario, result)
                
                return result
            else:
                # Failed evaluation - increment failure count
                self._increment_failure_count(agent.name, scenario.scenario_id)
                logger.warning(f"⚠️ Agent {agent.name} failed scenario {scenario.scenario_id}")
                return None
                
        except Exception as e:
            # Exception during evaluation - increment failure count
            self._increment_failure_count(agent.name, scenario.scenario_id)
            logger.error(f"❌ Agent {agent.name} error on scenario {scenario.scenario_id}: {e}")
            return None
    
    def _create_agent_copy(self, agent: BaseAgent) -> BaseAgent:
        """Create a separate copy of the agent to prevent conversation history contamination"""
        try:
            # Import the agent factory
            from ..agents.agent_factory import AgentFactory, AgentConfig, AgentType
            
            # Determine agent type based on the agent class
            agent_type = None
            agent_class_name = agent.__class__.__name__.lower()
            
            if 'openai' in agent_class_name:
                agent_type = AgentType.OPENAI
            elif 'anthropic' in agent_class_name:
                agent_type = AgentType.ANTHROPIC
            elif 'google' in agent_class_name:
                agent_type = AgentType.GOOGLE
            elif 'autogen' in agent_class_name:
                agent_type = AgentType.AUTOGEN
            elif 'langchain' in agent_class_name:
                agent_type = AgentType.LANGCHAIN
            elif 'crewai' in agent_class_name:
                agent_type = AgentType.CREWAI
            elif 'swarm' in agent_class_name:
                agent_type = AgentType.SWARM
            else:
                agent_type = AgentType.CUSTOM
            
            # Create agent configuration based on the original agent
            # CRITICAL FIX: Use consistent name for resume to work properly
            # The agent name must be the same across runs for checkpoint recovery
            agent_config = AgentConfig(
                name=agent.name,  # Use original name for resume consistency
                agent_type=agent_type,
                model_name=getattr(agent, 'model', None),
                api_key=getattr(agent, 'api_key', None),
                temperature=getattr(agent, 'temperature', 0.1),
                max_tokens=getattr(agent, 'max_tokens', 4096),
                custom_config=getattr(agent, 'config', {})
            )
            
            # Create a new agent instance
            agent_copy = AgentFactory.create_agent(agent_config)
            
            logger.debug(f"Created agent copy: {agent_copy.name} from {agent.name}")
            return agent_copy
            
        except Exception as e:
            logger.warning(f"Failed to create agent copy, using original agent (may cause contamination): {e}")
            # Clear the original agent's history as a fallback
            agent.clear_history()
            return agent
    
    async def _save_conversation_transcript(self, agent: BaseAgent, scenario: InteractiveScenario, result: AgentEvaluationResults):
        """Save the full conversation transcript for this evaluation"""
        try:
            # Use agent name directly (no more _copy_ suffix after fix)
            agent_name = agent.name
            
            # Create model-specific conversation directory to avoid overwriting
            conversations_dir = Path("intermediate_agent_results") / f"conversations_{agent_name}"
            conversations_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_file = conversations_dir / f"{agent_name}_{scenario.scenario_id}_{timestamp}.json"
            
            # Get session data from the base evaluator (if available)
            session_data = self.base_evaluator.get_session_data(result.session_id) if hasattr(self.base_evaluator, 'get_session_data') else None
            
            # Prepare conversation data
            conversation_data = {
                "evaluation_info": {
                    "agent_name": agent_name,  # Use agent name directly
                    "scenario_id": scenario.scenario_id,
                    "session_id": result.session_id,
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "overall_score": result.overall_score,
                    "total_turns": result.total_turns,
                    "session_duration": result.session_duration
                },
                "scenario_context": scenario.to_dict(),
                "conversation_history": agent.get_conversation_history() if hasattr(agent, 'get_conversation_history') else [],
                "session_data": session_data,
                "agent_statistics": agent.get_session_statistics() if hasattr(agent, 'get_session_statistics') else {}
            }
            
            # Convert AgentMessage objects to dictionaries for JSON serialization
            if conversation_data["conversation_history"]:
                conversation_data["conversation_history"] = [
                    msg.to_dict() if hasattr(msg, 'to_dict') else str(msg) 
                    for msg in conversation_data["conversation_history"]
                ]
            
            # Save to JSON file
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved conversation transcript: {conversation_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save conversation transcript for {agent.name}/{scenario.scenario_id}: {e}")
    
    def generate_evaluation_summary(self, results: Dict[str, List[AgentEvaluationResults]]) -> Dict[str, AgentEvaluationSummary]:
        """Generate comprehensive evaluation summaries for agents"""
        
        summaries = {}
        
        for agent_name, agent_results in results.items():
            if not agent_results:
                continue
            
            # Calculate basic stats
            total_scenarios = len(agent_results)
            # A scenario is considered "completed" if it has a valid session_status that's not "failed" or "error"
            # and has a non-None overall_score
            completed_scenarios = len([
                r for r in agent_results 
                if r.session_status not in ["failed", "error", "unknown"] 
                and r.overall_score is not None 
                and r.overall_score > 0
            ])
            failed_scenarios = total_scenarios - completed_scenarios
            
            # Calculate averages
            avg_overall_score = sum(r.overall_score for r in agent_results) / total_scenarios
            avg_conversation_turns = sum(r.total_turns for r in agent_results) / total_scenarios
            avg_session_duration = sum(r.session_duration for r in agent_results) / total_scenarios
            
            # Calculate LCBA Final Scores (Primary evaluation metrics)
            avg_lcba_comprehension = sum(getattr(r, 'lcba_comprehension', 0.0) for r in agent_results) / total_scenarios
            avg_lcba_efficiency = sum(getattr(r, 'lcba_efficiency', 0.0) for r in agent_results) / total_scenarios
            
            # Category and difficulty breakdowns
            category_scores = {}
            difficulty_scores = {}
            
            # Success rates
            parsing_success_rate = completed_scenarios / total_scenarios if total_scenarios > 0 else 0
            tool_usage_success_rate = 0.8  # Placeholder - would need actual tool usage data
            phase_completion_rate = 0.7  # Placeholder - would need actual phase completion data
            
            summary = AgentEvaluationSummary(
                agent_name=agent_name,
                total_scenarios=total_scenarios,
                completed_scenarios=completed_scenarios,
                failed_scenarios=failed_scenarios,
                avg_overall_score=avg_overall_score,
                avg_lcba_comprehension=avg_lcba_comprehension,
                avg_lcba_efficiency=avg_lcba_efficiency,
                avg_conversation_turns=avg_conversation_turns,
                avg_session_duration=avg_session_duration,
                category_scores=category_scores,
                difficulty_scores=difficulty_scores,
                parsing_success_rate=parsing_success_rate,
                tool_usage_success_rate=tool_usage_success_rate,
                phase_completion_rate=phase_completion_rate
            )
            
            summaries[agent_name] = summary
        
        return summaries
    
    def save_results(self, results: Dict[str, List[AgentEvaluationResults]], 
                    summaries: Dict[str, AgentEvaluationSummary], 
                    output_file: Path):
        """Save comprehensive agent evaluation results to file"""
        
        # Create comprehensive output data
        output_data = {
            "evaluation_metadata": {
                "framework": "LoCoBench-Agent",
                "version": "2.0",
                "evaluation_type": "multi_turn_agent",
                "timestamp": datetime.now().isoformat(),
                "total_agents": len(results),
                "total_scenarios": sum(len(agent_results) for agent_results in results.values()),
                "checkpoint_file": str(self.checkpoint_file),
                "incremental_file": str(self.incremental_file)
            },
            "agent_summaries": {
                agent_name: asdict(summary) for agent_name, summary in summaries.items()
            },
            "detailed_results": {
                agent_name: [result.to_dict() for result in agent_results]
                for agent_name, agent_results in results.items()
            },
            "cross_agent_analysis": self._generate_cross_agent_analysis(summaries),
            "configuration": {
                "checkpoint_version": self.checkpoint.checkpoint_version if self.checkpoint else "2.0",
                "max_retry_limit": self.checkpoint.max_retry_limit if self.checkpoint else 3,
                "evaluation_started": self.checkpoint.started_at if self.checkpoint else datetime.now().isoformat()
            }
        }
        
        # Save main results file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save markdown summary
        markdown_file = Path(str(output_file).replace('.json', '_summary.md'))
        self._save_markdown_summary(summaries, markdown_file)
        
        console.print(f"💾 Agent results saved to: {output_file}")
        console.print(f"📄 Agent summary saved to: {markdown_file}")
        console.print(f"📊 Saved {len(results)} agents × {sum(len(agent_results) for agent_results in results.values())} total evaluations")
    
    def _generate_cross_agent_analysis(self, summaries: Dict[str, AgentEvaluationSummary]) -> Dict[str, Any]:
        """Generate cross-agent comparison analysis"""
        
        if not summaries:
            return {}
        
        # Find best performing agent
        best_agent = max(summaries.items(), key=lambda x: x[1].avg_overall_score)
        
        # Calculate overall statistics
        all_scores = [s.avg_overall_score for s in summaries.values()]
        all_turns = [s.avg_conversation_turns for s in summaries.values()]
        
        return {
            "best_performing_agent": {
                "name": best_agent[0],
                "score": best_agent[1].avg_overall_score
            },
            "overall_statistics": {
                "avg_score_across_agents": sum(all_scores) / len(all_scores),
                "score_std_dev": self._calculate_std_dev(all_scores),
                "avg_turns_across_agents": sum(all_turns) / len(all_turns)
            },
            "agent_rankings": sorted(
                [(name, summary.avg_overall_score) for name, summary in summaries.items()],
                key=lambda x: x[1], reverse=True
            )
        }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _save_markdown_summary(self, summaries: Dict[str, AgentEvaluationSummary], output_file: Path):
        """Save a clear markdown summary of agent evaluation results"""
        
        markdown_content = []
        
        # Header
        markdown_content.append("# 🤖 LoCoBench-Agent Results Summary")
        markdown_content.append("")
        markdown_content.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**Framework Version:** LoCoBench-Agent v2.0")
        markdown_content.append(f"**Evaluation Type:** Multi-turn Agent Evaluation")
        markdown_content.append("")
        
        # Agent Performance Table
        markdown_content.append("## 🏆 Agent Performance Comparison")
        markdown_content.append("")
        markdown_content.append("| Rank | Agent | LCBA-Comp | LCBA-Eff | Completed | Failed | Avg Turns | Success Rate |")
        markdown_content.append("|------|-------|-----------|----------|-----------|--------|-----------|--------------|")
        
        # Sort agents by LCBA-Comprehension score (primary metric)
        sorted_agents = sorted(summaries.items(), key=lambda x: x[1].avg_lcba_comprehension, reverse=True)
        
        for i, (agent_name, summary) in enumerate(sorted_agents, 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            markdown_content.append(
                f"| {rank_emoji} | {agent_name} | {summary.avg_lcba_comprehension:.3f} | "
                f"{summary.avg_lcba_efficiency:.3f} | "
                f"{summary.completed_scenarios} | {summary.failed_scenarios} | "
                f"{summary.avg_conversation_turns:.1f} | {summary.parsing_success_rate:.1%} |"
            )
        
        markdown_content.append("")
        markdown_content.append("### 📊 Score Interpretation")
        markdown_content.append("")
        markdown_content.append("- **LCBA-Comp (Comprehension):** Quality, depth, correctness (23 metrics, 0-5 scale) - Higher is better")
        markdown_content.append("- **LCBA-Eff (Efficiency):** Speed, conciseness, resource optimization (8 metrics, 0-5 scale) - Higher is better")
        markdown_content.append("")
        markdown_content.append("**Note:** These are the two primary evaluation metrics. Different models may excel in different dimensions.")
        
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append("*Generated by LoCoBench-Agent v2.0 - Robust multi-turn agent evaluation*")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))
