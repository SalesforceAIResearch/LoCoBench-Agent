"""
LoCoBench Evaluation Pipeline

This module provides comprehensive evaluation capabilities for testing LLMs
on long-context development tasks using our automated validation framework.
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

from ..core.config import Config
from ..core.task import TaskCategory, DifficultyLevel
from ..generation.validation_framework import AutomatedValidator, ValidationResult
from ..generation.synthetic_generator import MultiLLMGenerator
from ..utils.llm_parsing import parse_llm_response

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class EvaluationCheckpoint:
    """Checkpoint state for resumable evaluations"""
    started_at: str
    checkpoint_version: str = "1.0"
    models: List[str] = None
    scenarios: List[str] = None  # scenario IDs
    task_categories: Optional[List[str]] = None
    difficulty_levels: Optional[List[str]] = None
    completed_evaluations: Dict[str, List[str]] = None  # model -> [scenario_ids]
    total_scenarios: int = 0
    completed_count: int = 0
    last_updated: str = ""
    output_file: Optional[str] = None
    
    # New: Retry tracking for failed scenarios
    failed_attempts: Dict[str, Dict[str, int]] = None  # model -> {scenario_id: attempt_count}
    max_retry_limit: int = 10  # Configurable retry limit
    
    def __post_init__(self):
        if self.completed_evaluations is None:
            self.completed_evaluations = {}
        if self.failed_attempts is None:
            self.failed_attempts = {}


@dataclass
class ModelEvaluationResult:
    """Results for a single model on a single scenario"""
    model_name: str
    scenario_id: str
    scenario_title: str
    task_category: str
    difficulty: str
    
    # Core scores (from ValidationResult) - 4 Evaluation Dimensions
    software_engineering_score: float    # 40% - Software Engineering Excellence (8 metrics)
    functional_correctness_score: float  # 30% - Functional Correctness (4 metrics)  
    code_quality_score: float           # 20% - Code Quality Assessment (3 metrics)
    longcontext_utilization_score: float # 10% - Long-Context Utilization (2 metrics)
    total_score: float
    
    # Additional metrics
    generation_time: float
    code_files_generated: int
    total_lines_generated: int
    parsing_success: bool
    
    # Solution code preservation
    solution_code: Dict[str, str]  # filename -> code content
    generated_files: List[str]     # list of filenames generated
    
    # Detailed breakdown
    detailed_results: Dict[str, Any]
    timestamp: str


@dataclass
class EvaluationSummary:
    """Summary statistics for model evaluation"""
    model_name: str
    total_scenarios: int
    completed_scenarios: int
    failed_scenarios: int
    
    # Average scores - 4 Evaluation Dimensions
    avg_software_engineering_score: float    # 40% - Software Engineering Excellence
    avg_functional_correctness_score: float  # 30% - Functional Correctness
    avg_code_quality_score: float           # 20% - Code Quality Assessment
    avg_longcontext_utilization_score: float # 10% - Long-Context Utilization
    avg_total_score: float
    
    # Performance stats
    avg_generation_time: float
    total_evaluation_time: float
    parsing_success_rate: float
    
    # Category breakdown
    category_results: Dict[str, Dict[str, float]]
    difficulty_results: Dict[str, Dict[str, float]]


class LoCoBenchEvaluator:
    """Main evaluator for LoCoBench benchmark"""
    
    def __init__(self, config: Config, model_name: str = None):
        self.config = config
        self.validator = AutomatedValidator(config)
        self.llm_generator = MultiLLMGenerator(config)
        self.results: List[ModelEvaluationResult] = []
        self.checkpoint: Optional[EvaluationCheckpoint] = None
        
        # Create intermediate_results directory if it doesn't exist
        intermediate_dir = Path("intermediate_results")
        intermediate_dir.mkdir(exist_ok=True)
        
        # Model-specific checkpoint files to avoid conflicts between concurrent evaluations
        if model_name:
            # Sanitize model name for filename
            safe_model_name = model_name.replace('-', '_').replace('.', '_').lower()
            self.checkpoint_file = intermediate_dir / f"evaluation_checkpoint_{safe_model_name}.json"
            self.incremental_file = intermediate_dir / f"evaluation_incremental_results_{safe_model_name}.json"
        else:
            # Fallback to original naming for backward compatibility
            self.checkpoint_file = intermediate_dir / "evaluation_checkpoint.json"
            self.incremental_file = intermediate_dir / "evaluation_incremental_results.json"
        
        self.current_model = model_name
        self._interrupted = False
        self._start_time = None
        self._scenario_times = []
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _calculate_eta(self, completed: int, total: int) -> str:
        """Calculate estimated time of arrival"""
        if completed == 0 or not self._start_time:
            return "Unknown"
        
        elapsed = time.time() - self._start_time
        avg_time_per_scenario = elapsed / completed
        remaining = total - completed
        eta_seconds = remaining * avg_time_per_scenario
        
        # Convert to human readable format
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.0f}m {eta_seconds%60:.0f}s"
        else:
            hours = eta_seconds // 3600
            minutes = (eta_seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def _display_evaluation_status(self, completed: int, total: int):
        """Display current evaluation status with progress and ETA"""
        if not self._start_time:
            self._start_time = time.time()
        
        elapsed = time.time() - self._start_time
        eta = self._calculate_eta(completed, total)
        percentage = (completed / total * 100) if total > 0 else 0
        
        console.print(f"📊 Progress: {completed}/{total} ({percentage:.1f}%) | "
                     f"⏱️  Elapsed: {elapsed/60:.1f}m | "
                     f"🎯 ETA: {eta}")
        
        if len(self._scenario_times) > 0:
            avg_time = sum(self._scenario_times) / len(self._scenario_times)
            console.print(f"⚡ Avg per scenario: {avg_time:.1f}s")
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        console.print(f"\n⚠️  Received {signal_name} (Ctrl+C). Saving progress...", style="yellow")
        
        self._interrupted = True
        
        # Save current state
        if self.checkpoint:
            self._save_checkpoint()
            console.print("💾 Checkpoint saved successfully!", style="green")
        
        # Clean exit message
        console.print("\n✅ Evaluation safely interrupted. Resume later with:", style="bold green")
        console.print("   locobench evaluate --resume [other-options]", style="cyan")
        
        sys.exit(0)
    
    def _save_checkpoint(self):
        """Save current evaluation state to checkpoint file"""
        if self.checkpoint:
            self.checkpoint.last_updated = datetime.now().isoformat()
            self.checkpoint.completed_count = sum(len(scenarios) for scenarios in self.checkpoint.completed_evaluations.values())
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(asdict(self.checkpoint), f, indent=2)
            logger.info(f"💾 Checkpoint saved: {self.checkpoint.completed_count}/{self.checkpoint.total_scenarios} completed")
    
    def _load_checkpoint(self) -> Optional[EvaluationCheckpoint]:
        """Load checkpoint from file if it exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                checkpoint = EvaluationCheckpoint(**data)
                
                # Check if this looks like a crash recovery situation
                if self._detect_crash_recovery(checkpoint):
                    console.print("🔄 Detected potential crash recovery situation", style="yellow")
                    console.print(f"📊 Last checkpoint: {checkpoint.completed_count}/{checkpoint.total_scenarios} completed")
                    console.print("💡 Tip: Use --resume to continue from where you left off")
                
                logger.info(f"📂 Loaded checkpoint: {checkpoint.completed_count}/{checkpoint.total_scenarios} completed")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    def _detect_crash_recovery(self, checkpoint: EvaluationCheckpoint) -> bool:
        """Detect if this appears to be a crash recovery situation"""
        # Check if checkpoint is recent (within last 24 hours) and incomplete
        try:
            last_update = datetime.fromisoformat(checkpoint.last_updated) if checkpoint.last_updated else None
            if last_update:
                hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
                is_recent = hours_since_update < 24
                is_incomplete = checkpoint.completed_count < checkpoint.total_scenarios
                return is_recent and is_incomplete
        except:
            pass
        return False
    
    def _check_for_auto_recovery(self) -> bool:
        """Check if we should automatically suggest recovery"""
        checkpoint = self._load_checkpoint()
        if checkpoint and self._detect_crash_recovery(checkpoint):
            console.print("\n" + "="*60, style="yellow")
            console.print("🚨 RECOVERY OPPORTUNITY DETECTED", style="bold yellow")
            console.print("="*60, style="yellow")
            console.print(f"Found incomplete evaluation from {checkpoint.last_updated}")
            console.print(f"Progress: {checkpoint.completed_count}/{checkpoint.total_scenarios} scenarios")
            
            if checkpoint.completed_count > 0:
                completion_percent = (checkpoint.completed_count / checkpoint.total_scenarios) * 100
                console.print(f"Completion: {completion_percent:.1f}%")
                console.print("\n💡 To resume this evaluation, run:", style="bold cyan")
                console.print("   locobench evaluate --resume [your-original-options]", style="cyan")
                console.print("\n🔄 Or start fresh by continuing with current command...", style="dim")
                console.print("="*60, style="yellow")
                return True
        return False
    
    def _update_checkpoint_completion(self, model_name: str, scenario_id: str):
        """Mark a model-scenario combination as completed in checkpoint"""
        if self.checkpoint:
            if model_name not in self.checkpoint.completed_evaluations:
                self.checkpoint.completed_evaluations[model_name] = []
            if scenario_id not in self.checkpoint.completed_evaluations[model_name]:
                self.checkpoint.completed_evaluations[model_name].append(scenario_id)
            self._save_checkpoint()
    
    def _is_evaluation_completed(self, model_name: str, scenario_id: str) -> bool:
        """Check if a model-scenario combination has already been completed"""
        if not self.checkpoint:
            return False
        model_completed = self.checkpoint.completed_evaluations.get(model_name, [])
        return scenario_id in model_completed
    
    def _get_remaining_work(self, models: List[str], scenarios: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Get list of (model, scenario) pairs that still need to be evaluated"""
        remaining = []
        for model in models:
            for scenario in scenarios:
                if not self._is_evaluation_completed(model, scenario.get('id', '')):
                    remaining.append((model, scenario))
        return remaining
    
    def _has_exceeded_retry_limit(self, model_name: str, scenario_id: str) -> bool:
        """Check if a scenario has exceeded the retry limit for a model"""
        if not self.checkpoint or not self.checkpoint.failed_attempts:
            return False
        
        model_failures = self.checkpoint.failed_attempts.get(model_name, {})
        attempt_count = model_failures.get(scenario_id, 0)
        return attempt_count >= self.checkpoint.max_retry_limit
    
    def _increment_failure_count(self, model_name: str, scenario_id: str):
        """Increment the failure count for a model-scenario combination"""
        if not self.checkpoint:
            return
        
        # Initialize failed_attempts if needed
        if not self.checkpoint.failed_attempts:
            self.checkpoint.failed_attempts = {}
        
        if model_name not in self.checkpoint.failed_attempts:
            self.checkpoint.failed_attempts[model_name] = {}
        
        current_count = self.checkpoint.failed_attempts[model_name].get(scenario_id, 0)
        self.checkpoint.failed_attempts[model_name][scenario_id] = current_count + 1
        
        # Save checkpoint after updating failure count
        self._save_checkpoint()
        
        logger.debug(f"Failure count for {model_name} on {scenario_id}: {current_count + 1}/{self.checkpoint.max_retry_limit}")
    
    def _get_failure_count(self, model_name: str, scenario_id: str) -> int:
        """Get the current failure count for a model-scenario combination"""
        if not self.checkpoint or not self.checkpoint.failed_attempts:
            return 0
        return self.checkpoint.failed_attempts.get(model_name, {}).get(scenario_id, 0)
    
    def _save_incremental_result(self, result: ModelEvaluationResult):
        """Save individual evaluation result to incremental results file using SAFE append-only approach"""
        incremental_file = self.incremental_file
        
        # SAFETY CHECK: Verify the result object is valid before saving
        if not result or not result.scenario_id:
            logger.error("Attempted to save invalid result object - skipping to prevent corruption")
            return
        
        # SAFE APPROACH: Use append-only JSONL format to prevent data loss
        # Each line is a complete JSON object, no risk of corruption from concurrent access
        try:
            # Convert result to JSON and append as a new line
            result_dict = asdict(result)
            
            # Validate the JSON can be serialized before writing
            result_json = json.dumps(result_dict)
            
            # Double-check the JSON is valid by parsing it back
            json.loads(result_json)  # This will raise exception if invalid
            
            # Append to file (thread-safe, atomic operation)
            with open(incremental_file, 'a', encoding='utf-8') as f:
                f.write(result_json + '\n')
                f.flush()  # Ensure data is written immediately
                # Verify the write was successful
                f.tell()  # This will raise exception if file is corrupted
            
            logger.debug(f"💾 Safely appended result for {result.model_name} on {result.scenario_id}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to save incremental result for {result.scenario_id}: {e}")
            logger.error("This result will be lost, but existing data is preserved.")
            # In case of any error, DO NOT reset or lose existing data - just log and continue
            # The failed result will be retried on next evaluation cycle
    
    def _migrate_json_to_jsonl(self) -> bool:
        """Migrate old JSON format to new JSONL format to prevent data loss"""
        incremental_file = self.incremental_file
        if not incremental_file.exists():
            return True
        
        try:
            # Check if file is already in JSONL format
            with open(incremental_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return True  # Empty file, nothing to migrate
                
                # Try to parse first line as JSON - if it works, likely already JSONL
                try:
                    json.loads(first_line)
                    logger.debug(f"File {incremental_file} appears to be already in JSONL format")
                    return True
                except json.JSONDecodeError:
                    pass  # Not JSONL, might be old JSON format
            
            # Try to load as old JSON array format
            try:
                with open(incremental_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                
                if isinstance(old_data, list):
                    # Backup the old file
                    backup_file = incremental_file.with_suffix('.json.backup')
                    import shutil
                    shutil.copy2(incremental_file, backup_file)
                    logger.info(f"Backed up old format to {backup_file}")
                    
                    # Convert to JSONL
                    with open(incremental_file, 'w', encoding='utf-8') as f:
                        for item in old_data:
                            f.write(json.dumps(item) + '\n')
                    
                    logger.info(f"Successfully migrated {len(old_data)} results from JSON to JSONL format")
                    return True
                    
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Could not parse {incremental_file} as JSON array, assuming JSONL format")
                return True
                
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False
        
        return True
    
    def _load_incremental_results(self) -> List[ModelEvaluationResult]:
        """Load all previously saved incremental results from JSONL format"""
        incremental_file = self.incremental_file
        if not incremental_file.exists():
            return []
        
        # Migrate old format if needed
        if not self._migrate_json_to_jsonl():
            logger.error("Failed to migrate old format, proceeding with JSONL parsing")
        
        results = []
        try:
            with open(incremental_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Parse each line as a separate JSON object
                        item = json.loads(line)
                        # Convert dict back to ModelEvaluationResult with backward compatibility
                        result = self._create_result_with_compatibility(item)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping corrupted line {line_num} in {incremental_file}: {e}")
                        # Continue processing other lines - don't lose all data due to one bad line
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {incremental_file}: {e}")
                        continue
            
            logger.info(f"📂 Loaded {len(results)} incremental results from JSONL format")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load incremental results file {incremental_file}: {e}")
            # Even if file reading fails, return empty list rather than crashing
            return []
    
    def _create_result_with_compatibility(self, item: Dict[str, Any]) -> ModelEvaluationResult:
        """Create ModelEvaluationResult with backward compatibility for old field names"""
        # Map old field names to new field names
        field_mapping = {
            'functional_score': 'functional_correctness_score',
            'agent_metrics_score': 'software_engineering_score', 
            'longcontext_metrics_score': 'software_engineering_score',
            'quality_score': 'code_quality_score',
            'style_score': 'longcontext_utilization_score'
        }
        
        # Create a copy of the item to avoid modifying original
        mapped_item = item.copy()
        
        # Map old field names to new ones
        for old_field, new_field in field_mapping.items():
            if old_field in mapped_item and new_field not in mapped_item:
                mapped_item[new_field] = mapped_item.pop(old_field)
        
        # Set default values for any missing new fields
        default_values = {
            'software_engineering_score': 0.0,
            'functional_correctness_score': 0.0,
            'code_quality_score': 0.0,
            'longcontext_utilization_score': 0.0
        }
        
        for field, default_value in default_values.items():
            if field not in mapped_item:
                mapped_item[field] = default_value
        
        return ModelEvaluationResult(**mapped_item)

    def _create_failed_scenario_response(self, model_name: str, scenario: Dict[str, Any], 
                                        failure_reason: str) -> ModelEvaluationResult:
        """Create a template response for scenarios that exceed retry limit"""
        from datetime import datetime
        
        return ModelEvaluationResult(
            model_name=model_name,
            scenario_id=scenario.get('id', 'unknown'),
            scenario_title=scenario.get('title', 'Unknown'),
            task_category=scenario.get('task_category', 'unknown'),
            difficulty=scenario.get('difficulty', 'unknown'),
            
            # Zero scores indicate model limitation/failure
            software_engineering_score=0.0,
            functional_correctness_score=0.0,
            code_quality_score=0.0,
            longcontext_utilization_score=0.0,
            total_score=0.0,
            
            generation_time=0.0,
            code_files_generated=0,
            total_lines_generated=0,
            parsing_success=False,  # Mark as parsing failure
            
            solution_code={},  # Empty solution
            generated_files=[],
            
            detailed_results={
                "failure_reason": failure_reason,
                "retry_limit_exceeded": True,
                "max_retries": self.checkpoint.max_retry_limit if self.checkpoint else 10,
                "failure_count": self._get_failure_count(model_name, scenario.get('id', 'unknown'))
            },
            timestamp=datetime.now().isoformat()
        )

    async def evaluate_model_on_scenario(self, model_name: str, scenario: Dict[str, Any]) -> Optional[ModelEvaluationResult]:
        """Evaluate a single model on a single scenario with timeout enforcement"""
        
        scenario_id = scenario.get('id', 'unknown')
        
        # Check if this evaluation was already completed
        if self._is_evaluation_completed(model_name, scenario_id):
            logger.info(f"⏭️  Skipping completed evaluation: {model_name} on {scenario_id}")
            return None
        
        # Check if scenario has exceeded retry limit
        if self._has_exceeded_retry_limit(model_name, scenario_id):
            # Create failed response and mark as completed
            failure_count = self._get_failure_count(model_name, scenario_id)
            failed_response = self._create_failed_scenario_response(
                model_name, scenario, f"Max retry limit exceeded ({failure_count} attempts)"
            )
            
            # Save the failed response and mark as completed
            try:
                self._save_incremental_result(failed_response)
                self._update_checkpoint_completion(model_name, scenario_id)
                logger.warning(f"🚫 Scenario {scenario_id} exceeded {self.checkpoint.max_retry_limit} retries - marking as failed")
            except Exception as e:
                logger.error(f"❌ Failed to save failed response for {model_name} on {scenario_id}: {e}")
            
            return failed_response
        
        try:
            # Determine timeout based on scenario type
            is_multi_session = scenario.get('task_category') == 'multi_session_development'
            timeout_seconds = (self.config.phase4.session_timeout if is_multi_session 
                              else self.config.phase4.task_timeout)
            
            # Wrap entire evaluation in timeout
            result = await asyncio.wait_for(
                self._evaluate_model_on_scenario_internal(model_name, scenario),
                timeout=timeout_seconds
            )
            
            # If evaluation succeeded, save result and mark as completed atomically
            # Only save and mark as completed if we have a successful result (parsing_success=True)
            if result and result.parsing_success:
                # Attempt to save result - only mark completed if save succeeds
                try:
                    self._save_incremental_result(result)
                    # Only mark as completed after successful save
                    self._update_checkpoint_completion(model_name, scenario_id)
                    logger.debug(f"✅ Successfully completed {model_name} on {scenario_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to save result for {model_name} on {scenario_id}: {e}")
                    # Don't mark as completed if save failed - scenario will be retried
                    return None
            elif result and not result.parsing_success:
                # Failed result - increment failure count, don't save or mark as completed, will be retried
                self._increment_failure_count(model_name, scenario_id)
                failure_count = self._get_failure_count(model_name, scenario_id)
                logger.warning(f"⚠️ Failed result for {model_name} on {scenario_id} (attempt {failure_count}/{self.checkpoint.max_retry_limit if self.checkpoint else 10}) - not marking as completed")
                return None
            
            return result
            
        except asyncio.TimeoutError:
            # Increment failure count for timeouts
            self._increment_failure_count(model_name, scenario_id)
            failure_count = self._get_failure_count(model_name, scenario_id)
            logger.warning(
                f"⏰ Timeout: Model {model_name} exceeded {timeout_seconds}s on scenario {scenario_id} (attempt {failure_count}/{self.checkpoint.max_retry_limit if self.checkpoint else 10})"
            )
            return None
        except Exception as e:
            # Increment failure count for exceptions
            self._increment_failure_count(model_name, scenario_id)
            failure_count = self._get_failure_count(model_name, scenario_id)
            logger.error(f"Error evaluating {model_name} on scenario {scenario_id} (attempt {failure_count}/{self.checkpoint.max_retry_limit if self.checkpoint else 10}): {e}")
            return None
    
    async def _evaluate_model_on_scenario_internal(self, model_name: str, scenario: Dict[str, Any]) -> Optional[ModelEvaluationResult]:
        """Internal evaluation method without timeout wrapper"""
        
        scenario_id = scenario.get('id', 'unknown')
        
        try:
            # Generate solution with the model
            start_time = time.time()
            solution_code = await self._generate_solution(model_name, scenario)
            generation_time = time.time() - start_time
            
            if not solution_code:
                logger.warning(f"Model {model_name} failed to parse solution for scenario {scenario_id}")
                # Create a result with parsing failure
                return ModelEvaluationResult(
                    model_name=model_name,
                    scenario_id=scenario_id,
                    scenario_title=scenario.get('title', 'Unknown'),
                    task_category=scenario.get('task_category', 'unknown'),
                    difficulty=scenario.get('difficulty', 'unknown'),
                    
                    software_engineering_score=0.0,
                    functional_correctness_score=0.0,
                    code_quality_score=0.0,
                    longcontext_utilization_score=0.0,
                    total_score=0.0,
                    
                    generation_time=generation_time,
                    code_files_generated=0,
                    total_lines_generated=0,
                    parsing_success=False,  # True parsing failure
                    
                    solution_code={},
                    generated_files=[],
                    
                    detailed_results={},
                    timestamp=datetime.now().isoformat()
                )
            
            # Sanitize solution_code to ensure all values are strings (fix for 79 failing multi-session scenarios)
            from ..generation.metric_algorithms import LoCoBenchMetricsCalculator
            sanitizer = LoCoBenchMetricsCalculator()
            solution_code = sanitizer._sanitize_solution_code(solution_code)
            
            # Parse solution statistics
            code_files_count = len(solution_code)
            total_lines = sum(len(code.split('\n')) for code in solution_code.values())
            # If we got solution_code, parsing succeeded (even if solution is minimal)
            parsing_success = True  # JSON parsing succeeded if we reached this point
            
            # Generate test suite for this scenario
            test_suite = await self.validator.generate_test_suite(scenario)
            
            # Validate the solution using our framework
            validation_result = await self.validator.validate_solution(
                scenario, solution_code, test_suite
            )
            
            # Create evaluation result
            result = ModelEvaluationResult(
                model_name=model_name,
                scenario_id=scenario_id,
                scenario_title=scenario.get('title', 'Unknown'),
                task_category=scenario.get('task_category', 'unknown'),
                difficulty=scenario.get('difficulty', 'unknown'),
                
                software_engineering_score=validation_result.software_engineering_score,
                functional_correctness_score=validation_result.functional_correctness_score,
                code_quality_score=validation_result.code_quality_score,
                longcontext_utilization_score=validation_result.longcontext_utilization_score,
                total_score=validation_result.total_score,
                
                generation_time=generation_time,
                code_files_generated=code_files_count,
                total_lines_generated=total_lines,
                parsing_success=parsing_success,
                
                # Preserve solution code for qualitative analysis
                solution_code=solution_code,  # Dict[str, str] of filename -> code
                generated_files=list(solution_code.keys()),  # List of filenames
                
                detailed_results=validation_result.detailed_results,
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for model {model_name} on scenario {scenario_id}: {e}")
            return None

    async def evaluate_models(self, model_names: List[str], scenarios: List[Dict[str, Any]], 
                            task_categories: Optional[List[str]] = None,
                            difficulty_levels: Optional[List[str]] = None,
                            max_concurrent_scenarios: int = 1,
                            resume: bool = True) -> Dict[str, List[ModelEvaluationResult]]:
        
        # Filter scenarios by task category and difficulty
        filtered_scenarios = self._filter_scenarios(scenarios, task_categories, difficulty_levels)
        
        # Initialize or load checkpoint
        if resume:
            self.checkpoint = self._load_checkpoint()
            if self.checkpoint:
                console.print(f"🔄 Resuming evaluation from checkpoint...")
                console.print(f"📊 Progress: {self.checkpoint.completed_count}/{self.checkpoint.total_scenarios} completed")
            else:
                console.print("⚠️  No checkpoint found, starting fresh evaluation")
                resume = False
        else:
            # Check for potential recovery opportunity
            if self._check_for_auto_recovery():
                console.print("\n⏸️  Continuing with fresh evaluation (as requested)...\n")
        
        if not resume or not self.checkpoint:
            # Create new checkpoint
            self.checkpoint = EvaluationCheckpoint(
                started_at=datetime.now().isoformat(),
                models=model_names,
                scenarios=[s.get('id', '') for s in filtered_scenarios],
                task_categories=task_categories,
                difficulty_levels=difficulty_levels,
                total_scenarios=len(filtered_scenarios) * len(model_names),
                output_file=None  # Will be set later
            )
            self._save_checkpoint()
        
        # Get work remaining (all work if not resuming, or just incomplete work if resuming)
        remaining_work = self._get_remaining_work(model_names, filtered_scenarios)
        
        console.print(f"🎯 Total work: {len(remaining_work)} evaluations")
        if resume and self.checkpoint:
            skipped_count = (len(model_names) * len(filtered_scenarios)) - len(remaining_work)
            console.print(f"⏭️  Skipping {skipped_count} completed evaluations")
        
        # Load any existing incremental results
        all_results = self._load_incremental_results()
        results = {}
        
        # Organize existing results by model
        for result in all_results:
            if result.model_name not in results:
                results[result.model_name] = []
            results[result.model_name].append(result)
        
        # Process remaining work
        for model_name in model_names:
            if model_name not in results:
                results[model_name] = []
            
            # Get remaining scenarios for this model
            model_remaining = [(model, scenario) for model, scenario in remaining_work if model == model_name]
            
            if not model_remaining:
                console.print(f"✅ {model_name}: All scenarios already completed")
                continue
            
            console.print(f"\n🤖 Evaluating model: {model_name}")
            if max_concurrent_scenarios > 1:
                console.print(f"⚡ Running up to {max_concurrent_scenarios} scenarios concurrently")
            
            model_results = results[model_name]  # Start with existing results
            failed_count = 0
            
            # Display initial status
            total_work = len(remaining_work)
            current_completed = sum(len(scenarios) for scenarios in self.checkpoint.completed_evaluations.values()) if self.checkpoint else 0
            self._display_evaluation_status(current_completed, self.checkpoint.total_scenarios if self.checkpoint else total_work)
            
            # Create semaphore for scenario concurrency
            scenario_semaphore = asyncio.Semaphore(max_concurrent_scenarios)
            
            async def evaluate_scenario_with_semaphore(scenario_data):
                """Wrapper to evaluate scenario with concurrency control and timeout handling"""
                async with scenario_semaphore:
                    _, scenario = scenario_data
                    scenario_start = time.time()
                    
                    # Set timeout based on model type (Claude gets longer timeout for long-context scenarios)
                    timeout_seconds = 600 if 'claude' in model_name.lower() else 300  # 10min for Claude, 5min for others
                    
                    try:
                        # Apply timeout to individual scenario evaluation
                        result = await asyncio.wait_for(
                            self.evaluate_model_on_scenario(model_name, scenario),
                            timeout=timeout_seconds
                        )
                        scenario_time = time.time() - scenario_start
                        self._scenario_times.append(scenario_time)
                        return result, scenario, scenario_time
                        
                    except asyncio.TimeoutError:
                        scenario_time = time.time() - scenario_start
                        self._scenario_times.append(scenario_time)
                        scenario_title = scenario.get('title', scenario.get('id', 'Unknown'))[:60]
                        logger.warning(f"⏰ Scenario timeout after {scenario_time:.1f}s: {scenario_title}")
                        # Return None to indicate timeout failure
                        return None, scenario, scenario_time
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Evaluating {model_name}", total=len(model_remaining))
                
                # Process scenarios with true streaming concurrency
                scenarios_only = [scenario for _, scenario in model_remaining]
                
                # Create ALL tasks at once for true concurrent streaming
                all_tasks = [
                        evaluate_scenario_with_semaphore((model_name, scenario))
                    for scenario in scenarios_only
                ]
                
                # Process results as they complete (streaming)
                completed_count = 0
                batch_start_count = len(model_results)  # Track successes from before this batch
                
                # Use asyncio.as_completed for true streaming
                for coro in asyncio.as_completed(all_tasks):
                    try:
                        result = await coro
                        completed_count += 1
                        
                        # Update progress as each task completes
                        progress.advance(task)
                        
                        # Process result immediately (result contains scenario info)
                        if isinstance(result, Exception):
                            failed_count += 1
                            console.print(f"  ❌ Task Exception: {result}")
                        elif result is None or result[0] is None:
                            # Scenario failed due to errors or timeout
                            failed_count += 1
                            if result and len(result) > 2 and result[2] >= 300:  # Check if timeout (>=5min)
                                console.print(f"  ⏰ Scenario timeout after {result[2]:.1f}s")
                            else:
                                console.print(f"  ❌ Scenario failed due to errors")
                        else:
                            result_obj, scenario, scenario_time = result
                            scenario_title = scenario.get('title', 'Unknown')[:50]
                            
                            if result_obj:
                                model_results.append(result_obj)
                                grade = self._get_letter_grade(result_obj.total_score)
                                console.print(f"  ✅ {scenario_title}: {result_obj.total_score:.3f} ({grade}) [{scenario_time:.1f}s]")
                            else:
                                failed_count += 1
                                console.print(f"  ❌ {scenario_title}: Failed [{scenario_time:.1f}s]")
                        
                        # Update overall progress periodically (only count actual successes)
                        if completed_count % 10 == 0:  # Every 10 completions
                            # Count only the NEW scenarios completed in this session
                            new_scenarios_completed = len(model_results) - batch_start_count
                            checkpoint_completed = sum(len(scenarios) for scenarios in self.checkpoint.completed_evaluations.values()) if self.checkpoint else 0
                            total_completed = checkpoint_completed + new_scenarios_completed
                            self._display_evaluation_status(total_completed, self.checkpoint.total_scenarios if self.checkpoint else len(scenarios_only))
                            
                    except Exception as e:
                        failed_count += 1
                        completed_count += 1
                        progress.advance(task)
                        console.print(f"  ❌ Task Exception: {e}")
                    
                    # Check for interruption during streaming
                    if self._interrupted:
                        console.print("🛑 Evaluation interrupted, stopping gracefully...")
                        # Cancel remaining tasks
                        for task_obj in all_tasks:
                            if not task_obj.done():
                                task_obj.cancel()
                        break
                    
                
                # All results are already processed in the streaming loop above
                
                # Check for interruption after model completion
                if self._interrupted:
                    console.print("🛑 Evaluation interrupted, exiting...")
                    break
            
            results[model_name] = model_results
            
            # Show model summary
            if model_results:
                avg_score = sum(r.total_score for r in model_results) / len(model_results)
                total_attempted = len(model_results) + failed_count
                success_rate = len(model_results) / total_attempted * 100 if total_attempted > 0 else 0
                console.print(f"📊 {model_name} Summary: {len(model_results)} completed, {failed_count} failed ({success_rate:.1f}% success), avg score: {avg_score:.3f}")
                
                # Show timeout-specific info for Claude models
                if 'claude' in model_name.lower() and failed_count > 0:
                    timeout_info = f" (Note: Failed scenarios may include timeouts due to long-context processing)"
                    console.print(f"   ⏰ Claude long-context note: Timeouts reflect processing time limitations{timeout_info}")
        
        return results

    async def evaluate_models_parallel(self, model_names: List[str], scenarios: List[Dict[str, Any]], 
                                      task_categories: Optional[List[str]] = None,
                                      difficulty_levels: Optional[List[str]] = None,
                                      max_concurrent_models: int = 2,
                                      max_concurrent_scenarios: int = 1,
                                      resume: bool = True) -> Dict[str, List[ModelEvaluationResult]]:
        """Evaluate multiple models in parallel with coordination"""
        
        # Use regular sequential evaluation for setup and filtering
        filtered_scenarios = self._filter_scenarios(scenarios, task_categories, difficulty_levels)
        
        # Initialize checkpoint system (same as sequential)
        if resume:
            self.checkpoint = self._load_checkpoint()
            if self.checkpoint:
                console.print(f"🔄 Resuming parallel evaluation from checkpoint...")
                console.print(f"📊 Progress: {self.checkpoint.completed_count}/{self.checkpoint.total_scenarios} completed")
            else:
                console.print("⚠️  No checkpoint found, starting fresh evaluation")
                resume = False
        else:
            if self._check_for_auto_recovery():
                console.print("\n⏸️  Continuing with fresh parallel evaluation (as requested)...\n")
        
        if not resume or not self.checkpoint:
            self.checkpoint = EvaluationCheckpoint(
                started_at=datetime.now().isoformat(),
                models=model_names,
                scenarios=[s.get('id', '') for s in filtered_scenarios],
                task_categories=task_categories,
                difficulty_levels=difficulty_levels,
                total_scenarios=len(filtered_scenarios) * len(model_names),
                output_file=None
            )
            self._save_checkpoint()
        
        # Get remaining work and organize by model
        remaining_work = self._get_remaining_work(model_names, filtered_scenarios)
        work_by_model = {}
        for model, scenario in remaining_work:
            if model not in work_by_model:
                work_by_model[model] = []
            work_by_model[model].append(scenario)
        
        console.print(f"🚀 Parallel evaluation: {len(model_names)} models, {len(remaining_work)} total evaluations")
        console.print(f"⚡ Max concurrent models: {max_concurrent_models}")
        
        # Load existing results
        all_results = self._load_incremental_results()
        results = {}
        for result in all_results:
            if result.model_name not in results:
                results[result.model_name] = []
            results[result.model_name].append(result)
        
        # Create semaphore to limit concurrent models
        model_semaphore = asyncio.Semaphore(max_concurrent_models)
        
        # Create tasks for each model
        model_tasks = []
        for model_name in model_names:
            if model_name in work_by_model and work_by_model[model_name]:
                task = asyncio.create_task(
                    self._evaluate_single_model_parallel(
                        model_name, 
                        work_by_model[model_name], 
                        model_semaphore,
                        results.get(model_name, []),
                        max_concurrent_scenarios
                    )
                )
                model_tasks.append((model_name, task))
            else:
                console.print(f"✅ {model_name}: All scenarios already completed")
                if model_name not in results:
                    results[model_name] = []
        
        # Run parallel evaluation
        if model_tasks:
            console.print(f"🏃‍♂️ Starting parallel evaluation of {len(model_tasks)} models...")
            
            completed_tasks = await asyncio.gather(
                *[task for _, task in model_tasks], 
                return_exceptions=True
            )
            
            # Collect results
            for i, (model_name, _) in enumerate(model_tasks):
                if isinstance(completed_tasks[i], Exception):
                    console.print(f"❌ {model_name}: Failed with error: {completed_tasks[i]}")
                    results[model_name] = results.get(model_name, [])
                else:
                    results[model_name] = completed_tasks[i]
        
        return results
    
    async def _evaluate_single_model_parallel(self, model_name: str, scenarios: List[Dict[str, Any]], 
                                            semaphore: asyncio.Semaphore, 
                                            existing_results: List[ModelEvaluationResult],
                                            max_concurrent_scenarios: int) -> List[ModelEvaluationResult]:
        """Evaluate a single model on its scenarios (used in parallel execution)"""
        
        async with semaphore:  # Limit concurrent models
            console.print(f"🤖 Starting {model_name} ({len(scenarios)} scenarios)")
            if max_concurrent_scenarios > 1:
                console.print(f"⚡ {model_name}: Running up to {max_concurrent_scenarios} scenarios concurrently")
            
            model_results = existing_results.copy()
            failed_count = 0
            
            # Create semaphore for scenario concurrency within this model
            scenario_semaphore = asyncio.Semaphore(max_concurrent_scenarios)
            
            async def evaluate_scenario_with_semaphore(scenario):
                """Wrapper to evaluate scenario with concurrency control"""
                async with scenario_semaphore:
                    scenario_start = time.time()
                    result = await self.evaluate_model_on_scenario(model_name, scenario)
                    scenario_time = time.time() - scenario_start
                    return result, scenario, scenario_time
            
            # Process scenarios in batches
            batch_size = max_concurrent_scenarios
            for i in range(0, len(scenarios), batch_size):
                batch = scenarios[i:i + batch_size]
                
                # Check for interruption
                if self._interrupted:
                    console.print(f"🛑 {model_name}: Interrupted, stopping...")
                    break
                
                # Create tasks for current batch
                batch_tasks = [evaluate_scenario_with_semaphore(scenario) for scenario in batch]
                
                # Wait for all scenarios in batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for j, batch_result in enumerate(batch_results):
                    scenario = batch[j]
                    scenario_title = scenario.get('title', 'Unknown')[:50]
                    
                    if isinstance(batch_result, Exception):
                        failed_count += 1
                        console.print(f"  ❌ {model_name}: {scenario_title} → Exception - {batch_result}")
                    else:
                        result, _, scenario_time = batch_result
                        if result:
                            model_results.append(result)
                            grade = self._get_letter_grade(result.total_score)
                            console.print(f"  ✅ {model_name}: {scenario_title} → {result.total_score:.3f} ({grade}) [{scenario_time:.1f}s]")
                        else:
                            failed_count += 1
                            console.print(f"  ❌ {model_name}: {scenario_title} → Failed [{scenario_time:.1f}s]")
                
                # Update progress every few batches
                if (i + batch_size) % (batch_size * 2) == 0:
                    current_completed = sum(len(scenarios) for scenarios in self.checkpoint.completed_evaluations.values()) if self.checkpoint else 0
                    console.print(f"📊 {model_name}: {i + batch_size}/{len(scenarios)} scenarios completed")
            
            # Show final summary for this model
            if model_results:
                avg_score = sum(r.total_score for r in model_results) / len(model_results)
                console.print(f"🏁 {model_name} Complete: {len(model_results)} scenarios, avg score: {avg_score:.3f}")
            
            return model_results

    def generate_evaluation_summary(self, results: Dict[str, List[ModelEvaluationResult]]) -> Dict[str, EvaluationSummary]:
        """Generate comprehensive evaluation summaries"""
        
        summaries = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Calculate averages
            total_scenarios = len(model_results)
            completed_scenarios = len([r for r in model_results if r.parsing_success])
            failed_scenarios = total_scenarios - completed_scenarios
            
            avg_software_engineering = sum(r.software_engineering_score for r in model_results) / total_scenarios
            avg_functional_correctness = sum(r.functional_correctness_score for r in model_results) / total_scenarios
            avg_code_quality = sum(r.code_quality_score for r in model_results) / total_scenarios
            avg_longcontext_utilization = sum(r.longcontext_utilization_score for r in model_results) / total_scenarios
            avg_total = sum(r.total_score for r in model_results) / total_scenarios
            
            avg_generation_time = sum(r.generation_time for r in model_results) / total_scenarios
            parsing_success_rate = completed_scenarios / total_scenarios
            
            # Category breakdown
            category_results = {}
            for category in TaskCategory:
                category_name = category.value
                category_scores = [r for r in model_results if r.task_category == category_name]
                if category_scores:
                    category_results[category_name] = {
                        'count': len(category_scores),
                        'avg_total_score': sum(r.total_score for r in category_scores) / len(category_scores),
                        'avg_software_engineering': sum(r.software_engineering_score for r in category_scores) / len(category_scores),
                        'avg_functional_correctness': sum(r.functional_correctness_score for r in category_scores) / len(category_scores),
                        'avg_code_quality': sum(r.code_quality_score for r in category_scores) / len(category_scores),
                        'avg_longcontext_utilization': sum(r.longcontext_utilization_score for r in category_scores) / len(category_scores)
                    }
            
            # Difficulty breakdown
            difficulty_results = {}
            for difficulty in DifficultyLevel:
                difficulty_name = difficulty.value
                difficulty_scores = [r for r in model_results if r.difficulty == difficulty_name]
                if difficulty_scores:
                    difficulty_results[difficulty_name] = {
                        'count': len(difficulty_scores),
                        'avg_total_score': sum(r.total_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_software_engineering': sum(r.software_engineering_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_functional_correctness': sum(r.functional_correctness_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_code_quality': sum(r.code_quality_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_longcontext_utilization': sum(r.longcontext_utilization_score for r in difficulty_scores) / len(difficulty_scores)
                    }
            
            summary = EvaluationSummary(
                model_name=model_name,
                total_scenarios=total_scenarios,
                completed_scenarios=completed_scenarios,
                failed_scenarios=failed_scenarios,
                
                avg_software_engineering_score=avg_software_engineering,
                avg_functional_correctness_score=avg_functional_correctness,
                avg_code_quality_score=avg_code_quality,
                avg_longcontext_utilization_score=avg_longcontext_utilization,
                avg_total_score=avg_total,
                
                avg_generation_time=avg_generation_time,
                total_evaluation_time=sum(r.generation_time for r in model_results),
                parsing_success_rate=parsing_success_rate,
                
                category_results=category_results,
                difficulty_results=difficulty_results
            )
            
            summaries[model_name] = summary
        
        return summaries

    def display_results(self, summaries: Dict[str, EvaluationSummary]):
        """Display formatted evaluation results"""
        
        if not summaries:
            console.print("❌ No evaluation results to display")
            return
        
        # Overall comparison table
        console.print(Panel.fit("🏆 LoCoBench Results", style="bold green"))
        
        comparison_table = Table(title="Model Performance Comparison")
        comparison_table.add_column("Model", style="bold")
        comparison_table.add_column("Total Score", style="green")
        comparison_table.add_column("Grade", style="yellow")
        comparison_table.add_column("Software Engineering", style="purple")
        comparison_table.add_column("Functional Correctness", style="blue")
        comparison_table.add_column("Code Quality", style="cyan")
        comparison_table.add_column("Long-Context Util", style="magenta")
        comparison_table.add_column("Success Rate", style="dim")
        
        # Sort by total score
        sorted_summaries = sorted(summaries.items(), key=lambda x: x[1].avg_total_score, reverse=True)
        
        for i, (model_name, summary) in enumerate(sorted_summaries):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            model_display = f"{medal} {model_name}"
            
            comparison_table.add_row(
                model_display,
                f"{summary.avg_total_score:.3f}",
                self._get_letter_grade(summary.avg_total_score),
                f"{summary.avg_software_engineering_score:.3f}",
                f"{summary.avg_functional_correctness_score:.3f}",
                f"{summary.avg_code_quality_score:.3f}",
                f"{summary.avg_longcontext_utilization_score:.3f}",
                f"{summary.parsing_success_rate:.1%}"
            )
        
        console.print(comparison_table)
        
        # Category breakdown
        if len(summaries) > 0:
            first_summary = next(iter(summaries.values()))
            if first_summary.category_results:
                self._display_category_breakdown(summaries)
        
        # Display comprehensive metrics for each model
        self._display_comprehensive_metrics(summaries)

    def _display_comprehensive_metrics(self, summaries: Dict[str, EvaluationSummary]):
        """Display comprehensive metrics analysis for all models"""
        
        console.print("\n" + "="*80)
        console.print(Panel.fit("📊 COMPREHENSIVE METRICS ANALYSIS", style="bold cyan"))
        console.print("="*80)
        
        for model_name, summary in summaries.items():
            # Get detailed results for this model
            model_results = []
            for result in self.results:
                if result.model_name == model_name:
                    model_results.append(result)
            
            if not model_results:
                continue
                
            console.print(f"\n🤖 [bold blue]MODEL: {model_name.upper()}[/bold blue]")
            console.print("="*60)
            
            # Extract and display software engineering metrics
            se_metrics = self._extract_software_engineering_metrics_summary(model_results)
            self._display_longcontext_metrics(se_metrics)
            
            # Extract and display functional metrics
            functional_metrics = self._extract_functional_correctness_metrics_summary(model_results)
            self._display_functional_metrics(functional_metrics)
            
            # Extract and display quality metrics
            quality_metrics = self._extract_code_quality_metrics_summary(model_results)
            self._display_quality_metrics(quality_metrics)
            
            # Extract and display long-context utilization metrics
            lcu_metrics = self._extract_longcontext_utilization_metrics_summary(model_results)
            
            # Display category breakdown for software engineering metrics
            self._display_longcontext_metrics_by_category(se_metrics)
            
            # Display summary
            self._display_model_summary(se_metrics, functional_metrics, quality_metrics, lcu_metrics, model_name)

    def _extract_software_engineering_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract software engineering excellence metrics (8 metrics: ACS, DTA, CFRD, STS, RS, CS, IS, SES)"""
        from collections import defaultdict
        import statistics
        
        metrics_by_category = defaultdict(lambda: defaultdict(list))
        all_individual_scores = defaultdict(list)
        
        for result in model_results:
            category = result.task_category
            
            # Get software engineering metrics from both old and new structure (backward compatibility)
            se_details = result.detailed_results.get('software_engineering_details', {})
            individual_scores = se_details.get('individual_scores', {})
            
            # Initialize traditional_scores to avoid scope issues
            traditional_scores = {}
            
            # Also check old structure for backward compatibility
            if not individual_scores:
                # Get from traditional_agent_metrics_details (ACS, DTA, CFRD, ICU, MMR)
                traditional_details = result.detailed_results.get('traditional_agent_metrics_details', {})
                traditional_scores = traditional_details.get('individual_scores', {})
                
            # Get from advanced_metrics_details (STS, RS, CS, IS, SES)
            advanced_details = result.detailed_results.get('advanced_metrics_details', {})
            advanced_scores = advanced_details.get('individual_scores', {})
            
            # Combine into software engineering metrics (exclude ICU, MMR which go to LCU)
            se_metrics_from_traditional = {k: v for k, v in traditional_scores.items() 
                                         if k not in ['information_coverage_utilization', 'multi_session_memory_retention']}
            individual_scores = {**se_metrics_from_traditional, **advanced_scores}
            
            # Collect software engineering metric scores (8 metrics)
            for metric_name, score in individual_scores.items():
                all_individual_scores[metric_name].append(score)
                metrics_by_category[category][metric_name].append(score)
        
        # Calculate averages
        overall_averages = {}
        for metric_name, scores in all_individual_scores.items():
            if scores:
                overall_averages[metric_name] = {
                    'average': statistics.mean(scores),
                    'count': len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0
                }
        
        # Calculate category averages
        category_averages = {}
        for category, metrics in metrics_by_category.items():
            category_averages[category] = {}
            for metric_name, scores in metrics.items():
                if scores:
                    category_averages[category][metric_name] = {
                        'average': statistics.mean(scores),
                        'count': len(scores)
                    }
        
        return {
            'overall_averages': overall_averages,
            'category_breakdown': category_averages
        }

    def _extract_functional_correctness_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract functional correctness metrics (4 metrics: Compilation, Unit Tests, Integration Tests, IDC)"""
        from collections import defaultdict
        import statistics
        
        compilation_scores = []
        unit_test_scores = []
        integration_scores = []
        idc_scores = []
        
        for result in model_results:
            # Get functional correctness metrics from new or old structure (backward compatibility)
            fc_details = result.detailed_results.get('functional_correctness_details', {})
            
            # Extract functional sub-scores
            if 'overall_breakdown' in fc_details:
                breakdown = fc_details['overall_breakdown']
            else:
                # Fall back to old structure
                functional_details = result.detailed_results.get('functional_details', {})
                breakdown = functional_details.get('overall_breakdown', {})
            
            if breakdown:
                comp_score = breakdown.get('compilation_score', 0)
                unit_score = breakdown.get('unit_test_score', 0)
                integ_score = breakdown.get('integration_score', 0)
                
                compilation_scores.append(comp_score)
                unit_test_scores.append(unit_score)
                integration_scores.append(integ_score)
            
            # Get IDC from functional correctness breakdown (new structure) or fallback to old structure
            idc_score = None
            if breakdown and 'idc_score' in breakdown:
                idc_score = breakdown['idc_score']
            else:
                # Fall back to old structure - IDC might be in traditional_agent_metrics_details
                traditional_details = result.detailed_results.get('traditional_agent_metrics_details', {})
                individual_scores = traditional_details.get('individual_scores', {})
                idc_score = individual_scores.get('incremental_development_capability')
            
            if idc_score is not None:
                idc_scores.append(idc_score)
        
        # Calculate overall averages
        overall = {}
        if compilation_scores:
            overall['compilation'] = {
                'average': statistics.mean(compilation_scores),
                'count': len(compilation_scores)
            }
        if unit_test_scores:
            overall['unit_tests'] = {
                'average': statistics.mean(unit_test_scores),
                'count': len(unit_test_scores)
            }
        if integration_scores:
            overall['integration'] = {
                'average': statistics.mean(integration_scores),
                'count': len(integration_scores)
            }
        if idc_scores:
            overall['incremental_development_capability'] = {
                'average': statistics.mean(idc_scores),
                'count': len(idc_scores)
            }
        
        return {'overall_averages': overall}

    def _extract_code_quality_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract code quality assessment metrics (3 metrics: Security, Quality Score, Issues Found)"""
        import statistics
        
        security_scores = []
        quality_scores = []
        issues_counts = []
        
        for result in model_results:
            # Get code quality metrics from new or old structure (backward compatibility)
            cq_details = result.detailed_results.get('code_quality_details', {})
            
            if cq_details:
                security_score = cq_details.get('security_analysis', {}).get('security_score', 0)
                overall_quality = cq_details.get('overall_quality_score', 0)
                issues_count = len(cq_details.get('issues_found', []))
            else:
                # Fall back to old structure
                quality_details = result.detailed_results.get('quality_details', {})
                security_score = quality_details.get('security_analysis', {}).get('security_score', 0)
                overall_quality = quality_details.get('overall_quality_score', 0)
                issues_count = len(quality_details.get('issues_found', []))
            
            security_scores.append(security_score)
            quality_scores.append(overall_quality)
            issues_counts.append(issues_count)
        
        # Calculate overall averages
        overall = {}
        if security_scores:
            overall['security'] = {
                'average': statistics.mean(security_scores),
                'count': len(security_scores)
            }
        if quality_scores:
            overall['overall_quality'] = {
                'average': statistics.mean(quality_scores),
                'count': len(quality_scores)
            }
        if issues_counts:
            overall['avg_issues_count'] = {
                'average': statistics.mean(issues_counts),
                'count': len(issues_counts)
            }
        
        return {'overall_averages': overall}

    def _extract_longcontext_utilization_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract long-context utilization metrics (2 metrics: ICU, MMR)"""
        from collections import defaultdict
        import statistics
        
        icu_scores = []
        mmr_scores = []
        
        for result in model_results:
            # Get long-context utilization metrics from new or old structure (backward compatibility)
            lcu_details = result.detailed_results.get('longcontext_utilization_details', {})
            individual_scores = lcu_details.get('individual_scores', {})
            
            # Fall back to old structure if needed
            if not individual_scores:
                traditional_details = result.detailed_results.get('traditional_agent_metrics_details', {})
                traditional_scores = traditional_details.get('individual_scores', {})
                # Only get ICU and MMR for long-context utilization
                individual_scores = {k: v for k, v in traditional_scores.items() 
                                   if k in ['information_coverage_utilization', 'multi_session_memory_retention']}
            
            if 'information_coverage_utilization' in individual_scores:
                icu_scores.append(individual_scores['information_coverage_utilization'])
            if 'multi_session_memory_retention' in individual_scores:
                mmr_scores.append(individual_scores['multi_session_memory_retention'])
        
        # Calculate overall averages
        overall = {}
        if icu_scores:
            overall['information_coverage_utilization'] = {
                'average': statistics.mean(icu_scores),
                'count': len(icu_scores),
                'min': min(icu_scores),
                'max': max(icu_scores),
                'std_dev': statistics.stdev(icu_scores) if len(icu_scores) > 1 else 0.0
            }
        if mmr_scores:
            overall['multi_session_memory_retention'] = {
                'average': statistics.mean(mmr_scores),
                'count': len(mmr_scores),
                'min': min(mmr_scores),
                'max': max(mmr_scores),
                'std_dev': statistics.stdev(mmr_scores) if len(mmr_scores) > 1 else 0.0
            }
        
        return {'overall_averages': overall}

    def _display_longcontext_metrics(self, longcontext_metrics: Dict[str, Any]):
        """Display software engineering metrics in a formatted table"""
        
        longcontext_overall = longcontext_metrics['overall_averages']
        
        # Software Engineering Capabilities - Core software engineering skills
        se_capabilities_metrics = {
            'architectural_coherence_score': 'Architectural Coherence Score (ACS)',
            'dependency_traversal_accuracy': 'Dependency Traversal Accuracy (DTA)',
            'multi_session_memory_retention': 'Multi-Session Memory Retention (MMR)',
            'cross_file_reasoning_depth': 'Cross-File Reasoning Depth (CFRD)',
            'incremental_development_capability': 'Incremental Development Capability (IDC)',
            'information_coverage_utilization': 'Information Coverage Utilization (ICU)'
        }
        
        # Software Engineering Excellence - Advanced development practices
        engineering_excellence_metrics = {
            'robustness_score': 'Robustness Score (RS)',
            'comprehensiveness_score': 'Comprehensiveness Score (CS)',
            'innovation_score': 'Innovation Score (IS)',
            'system_thinking_score': 'System Thinking Score (STS)',
            'solution_elegance_score': 'Solution Elegance Score (SES)'
        }
        
        # Display Software Engineering Capabilities metrics
        capabilities_found = any(metric in longcontext_overall for metric in se_capabilities_metrics.keys())
        if capabilities_found:
            console.print("\n🏗️ [bold blue]SOFTWARE ENGINEERING EXCELLENCE (8 metrics)[/bold blue]")
            console.print("[dim]Core long-context software development capabilities[/dim]")
            console.print("-" * 55)
            
            for metric_name, display_name in se_capabilities_metrics.items():
                if metric_name in longcontext_overall:
                    stats = longcontext_overall[metric_name]
                    console.print(f"  📈 [bold]{display_name}[/bold]:")
                    console.print(f"     Average: [blue]{stats['average']:.3f}[/blue] | Count: {stats['count']} | Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Display Software Engineering Excellence metrics
        engineering_found = any(metric in longcontext_overall for metric in engineering_excellence_metrics.keys())
        if engineering_found:
            console.print("\n🛡️ [bold magenta]SOFTWARE ENGINEERING EXCELLENCE (5 metrics)[/bold magenta]")
            console.print("[dim]Advanced software engineering practices and code quality[/dim]")
            console.print("-" * 60)
            
            for metric_name, display_name in engineering_excellence_metrics.items():
                if metric_name in longcontext_overall:
                    stats = longcontext_overall[metric_name]
                    console.print(f"  🛡️ [bold]{display_name}[/bold]:")
                    console.print(f"     Average: [magenta]{stats['average']:.3f}[/magenta] | Count: {stats['count']} | Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Show evaluation scope
        total_metrics = len([m for m in longcontext_overall.keys() if m in {**se_capabilities_metrics, **engineering_excellence_metrics}])
        if engineering_found and capabilities_found:
            console.print(f"\n🎯 [bold green]Comprehensive Long-Context Evaluation ({total_metrics} metrics)[/bold green]")
        elif capabilities_found:
            console.print("\n📊 [bold blue]Core Software Engineering Excellence Evaluation (8 metrics)[/bold blue]")
        elif engineering_found:
            console.print("\n🔬 [bold magenta]Software Engineering Excellence Evaluation (5 metrics)[/bold magenta]")

    def _display_functional_metrics(self, functional_metrics: Dict[str, Any]):
        """Display functional metrics"""
        
        console.print("\n⚙️  [bold green]FUNCTIONAL CORRECTNESS[/bold green]")
        console.print("[dim]Code compilation, testing, and basic functionality validation[/dim]")
        console.print("-" * 30)
        
        functional_overall = functional_metrics['overall_averages']
        
        functional_display_names = {
            'compilation': 'Code Compilation Success',
            'unit_tests': 'Unit Test Performance',
            'integration': 'Integration Test Performance'
        }
        
        for metric_name, display_name in functional_display_names.items():
            if metric_name in functional_overall:
                stats = functional_overall[metric_name]
                console.print(f"  🔧 {display_name}: [blue]{stats['average']:.3f}[/blue] (n={stats['count']})")

    def _display_quality_metrics(self, quality_metrics: Dict[str, Any]):
        """Display quality metrics"""
        
        console.print("\n🔍 [bold yellow]CODE QUALITY ASSESSMENT[/bold yellow]")
        console.print("[dim]Static analysis, security scanning, and code quality evaluation[/dim]")
        console.print("-" * 25)
        
        quality_overall = quality_metrics['overall_averages']
        
        quality_display_names = {
            'security': 'Security Analysis Score',
            'overall_quality': 'Overall Code Quality',
            'avg_issues_count': 'Average Issues Found'
        }
        
        for metric_name, display_name in quality_display_names.items():
            if metric_name in quality_overall:
                stats = quality_overall[metric_name]
                console.print(f"  🔍 {display_name}: [purple]{stats['average']:.3f}[/purple] (n={stats['count']})")

    def _display_longcontext_metrics_by_category(self, longcontext_metrics: Dict[str, Any]):
        """Display software engineering metrics breakdown by category"""
        
        console.print("\n📋 [bold cyan]SOFTWARE ENGINEERING METRICS BY CATEGORY[/bold cyan]")
        console.print("-" * 35)
        
        longcontext_categories = longcontext_metrics['category_breakdown']
        
        # Map metric names to display names
        metric_display_names = {
            'architectural_coherence_score': 'ACS',
            'dependency_traversal_accuracy': 'DTA',
            'multi_session_memory_retention': 'MMR',
            'cross_file_reasoning_depth': 'CFRD',
            'incremental_development_capability': 'IDC',
            'information_coverage_utilization': 'ICU'
        }
        
        for category, metrics in sorted(longcontext_categories.items()):
            console.print(f"\n  📂 [bold]{category.replace('_', ' ').title()}[/bold]:")
            for metric_name, stats in metrics.items():
                short_name = metric_display_names.get(metric_name, metric_name)
                console.print(f"     {short_name}: [cyan]{stats['average']:.3f}[/cyan] (n={stats['count']})")

    def _display_model_summary(self, longcontext_metrics: Dict[str, Any], functional_metrics: Dict[str, Any], 
                              quality_metrics: Dict[str, Any], lcu_metrics: Dict[str, Any], model_name: str):
        """Display high-level summary for the model"""
        import statistics
        
        console.print(f"\n📊 [bold green]SUMMARY FOR {model_name.upper()}[/bold green]")
        console.print("-" * 30)
        
        # Calculate overall software engineering metrics average
        longcontext_overall = longcontext_metrics['overall_averages']
        all_longcontext_scores = []
        for metric_stats in longcontext_overall.values():
            all_longcontext_scores.append(metric_stats['average'])
        
        if all_longcontext_scores:
            overall_longcontext_avg = statistics.mean(all_longcontext_scores)
            console.print(f"  🎯 Overall Software Engineering Score: [yellow]{overall_longcontext_avg:.3f}[/yellow]")
        
        # Calculate overall functional average
        functional_overall = functional_metrics['overall_averages']
        all_functional_scores = []
        for metric_stats in functional_overall.values():
            all_functional_scores.append(metric_stats['average'])
        
        if all_functional_scores:
            overall_functional_avg = statistics.mean(all_functional_scores)
            console.print(f"  ⚙️  Overall Functional Score: [blue]{overall_functional_avg:.3f}[/blue]")
        
        # Calculate overall quality average
        quality_overall = quality_metrics['overall_averages']
        all_quality_scores = []
        for metric_name, metric_stats in quality_overall.items():
            if metric_name != 'avg_issues_count':  # Exclude issues count from quality average
                all_quality_scores.append(metric_stats['average'])
        
        if all_quality_scores:
            overall_quality_avg = statistics.mean(all_quality_scores)
            console.print(f"  🛡️  Overall Quality Score: [purple]{overall_quality_avg:.3f}[/purple]")
        
        console.print("\n" + "="*60)

    def save_results(self, results: Dict[str, List[ModelEvaluationResult]], 
                    summaries: Dict[str, EvaluationSummary], 
                    output_file: Path):
        """Save comprehensive evaluation results to file"""
        
        # Calculate additional analytics
        all_scenarios = []
        for model_results in results.values():
            all_scenarios.extend([r.scenario_id for r in model_results])
        unique_scenarios = list(set(all_scenarios))
        
        # Analyze category distribution
        category_distribution = {}
        difficulty_distribution = {}
        for model_results in results.values():
            for result in model_results:
                cat = result.task_category
                diff = result.difficulty
                category_distribution[cat] = category_distribution.get(cat, 0) + 1
                difficulty_distribution[diff] = difficulty_distribution.get(diff, 0) + 1
        
        # Calculate cross-model comparison if multiple models
        model_comparison = {}
        if len(results) > 1:
            models = list(results.keys())
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models[i+1:], i+1):
                    if model1 in summaries and model2 in summaries:
                        comparison_key = f"{model1}_vs_{model2}"
                        model_comparison[comparison_key] = {
                            'total_score_diff': summaries[model1].avg_total_score - summaries[model2].avg_total_score,
                            'software_engineering_diff': summaries[model1].avg_software_engineering_score - summaries[model2].avg_software_engineering_score,
                            'functional_correctness_diff': summaries[model1].avg_functional_correctness_score - summaries[model2].avg_functional_correctness_score,
                            'code_quality_diff': summaries[model1].avg_code_quality_score - summaries[model2].avg_code_quality_score,
                            'longcontext_utilization_diff': summaries[model1].avg_longcontext_utilization_score - summaries[model2].avg_longcontext_utilization_score,
                            'generation_time_diff': summaries[model1].avg_generation_time - summaries[model2].avg_generation_time
                        }
        
        output_data = {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'config_file': str(self.config.config_path) if hasattr(self.config, 'config_path') else 'default',
                'total_models': len(results),
                'total_scenarios': sum(len(model_results) for model_results in results.values()),
                'unique_scenarios': len(unique_scenarios),
                'models_evaluated': list(results.keys()),
                'evaluation_scope': {
                    'category_distribution': category_distribution,
                    'difficulty_distribution': difficulty_distribution,
                    'unique_scenario_ids': unique_scenarios
                },
                'system_info': {
                    'total_evaluation_time': sum(s.total_evaluation_time for s in summaries.values()),
                    'avg_parsing_success_rate': sum(s.parsing_success_rate for s in summaries.values()) / len(summaries) if summaries else 0
                }
            },
            'configuration': {
                'api_settings': {
                    'max_requests_per_minute': getattr(self.config.api, 'max_requests_per_minute', 'N/A'),
                    'default_models': {
                        'openai': getattr(self.config.api, 'default_model_openai', 'N/A'),
                        'google': getattr(self.config.api, 'default_model_google', 'N/A')
                    }
                },
                'evaluation_weights': {
                    'architectural_coherence': self.config.phase4.software_engineering_weights.get('architectural_coherence', 0.125),
                    'dependency_traversal': self.config.phase4.software_engineering_weights.get('dependency_traversal', 0.125),
                    'cross_file_reasoning': self.config.phase4.software_engineering_weights.get('cross_file_reasoning', 0.125),
                    'system_thinking': self.config.phase4.software_engineering_weights.get('system_thinking', 0.125),
                    'robustness': self.config.phase4.software_engineering_weights.get('robustness', 0.125),
                    'comprehensiveness': self.config.phase4.software_engineering_weights.get('comprehensiveness', 0.125),
                    'innovation': self.config.phase4.software_engineering_weights.get('innovation', 0.125),
                    'solution_elegance': self.config.phase4.software_engineering_weights.get('solution_elegance', 0.125),
                    'information_coverage': self.config.phase4.longcontext_utilization_weights.get('information_coverage', 0.50),
                    'multi_session_memory': self.config.phase4.longcontext_utilization_weights.get('multi_session_memory', 0.50)
                },
                'benchmark_settings': {
                    'total_instances': getattr(self.config.phase3, 'total_instances', 'N/A'),
                    'min_information_coverage': getattr(self.config.phase3, 'min_information_coverage', 'N/A')
                }
            },
            'analysis': {
                'model_comparison': model_comparison,
                'performance_ranking': sorted(
                    [(model, summary.avg_total_score) for model, summary in summaries.items()],
                    key=lambda x: x[1], reverse=True
                ),
                'category_performance': {
                    model: summary.category_results for model, summary in summaries.items()
                }
            },
            'summaries': {model: asdict(summary) for model, summary in summaries.items()},
            'detailed_results': {
                model: [asdict(result) for result in model_results] 
                for model, model_results in results.items()
            },
            'scenario_lookup': {
                scenario_id: {
                    'models_evaluated': [
                        model for model, model_results in results.items()
                        if any(r.scenario_id == scenario_id for r in model_results)
                    ],
                    'results': {
                        model: next(
                            (asdict(r) for r in model_results if r.scenario_id == scenario_id), 
                            None
                        )
                        for model, model_results in results.items()
                    }
                }
                for scenario_id in unique_scenarios
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Also save a markdown summary
        markdown_file = Path(str(output_file).replace('.json', '_summary.md'))
        self._save_markdown_summary(summaries, markdown_file)
        
        # Print comprehensive save summary
        console.print(f"💾 Results saved to: {output_file}")
        console.print(f"📄 Summary saved to: {markdown_file}")
        console.print(f"📊 Saved {len(results)} models × {len(unique_scenarios)} scenarios = {sum(len(model_results) for model_results in results.values())} total evaluations")
        console.print(f"📈 File includes: summaries, detailed results, cross-model analysis, configuration, and scenario lookup")
        console.print(f"💡 Use this file for research analysis, visualization, and detailed performance investigation")

    def _save_markdown_summary(self, summaries: Dict[str, EvaluationSummary], output_file: Path):
        """Save a clear and well-organized markdown summary of evaluation results"""
        
        markdown_content = []
        
        # Header
        markdown_content.append("# 📊 LoCoBench Results Summary")
        markdown_content.append("")
        markdown_content.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**Framework Version:** LoCoBench v1.0")
        markdown_content.append(f"**Benchmark:** Multi-language Software Development Tasks")
        markdown_content.append("")
        
        # Overall Model Performance Table
        markdown_content.append("## 🏆 Model Performance Comparison")
        markdown_content.append("")
        
        # Create the main performance table
        table_headers = ["Model", "Total Score", "Grade", "Software Engineering", "Functional Correctness", "Code Quality", "Long-Context Util", "Success Rate"]
        markdown_content.append("| " + " | ".join(table_headers) + " |")
        markdown_content.append("| " + " | ".join(["---"] * len(table_headers)) + " |")
        
        # Sort models by total score (descending)
        sorted_models = sorted(summaries.items(), key=lambda x: x[1].avg_total_score, reverse=True)
        
        for i, (model_name, summary) in enumerate(sorted_models):
            # Add medal emoji for top performers
            model_display = model_name
            if i == 0:
                model_display = f"🥇 {model_name}"
            elif i == 1:
                model_display = f"🥈 {model_name}"
            elif i == 2:
                model_display = f"🥉 {model_name}"
            
            grade = self._get_letter_grade(summary.avg_total_score)
            success_rate = f"{(summary.completed_scenarios / summary.total_scenarios):.1%}" if summary.total_scenarios > 0 else "0.0%"
            
            row_data = [
                model_display,
                f"{summary.avg_total_score:.3f}",
                grade,
                f"{summary.avg_software_engineering_score:.3f}",
                f"{summary.avg_functional_correctness_score:.3f}",
                f"{summary.avg_code_quality_score:.3f}",
                f"{summary.avg_longcontext_utilization_score:.3f}",
                success_rate
            ]
            
            markdown_content.append("| " + " | ".join(row_data) + " |")
        
        markdown_content.append("")
        
        # DETAILED MODEL ANALYSIS
        for model_name, summary in sorted_models:
            # Get detailed results for this model
            model_results = []
            for result in self.results:
                if result.model_name == model_name:
                    model_results.append(result)
            
            if not model_results:
                continue
                
            markdown_content.append(f"## 🤖 {model_name.upper()} - Detailed Analysis")
            markdown_content.append("")
            
            # EXECUTIVE SUMMARY for this model
            markdown_content.append("### 📈 Performance Overview")
            markdown_content.append("")
            markdown_content.append(f"- **🎯 Total Score:** {summary.avg_total_score:.3f} ({self._get_letter_grade(summary.avg_total_score)})")
            markdown_content.append(f"- **🏗️ Software Engineering:** {summary.avg_software_engineering_score:.3f} (Weight: 40%)")
            markdown_content.append(f"- **⚙️ Functional Correctness:** {summary.avg_functional_correctness_score:.3f} (Weight: 30%)")
            markdown_content.append(f"- **🔍 Code Quality:** {summary.avg_code_quality_score:.3f} (Weight: 20%)")
            markdown_content.append(f"- **🧠 Long-Context Utilization:** {summary.avg_longcontext_utilization_score:.3f} (Weight: 10%)")
            markdown_content.append(f"- **✅ Success Rate:** {(summary.completed_scenarios / summary.total_scenarios):.1%}")
            markdown_content.append("")
            
            # Extract detailed metrics using the new 4-dimension structure
            software_engineering_metrics = self._extract_software_engineering_metrics_summary(model_results)
            functional_correctness_metrics = self._extract_functional_correctness_metrics_summary(model_results)
            code_quality_metrics = self._extract_code_quality_metrics_summary(model_results)
            longcontext_utilization_metrics = self._extract_longcontext_utilization_metrics_summary(model_results)
            
            # Get all detailed metrics data
            software_engineering_overall = software_engineering_metrics['overall_averages']
            functional_correctness_overall = functional_correctness_metrics['overall_averages']
            code_quality_overall = code_quality_metrics['overall_averages']
            longcontext_utilization_overall = longcontext_utilization_metrics['overall_averages']
            
            # 1. SOFTWARE ENGINEERING EXCELLENCE (8 metrics)
            markdown_content.append("### 🏗️ Software Engineering Excellence (8 metrics)")
            markdown_content.append("")
            markdown_content.append("*Advanced software engineering practices and architectural understanding*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            # Define Software Engineering metrics in order
            se_metrics = [
                ('architectural_coherence_score', 'ACS', 'System design consistency and architectural principles'),
                ('dependency_traversal_accuracy', 'DTA', 'Cross-module dependency navigation ability'),
                ('cross_file_reasoning_depth', 'CFRD', 'Multi-file relationship understanding'),
                ('system_thinking_score', 'STS', 'Scalability and maintainability considerations'),
                ('robustness_score', 'RS', 'Error handling and security practices'),
                ('comprehensiveness_score', 'CS', 'Documentation and API completeness'),
                ('innovation_score', 'IS', 'Modern patterns and algorithmic efficiency'),
                ('solution_elegance_score', 'SES', 'Code clarity and abstraction appropriateness')
            ]
            
            for metric_key, short_name, description in se_metrics:
                if metric_key in software_engineering_overall:
                    stats = software_engineering_overall[metric_key]
                    markdown_content.append(f"| **{short_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # 2. FUNCTIONAL CORRECTNESS (4 metrics)
            markdown_content.append("### ⚙️ Functional Correctness (4 metrics)")
            markdown_content.append("")
            markdown_content.append("*Code compilation, testing, and incremental development*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            # Functional metrics
            functional_metrics_list = [
                ('compilation', 'Code Compilation', 'Successful compilation across languages'),
                ('unit_tests', 'Unit Tests', 'Unit test execution and passing rate'),
                ('integration', 'Integration Tests', 'End-to-end functionality validation'),
                ('incremental_development_capability', 'IDC', 'Building effectively on previous work')
            ]
            
            for metric_key, display_name, description in functional_metrics_list:
                if metric_key in functional_correctness_overall:
                    stats = functional_correctness_overall[metric_key]
                    markdown_content.append(f"| **{display_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # 3. CODE QUALITY ASSESSMENT (3 metrics)
            markdown_content.append("### 🔍 Code Quality Assessment (3 metrics)")
            markdown_content.append("")
            markdown_content.append("*Static analysis, security, and maintainability*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            quality_metrics_list = [
                ('security', 'Security Analysis', 'Vulnerability detection and security practices'),
                ('overall_quality', 'Code Quality', 'Overall maintainability and readability'),
                ('avg_issues_count', 'Issues Found', 'Average code issues detected (lower is better)')
            ]
            
            for metric_key, display_name, description in quality_metrics_list:
                if metric_key in code_quality_overall:
                    stats = code_quality_overall[metric_key]
                    markdown_content.append(f"| **{display_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # 4. LONG-CONTEXT UTILIZATION (2 metrics)
            markdown_content.append("### 🧠 Long-Context Utilization (2 metrics)")
            markdown_content.append("")
            markdown_content.append("*Context usage efficiency and memory retention*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            longcontext_util_metrics = [
                ('information_coverage_utilization', 'ICU', 'Effective usage of provided context information'),
                ('multi_session_memory_retention', 'MMR', 'Context persistence across development sessions')
            ]
            
            for metric_key, short_name, description in longcontext_util_metrics:
                if metric_key in longcontext_utilization_overall:
                    stats = longcontext_utilization_overall[metric_key]
                    markdown_content.append(f"| **{short_name}** | {stats['average']:.3f} | {description} |")
            
                markdown_content.append("")
            
            # PERFORMANCE BY TASK CATEGORY
            markdown_content.append("### 📊 Performance by Task Category")
            markdown_content.append("")
            
            # Best and worst performing categories
            if summary.category_results:
                best_category = max(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                worst_category = min(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                
                markdown_content.append(f"- **🏆 Strongest Category:** {best_category[0].replace('_', ' ').title()} ({best_category[1]['avg_total_score']:.3f})")
                markdown_content.append(f"- **📈 Improvement Area:** {worst_category[0].replace('_', ' ').title()} ({worst_category[1]['avg_total_score']:.3f})")
            markdown_content.append("")
            
            # Category breakdown table
            markdown_content.append("| Category | Total Score | Software Engineering | Scenarios |")
            markdown_content.append("|----------|-------------|---------------------|-----------|")
            
            for category, stats in sorted(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'], reverse=True):
                category_name = category.replace('_', ' ').title()
                markdown_content.append(f"| {category_name} | {stats['avg_total_score']:.3f} | {stats['avg_software_engineering']:.3f} | {stats['count']} |")
            
            markdown_content.append("")
            markdown_content.append("---")
            markdown_content.append("")
        
        # CROSS-MODEL INSIGHTS (only if multiple models)
        if len(summaries) > 1:
            markdown_content.append("## 🔄 Multi-Model Comparison")
        markdown_content.append("")
        
        # Get all unique categories
        all_categories = set()
        for summary in summaries.values():
            all_categories.update(summary.category_results.keys())
        
        for category in sorted(all_categories):
            category_title = category.replace('_', ' ').title()
            markdown_content.append(f"### {category_title}")
            markdown_content.append("")
            
            # Category table headers
            cat_headers = ["Model", "Count", "Total Score", "Software Engineering", "Performance"]
            markdown_content.append("| " + " | ".join(cat_headers) + " |")
            markdown_content.append("| " + " | ".join(["---"] * len(cat_headers)) + " |")
            
            for model_name, summary in sorted_models:
                if category in summary.category_results:
                    cat_result = summary.category_results[category]
                    cat_row = [
                        model_name,
                        str(cat_result['count']),
                        f"{cat_result['avg_total_score']:.3f}",
                            f"{cat_result['avg_software_engineering']:.3f}",
                            "✅" if cat_result['avg_software_engineering'] > 0.3 else "📈"
                    ]
                    markdown_content.append("| " + " | ".join(cat_row) + " |")
            
            markdown_content.append("")
        
        # SUMMARY INSIGHTS
        markdown_content.append("## 💡 Summary & Insights")
        markdown_content.append("")
        
        if len(summaries) > 0:
            # Overall statistics
            total_evaluations = sum(s.completed_scenarios for s in summaries.values())
            avg_success_rate = sum((s.completed_scenarios / s.total_scenarios) if s.total_scenarios > 0 else 0 for s in summaries.values()) / len(summaries)
            
            markdown_content.append("### 📊 Evaluation Statistics")
            markdown_content.append("")
            markdown_content.append(f"- **📈 Total Evaluations:** {total_evaluations:,} scenarios")
            markdown_content.append(f"- **✅ Success Rate:** {avg_success_rate:.1%}")
            markdown_content.append(f"- **🎯 Coverage:** 8 task categories across multiple difficulty levels")
            markdown_content.append(f"- **🌐 Languages:** Multi-language evaluation (Python, Java, JavaScript, etc.)")
            markdown_content.append("")
        
            if len(summaries) == 1:
                # Single model insights
                model_name, summary = list(summaries.items())[0]
                
                markdown_content.append("### 🎯 Key Findings")
                markdown_content.append("")
                markdown_content.append(f"- **Overall Performance:** {summary.avg_total_score:.3f} ({self._get_letter_grade(summary.avg_total_score)})")
                
                # Strengths and weaknesses
                dimension_scores = [
                    ("Software Engineering", summary.avg_software_engineering_score),
                    ("Functional Correctness", summary.avg_functional_correctness_score), 
                    ("Code Quality", summary.avg_code_quality_score),
                    ("Long-Context Utilization", summary.avg_longcontext_utilization_score)
                ]
                
                best_dimension = max(dimension_scores, key=lambda x: x[1])
                worst_dimension = min(dimension_scores, key=lambda x: x[1])
                
                markdown_content.append(f"- **Strongest Dimension:** {best_dimension[0]} ({best_dimension[1]:.3f})")
                markdown_content.append(f"- **Improvement Area:** {worst_dimension[0]} ({worst_dimension[1]:.3f})")
                
                # Category performance
                if summary.category_results:
                    best_category = max(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                    worst_category = min(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                    
                    markdown_content.append(f"- **Best Category:** {best_category[0].replace('_', ' ').title()} ({best_category[1]['avg_total_score']:.3f})")
                    markdown_content.append(f"- **Challenging Category:** {worst_category[0].replace('_', ' ').title()} ({worst_category[1]['avg_total_score']:.3f})")
            else:
                # Multi-model insights
                best_model = sorted_models[0][0]
                best_score = sorted_models[0][1].avg_total_score
                
                markdown_content.append("### 🏆 Model Ranking")
                markdown_content.append("")
                markdown_content.append(f"- **🥇 Top Performer:** {best_model} ({best_score:.3f})")
                
                if len(sorted_models) > 1:
                    score_gap = sorted_models[0][1].avg_total_score - sorted_models[1][1].avg_total_score
                    markdown_content.append(f"- **Performance Gap:** {score_gap:.3f} points between 1st and 2nd place")
        
        markdown_content.append("")
        markdown_content.append("## 📖 Evaluation Framework")
        markdown_content.append("")
        markdown_content.append("LoCoBench uses **17 metrics across 4 dimensions**:")
        markdown_content.append("")
        markdown_content.append("- **🏗️ Software Engineering Excellence (8 metrics):** ACS, DTA, CFRD, STS, RS, CS, IS, SES")
        markdown_content.append("- **⚙️ Functional Correctness (4 metrics):** Code Compilation, Unit Tests, Integration Tests, IDC")
        markdown_content.append("- **🔍 Code Quality Assessment (3 metrics):** Security Analysis, Code Quality, Issue Detection") 
        markdown_content.append("- **🧠 Long-Context Utilization (2 metrics):** ICU, MMR")
        markdown_content.append("")
        markdown_content.append("**Scoring:** LoCoBench Score (LCBS) = 40% SE + 30% FC + 20% CQ + 10% LCU")
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append("*Generated by LoCoBench v1.0 - Comprehensive long-context software development evaluation*")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))

    # Helper methods
    
    async def _generate_solution(self, model_name: str, scenario: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate solution using specified model with enhanced prompts and retry logic"""
        
        # Detect language from scenario
        def detect_language_from_scenario_id(scenario_id: str) -> str:
            """Extract language from scenario ID like 'java_web_ecommerce_easy_001'"""
            if not scenario_id:
                return 'python'  # Default fallback
            
            parts = scenario_id.split('_')
            if not parts:
                return 'python'
            
            language_part = parts[0].lower()
            
            # Map scenario language prefixes to language names
            language_mapping = {
                'c': 'c',
                'cpp': 'cpp', 
                'csharp': 'csharp',
                'go': 'go',
                'java': 'java',
                'javascript': 'javascript',
                'php': 'php',
                'python': 'python',
                'rust': 'rust',
                'typescript': 'typescript'
            }
            
            return language_mapping.get(language_part, 'python')
        
        # Get language and create language-specific prompt
        scenario_id = scenario.get('id', '')
        language = detect_language_from_scenario_id(scenario_id)
        
        # Language-specific configurations
        language_configs = {
            'c': {
                'engineer': 'C software engineer',
                'files': {'main.c': '#include <stdio.h>\\n\\nint main() {\\n    printf(\\"Hello World\\\\n\\");\\n    return 0;\\n}'},
                'imports': '#include <stdio.h>',
                'practices': 'proper memory management, error handling, and C best practices'
            },
            'cpp': {
                'engineer': 'C++ software engineer', 
                'files': {'main.cpp': '#include <iostream>\\n\\nint main() {\\n    std::cout << \\"Hello World\\" << std::endl;\\n    return 0;\\n}'},
                'imports': '#include <iostream>',
                'practices': 'proper memory management, RAII, and modern C++ best practices'
            },
            'csharp': {
                'engineer': 'C# software engineer',
                'files': {'Program.cs': 'using System;\\n\\nnamespace Solution {\\n    class Program {\\n        static void Main(string[] args) {\\n            Console.WriteLine(\\"Hello World\\");\\n        }\\n    }\\n}'},
                'imports': 'using System;',
                'practices': 'proper exception handling, LINQ, and C# best practices'
            },
            'go': {
                'engineer': 'Go software engineer',
                'files': {'main.go': 'package main\\n\\nimport \\"fmt\\"\\n\\nfunc main() {\\n    fmt.Println(\\"Hello\\")\\n}'},
                'imports': 'import "fmt"',
                'practices': 'proper Go imports, error handling, and best practices'
            },
            'java': {
                'engineer': 'Java software engineer',
                'files': {'Main.java': 'public class Main {\\n    public static void main(String[] args) {\\n        System.out.println(\\"Hello World\\");\\n    }\\n}'},
                'imports': 'import java.util.*;',
                'practices': 'proper Java imports, exception handling, and best practices'
            },
            'javascript': {
                'engineer': 'JavaScript software engineer',
                'files': {'main.js': 'console.log(\\"Hello World\\");'},
                'imports': 'const fs = require("fs");',
                'practices': 'proper JavaScript modules, error handling, and best practices'
            },
            'php': {
                'engineer': 'PHP software engineer',
                'files': {'index.php': '<?php\\necho \\"Hello World\\";\\n?>'},
                'imports': '<?php',
                'practices': 'proper PHP syntax, error handling, and best practices'
            },
            'python': {
                'engineer': 'Python software engineer',
                'files': {'main.py': 'def main():\\n    print(\\"Hello World\\")\\n\\nif __name__ == \\"__main__\\":\\n    main()'},
                'imports': 'import sys',
                'practices': 'proper Python imports, error handling, and best practices'
            },
            'rust': {
                'engineer': 'Rust software engineer',
                'files': {'main.rs': 'fn main() {\\n    println!(\\"Hello World\\");\\n}'},
                'imports': 'use std::collections::HashMap;',
                'practices': 'proper Rust ownership, error handling, and best practices'
            },
            'typescript': {
                'engineer': 'TypeScript software engineer',
                'files': {'main.ts': 'console.log(\\"Hello World\\");'},
                'imports': 'import * as fs from "fs";',
                'practices': 'proper TypeScript types, imports, and best practices'
            }
        }
        
        config = language_configs.get(language, language_configs['python'])
        file_examples = '\\n'.join([f'        "{filename}": "{content}"' for filename, content in config['files'].items()])
        
        # Handle multi-session task prompts properly
        task_prompt = scenario.get('task_prompt', '')
        if isinstance(task_prompt, dict):
            # Multi-session development scenario - combine all sessions
            session_requirements = []
            for session_key in sorted(task_prompt.keys()):
                session_content = task_prompt[session_key]
                session_requirements.append(f"**{session_key.upper()}**: {session_content}")
            formatted_requirements = '\n\n'.join(session_requirements)
        else:
            # Regular scenario with string task_prompt
            formatted_requirements = str(task_prompt)
        
        # Create enhanced solution prompt (now language-aware)
        solution_prompt = f"""You are an expert {config['engineer']}. Your task is to provide a complete, working solution.

**TASK**: {scenario.get('title', 'Development Task')}

**DESCRIPTION**: {scenario.get('description', '')}

**REQUIREMENTS**: 
{formatted_requirements}

**CONTEXT FILES**: {', '.join(scenario.get('context_files', []))}

**CRITICAL INSTRUCTIONS**:
1. You MUST respond with valid JSON in the exact format shown below
2. Each file MUST contain complete, syntactically correct {language.upper()} code
3. Do NOT truncate your response - provide the complete solution
4. Use {config['practices']}

**REQUIRED RESPONSE FORMAT**:
```json
{{
    "approach": "Your solution strategy (keep under 200 words)",
    "files": {{
{file_examples}
    }},
    "explanation": "Implementation details (keep under 300 words)"
}}
```

**VALIDATION CHECKLIST**:
- ✅ Response is valid JSON wrapped in ```json blocks
- ✅ All strings are properly escaped (\\n for newlines, \\" for quotes)
- ✅ Each file contains complete {language.upper()} code
- ✅ Code compiles and addresses all requirements
- ✅ Response is complete (not truncated)

Generate your response now:"""

        # Map model names to our generator keys  
        model_key_mapping = {
            # Generic mappings
            'openai': 'openai',
            'gemini': 'google',
            
            # === OpenAI Models ===
            # Latest GPT-5 series
            'gpt-5': 'openai',
            'gpt-5-mini': 'openai',
            'gpt-5-nano': 'openai',
            'gpt-5-chat-latest': 'openai',
            
            # GPT-4.1 series
            'gpt-4.1': 'openai',
            'gpt-4.1-2025-04-14': 'openai',
            'gpt-4.1-mini': 'openai',
            'gpt-4.1-mini-2025-04-14': 'openai',
            'gpt-4.1-nano': 'openai',
            'gpt-4.1-nano-2025-04-14': 'openai',
            'gpt-4.5-preview': 'openai',
            'gpt-4.5-preview-2025-02-27': 'openai',
            
            # GPT-4o series (text-based, suitable for code)
            'gpt-4o': 'openai',
            'gpt-4o-2024-05-13': 'openai',
            'gpt-4o-2024-08-06': 'openai',
            'gpt-4o-mini': 'openai',
            'gpt-4o-mini-2024-07-18': 'openai',
            'gpt-4o-search-preview': 'openai',
            'gpt-4o-search-preview-2025-03-11': 'openai',
            'gpt-4o-mini-search-preview': 'openai',
            'gpt-4o-mini-search-preview-2025-03-11': 'openai',
            
            # o-series reasoning models
            'o1': 'openai',
            'o1-2024-12-17': 'openai',
            'o1-pro': 'openai',
            'o1-pro-2025-03-19': 'openai',
            'o1-mini': 'openai',
            'o1-mini-2024-09-12': 'openai',
            
            # o3-series models
            'o3': 'openai',
            'o3-2025-04-16': 'openai',
            'o3-pro': 'openai',
            'o3-pro-2025-06-10': 'openai',
            'o3-deep-research': 'openai',
            'o3-deep-research-2025-06-26': 'openai',
            'o3-mini': 'openai',
            'o3-mini-2025-01-31': 'openai',
            
            # o4-series models
            'o4-mini': 'openai',
            'o4-mini-2025-04-16': 'openai',
            'o4-mini-deep-research': 'openai',
            'o4-mini-deep-research-2025-06-26': 'openai',
            
            # Codex models (specialized for code)
            'codex-mini-latest': 'openai',
            
            # Computer use model (can handle code tasks)
            'computer-use-preview': 'openai',
            'computer-use-preview-2025-03-11': 'openai',
            
            # Legacy GPT-4 models
            'gpt-4': 'openai',
            'gpt-4-turbo': 'openai',
            
            # Legacy OpenAI naming
            'openai-o3': 'openai',
            
            # === Google Gemini Models ===
            # Gemini 2.5 series (text-capable)
            'gemini-2.5-pro': 'google',
            'gemini-2.5-flash': 'google',
            'gemini-2.5-flash-lite': 'google',
            
            # Gemini 2.0 series (text-capable)
            'gemini-2.0-flash': 'google',
            'gemini-2.0-flash-lite': 'google',
            
            # === Claude Models (Bearer Token Authentication) ===
            # Claude 4 series - Latest generation
            'claude-sonnet-4': 'claude-sonnet-4',          # Balanced performance
            'claude-opus-4': 'claude-opus-4',              # High capability
            'claude-sonnet-3.7': 'claude-sonnet-3.7',      # Hybrid reasoning
            
            # Note: Excluding audio/video/TTS-only models as they're not suitable for code evaluation:
            # - gpt-4o-audio-preview, gpt-4o-realtime-preview (audio focus)
            # - gpt-4o-mini-audio-preview, gpt-4o-mini-realtime-preview (audio focus)
            # - gpt-image-1 (image generation only)
            # - gemini-live-* (live interaction, not text generation)
            # - gemini-*-preview-tts (text-to-speech only)
            # - gemini-*-native-audio-dialog (audio focus)
            # - gemini-2.0-flash-preview-image-generation (image generation focus)
        }
        
        # Ensure model_name is a string (fix for dict error)
        if not isinstance(model_name, str):
            logger.warning(f"Model name is not a string: {type(model_name)} = {model_name}")
            model_name = str(model_name)
        
        model_key = model_key_mapping.get(model_name.lower(), 'openai')
        
        # Retry logic for empty responses
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # For evaluation, use the specific model name instead of default config
                if model_key == 'openai':
                    # Temporarily override the default model for this evaluation
                    original_model = self.llm_generator.config.api.default_model_openai
                    self.llm_generator.config.api.default_model_openai = model_name
                    response = await self.llm_generator.generate_with_model(model_key, solution_prompt)
                    # Restore original model
                    self.llm_generator.config.api.default_model_openai = original_model
                elif model_key == 'google':
                    # Temporarily override the default model for this evaluation  
                    original_model = self.llm_generator.config.api.default_model_google
                    self.llm_generator.config.api.default_model_google = model_name
                    response = await self.llm_generator.generate_with_model(model_key, solution_prompt)
                    # Restore original model
                    self.llm_generator.config.api.default_model_google = original_model
                else:
                    response = await self.llm_generator.generate_with_model(model_key, solution_prompt)
                
                # Validate response before parsing
                if not response or len(response.strip()) < 50:
                    logger.warning(f"Empty/tiny response from {model_name} (attempt {attempt + 1}/{max_retries}): {len(response)} chars")
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        logger.error(f"All retry attempts failed for {model_name}")
                        return None
                
                # Parse the response using our enhanced parser
                solution_code = parse_llm_response(response, expected_language=language)
                
                # Debug logging for multi-session scenarios
                if scenario.get('task_category') == 'multi_session_development' and solution_code:
                    logger.info(f"🔍 DEBUG: Multi-session solution_code type: {type(solution_code)}")
                    logger.info(f"🔍 DEBUG: Multi-session solution_code keys: {list(solution_code.keys())}")
                    for key, value in solution_code.items():
                        logger.info(f"🔍 DEBUG: {key}: {type(value)} = {repr(value)[:100]}...")
                
                # Validate parsed result
                if not solution_code:
                    logger.warning(f"Failed to parse response from {model_name} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        return None
                
                # Check if solution has reasonable content
                total_content = sum(len(code) for code in solution_code.values())
                if total_content < 100:
                    logger.warning(f"Suspiciously short solution from {model_name} (attempt {attempt + 1}/{max_retries}): {total_content} chars")
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        return solution_code  # Accept even short solutions on final attempt
                
                # Success!
                logger.info(f"✅ Successfully generated solution from {model_name}: {len(solution_code)} files, {total_content} chars")
                return solution_code
                
            except Exception as e:
                logger.error(f"Solution generation error for {model_name} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    return None
        
        return None

    def _filter_scenarios(self, scenarios: List[Dict[str, Any]], 
                         task_categories: Optional[List[str]], 
                         difficulty_levels: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Filter scenarios based on criteria"""
        
        filtered = scenarios
        
        if task_categories:
            filtered = [s for s in filtered if s.get('task_category') in task_categories]
        
        if difficulty_levels:
            filtered = [s for s in filtered if s.get('difficulty') in difficulty_levels]
        
        return filtered

    def _get_letter_grade(self, score: float) -> str:
        """Convert numeric score to letter grade using config thresholds"""
        thresholds = self.config.phase4.score_thresholds
        
        if score >= thresholds["excellent"]["min"]:
            return "A (Excellent)"
        elif score >= thresholds["good"]["min"]:
            return "B (Good)" 
        elif score >= thresholds["fair"]["min"]:
            return "C (Fair)"
        else:
            return "F (Poor)"
    
    def _get_score_classification(self, score: float) -> str:
        """Classify score using config thresholds"""
        thresholds = self.config.phase4.score_thresholds
        
        if score >= thresholds["excellent"]["min"]:
            return "excellent"
        elif score >= thresholds["good"]["min"]:
            return "good"
        elif score >= thresholds["fair"]["min"]:
            return "fair"
        else:
            return "poor"
    
    def _determine_pass_fail(self, score: float) -> bool:
        """Determine if score represents a pass using config thresholds"""
        # Consider "fair" (2.0+) and above as passing
        return score >= self.config.phase4.score_thresholds["fair"]["min"]

    def _display_category_breakdown(self, summaries: Dict[str, EvaluationSummary]):
        """Display category-wise performance breakdown"""
        
        console.print(Panel.fit("📊 Category Performance Breakdown", style="bold blue"))
        
        # Get all categories
        all_categories = set()
        for summary in summaries.values():
            all_categories.update(summary.category_results.keys())
        
        for category in sorted(all_categories):
            category_table = Table(title=f"{category.replace('_', ' ').title()} Category")
            category_table.add_column("Model", style="bold")
            category_table.add_column("Count", style="dim")
            category_table.add_column("Avg Score", style="green")
            category_table.add_column("Software Engineering", style="purple")
            
            for model_name, summary in summaries.items():
                if category in summary.category_results:
                    data = summary.category_results[category]
                    category_table.add_row(
                        model_name,
                        str(data['count']),
                        f"{data['avg_total_score']:.3f}",
                        f"{data['avg_software_engineering']:.3f}"
                    )
            
            console.print(category_table)

    def _prioritize_scenarios(self, scenarios: List[Dict[str, Any]], priority_order: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """Prioritize scenarios based on difficulty, category, or custom priority"""
        
        if not priority_order:
            # Default priority: expert > hard > medium > easy (run challenging scenarios first)
            difficulty_priority = {
                'expert': 4,
                'hard': 3,
                'medium': 2,
                'easy': 1
            }
            
            # Category priority (can be customized)
            category_priority = {
                'architectural_understanding': 5,
                'integration_testing': 4,
                'multi_session_development': 4,
                'feature_implementation': 3,
                'code_comprehension': 2,
                'bug_investigation': 2,
                'security_analysis': 1,
                'performance_optimization': 1
            }
        else:
            difficulty_priority = priority_order.get('difficulty', {})
            category_priority = priority_order.get('category', {})
        
        def scenario_priority(scenario):
            difficulty = scenario.get('difficulty', 'medium')
            category = scenario.get('task_category', 'bug_investigation')
            
            diff_score = difficulty_priority.get(difficulty, 0)
            cat_score = category_priority.get(category, 0)
            
            # Combine scores (difficulty weighted higher)
            return (diff_score * 10) + cat_score
        
        # Sort by priority (highest first)
        prioritized = sorted(scenarios, key=scenario_priority, reverse=True)
        
        console.print(f"📋 Prioritized {len(prioritized)} scenarios (expert/hard → easy, architectural → performance)")
        return prioritized
    
    def _create_evaluation_queue(self, models: List[str], scenarios: List[Dict[str, Any]], 
                                prioritize: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        """Create an optimized evaluation queue with priority scheduling"""
        
        if prioritize:
            scenarios = self._prioritize_scenarios(scenarios)
        
        # Create all model-scenario combinations
        queue = []
        for scenario in scenarios:
            for model in models:
                queue.append((model, scenario))
        
        if prioritize:
            # Further optimize: group by scenario to minimize context switching
            # This runs all models on the same scenario before moving to next scenario
            queue_by_scenario = {}
            for model, scenario in queue:
                scenario_id = scenario.get('id', '')
                if scenario_id not in queue_by_scenario:
                    queue_by_scenario[scenario_id] = []
                queue_by_scenario[scenario_id].append((model, scenario))
            
            # Flatten back to queue, maintaining scenario priority
            optimized_queue = []
            for scenario in scenarios:  # scenarios are already prioritized
                scenario_id = scenario.get('id', '')
                if scenario_id in queue_by_scenario:
                    optimized_queue.extend(queue_by_scenario[scenario_id])
            
            queue = optimized_queue
        
        console.print(f"📊 Created evaluation queue: {len(queue)} total evaluations")
        return queue 


def run_evaluation(config: Config, models: Optional[List[str]] = None, 
                  categories: Optional[List[str]] = None, 
                  difficulty: Optional[str] = None,
                  resume: bool = True,
                  parallel: bool = False,
                  max_concurrent_models: int = 2,
                  max_concurrent_scenarios: int = 1) -> Dict[str, Any]:
    """Main evaluation function called by CLI"""
    
    async def _async_evaluation():
        # Load scenarios from Phase 3
        scenarios_dir = Path(config.data.output_dir) / "scenarios"
        if not scenarios_dir.exists():
            raise FileNotFoundError("No scenarios found. Run Phase 3 first!")
        
        # Load all scenarios
        all_scenarios = []
        for scenario_file in scenarios_dir.glob("*.json"):
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
                # Each file contains a single scenario object, not an array
                all_scenarios.append(scenario_data)
        
        if not all_scenarios:
            raise ValueError("No scenarios found in scenario files!")
        
        # Default models if none specified
        if not models:
            available_models = ['openai-o3', 'gemini-2.5-pro']
        else:
            available_models = list(models)
        
        # Convert difficulty to list if specified
        difficulty_levels = [difficulty] if difficulty else None
        
        # For multiple models, evaluate each one separately with its own checkpoint
        all_results = {}
        
        for model_name in available_models:
            console.print(f"\n🤖 Evaluating model: {model_name}")
            
            # Create model-specific evaluator with model-specific checkpoint files
            evaluator = LoCoBenchEvaluator(config, model_name=model_name)
            
            # Evaluate this model
            model_results = await evaluator.evaluate_models(
                [model_name], all_scenarios, categories, difficulty_levels, 
                max_concurrent_scenarios=max_concurrent_scenarios, resume=resume
            )
            
            # Merge results
            all_results.update(model_results)
        
        results = all_results
        
        # Generate summaries using the last evaluator instance (any will work for summary generation)
        if available_models:
            last_evaluator = LoCoBenchEvaluator(config)  # Use generic evaluator for summary
            summaries = last_evaluator.generate_evaluation_summary(results)
            
            # Load all results into the evaluator for comprehensive summary generation
            last_evaluator.results = []
            for model_results in results.values():
                last_evaluator.results.extend(model_results)
                
            # Return the evaluator with all results loaded
            return_evaluator = last_evaluator
        else:
            summaries = {}
            return_evaluator = evaluator  # Fallback to last model evaluator
        
        return {
            'evaluator': return_evaluator,
            'results': results,
            'summaries': summaries,
            'success': True
        }
    
    # Run the async evaluation
    try:
        return asyncio.run(_async_evaluation())
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': {},
            'summaries': {}
        } 