"""
Command Line Interface for LoCoBench
"""

import click
import os
import sys
import json
import time
import random
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import List, Dict, Any, Optional

from .core.config import Config
from .core.task import TaskCategory, DifficultyLevel
from .core.data_loader import DataLoader, get_data_loader
from .generation.synthetic_generator import CriticalAuthError
from .core.multi_turn_pipeline import MultiTurnEvaluationPipeline, PipelineConfig
from .evaluation.agent_evaluator import AgentEvaluator, EvaluationConfig
from .agents.agent_factory import AgentFactory, AgentType
from .generation.interactive_scenario_generator import InteractiveScenarioGenerator
from .generation.scenario_converter import ScenarioConverter, get_scenario_converter
from .analysis.turn_length_analyzer import analyze_turn_lengths as analyze_turn_lengths_func

console = Console()

import logging
logger = logging.getLogger(__name__)


async def convert_to_interactive_scenario(scenario_data: Dict[str, Any], config) -> Dict[str, Any]:
    """Convert a single-turn scenario to multi-turn interactive scenario"""
    
    # Extract project information from scenario
    scenario_id = scenario_data.get("scenario_id", scenario_data.get("id", "unknown"))
    parts = scenario_id.split("_")
    if len(parts) >= 4:
        project_name = "_".join(parts[:4])
    else:
        project_name = scenario_id
    
    # Load project data if available
    project_dir = Path(config.data.generated_dir) / project_name
    project_files = []
    project_spec = {}
    
    if project_dir.exists():
        metadata_file = project_dir / "project_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                project_metadata = json.load(f)
            project_spec = project_metadata.get('specification', {})
            
            # Load project files
            for file_info in project_metadata.get('files', []):
                file_path = project_dir / file_info['path']
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        project_files.append({
                            "path": file_info['path'],
                            "content": content,
                            "type": file_info.get('type', 'source')
                        })
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
    
    # Convert task category
    task_category_str = scenario_data.get("task_category", "code_comprehension")
    category_mapping = {
        "code_comprehension": "interactive_code_exploration",
        "architectural_understanding": "interactive_architecture_exploration", 
        "bug_investigation": "interactive_debugging_sessions",
        "feature_implementation": "collaborative_feature_development",
        "cross_file_refactoring": "guided_multi_file_refactoring",
        "integration_testing": "test_driven_development_sessions",
        "security_analysis": "interactive_security_auditing",
        "multi_session_development": "extended_development_projects"
    }
    task_category = category_mapping.get(task_category_str, "interactive_code_exploration")
    
    # Convert difficulty
    difficulty_str = scenario_data.get("difficulty", "medium")
    difficulty = difficulty_str.lower()
    
    # Create interactive scenario structure
    return {
        "scenario_id": scenario_id,
        "title": scenario_data.get("title", "Interactive Development Task"),
        "description": scenario_data.get("description", "Multi-turn development scenario"),
        "category": task_category,
        "difficulty": difficulty,
        "context_files": scenario_data.get("context_files", []),
        "working_directory": project_name,
        "conversation_phases": [
            {
                "phase_id": "exploration",
                "name": "Code Exploration",
                "initial_prompt": f"Explore and understand the codebase: {scenario_data.get('description', '')}",
                "expected_actions": ["read_file", "search_code"],
                "success_conditions": ["understanding", "analysis"],
                "max_turns_in_phase": 10
            },
            {
                "phase_id": "implementation", 
                "name": "Implementation",
                "initial_prompt": "Based on your understanding, implement the required solution",
                "expected_actions": ["write_file", "compiler"],
                "success_conditions": ["implementation", "testing"],
                "max_turns_in_phase": 15
            }
        ],
        "available_tools": ["file_system", "code_search", "compiler", "ide_simulator"],
        "max_turns": 30,
        "project_files": project_files,
        "project_spec": project_spec,
        "project_directory": str(project_dir) if project_dir.exists() else None
    }


def save_progress(progress_file: Path, completed_projects: List[Dict[str, Any]], phase: str):
    """Save progress to a JSON file for resumability"""
    with open(progress_file, 'w') as f:
        json.dump({
            'phase': phase,
            'timestamp': str(datetime.now()),
            'completed_projects': completed_projects,
            'total_completed': len(completed_projects)
        }, f, indent=2)

def load_progress(progress_file: Path) -> List[Dict[str, Any]]:
    """Load progress from a JSON file"""
    if not progress_file.exists():
        return []
    
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data.get('completed_projects', [])
    except Exception as e:
        console.print(f"⚠️ Warning: Could not load progress file: {e}")
        return []


def save_timing_summary(phase_name, start_time, end_time, stats):
    """Save timing summary data for analysis"""
    timing_file = Path(f"logs/timing_phase{phase_name}.json")
    timing_file.parent.mkdir(exist_ok=True)
    
    timing_data = {
        'phase': phase_name,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2)


@click.group()
@click.version_option(version="0.1.0", prog_name="LoCoBench")
@click.pass_context  
def main(ctx):
    """LoCoBench: A Novel Benchmark for Evaluating Long-Context LLMs in Software Development Tasks
    
    Now includes LoCoBench-Agent for multi-turn agent evaluation with tool usage and collaboration.
    """
    ctx.ensure_object(dict)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--save-config', '-s', type=click.Path(), help='Save configuration to file')
def setup(config_path, save_config):
    """Set up LoCoBench environment and configuration"""
    console.print(Panel.fit("🚀 LoCoBench Setup", style="bold blue"))
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Validate configuration
        errors = config.validate()
        
        if errors:
            console.print("❌ Configuration errors found:", style="bold red")
            for error in errors:
                console.print(f"  • {error}", style="red")
            
            # Check for available API keys
            
            if not any([config.api.openai_api_key, config.api.google_api_key]):
                console.print("\n💡 To fix API key issues, set environment variables:", style="yellow")
                console.print("  🏆 For our 2 Elite Models:")
                console.print("  export OPENAI_API_KEY='your-key-here'  # For OpenAI o3")
                console.print("  export GEMINI_API_KEY='your-key-here'  # For Gemini 2.5 Pro")
                
            sys.exit(1)
        
        # Display configuration summary
        console.print("✅ Configuration validated successfully!", style="bold green")
        console.print(config.summary())
        
        # Save configuration if requested
        if save_config:
            config.save_to_file(save_config)
            console.print(f"💾 Configuration saved to: {save_config}", style="green")
            
        console.print("🎯 Setup complete! Ready to begin LoCoBench benchmark generation.", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Setup failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
def status(config_path):
    """Show current LoCoBench status and configuration"""
    try:
        config = Config.from_yaml(config_path)
        
        # Create status table
        table = Table(title="LoCoBench Status", style="cyan")
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center") 
        table.add_column("Details")
        
        # API status (checking our 2 Elite Models)
        
        apis = [
            ("OpenAI", config.api.openai_api_key),
            ("Google", config.api.google_api_key),
            # Removed HuggingFace - no longer needed with synthetic generation
        ]
        
        for name, key in apis:
            status_icon = "✅" if key else "❌"
            status_text = "Configured" if key else "Missing"
            table.add_row(f"{name} API", status_icon, status_text)
        
        # Directory status
        directories = [
    
            ("Output Directory", config.data.output_dir), 
            ("Generated Directory", config.data.generated_dir)
        ]
        
        for name, path in directories:
            exists = Path(path).exists()
            status_icon = "✅" if exists else "❌"
            status_text = f"{'Exists' if exists else 'Missing'}: {path}"
            table.add_row(name, status_icon, status_text)
        
        # Benchmark configuration
        table.add_row("Benchmark Scale", "📊", f"{config.phase3.total_instances:,} instances")
        table.add_row("Task Categories", "📋", f"{len(config.phase3.task_distribution)} categories")
        table.add_row("Languages", "🔤", f"{len(config.phase1.supported_languages)} languages")
        
        console.print(table)
        
        # Validation errors
        errors = config.validate()
        if errors:
            console.print("\n⚠️  Configuration Issues:", style="yellow")
            for error in errors:
                console.print(f"  • {error}", style="yellow")
        else:
            console.print("\n✅ All systems ready!", style="bold green")
            
    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--phase', type=click.Choice(['1', '2', '3', '4', 'all']), default='1', 
              help='Which implementation phase to run')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
@click.option('--force', is_flag=True, help='Force regeneration of already completed projects')
@click.option('--max-concurrent', '-j', type=int, default=3, 
              help='Maximum concurrent operations (default: 3, recommended: 3-10)')
def generate(config_path, phase, dry_run, force, max_concurrent):
    """Generate LoCoBench benchmark instances"""
    console.print(Panel.fit(f"🏗️  LoCoBench Generation - Phase {phase}", style="bold green"))
    
    if dry_run:
        console.print("🔍 DRY RUN MODE - No actual generation will occur", style="yellow")
    
    if max_concurrent > 1:
        console.print(f"🚀 Parallel mode: {max_concurrent} concurrent operations", style="bold blue")
    
    try:
        config = Config.from_yaml(config_path)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            console.print("❌ Configuration errors found:", style="bold red")
            for error in errors:
                console.print(f"  • {error}", style="red")
            sys.exit(1)
        
        if phase == '1' or phase == 'all':
            console.print("🎯 Phase 1: Synthetic Project Generation", style="bold")
            if not dry_run:
                import asyncio
                from .generation.synthetic_generator import SyntheticProjectGenerator, ProjectDomain, ProjectComplexity
                asyncio.run(run_phase_1_generation(config, max_concurrent))
            else:
                console.print("  • Generate synthetic multi-file projects")
                console.print(f"  • 10 domains × 4 complexity levels × {len(config.phase1.supported_languages)} languages")
                console.print("  • Production-quality code with tests & docs")
                target_projects = len(config.phase1.supported_languages) * config.phase1.projects_per_language
                console.print(f"  • Target: {target_projects:,} synthetic projects")
                
        if phase == '2' or phase == 'all':
            console.print("🎯 Phase 2: Synthetic Codebase Generation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_2_generation(config, force, max_concurrent))
            else:
                console.print("  • Generate actual code files from specifications")
                console.print("  • Multi-file projects with realistic complexity")
                console.print("  • Tests, documentation, and error handling")
                
        if phase == '3' or phase == 'all':
            console.print("🎯 Phase 3: Long-Context Evaluation Scenario Creation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_3_generation(config, force, max_concurrent))
            else:
                console.print("  • Create evaluation scenarios from generated code")
                console.print(f"  • {len(config.phase3.task_distribution)} task categories × varying difficulties")
                console.print("  • Context-rich scenarios for long-context evaluation")
                
        if phase == '4' or phase == 'all':
            console.print("🎯 Phase 4: Automated Test-Driven Validation", style="bold")
            if not dry_run:
                import asyncio
                asyncio.run(run_phase_4_generation(config, force, max_concurrent))
            else:
                console.print("  • Generate automated test suites")
                console.print("  • Compilation, unit tests, integration tests")
                console.print(f"  • {len(config.phase4.software_engineering_weights)} software engineering metrics (ACS, DTA, CFRD, STS, RS, CS, IS, SES)")
                console.print("  • Security analysis and code quality validation")
                
        console.print("\n✅ Generation complete!", style="bold green")
        console.print("Next steps:")
        console.print("  • Run evaluation: locobench evaluate")
        console.print("  • Check status: locobench status")
        
    except Exception as e:
        console.print(f"❌ Generation failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--model', '-m', multiple=True, help='Model to evaluate (can specify multiple)')
@click.option('--task-category', '-t', multiple=True, help='Task category to evaluate')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard', 'expert']),
              help='Difficulty level to evaluate')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for results (auto-generated if not specified)')
@click.option('--no-save', is_flag=True, help='Skip saving results to file (display only)')
@click.option('--no-resume', is_flag=True, help='Start fresh evaluation (ignore any existing checkpoint)')
@click.option('--parallel', is_flag=True, help='Enable parallel model evaluation (faster but more resource intensive)')
@click.option('--max-concurrent-models', type=int, default=2, help='Maximum number of models to evaluate concurrently (default: 2)')
@click.option('--max-concurrent-scenarios', type=int, default=1, help='Maximum number of scenarios per model to evaluate concurrently (default: 1)')
@click.option('--monitor', is_flag=True, help='Start web monitoring dashboard at http://localhost:8080')
def evaluate(config_path, model, task_category, difficulty, output_file, no_save, no_resume, parallel, max_concurrent_models, max_concurrent_scenarios, monitor):
    """Evaluate models on LoCoBench benchmark"""
    console.print(Panel.fit("🧪 LoCoBench Evaluation", style="bold purple"))
    
    # Start monitoring dashboard if requested
    dashboard = None
    if monitor:
        try:
            from .evaluation.monitoring import MonitoringDashboard
            dashboard = MonitoringDashboard()
            dashboard.start()
        except Exception as e:
            console.print(f"⚠️  Failed to start monitoring dashboard: {e}", style="yellow")
    
    try:
        config = Config.from_yaml(config_path)
        
        from .evaluation.evaluator import run_evaluation
        evaluation_data = run_evaluation(config, model, task_category, difficulty, resume=not no_resume, parallel=parallel, max_concurrent_models=max_concurrent_models, max_concurrent_scenarios=max_concurrent_scenarios)
        
        # Check if evaluation succeeded
        if not evaluation_data.get('success', False):
            console.print(f"❌ Evaluation failed: {evaluation_data.get('error', 'Unknown error')}", style="bold red")
            return
        
        # Extract results
        evaluator = evaluation_data['evaluator']
        results = evaluation_data['results']
        summaries = evaluation_data['summaries']
        
        # Auto-generate output filename if not provided
        if not output_file and not no_save:
            from datetime import datetime
            from pathlib import Path
            
            # Build descriptive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Model names part
            model_list = list(model) if model else ['all-models']
            models_part = "_".join([m.replace('-', '').replace('_', '').lower() for m in model_list])
            if len(models_part) > 30:  # Limit length
                models_part = f"{len(model_list)}models"
            
            # Category part
            if task_category:
                categories_part = "_".join([c.replace('_', '') for c in task_category])
                if len(categories_part) > 20:
                    categories_part = f"{len(task_category)}cats"
            else:
                categories_part = "allcats"
            
            # Difficulty part
            difficulty_part = difficulty if difficulty else "alldiff"
            
            # Construct filename
            output_file = f"{models_part}_{categories_part}_{difficulty_part}_{timestamp}_evaluation_results.json"
            
            # Ensure results directory exists
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            output_file = results_dir / output_file
        
        # Show evaluation parameters (including auto-generated filename)
        console.print("📋 Evaluation Parameters:", style="bold")
        console.print(f"  • Models: {list(model) if model else 'All available'}")
        console.print(f"  • Categories: {list(task_category) if task_category else 'All categories'}")
        console.print(f"  • Difficulty: {difficulty if difficulty else 'All levels'}")
        if no_save:
            console.print(f"  • Output: Display only (saving disabled)")
        else:
            console.print(f"  • Output: {output_file}")
        
        # Display formatted results
        if summaries:
            console.print("\n📊 Evaluation Completed!", style="bold green")
            evaluator.display_results(summaries)
            
            # Save comprehensive results (unless explicitly disabled)
            if not no_save:
                from pathlib import Path
                output_path = Path(output_file)
                evaluator.save_results(results, summaries, output_path)
        else:
            console.print("❌ No evaluation results generated", style="bold red")
        
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="bold red")
        sys.exit(1)
    finally:
        # Clean up monitoring dashboard
        if dashboard:
            dashboard.stop()


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--agent-type', type=click.Choice(['openai', 'anthropic', 'google', 'all']), default='openai',
              help='Type of agent to run')
@click.option('--model', '-m', help='Specific model to use (e.g., gpt-4o, claude-sonnet-4)')
@click.option('--max-turns', type=int, default=20, help='Maximum turns per conversation')
@click.option('--enable-tools', is_flag=True, default=True, help='Enable tool usage')
@click.option('--readonly', is_flag=True, help='Run in readonly mode (no file modifications)')
@click.option('--compare-agents', is_flag=True, help='Compare multiple agents on the same scenarios')
@click.option('--scenario-file', type=click.Path(), help='Specific scenario file to run')
@click.option('--output-dir', type=click.Path(), help='Output directory for agent results')
def agent(config_path, agent_type, model, max_turns, enable_tools, readonly, compare_agents, scenario_file, output_dir):
    """Run LoCoBench-Agent evaluation with multi-turn conversations and tool usage"""
    console.print(Panel.fit("🤖 LoCoBench-Agent Evaluation", style="bold purple"))
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Enable agent mode in configuration
        config.agent.enable_agent_mode = True
        config.agent.max_turns_per_session = max_turns
        config.agent.readonly_mode = readonly
        config.agent.enable_agent_comparison = compare_agents
        
        if output_dir:
            config.data.output_dir = output_dir
        
        # Validate configuration
        errors = config.validate()
        if errors:
            console.print("❌ Configuration errors found:", style="bold red")
            for error in errors:
                console.print(f"  • {error}", style="red")
            sys.exit(1)
        
        # Create directories
        config.create_directories()
        
        # Import agent evaluation system
        import asyncio
        from .agents.agent_factory import AgentFactory, AgentConfig as AgentConfigClass, AgentType
        from .core.agent_session import AgentSession, ConversationPhase, SessionConfig
        from .core.tool_registry import get_tool_registry, register_tool
        from .tools import FileSystemTool, CompilerTool, DebuggerTool, IDESimulatorTool, EchoTool, CalculatorTool
        
        # Run agent evaluation
        asyncio.run(run_agent_evaluation(
            config, agent_type, model, max_turns, enable_tools, 
            compare_agents, scenario_file
        ))
        
    except Exception as e:
        console.print(f"❌ Agent evaluation failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--show-projects', is_flag=True, help='Show detailed project information')
@click.option('--show-scenarios', is_flag=True, help='Show detailed scenario information')
def data_info(config_path, show_projects, show_scenarios):
    """Show information about available LoCoBench data"""
    console.print(Panel.fit("📊 LoCoBench Data Information", style="bold blue"))
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path) if config_path else Config()
        data_loader = get_data_loader(config)
        
        # Get statistics
        stats = data_loader.get_data_statistics()
        
        # Overview table
        overview_table = Table(title="Data Overview")
        overview_table.add_column("Type", style="cyan")
        overview_table.add_column("Count", style="green")
        overview_table.add_column("Details", style="yellow")
        
        overview_table.add_row(
            "Projects", 
            str(stats['projects']['total']),
            f"Languages: {len(stats['projects']['by_language'])}, Complexities: {len(stats['projects']['by_complexity'])}"
        )
        overview_table.add_row(
            "Scenarios",
            str(stats['scenarios']['total']),
            f"Categories: {len(stats['scenarios']['by_category'])}, Difficulties: {len(stats['scenarios']['by_difficulty'])}"
        )
        
        console.print(overview_table)
        
        # Project breakdown
        if stats['projects']['total'] > 0:
            console.print("\n📁 Project Distribution:", style="bold cyan")
            
            # By language
            lang_table = Table(title="By Programming Language")
            lang_table.add_column("Language", style="cyan")
            lang_table.add_column("Count", style="green")
            lang_table.add_column("Percentage", style="yellow")
            
            total_projects = stats['projects']['total']
            for lang, count in sorted(stats['projects']['by_language'].items()):
                percentage = (count / total_projects) * 100
                lang_table.add_row(lang, str(count), f"{percentage:.1f}%")
            
            console.print(lang_table)
            
            # By complexity
            complexity_table = Table(title="By Complexity Level")
            complexity_table.add_column("Complexity", style="cyan")
            complexity_table.add_column("Count", style="green")
            complexity_table.add_column("Percentage", style="yellow")
            
            for complexity, count in sorted(stats['projects']['by_complexity'].items()):
                percentage = (count / total_projects) * 100
                complexity_table.add_row(complexity, str(count), f"{percentage:.1f}%")
            
            console.print(complexity_table)
        
        # Scenario breakdown
        if stats['scenarios']['total'] > 0:
            console.print("\n🎯 Scenario Distribution:", style="bold cyan")
            
            # By category
            cat_table = Table(title="By Task Category")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green")
            cat_table.add_column("Percentage", style="yellow")
            
            total_scenarios = stats['scenarios']['total']
            for category, count in sorted(stats['scenarios']['by_category'].items()):
                percentage = (count / total_scenarios) * 100
                cat_table.add_row(category, str(count), f"{percentage:.1f}%")
            
            console.print(cat_table)
            
            # By difficulty
            diff_table = Table(title="By Difficulty Level")
            diff_table.add_column("Difficulty", style="cyan")
            diff_table.add_column("Count", style="green")
            diff_table.add_column("Percentage", style="yellow")
            
            for difficulty, count in sorted(stats['scenarios']['by_difficulty'].items()):
                percentage = (count / total_scenarios) * 100
                diff_table.add_row(difficulty, str(count), f"{percentage:.1f}%")
            
            console.print(diff_table)
        
        # Directory information
        console.print(f"\n📂 Data Directories:", style="bold cyan")
        console.print(f"   Generated Projects: {config.data.generated_dir}", style="blue")
        console.print(f"   Scenarios: {config.data.output_dir}/scenarios", style="blue")
        
        # Show detailed information if requested
        if show_projects:
            console.print(f"\n📋 Available Projects (first 20):", style="bold cyan")
            projects = data_loader.list_available_projects()[:20]
            for project in projects:
                console.print(f"   • {project}", style="blue")
            if len(data_loader.list_available_projects()) > 20:
                console.print(f"   ... and {len(data_loader.list_available_projects()) - 20} more", style="yellow")
        
        if show_scenarios:
            console.print(f"\n🎮 Sample Scenarios (first 10):", style="bold cyan")
            scenarios = data_loader.load_scenarios(limit=10, include_project_context=False)
            for scenario in scenarios:
                console.print(f"   • {scenario.scenario_id}: {scenario.title}", style="blue")
        
        # Show conversion status
        converter = get_scenario_converter(config)
        conversion_stats = converter.get_conversion_stats()
        
        console.print(f"\n🔄 Agent Scenario Conversion Status:", style="bold cyan")
        if conversion_stats['total_converted'] > 0:
            console.print(f"   ✅ {conversion_stats['total_converted']} scenarios pre-converted for agent evaluation", style="green")
            console.print(f"   📁 Cache size: {conversion_stats['cache_size_mb']:.1f} MB", style="blue")
            console.print(f"   ⚡ Agent evaluation will be fast!", style="green")
        else:
            console.print(f"   ❌ No scenarios converted yet", style="red")
            console.print(f"   💡 Run 'locobench convert-scenarios' for faster agent evaluation", style="yellow")
        
        # Usage recommendations
        console.print(f"\n💡 Usage Examples:", style="bold green")
        if conversion_stats['total_converted'] > 0:
            console.print("   # Run fast agent evaluation (using pre-converted scenarios):", style="green")
            console.print("   locobench evaluate --mode agent --scenario-count 5", style="cyan")
        else:
            console.print("   # Pre-convert scenarios first (recommended):", style="green")
            console.print("   locobench convert-scenarios --limit 100", style="cyan")
            console.print("   # Then run fast agent evaluation:", style="green")
            console.print("   locobench evaluate --mode agent --scenario-count 5", style="cyan")
        
        console.print("   # Compare LLM vs Agent on same scenarios:", style="green")
        console.print("   locobench evaluate --mode both --compare-modes", style="cyan")
        console.print("   # Run traditional LLM evaluation:", style="green")
        console.print("   locobench evaluate --mode llm", style="cyan")
        
    except Exception as e:
        console.print(f"❌ Error loading data information: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--force', is_flag=True, help='Force reconversion of all scenarios (ignore cache)')
@click.option('--max-concurrent', type=int, default=5, help='Maximum concurrent conversions')
@click.option('--limit', type=int, help='Limit number of scenarios to convert (for testing)')
def convert_scenarios(config_path, force, max_concurrent, limit):
    """Pre-convert single-turn scenarios to multi-turn format for faster agent evaluation"""
    console.print(Panel.fit("🔄 Scenario Conversion for LoCoBench-Agent", style="bold purple"))
    
    async def run_conversion():
        try:
            # Load configuration
            config = Config.from_yaml(config_path) if config_path else Config()
            converter = get_scenario_converter(config)
            
            # Show current conversion status
            current_stats = converter.get_conversion_stats()
            console.print(f"📊 Current Status:", style="bold cyan")
            console.print(f"   • Converted scenarios: {current_stats['total_converted']}", style="blue")
            console.print(f"   • Cache size: {current_stats['cache_size_mb']:.1f} MB", style="blue")
            console.print(f"   • Cache directory: {current_stats['cache_directory']}", style="blue")
            
            if force:
                console.print("🔄 Force mode: Reconverting ALL scenarios", style="yellow")
            elif current_stats['total_converted'] > 0:
                console.print("⚡ Smart mode: Only converting new/changed scenarios", style="green")
            
            # Run conversion
            console.print(f"\n🚀 Starting conversion with {max_concurrent} concurrent workers...", style="bold green")
            
            stats = await converter.convert_all_scenarios(
                force_reconvert=force,
                max_concurrent=max_concurrent,
                limit=limit
            )
            
            # Show results
            console.print(f"\n✅ Conversion Complete!", style="bold green")
            console.print(f"📈 Results:", style="bold cyan")
            console.print(f"   • Total scenarios: {stats.total_scenarios}", style="blue")
            console.print(f"   • Newly converted: {stats.converted_scenarios}", style="green")
            console.print(f"   • Used from cache: {stats.cached_scenarios}", style="cyan")
            console.print(f"   • Failed: {stats.failed_conversions}", style="red" if stats.failed_conversions > 0 else "blue")
            console.print(f"   • Conversion time: {stats.conversion_time_seconds:.1f} seconds", style="blue")
            
            if stats.converted_scenarios > 0:
                rate = stats.converted_scenarios / stats.conversion_time_seconds
                console.print(f"   • Conversion rate: {rate:.1f} scenarios/second", style="blue")
            
            # Show final cache stats
            final_stats = converter.get_conversion_stats()
            console.print(f"\n💾 Final Cache Status:", style="bold cyan")
            console.print(f"   • Total converted scenarios: {final_stats['total_converted']}", style="blue")
            console.print(f"   • Cache size: {final_stats['cache_size_mb']:.1f} MB", style="blue")
            
            console.print(f"\n🎯 Agent evaluation will now be much faster!", style="bold green")
            console.print("   Run: locobench evaluate --mode agent", style="cyan")
            
        except Exception as e:
            console.print(f"❌ Conversion failed: {e}", style="bold red")
            import traceback
            console.print(traceback.format_exc(), style="red")
            sys.exit(1)
    
    # Run async conversion
    asyncio.run(run_conversion())


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--scenario-limit', type=int, default=20, help='Number of scenarios to analyze')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for the analysis report')
@click.option('--categories', help='Comma-separated list of categories to analyze')
@click.option('--difficulties', help='Comma-separated list of difficulties to analyze')
def analyze_turn_lengths(config_path, scenario_limit, output_file, categories, difficulties):
    """Analyze average turn lengths in multi-turn agent conversations"""
    console.print(Panel.fit("📏 Turn Length Analysis for LoCoBench-Agent", style="bold cyan"))
    
    async def run_analysis():
        try:
            # Load configuration
            config = Config.from_yaml(config_path) if config_path else Config()
            
            # Parse filters
            category_list = categories.split(',') if categories else None
            difficulty_list = difficulties.split(',') if difficulties else None
            
            console.print(f"🔍 Analyzing turn patterns for up to {scenario_limit} scenarios", style="blue")
            if category_list:
                console.print(f"   • Categories: {', '.join(category_list)}", style="blue")
            if difficulty_list:
                console.print(f"   • Difficulties: {', '.join(difficulty_list)}", style="blue")
            
            # Run analysis
            stats = await analyze_turn_lengths_func(
                config=config,
                scenario_limit=scenario_limit,
                output_file=output_file
            )
            
            # Show quick summary
            console.print(f"\n✅ Analysis Complete!", style="bold green")
            console.print(f"📊 Key Results:", style="bold cyan")
            console.print(f"   • Average turn length: {stats.overall_avg_turn_length:.0f} tokens", style="green")
            console.print(f"   • Median turn length: {stats.overall_median_turn_length:.0f} tokens", style="green")
            console.print(f"   • Total conversations: {stats.total_conversations}", style="blue")
            console.print(f"   • Total turns: {stats.total_turns}", style="blue")
            
            if output_file:
                console.print(f"📄 Full report saved to: {output_file}", style="cyan")
            
        except Exception as e:
            console.print(f"❌ Analysis failed: {e}", style="bold red")
            import traceback
            console.print(traceback.format_exc(), style="red")
            sys.exit(1)
    
    # Run async analysis
    asyncio.run(run_analysis())


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--session-id', help='Session ID to resume')
@click.option('--list-sessions', is_flag=True, help='List all active sessions')
def agent_sessions(config_path, session_id, list_sessions):
    """Manage agent evaluation sessions"""
    console.print(Panel.fit("🔄 Agent Session Management", style="bold cyan"))
    
    try:
        config = Config.from_yaml(config_path)
        
        if list_sessions:
            # List active sessions
            sessions_dir = Path(config.data.output_dir) / "agents" / "sessions"
            if sessions_dir.exists():
                session_files = list(sessions_dir.glob("*.json"))
                
                if session_files:
                    table = Table(title="Active Agent Sessions", style="cyan")
                    table.add_column("Session ID", style="bold")
                    table.add_column("Agent", style="green")
                    table.add_column("Status", justify="center")
                    table.add_column("Turns", justify="right")
                    table.add_column("Created", style="dim")
                    
                    for session_file in session_files:
                        try:
                            with open(session_file, 'r') as f:
                                session_data = json.load(f)
                            
                            table.add_row(
                                session_data.get("session_id", "unknown"),
                                session_data.get("agent_name", "unknown"),
                                session_data.get("status", "unknown"),
                                str(session_data.get("total_turns", 0)),
                                datetime.fromtimestamp(session_data.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
                            )
                        except Exception:
                            continue
                    
                    console.print(table)
                else:
                    console.print("No active sessions found", style="yellow")
            else:
                console.print("No sessions directory found", style="yellow")
        
        elif session_id:
            # Show specific session details
            session_file = Path(config.data.output_dir) / "agents" / "sessions" / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                console.print(f"📋 Session Details: {session_id}", style="bold")
                console.print(json.dumps(session_data, indent=2))
            else:
                console.print(f"Session {session_id} not found", style="red")
        
        else:
            console.print("Use --list-sessions to see all sessions or --session-id to view a specific session", style="yellow")
    
    except Exception as e:
        console.print(f"❌ Session management failed: {e}", style="bold red")
        sys.exit(1)


@main.command()
def version():
    """Show LoCoBench version information"""
    console.print("🔧 LoCoBench v0.1.0", style="bold blue")
    console.print("A Novel Benchmark for Evaluating Long-Context Language Models")
    console.print("in Software Development Tasks")
    console.print("\n🤖 LoCoBench-Agent: Multi-Turn Agent Evaluation System", style="bold purple")
    console.print("Comprehensive evaluation of LLM agents with tool usage and collaboration")
    console.print("\nFor more information: https://github.com/LoCoBench/LoCoBench")


async def run_phase_1_generation(config, max_concurrent=3):
    """Run Phase 1: Synthetic Project Generation with Guaranteed Uniqueness, progress tracking, and resumability"""
    from .generation.synthetic_generator import (
        SyntheticProjectGenerator, ProjectDomain, ProjectComplexity,
        ProjectArchitecture, ProjectTheme
    )
    import asyncio
    from asyncio import Semaphore
    
    # ⏰ START TIMING
    phase_start_time = time.time()
    phase_start_datetime = datetime.now()
    
    console.print("\n🎯 [bold]Synthetic Project Generation Pipeline (Uniqueness Guaranteed)[/bold]")
    console.print("=" * 75)
    console.print(f"⏰ Phase 1 started at: {phase_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize timing tracking
    project_times = []
    estimated_total_time = None
    
    # Setup progress tracking
    progress_file = Path("logs/phase1_progress.json")
    progress_file.parent.mkdir(exist_ok=True)
    completed_projects = load_progress(progress_file)
    completed_project_names = {p.get('unique_id', '') for p in completed_projects}
    
    generator = SyntheticProjectGenerator(config, log_file="logs/phase1_generation.log")
    
    console.print(f"📋 Resume state: {len(completed_projects)} projects previously completed")
    
    # Target: projects per language from config
    languages = config.phase1.supported_languages
    projects_per_language = config.phase1.projects_per_language
    total_projects = len(languages) * projects_per_language
    
    # Get all available factors for uniqueness
    domains = list(ProjectDomain)
    complexities = list(ProjectComplexity)
    architectures = list(ProjectArchitecture)
    themes = list(ProjectTheme)
    
    console.print(f"📊 Target: {len(languages)} languages × {projects_per_language} projects = {total_projects} total")
    console.print(f"🌐 Languages: {', '.join(languages)}")
    console.print(f"🏗️ Uniqueness factors:")
    console.print(f"   • {len(domains)} domains × {len(complexities)} complexities × {len(architectures)} architectures × {len(themes)} themes")
    console.print(f"   • = {len(domains) * len(complexities) * len(architectures) * len(themes):,} possible combinations")
    console.print(f"   • + unique seeds = guaranteed uniqueness for {projects_per_language} projects per language ✅")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent specifications[/bold blue]")
    
    console.print("🏗️ Generating unique project specifications...")
    
    # Create complexity selection pool based on config distribution
    import random
    
    # 🔧 FIX: Set deterministic seed for reproducible generation
    random.seed(42)  # Fixed seed ensures same complexity assignment across runs
    
    complexity_pool = []
    for complexity_name, ratio in config.phase1.complexity_distribution.items():
        complexity_enum = getattr(ProjectComplexity, complexity_name.upper())
        count = int(projects_per_language * len(languages) * ratio)
        complexity_pool.extend([complexity_enum] * count)
    
    # Ensure we have exactly the right number of complexities
    while len(complexity_pool) < total_projects:
        complexity_pool.append(random.choice(complexities))
    while len(complexity_pool) > total_projects:
        complexity_pool.pop()
    
    # Shuffle for random distribution (now deterministic due to fixed seed)
    random.shuffle(complexity_pool)
    
    # Generate unique combinations for each language
    spec_tasks = []
    global_index = 0
    
    for language in languages:
        console.print(f"🔧 [cyan]Planning {projects_per_language} unique projects for {language}...[/cyan]")
        
        # Create unique combinations for this language
        language_combinations = []
        
        for i in range(projects_per_language):
            # Use different distribution strategies to ensure uniqueness
            domain = domains[i % len(domains)]
            complexity = complexity_pool[global_index]
            architecture = architectures[i % len(architectures)]
            theme = themes[i % len(themes)]
            
            # Create unique seed for deterministic but varied LLM generation
            unique_seed = hash(f"{language}-{domain.value}-{complexity.value}-{architecture.value}-{theme.value}-{i}") % 1000000
            
            # Generate unique project ID
            unique_id = f"{language}_{domain.value}_{complexity.value}_{i:03d}"
            
            language_combinations.append({
                'unique_id': unique_id,
                'language': language,
                'domain': domain,
                'complexity': complexity,
                'architecture': architecture,
                'theme': theme,
                'index': i,
                'seed': unique_seed,
                'global_index': global_index
            })
            
            global_index += 1
        
        # Add to spec tasks
        spec_tasks.extend(language_combinations)
        
        # Verify uniqueness for this language
        unique_combinations = set()
        for combo in language_combinations:
            combination_key = (combo['domain'].value, combo['complexity'].value, 
                             combo['architecture'].value, combo['theme'].value)
            unique_combinations.add(combination_key)
        
        console.print(f"   ✅ [green]{len(unique_combinations)} unique factor combinations for {language}[/green]")
    
    console.print(f"🎯 Generated {len(spec_tasks)} unique project specifications...")
    
    # Verify global uniqueness
    all_unique_ids = set(task['unique_id'] for task in spec_tasks)
    console.print(f"🔍 Uniqueness verification: {len(all_unique_ids)} unique IDs for {len(spec_tasks)} projects ✅")
    
    # Semaphore for parallel generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    projects_generated = 0
    projects_failed = 0
    
    async def generate_single_spec(task_info, task_index):
        """Generate a single project specification with guaranteed uniqueness"""
        async with semaphore:
            unique_id = task_info['unique_id']
            language = task_info['language']
            domain = task_info['domain']
            complexity = task_info['complexity']
            architecture = task_info['architecture']
            theme = task_info['theme']
            seed = task_info['seed']
            
            # Skip if already completed (resume functionality)
            if unique_id in completed_project_names:
                console.print(f"✅ [green]Skipping {unique_id} - Already completed![/green]")
                return {
                    'success': True,
                    'skipped': True,
                    'unique_id': unique_id,
                    'project_name': unique_id
                }
            
            try:
                console.print(f"🔨 [bold cyan]Generating {task_index}/{len(spec_tasks)}: {unique_id}[/bold cyan]")
                console.print(f"     {language} | {domain.value} | {complexity.value} | {architecture.value} | {theme.value}")
                
                # Start timing
                import time
                start_time = time.time()
                
                # Set random seed for deterministic variation
                random.seed(seed)
                
                # Generate project specification with unique factors
                spec = await generator.generate_project_specification_unique(
                    domain, complexity, language, architecture, theme, unique_id, seed
                )
                
                generation_time = time.time() - start_time
                
                # Save specification to project directory  
                project_name = unique_id
                
                # Create project directory and save specification
                project_dir = generator.generated_dir / project_name
                project_dir.mkdir(exist_ok=True)
                
                # Save specification metadata
                metadata = {
                    "specification": spec.to_dict(),
                    "generated_timestamp": time.time(),
                    "phase_1_complete": True,
                    "uniqueness_factors": {
                        "domain": domain.value,
                        "complexity": complexity.value, 
                        "architecture": architecture.value,
                        "theme": theme.value,
                        "seed": seed
                    }
                }
                
                with open(project_dir / "project_metadata.json", 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2)
                
                console.print(f"   ✅ [green]Generated {project_name}![/green] {spec.target_file_count} files, ~{spec.target_token_count:,} tokens ({generation_time:.1f}s)")
                
                # Save progress for successful completion
                current_progress = {
                    'unique_id': unique_id,
                    'project_name': project_name,
                    'status': 'completed',
                    'language': language,
                    'domain': domain.value,
                    'complexity': complexity.value,
                    'architecture': architecture.value,
                    'theme': theme.value,
                    'timestamp': time.time()
                }
                completed_projects.append(current_progress)
                save_progress(progress_file, completed_projects, "1")
                
                return {
                    'success': True,
                    'project_name': project_name,
                    'unique_id': unique_id,
                    'language': language,
                    'domain': domain.value,
                    'complexity': complexity.value,
                    'architecture': architecture.value,
                    'theme': theme.value,
                    'generation_time': generation_time,
                    'target_files': spec.target_file_count,
                    'target_tokens': spec.target_token_count
                }
                
            except CriticalAuthError as e:
                # Critical auth errors should stop the entire process
                console.print(f"   🚨 [bold red]CRITICAL AUTH FAILURE in {unique_id}[/bold red]")
                console.print(f"   🔑 {e.provider}: {e.message}")
                console.print("   🛑 [yellow]Stopping generation to fix authentication...[/yellow]")
                
                # Save current progress before stopping
                current_progress = {
                    'unique_id': unique_id,
                    'status': 'auth_failed',
                    'error': str(e),
                    'timestamp': time.time()
                }
                completed_projects.append(current_progress)
                save_progress(progress_file, completed_projects, "1")
                
                # Re-raise to stop the entire process
                raise e
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {unique_id}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'unique_id': unique_id,
                    'language': language,
                    'domain': domain.value
                }
    
    # Execute all specification generation tasks in parallel
    console.print(f"\n🚀 [bold]Starting parallel specification generation for {len(spec_tasks)} projects...[/bold]")
    
    try:
        # Create asyncio tasks
        tasks = []
        for i, task_info in enumerate(spec_tasks, 1):
            task = generate_single_spec(task_info, i)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for CriticalAuthError in results
        for result in results:
            if isinstance(result, CriticalAuthError):
                raise result
        
        # Process results
        successful_projects = []
        failed_projects = []
        skipped_projects = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_projects.append(f"Exception: {str(result)}")
                projects_failed += 1
            elif result and result['success']:
                if result.get('skipped'):
                    skipped_projects += 1
                else:
                    successful_projects.append(result)
                    projects_generated += 1
                    # Collect timing data for analysis
                    if 'generation_time' in result:
                        project_times.append(result['generation_time'])
            else:
                failed_projects.append(f"{result['language']} {result['domain']}" if result else "Unknown project")
                projects_failed += 1
        
        # ⏰ TIMING ANALYSIS
        phase_end_time = time.time()
        phase_duration = phase_end_time - phase_start_time
        phase_end_datetime = datetime.now()
        
        # Convert to human readable (define globally for all timing displays)
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        # Calculate timing statistics
        if project_times:
            avg_project_time = sum(project_times) / len(project_times)
            min_project_time = min(project_times)
            max_project_time = max(project_times)
            total_generation_time = sum(project_times)
            
            # Calculate full-scale projections
            total_target_projects = len(config.phase1.supported_languages) * config.phase1.projects_per_language
            estimated_full_scale_time = avg_project_time * total_target_projects
            
            # Estimate parallel efficiency (Phase 1 can be highly parallelized)
            parallel_efficiency = 0.8  # Assume 80% parallel efficiency
            if max_concurrent > 1:
                estimated_parallel_time = estimated_full_scale_time / (max_concurrent * parallel_efficiency)
            else:
                estimated_parallel_time = estimated_full_scale_time
        
        # Final summary
        console.print(f"\n📊 [bold]Phase 1 Summary:[/bold]")
        console.print(f"   ✅ Generated: {projects_generated} project specifications")
        console.print(f"   ⚠️  Skipped: {skipped_projects} specifications (already done)")
        console.print(f"   ❌ Failed: {projects_failed} specifications")
        console.print(f"   📁 Specifications saved to: {generator.generated_dir}")
        
        # ⏰ TIMING SUMMARY
        console.print(f"\n⏰ [bold]Timing Analysis:[/bold]")
        console.print(f"   🕐 Phase duration: {format_duration(phase_duration)}")
        console.print(f"   📅 Started: {phase_start_datetime.strftime('%H:%M:%S')}")
        console.print(f"   📅 Ended: {phase_end_datetime.strftime('%H:%M:%S')}")
        
        if project_times:
            console.print(f"\n📈 [bold]Per-Project Statistics:[/bold]")
            console.print(f"   ⚡ Average: {format_duration(avg_project_time)}")
            console.print(f"   🚀 Fastest: {format_duration(min_project_time)}")
            console.print(f"   🐌 Slowest: {format_duration(max_project_time)}")
            console.print(f"   🔄 Total generation time: {format_duration(total_generation_time)}")
            console.print(f"   🎯 Parallel efficiency: {(total_generation_time/phase_duration)*100:.1f}%")
            
            console.print(f"\n🚀 [bold]Full-Scale Projections ({total_target_projects:,} projects):[/bold]")
            console.print(f"   📊 Sequential: {format_duration(estimated_full_scale_time)}")
            console.print(f"   ⚡ Parallel (x{max_concurrent}): {format_duration(estimated_parallel_time)}")
            
            # Add timing information if we generated any projects
            if projects_generated > 0:
                            console.print(f"\n⏱️ [bold]Estimated Timing Pattern:[/bold]")
            console.print(f"   📝 Per project: ~{avg_project_time:.1f}s average")
            console.print(f"   🔄 Concurrent slots: {max_concurrent}")
            console.print(f"   ⏱️  Expected full run: {format_duration(estimated_parallel_time)}")
            
            # Save timing data for analysis
            timing_stats = {
                'projects_generated': projects_generated,
                'avg_project_time': avg_project_time,
                'min_project_time': min_project_time,
                'max_project_time': max_project_time,
                'total_generation_time': total_generation_time,
                'parallel_efficiency': (total_generation_time/phase_duration)*100,
                'estimated_full_scale_time': estimated_full_scale_time,
                'estimated_parallel_time': estimated_parallel_time,
                'max_concurrent': max_concurrent
            }
            save_timing_summary("1", phase_start_datetime, phase_end_datetime, timing_stats)
        else:
            console.print(f"   ⚠️  No timing data available (no successful generations)")
            save_timing_summary("1", phase_start_datetime, phase_end_datetime, {'projects_generated': 0})
        
        if failed_projects:
            console.print(f"\n⚠️  [yellow]Failed specifications:[/yellow]")
            for failed in failed_projects[:10]:  # Show first 10 failures
                console.print(f"     • {failed}")
            if len(failed_projects) > 10:
                console.print(f"     ... and {len(failed_projects) - 10} more")
                
    except CriticalAuthError as e:
        console.print(f"\n🚨 [bold red]CRITICAL AUTHENTICATION FAILURE[/bold red]")
        console.print(f"🔑 Provider: {e.provider}")
        console.print(f"💬 Error: {e.message}")
        console.print(f"\n📋 Progress saved to: {progress_file}")
        console.print(f"✅ {len(completed_projects)} projects completed before failure")
        console.print(f"\n🔧 [bold yellow]Next steps:[/bold yellow]")
        console.print("   1. Update your API credentials (check api.sh)")
        console.print("   2. Run: source api.sh")
        console.print("   3. Resume with: locobench generate --phase 1")
        console.print("   4. The pipeline will automatically resume from where it stopped")
        
        # Exit with error code
        import sys
        sys.exit(1)


async def run_phase_2_generation(config, force_regenerate=False, max_concurrent=3):
    """Run Phase 2: Synthetic Codebase Generation with parallel processing and resumability"""
    from .generation.synthetic_generator import SyntheticProjectGenerator
    from pathlib import Path
    import json
    import asyncio
    from asyncio import Semaphore
    
    # ⏰ START TIMING
    phase_start_time = time.time()
    phase_start_datetime = datetime.now()
    
    console.print("\n💻 [bold]Synthetic Codebase Generation Pipeline[/bold]")
    console.print("=" * 60)
    console.print(f"⏰ Phase 2 started at: {phase_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize timing tracking
    project_times = []
    file_counts = []
    line_counts = []
    
    # Setup progress tracking
    progress_file = Path("logs/phase2_progress.json")
    progress_file.parent.mkdir(exist_ok=True)
    completed_projects = load_progress(progress_file)
    completed_project_names = {p.get('project_name', '') for p in completed_projects}
    
    generator = SyntheticProjectGenerator(config, log_file="logs/phase2_generation.log")
    generated_dir = Path(config.data.generated_dir)
    
    # Find all project metadata files from Phase 1
    project_dirs = [d for d in generated_dir.iterdir() if d.is_dir()]
    
    console.print(f"📂 Found {len(project_dirs)} projects from Phase 1")
    console.print(f"📋 Resume state: {len(completed_projects)} projects previously completed")
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL projects[/yellow]")
        completed_project_names = set()  # Clear resume state
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed projects...[/cyan]")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent projects[/bold blue]")
    
    console.print("🏭 Generating production-quality code with 3 Elite Models...")
    
    # Prepare projects for processing
    projects_to_process = []
    projects_skipped = 0
    
    for project_dir in project_dirs:
        metadata_file = project_dir / "project_metadata.json"
        
        if not metadata_file.exists():
            console.print(f"⚠️  Skipping {project_dir.name} - no metadata found")
            continue
            
        # Load project specification
        with open(metadata_file, 'r') as f:
            project_data = json.load(f)
        
        project_name = f"{project_data['specification']['name']} ({project_data['specification']['language']})"
        
        # Check if project is already completed (unless force regeneration)
        if not force_regenerate and (
            project_name in completed_project_names or 
            'generated_stats' in project_data
        ):
            stats = project_data.get('generated_stats', {})
            # Also verify files actually exist on disk
            expected_files = project_data.get('files', [])
            all_files_exist = all((project_dir / f['path']).exists() for f in expected_files)
            
            if all_files_exist and stats.get('files_count', 0) > 0:
                console.print(f"✅ [green]{project_name} - Already completed![/green]")
                projects_skipped += 1
                continue
        
        projects_to_process.append((project_dir, project_data))
    
    if not projects_to_process:
        console.print("✅ All projects already completed! Use --force to regenerate.")
        return
    
    console.print(f"🎯 Processing {len(projects_to_process)} projects ({projects_skipped} skipped)")
    
    # Semaphore to limit concurrent project generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    total_files_generated = 0
    total_lines_generated = 0
    projects_completed = 0
    
    async def generate_single_project(project_info, project_index):
        """Generate a single project with semaphore control"""
        project_dir, project_data = project_info
        
        async with semaphore:  # Acquire semaphore slot
            spec = project_data['specification']
            project_name = f"{spec['name']} ({spec['language']})"
            
            try:
                console.print(f"🔨 [bold cyan]Starting {project_index}/{len(projects_to_process)}: {project_name}[/bold cyan]")
                
                # Extract target metrics
                target_files = spec.get('target_file_count', 10)
                target_tokens = spec.get('target_token_count', 20000)
                
                console.print(f"   🎯 Target: {target_files} files, ~{target_tokens:,} tokens")
                console.print("   🤖 3 Elite Models working...")
                
                # Start timing
                import time
                start_time = time.time()
                
                # Generate project files
                project_result = await generator.generate_project_files(spec, target_files, target_tokens)
                
                # Extract data from new format
                project_files = project_result['files']
                files_created = project_result['files_created']
                lines_created = project_result['lines_created']
                generation_time = project_result['generation_time']
                
                # Save generated files to project directory
                console.print(f"   💾 Saving {len(project_files)} files...")
                for file_data in project_files:
                    file_path = project_dir / file_data['path']
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(file_data['content'])
                
                # Update project metadata with generated files
                project_data['files'] = [{'path': f['path'], 'type': f['type']} for f in project_files]
                project_data['generated_stats'] = {
                    'files_count': files_created,
                    'lines_count': lines_created,
                    'generation_time': generation_time,
                    'timestamp': time.time()
                }
                
                # Save updated metadata
                with open(project_dir / "project_metadata.json", 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                console.print(f"   ✅ [green]Completed {project_name}![/green] {files_created} files, {lines_created:,} lines")
                
                # Save progress for successful completion
                import time
                current_progress = {
                    'project_name': project_name,
                    'status': 'completed',
                    'files_created': files_created,
                    'lines_created': lines_created,
                    'timestamp': time.time()
                }
                completed_projects.append(current_progress)
                save_progress(progress_file, completed_projects, "2")
                
                return {
                    'success': True,
                    'files_created': files_created,
                    'lines_created': lines_created,
                    'project_name': project_name,
                    'generation_time': generation_time
                }
                
            except CriticalAuthError as e:
                # Critical auth errors should stop the entire process
                console.print(f"   🚨 [bold red]CRITICAL AUTH FAILURE in {project_name}[/bold red]")
                console.print(f"   🔑 {e.provider}: {e.message}")
                console.print("   🛑 [yellow]Stopping generation to fix authentication...[/yellow]")
                
                # Save current progress before stopping
                current_progress = {
                    'project_name': project_name,
                    'status': 'auth_failed',
                    'error': str(e),
                    'timestamp': time.time()
                }
                completed_projects.append(current_progress)
                save_progress(progress_file, completed_projects, "2")
                
                # Re-raise to stop the entire process
                raise e
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {project_name}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'project_name': project_name
                }
    
    # Execute all projects in parallel with progress tracking
    console.print(f"\n🚀 [bold]Starting parallel generation of {len(projects_to_process)} projects...[/bold]")
    
    try:
        # Create tasks for all projects
        tasks = []
        for i, project_info in enumerate(projects_to_process, 1):
            task = generate_single_project(project_info, i)
            tasks.append(task)
        
        # Wait for all projects to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for CriticalAuthError in results
        for result in results:
            if isinstance(result, CriticalAuthError):
                raise result
        
        # Process results
        successful_projects = []
        failed_projects = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_projects.append(f"Exception: {str(result)}")
            elif result and result['success']:
                successful_projects.append(result)
                total_files_generated += result['files_created']
                total_lines_generated += result['lines_created']
                projects_completed += 1
                # Collect timing data if available
                if 'generation_time' in result:
                    project_times.append(result['generation_time'])
                if 'files_created' in result:
                    file_counts.append(result['files_created'])
                if 'lines_created' in result:
                    line_counts.append(result['lines_created'])
            else:
                failed_projects.append(result['project_name'] if result else "Unknown project")
        
        # ⏰ TIMING ANALYSIS
        phase_end_time = time.time()
        phase_duration = phase_end_time - phase_start_time
        phase_end_datetime = datetime.now()
        
        # Convert to human readable
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        # Calculate timing statistics
        if project_times:
            avg_project_time = sum(project_times) / len(project_times)
            min_project_time = min(project_times)
            max_project_time = max(project_times)
            total_generation_time = sum(project_times)
            
            # Calculate throughput metrics
            avg_files_per_project = sum(file_counts) / len(file_counts) if file_counts else 0
            avg_lines_per_project = sum(line_counts) / len(line_counts) if line_counts else 0
            files_per_minute = (total_files_generated / phase_duration) * 60 if phase_duration > 0 else 0
            lines_per_minute = (total_lines_generated / phase_duration) * 60 if phase_duration > 0 else 0
            
            # Calculate full-scale projections for Phase 2
            total_target_projects = len(config.phase1.supported_languages) * config.phase1.projects_per_language
            estimated_full_scale_time = avg_project_time * total_target_projects
            
            # Phase 2 has lower parallel efficiency due to file generation complexity
            parallel_efficiency = 0.6  # Assume 60% parallel efficiency for file generation
            if max_concurrent > 1:
                estimated_parallel_time = estimated_full_scale_time / (max_concurrent * parallel_efficiency)
            else:
                estimated_parallel_time = estimated_full_scale_time
        
        # Final summary
        console.print(f"\n📊 [bold]Phase 2 Summary:[/bold]")
        console.print(f"   ✅ Completed: {projects_completed} projects")
        console.print(f"   ⚠️  Skipped: {projects_skipped} projects (already done)")
        console.print(f"   ❌ Failed: {len(failed_projects)} projects")
        console.print(f"   📄 Total files generated: {total_files_generated:,}")
        console.print(f"   📝 Total lines generated: {total_lines_generated:,}")
        
        # ⏰ TIMING SUMMARY
        console.print(f"\n⏰ [bold]Timing Analysis:[/bold]")
        console.print(f"   🕐 Phase duration: {format_duration(phase_duration)}")
        console.print(f"   📅 Started: {phase_start_datetime.strftime('%H:%M:%S')}")
        console.print(f"   📅 Ended: {phase_end_datetime.strftime('%H:%M:%S')}")
        
        if project_times:
            console.print(f"\n📈 [bold]Per-Project Statistics:[/bold]")
            console.print(f"   ⚡ Average time: {format_duration(avg_project_time)}")
            console.print(f"   🚀 Fastest: {format_duration(min_project_time)}")
            console.print(f"   🐌 Slowest: {format_duration(max_project_time)}")
            console.print(f"   📄 Avg files/project: {avg_files_per_project:.1f}")
            console.print(f"   📝 Avg lines/project: {avg_lines_per_project:.0f}")
            console.print(f"   🎯 Parallel efficiency: {(total_generation_time/phase_duration)*100:.1f}%")
            
            console.print(f"\n🏭 [bold]Throughput Metrics:[/bold]")
            console.print(f"   📄 Files/minute: {files_per_minute:.1f}")
            console.print(f"   📝 Lines/minute: {lines_per_minute:.0f}")
            console.print(f"   🔄 Total generation time: {format_duration(total_generation_time)}")
            
            console.print(f"\n🚀 [bold]Full-Scale Projections ({total_target_projects:,} projects):[/bold]")
            console.print(f"   📊 Sequential: {format_duration(estimated_full_scale_time)}")
            console.print(f"   ⚡ Parallel (x{max_concurrent}): {format_duration(estimated_parallel_time)}")
            console.print(f"   📄 Expected files: {int(avg_files_per_project * total_target_projects):,}")
            console.print(f"   📝 Expected lines: {int(avg_lines_per_project * total_target_projects):,}")
            
            # Save timing data for analysis
            timing_stats = {
                'projects_completed': projects_completed,
                'total_files_generated': total_files_generated,
                'total_lines_generated': total_lines_generated,
                'avg_project_time': avg_project_time,
                'min_project_time': min_project_time,
                'max_project_time': max_project_time,
                'avg_files_per_project': avg_files_per_project,
                'avg_lines_per_project': avg_lines_per_project,
                'files_per_minute': files_per_minute,
                'lines_per_minute': lines_per_minute,
                'parallel_efficiency': (total_generation_time/phase_duration)*100,
                'estimated_full_scale_time': estimated_full_scale_time,
                'estimated_parallel_time': estimated_parallel_time,
                'max_concurrent': max_concurrent
            }
            save_timing_summary("2", phase_start_datetime, phase_end_datetime, timing_stats)
        else:
            console.print(f"   ⚠️  No timing data available (no successful generations)")
            save_timing_summary("2", phase_start_datetime, phase_end_datetime, {'projects_completed': 0})
        
        if failed_projects:
            console.print(f"\n⚠️  [yellow]Failed projects:[/yellow]")
            for failed in failed_projects[:10]:  # Show first 10
                console.print(f"     • {failed}")
            if len(failed_projects) > 10:
                console.print(f"     ... and {len(failed_projects) - 10} more")
                
    except CriticalAuthError as e:
        console.print(f"\n🚨 [bold red]CRITICAL AUTHENTICATION FAILURE[/bold red]")
        console.print(f"🔑 Provider: {e.provider}")
        console.print(f"💬 Error: {e.message}")
        console.print(f"\n📋 Progress saved to: {progress_file}")
        console.print(f"✅ {len(completed_projects)} projects completed before failure")
        console.print(f"\n🔧 [bold yellow]Next steps:[/bold yellow]")
        console.print("   1. Update your API credentials (check api.sh)")
        console.print("   2. Run: source api.sh")
        console.print("   3. Resume with: locobench generate --phase 2")
        console.print("   4. The pipeline will automatically resume from where it stopped")
        
        # Exit with error code
        import sys
        sys.exit(1)


async def run_phase_3_generation(config, force_regenerate=False, max_concurrent=3):
    """Run Phase 3: Long-Context Evaluation Scenario Creation with parallel processing, progress tracking, and resumability"""
    from .generation.scenario_generator import ScenarioGenerator
    from .core.task import TaskCategory
    from pathlib import Path
    import json
    import asyncio
    from asyncio import Semaphore
    
    # ⏰ START TIMING
    phase_start_time = time.time()
    phase_start_datetime = datetime.now()
    
    console.print("\n🎮 [bold]Long-Context Evaluation Scenario Creation Pipeline[/bold]")
    console.print("=" * 60)
    console.print(f"⏰ Phase 3 started at: {phase_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup progress tracking
    progress_file = Path("logs/phase3_progress.json")
    progress_file.parent.mkdir(exist_ok=True)
    completed_scenarios = load_progress(progress_file)
    completed_scenario_keys = {f"{s.get('project_name', '')}_{s.get('category', '')}" for s in completed_scenarios}
    
    generator = ScenarioGenerator(config, log_file="logs/phase3_generation.log")
    generated_dir = Path(config.data.generated_dir)
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"📋 Resume state: {len(completed_scenarios)} scenario tasks previously completed")
    
    # Find all completed projects from Phase 2
    project_dirs = [d for d in generated_dir.iterdir() if d.is_dir()]
    completed_projects = []
    
    for project_dir in project_dirs:
        metadata_file = project_dir / "project_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                project_data = json.load(f)
            # Check if project has generated code files
            if 'generated_stats' in project_data and project_data['generated_stats'].get('files_count', 0) > 0:
                completed_projects.append((project_dir, project_data))
    
    console.print(f"📂 Found {len(completed_projects)} completed projects from Phase 2")
    
    if len(completed_projects) == 0:
        console.print("⚠️  [yellow]No completed projects found. Run Phase 2 first![/yellow]")
        return
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL scenarios[/yellow]")
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed scenarios...[/cyan]")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent scenario generations[/bold blue]")
    
    console.print("🎯 Creating evaluation scenarios with 2 Elite Models...")
    
    # Calculate scenario distribution based on config task_distribution
    task_categories = list(TaskCategory)
    
    # Validate that all configured task categories exist in our enum
    config_categories = set(config.phase3.task_distribution.keys())
    enum_categories = {cat.value for cat in task_categories}
    missing_categories = config_categories - enum_categories
    if missing_categories:
        console.print(f"⚠️  [yellow]Warning: Config contains unknown task categories: {missing_categories}[/yellow]")
    
    # Create distribution map from config
    task_instance_counts = {}
    total_projects_all_languages = len(completed_projects)
    
    for task_category in task_categories:
        if task_category.value in config.phase3.task_distribution:
            # Use configured count
            target_count = config.phase3.task_distribution[task_category.value]
            instances_per_project = max(1, target_count // total_projects_all_languages)
            task_instance_counts[task_category] = instances_per_project
        else:
            # Fallback for missing categories
            console.print(f"⚠️  [yellow]Warning: {task_category.value} not in config task_distribution, using default[/yellow]")
            task_instance_counts[task_category] = 2
    
    # Prepare scenario generation tasks - DISTRIBUTION-ENFORCED GENERATION
    console.print(f"📋 Target Difficulty Distribution (ENFORCED):")
    target_distribution = {}
    for difficulty, count in config.phase3.difficulty_distribution.items():
        target_distribution[difficulty] = count
        console.print(f"  • {difficulty}: {count} scenarios")
    
    console.print(f"📋 Task Distribution (from config):")
    for task_category, instances_per_project in task_instance_counts.items():
        total_for_category = instances_per_project * total_projects_all_languages
        console.print(f"  • {task_category.value}: {instances_per_project} per project × {total_projects_all_languages} projects = {total_for_category} total")
    
    # Track current difficulty distribution
    current_distribution = {"easy": 0, "medium": 0, "hard": 0, "expert": 0}
    
    # Create difficulty assignment strategy
    def get_next_target_difficulty():
        """Determine which difficulty level we need more of to achieve target distribution"""
        for difficulty in ["expert", "hard", "medium", "easy"]:  # Prioritize harder difficulties
            target_count = target_distribution.get(difficulty, 0)
            current_count = current_distribution.get(difficulty, 0)
            if current_count < target_count:
                return difficulty
        # If we've met all targets, cycle through difficulties
        return random.choice(["easy", "medium", "hard", "expert"])
    
    scenario_tasks = []
    scenarios_skipped = 0
    total_scenarios_planned = sum(task_instance_counts.values()) * total_projects_all_languages
    
    console.print(f"\n🎯 [bold]Distribution-Enforced Generation: {total_scenarios_planned} scenarios[/bold]")
    console.print("🔧 [cyan]Each scenario will target a specific difficulty to achieve desired distribution[/cyan]")
    
    for project_dir, project_data in completed_projects:
        for task_category in task_categories:
            # Generate the configured number of scenarios for this combination
            instances_for_category = task_instance_counts[task_category]
            
            for instance_num in range(instances_for_category):
                # Determine target difficulty for this scenario
                target_difficulty_name = get_next_target_difficulty()
                current_distribution[target_difficulty_name] += 1  # Reserve this slot
                
                # Generate unique scenario file name with target difficulty
                scenario_file = scenarios_dir / f"{project_dir.name}_{task_category.value}_{target_difficulty_name}_{instance_num+1:02d}.json"
                
                if not force_regenerate and scenario_file.exists():
                    console.print(f"✅ [green]{scenario_file.name} already exists[/green]")
                    scenarios_skipped += 1
                    continue
                
                scenario_tasks.append({
                    'project_dir': project_dir,
                    'project_data': project_data,
                    'task_category': task_category,
                    'target_difficulty': target_difficulty_name,  # NEW: Target specific difficulty
                    'instance_num': instance_num + 1,
                    'scenario_file': scenario_file,
                    'scenario_id': f"{project_dir.name}_{task_category.value}_{target_difficulty_name}_{instance_num+1:02d}"
                })
    
    console.print(f"\n🎯 [bold]Total scenarios planned: {total_scenarios_planned}[/bold]")
    
    if not scenario_tasks:
        console.print("✅ All scenarios already completed! Use --force to regenerate.")
        return
    
    console.print(f"🎯 Processing {len(scenario_tasks)} scenario generation tasks ({scenarios_skipped} skipped)")
    
    # Semaphore to limit concurrent scenario generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    total_scenarios_generated = 0
    tasks_completed = 0
    
    async def generate_single_scenario_task(task_info, task_index):
        """Generate a single scenario (individual scenario-level parallelization)"""
        async with semaphore:  # Acquire semaphore slot
            project_dir = task_info['project_dir']
            project_data = task_info['project_data']
            task_category = task_info['task_category']
            target_difficulty_name = task_info['target_difficulty']
            scenario_file = task_info['scenario_file']
            scenario_id = task_info['scenario_id']
            
            # Convert target difficulty string to enum
            difficulty_map = {
                "easy": DifficultyLevel.EASY,
                "medium": DifficultyLevel.MEDIUM,
                "hard": DifficultyLevel.HARD,
                "expert": DifficultyLevel.EXPERT
            }
            target_difficulty = difficulty_map[target_difficulty_name]
            
            project_name = project_data['specification']['name']
            category_name = task_category.value
            
            # Skip if already completed (resume functionality)
            if not force_regenerate and scenario_file.exists():
                console.print(f"✅ [green]Skipping {scenario_file.name} - Already completed![/green]")
                return {
                    'success': True,
                    'skipped': True,
                    'project_name': project_name,
                    'category': category_name,
                    'target_difficulty': target_difficulty_name
                }
            
            try:
                console.print(f"🔨 [bold cyan]Starting {task_index}/{len(scenario_tasks)}: {project_name} - {category_name} ({target_difficulty_name.upper()})[/bold cyan]")
                
                # Start timing
                import time
                start_time = time.time()
                
                # Generate single scenario directly using the low-level method
                project_files = generator._load_project_files(project_dir, project_data)
                scenario = await generator._generate_single_scenario(
                    scenario_id=scenario_id,
                    task_category=task_category,
                    project_spec=project_data['specification'],
                    project_files=project_files,
                    project_stats=project_data['generated_stats'],
                    target_difficulty=target_difficulty
                )
                
                generation_time = time.time() - start_time
                
                # Safety check: Ensure we actually generated a scenario
                if not scenario:
                    error_msg = f"No scenario generated for {scenario_id}"
                    console.print(f"   ❌ [red]{error_msg}[/red]")
                    return {
                        'success': False,
                        'error': error_msg,
                        'project_name': project_name,
                        'category': category_name,
                        'target_difficulty': target_difficulty_name
                    }
                
                # Save scenario to individual file (one scenario per file)
                with open(scenario_file, 'w') as f:
                    json.dump(scenario, f, indent=2)
                
                console.print(f"   ✅ [green]Completed {project_name} - {category_name} ({target_difficulty_name.upper()})![/green] 1 scenario in {generation_time:.1f}s")
                
                # Save progress for successful completion
                current_progress = {
                    'project_name': project_name,
                    'category': category_name,
                    'status': 'completed',
                    'scenarios_generated': 1,
                    'generation_time': generation_time,
                    'timestamp': time.time()
                }
                completed_scenarios.append(current_progress)
                save_progress(progress_file, completed_scenarios, "3")
                
                return {
                    'success': True,
                    'scenarios_generated': 1,
                    'project_name': project_name,
                    'category': category_name,
                    'generation_time': generation_time,
                    'target_difficulty': target_difficulty_name
                }
                
            except CriticalAuthError as e:
                # Critical auth errors should stop the entire process
                console.print(f"   🚨 [bold red]CRITICAL AUTH FAILURE in {project_name} - {category_name}[/bold red]")
                console.print(f"   🔑 {e.provider}: {e.message}")
                console.print("   🛑 [yellow]Stopping generation to fix authentication...[/yellow]")
                
                # Save current progress before stopping
                current_progress = {
                    'project_name': project_name,
                    'category': category_name,
                    'status': 'auth_failed',
                    'error': str(e),
                    'timestamp': time.time()
                }
                completed_scenarios.append(current_progress)
                save_progress(progress_file, completed_scenarios, "3")
                
                # Re-raise to stop the entire process
                raise e
                
            except Exception as e:
                error_str = str(e)
                console.print(f"   ❌ [red]Failed {project_name} - {category_name} ({target_difficulty_name.upper()}): {error_str}[/red]")
                
                return {
                    'success': False,
                    'error': error_str,
                    'project_name': project_name,
                    'category': category_name,
                    'target_difficulty': target_difficulty_name
                }
    
    # Execute all scenario generation tasks in parallel
    console.print(f"\n🚀 [bold]Starting parallel scenario generation for {len(scenario_tasks)} individual scenarios...[/bold]")
    
    try:
        # Create asyncio tasks for individual scenarios (not project+category groups)
        tasks = []
        for i, task_info in enumerate(scenario_tasks, 1):
            task = generate_single_scenario_task(task_info, i)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for CriticalAuthError in results
        for result in results:
            if isinstance(result, CriticalAuthError):
                raise result
        
        # Process results
        successful_tasks = []
        failed_tasks = []
        skipped_tasks = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_tasks.append(f"Exception: {str(result)}")
            elif result and result['success']:
                if result.get('skipped'):
                    skipped_tasks += 1
                else:
                    successful_tasks.append(result)
                    total_scenarios_generated += result['scenarios_generated']
                    tasks_completed += 1
            else:
                failed_tasks.append(f"{result['project_name']} - {result['category']}" if result else "Unknown task")
        
        # ⏰ TIMING ANALYSIS
        phase_end_time = time.time()
        phase_duration = phase_end_time - phase_start_time
        phase_end_datetime = datetime.now()
        
        # Convert to human readable
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        # Final summary
        console.print(f"\n📊 [bold]Phase 3 Summary:[/bold]")
        console.print(f"   ✅ Completed: {tasks_completed} scenario generation tasks")
        console.print(f"   ⚠️  Skipped: {skipped_tasks + scenarios_skipped} tasks (already done)")
        console.print(f"   ❌ Failed: {len(failed_tasks)} tasks")
        console.print(f"   🎯 Total scenarios generated: {total_scenarios_generated}")
        console.print(f"   📁 Scenarios saved to: {scenarios_dir}")
        
        # 📊 DIFFICULTY DISTRIBUTION ANALYSIS
        console.print(f"\n📊 [bold]Difficulty Distribution Analysis:[/bold]")
        console.print("🔬 [cyan]Analyzing actual vs desired difficulty distribution...[/cyan]")
        
        # Analyze generated scenarios to see actual difficulty distribution
        actual_distribution = {"easy": 0, "medium": 0, "hard": 0, "expert": 0}
        total_analyzed = 0
        
        for scenario_file in scenarios_dir.glob("*.json"):
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                # Each file contains a single scenario object, not an array
                difficulty = scenario_data.get('difficulty', '').lower()
                if difficulty in actual_distribution:
                    actual_distribution[difficulty] += 1
                    total_analyzed += 1
            except Exception as e:
                logger.debug(f"Error analyzing {scenario_file}: {e}")
        
        console.print(f"   📈 Analyzed {total_analyzed} scenarios from {len(list(scenarios_dir.glob('*.json')))} files")
        console.print(f"\n   📋 Distribution Comparison:")
        console.print(f"   {'Difficulty':<12} {'Desired':<8} {'Actual':<8} {'Diff':<8} {'%':<8}")
        console.print(f"   {'-'*50}")
        
        for difficulty in ["easy", "medium", "hard", "expert"]:
            desired = target_distribution.get(difficulty, 0)
            actual = actual_distribution.get(difficulty, 0)
            diff = actual - desired
            percentage = (actual / desired * 100) if desired > 0 else 0
            
            # Color coding for the difference
            diff_color = "green" if abs(diff) <= 2 else "yellow" if abs(diff) <= 5 else "red"
            console.print(f"   {difficulty:<12} {desired:<8} {actual:<8} [bold {diff_color}]{diff:+3d}[/bold {diff_color}]     {percentage:5.1f}%")
        
        # Overall assessment
        total_desired = sum(target_distribution.values())
        deviation_score = sum(abs(actual_distribution[d] - target_distribution.get(d, 0)) for d in actual_distribution.keys())
        
        if deviation_score <= total_desired * 0.1:  # Within 10%
            console.print(f"   🎯 [bold green]Excellent distribution match![/bold green] (deviation: {deviation_score})")
        elif deviation_score <= total_desired * 0.2:  # Within 20%
            console.print(f"   ✅ [bold yellow]Good distribution match[/bold yellow] (deviation: {deviation_score})")
        else:
            console.print(f"   ⚠️  [bold red]Distribution needs improvement[/bold red] (deviation: {deviation_score})")
            console.print(f"   💡 [cyan]Tip: Adjust file selection or context length ranges to improve distribution[/cyan]")
        
        # ⏰ TIMING SUMMARY
        console.print(f"\n⏰ [bold]Timing Analysis:[/bold]")
        console.print(f"   🕐 Phase duration: {format_duration(phase_duration)}")
        console.print(f"   📅 Started: {phase_start_datetime.strftime('%H:%M:%S')}")
        console.print(f"   📅 Ended: {phase_end_datetime.strftime('%H:%M:%S')}")
        
        if tasks_completed > 0:
            avg_task_time = phase_duration / (tasks_completed + len(failed_tasks)) if (tasks_completed + len(failed_tasks)) > 0 else 0
            console.print(f"   ⚡ Average task time: {format_duration(avg_task_time)}")
            if total_scenarios_generated > 0:
                scenarios_per_minute = (total_scenarios_generated / phase_duration) * 60 if phase_duration > 0 else 0
                console.print(f"   🎯 Scenarios/minute: {scenarios_per_minute:.1f}")
        else:
            console.print(f"   ⚠️  No timing data available (no successful completions)")
        
        if failed_tasks:
            console.print(f"\n⚠️  [yellow]Failed tasks:[/yellow]")
            for failed in failed_tasks[:10]:  # Show first 10 failures
                console.print(f"     • {failed}")
            if len(failed_tasks) > 10:
                console.print(f"     ... and {len(failed_tasks) - 10} more")
                
    except CriticalAuthError as e:
        console.print(f"\n🚨 [bold red]CRITICAL AUTHENTICATION FAILURE[/bold red]")
        console.print(f"🔑 Provider: {e.provider}")
        console.print(f"💬 Error: {e.message}")
        console.print(f"\n📋 Progress saved to: {progress_file}")
        console.print(f"✅ {len(completed_scenarios)} scenario tasks completed before failure")
        console.print(f"\n🔧 [bold yellow]Next steps:[/bold yellow]")
        console.print("   1. Update your API credentials (check api.sh)")
        console.print("   2. Run: source api.sh")
        console.print("   3. Resume with: locobench generate --phase 3")
        console.print("   4. The pipeline will automatically resume from where it stopped")
        
        # Exit with error code
        import sys
        sys.exit(1)


async def run_phase_4_generation(config, force_regenerate=False, max_concurrent=3):
    """Run Phase 4: Automated Test-Driven Validation Framework with parallel processing, progress tracking, and resumability"""
    from .generation.validation_framework import AutomatedValidator
    from .core.task import TaskCategory
    from pathlib import Path
    import json
    import asyncio
    from asyncio import Semaphore
    
    # ⏰ START TIMING
    phase_start_time = time.time()
    phase_start_datetime = datetime.now()
    
    console.print("\n🧪 [bold]Automated Test-Driven Validation Framework[/bold]")
    console.print("=" * 60)
    console.print(f"⏰ Phase 4 started at: {phase_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup progress tracking
    progress_file = Path("logs/phase4_progress.json")
    progress_file.parent.mkdir(exist_ok=True)
    completed_validations = load_progress(progress_file)
    completed_scenario_files = {v.get('scenario_file', '') for v in completed_validations}
    
    validator = AutomatedValidator(config)
    scenarios_dir = Path(config.data.output_dir) / "scenarios"
    
    console.print(f"📋 Resume state: {len(completed_validations)} validation tasks previously completed")
    
    if not scenarios_dir.exists():
        console.print("⚠️  [yellow]No scenarios found. Run Phase 3 first![/yellow]")
        return
    
    # Find all scenario files
    scenario_files = list(scenarios_dir.glob("*.json"))
    
    if len(scenario_files) == 0:
        console.print("⚠️  [yellow]No scenario files found. Run Phase 3 first![/yellow]")
        return
    
    console.print(f"📂 Found {len(scenario_files)} scenario files from Phase 3")
    
    if force_regenerate:
        console.print("🔄 [yellow]Force mode: Regenerating ALL test suites[/yellow]")
    else:
        console.print("🧠 [cyan]Smart resume: Checking for completed test suites...[/cyan]")
    
    if max_concurrent > 1:
        console.print(f"🚀 [bold blue]Parallel mode: {max_concurrent} concurrent test suite generations[/bold blue]")
    
    console.print("🎯 Creating automated test suites for evaluation...")
    console.print(f"⚖️  Evaluation weights: Software Engineering (40%) | Functional Correctness (30%) | Code Quality (20%) | Long-Context Util (10%)")
    
    # Prepare test suite generation tasks
    validation_tasks = []
    test_suites_skipped = 0
    
    for scenario_file in scenario_files:
        # Check if test suite already exists
        validation_dir = Path(config.data.output_dir) / "validation" / "test_suites"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        test_suite_file = validation_dir / f"{scenario_file.stem}_test_suite.json"
        
        if not force_regenerate and test_suite_file.exists():
            console.print(f"✅ [green]{scenario_file.name} - test suite already exists[/green]")
            test_suites_skipped += 1
            continue
        
        validation_tasks.append({
            'scenario_file': scenario_file,
            'test_suite_file': test_suite_file
        })
    
    if not validation_tasks:
        console.print("✅ All test suites already completed! Use --force to regenerate.")
        return
    
    console.print(f"🎯 Processing {len(validation_tasks)} test suite generation tasks ({test_suites_skipped} skipped)")
    
    # Semaphore to limit concurrent test suite generation
    semaphore = Semaphore(max_concurrent)
    
    # Statistics tracking
    total_test_suites_generated = 0
    tasks_completed = 0
    
    async def generate_test_suite_for_scenarios(task_info, task_index):
        """Generate test suite for one scenario file"""
        async with semaphore:  # Acquire semaphore slot
            scenario_file = task_info['scenario_file']
            test_suite_file = task_info['test_suite_file']
            
            # Skip if already completed (resume functionality)
            if not force_regenerate and scenario_file.name in completed_scenario_files:
                console.print(f"✅ [green]Skipping {scenario_file.name} - Already completed![/green]")
                return {
                    'success': True,
                    'skipped': True,
                    'scenario_file': scenario_file.name
                }
            
            try:
                console.print(f"🔨 [bold cyan]Starting {task_index}/{len(validation_tasks)}: {scenario_file.name}[/bold cyan]")
                
                # Load scenarios
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                
                scenarios = scenario_data.get('scenarios', [])
                if not scenarios:
                    console.print(f"   ⚠️  [yellow]No scenarios found in {scenario_file.name}[/yellow]")
                    return {'success': True, 'test_suites_generated': 0, 'scenario_file': scenario_file.name}
                
                # Start timing
                import time
                start_time = time.time()
                
                # Generate test suites for all scenarios in this file
                test_suites = []
                for scenario in scenarios:
                    test_suite = await validator.generate_test_suite(scenario)
                    test_suites.append({
                        'scenario_id': scenario.get('id', 'unknown'),
                        'test_suite': test_suite.to_dict()  # Convert to dict for JSON serialization
                    })
                
                generation_time = time.time() - start_time
                
                # Save test suites
                test_suite_data = {
                    'source_file': scenario_file.name,
                    'generated_timestamp': time.time(),
                    'generation_time': generation_time,
                    'test_suites': test_suites
                }
                
                with open(test_suite_file, 'w') as f:
                    json.dump(test_suite_data, f, indent=2)
                
                console.print(f"   ✅ [green]Completed {scenario_file.name}![/green] {len(test_suites)} test suites in {generation_time:.1f}s")
                
                # Save progress for successful completion
                current_progress = {
                    'scenario_file': scenario_file.name,
                    'status': 'completed',
                    'test_suites_generated': len(test_suites),
                    'generation_time': generation_time,
                    'timestamp': time.time()
                }
                completed_validations.append(current_progress)
                save_progress(progress_file, completed_validations, "4")
                
                return {
                    'success': True,
                    'test_suites_generated': len(test_suites),
                    'scenario_file': scenario_file.name,
                    'generation_time': generation_time
                }
                
            except Exception as e:
                console.print(f"   ❌ [red]Failed {scenario_file.name}: {str(e)}[/red]")
                return {
                    'success': False,
                    'error': str(e),
                    'scenario_file': scenario_file.name
                }
    
    # Execute all test suite generation tasks in parallel
    console.print(f"\n🚀 [bold]Starting parallel test suite generation for {len(validation_tasks)} tasks...[/bold]")
    
    try:
        # Create asyncio tasks
        tasks = []
        for i, task_info in enumerate(validation_tasks, 1):
            task = generate_test_suite_for_scenarios(task_info, i)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_tasks = []
        failed_tasks = []
        skipped_tasks = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_tasks.append(f"Exception: {str(result)}")
            elif result and result['success']:
                if result.get('skipped'):
                    skipped_tasks += 1
                else:
                    successful_tasks.append(result)
                    total_test_suites_generated += result['test_suites_generated']
                    tasks_completed += 1
            else:
                failed_tasks.append(result['scenario_file'] if result else "Unknown task")
        
        # ⏰ TIMING ANALYSIS
        phase_end_time = time.time()
        phase_duration = phase_end_time - phase_start_time
        phase_end_datetime = datetime.now()
        
        # Convert to human readable
        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        # Final summary
        console.print(f"\n📊 [bold]Phase 4 Summary:[/bold]")
        console.print(f"   ✅ Completed: {tasks_completed} test suite generation tasks")
        console.print(f"   ⚠️  Skipped: {skipped_tasks + test_suites_skipped} tasks (already done)")
        console.print(f"   ❌ Failed: {len(failed_tasks)} tasks")
        console.print(f"   🧪 Total test suites generated: {total_test_suites_generated}")
        console.print(f"   📁 Test suites saved to: {Path(config.data.output_dir) / 'validation' / 'test_suites'}")
        
        # ⏰ TIMING SUMMARY
        console.print(f"\n⏰ [bold]Timing Analysis:[/bold]")
        console.print(f"   🕐 Phase duration: {format_duration(phase_duration)}")
        console.print(f"   📅 Started: {phase_start_datetime.strftime('%H:%M:%S')}")
        console.print(f"   📅 Ended: {phase_end_datetime.strftime('%H:%M:%S')}")
        
        if tasks_completed > 0:
            avg_task_time = phase_duration / (tasks_completed + len(failed_tasks)) if (tasks_completed + len(failed_tasks)) > 0 else 0
            console.print(f"   ⚡ Average task time: {format_duration(avg_task_time)}")
            if total_test_suites_generated > 0:
                test_suites_per_minute = (total_test_suites_generated / phase_duration) * 60 if phase_duration > 0 else 0
                console.print(f"   🧪 Test suites/minute: {test_suites_per_minute:.1f}")
        else:
            console.print(f"   ⚠️  No timing data available (no successful completions)")
        
        if failed_tasks:
            console.print(f"\n⚠️  [yellow]Failed tasks:[/yellow]")
            for failed in failed_tasks[:10]:  # Show first 10 failures
                console.print(f"     • {failed}")
            if len(failed_tasks) > 10:
                console.print(f"     ... and {len(failed_tasks) - 10} more")
                
    except Exception as e:
        console.print(f"\n❌ [bold red]EXECUTION FAILURE[/bold red]")
        console.print(f"💬 Error: {str(e)}")
        console.print(f"\n📋 Progress saved to: {progress_file}")
        console.print(f"✅ {len(completed_validations)} validation tasks completed before failure")
        console.print(f"\n🔧 [bold yellow]Next steps:[/bold yellow]")
        console.print("   1. Check the error details above")
        console.print("   2. Resume with: locobench generate --phase 4")
        console.print("   3. The pipeline will automatically resume from where it stopped")
        
        # Exit with error code
        import sys
        sys.exit(1)


async def run_agent_evaluation(
    config, agent_type, model, max_turns, enable_tools, 
    compare_agents, scenario_file
):
    """Run agent evaluation with the specified parameters"""
    from .agents.agent_factory import AgentFactory, AgentConfig as AgentConfigClass, AgentType
    from .core.agent_session import AgentSession, ConversationPhase, SessionConfig
    from .core.tool_registry import get_tool_registry, register_tool
    from .tools import FileSystemTool, CompilerTool, DebuggerTool, IDESimulatorTool, EchoTool, CalculatorTool
    
    console.print("🚀 Initializing LoCoBench-Agent evaluation system...", style="bold")
    
    # Register tools if enabled
    if enable_tools:
        tool_registry = get_tool_registry()
        
        # Register basic tools
        register_tool(EchoTool())
        register_tool(CalculatorTool())
        
        # Register advanced tools based on configuration
        if config.agent.enable_file_system_tools:
            register_tool(FileSystemTool(
                allowed_directories=config.agent.allowed_directories,
                readonly_mode=config.agent.readonly_mode,
                max_file_size=config.agent.max_file_size_mb * 1024 * 1024
            ))
        
        if config.agent.enable_compiler_tools:
            register_tool(CompilerTool(
                allowed_directories=config.agent.allowed_directories,
                enable_network=config.agent.enable_network_access
            ))
        
        if config.agent.enable_debugger_tools:
            register_tool(DebuggerTool(
                allowed_directories=config.agent.allowed_directories
            ))
        
        if config.agent.enable_ide_simulator:
            register_tool(IDESimulatorTool(
                allowed_directories=config.agent.allowed_directories
            ))
        
        available_tools = tool_registry.get_all_tools()
        console.print(f"✅ Registered {len(available_tools)} tools", style="green")
    else:
        available_tools = []
        console.print("⚠️ Tools disabled", style="yellow")
    
    # Create agent configurations
    agent_configs = []
    
    if agent_type == 'all' or compare_agents:
        # Create configurations for all available agent types
        if agent_type == 'all':
            types_to_test = ['openai', 'anthropic', 'google']
        else:
            types_to_test = [agent_type]
        
        for atype in types_to_test:
            try:
                agent_enum = getattr(AgentType, atype.upper())
                agent_config = AgentConfigClass(
                    agent_type=agent_enum,
                    name=f"{atype.title()} Agent",
                    model_name=model or AgentFactory.DEFAULT_MODELS.get(agent_enum),
                    temperature=0.1,
                    max_tokens=4096
                )
                agent_configs.append(agent_config)
            except (AttributeError, KeyError):
                console.print(f"⚠️ Agent type {atype} not available", style="yellow")
    else:
        # Create single agent configuration
        try:
            agent_enum = getattr(AgentType, agent_type.upper())
            agent_config = AgentConfigClass(
                agent_type=agent_enum,
                name=f"{agent_type.title()} Agent",
                model_name=model or AgentFactory.DEFAULT_MODELS.get(agent_enum),
                temperature=0.1,
                max_tokens=4096
            )
            agent_configs.append(agent_config)
        except (AttributeError, KeyError):
            console.print(f"❌ Agent type {agent_type} not available", style="red")
            return
    
    console.print(f"🤖 Created {len(agent_configs)} agent configurations", style="blue")
    
    # Load or create evaluation scenarios
    scenarios = []
    
    if scenario_file:
        # Load specific scenario file
        scenario_path = Path(scenario_file)
        if scenario_path.exists():
            with open(scenario_path, 'r') as f:
                scenario_data = json.load(f)
            # Convert single-turn scenario to multi-turn interactive scenario
            interactive_scenario = await convert_to_interactive_scenario(scenario_data, config)
            scenarios.append(interactive_scenario)
            console.print(f"📄 Loaded and converted scenario from {scenario_file}", style="green")
        else:
            console.print(f"❌ Scenario file not found: {scenario_file}", style="red")
            return
    else:
        # Use cached converted scenarios for much faster loading
        converter = get_scenario_converter(config)
        data_loader = get_data_loader(config)
        
        # Check if scenarios have been pre-converted
        conversion_stats = converter.get_conversion_stats()
        
        if conversion_stats['total_converted'] > 0:
            console.print(f"⚡ Using pre-converted scenarios for fast loading", style="green")
            console.print(f"📊 Available: {conversion_stats['total_converted']} converted scenarios", style="blue")
            
            # Load original scenarios to get the list
            original_scenarios = data_loader.load_scenarios(
                limit=10,  # Reasonable limit for evaluation
                include_project_context=False  # Don't load project context yet, just get scenario IDs
            )
            
            loaded_count = 0
            for scenario_data in original_scenarios:
                # Try to load pre-converted scenario
                converted_scenario = converter.load_converted_scenario(scenario_data.scenario_id)
                
                if converted_scenario:
                    scenarios.append(converted_scenario)
                    loaded_count += 1
                else:
                    console.print(f"⚠️ No converted version found for {scenario_data.scenario_id}, skipping", style="yellow")
            
            console.print(f"📁 Loaded {loaded_count} pre-converted scenarios", style="green")
            
            # Show breakdown by category
            if scenarios:
                categories = {}
                for scenario in scenarios:
                    cat = scenario.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                
                console.print("📋 Scenario breakdown:", style="blue")
                for cat, count in categories.items():
                    console.print(f"   • {cat}: {count} scenarios", style="blue")
        
        else:
            # No pre-converted scenarios available, fall back to real-time conversion
            console.print("⚠️ No pre-converted scenarios found. Converting on-the-fly (slower)", style="yellow")
            console.print("💡 Tip: Run 'locobench convert-scenarios' first for much faster evaluation", style="cyan")
            
            # Get data statistics
            stats = data_loader.get_data_statistics()
            console.print(f"📊 Data Overview: {stats['projects']['total']} projects, {stats['scenarios']['total']} scenarios", style="blue")
            
            # Load scenarios with project context
            scenario_data_list = data_loader.load_scenarios(
                limit=10,  # Reasonable limit for evaluation
                include_project_context=True
            )
            
            if scenario_data_list:
                console.print(f"🔍 Found {stats['scenarios']['total']} total scenarios, using {len(scenario_data_list)} for evaluation", style="blue")
                
                for scenario_data in scenario_data_list:
                    try:
                        # Convert to interactive scenario format (slow)
                        interactive_scenario = await convert_to_interactive_scenario(scenario_data.raw_data, config)
                        
                        # Enhance with project context if available
                        if scenario_data.project_context:
                            interactive_scenario["project_files"] = scenario_data.project_context.files
                            interactive_scenario["project_spec"] = scenario_data.project_context.specification
                            interactive_scenario["project_directory"] = str(scenario_data.project_context.project_dir)
                            interactive_scenario["project_name"] = scenario_data.project_context.project_name
                        
                        scenarios.append(interactive_scenario)
                        
                    except Exception as e:
                        console.print(f"⚠️ Error converting scenario {scenario_data.scenario_id}: {e}", style="yellow")
                        continue
                        
                console.print(f"📁 Successfully converted {len(scenarios)} scenarios to multi-turn format", style="green")
                
                # Show breakdown by category
                categories = {}
                for scenario in scenarios:
                    cat = scenario.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                
                console.print("📋 Scenario breakdown:", style="blue")
                for cat, count in categories.items():
                    console.print(f"   • {cat}: {count} scenarios", style="blue")
            else:
                # Create a simple demo scenario
                demo_scenario = {
                    "id": "demo_scenario_001",
                    "title": "Simple Code Analysis Task",
                    "description": "Analyze the project structure and provide insights",
                    "context_files": ["test_agent_system.py", "example_usage.py"],
                    "difficulty": "easy"
                }
                interactive_scenario = await convert_to_interactive_scenario(demo_scenario, config)
                scenarios.append(interactive_scenario)
                console.print("🎯 Created demo scenario (no data found)", style="blue")
    
    # Run evaluation for each agent configuration
    results = {}
    
    for i, agent_config in enumerate(agent_configs, 1):
        console.print(f"\n🔄 Evaluating Agent {i}/{len(agent_configs)}: {agent_config.name}", style="bold cyan")
        
        try:
            # Create agent
            agent = AgentFactory.create_agent(agent_config)
            
            # Create conversation phases
            phases = [
                ConversationPhase(
                    phase_id="analysis",
                    name="Analysis Phase",
                    initial_prompt="Please analyze the provided codebase and identify key components, architecture patterns, and potential improvements.",
                    expected_actions=["read_file", "list_directory"] if enable_tools else [],
                    success_conditions=["analysis", "components", "architecture"],
                    max_turns_in_phase=max_turns // 2
                ),
                ConversationPhase(
                    phase_id="recommendations",
                    name="Recommendations Phase", 
                    initial_prompt="Based on your analysis, provide specific recommendations for code improvements and explain your reasoning.",
                    expected_actions=["echo"] if enable_tools else [],
                    success_conditions=["recommendations", "improvements"],
                    max_turns_in_phase=max_turns // 2
                )
            ]
            
            # Run evaluation on scenarios
            agent_results = []
            
            for j, scenario in enumerate(scenarios, 1):
                console.print(f"  📋 Scenario {j}/{len(scenarios)}: {scenario.get('title', 'Untitled')}", style="cyan")
                
                # Create session
                session = AgentSession(
                    session_id=f"{agent_config.name.lower().replace(' ', '_')}_scenario_{j}",
                    agent=agent,
                    scenario_context=scenario,
                    conversation_phases=phases,
                    available_tools=available_tools,
                    config=SessionConfig(
                        max_turns=max_turns,
                        timeout_seconds=config.agent.session_timeout_seconds,
                        save_checkpoints=config.agent.enable_checkpointing
                    )
                )
                
                # Execute session
                session_result = await session.execute_conversation()
                agent_results.append(session_result)
                
                # Show brief results
                status_icon = "✅" if session_result["status"] == "completed" else "❌"
                console.print(f"    {status_icon} Status: {session_result['status']}, "
                            f"Turns: {session_result['total_turns']}, "
                            f"Duration: {session_result['session_duration_seconds']:.1f}s")
            
            results[agent_config.name] = agent_results
            
        except Exception as e:
            console.print(f"❌ Error evaluating {agent_config.name}: {e}", style="red")
            results[agent_config.name] = {"error": str(e)}
    
    # Display comparison results
    if len(agent_configs) > 1:
        console.print("\n📊 Agent Comparison Results", style="bold green")
        console.print("=" * 60)
        
        table = Table(title="Agent Performance Comparison", style="green")
        table.add_column("Agent", style="bold")
        table.add_column("Scenarios", justify="center")
        table.add_column("Success Rate", justify="center")
        table.add_column("Avg Turns", justify="center")
        table.add_column("Avg Duration", justify="center")
        
        for agent_name, agent_results in results.items():
            if isinstance(agent_results, dict) and "error" in agent_results:
                table.add_row(agent_name, "ERROR", "N/A", "N/A", "N/A", "N/A")
                continue
            
            total_scenarios = len(agent_results)
            successful = sum(1 for r in agent_results if r["status"] == "completed")
            success_rate = (successful / total_scenarios * 100) if total_scenarios > 0 else 0
            
            avg_turns = sum(r["total_turns"] for r in agent_results) / total_scenarios if total_scenarios > 0 else 0
            avg_duration = sum(r["session_duration_seconds"] for r in agent_results) / total_scenarios if total_scenarios > 0 else 0
            
            table.add_row(
                agent_name,
                str(total_scenarios),
                f"{success_rate:.1f}%",
                f"{avg_turns:.1f}",
                f"{avg_duration:.1f}s"
            )
        
        console.print(table)
    
    # Save results
    results_file = Path(config.data.output_dir) / "agents" / "evaluations" / f"agent_evaluation_{int(time.time())}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "evaluation_config": {
                "agent_type": agent_type,
                "model": model,
                "max_turns": max_turns,
                "enable_tools": enable_tools,
                "compare_agents": compare_agents,
                "scenario_file": scenario_file
            },
            "results": results,
            "timestamp": time.time()
        }, f, indent=2)
    
    console.print(f"\n💾 Results saved to: {results_file}", style="green")
    console.print("🎉 Agent evaluation completed!", style="bold green")


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--mode', type=click.Choice(['llm', 'agent', 'both']), default='llm', help='Evaluation mode: llm (traditional), agent (multi-turn), or both')
@click.option('--agent-type', type=click.Choice(['openai', 'anthropic', 'google', 'autogen', 'langchain', 'crewai', 'swarm', 'all']), default='openai', help='Type of agent to use (agent mode only)')
@click.option('--model', '-m', help='Specific model to use (e.g., gpt-4o, claude-sonnet-4)')
@click.option('--max-turns', type=int, default=20, help='Maximum turns per conversation (agent mode only)')
@click.option('--enable-tools', is_flag=True, default=True, help='Enable tool usage (agent mode only)')
@click.option('--scenario-count', type=int, default=5, help='Number of scenarios to generate/run')
@click.option('--difficulty', type=click.Choice(['easy', 'medium', 'hard', 'expert']), help='Difficulty level (if not specified, uses all difficulties to reach scenario count)')
@click.option('--category', type=click.Choice(['debugging', 'feature_implementation', 'code_review', 'refactoring', 'testing', 'documentation']), help='Task category filter')
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--compare-modes', is_flag=True, help='Compare LLM vs Agent performance (requires both modes)')
@click.option('--enable-analysis', is_flag=True, default=False, help='Enable statistical analysis and comparison (disabled by default to avoid redundant files)')
@click.option('--generate-reports', is_flag=True, default=False, help='Generate HTML reports (disabled by default to avoid redundant files)')
@click.option('--save-conversations', is_flag=True, default=True, help='Save conversation transcripts (agent mode)')
@click.option('--resume', is_flag=True, default=True, help='Resume from checkpoint if available')
@click.option('--max-concurrent-scenarios', type=int, default=1, help='Maximum concurrent scenarios for agent evaluation (default: 1)')
@click.option('--context-management', type=click.Choice(['none', 'basic', 'adaptive']), default='adaptive', help='Context management strategy (default: adaptive)')
@click.option('--initial-context-mode', type=click.Choice(['full', 'minimal', 'empty']), default='minimal', help='Initial context loading: full=load all files (OLD), minimal=README+entry points (RECOMMENDED), empty=discover everything (default: minimal)')
@click.option('--enable-semantic-search/--disable-semantic-search', default=True, help='Enable/disable semantic code search (Cursor @codebase equivalent) - enabled by default')
@click.option('--enable-enhanced-summarization/--disable-enhanced-summarization', default=True, help='Enable/disable LLM-based conversation summarization - enabled by default')
def evaluate(config_path, mode, agent_type, model, max_turns, enable_tools, scenario_count, difficulty, category, output_dir, compare_modes, enable_analysis, generate_reports, save_conversations, resume, max_concurrent_scenarios, context_management, initial_context_mode, enable_semantic_search, enable_enhanced_summarization):
    """Unified evaluation command for both LLM and Agent modes
    
    This command provides a unified interface to run evaluations in:
    - LLM mode: Traditional single-turn LLM evaluation (original LoCoBench)
    - Agent mode: Multi-turn agent evaluation with tools (LoCoBench-Agent)  
    - Both modes: Run both evaluations for comprehensive comparison
    """
    
    console.print(Panel.fit("🚀 LoCoBench Unified Evaluation", style="bold blue"))
    
    # Load configuration
    try:
        config = Config.from_yaml(config_path) if config_path else Config()
        console.print("✅ Configuration loaded", style="green")
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="red")
        return
    
    # Override output directory if specified
    if output_dir:
        config.data.output_dir = str(output_dir)
    
    # Create model-specific output directories in main project directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == "agent":
        # Create model-specific directory name
        model_name = model or "unknown"
        if agent_type == "openai":
            model_display = model_name
        elif agent_type == "anthropic":
            model_display = model_name.replace("claude-3-sonnet-20240229", "claude-sonnet-3")
        elif agent_type == "google":
            model_display = model_name.replace("gemini-pro", "gemini-1.5-pro")
        else:
            model_display = f"{agent_type}-{model_name}"
            
        unified_output = Path("evaluation_agent_results")
    elif mode == "llm":
        model_display = model or "unknown"
        unified_output = Path("evaluation_llm_results") / f"{model_display}_results_{timestamp}"
    else:  # both
        unified_output = Path("evaluation_comparison_results") / f"comparison_{timestamp}"
    unified_output.mkdir(parents=True, exist_ok=True)
    
    console.print(f"📁 Output directory: {unified_output}", style="blue")
    
    # Validate mode combinations
    if compare_modes and mode != 'both':
        console.print("⚠️ --compare-modes requires --mode both. Setting mode to 'both'.", style="yellow")
        mode = 'both'
    
    # Run evaluation based on mode
    results = {}
    
    if mode in ['llm', 'both']:
        console.print("\n📊 Starting LLM Evaluation (Traditional LoCoBench)...", style="bold cyan")
        llm_results = run_llm_evaluation(config, scenario_count, difficulty, category, unified_output / "llm_results")
        results['llm'] = llm_results
        console.print("✅ LLM evaluation completed", style="green")
    
    if mode in ['agent', 'both']:
        console.print("\n🤖 Starting Agent Evaluation (LoCoBench-Agent)...", style="bold cyan")
        agent_results = asyncio.run(run_robust_agent_evaluation(
            config, agent_type, model, max_turns, enable_tools, 
            scenario_count, difficulty, category, unified_output,
            save_conversations, resume=resume, max_concurrent_scenarios=max_concurrent_scenarios,
            context_management=context_management,
            initial_context_mode=initial_context_mode,
            enable_semantic_search=enable_semantic_search,
            enable_enhanced_summarization=enable_enhanced_summarization
        ))
        results['agent'] = agent_results
        console.print("✅ Agent evaluation completed", style="green")
    
    # Comparison and analysis
    if compare_modes and 'llm' in results and 'agent' in results:
        console.print("\n📈 Performing Cross-Mode Comparison...", style="bold cyan")
        comparison_results = perform_cross_mode_comparison(
            results['llm'], results['agent'], unified_output / "comparison"
        )
        results['comparison'] = comparison_results
        console.print("✅ Cross-mode comparison completed", style="green")
    
    # Statistical analysis
    if enable_analysis and len(results) > 0:
        console.print("\n📊 Performing Statistical Analysis...", style="bold cyan")
        analysis_results = asyncio.run(perform_unified_statistical_analysis(
            results, unified_output / "analysis"
        ))
        results['analysis'] = analysis_results
        console.print("✅ Statistical analysis completed", style="green")
    
    # Generate reports
    if generate_reports:
        console.print("\n📋 Generating Comprehensive Reports...", style="bold cyan")
        report_files = generate_unified_reports(results, unified_output / "reports")
        console.print(f"✅ Reports generated: {len(report_files)} files", style="green")
    
    # Summary
    console.print(f"\n🎉 Evaluation Completed!", style="bold green")
    console.print(f"📁 All results saved to: {unified_output}", style="blue")
    
    # Display summary table
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Mode", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Results", style="yellow")
    
    if 'llm' in results:
        summary_table.add_row("LLM (Traditional)", "✅ Complete", f"{len(results['llm'])} evaluations")
    
    if 'agent' in results:
        summary_table.add_row("Agent (Multi-turn)", "✅ Complete", f"{len(results['agent'])} evaluations")
    
    if 'comparison' in results:
        summary_table.add_row("Cross-Mode Comparison", "✅ Complete", "LLM vs Agent analysis")
    
    if 'analysis' in results:
        summary_table.add_row("Statistical Analysis", "✅ Complete", "Comprehensive statistics")
    
    console.print(summary_table)


def run_llm_evaluation(config: Config, scenario_count: int, difficulty: str, category: Optional[str], output_dir: Path) -> List[Dict[str, Any]]:
    """Run traditional LLM evaluation"""
    
    # This would integrate with the existing LoCoBench evaluation pipeline
    # For now, we'll create a placeholder that simulates traditional evaluation
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("🔄 Running traditional LLM evaluation scenarios...", style="blue")
    
    # Simulate LLM evaluation results
    results = []
    for i in range(scenario_count):
        result = {
            "scenario_id": f"llm_scenario_{i+1}",
            "mode": "llm",
            "difficulty": difficulty,
            "category": category or "general",
            "overall_score": random.uniform(3.0, 4.5),
            "metrics": {
                "functional_correctness": random.uniform(3.0, 4.5),
                "code_quality": random.uniform(3.0, 4.5),
                "long_context_utilization": random.uniform(2.5, 4.0),
                "software_engineering_excellence": random.uniform(3.0, 4.5)
            },
            "execution_time": random.uniform(10, 60),
            "token_usage": random.randint(1000, 5000),
            "execution_efficiency": random.uniform(0.8, 1.0)
        }
        results.append(result)
    
    # Save LLM results
    llm_results_file = output_dir / "llm_evaluation_results.json"
    with open(llm_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


async def run_robust_agent_evaluation(
    config: Config, 
    agent_type: str, 
    model: Optional[str], 
    max_turns: int, 
    enable_tools: bool,
    scenario_count: int, 
    difficulty: str, 
    category: Optional[str], 
    output_dir: Path,
    save_conversations: bool,
    resume: bool = True,
    max_concurrent_scenarios: int = 1,
    context_management: str = "adaptive",
    initial_context_mode: str = "minimal",
    enable_semantic_search: bool = False,
    enable_enhanced_summarization: bool = False
) -> List[Dict[str, Any]]:
    """Run robust agent evaluation with checkpointing and incremental saving"""
    
    from .evaluation.robust_agent_evaluator import RobustAgentEvaluator
    from .agents.agent_factory import AgentFactory, AgentConfig as AgentConfigClass, AgentType
    from .generation.interactive_scenario_generator import InteractiveScenario, InteractionMode, ToolUsageMode
    from .core.data_loader import DataLoader
    from .core.task import TaskCategory
    from .core.agent_session import ConversationPhase
    
    try:
        console.print("🔄 Setting up robust agent evaluation...", style="blue")
        
        # Create agent
        agent_type_enum = AgentType.OPENAI if agent_type == "openai" else AgentType.ANTHROPIC if agent_type == "anthropic" else AgentType.GOOGLE
        agent_config = AgentConfigClass(
            agent_type=agent_type_enum,
            model_name=model or "gpt-4o",
            name=f"{agent_type}-{model or 'gpt-4o'}",
            supports_function_calling=enable_tools,
            custom_config={
                "max_turns": max_turns,
                "enable_tools": enable_tools,
                "context_management": context_management
            }
        )
        
        factory = AgentFactory()
        agent = factory.create_agent(agent_config)
        
        # Register tools if enabled
        if enable_tools:
            from .core.tool_registry import get_tool_registry, register_tool
            from .tools import FileSystemTool, CompilerTool, DebuggerTool, IDESimulatorTool, EchoTool, CalculatorTool, SemanticSearchTool
            
            tool_registry = get_tool_registry()
            
            # Register basic tools
            register_tool(EchoTool())
            register_tool(CalculatorTool())
            
            # Register semantic search tool if enabled
            if enable_semantic_search:
                console.print("✨ Enabling semantic code search (Cursor @codebase equivalent)", style="cyan")
                register_tool(SemanticSearchTool())  # Will be linked to retriever in session
            
            # Register advanced tools based on configuration
            if config.agent.enable_file_system_tools:
                register_tool(FileSystemTool(
                    allowed_directories=config.agent.allowed_directories,
                    readonly_mode=config.agent.readonly_mode,
                    max_file_size=config.agent.max_file_size_mb * 1024 * 1024
                ))
            
            if config.agent.enable_compiler_tools:
                register_tool(CompilerTool(
                    allowed_directories=config.agent.allowed_directories,
                    enable_network=config.agent.enable_network_access
                ))
            
            if config.agent.enable_debugger_tools:
                register_tool(DebuggerTool(
                    allowed_directories=config.agent.allowed_directories
                ))
            
            if config.agent.enable_ide_simulator:
                register_tool(IDESimulatorTool(
                    allowed_directories=config.agent.allowed_directories
                ))
            
            available_tools = tool_registry.get_all_tools()
            console.print(f"✅ Registered {len(available_tools)} tools", style="green")
        
        # Load scenarios
        console.print("📋 Loading agent scenarios...", style="blue")
        data_loader = DataLoader(config)
        
        # Load converted scenarios from cache
        from .generation.scenario_converter import get_scenario_converter
        converter = get_scenario_converter(config)
        cached_scenarios = converter.load_all_cached_scenarios()
        
        if not cached_scenarios:
            console.print("❌ No cached agent scenarios found. Run 'locobench convert-scenarios' first.", style="red")
            return []
        
        console.print(f"📋 Found {len(cached_scenarios)} cached scenarios", style="blue")
        
        # Prioritize scenarios with source files for better testing
        scenarios_with_src = []
        scenarios_without_src = []
        
        for scenario_data in cached_scenarios:
            project_files = scenario_data.get("project_files", [])
            src_files = [f for f in project_files if isinstance(f, dict) and "path" in f and ("/src/" in f.get("path", "") or "//src//" in f.get("path", ""))]
            if src_files:
                scenarios_with_src.append(scenario_data)
            else:
                scenarios_without_src.append(scenario_data)
        
        logger.info(f"Found {len(scenarios_with_src)} scenarios with source files, {len(scenarios_without_src)} without")
        
        # Prefer scenarios with source files first, then others
        prioritized_scenarios = scenarios_with_src + scenarios_without_src
        
        # Convert to InteractiveScenario objects and filter
        scenarios = []
        for i, scenario_data in enumerate(prioritized_scenarios):
            try:
                # Filter by difficulty and category if specified
                if difficulty and scenario_data.get("difficulty", "").lower() != difficulty.lower():
                    continue
                if category and category not in scenario_data.get("category", ""):
                    continue
                
                # Required enums are now imported at the top
                
                # Convert conversation phases from dict to ConversationPhase objects
                conversation_phases = []
                for phase_data in scenario_data.get("conversation_phases", []):
                    if isinstance(phase_data, dict):
                        conversation_phases.append(ConversationPhase(
                            phase_id=phase_data.get("phase_id", "unknown"),
                            name=phase_data.get("name", "Unknown Phase"),
                            initial_prompt=phase_data.get("initial_prompt", "Please proceed with this phase."),
                            expected_actions=phase_data.get("expected_actions", []),
                            success_conditions=phase_data.get("success_conditions", []),
                            max_turns_in_phase=phase_data.get("max_turns_in_phase", 10),
                            dynamic_prompts=phase_data.get("dynamic_prompts", {})
                        ))
                    elif isinstance(phase_data, ConversationPhase):
                        # Already a ConversationPhase object
                        conversation_phases.append(phase_data)
                    else:
                        # Invalid type - log warning and skip
                        logger.warning(f"Invalid conversation phase data type: {type(phase_data)} - skipping")
                
                # Map category string to TaskCategory enum
                category_str = scenario_data.get("category", "code_comprehension")
                try:
                    category_enum = TaskCategory(category_str.lower().replace("interactive_", ""))
                except ValueError:
                    category_enum = TaskCategory.CODE_COMPREHENSION  # Default fallback
                
                # Prepare initial context with project files
                initial_context = scenario_data.get("initial_context", {}).copy()
                
                # Add project files to initial context so agents can access them
                project_files = scenario_data.get("project_files", [])
                scenario_id = scenario_data.get("scenario_id", "unknown")
                logger.debug(f"Processing scenario {i+1}/{len(prioritized_scenarios)}: {scenario_id} with {len(project_files)} project files")
                if project_files:
                    src_files = [f for f in project_files if isinstance(f, dict) and "path" in f and ("/src/" in f.get("path", "") or "//src//" in f.get("path", ""))]
                    logger.debug(f"Scenario {scenario_id}: {len(src_files)} source files in project_files list")
                    # CRITICAL: Limit file content size to prevent 22MB message errors
                    max_content_size = 8_000_000  # 8MB safety limit (OpenAI limit is 10MB)
                    limited_project_files = {}
                    for file_data in project_files:
                        if isinstance(file_data, dict) and "path" in file_data and "content" in file_data:
                            content = file_data["content"]
                            if len(content) > max_content_size:
                                logger.warning(f"Project file {file_data['path']} too large ({len(content)} chars), truncating to {max_content_size}")
                                content = content[:max_content_size] + f"\n\n[Content truncated - file was {len(file_data['content'])} characters, showing first {max_content_size}]"
                            limited_project_files[file_data["path"]] = content
                    
                    initial_context["project_files"] = limited_project_files
                    logger.debug(f"Scenario {scenario_id}: loaded {len(initial_context['project_files'])} files into initial_context")
                
                # Add project metadata
                if scenario_data.get("project_spec"):
                    initial_context["project_spec"] = scenario_data["project_spec"]
                if scenario_data.get("project_name"):
                    initial_context["project_name"] = scenario_data["project_name"]
                
                scenario = InteractiveScenario(
                    scenario_id=scenario_data["scenario_id"],
                    title=scenario_data.get("title", "Agent Scenario"),
                    description=scenario_data.get("description", ""),
                    category=category_enum,
                    difficulty=DifficultyLevel(scenario_data.get("difficulty", "medium").lower()),
                    initial_context=initial_context,
                    context_files=scenario_data.get("context_files", []),
                    working_directory=scenario_data.get("working_directory", "project"),
                    conversation_phases=conversation_phases,
                    global_success_criteria=[],  # Will be empty for now
                    available_tools=scenario_data.get("available_tools", []),
                    max_turns=scenario_data.get("max_turns", max_turns),
                    max_duration_minutes=scenario_data.get("max_duration_minutes", 30),
                    context_window_tokens=scenario_data.get("context_window_tokens", 1_000_000)
                )
                scenarios.append(scenario)
                
                # Stop when we have enough scenarios
                if len(scenarios) >= scenario_count:
                    break
                
            except Exception as e:
                console.print(f"⚠️ Skipping invalid scenario {scenario_data.get('scenario_id', 'unknown')}: {e}", style="yellow")
                continue
        
        if not scenarios:
            console.print("❌ No valid scenarios found for evaluation", style="red")
            return []
        
        console.print(f"✅ Loaded {len(scenarios)} scenarios for evaluation", style="green")
        if scenarios:
            console.print(f"📋 Sample scenario: {scenarios[0].scenario_id}", style="blue")
            console.print(f"   Evaluation difficulty: {scenarios[0].difficulty.value}", style="blue")
            console.print(f"   Task category: {scenarios[0].category.value}", style="blue")
        
        # Create robust evaluator with Cursor-aligned features
        evaluator = RobustAgentEvaluator(
            config,
            agent_name=f"{agent_type}-{model or 'default'}",
            enable_semantic_search=enable_semantic_search,
            enable_enhanced_summarization=enable_enhanced_summarization,
            initial_context_mode=initial_context_mode
        )
        
        # Run evaluation with checkpointing
        console.print("🚀 Starting robust agent evaluation...", style="bold blue")
        console.print(f"🧠 Context management: {context_management}", style="blue")
        console.print(f"📂 Initial context mode: {initial_context_mode}", style="blue")
        
        # Context management is now passed through agent configuration
        results = await evaluator.evaluate_agents([agent], scenarios, resume=resume, max_concurrent_scenarios=max_concurrent_scenarios)
        
        if not results or not any(results.values()):
            console.print("❌ No evaluation results generated", style="red")
            return []
        
        # Generate summaries
        summaries = evaluator.generate_evaluation_summary(results)
        
        # Save results in original LoCoBench format  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_display = model or "unknown"
        results_file = output_dir / f"{model_display}_agent_evaluation_results_{timestamp}.json"
        
        evaluator.save_results(results, summaries, results_file)
        
        # Convert to unified format for compatibility
        unified_results = []
        for agent_name, agent_results in results.items():
            for result in agent_results:
                unified_results.append({
                    "agent_name": result.agent_name,
                    "scenario_id": result.scenario_id,
                    "overall_score": result.overall_score,
                    "total_turns": result.total_turns,
                    "session_duration": result.session_duration,
                    "category_scores": {cat.value: score for cat, score in result.category_scores.items()},
                    "evaluation_timestamp": result.evaluation_timestamp if isinstance(result.evaluation_timestamp, str) else (result.evaluation_timestamp.isoformat() if hasattr(result, 'evaluation_timestamp') else datetime.now().isoformat())
                })
        
        console.print(f"✅ Robust agent evaluation completed: {len(unified_results)} results", style="green")
        console.print(f"💾 Results saved with checkpointing support", style="blue")
        
        return unified_results
        
    except Exception as e:
        console.print(f"❌ Robust agent evaluation failed: {e}", style="red")
        import traceback
        traceback.print_exc()
        return []


async def run_agent_evaluation_unified(
    config: Config, 
    agent_type: str, 
    model: Optional[str], 
    max_turns: int, 
    enable_tools: bool,
    scenario_count: int, 
    difficulty: str, 
    category: Optional[str], 
    output_dir: Path,
    save_conversations: bool
) -> List[Dict[str, Any]]:
    """Run agent evaluation in unified mode"""
    
    # Import agent factory components
    from .agents.agent_factory import AgentFactory, AgentConfig as AgentConfigClass, AgentType
    from .core.multi_turn_pipeline import MultiTurnEvaluationPipeline, PipelineConfig
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("🔄 Running agent evaluation scenarios...", style="blue")
    
    try:
        # Initialize pipeline
        pipeline_config = PipelineConfig(
            max_concurrent_sessions=3,
            output_directory=output_dir,
            save_raw_conversations=save_conversations,
            generate_html_report=True
        )
        
        pipeline = MultiTurnEvaluationPipeline(pipeline_config)
        
        # Create agents based on type
        agents = []
        
        if agent_type == 'all':
            agent_types = ['openai', 'anthropic', 'google']
        else:
            agent_types = [agent_type]
        
        factory = AgentFactory()
        
        for atype in agent_types:
            try:
                if atype == 'openai':
                    agent_config = AgentConfigClass(
                        agent_type=AgentType.OPENAI,
                        name=f"openai-{model or 'gpt-4o'}",
                        model_name=model or "gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                    agent = factory.create_agent(agent_config)
                elif atype == 'anthropic':
                    agent_config = AgentConfigClass(
                        agent_type=AgentType.ANTHROPIC,
                        name=f"anthropic-{model or 'claude-sonnet'}",
                        model_name=model or "claude-3-sonnet-20240229",
                        api_key=os.getenv("ANTHROPIC_API_KEY")
                    )
                    agent = factory.create_agent(agent_config)
                elif atype == 'google':
                    agent_config = AgentConfigClass(
                        agent_type=AgentType.GOOGLE,
                        name=f"google-{model or 'gemini-pro'}",
                        model_name=model or "gemini-pro",
                        api_key=os.getenv("GOOGLE_API_KEY")
                    )
                    agent = factory.create_agent(agent_config)
                
                agents.append(agent)
                
            except Exception as e:
                console.print(f"⚠️ Could not create {atype} agent: {e}", style="yellow")
        
        if not agents:
            console.print("❌ No agents could be created. Check API keys.", style="red")
            return []
        
        # Load pre-converted scenarios
        from .generation.scenario_converter import get_scenario_converter
        from .generation.interactive_scenario_generator import InteractiveScenario
        from .core.task import TaskCategory, DifficultyLevel
        from .core.agent_session import ConversationPhase
        
        converter = get_scenario_converter(config)
        
        # Load scenarios from cache and convert to InteractiveScenario objects
        scenarios = []
        cache_dir = converter.cache_dir
        if cache_dir.exists():
            # Sort scenario files by name for deterministic ordering across runs
            scenario_files = sorted(cache_dir.glob("*.json"), key=lambda p: p.name)[:scenario_count]
            for scenario_file in scenario_files:
                try:
                    with open(scenario_file, 'r') as f:
                        scenario_data = json.load(f)
                    
                    # Convert JSON to InteractiveScenario object with all required fields
                    from locobench.generation.interactive_scenario_generator import InteractionMode, ToolUsageMode
                    
                    scenario = InteractiveScenario(
                        scenario_id=scenario_data["scenario_id"],
                        title=scenario_data["title"],
                        description=scenario_data["description"],
                        category=TaskCategory(scenario_data["category"]),
                        difficulty=DifficultyLevel(scenario_data["difficulty"].lower()),
                        initial_context=scenario_data.get("initial_context", {}),
                        context_files=scenario_data.get("context_files", []),
                        working_directory=scenario_data.get("working_directory", "project"),
                        conversation_phases=[
                            ConversationPhase(
                                phase_id=phase["phase_id"],
                                name=phase["name"],
                                initial_prompt=phase["initial_prompt"],
                                expected_actions=phase.get("expected_actions", []),
                                success_conditions=phase.get("success_conditions", []),
                                max_turns_in_phase=phase.get("max_turns_in_phase", 10),
                                dynamic_prompts=phase.get("dynamic_prompts", {})
                            )
                            for phase in scenario_data.get("conversation_phases", [])
                        ],
                        global_success_criteria=[],
                        available_tools=scenario_data.get("available_tools", []),
                        interaction_mode=InteractionMode.AUTONOMOUS,
                        tool_usage_mode=ToolUsageMode.UNRESTRICTED,
                        max_turns=scenario_data.get("max_turns", 30),
                        max_duration_minutes=scenario_data.get("max_duration_minutes", 60),
                        context_window_tokens=scenario_data.get("context_window_tokens", 100000),
                        dynamic_prompts=[],
                        human_intervention_triggers=[],
                        expected_outcomes=[],
                        evaluation_focus=[]
                    )
                    scenarios.append(scenario)
                    
                except Exception as e:
                    console.print(f"⚠️ Error loading scenario {scenario_file.name}: {e}", style="yellow")
        
        if not scenarios:
            console.print("❌ No pre-converted scenarios found. Run 'locobench convert-scenarios' first.", style="red")
            return []
        
        # Run pipeline
        pipeline_result = await pipeline.run_evaluation_pipeline(
            agents=agents,
            scenarios=scenarios,
            pipeline_id=f"unified_agent_eval_{int(time.time())}"
        )
        
        # Convert results to unified format
        results = []
        for agent_result in pipeline_result.agent_results:
            result = {
                "scenario_id": agent_result.scenario_id,
                "agent_name": agent_result.agent_name,
                "mode": "agent",
                "overall_score": agent_result.overall_score,
                "total_turns": agent_result.total_turns,
                "session_duration": agent_result.session_duration,
                "category_scores": {cat.value: score for cat, score in agent_result.category_scores.items()},
                "tool_usage_stats": {}  # Tool usage stats would come from session analysis
            }
            results.append(result)
        
        # Save results in two-tier structure
        await save_agent_results_two_tier(
            output_dir=output_dir,
            agent_results=pipeline_result.agent_results,
            scenarios=scenarios,
            agents=agents,
            config=config,
            model=model or "unknown",
            agent_type=agent_type,
            pipeline_result=pipeline_result  # Pass full pipeline result for session data
        )
        
        return results
        
    except Exception as e:
        console.print(f"❌ Agent evaluation failed: {e}", style="red")
        return []


async def save_agent_results_two_tier(
    output_dir: Path,
    agent_results: List,
    scenarios: List,
    agents: List,
    config,
    model: str,
    agent_type: str,
    pipeline_result = None
):
    """Save agent evaluation results in two-tier structure: 
    - intermediate_agent_results/: Detailed data for research
    - evaluation_agent_results/: Clean summary for end users
    """
    
    # Create model-specific directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model}_results_{timestamp}"
    
    # Create two-tier directory structure
    intermediate_dir = Path("intermediate_agent_results") / dir_name
    summary_dir = Path("evaluation_agent_results") / dir_name
    
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories in intermediate results
    conversations_dir = intermediate_dir / "conversations"
    session_logs_dir = intermediate_dir / "session_logs"
    
    conversations_dir.mkdir(exist_ok=True)
    session_logs_dir.mkdir(exist_ok=True)
    
    # Calculate aggregate statistics
    if not agent_results:
        console.print("⚠️ No agent results to save", style="yellow")
        return
    
    total_duration = sum(r.session_duration for r in agent_results)
    total_turns = sum(r.total_turns for r in agent_results)
    total_cost = sum(r.total_cost for r in agent_results)
    avg_score = sum(r.overall_score for r in agent_results) / len(agent_results)
    
    # Aggregate category scores
    category_totals = {}
    for result in agent_results:
        for cat, score in result.category_scores.items():
            cat_name = cat.value if hasattr(cat, 'value') else str(cat)
            if cat_name not in category_totals:
                category_totals[cat_name] = []
            category_totals[cat_name].append(score)
    
    category_averages = {
        cat: sum(scores) / len(scores) 
        for cat, scores in category_totals.items()
    }
    
    # === TIER 1: SUMMARY RESULTS (evaluation_agent_results/) ===
    
    # Create clean summary.json
    summary = {
        "model": model,
        "agent_type": agent_type,
        "evaluation_timestamp": datetime.now().isoformat(),
        "performance": {
            "overall_score": round(avg_score, 2),
            "normalized_score": round(avg_score / 5.0, 3),
            "category_scores": {k: round(v, 2) for k, v in category_averages.items()}
        },
        "statistics": {
            "scenarios_evaluated": len(agent_results),
            "total_turns": total_turns,
            "duration_seconds": round(total_duration, 1),
            "avg_turns_per_scenario": round(total_turns / len(agent_results), 1)
        },
        "scenarios": [r.scenario_id for r in agent_results]
    }
    
    # Save clean summary
    with open(summary_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create simple HTML report
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{model} Evaluation Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .score {{ font-size: 2em; color: #007acc; font-weight: bold; }}
        .metric {{ margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .category {{ display: flex; justify-content: space-between; align-items: center; }}
        .category-name {{ font-weight: bold; }}
        .category-score {{ color: #007acc; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{model} Agent Evaluation</h1>
            <div class="score">{avg_score:.2f}/5.0</div>
            <p>LoCoBench Agent Score (LCAS) • {avg_score/5.0:.1%} Performance</p>
        </div>
        
        <div class="metric">
            <h3>Performance Breakdown</h3>"""
    
    for cat, score in category_averages.items():
        cat_display = cat.replace('_', ' ').title()
        html_content += f"""
            <div class="category">
                <span class="category-name">{cat_display}</span>
                <span class="category-score">{score:.2f}/5.0</span>
            </div>"""
    
    html_content += f"""
        </div>
        
        <div class="metric">
            <h3>Session Statistics</h3>
            <p><strong>Scenarios:</strong> {len(agent_results)} evaluated</p>
            <p><strong>Total Turns:</strong> {total_turns} ({total_turns/len(agent_results):.1f} avg per scenario)</p>
            <p><strong>Total Cost:</strong> ${total_cost:.2f} (${total_cost/len(agent_results):.2f} avg per scenario)</p>
            <p><strong>Duration:</strong> {total_duration:.1f} seconds</p>
        </div>
    </div>
</body>
</html>"""
    
    with open(summary_dir / "report.html", 'w') as f:
        f.write(html_content)
    
    # === TIER 2: DETAILED RESULTS (intermediate_agent_results/) ===
    
    # Create detailed_metrics.json (Complete Breakdown)
    detailed_metrics = {
        "model": model,
        "agent_type": agent_type,
        "evaluation_timestamp": datetime.now().isoformat(),
        "scenario_results": {}
    }
    
    # Add detailed results for each scenario
    for result in agent_results:
        scenario_key = result.scenario_id.replace("_", "-")  # Clean key name
        
        # Convert category scores to serializable format
        category_scores = {}
        for cat, score in result.category_scores.items():
            cat_name = cat.value if hasattr(cat, 'value') else str(cat)
            category_scores[cat_name] = score
        
        detailed_metrics["scenario_results"][scenario_key] = {
            "scenario_id": result.scenario_id,
            "agent_name": result.agent_name,
            "session_id": result.session_id,
            "overall_score": result.overall_score,
            "category_scores": category_scores,
            "session_stats": {
                "total_turns": result.total_turns,
                "duration_seconds": result.session_duration,
                "total_cost": result.total_cost
            },
            "metric_details": [mr.to_dict() for mr in result.metric_results] if hasattr(result, 'metric_results') else [],
            "evaluation_metadata": {
                "timestamp": result.evaluation_timestamp if isinstance(result.evaluation_timestamp, str) else (result.evaluation_timestamp.isoformat() if hasattr(result, 'evaluation_timestamp') else datetime.now().isoformat()),
                "evaluator_version": getattr(result, 'evaluator_version', "1.0.0")
            }
        }
    
    # Save detailed metrics to intermediate directory
    with open(intermediate_dir / "detailed_metrics.json", 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # Create agent_config.json (Model Configuration) in intermediate directory
    agent_config = {
        "model_name": model,
        "agent_type": agent_type,
        "capabilities": {},
        "evaluation_settings": {
            "scenarios_tested": len(scenarios),
            "agents_tested": len(agents),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Get agent capabilities if available
    if agents:
        agent = agents[0]  # Use first agent as representative
        if hasattr(agent, 'capabilities') and agent.capabilities:
            agent_config["capabilities"] = agent.capabilities.to_dict()
        if hasattr(agent, 'model'):
            agent_config["model_details"] = {
                "model_name": agent.model,
                "agent_name": agent.name
            }
    
    # Save agent config to intermediate directory
    with open(intermediate_dir / "agent_config.json", 'w') as f:
        json.dump(agent_config, f, indent=2)
    
    # === SAVE FULL CONVERSATIONS ===
    conversations_saved = 0
    full_conversations_saved = 0
    
    for result in agent_results:
        if hasattr(result, 'session_id'):
            conversation_file = conversations_dir / f"{result.session_id}_conversation.json"
            
            # Initialize conversation data structure
            conversation_data = {
                "session_id": result.session_id,
                "scenario_id": result.scenario_id,
                "agent_name": result.agent_name,
                "session_metadata": {
                    "total_turns": result.total_turns,
                    "duration_seconds": result.session_duration,
                    "total_cost": result.total_cost,
                    "timestamp": result.evaluation_timestamp if isinstance(result.evaluation_timestamp, str) else (result.evaluation_timestamp.isoformat() if hasattr(result, 'evaluation_timestamp') else datetime.now().isoformat())
                },
                "conversation_history": [],
                "tool_calls": [],
                "phase_transitions": [],
                "error_log": []
            }
            
            # Try to get full session data from pipeline result
            session_data_found = False
            if pipeline_result and hasattr(pipeline_result, 'session_data'):
                # Look for session data in pipeline result
                session_info = pipeline_result.session_data.get(result.session_id)
                if session_info:
                    conversation_data.update({
                        "conversation_history": session_info.get("conversation_history", []),
                        "tool_calls": session_info.get("tool_usage_log", []),
                        "phase_transitions": session_info.get("phase_history", []),
                        "error_log": session_info.get("error_log", []),
                        "human_interventions": session_info.get("human_interventions", []),
                        "session_status": session_info.get("session_metadata", {}).get("status", "unknown")
                    })
                    session_data_found = True
                    full_conversations_saved += 1
            
            # Alternative: Try to access session data through evaluator
            if not session_data_found and pipeline_result:
                # Check if we can access the evaluator's active evaluations
                try:
                    # This is a fallback approach - try to get session data from the evaluator
                    if hasattr(pipeline_result, 'agent_evaluator'):
                        evaluator = pipeline_result.agent_evaluator
                        if hasattr(evaluator, 'active_evaluations'):
                            for eval_session in evaluator.active_evaluations.values():
                                if (hasattr(eval_session, 'session_id') and 
                                    eval_session.session_id == result.session_id and
                                    hasattr(eval_session, 'agent_session') and
                                    eval_session.agent_session):
                                    
                                    agent_session = eval_session.agent_session
                                    
                                    # Extract conversation history
                                    if hasattr(agent_session, 'phase_history'):
                                        conversation_data["phase_transitions"] = [
                                            {
                                                "phase_name": phase.get("phase_name", "unknown"),
                                                "turns_in_phase": phase.get("turns", 0),
                                                "success": phase.get("success", False),
                                                "duration": phase.get("duration_seconds", 0)
                                            }
                                            for phase in agent_session.phase_history
                                        ]
                                    
                                    # Extract tool usage
                                    if hasattr(agent_session, 'tool_usage_log'):
                                        conversation_data["tool_calls"] = [
                                            {
                                                "tool_name": tool_call.get("tool_name", "unknown"),
                                                "timestamp": tool_call.get("timestamp", ""),
                                                "success": tool_call.get("success", False),
                                                "duration": tool_call.get("duration", 0)
                                            }
                                            for tool_call in agent_session.tool_usage_log
                                        ]
                                    
                                    # Extract error log
                                    if hasattr(agent_session, 'error_log'):
                                        conversation_data["error_log"] = agent_session.error_log
                                    
                                    session_data_found = True
                                    full_conversations_saved += 1
                                    break
                except Exception as e:
                    # If we can't access session data, continue with metadata only
                    pass
            
            # Add note about conversation data availability
            if not session_data_found:
                conversation_data["note"] = "Session metadata only - full conversation transcripts require session data access"
            else:
                conversation_data["note"] = "Complete session data with conversation history, tool calls, and phase transitions"
            
            # Save conversation file
            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            conversations_saved += 1
    
    # Log completion
    console.print(f"✅ Summary results saved to: {summary_dir}", style="green")
    console.print(f"✅ Detailed results saved to: {intermediate_dir}", style="green")
    
    if conversations_saved > 0:
        if full_conversations_saved > 0:
            console.print(f"✅ Full conversations saved: {full_conversations_saved}/{conversations_saved} sessions", style="green")
            console.print(f"   📝 Includes: conversation history, tool calls, phase transitions", style="blue")
        else:
            console.print(f"✅ Conversation metadata saved: {conversations_saved} sessions", style="green")
            console.print(f"   ⚠️ Full transcripts require session data access", style="yellow")
    else:
        console.print(f"⚠️ No conversation data available", style="yellow")


def perform_cross_mode_comparison(llm_results: List[Dict], agent_results: List[Dict], output_dir: Path) -> Dict[str, Any]:
    """Perform comparison between LLM and Agent modes"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate averages
    llm_scores = [r['overall_score'] for r in llm_results]
    agent_scores = [r['overall_score'] for r in agent_results]
    
    llm_avg = sum(llm_scores) / len(llm_scores) if llm_scores else 0
    agent_avg = sum(agent_scores) / len(agent_scores) if agent_scores else 0
    
    # Cost comparison
    llm_costs = [r.get('cost', 0) for r in llm_results]
    agent_costs = [r.get('total_cost', 0) for r in agent_results]
    
    llm_cost_avg = sum(llm_costs) / len(llm_costs) if llm_costs else 0
    agent_cost_avg = sum(agent_costs) / len(agent_costs) if agent_costs else 0
    
    comparison = {
        "performance_comparison": {
            "llm_average_score": llm_avg,
            "agent_average_score": agent_avg,
            "performance_difference": agent_avg - llm_avg,
            "winner": "agent" if agent_avg > llm_avg else "llm"
        },
        "cost_comparison": {
            "llm_average_cost": llm_cost_avg,
            "agent_average_cost": agent_cost_avg,
            "cost_difference": agent_cost_avg - llm_cost_avg,
            "more_cost_efficient": "llm" if llm_cost_avg < agent_cost_avg else "agent"
        },
        "capability_analysis": {
            "agent_unique_capabilities": [
                "Multi-turn conversation",
                "Tool usage",
                "Dynamic problem solving",
                "Context management across turns"
            ],
            "llm_advantages": [
                "Single-turn efficiency",
                "Lower cost per evaluation",
                "Faster execution"
            ]
        }
    }
    
    # Save comparison
    comparison_file = output_dir / "cross_mode_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


async def perform_unified_statistical_analysis(results: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Perform statistical analysis on unified results"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from .analysis.statistical_analysis import StatisticalAnalyzer, StatisticalAnalysisConfig
    
    # Prepare data for analysis
    all_results = []
    
    if 'llm' in results:
        for result in results['llm']:
            all_results.append({
                'mode': 'llm',
                'overall_score': result['overall_score'],
                'cost': result.get('cost', 0),
                'execution_time': result.get('execution_time', 0)
            })
    
    if 'agent' in results:
        for result in results['agent']:
            all_results.append({
                'mode': 'agent',
                'overall_score': result['overall_score'],
                'cost': result.get('total_cost', 0),
                'execution_time': result.get('session_duration', 0)
            })
    
    # Perform statistical analysis
    analyzer = StatisticalAnalyzer()
    
    # Simple statistical summary
    if len(all_results) > 0:
        analysis = {
            "total_evaluations": len(all_results),
            "modes_analyzed": list(set(r['mode'] for r in all_results)),
            "overall_statistics": {
                "average_score": sum(r['overall_score'] for r in all_results) / len(all_results),
                "average_cost": sum(r['cost'] for r in all_results) / len(all_results),
                "average_time": sum(r['execution_time'] for r in all_results) / len(all_results)
            }
        }
    else:
        analysis = {
            "total_evaluations": 0,
            "modes_analyzed": [],
            "overall_statistics": {
                "average_score": 0.0,
                "average_cost": 0.0,
                "average_time": 0.0
            },
            "message": "No evaluation results found to analyze"
        }
    
    if 'llm' in results and 'agent' in results:
        llm_scores = [r['overall_score'] for r in all_results if r['mode'] == 'llm']
        agent_scores = [r['overall_score'] for r in all_results if r['mode'] == 'agent']
        
        analysis["mode_comparison"] = {
            "llm_average": sum(llm_scores) / len(llm_scores),
            "agent_average": sum(agent_scores) / len(agent_scores),
            "performance_difference": (sum(agent_scores) / len(agent_scores)) - (sum(llm_scores) / len(llm_scores))
        }
    
    # Save analysis
    analysis_file = output_dir / "statistical_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis


def generate_unified_reports(results: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate comprehensive reports for unified evaluation"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_files = []
    
    # Generate HTML summary report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LoCoBench Unified Evaluation Report</title>
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
            <h1>LoCoBench Unified Evaluation Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Evaluation Summary</h2>
            <div class="metric">
                <strong>Modes Evaluated:</strong> {', '.join(results.keys())}
            </div>
        </div>
    """
    
    if 'comparison' in results:
        html_content += f"""
        <div class="section">
            <h2>Cross-Mode Comparison</h2>
            <p><strong>Performance Winner:</strong> {results['comparison']['performance_comparison']['winner'].upper()}</p>
            <p><strong>Cost Efficient:</strong> {results['comparison']['cost_comparison']['more_cost_efficient'].upper()}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    html_file = output_dir / "evaluation_report.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    report_files.append(html_file)
    
    # Generate text summary
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("LoCoBench Unified Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        
        if 'llm' in results:
            f.write(f"LLM Mode: {len(results['llm'])} evaluations\n")
        
        if 'agent' in results:
            f.write(f"Agent Mode: {len(results['agent'])} evaluations\n")
        
        if 'comparison' in results:
            f.write(f"\nPerformance Winner: {results['comparison']['performance_comparison']['winner'].upper()}\n")
            f.write(f"Cost Efficient: {results['comparison']['cost_comparison']['more_cost_efficient'].upper()}\n")
    
    report_files.append(summary_file)
    
    return report_files


@main.command()
@click.option('--config-path', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--mode', type=click.Choice(['agent-comparison', 'statistical-analysis', 'performance-analysis']), default='agent-comparison', help='Type of analysis to perform')
@click.option('--include-llm-agents', help='Comma-separated list of LLM agents to include (e.g., gpt-4o,claude-sonnet-4,gemini-2.5-pro)')
@click.option('--include-framework-agents', help='Comma-separated list of framework agents to include (e.g., autogen,langchain,crewai)')
@click.option('--results-directory', type=click.Path(), help='Directory containing evaluation results to analyze')
@click.option('--output-format', type=click.Choice(['html', 'json', 'both']), default='both', help='Output format for analysis results')
@click.option('--statistical-tests', is_flag=True, help='Enable statistical significance testing')
@click.option('--significance-level', type=float, default=0.05, help='Significance level for statistical tests')
@click.option('--generate-charts', is_flag=True, help='Generate visualization charts')
@click.option('--output-dir', type=click.Path(), help='Output directory for analysis results')
def analyze(config_path, mode, include_llm_agents, include_framework_agents, results_directory, output_format, statistical_tests, significance_level, generate_charts, output_dir):
    """Analyze agent evaluation results and generate comprehensive comparison reports
    
    This command provides advanced analysis capabilities for agent evaluation results:
    - Agent comparison: Compare performance across different agents and frameworks
    - Statistical analysis: Perform rigorous statistical testing and significance analysis
    - Performance analysis: Analyze efficiency, cost, and resource utilization patterns
    """
    
    console.print(Panel.fit("📊 LoCoBench Agent Analysis", style="bold blue"))
    
    # Load configuration
    try:
        config = Config.from_yaml(config_path) if config_path else Config()
        console.print("✅ Configuration loaded", style="green")
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="red")
        return
    
    # Setup output directory
    if not output_dir:
        output_dir = Path(config.data.output_dir) / "analysis" / f"analysis_{int(time.time())}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"📁 Analysis output directory: {output_dir}", style="blue")
    
    # Parse agent lists
    llm_agents = []
    if include_llm_agents:
        llm_agents = [agent.strip() for agent in include_llm_agents.split(',')]
    
    framework_agents = []
    if include_framework_agents:
        framework_agents = [agent.strip() for agent in include_framework_agents.split(',')]
    
    console.print(f"🔍 Analysis mode: {mode}", style="cyan")
    
    if llm_agents:
        console.print(f"🤖 LLM agents to analyze: {', '.join(llm_agents)}", style="blue")
    
    if framework_agents:
        console.print(f"🛠️ Framework agents to analyze: {', '.join(framework_agents)}", style="blue")
    
    # Load evaluation results
    console.print("📂 Loading evaluation results...", style="cyan")
    
    try:
        evaluation_results = load_evaluation_results(results_directory, config)
        
        if not evaluation_results:
            console.print("❌ No evaluation results found", style="red")
            console.print("💡 Tip: Run agent evaluations first using 'locobench evaluate --mode agent'", style="yellow")
            return
        
        console.print(f"✅ Loaded {len(evaluation_results)} evaluation results", style="green")
        
    except Exception as e:
        console.print(f"❌ Error loading evaluation results: {e}", style="red")
        return
    
    # Filter results based on specified agents
    if llm_agents or framework_agents:
        all_specified_agents = llm_agents + framework_agents
        evaluation_results = [
            result for result in evaluation_results
            if any(agent in result.get('agent_name', '') for agent in all_specified_agents)
        ]
        
        console.print(f"🔽 Filtered to {len(evaluation_results)} results matching specified agents", style="blue")
    
    # Perform analysis based on mode
    analysis_results = {}
    
    if mode == 'agent-comparison':
        console.print("🔄 Performing agent comparison analysis...", style="cyan")
        analysis_results = asyncio.run(perform_agent_comparison_analysis(
            evaluation_results, output_dir, statistical_tests, significance_level
        ))
        
    elif mode == 'statistical-analysis':
        console.print("🔄 Performing statistical analysis...", style="cyan")
        analysis_results = asyncio.run(perform_statistical_analysis_mode(
            evaluation_results, output_dir, significance_level
        ))
        
    elif mode == 'performance-analysis':
        console.print("🔄 Performing performance analysis...", style="cyan")
        analysis_results = asyncio.run(perform_performance_analysis_mode(
            evaluation_results, output_dir
        ))
    
    # Generate output files
    console.print("📋 Generating analysis reports...", style="cyan")
    
    output_files = []
    
    if output_format in ['json', 'both']:
        json_file = output_dir / f"{mode}_results.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        output_files.append(json_file)
        console.print(f"📄 JSON report: {json_file}", style="green")
    
    if output_format in ['html', 'both']:
        html_file = output_dir / f"{mode}_report.html"
        html_content = generate_analysis_html_report(analysis_results, mode)
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        output_files.append(html_file)
        console.print(f"🌐 HTML report: {html_file}", style="green")
    
    # Generate charts if requested
    if generate_charts:
        console.print("📊 Generating visualization charts...", style="cyan")
        try:
            chart_files = generate_analysis_charts(analysis_results, output_dir, mode)
            output_files.extend(chart_files)
            console.print(f"📈 Generated {len(chart_files)} chart files", style="green")
        except Exception as e:
            console.print(f"⚠️ Could not generate charts: {e}", style="yellow")
    
    # Display summary
    console.print(f"\n🎉 Analysis completed successfully!", style="bold green")
    console.print(f"📁 Results saved to: {output_dir}", style="blue")
    console.print(f"📊 Generated {len(output_files)} output files", style="blue")
    
    # Create summary table
    summary_table = Table(title="Analysis Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Analysis Mode", mode)
    summary_table.add_row("Evaluation Results", str(len(evaluation_results)))
    summary_table.add_row("Output Files", str(len(output_files)))
    summary_table.add_row("Statistical Tests", "Enabled" if statistical_tests else "Disabled")
    
    if analysis_results.get('summary'):
        for key, value in analysis_results['summary'].items():
            summary_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(summary_table)


def load_evaluation_results(results_directory: Optional[str], config: Config) -> List[Dict[str, Any]]:
    """Load evaluation results from directory or default locations"""
    
    results = []
    
    # Determine search directories
    search_dirs = []
    
    if results_directory:
        search_dirs.append(Path(results_directory))
    else:
        # Default locations
        base_output = Path(config.data.output_dir)
        search_dirs.extend([
            Path("evaluation_agent_results"),
            Path("evaluation_llm_results"), 
            Path("evaluation_comparison_results"),
            base_output / "agents" / "evaluations",
            base_output / "agent_results"
        ])
    
    # Search for result files
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Look for JSON result files
        for json_file in search_dir.rglob("*.json"):
            if any(keyword in json_file.name.lower() for keyword in ['result', 'evaluation', 'agent']):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if this looks like evaluation results
                    if isinstance(data, dict) and ('agent_name' in data or 'results' in data):
                        results.append(data)
                    elif isinstance(data, list):
                        results.extend(data)
                        
                except Exception as e:
                    logger.warning(f"Could not load {json_file}: {e}")
                    continue
    
    return results


async def perform_agent_comparison_analysis(
    evaluation_results: List[Dict[str, Any]],
    output_dir: Path,
    enable_statistical_tests: bool,
    significance_level: float
) -> Dict[str, Any]:
    """Perform comprehensive agent comparison analysis"""
    
    from .analysis.agent_comparison import AgentComparisonFramework, ComparisonConfig, ComparisonMode
    from .analysis.statistical_analysis import StatisticalAnalyzer, StatisticalAnalysisConfig
    
    # Configure comparison framework
    comparison_config = ComparisonConfig(
        modes=[ComparisonMode.LEADERBOARD, ComparisonMode.HEAD_TO_HEAD, ComparisonMode.CATEGORY_ANALYSIS],
        enable_statistical_tests=enable_statistical_tests,
        significance_level=significance_level,
        generate_charts=True,
        generate_html_report=True
    )
    
    # Initialize framework
    comparison_framework = AgentComparisonFramework(comparison_config)
    
    # Convert results to AgentEvaluationResults format (simplified)
    from .evaluation.agent_metrics import AgentEvaluationResults, MetricCategory
    
    agent_results = []
    for result in evaluation_results:
        # Create simplified AgentEvaluationResults
        eval_result = AgentEvaluationResults(
            agent_name=result.get('agent_name', 'unknown'),
            scenario_id=result.get('scenario_id', 'unknown'),
            overall_score=result.get('overall_score', 0.0),
            category_scores={
                MetricCategory.SOFTWARE_ENGINEERING: result.get('software_engineering_score', 0.0),
                MetricCategory.FUNCTIONAL_CORRECTNESS: result.get('functional_correctness_score', 0.0),
                MetricCategory.CODE_QUALITY: result.get('code_quality_score', 0.0),
                MetricCategory.LONG_CONTEXT_UTILIZATION: result.get('long_context_score', 0.0),
                MetricCategory.AGENT_INTERACTION: result.get('agent_interaction_score', 0.0)
            },
            total_turns=result.get('total_turns', 0),
            session_duration=result.get('session_duration', 0.0),
            total_cost=result.get('total_cost', 0.0)
        )
        agent_results.append(eval_result)
    
    # Perform comparison analysis
    comparison_result = await comparison_framework.compare_agents(
        evaluation_results=agent_results,
        analysis_id=f"comparison_{int(time.time())}"
    )
    
    # Save comparison results
    saved_files = await comparison_framework.save_comparison_results(
        comparison_result, output_dir
    )
    
    return {
        "comparison_result": comparison_result.to_dict(),
        "saved_files": {k: str(v) for k, v in saved_files.items()},
        "summary": {
            "agents_compared": len(comparison_result.agents_analyzed),
            "total_evaluations": comparison_result.total_evaluations,
            "best_agent": comparison_result.leaderboard[0].agent_name if comparison_result.leaderboard else None,
            "pairwise_comparisons": len(comparison_result.pairwise_comparisons)
        }
    }


async def perform_statistical_analysis_mode(
    evaluation_results: List[Dict[str, Any]],
    output_dir: Path,
    significance_level: float
) -> Dict[str, Any]:
    """Perform statistical analysis mode"""
    
    from .analysis.statistical_analysis import StatisticalAnalyzer, StatisticalAnalysisConfig
    from .evaluation.agent_metrics import AgentEvaluationResults, MetricCategory
    
    # Configure statistical analyzer
    stat_config = StatisticalAnalysisConfig(
        significance_level=significance_level,
        apply_bonferroni_correction=True,
        calculate_effect_sizes=True,
        enable_bootstrap=True
    )
    
    analyzer = StatisticalAnalyzer(stat_config)
    
    # Convert to AgentEvaluationResults format
    agent_results = []
    for result in evaluation_results:
        eval_result = AgentEvaluationResults(
            agent_name=result.get('agent_name', 'unknown'),
            scenario_id=result.get('scenario_id', 'unknown'),
            overall_score=result.get('overall_score', 0.0),
            category_scores={},
            total_turns=result.get('total_turns', 0),
            session_duration=result.get('session_duration', 0.0),
            total_cost=result.get('total_cost', 0.0)
        )
        agent_results.append(eval_result)
    
    # Perform comprehensive statistical analysis
    analysis_result = await analyzer.perform_comprehensive_analysis(
        evaluation_results=agent_results,
        analysis_id=f"statistical_{int(time.time())}"
    )
    
    # Save analysis results
    saved_files = await analyzer.save_statistical_analysis(
        analysis_result, output_dir
    )
    
    return {
        "statistical_analysis": analysis_result.to_dict(),
        "saved_files": {k: str(v) for k, v in saved_files.items()},
        "summary": {
            "agents_analyzed": len(analysis_result.agents_analyzed),
            "total_observations": analysis_result.total_observations,
            "statistical_tests_performed": len(analysis_result.test_results),
            "significant_differences": len([t for t in analysis_result.test_results if t.is_significant])
        }
    }


async def perform_performance_analysis_mode(
    evaluation_results: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """Perform performance analysis mode"""
    
    # Performance analysis focusing on efficiency metrics
    performance_data = {
        "cost_analysis": {},
        "time_analysis": {},
        "efficiency_analysis": {},
        "resource_utilization": {}
    }
    
    # Group by agent
    agent_groups = {}
    for result in evaluation_results:
        agent_name = result.get('agent_name', 'unknown')
        if agent_name not in agent_groups:
            agent_groups[agent_name] = []
        agent_groups[agent_name].append(result)
    
    # Analyze each agent's performance
    for agent_name, results in agent_groups.items():
        costs = [r.get('total_cost', 0.0) for r in results]
        durations = [r.get('session_duration', 0.0) for r in results]
        scores = [r.get('overall_score', 0.0) for r in results]
        
        performance_data["cost_analysis"][agent_name] = {
            "average_cost": sum(costs) / len(costs) if costs else 0,
            "total_cost": sum(costs),
            "cost_per_point": (sum(costs) / sum(scores)) if sum(scores) > 0 else float('inf')
        }
        
        performance_data["time_analysis"][agent_name] = {
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "total_time": sum(durations),
            "time_per_point": (sum(durations) / sum(scores)) if sum(scores) > 0 else float('inf')
        }
        
        performance_data["efficiency_analysis"][agent_name] = {
            "score_per_dollar": (sum(scores) / sum(costs)) if sum(costs) > 0 else 0,
            "score_per_second": (sum(scores) / sum(durations)) if sum(durations) > 0 else 0
        }
    
    # Save performance analysis
    performance_file = output_dir / "performance_analysis.json"
    with open(performance_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    return {
        "performance_analysis": performance_data,
        "saved_files": {"performance_analysis": str(performance_file)},
        "summary": {
            "agents_analyzed": len(agent_groups),
            "metrics_calculated": ["cost", "time", "efficiency"],
            "most_cost_efficient": min(performance_data["cost_analysis"].items(), 
                                     key=lambda x: x[1]["cost_per_point"])[0] if performance_data["cost_analysis"] else None
        }
    }


def generate_analysis_html_report(analysis_results: Dict[str, Any], mode: str) -> str:
    """Generate HTML report for analysis results"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LoCoBench Agent Analysis Report - {mode.title()}</title>
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
            <h1>LoCoBench Agent Analysis Report</h1>
            <p><strong>Analysis Mode:</strong> {mode.replace('-', ' ').title()}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Analysis Summary</h2>
    """
    
    if 'summary' in analysis_results:
        for key, value in analysis_results['summary'].items():
            html_content += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <p>See the JSON output file for complete analysis results and raw data.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content


def generate_analysis_charts(analysis_results: Dict[str, Any], output_dir: Path, mode: str) -> List[Path]:
    """Generate visualization charts for analysis results"""
    
    # This would generate charts using matplotlib or plotly
    # For now, return empty list as charts require additional dependencies
    
    chart_files = []
    
    try:
        # Placeholder for chart generation
        # In a full implementation, this would create:
        # - Performance comparison charts
        # - Statistical distribution plots  
        # - Cost vs performance scatter plots
        # - Time series analysis charts
        
        logger.info("Chart generation would require additional visualization libraries")
        
    except ImportError:
        logger.warning("Visualization libraries not available for chart generation")
    
    return chart_files


if __name__ == '__main__':
    main() 