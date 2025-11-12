"""
Data Loading Utilities for LoCoBench-Agent

This module provides unified data loading capabilities for both traditional
LLM evaluation and agent evaluation modes, ensuring seamless integration
with existing LoCoBench data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .config import Config
from .task import TaskCategory, DifficultyLevel

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Container for project context information"""
    project_id: str
    project_name: str
    project_dir: Path
    metadata: Dict[str, Any]
    files: List[Dict[str, str]]
    specification: Dict[str, Any]
    
    @property
    def exists(self) -> bool:
        """Check if project directory and metadata exist"""
        return self.project_dir.exists() and (self.project_dir / "project_metadata.json").exists()


@dataclass
class ScenarioData:
    """Container for scenario information"""
    scenario_id: str
    title: str
    description: str
    task_category: str
    difficulty: str
    context_files: List[str]
    project_context: Optional[ProjectContext] = None
    raw_data: Dict[str, Any] = None


class DataLoader:
    """Unified data loader for LoCoBench and LoCoBench-Agent"""
    
    def __init__(self, config: Config):
        self.config = config
        self.generated_dir = Path(config.data.generated_dir)
        self.scenarios_dir = Path(config.data.output_dir) / "scenarios"
    
    def load_project_context(self, project_id: str) -> Optional[ProjectContext]:
        """Load complete project context from generated data"""
        
        project_dir = self.generated_dir / project_id
        
        # If exact match doesn't exist, try to find a directory with this prefix
        if not project_dir.exists():
            # Look for directories that start with the project_id
            matching_dirs = list(self.generated_dir.glob(f"{project_id}_*"))
            if matching_dirs:
                # Use the first matching directory
                project_dir = matching_dirs[0]
                logger.debug(f"Using project directory: {project_dir} for project_id: {project_id}")
            else:
                logger.warning(f"Project directory not found: {project_dir}")
                return None
        
        metadata_file = project_dir / "project_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Project metadata not found: {metadata_file}")
            return None
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            specification = metadata.get('specification', {})
            project_name = specification.get('name', project_id)
            
            # Load all project files
            files = []
            for file_info in metadata.get('files', []):
                file_path = project_dir / file_info['path']
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        files.append({
                            "path": file_info['path'],
                            "content": content,
                            "type": file_info.get('type', 'source'),
                            "size": len(content)
                        })
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                else:
                    logger.warning(f"File not found: {file_path}")
            
            return ProjectContext(
                project_id=project_id,
                project_name=project_name,
                project_dir=project_dir,
                metadata=metadata,
                files=files,
                specification=specification
            )
            
        except Exception as e:
            logger.error(f"Error loading project context for {project_id}: {e}")
            return None
    
    def load_scenarios(
        self, 
        limit: Optional[int] = None,
        category_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        include_project_context: bool = True
    ) -> List[ScenarioData]:
        """Load evaluation scenarios from data/output/scenarios/"""
        
        scenarios = []
        
        if not self.scenarios_dir.exists():
            logger.warning(f"Scenarios directory not found: {self.scenarios_dir}")
            return scenarios
        
        scenario_files = list(self.scenarios_dir.glob("*.json"))
        
        # Apply filters
        if category_filter:
            scenario_files = [f for f in scenario_files if category_filter in f.name]
        if difficulty_filter:
            scenario_files = [f for f in scenario_files if difficulty_filter in f.name]
        
        # Apply limit
        if limit:
            scenario_files = scenario_files[:limit]
        
        logger.info(f"Loading {len(scenario_files)} scenarios from {self.scenarios_dir}")
        
        for scenario_file in scenario_files:
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                
                # Extract project ID from scenario ID
                scenario_id = scenario_data.get("id", scenario_file.stem)
                project_id = self._extract_project_id(scenario_id)
                
                # Load project context if requested
                project_context = None
                if include_project_context and project_id:
                    project_context = self.load_project_context(project_id)
                
                scenarios.append(ScenarioData(
                    scenario_id=scenario_id,
                    title=scenario_data.get("title", "Untitled Scenario"),
                    description=scenario_data.get("description", ""),
                    task_category=scenario_data.get("task_category", "code_comprehension"),
                    difficulty=scenario_data.get("difficulty", "medium"),
                    context_files=scenario_data.get("context_files", []),
                    project_context=project_context,
                    raw_data=scenario_data
                ))
                
            except Exception as e:
                logger.warning(f"Error loading scenario {scenario_file.name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(scenarios)} scenarios")
        return scenarios
    
    def load_scenario_by_id(self, scenario_id: str, include_project_context: bool = True) -> Optional[ScenarioData]:
        """Load a specific scenario by ID"""
        
        scenario_file = self.scenarios_dir / f"{scenario_id}.json"
        if not scenario_file.exists():
            logger.warning(f"Scenario file not found: {scenario_file}")
            return None
        
        try:
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            project_id = self._extract_project_id(scenario_id)
            project_context = None
            if include_project_context and project_id:
                project_context = self.load_project_context(project_id)
            
            return ScenarioData(
                scenario_id=scenario_id,
                title=scenario_data.get("title", "Untitled Scenario"),
                description=scenario_data.get("description", ""),
                task_category=scenario_data.get("task_category", "code_comprehension"),
                difficulty=scenario_data.get("difficulty", "medium"),
                context_files=scenario_data.get("context_files", []),
                project_context=project_context,
                raw_data=scenario_data
            )
            
        except Exception as e:
            logger.error(f"Error loading scenario {scenario_id}: {e}")
            return None
    
    def list_available_projects(self) -> List[str]:
        """List all available project IDs"""
        
        if not self.generated_dir.exists():
            return []
        
        projects = []
        for project_dir in self.generated_dir.iterdir():
            if project_dir.is_dir() and (project_dir / "project_metadata.json").exists():
                projects.append(project_dir.name)
        
        return sorted(projects)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about available data"""
        
        stats = {
            "projects": {
                "total": 0,
                "by_language": {},
                "by_complexity": {},
                "by_domain": {}
            },
            "scenarios": {
                "total": 0,
                "by_category": {},
                "by_difficulty": {}
            }
        }
        
        # Project statistics
        if self.generated_dir.exists():
            for project_dir in self.generated_dir.iterdir():
                if project_dir.is_dir():
                    metadata_file = project_dir / "project_metadata.json"
                    if metadata_file.exists():
                        stats["projects"]["total"] += 1
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            spec = metadata.get('specification', {})
                            language = spec.get('language', 'unknown')
                            complexity = spec.get('complexity', 'unknown')
                            domain = spec.get('domain', 'unknown')
                            
                            stats["projects"]["by_language"][language] = stats["projects"]["by_language"].get(language, 0) + 1
                            stats["projects"]["by_complexity"][complexity] = stats["projects"]["by_complexity"].get(complexity, 0) + 1
                            stats["projects"]["by_domain"][domain] = stats["projects"]["by_domain"].get(domain, 0) + 1
                            
                        except Exception as e:
                            logger.warning(f"Error reading metadata for {project_dir.name}: {e}")
        
        # Scenario statistics
        if self.scenarios_dir.exists():
            for scenario_file in self.scenarios_dir.glob("*.json"):
                stats["scenarios"]["total"] += 1
                try:
                    with open(scenario_file, 'r') as f:
                        scenario_data = json.load(f)
                    
                    category = scenario_data.get("task_category", "unknown")
                    difficulty = scenario_data.get("difficulty", "unknown")
                    
                    stats["scenarios"]["by_category"][category] = stats["scenarios"]["by_category"].get(category, 0) + 1
                    stats["scenarios"]["by_difficulty"][difficulty] = stats["scenarios"]["by_difficulty"].get(difficulty, 0) + 1
                    
                except Exception as e:
                    logger.warning(f"Error reading scenario {scenario_file.name}: {e}")
        
        return stats
    
    def _extract_project_id(self, scenario_id: str) -> Optional[str]:
        """Extract project ID from scenario ID"""
        
        # Scenario IDs follow pattern: {project_id}_{task_category}_{difficulty}_{instance}
        # Example: c_api_gateway_easy_009_architectural_understanding_expert_01
        # Project IDs follow pattern: language_domain_complexity_number
        # Example: c_api_gateway_easy_009
        
        # Full task category names (not partial matches)
        task_categories = {
            "architectural_understanding", "bug_investigation", "code_comprehension", 
            "cross_file_refactoring", "feature_implementation", "integration_testing", 
            "multi_session_development", "security_analysis"
        }
        
        # Look for task category patterns in the scenario ID
        for task_category in task_categories:
            if f"_{task_category}_" in scenario_id:
                # Find the position and extract project ID
                pos = scenario_id.find(f"_{task_category}_")
                return scenario_id[:pos]
        
        # Fallback: assume first 4 parts (original logic)
        parts = scenario_id.split("_")
        if len(parts) >= 4:
            return "_".join(parts[:4])
        
        return None


def get_data_loader(config: Config = None) -> DataLoader:
    """Get a configured data loader instance"""
    
    if config is None:
        config = Config()
    
    return DataLoader(config)
