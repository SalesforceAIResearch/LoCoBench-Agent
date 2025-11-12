"""
Human Validation Framework for LoCoBench Metrics Revision

This module provides the framework for human expert validation of our
bias-free evaluation metrics. It enables systematic validation of metric
rankings against human expert judgment.

Key Features:
- Expert evaluation interface
- Metric ranking validation
- Statistical agreement analysis
- Bias detection in metric results
- Continuous validation pipeline
- Expert consensus measurement

This ensures our objective metrics align with human expert judgment
while maintaining bias-free evaluation.
"""

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ExpertiseLevel(Enum):
    """Expert expertise levels"""
    JUNIOR = "junior"           # 1-3 years experience
    INTERMEDIATE = "intermediate"  # 3-7 years experience
    SENIOR = "senior"           # 7-15 years experience
    EXPERT = "expert"           # 15+ years experience


class ValidationTask(Enum):
    """Types of validation tasks"""
    METRIC_RANKING = "metric_ranking"      # Rank solutions by quality
    BINARY_COMPARISON = "binary_comparison"  # A vs B comparison
    SCORE_ESTIMATION = "score_estimation"   # Estimate numeric score
    CATEGORY_CLASSIFICATION = "category_classification"  # Classify solution type


@dataclass
class ExpertProfile:
    """Expert evaluator profile"""
    expert_id: str
    name: str
    expertise_level: ExpertiseLevel
    specializations: List[str]  # e.g., ["web_development", "security", "performance"]
    years_experience: int
    validation_count: int = 0
    agreement_rate: float = 0.0
    last_validation: Optional[datetime] = None


@dataclass
class ValidationSample:
    """A sample for human validation"""
    sample_id: str
    scenario_id: str
    model_solutions: Dict[str, Dict[str, str]]  # model_name -> solution_code
    task_type: str
    complexity_level: str
    ground_truth_ranking: Optional[List[str]] = None  # If available
    created_at: datetime = None


@dataclass
class ExpertEvaluation:
    """Expert's evaluation of a validation sample"""
    evaluation_id: str
    expert_id: str
    sample_id: str
    task_type: ValidationTask
    rankings: Optional[List[str]] = None  # Model rankings (best to worst)
    scores: Optional[Dict[str, float]] = None  # Model scores (0-5)
    comparisons: Optional[Dict[str, str]] = None  # Pairwise comparisons
    confidence: float = 1.0  # Expert's confidence (0-1)
    time_spent_minutes: int = 0
    comments: str = ""
    submitted_at: datetime = None


@dataclass
class ValidationResult:
    """Result of validation analysis"""
    sample_id: str
    metric_ranking: List[str]
    expert_rankings: List[List[str]]
    agreement_scores: List[float]
    consensus_ranking: List[str]
    metric_accuracy: float
    expert_consensus: float
    validation_confidence: float


@dataclass
class MetricValidationReport:
    """Comprehensive metric validation report"""
    metric_name: str
    total_samples: int
    accuracy_rate: float
    agreement_rate: float
    bias_indicators: Dict[str, float]
    expert_feedback: List[str]
    recommendations: List[str]
    validation_date: datetime


class HumanValidationFramework:
    """
    Human validation framework for bias-free metric evaluation.
    
    This framework enables systematic validation of our objective metrics
    against human expert judgment to ensure they align with real-world
    software engineering expertise.
    """
    
    def __init__(self, validation_data_dir: str = "validation_data"):
        self.validation_data_dir = Path(validation_data_dir)
        self.validation_data_dir.mkdir(exist_ok=True)
        
        self.experts_file = self.validation_data_dir / "experts.json"
        self.samples_file = self.validation_data_dir / "validation_samples.json"
        self.evaluations_file = self.validation_data_dir / "expert_evaluations.json"
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self.experts = self._load_experts()
        self.validation_samples = self._load_validation_samples()
        self.expert_evaluations = self._load_expert_evaluations()
    
    # ===== EXPERT MANAGEMENT =====
    
    def register_expert(self, name: str, expertise_level: ExpertiseLevel, 
                       specializations: List[str], years_experience: int) -> str:
        """Register a new expert evaluator"""
        expert_id = self._generate_expert_id(name)
        
        expert = ExpertProfile(
            expert_id=expert_id,
            name=name,
            expertise_level=expertise_level,
            specializations=specializations,
            years_experience=years_experience
        )
        
        self.experts[expert_id] = expert
        self._save_experts()
        
        self.logger.info(f"Registered expert: {name} ({expert_id})")
        return expert_id
    
    def get_expert_profile(self, expert_id: str) -> Optional[ExpertProfile]:
        """Get expert profile by ID"""
        return self.experts.get(expert_id)
    
    def list_experts(self, expertise_level: Optional[ExpertiseLevel] = None,
                    specialization: Optional[str] = None) -> List[ExpertProfile]:
        """List experts with optional filtering"""
        experts = list(self.experts.values())
        
        if expertise_level:
            experts = [e for e in experts if e.expertise_level == expertise_level]
        
        if specialization:
            experts = [e for e in experts if specialization in e.specializations]
        
        return experts
    
    # ===== VALIDATION SAMPLE MANAGEMENT =====
    
    def create_validation_sample(self, scenario_id: str, model_solutions: Dict[str, Dict[str, str]],
                               task_type: str, complexity_level: str) -> str:
        """Create a new validation sample"""
        sample_id = self._generate_sample_id(scenario_id, model_solutions)
        
        sample = ValidationSample(
            sample_id=sample_id,
            scenario_id=scenario_id,
            model_solutions=model_solutions,
            task_type=task_type,
            complexity_level=complexity_level,
            created_at=datetime.now()
        )
        
        self.validation_samples[sample_id] = sample
        self._save_validation_samples()
        
        self.logger.info(f"Created validation sample: {sample_id}")
        return sample_id
    
    def get_validation_sample(self, sample_id: str) -> Optional[ValidationSample]:
        """Get validation sample by ID"""
        return self.validation_samples.get(sample_id)
    
    def list_validation_samples(self, task_type: Optional[str] = None,
                              complexity_level: Optional[str] = None) -> List[ValidationSample]:
        """List validation samples with optional filtering"""
        samples = list(self.validation_samples.values())
        
        if task_type:
            samples = [s for s in samples if s.task_type == task_type]
        
        if complexity_level:
            samples = [s for s in samples if s.complexity_level == complexity_level]
        
        return samples
    
    # ===== EXPERT EVALUATION =====
    
    def submit_expert_evaluation(self, expert_id: str, sample_id: str, 
                               task_type: ValidationTask, **evaluation_data) -> str:
        """Submit an expert evaluation"""
        evaluation_id = self._generate_evaluation_id(expert_id, sample_id)
        
        evaluation = ExpertEvaluation(
            evaluation_id=evaluation_id,
            expert_id=expert_id,
            sample_id=sample_id,
            task_type=task_type,
            submitted_at=datetime.now(),
            **evaluation_data
        )
        
        self.expert_evaluations[evaluation_id] = evaluation
        self._save_expert_evaluations()
        
        # Update expert profile
        if expert_id in self.experts:
            self.experts[expert_id].validation_count += 1
            self.experts[expert_id].last_validation = datetime.now()
            self._save_experts()
        
        self.logger.info(f"Submitted evaluation: {evaluation_id}")
        return evaluation_id
    
    def get_expert_evaluation(self, evaluation_id: str) -> Optional[ExpertEvaluation]:
        """Get expert evaluation by ID"""
        return self.expert_evaluations.get(evaluation_id)
    
    def list_expert_evaluations(self, expert_id: Optional[str] = None,
                              sample_id: Optional[str] = None) -> List[ExpertEvaluation]:
        """List expert evaluations with optional filtering"""
        evaluations = list(self.expert_evaluations.values())
        
        if expert_id:
            evaluations = [e for e in evaluations if e.expert_id == expert_id]
        
        if sample_id:
            evaluations = [e for e in evaluations if e.sample_id == sample_id]
        
        return evaluations
    
    # ===== VALIDATION ANALYSIS =====
    
    def validate_metric_rankings(self, sample_id: str, metric_rankings: Dict[str, List[str]]) -> ValidationResult:
        """
        Validate metric rankings against expert evaluations.
        This is the core validation function.
        """
        sample = self.validation_samples.get(sample_id)
        if not sample:
            raise ValueError(f"Sample not found: {sample_id}")
        
        # Get expert evaluations for this sample
        expert_evaluations = self.list_expert_evaluations(sample_id=sample_id)
        
        if not expert_evaluations:
            raise ValueError(f"No expert evaluations found for sample: {sample_id}")
        
        # Extract expert rankings
        expert_rankings = []
        for evaluation in expert_evaluations:
            if evaluation.rankings:
                expert_rankings.append(evaluation.rankings)
        
        if not expert_rankings:
            raise ValueError(f"No ranking evaluations found for sample: {sample_id}")
        
        # Calculate consensus ranking
        consensus_ranking = self._calculate_consensus_ranking(expert_rankings)
        
        # Calculate agreement scores for each metric
        agreement_scores = {}
        for metric_name, metric_ranking in metric_rankings.items():
            agreement_scores[metric_name] = self._calculate_ranking_agreement(
                metric_ranking, consensus_ranking
            )
        
        # Find best metric
        best_metric = max(agreement_scores.keys(), key=lambda k: agreement_scores[k])
        
        # Calculate expert consensus
        expert_consensus = self._calculate_expert_consensus(expert_rankings)
        
        # Calculate validation confidence
        validation_confidence = self._calculate_validation_confidence(
            expert_evaluations, expert_consensus
        )
        
        return ValidationResult(
            sample_id=sample_id,
            metric_ranking=metric_rankings[best_metric],
            expert_rankings=expert_rankings,
            agreement_scores=list(agreement_scores.values()),
            consensus_ranking=consensus_ranking,
            metric_accuracy=agreement_scores[best_metric],
            expert_consensus=expert_consensus,
            validation_confidence=validation_confidence
        )
    
    def generate_metric_validation_report(self, metric_name: str, 
                                        validation_results: List[ValidationResult]) -> MetricValidationReport:
        """Generate comprehensive validation report for a metric"""
        if not validation_results:
            raise ValueError("No validation results provided")
        
        # Calculate overall accuracy
        accuracy_scores = [result.metric_accuracy for result in validation_results]
        accuracy_rate = statistics.mean(accuracy_scores)
        
        # Calculate agreement rate
        agreement_scores = []
        for result in validation_results:
            agreement_scores.extend(result.agreement_scores)
        agreement_rate = statistics.mean(agreement_scores) if agreement_scores else 0.0
        
        # Detect bias indicators
        bias_indicators = self._detect_bias_indicators(validation_results)
        
        # Collect expert feedback
        expert_feedback = self._collect_expert_feedback(validation_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(accuracy_rate, bias_indicators)
        
        return MetricValidationReport(
            metric_name=metric_name,
            total_samples=len(validation_results),
            accuracy_rate=accuracy_rate,
            agreement_rate=agreement_rate,
            bias_indicators=bias_indicators,
            expert_feedback=expert_feedback,
            recommendations=recommendations,
            validation_date=datetime.now()
        )
    
    # ===== CONSENSUS AND AGREEMENT CALCULATION =====
    
    def _calculate_consensus_ranking(self, expert_rankings: List[List[str]]) -> List[str]:
        """Calculate consensus ranking from multiple expert rankings"""
        if not expert_rankings:
            return []
        
        # Get all models
        all_models = set()
        for ranking in expert_rankings:
            all_models.update(ranking)
        
        # Calculate average position for each model
        model_positions = {}
        for model in all_models:
            positions = []
            for ranking in expert_rankings:
                if model in ranking:
                    positions.append(ranking.index(model))
            
            if positions:
                model_positions[model] = statistics.mean(positions)
            else:
                model_positions[model] = len(all_models)  # Worst position
        
        # Sort by average position
        consensus_ranking = sorted(all_models, key=lambda m: model_positions[m])
        
        return consensus_ranking
    
    def _calculate_ranking_agreement(self, ranking1: List[str], ranking2: List[str]) -> float:
        """
        Calculate agreement between two rankings using Kendall's tau.
        Returns value between 0 (no agreement) and 1 (perfect agreement).
        """
        if not ranking1 or not ranking2:
            return 0.0
        
        # Find common items
        common_items = set(ranking1) & set(ranking2)
        if len(common_items) < 2:
            return 0.0
        
        # Create position mappings
        pos1 = {item: i for i, item in enumerate(ranking1) if item in common_items}
        pos2 = {item: i for i, item in enumerate(ranking2) if item in common_items}
        
        # Calculate concordant and discordant pairs
        concordant = 0
        discordant = 0
        
        items = list(common_items)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1, item2 = items[i], items[j]
                
                # Check if pair order is same in both rankings
                order1 = pos1[item1] < pos1[item2]
                order2 = pos2[item1] < pos2[item2]
                
                if order1 == order2:
                    concordant += 1
                else:
                    discordant += 1
        
        # Calculate Kendall's tau
        total_pairs = concordant + discordant
        if total_pairs == 0:
            return 1.0
        
        tau = (concordant - discordant) / total_pairs
        
        # Convert to 0-1 scale
        return (tau + 1) / 2
    
    def _calculate_expert_consensus(self, expert_rankings: List[List[str]]) -> float:
        """Calculate consensus level among expert rankings"""
        if len(expert_rankings) < 2:
            return 1.0
        
        # Calculate pairwise agreements
        agreements = []
        for i in range(len(expert_rankings)):
            for j in range(i + 1, len(expert_rankings)):
                agreement = self._calculate_ranking_agreement(
                    expert_rankings[i], expert_rankings[j]
                )
                agreements.append(agreement)
        
        return statistics.mean(agreements) if agreements else 0.0
    
    def _calculate_validation_confidence(self, expert_evaluations: List[ExpertEvaluation], 
                                       expert_consensus: float) -> float:
        """Calculate confidence in validation results"""
        if not expert_evaluations:
            return 0.0
        
        # Factor 1: Number of experts
        num_experts = len(expert_evaluations)
        expert_factor = min(num_experts / 5, 1.0)  # Optimal: 5+ experts
        
        # Factor 2: Expert consensus
        consensus_factor = expert_consensus
        
        # Factor 3: Expert confidence
        confidence_scores = [e.confidence for e in expert_evaluations if e.confidence > 0]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5
        
        # Factor 4: Expert expertise
        expertise_weights = {
            ExpertiseLevel.JUNIOR: 0.5,
            ExpertiseLevel.INTERMEDIATE: 0.7,
            ExpertiseLevel.SENIOR: 0.9,
            ExpertiseLevel.EXPERT: 1.0
        }
        
        expertise_scores = []
        for evaluation in expert_evaluations:
            expert = self.experts.get(evaluation.expert_id)
            if expert:
                weight = expertise_weights.get(expert.expertise_level, 0.5)
                expertise_scores.append(weight)
        
        avg_expertise = statistics.mean(expertise_scores) if expertise_scores else 0.5
        
        # Weighted combination
        validation_confidence = (
            expert_factor * 0.3 +
            consensus_factor * 0.3 +
            avg_confidence * 0.2 +
            avg_expertise * 0.2
        )
        
        return validation_confidence
    
    # ===== BIAS DETECTION =====
    
    def _detect_bias_indicators(self, validation_results: List[ValidationResult]) -> Dict[str, float]:
        """Detect potential bias indicators in validation results"""
        bias_indicators = {}
        
        # Bias 1: Systematic ranking bias (always favoring certain models)
        model_position_bias = self._calculate_model_position_bias(validation_results)
        bias_indicators["model_position_bias"] = model_position_bias
        
        # Bias 2: Complexity bias (performance varies by task complexity)
        complexity_bias = self._calculate_complexity_bias(validation_results)
        bias_indicators["complexity_bias"] = complexity_bias
        
        # Bias 3: Expert disagreement bias (low consensus indicates potential bias)
        disagreement_bias = self._calculate_disagreement_bias(validation_results)
        bias_indicators["disagreement_bias"] = disagreement_bias
        
        # Bias 4: Confidence bias (low expert confidence indicates uncertainty)
        confidence_bias = self._calculate_confidence_bias(validation_results)
        bias_indicators["confidence_bias"] = confidence_bias
        
        return bias_indicators
    
    def _calculate_model_position_bias(self, validation_results: List[ValidationResult]) -> float:
        """Calculate bias in model positioning"""
        # This would analyze if certain models are systematically ranked higher/lower
        # Simplified implementation
        return 0.0
    
    def _calculate_complexity_bias(self, validation_results: List[ValidationResult]) -> float:
        """Calculate bias related to task complexity"""
        # This would analyze if metric accuracy varies significantly by complexity
        # Simplified implementation
        return 0.0
    
    def _calculate_disagreement_bias(self, validation_results: List[ValidationResult]) -> float:
        """Calculate bias from expert disagreement"""
        consensus_scores = [result.expert_consensus for result in validation_results]
        avg_consensus = statistics.mean(consensus_scores) if consensus_scores else 1.0
        
        # Low consensus indicates potential bias
        return 1.0 - avg_consensus
    
    def _calculate_confidence_bias(self, validation_results: List[ValidationResult]) -> float:
        """Calculate bias from low validation confidence"""
        confidence_scores = [result.validation_confidence for result in validation_results]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 1.0
        
        # Low confidence indicates potential bias
        return 1.0 - avg_confidence
    
    # ===== EXPERT FEEDBACK AND RECOMMENDATIONS =====
    
    def _collect_expert_feedback(self, validation_results: List[ValidationResult]) -> List[str]:
        """Collect expert feedback from validation results"""
        feedback = []
        
        for result in validation_results:
            sample_evaluations = self.list_expert_evaluations(sample_id=result.sample_id)
            for evaluation in sample_evaluations:
                if evaluation.comments:
                    feedback.append(evaluation.comments)
        
        return feedback
    
    def _generate_recommendations(self, accuracy_rate: float, 
                                bias_indicators: Dict[str, float]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Accuracy-based recommendations
        if accuracy_rate < 0.6:
            recommendations.append("CRITICAL: Metric accuracy is below 60%. Consider complete redesign.")
        elif accuracy_rate < 0.75:
            recommendations.append("WARNING: Metric accuracy is below 75%. Significant improvements needed.")
        elif accuracy_rate < 0.85:
            recommendations.append("MODERATE: Metric accuracy is below 85%. Minor improvements recommended.")
        else:
            recommendations.append("GOOD: Metric accuracy is above 85%. Monitor for consistency.")
        
        # Bias-based recommendations
        for bias_type, bias_score in bias_indicators.items():
            if bias_score > 0.3:
                recommendations.append(f"HIGH BIAS DETECTED: {bias_type} score is {bias_score:.2f}. Investigate immediately.")
            elif bias_score > 0.2:
                recommendations.append(f"MODERATE BIAS: {bias_type} score is {bias_score:.2f}. Monitor closely.")
        
        return recommendations
    
    # ===== DATA PERSISTENCE =====
    
    def _load_experts(self) -> Dict[str, ExpertProfile]:
        """Load expert profiles from file"""
        if not self.experts_file.exists():
            return {}
        
        try:
            with open(self.experts_file, 'r') as f:
                data = json.load(f)
            
            experts = {}
            for expert_id, expert_data in data.items():
                expert_data['expertise_level'] = ExpertiseLevel(expert_data['expertise_level'])
                if expert_data.get('last_validation'):
                    expert_data['last_validation'] = datetime.fromisoformat(expert_data['last_validation'])
                experts[expert_id] = ExpertProfile(**expert_data)
            
            return experts
        except Exception as e:
            self.logger.error(f"Failed to load experts: {e}")
            return {}
    
    def _save_experts(self):
        """Save expert profiles to file"""
        try:
            data = {}
            for expert_id, expert in self.experts.items():
                expert_dict = asdict(expert)
                expert_dict['expertise_level'] = expert.expertise_level.value
                if expert.last_validation:
                    expert_dict['last_validation'] = expert.last_validation.isoformat()
                data[expert_id] = expert_dict
            
            with open(self.experts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save experts: {e}")
    
    def _load_validation_samples(self) -> Dict[str, ValidationSample]:
        """Load validation samples from file"""
        if not self.samples_file.exists():
            return {}
        
        try:
            with open(self.samples_file, 'r') as f:
                data = json.load(f)
            
            samples = {}
            for sample_id, sample_data in data.items():
                if sample_data.get('created_at'):
                    sample_data['created_at'] = datetime.fromisoformat(sample_data['created_at'])
                samples[sample_id] = ValidationSample(**sample_data)
            
            return samples
        except Exception as e:
            self.logger.error(f"Failed to load validation samples: {e}")
            return {}
    
    def _save_validation_samples(self):
        """Save validation samples to file"""
        try:
            data = {}
            for sample_id, sample in self.validation_samples.items():
                sample_dict = asdict(sample)
                if sample.created_at:
                    sample_dict['created_at'] = sample.created_at.isoformat()
                data[sample_id] = sample_dict
            
            with open(self.samples_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save validation samples: {e}")
    
    def _load_expert_evaluations(self) -> Dict[str, ExpertEvaluation]:
        """Load expert evaluations from file"""
        if not self.evaluations_file.exists():
            return {}
        
        try:
            with open(self.evaluations_file, 'r') as f:
                data = json.load(f)
            
            evaluations = {}
            for eval_id, eval_data in data.items():
                eval_data['task_type'] = ValidationTask(eval_data['task_type'])
                if eval_data.get('submitted_at'):
                    eval_data['submitted_at'] = datetime.fromisoformat(eval_data['submitted_at'])
                evaluations[eval_id] = ExpertEvaluation(**eval_data)
            
            return evaluations
        except Exception as e:
            self.logger.error(f"Failed to load expert evaluations: {e}")
            return {}
    
    def _save_expert_evaluations(self):
        """Save expert evaluations to file"""
        try:
            data = {}
            for eval_id, evaluation in self.expert_evaluations.items():
                eval_dict = asdict(evaluation)
                eval_dict['task_type'] = evaluation.task_type.value
                if evaluation.submitted_at:
                    eval_dict['submitted_at'] = evaluation.submitted_at.isoformat()
                data[eval_id] = eval_dict
            
            with open(self.evaluations_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save expert evaluations: {e}")
    
    # ===== UTILITY METHODS =====
    
    def _generate_expert_id(self, name: str) -> str:
        """Generate unique expert ID"""
        return f"expert_{hashlib.md5(name.encode()).hexdigest()[:8]}"
    
    def _generate_sample_id(self, scenario_id: str, model_solutions: Dict[str, Dict[str, str]]) -> str:
        """Generate unique sample ID"""
        content = f"{scenario_id}_{sorted(model_solutions.keys())}"
        return f"sample_{hashlib.md5(content.encode()).hexdigest()[:8]}"
    
    def _generate_evaluation_id(self, expert_id: str, sample_id: str) -> str:
        """Generate unique evaluation ID"""
        content = f"{expert_id}_{sample_id}_{datetime.now().isoformat()}"
        return f"eval_{hashlib.md5(content.encode()).hexdigest()[:8]}"
