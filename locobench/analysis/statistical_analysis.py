"""
Statistical Analysis Module for LoCoBench-Agent

This module provides comprehensive statistical analysis capabilities
for agent evaluation results, including hypothesis testing, effect sizes,
confidence intervals, and advanced statistical methods.
"""

import json
import logging
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from ..evaluation.agent_metrics import AgentEvaluationResults, MetricCategory

logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Statistical tests available for analysis"""
    T_TEST = "t_test"
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    KRUSKAL_WALLIS = "kruskal_wallis"
    ANOVA_ONE_WAY = "anova_one_way"
    FRIEDMAN = "friedman"
    CHI_SQUARE = "chi_square"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    SHAPIRO_WILK = "shapiro_wilk"


class EffectSizeMethod(Enum):
    """Effect size calculation methods"""
    COHEN_D = "cohen_d"
    GLASS_DELTA = "glass_delta"
    HEDGE_G = "hedge_g"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CLIFF_DELTA = "cliff_delta"


class ConfidenceLevel(Enum):
    """Standard confidence levels"""
    CI_90 = 0.90
    CI_95 = 0.95
    CI_99 = 0.99
    CI_99_9 = 0.999


@dataclass
class StatisticalTestResult:
    """Result from a statistical test"""
    
    test_name: str
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    
    # Interpretation
    is_significant: bool = False
    significance_level: float = 0.05
    
    # Effect size
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    
    # Confidence interval
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: float = 0.95
    
    # Additional statistics
    power: Optional[float] = None
    sample_sizes: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "is_significant": self.is_significant,
            "significance_level": self.significance_level,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "confidence_level": self.confidence_level,
            "power": self.power,
            "sample_sizes": self.sample_sizes
        }


@dataclass
class DescriptiveStatistics:
    """Comprehensive descriptive statistics"""
    
    # Basic statistics
    count: int
    mean: float
    median: float
    mode: Optional[float] = None
    
    # Variability
    std_dev: float = 0.0
    variance: float = 0.0
    range_value: float = 0.0
    iqr: float = 0.0
    
    # Distribution shape
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Extremes
    minimum: float = 0.0
    maximum: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    
    # Percentiles
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    # Outliers
    outliers: List[float] = field(default_factory=list)
    outlier_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std_dev": self.std_dev,
            "variance": self.variance,
            "range": self.range_value,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "q1": self.q1,
            "q3": self.q3,
            "percentiles": self.percentiles,
            "outliers": self.outliers,
            "outlier_count": self.outlier_count
        }


@dataclass
class StatisticalAnalysisConfig:
    """Configuration for statistical analysis"""
    
    # Significance testing
    significance_level: float = 0.05
    confidence_level: float = 0.95
    
    # Multiple comparisons
    apply_bonferroni_correction: bool = True
    apply_fdr_correction: bool = False
    
    # Effect sizes
    calculate_effect_sizes: bool = True
    effect_size_methods: List[EffectSizeMethod] = field(
        default_factory=lambda: [EffectSizeMethod.COHEN_D, EffectSizeMethod.CLIFF_DELTA]
    )
    
    # Tests to perform
    tests_to_perform: List[StatisticalTest] = field(
        default_factory=lambda: [
            StatisticalTest.T_TEST,
            StatisticalTest.MANN_WHITNEY_U,
            StatisticalTest.ANOVA_ONE_WAY,
            StatisticalTest.KRUSKAL_WALLIS
        ]
    )
    
    # Normality testing
    test_normality: bool = True
    normality_tests: List[StatisticalTest] = field(
        default_factory=lambda: [StatisticalTest.SHAPIRO_WILK, StatisticalTest.KOLMOGOROV_SMIRNOV]
    )
    
    # Outlier detection
    detect_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "z_score", "modified_z_score"
    outlier_threshold: float = 1.5
    
    # Bootstrap
    enable_bootstrap: bool = True
    bootstrap_iterations: int = 1000


@dataclass
class ComprehensiveStatisticalAnalysis:
    """Complete statistical analysis results"""
    
    analysis_id: str
    timestamp: datetime
    config: StatisticalAnalysisConfig
    
    # Input data summary
    agents_analyzed: List[str]
    total_observations: int
    
    # Descriptive statistics
    descriptive_stats: Dict[str, DescriptiveStatistics] = field(default_factory=dict)
    
    # Inferential statistics
    test_results: List[StatisticalTestResult] = field(default_factory=list)
    
    # Comparative analysis
    pairwise_comparisons: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    multiple_comparisons: Dict[str, Any] = field(default_factory=dict)
    
    # Distribution analysis
    normality_tests: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    distribution_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Effect size analysis
    effect_sizes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Bootstrap analysis
    bootstrap_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recommendations
    statistical_recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "agents_analyzed": self.agents_analyzed,
            "total_observations": self.total_observations,
            "descriptive_stats": {k: v.to_dict() for k, v in self.descriptive_stats.items()},
            "test_results": [test.to_dict() for test in self.test_results],
            "pairwise_comparisons": {k: v.to_dict() for k, v in self.pairwise_comparisons.items()},
            "multiple_comparisons": self.multiple_comparisons,
            "normality_tests": {k: v.to_dict() for k, v in self.normality_tests.items()},
            "distribution_summaries": self.distribution_summaries,
            "effect_sizes": self.effect_sizes,
            "bootstrap_results": self.bootstrap_results,
            "statistical_recommendations": self.statistical_recommendations
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis engine for LoCoBench-Agent
    
    This analyzer provides:
    1. Descriptive statistics for all metrics
    2. Inferential statistical testing
    3. Effect size calculations
    4. Multiple comparison corrections
    5. Distribution analysis and normality testing
    6. Bootstrap confidence intervals
    7. Power analysis
    8. Outlier detection and analysis
    """
    
    def __init__(self, config: StatisticalAnalysisConfig = None):
        self.config = config or StatisticalAnalysisConfig()
        
        # Check for optional statistical packages
        self.scipy_available = self._check_scipy()
        self.numpy_available = self._check_numpy()
        
        logger.info(f"StatisticalAnalyzer initialized (scipy: {self.scipy_available}, numpy: {self.numpy_available})")
    
    def _check_scipy(self) -> bool:
        """Check if scipy is available"""
        try:
            import scipy
            return True
        except ImportError:
            return False
    
    def _check_numpy(self) -> bool:
        """Check if numpy is available"""
        try:
            import numpy
            return True
        except ImportError:
            return False
    
    async def perform_comprehensive_analysis(
        self,
        evaluation_results: List[AgentEvaluationResults],
        analysis_id: str = None
    ) -> ComprehensiveStatisticalAnalysis:
        """
        Perform comprehensive statistical analysis of agent evaluation results
        
        Args:
            evaluation_results: List of agent evaluation results
            analysis_id: Optional analysis ID
            
        Returns:
            Complete statistical analysis results
        """
        
        if not analysis_id:
            analysis_id = f"stat_analysis_{int(datetime.now().timestamp())}"
        
        logger.info(f"Starting comprehensive statistical analysis: {analysis_id}")
        
        # Initialize analysis result
        analysis = ComprehensiveStatisticalAnalysis(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            config=self.config,
            agents_analyzed=list(set(r.agent_name for r in evaluation_results)),
            total_observations=len(evaluation_results)
        )
        
        # Group results by agent
        agent_results = self._group_results_by_agent(evaluation_results)
        
        # 1. Descriptive Statistics
        analysis.descriptive_stats = await self._calculate_descriptive_statistics(agent_results)
        
        # 2. Distribution Analysis
        if self.config.test_normality:
            analysis.normality_tests = await self._perform_normality_tests(agent_results)
            analysis.distribution_summaries = await self._analyze_distributions(agent_results)
        
        # 3. Inferential Statistics
        analysis.test_results = await self._perform_inferential_tests(agent_results)
        
        # 4. Pairwise Comparisons
        analysis.pairwise_comparisons = await self._perform_pairwise_comparisons(agent_results)
        
        # 5. Multiple Comparisons Correction
        analysis.multiple_comparisons = await self._apply_multiple_comparisons_correction(
            analysis.pairwise_comparisons
        )
        
        # 6. Effect Size Analysis
        if self.config.calculate_effect_sizes:
            analysis.effect_sizes = await self._calculate_effect_sizes(agent_results)
        
        # 7. Bootstrap Analysis
        if self.config.enable_bootstrap:
            analysis.bootstrap_results = await self._perform_bootstrap_analysis(agent_results)
        
        # 8. Generate Recommendations
        analysis.statistical_recommendations = await self._generate_statistical_recommendations(
            analysis
        )
        
        logger.info(f"Comprehensive statistical analysis completed: {analysis_id}")
        
        return analysis
    
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
    
    async def _calculate_descriptive_statistics(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, DescriptiveStatistics]:
        """Calculate comprehensive descriptive statistics for each agent"""
        
        logger.info("Calculating descriptive statistics")
        
        descriptive_stats = {}
        
        for agent_name, results in agent_results.items():
            scores = [r.overall_score for r in results]
            
            # Basic statistics
            count = len(scores)
            mean_val = statistics.mean(scores)
            median_val = statistics.median(scores)
            
            # Mode (if applicable)
            try:
                mode_val = statistics.mode(scores)
            except statistics.StatisticsError:
                mode_val = None
            
            # Variability
            if count > 1:
                std_dev = statistics.stdev(scores)
                variance = statistics.variance(scores)
            else:
                std_dev = 0.0
                variance = 0.0
            
            range_val = max(scores) - min(scores) if scores else 0.0
            
            # Quartiles and IQR
            sorted_scores = sorted(scores)
            q1 = self._percentile(sorted_scores, 25)
            q3 = self._percentile(sorted_scores, 75)
            iqr = q3 - q1
            
            # Percentiles
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentiles[p] = self._percentile(sorted_scores, p)
            
            # Outliers
            outliers = self._detect_outliers(scores)
            
            # Distribution shape (requires scipy)
            skewness = None
            kurtosis = None
            
            if self.scipy_available and count > 2:
                try:
                    from scipy import stats
                    skewness = stats.skew(scores)
                    kurtosis = stats.kurtosis(scores)
                except Exception as e:
                    logger.warning(f"Could not calculate skewness/kurtosis: {e}")
            
            descriptive_stats[agent_name] = DescriptiveStatistics(
                count=count,
                mean=mean_val,
                median=median_val,
                mode=mode_val,
                std_dev=std_dev,
                variance=variance,
                range_value=range_val,
                iqr=iqr,
                skewness=skewness,
                kurtosis=kurtosis,
                minimum=min(scores) if scores else 0.0,
                maximum=max(scores) if scores else 0.0,
                q1=q1,
                q3=q3,
                percentiles=percentiles,
                outliers=outliers,
                outlier_count=len(outliers)
            )
        
        return descriptive_stats
    
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile of sorted data"""
        
        if not sorted_data:
            return 0.0
        
        if percentile == 0:
            return sorted_data[0]
        if percentile == 100:
            return sorted_data[-1]
        
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _detect_outliers(self, data: List[float]) -> List[float]:
        """Detect outliers using the configured method"""
        
        if not data or len(data) < 4:
            return []
        
        if self.config.outlier_method == "iqr":
            return self._detect_outliers_iqr(data)
        elif self.config.outlier_method == "z_score":
            return self._detect_outliers_z_score(data)
        elif self.config.outlier_method == "modified_z_score":
            return self._detect_outliers_modified_z_score(data)
        else:
            return self._detect_outliers_iqr(data)
    
    def _detect_outliers_iqr(self, data: List[float]) -> List[float]:
        """Detect outliers using IQR method"""
        
        sorted_data = sorted(data)
        q1 = self._percentile(sorted_data, 25)
        q3 = self._percentile(sorted_data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.config.outlier_threshold * iqr
        upper_bound = q3 + self.config.outlier_threshold * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        
        return outliers
    
    def _detect_outliers_z_score(self, data: List[float]) -> List[float]:
        """Detect outliers using Z-score method"""
        
        if len(data) < 2:
            return []
        
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        
        if std_val == 0:
            return []
        
        threshold = self.config.outlier_threshold
        outliers = [x for x in data if abs((x - mean_val) / std_val) > threshold]
        
        return outliers
    
    def _detect_outliers_modified_z_score(self, data: List[float]) -> List[float]:
        """Detect outliers using modified Z-score method"""
        
        if len(data) < 2:
            return []
        
        median_val = statistics.median(data)
        mad = statistics.median([abs(x - median_val) for x in data])
        
        if mad == 0:
            return []
        
        threshold = self.config.outlier_threshold
        modified_z_scores = [0.6745 * (x - median_val) / mad for x in data]
        
        outliers = [data[i] for i, z in enumerate(modified_z_scores) if abs(z) > threshold]
        
        return outliers
    
    async def _perform_normality_tests(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, StatisticalTestResult]:
        """Perform normality tests on agent score distributions"""
        
        logger.info("Performing normality tests")
        
        normality_results = {}
        
        if not self.scipy_available:
            logger.warning("scipy not available for normality tests")
            return normality_results
        
        try:
            from scipy import stats
            
            for agent_name, results in agent_results.items():
                scores = [r.overall_score for r in results]
                
                if len(scores) < 3:
                    continue
                
                # Shapiro-Wilk test
                if StatisticalTest.SHAPIRO_WILK in self.config.normality_tests:
                    try:
                        statistic, p_value = stats.shapiro(scores)
                        
                        normality_results[f"{agent_name}_shapiro_wilk"] = StatisticalTestResult(
                            test_name="Shapiro-Wilk",
                            test_statistic=statistic,
                            p_value=p_value,
                            is_significant=p_value < self.config.significance_level,
                            significance_level=self.config.significance_level,
                            sample_sizes=[len(scores)]
                        )
                    except Exception as e:
                        logger.warning(f"Shapiro-Wilk test failed for {agent_name}: {e}")
                
                # Kolmogorov-Smirnov test (against normal distribution)
                if StatisticalTest.KOLMOGOROV_SMIRNOV in self.config.normality_tests:
                    try:
                        mean_val = statistics.mean(scores)
                        std_val = statistics.stdev(scores) if len(scores) > 1 else 1.0
                        
                        statistic, p_value = stats.kstest(scores, lambda x: stats.norm.cdf(x, mean_val, std_val))
                        
                        normality_results[f"{agent_name}_kolmogorov_smirnov"] = StatisticalTestResult(
                            test_name="Kolmogorov-Smirnov",
                            test_statistic=statistic,
                            p_value=p_value,
                            is_significant=p_value < self.config.significance_level,
                            significance_level=self.config.significance_level,
                            sample_sizes=[len(scores)]
                        )
                    except Exception as e:
                        logger.warning(f"Kolmogorov-Smirnov test failed for {agent_name}: {e}")
        
        except ImportError:
            logger.warning("scipy not available for normality tests")
        
        return normality_results
    
    async def _analyze_distributions(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze score distributions for each agent"""
        
        logger.info("Analyzing score distributions")
        
        distribution_summaries = {}
        
        for agent_name, results in agent_results.items():
            scores = [r.overall_score for r in results]
            
            if len(scores) < 2:
                continue
            
            # Basic distribution characteristics
            mean_val = statistics.mean(scores)
            median_val = statistics.median(scores)
            std_val = statistics.stdev(scores)
            
            # Distribution shape assessment
            distribution_summary = {
                "sample_size": len(scores),
                "mean": mean_val,
                "median": median_val,
                "std_dev": std_val,
                "coefficient_of_variation": std_val / mean_val if mean_val != 0 else float('inf'),
                "mean_median_difference": abs(mean_val - median_val),
                "distribution_shape": self._assess_distribution_shape(scores),
                "likely_normal": self._assess_normality_likelihood(scores)
            }
            
            distribution_summaries[agent_name] = distribution_summary
        
        return distribution_summaries
    
    def _assess_distribution_shape(self, scores: List[float]) -> str:
        """Assess the shape of a distribution"""
        
        if len(scores) < 3:
            return "insufficient_data"
        
        mean_val = statistics.mean(scores)
        median_val = statistics.median(scores)
        
        # Simple heuristic based on mean-median relationship
        diff = mean_val - median_val
        std_val = statistics.stdev(scores) if len(scores) > 1 else 1.0
        
        if abs(diff) < 0.1 * std_val:
            return "approximately_symmetric"
        elif diff > 0.1 * std_val:
            return "right_skewed"
        else:
            return "left_skewed"
    
    def _assess_normality_likelihood(self, scores: List[float]) -> str:
        """Assess likelihood of normal distribution"""
        
        if len(scores) < 5:
            return "insufficient_data"
        
        # Simple heuristics for normality assessment
        mean_val = statistics.mean(scores)
        median_val = statistics.median(scores)
        std_val = statistics.stdev(scores) if len(scores) > 1 else 1.0
        
        # Check mean-median closeness
        mean_median_close = abs(mean_val - median_val) < 0.2 * std_val
        
        # Check for outliers
        outliers = self._detect_outliers(scores)
        few_outliers = len(outliers) < 0.1 * len(scores)
        
        if mean_median_close and few_outliers:
            return "likely_normal"
        elif mean_median_close or few_outliers:
            return "possibly_normal"
        else:
            return "unlikely_normal"
    
    async def _perform_inferential_tests(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> List[StatisticalTestResult]:
        """Perform inferential statistical tests"""
        
        logger.info("Performing inferential statistical tests")
        
        test_results = []
        
        if len(agent_results) < 2:
            logger.warning("Need at least 2 agents for inferential tests")
            return test_results
        
        # Collect all scores for multi-group tests
        all_scores = []
        group_labels = []
        
        for agent_name, results in agent_results.items():
            scores = [r.overall_score for r in results]
            all_scores.extend(scores)
            group_labels.extend([agent_name] * len(scores))
        
        # Group scores by agent
        score_groups = [
            [r.overall_score for r in results] 
            for results in agent_results.values()
        ]
        
        if not self.scipy_available:
            logger.warning("scipy not available for statistical tests")
            return test_results
        
        try:
            from scipy import stats
            
            # One-way ANOVA
            if StatisticalTest.ANOVA_ONE_WAY in self.config.tests_to_perform:
                try:
                    f_stat, p_value = stats.f_oneway(*score_groups)
                    
                    test_results.append(StatisticalTestResult(
                        test_name="One-way ANOVA",
                        test_statistic=f_stat,
                        p_value=p_value,
                        degrees_of_freedom=len(agent_results) - 1,
                        is_significant=p_value < self.config.significance_level,
                        significance_level=self.config.significance_level,
                        sample_sizes=[len(group) for group in score_groups]
                    ))
                except Exception as e:
                    logger.warning(f"ANOVA test failed: {e}")
            
            # Kruskal-Wallis test
            if StatisticalTest.KRUSKAL_WALLIS in self.config.tests_to_perform:
                try:
                    h_stat, p_value = stats.kruskal(*score_groups)
                    
                    test_results.append(StatisticalTestResult(
                        test_name="Kruskal-Wallis",
                        test_statistic=h_stat,
                        p_value=p_value,
                        degrees_of_freedom=len(agent_results) - 1,
                        is_significant=p_value < self.config.significance_level,
                        significance_level=self.config.significance_level,
                        sample_sizes=[len(group) for group in score_groups]
                    ))
                except Exception as e:
                    logger.warning(f"Kruskal-Wallis test failed: {e}")
        
        except ImportError:
            logger.warning("scipy not available for statistical tests")
        
        return test_results
    
    async def _perform_pairwise_comparisons(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, StatisticalTestResult]:
        """Perform pairwise statistical comparisons between agents"""
        
        logger.info("Performing pairwise comparisons")
        
        pairwise_results = {}
        
        if len(agent_results) < 2:
            return pairwise_results
        
        if not self.scipy_available:
            logger.warning("scipy not available for pairwise tests")
            return pairwise_results
        
        try:
            from scipy import stats
            
            agent_names = list(agent_results.keys())
            
            for i, agent_a in enumerate(agent_names):
                for agent_b in agent_names[i + 1:]:
                    scores_a = [r.overall_score for r in agent_results[agent_a]]
                    scores_b = [r.overall_score for r in agent_results[agent_b]]
                    
                    comparison_key = f"{agent_a}_vs_{agent_b}"
                    
                    # T-test
                    if StatisticalTest.T_TEST in self.config.tests_to_perform:
                        try:
                            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                            
                            pairwise_results[f"{comparison_key}_ttest"] = StatisticalTestResult(
                                test_name="Independent t-test",
                                test_statistic=t_stat,
                                p_value=p_value,
                                degrees_of_freedom=len(scores_a) + len(scores_b) - 2,
                                is_significant=p_value < self.config.significance_level,
                                significance_level=self.config.significance_level,
                                sample_sizes=[len(scores_a), len(scores_b)]
                            )
                        except Exception as e:
                            logger.warning(f"T-test failed for {comparison_key}: {e}")
                    
                    # Mann-Whitney U test
                    if StatisticalTest.MANN_WHITNEY_U in self.config.tests_to_perform:
                        try:
                            u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
                            
                            pairwise_results[f"{comparison_key}_mannwhitney"] = StatisticalTestResult(
                                test_name="Mann-Whitney U",
                                test_statistic=u_stat,
                                p_value=p_value,
                                is_significant=p_value < self.config.significance_level,
                                significance_level=self.config.significance_level,
                                sample_sizes=[len(scores_a), len(scores_b)]
                            )
                        except Exception as e:
                            logger.warning(f"Mann-Whitney U test failed for {comparison_key}: {e}")
        
        except ImportError:
            logger.warning("scipy not available for pairwise tests")
        
        return pairwise_results
    
    async def _apply_multiple_comparisons_correction(
        self,
        pairwise_comparisons: Dict[str, StatisticalTestResult]
    ) -> Dict[str, Any]:
        """Apply multiple comparisons corrections"""
        
        logger.info("Applying multiple comparisons corrections")
        
        if not pairwise_comparisons:
            return {}
        
        # Extract p-values
        p_values = [result.p_value for result in pairwise_comparisons.values()]
        comparison_names = list(pairwise_comparisons.keys())
        
        corrections = {}
        
        # Bonferroni correction
        if self.config.apply_bonferroni_correction:
            bonferroni_alpha = self.config.significance_level / len(p_values)
            bonferroni_significant = [p < bonferroni_alpha for p in p_values]
            
            corrections["bonferroni"] = {
                "corrected_alpha": bonferroni_alpha,
                "significant_comparisons": [
                    comparison_names[i] for i, sig in enumerate(bonferroni_significant) if sig
                ],
                "total_comparisons": len(p_values),
                "significant_count": sum(bonferroni_significant)
            }
        
        # FDR correction (Benjamini-Hochberg)
        if self.config.apply_fdr_correction:
            try:
                if self.scipy_available:
                    from scipy.stats import false_discovery_control
                    
                    fdr_corrected = false_discovery_control(p_values, method='bh')
                    fdr_significant = [p < self.config.significance_level for p in fdr_corrected]
                    
                    corrections["fdr_bh"] = {
                        "corrected_p_values": fdr_corrected.tolist() if hasattr(fdr_corrected, 'tolist') else list(fdr_corrected),
                        "significant_comparisons": [
                            comparison_names[i] for i, sig in enumerate(fdr_significant) if sig
                        ],
                        "total_comparisons": len(p_values),
                        "significant_count": sum(fdr_significant)
                    }
                else:
                    # Simple BH procedure implementation
                    sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
                    fdr_significant = [False] * len(p_values)
                    
                    for i, idx in enumerate(sorted_indices):
                        threshold = (i + 1) / len(p_values) * self.config.significance_level
                        if p_values[idx] <= threshold:
                            fdr_significant[idx] = True
                        else:
                            break
                    
                    corrections["fdr_bh"] = {
                        "significant_comparisons": [
                            comparison_names[i] for i, sig in enumerate(fdr_significant) if sig
                        ],
                        "total_comparisons": len(p_values),
                        "significant_count": sum(fdr_significant)
                    }
            
            except Exception as e:
                logger.warning(f"FDR correction failed: {e}")
        
        return corrections
    
    async def _calculate_effect_sizes(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate effect sizes for pairwise comparisons"""
        
        logger.info("Calculating effect sizes")
        
        effect_sizes = {}
        
        if len(agent_results) < 2:
            return effect_sizes
        
        agent_names = list(agent_results.keys())
        
        for i, agent_a in enumerate(agent_names):
            for agent_b in agent_names[i + 1:]:
                scores_a = [r.overall_score for r in agent_results[agent_a]]
                scores_b = [r.overall_score for r in agent_results[agent_b]]
                
                comparison_key = f"{agent_a}_vs_{agent_b}"
                effect_sizes[comparison_key] = {}
                
                # Cohen's d
                if EffectSizeMethod.COHEN_D in self.config.effect_size_methods:
                    cohens_d = self._calculate_cohens_d(scores_a, scores_b)
                    effect_sizes[comparison_key]["cohens_d"] = cohens_d
                
                # Cliff's delta
                if EffectSizeMethod.CLIFF_DELTA in self.config.effect_size_methods:
                    cliffs_delta = self._calculate_cliffs_delta(scores_a, scores_b)
                    effect_sizes[comparison_key]["cliffs_delta"] = cliffs_delta
                
                # Glass's delta
                if EffectSizeMethod.GLASS_DELTA in self.config.effect_size_methods:
                    glass_delta = self._calculate_glass_delta(scores_a, scores_b)
                    effect_sizes[comparison_key]["glass_delta"] = glass_delta
        
        return effect_sizes
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        
        # Pooled standard deviation
        var1 = statistics.variance(group1)
        var2 = statistics.variance(group2)
        
        pooled_var = ((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / (len(group1) + len(group2) - 2)
        pooled_std = math.sqrt(pooled_var)
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta effect size"""
        
        if not group1 or not group2:
            return 0.0
        
        greater = 0
        less = 0
        
        for x in group1:
            for y in group2:
                if x > y:
                    greater += 1
                elif x < y:
                    less += 1
        
        total = len(group1) * len(group2)
        
        if total == 0:
            return 0.0
        
        return (greater - less) / total
    
    def _calculate_glass_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Glass's delta effect size"""
        
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        std2 = statistics.stdev(group2)
        
        if std2 == 0:
            return 0.0
        
        return (mean1 - mean2) / std2
    
    async def _perform_bootstrap_analysis(
        self,
        agent_results: Dict[str, List[AgentEvaluationResults]]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform bootstrap analysis for confidence intervals"""
        
        logger.info("Performing bootstrap analysis")
        
        bootstrap_results = {}
        
        import random
        
        for agent_name, results in agent_results.items():
            scores = [r.overall_score for r in results]
            
            if len(scores) < 2:
                continue
            
            # Bootstrap resampling
            bootstrap_means = []
            
            for _ in range(self.config.bootstrap_iterations):
                bootstrap_sample = random.choices(scores, k=len(scores))
                bootstrap_means.append(statistics.mean(bootstrap_sample))
            
            # Calculate confidence interval
            bootstrap_means.sort()
            
            alpha = 1 - self.config.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = self._percentile(bootstrap_means, lower_percentile)
            ci_upper = self._percentile(bootstrap_means, upper_percentile)
            
            bootstrap_results[agent_name] = {
                "bootstrap_mean": statistics.mean(bootstrap_means),
                "confidence_interval": [ci_lower, ci_upper],
                "confidence_level": self.config.confidence_level,
                "bootstrap_std": statistics.stdev(bootstrap_means),
                "iterations": self.config.bootstrap_iterations
            }
        
        return bootstrap_results
    
    async def _generate_statistical_recommendations(
        self,
        analysis: ComprehensiveStatisticalAnalysis
    ) -> List[str]:
        """Generate statistical recommendations based on analysis results"""
        
        recommendations = []
        
        # Sample size recommendations
        min_sample_size = min([stats.count for stats in analysis.descriptive_stats.values()])
        if min_sample_size < 10:
            recommendations.append(
                f"Consider increasing sample size (current minimum: {min_sample_size}). "
                "Small sample sizes may lead to unreliable statistical inferences."
            )
        
        # Normality recommendations
        normal_agents = 0
        for agent_name, dist_summary in analysis.distribution_summaries.items():
            if dist_summary.get("likely_normal") == "likely_normal":
                normal_agents += 1
        
        if normal_agents < len(analysis.distribution_summaries) / 2:
            recommendations.append(
                "Many distributions appear non-normal. Consider using non-parametric tests "
                "(Mann-Whitney U, Kruskal-Wallis) instead of parametric tests (t-test, ANOVA)."
            )
        
        # Multiple comparisons recommendations
        if len(analysis.pairwise_comparisons) > 3:
            recommendations.append(
                "Multiple pairwise comparisons detected. Consider applying Bonferroni or FDR "
                "correction to control for Type I error inflation."
            )
        
        # Effect size recommendations
        if analysis.effect_sizes:
            large_effects = []
            for comparison, effects in analysis.effect_sizes.items():
                if "cohens_d" in effects and abs(effects["cohens_d"]) > 0.8:
                    large_effects.append(comparison)
            
            if large_effects:
                recommendations.append(
                    f"Large effect sizes detected in: {', '.join(large_effects)}. "
                    "These differences may be practically significant."
                )
        
        # Outlier recommendations
        total_outliers = sum([stats.outlier_count for stats in analysis.descriptive_stats.values()])
        if total_outliers > 0:
            recommendations.append(
                f"Outliers detected ({total_outliers} total). Consider investigating these data points "
                "or using robust statistical methods."
            )
        
        return recommendations
    
    async def save_statistical_analysis(
        self,
        analysis: ComprehensiveStatisticalAnalysis,
        output_directory: Path
    ) -> Dict[str, Path]:
        """Save statistical analysis results to files"""
        
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save main analysis
        main_file = output_directory / f"{analysis.analysis_id}_statistical_analysis.json"
        with open(main_file, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)
        
        saved_files["main_analysis"] = main_file
        
        # Save descriptive statistics
        desc_file = output_directory / f"{analysis.analysis_id}_descriptive_stats.json"
        with open(desc_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in analysis.descriptive_stats.items()}, f, indent=2)
        
        saved_files["descriptive_stats"] = desc_file
        
        # Save test results
        if analysis.test_results:
            tests_file = output_directory / f"{analysis.analysis_id}_test_results.json"
            with open(tests_file, 'w') as f:
                json.dump([test.to_dict() for test in analysis.test_results], f, indent=2)
            
            saved_files["test_results"] = tests_file
        
        # Save recommendations
        if analysis.statistical_recommendations:
            rec_file = output_directory / f"{analysis.analysis_id}_recommendations.txt"
            with open(rec_file, 'w') as f:
                for i, rec in enumerate(analysis.statistical_recommendations, 1):
                    f.write(f"{i}. {rec}\n\n")
            
            saved_files["recommendations"] = rec_file
        
        logger.info(f"Statistical analysis results saved to: {output_directory}")
        
        return saved_files
