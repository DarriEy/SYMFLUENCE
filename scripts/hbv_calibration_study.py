#!/usr/bin/env python3
"""
HBV Calibration Study - Multi-Algorithm Comparison (Publication-Ready)

This script runs comprehensive calibration studies comparing all available
optimization algorithms for HBV model calibration on lumped catchments.

Features:
- Uses max_eval (function evaluations) for fair algorithm comparison
  (iterations mean different things to different algorithms)
- Automatically computes optimal iterations/population_size per algorithm
- Preprocesses domain data for lumped HBV (basin-averaged forcing)
- Runs all specified calibration algorithms with multiple replicates
- Compares performance metrics:
  * KGE, NSE, logNSE, PBIAS, RMSE, MAE
  * Convergence rate and evaluations to target
  * Computational efficiency (time, evals/second)
- Statistical analysis:
  * Mean, std, 95% confidence intervals across runs
  * Friedman test for overall significance
  * Wilcoxon signed-rank pairwise tests
  * Algorithm rankings with critical difference diagrams
- Publication-quality outputs:
  * Vector graphics (PDF) at 300 dpi
  * LaTeX tables for direct manuscript inclusion
  * Convergence curves vs cumulative evaluations
- Supports single basin or batch processing

Usage:
    # Single basin with 1000 evaluations per algorithm, 10 runs each
    python hbv_calibration_study.py --config config_CAN_01AD003_macro.yaml --max-eval 1000 --n-runs 10

    # With specific algorithms
    python hbv_calibration_study.py --config config_CAN_01AD003_macro.yaml \\
        --algorithms dds pso de adam cmaes --max-eval 2000 --n-runs 20

    # Batch mode for all basins
    python hbv_calibration_study.py --batch --config-dir 0_config_files/ --max-eval 1000

    # Quick test (fewer evaluations, single run)
    python hbv_calibration_study.py --config config_CAN_01AD003_macro.yaml --quick

    # Specify number of parameters for optimal config calculation
    python hbv_calibration_study.py --config config.yaml --max-eval 1000 --n-params 12
"""

import sys
import json
import time
import yaml
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Add src to path if running from scripts directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))


# ============================================================================
# DATA CLASSES FOR RESULTS
# ============================================================================

@dataclass
class AlgorithmResult:
    """Results from a single algorithm calibration run."""
    algorithm: str
    run_id: int  # Run number for multi-run support
    seed: int  # Random seed used for this run
    final_kge: float
    final_nse: float
    best_params: Dict[str, float]
    n_evaluations: int
    runtime_seconds: float
    convergence_history: List[Dict[str, float]] = field(default_factory=list)
    error: Optional[str] = None

    # Additional performance metrics
    final_lognse: Optional[float] = None  # Log-transformed NSE for low flows
    final_pbias: Optional[float] = None  # Percent bias
    final_rmse: Optional[float] = None  # Root mean square error
    final_mae: Optional[float] = None  # Mean absolute error
    final_kge_alpha: Optional[float] = None  # KGE variability component
    final_kge_beta: Optional[float] = None  # KGE bias component
    final_kge_r: Optional[float] = None  # KGE correlation component

    # Derived metrics (computed after run)
    evals_to_target: Optional[int] = None
    convergence_rate: Optional[float] = None
    evals_per_second: Optional[float] = None
    area_under_curve: Optional[float] = None  # AUCC - area under convergence curve

    def compute_derived_metrics(self, target_kge: float = 0.7) -> None:
        """Compute derived metrics from convergence history."""
        if not self.convergence_history:
            return

        # Build cumulative evaluations list
        cumulative_evals = []
        total_evals = 0
        for record in self.convergence_history:
            total_evals += record.get('n_evals', 1)
            cumulative_evals.append(total_evals)
            record['cumulative_evals'] = total_evals  # Store for plotting

        # Evaluations to target
        for i, record in enumerate(self.convergence_history):
            if record.get('best_score', -999) >= target_kge:
                self.evals_to_target = cumulative_evals[i]
                break

        # Convergence rate (average improvement per 100 evaluations in first half)
        if len(self.convergence_history) > 1:
            first_half = self.convergence_history[:len(self.convergence_history)//2]
            if len(first_half) > 1:
                scores = [r.get('best_score', -999) for r in first_half]
                improvement = scores[-1] - scores[0]
                n_evals = sum(r.get('n_evals', 1) for r in first_half)
                if n_evals > 0:
                    self.convergence_rate = improvement / n_evals * 100

        # Area Under Convergence Curve (AUCC) - normalized
        # Higher is better (more area = better performance throughout)
        if len(self.convergence_history) > 1 and cumulative_evals[-1] > 0:
            scores = [max(0, r.get('best_score', 0)) for r in self.convergence_history]
            # Trapezoidal integration normalized by total evaluations
            self.area_under_curve = np.trapz(scores, cumulative_evals) / cumulative_evals[-1]

        # Evaluations per second
        if self.runtime_seconds > 0:
            self.evals_per_second = self.n_evaluations / self.runtime_seconds


@dataclass
class MultiRunStatistics:
    """Statistical summary across multiple runs of an algorithm."""
    algorithm: str
    n_runs: int

    # KGE statistics
    kge_mean: float
    kge_std: float
    kge_ci_lower: float  # 95% CI
    kge_ci_upper: float
    kge_median: float
    kge_iqr: float  # Interquartile range

    # NSE statistics
    nse_mean: float
    nse_std: float
    nse_ci_lower: float
    nse_ci_upper: float

    # Additional metrics (mean ± std)
    lognse_mean: Optional[float] = None
    lognse_std: Optional[float] = None
    pbias_mean: Optional[float] = None
    pbias_std: Optional[float] = None
    rmse_mean: Optional[float] = None
    rmse_std: Optional[float] = None

    # Efficiency metrics
    evals_to_target_mean: Optional[float] = None
    evals_to_target_std: Optional[float] = None
    runtime_mean: float = 0.0
    runtime_std: float = 0.0
    aucc_mean: Optional[float] = None
    aucc_std: Optional[float] = None

    # Ranking
    mean_rank: Optional[float] = None

    # All individual results for detailed analysis
    individual_results: List[AlgorithmResult] = field(default_factory=list)

    @classmethod
    def from_results(cls, algorithm: str, results: List[AlgorithmResult]) -> 'MultiRunStatistics':
        """Compute statistics from a list of individual run results."""
        n_runs = len(results)
        if n_runs == 0:
            raise ValueError("Cannot compute statistics from empty results")

        # Extract KGE values
        kge_vals = np.array([r.final_kge for r in results if r.final_kge > -900])
        nse_vals = np.array([r.final_nse for r in results if r.final_nse > -900])

        # Compute KGE statistics
        kge_mean = float(np.mean(kge_vals)) if len(kge_vals) > 0 else -999.0
        kge_std = float(np.std(kge_vals, ddof=1)) if len(kge_vals) > 1 else 0.0
        kge_median = float(np.median(kge_vals)) if len(kge_vals) > 0 else -999.0
        kge_iqr = float(np.percentile(kge_vals, 75) - np.percentile(kge_vals, 25)) if len(kge_vals) > 0 else 0.0

        # 95% Confidence Interval
        if len(kge_vals) > 1:
            ci = scipy_stats.t.interval(
                0.95, len(kge_vals) - 1,
                loc=kge_mean, scale=scipy_stats.sem(kge_vals)
            )
            kge_ci_lower, kge_ci_upper = float(ci[0]), float(ci[1])
        else:
            kge_ci_lower, kge_ci_upper = kge_mean, kge_mean

        # NSE statistics
        nse_mean = float(np.mean(nse_vals)) if len(nse_vals) > 0 else -999.0
        nse_std = float(np.std(nse_vals, ddof=1)) if len(nse_vals) > 1 else 0.0
        if len(nse_vals) > 1:
            ci = scipy_stats.t.interval(
                0.95, len(nse_vals) - 1,
                loc=nse_mean, scale=scipy_stats.sem(nse_vals)
            )
            nse_ci_lower, nse_ci_upper = float(ci[0]), float(ci[1])
        else:
            nse_ci_lower, nse_ci_upper = nse_mean, nse_mean

        # Additional metrics
        lognse_vals = [r.final_lognse for r in results if r.final_lognse is not None]
        pbias_vals = [r.final_pbias for r in results if r.final_pbias is not None]
        rmse_vals = [r.final_rmse for r in results if r.final_rmse is not None]

        # Efficiency metrics
        ett_vals = [r.evals_to_target for r in results if r.evals_to_target is not None]
        runtime_vals = [r.runtime_seconds for r in results]
        aucc_vals = [r.area_under_curve for r in results if r.area_under_curve is not None]

        return cls(
            algorithm=algorithm,
            n_runs=n_runs,
            kge_mean=kge_mean,
            kge_std=kge_std,
            kge_ci_lower=kge_ci_lower,
            kge_ci_upper=kge_ci_upper,
            kge_median=kge_median,
            kge_iqr=kge_iqr,
            nse_mean=nse_mean,
            nse_std=nse_std,
            nse_ci_lower=nse_ci_lower,
            nse_ci_upper=nse_ci_upper,
            lognse_mean=float(np.mean(lognse_vals)) if lognse_vals else None,
            lognse_std=float(np.std(lognse_vals, ddof=1)) if len(lognse_vals) > 1 else None,
            pbias_mean=float(np.mean(pbias_vals)) if pbias_vals else None,
            pbias_std=float(np.std(pbias_vals, ddof=1)) if len(pbias_vals) > 1 else None,
            rmse_mean=float(np.mean(rmse_vals)) if rmse_vals else None,
            rmse_std=float(np.std(rmse_vals, ddof=1)) if len(rmse_vals) > 1 else None,
            evals_to_target_mean=float(np.mean(ett_vals)) if ett_vals else None,
            evals_to_target_std=float(np.std(ett_vals, ddof=1)) if len(ett_vals) > 1 else None,
            runtime_mean=float(np.mean(runtime_vals)),
            runtime_std=float(np.std(runtime_vals, ddof=1)) if len(runtime_vals) > 1 else 0.0,
            aucc_mean=float(np.mean(aucc_vals)) if aucc_vals else None,
            aucc_std=float(np.std(aucc_vals, ddof=1)) if len(aucc_vals) > 1 else None,
            individual_results=results,
        )


@dataclass
class CalibrationStudyResult:
    """Complete results from a calibration study with multi-run support."""
    domain_name: str
    model: str
    timestamp: str
    target_kge: float
    max_evaluations: int
    n_runs: int = 1

    # Individual run results (all runs, all algorithms)
    algorithm_results: List[AlgorithmResult] = field(default_factory=list)

    # Multi-run statistics per algorithm
    algorithm_statistics: Dict[str, MultiRunStatistics] = field(default_factory=dict)

    # Summary statistics
    best_algorithm: Optional[str] = None
    best_kge: Optional[float] = None
    fastest_to_target: Optional[str] = None

    # Statistical test results
    friedman_statistic: Optional[float] = None
    friedman_pvalue: Optional[float] = None
    pairwise_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)  # p-values
    algorithm_ranks: Dict[str, float] = field(default_factory=dict)

    def compute_statistics(self) -> None:
        """Compute multi-run statistics for each algorithm."""
        # Group results by algorithm
        results_by_algo: Dict[str, List[AlgorithmResult]] = {}
        for r in self.algorithm_results:
            if r.final_kge > -900:  # Valid result
                if r.algorithm not in results_by_algo:
                    results_by_algo[r.algorithm] = []
                results_by_algo[r.algorithm].append(r)

        # Compute statistics for each algorithm
        for algo, results in results_by_algo.items():
            if results:
                self.algorithm_statistics[algo] = MultiRunStatistics.from_results(algo, results)

    def compute_summary(self) -> None:
        """Compute summary statistics across all algorithms."""
        # Use multi-run statistics if available
        if self.algorithm_statistics:
            valid_stats = {k: v for k, v in self.algorithm_statistics.items() if v.kge_mean > -900}
            if valid_stats:
                best_algo = max(valid_stats.items(), key=lambda x: x[1].kge_mean)
                self.best_algorithm = best_algo[0]
                self.best_kge = best_algo[1].kge_mean

                # Fastest to target (by mean)
                with_target = {k: v for k, v in valid_stats.items() if v.evals_to_target_mean is not None}
                if with_target:
                    fastest = min(with_target.items(), key=lambda x: x[1].evals_to_target_mean)
                    self.fastest_to_target = fastest[0]
        else:
            # Fallback to single-run logic
            valid_results = [r for r in self.algorithm_results if r.final_kge > -900]
            if valid_results:
                best_result = max(valid_results, key=lambda r: r.final_kge)
                self.best_algorithm = best_result.algorithm
                self.best_kge = best_result.final_kge

                results_with_target = [r for r in valid_results if r.evals_to_target is not None]
                if results_with_target:
                    fastest = min(results_with_target, key=lambda r: r.evals_to_target)
                    self.fastest_to_target = fastest.algorithm

    def run_statistical_tests(self) -> None:
        """Run Friedman test and pairwise Wilcoxon signed-rank tests."""
        if not self.algorithm_statistics or len(self.algorithm_statistics) < 2:
            return

        algorithms = list(self.algorithm_statistics.keys())
        n_algorithms = len(algorithms)

        # Get KGE values for each algorithm (need same number of runs for Friedman)
        kge_matrix = []
        min_runs = min(len(s.individual_results) for s in self.algorithm_statistics.values())

        if min_runs < 2:
            return  # Need at least 2 runs for statistical tests

        for algo in algorithms:
            results = self.algorithm_statistics[algo].individual_results[:min_runs]
            kge_matrix.append([r.final_kge for r in results])

        kge_matrix = np.array(kge_matrix)  # Shape: (n_algorithms, n_runs)

        # Friedman test (non-parametric test for repeated measures)
        try:
            stat, pval = scipy_stats.friedmanchisquare(*kge_matrix)
            self.friedman_statistic = float(stat)
            self.friedman_pvalue = float(pval)
        except Exception:
            pass

        # Compute ranks for each run, then average
        ranks_per_run = []
        for run_idx in range(min_runs):
            run_kges = kge_matrix[:, run_idx]
            # Rank (higher KGE = lower rank = better)
            ranks = scipy_stats.rankdata(-run_kges)
            ranks_per_run.append(ranks)

        mean_ranks = np.mean(ranks_per_run, axis=0)
        for i, algo in enumerate(algorithms):
            self.algorithm_ranks[algo] = float(mean_ranks[i])
            if algo in self.algorithm_statistics:
                self.algorithm_statistics[algo].mean_rank = float(mean_ranks[i])

        # Pairwise Wilcoxon signed-rank tests with Bonferroni correction
        n_comparisons = n_algorithms * (n_algorithms - 1) // 2
        self.pairwise_tests = {}

        for i, algo1 in enumerate(algorithms):
            self.pairwise_tests[algo1] = {}
            for j, algo2 in enumerate(algorithms):
                if i < j:
                    try:
                        stat, pval = scipy_stats.wilcoxon(
                            kge_matrix[i], kge_matrix[j],
                            alternative='two-sided'
                        )
                        # Bonferroni correction
                        corrected_pval = min(1.0, pval * n_comparisons)
                        self.pairwise_tests[algo1][algo2] = float(corrected_pval)
                    except Exception:
                        self.pairwise_tests[algo1][algo2] = 1.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all individual results to pandas DataFrame."""
        records = []
        for r in self.algorithm_results:
            records.append({
                'domain': self.domain_name,
                'algorithm': r.algorithm,
                'run_id': r.run_id,
                'seed': r.seed,
                'final_kge': r.final_kge,
                'final_nse': r.final_nse,
                'final_lognse': r.final_lognse,
                'final_pbias': r.final_pbias,
                'final_rmse': r.final_rmse,
                'final_mae': r.final_mae,
                'n_evaluations': r.n_evaluations,
                'runtime_seconds': r.runtime_seconds,
                'evals_to_target': r.evals_to_target,
                'area_under_curve': r.area_under_curve,
                'evals_per_second': r.evals_per_second,
                'error': r.error,
            })
        return pd.DataFrame(records)

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Convert multi-run statistics to summary DataFrame."""
        records = []
        for algo, stats in self.algorithm_statistics.items():
            records.append({
                'algorithm': algo,
                'n_runs': stats.n_runs,
                'kge_mean': stats.kge_mean,
                'kge_std': stats.kge_std,
                'kge_ci_lower': stats.kge_ci_lower,
                'kge_ci_upper': stats.kge_ci_upper,
                'nse_mean': stats.nse_mean,
                'nse_std': stats.nse_std,
                'lognse_mean': stats.lognse_mean,
                'pbias_mean': stats.pbias_mean,
                'rmse_mean': stats.rmse_mean,
                'runtime_mean': stats.runtime_mean,
                'runtime_std': stats.runtime_std,
                'evals_to_target_mean': stats.evals_to_target_mean,
                'aucc_mean': stats.aucc_mean,
                'mean_rank': stats.mean_rank,
            })
        return pd.DataFrame(records)

    def to_latex_table(self, metrics: List[str] = None, caption: str = None) -> str:
        """
        Generate publication-ready LaTeX table.

        Args:
            metrics: List of metrics to include. Default: ['kge', 'nse', 'pbias', 'rmse']
            caption: Table caption

        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ['kge', 'nse', 'lognse', 'pbias']

        if not self.algorithm_statistics:
            return "% No statistics available"

        # Sort by mean KGE descending
        sorted_algos = sorted(
            self.algorithm_statistics.items(),
            key=lambda x: x[1].kge_mean,
            reverse=True
        )

        # Build column headers
        metric_labels = {
            'kge': 'KGE',
            'nse': 'NSE',
            'lognse': 'logNSE',
            'pbias': 'PBIAS (\\%)',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'runtime': 'Time (s)',
            'rank': 'Rank',
        }

        cols = ['Algorithm'] + [metric_labels.get(m, m) for m in metrics]
        if self.algorithm_ranks:
            cols.append('Rank')

        n_cols = len(cols)

        # LaTeX header
        caption_text = caption or 'Algorithm comparison results (mean $\\pm$ std)'
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption_text}}}",
            "\\label{tab:calibration_results}",
            "\\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
            "\\toprule",
            " & ".join(cols) + " \\\\",
            "\\midrule",
        ]

        # Data rows
        best_kge = max(s.kge_mean for _, s in sorted_algos)

        for algo, stats in sorted_algos:
            row = [algo.upper()]

            for metric in metrics:
                if metric == 'kge':
                    val = f"{stats.kge_mean:.3f} $\\pm$ {stats.kge_std:.3f}"
                    if stats.kge_mean == best_kge:
                        val = f"\\textbf{{{val}}}"
                elif metric == 'nse':
                    val = f"{stats.nse_mean:.3f} $\\pm$ {stats.nse_std:.3f}"
                elif metric == 'lognse':
                    if stats.lognse_mean is not None:
                        val = f"{stats.lognse_mean:.3f}"
                        if stats.lognse_std:
                            val += f" $\\pm$ {stats.lognse_std:.3f}"
                    else:
                        val = "--"
                elif metric == 'pbias':
                    if stats.pbias_mean is not None:
                        val = f"{stats.pbias_mean:.1f}"
                        if stats.pbias_std:
                            val += f" $\\pm$ {stats.pbias_std:.1f}"
                    else:
                        val = "--"
                elif metric == 'rmse':
                    if stats.rmse_mean is not None:
                        val = f"{stats.rmse_mean:.3f}"
                        if stats.rmse_std:
                            val += f" $\\pm$ {stats.rmse_std:.3f}"
                    else:
                        val = "--"
                elif metric == 'runtime':
                    val = f"{stats.runtime_mean:.1f} $\\pm$ {stats.runtime_std:.1f}"
                else:
                    val = "--"

                row.append(val)

            # Add rank
            if self.algorithm_ranks and algo in self.algorithm_ranks:
                rank = self.algorithm_ranks[algo]
                row.append(f"{rank:.2f}")

            lines.append(" & ".join(row) + " \\\\")

        # Footer with statistical test results
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        if self.friedman_pvalue is not None:
            sig = "p < 0.001" if self.friedman_pvalue < 0.001 else f"p = {self.friedman_pvalue:.3f}"
            lines.append("\\\\[2pt]")
            lines.append(f"\\footnotesize{{Friedman test: $\\chi^2$ = {self.friedman_statistic:.2f}, {sig}}}")

        lines.append("\\end{table}")

        return "\n".join(lines)

    def to_latex_pairwise_table(self) -> str:
        """Generate LaTeX table of pairwise significance tests."""
        if not self.pairwise_tests:
            return "% No pairwise tests available"

        algorithms = sorted(self.algorithm_statistics.keys())
        n = len(algorithms)

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Pairwise Wilcoxon signed-rank test p-values (Bonferroni corrected)}",
            "\\label{tab:pairwise_tests}",
            "\\begin{tabular}{l" + "c" * n + "}",
            "\\toprule",
            " & " + " & ".join([a.upper() for a in algorithms]) + " \\\\",
            "\\midrule",
        ]

        for i, algo1 in enumerate(algorithms):
            row = [algo1.upper()]
            for j, algo2 in enumerate(algorithms):
                if i == j:
                    row.append("--")
                elif i < j:
                    pval = self.pairwise_tests.get(algo1, {}).get(algo2, 1.0)
                    if pval < 0.001:
                        row.append("$<$0.001")
                    elif pval < 0.05:
                        row.append(f"\\textbf{{{pval:.3f}}}")
                    else:
                        row.append(f"{pval:.3f}")
                else:
                    # Mirror the upper triangle
                    pval = self.pairwise_tests.get(algo2, {}).get(algo1, 1.0)
                    if pval < 0.001:
                        row.append("$<$0.001")
                    elif pval < 0.05:
                        row.append(f"\\textbf{{{pval:.3f}}}")
                    else:
                        row.append(f"{pval:.3f}")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[2pt]",
            "\\footnotesize{Bold values indicate statistical significance at $\\alpha$ = 0.05}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def save(self, output_path: Path) -> None:
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'domain_name': self.domain_name,
            'model': self.model,
            'timestamp': self.timestamp,
            'target_kge': self.target_kge,
            'max_evaluations': self.max_evaluations,
            'n_runs': self.n_runs,
            'best_algorithm': self.best_algorithm,
            'best_kge': float(self.best_kge) if self.best_kge is not None else None,
            'fastest_to_target': self.fastest_to_target,
            'friedman_statistic': self.friedman_statistic,
            'friedman_pvalue': self.friedman_pvalue,
            'algorithm_ranks': self.algorithm_ranks,
            'algorithm_statistics': {},
            'algorithm_results': []
        }

        # Save statistics
        for algo, stats in self.algorithm_statistics.items():
            data['algorithm_statistics'][algo] = {
                'n_runs': stats.n_runs,
                'kge_mean': stats.kge_mean,
                'kge_std': stats.kge_std,
                'kge_ci_lower': stats.kge_ci_lower,
                'kge_ci_upper': stats.kge_ci_upper,
                'nse_mean': stats.nse_mean,
                'nse_std': stats.nse_std,
                'lognse_mean': stats.lognse_mean,
                'pbias_mean': stats.pbias_mean,
                'rmse_mean': stats.rmse_mean,
                'runtime_mean': stats.runtime_mean,
                'mean_rank': stats.mean_rank,
            }

        # Save individual results
        for r in self.algorithm_results:
            result_dict = {
                'algorithm': r.algorithm,
                'run_id': r.run_id,
                'seed': r.seed,
                'final_kge': float(r.final_kge) if r.final_kge is not None else None,
                'final_nse': float(r.final_nse) if r.final_nse is not None else None,
                'final_lognse': float(r.final_lognse) if r.final_lognse is not None else None,
                'final_pbias': float(r.final_pbias) if r.final_pbias is not None else None,
                'final_rmse': float(r.final_rmse) if r.final_rmse is not None else None,
                'best_params': {k: float(v) for k, v in r.best_params.items()} if r.best_params else {},
                'n_evaluations': r.n_evaluations,
                'runtime_seconds': float(r.runtime_seconds),
                'evals_to_target': r.evals_to_target,
                'area_under_curve': float(r.area_under_curve) if r.area_under_curve else None,
                'evals_per_second': float(r.evals_per_second) if r.evals_per_second is not None else None,
                'error': r.error,
            }
            data['algorithm_results'].append(result_dict)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> 'CalibrationStudyResult':
        """Load results from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        result = cls(
            domain_name=data['domain_name'],
            model=data['model'],
            timestamp=data['timestamp'],
            target_kge=data['target_kge'],
            max_evaluations=data.get('max_evaluations', data.get('max_iterations', 1000)),
            n_runs=data.get('n_runs', 1),
            best_algorithm=data.get('best_algorithm'),
            best_kge=data.get('best_kge'),
            fastest_to_target=data.get('fastest_to_target'),
            friedman_statistic=data.get('friedman_statistic'),
            friedman_pvalue=data.get('friedman_pvalue'),
            algorithm_ranks=data.get('algorithm_ranks', {}),
        )

        for r_data in data.get('algorithm_results', []):
            alg_result = AlgorithmResult(
                algorithm=r_data['algorithm'],
                run_id=r_data.get('run_id', 0),
                seed=r_data.get('seed', 0),
                final_kge=r_data['final_kge'] or -999.0,
                final_nse=r_data['final_nse'] or -999.0,
                final_lognse=r_data.get('final_lognse'),
                final_pbias=r_data.get('final_pbias'),
                final_rmse=r_data.get('final_rmse'),
                best_params=r_data.get('best_params', {}),
                n_evaluations=r_data['n_evaluations'],
                runtime_seconds=r_data['runtime_seconds'],
                evals_to_target=r_data.get('evals_to_target'),
                area_under_curve=r_data.get('area_under_curve'),
                evals_per_second=r_data.get('evals_per_second'),
                error=r_data.get('error'),
            )
            result.algorithm_results.append(alg_result)

        # Recompute statistics from loaded results
        result.compute_statistics()

        return result


# ============================================================================
# CALIBRATION STUDY CLASS
# ============================================================================

class HBVCalibrationStudy:
    """
    Run calibration studies comparing multiple optimization algorithms for HBV.

    This class orchestrates:
    1. Preprocessing domain data for lumped HBV
    2. Running multiple calibration algorithms
    3. Collecting comprehensive comparison metrics
    4. Generating reports and visualizations
    """

    # Default single-objective algorithms to compare
    DEFAULT_ALGORITHMS = [
        'dds', 'pso', 'de', 'sce-ua', 'cmaes',
        'adam', 'nelder-mead', 'ga', 'simulated-annealing',
    ]

    # Quick test subset
    QUICK_ALGORITHMS = ['dds', 'pso', 'adam']

    def __init__(
        self,
        config_path: Path,
        algorithms: Optional[List[str]] = None,
        max_evaluations: int = 1000,
        n_runs: int = 1,
        target_kge: float = 0.7,
        output_dir: Optional[Path] = None,
        random_seed: Optional[int] = 42,
        logger: Optional[logging.Logger] = None,
        quick: bool = False,
        n_params: int = 10,
    ):
        """
        Initialize calibration study.

        Args:
            config_path: Path to basin configuration YAML file
            algorithms: List of algorithm names to compare
            max_evaluations: Maximum function evaluations per algorithm (fair comparison)
            n_runs: Number of replicate runs per algorithm for statistical analysis
            target_kge: Target KGE for "evaluations to target" metric
            output_dir: Directory for results
            random_seed: Base random seed for reproducibility (incremented per run)
            logger: Logger instance
            quick: If True, use reduced evaluations and algorithm set for testing
            n_params: Number of parameters (used to compute optimal algorithm configs)
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load configuration
        with open(self.config_path) as f:
            self.config_dict = yaml.safe_load(f)

        # Ensure required config keys for lumped HBV mode
        self._ensure_lumped_config()

        # Quick mode adjustments
        if quick:
            max_evaluations = min(max_evaluations, 500)
            algorithms = algorithms or self.QUICK_ALGORITHMS
            n_runs = 1  # Single run for quick mode

        self.algorithms = algorithms or self.DEFAULT_ALGORITHMS
        self.max_evaluations = max_evaluations
        self.n_runs = n_runs
        self.n_params = n_params
        self.target_kge = target_kge
        self.random_seed = random_seed
        self.quick = quick

        # Setup logging
        self.logger = logger or logging.getLogger(__name__)

        # Setup paths
        self.data_dir = Path(self.config_dict.get('SYMFLUENCE_DATA_DIR', '.'))
        self.domain_name = self.config_dict.get('DOMAIN_NAME', 'unknown')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.project_dir / 'studies' / 'hbv_calibration_study'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_lumped_config(self) -> None:
        """Ensure config has all required keys for lumped HBV mode."""
        # Add SUB_GRID_DISCRETIZATION if missing (required by SymfluenceConfig)
        if 'SUB_GRID_DISCRETIZATION' not in self.config_dict:
            self.config_dict['SUB_GRID_DISCRETIZATION'] = 'lumped'

        # For lumped mode, override DOMAIN_DEFINITION_METHOD to lumped
        # (preserving the original if user wants distributed later)
        self._original_domain_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped')

        # Ensure HBV model settings
        if 'model' not in self.config_dict:
            self.config_dict['model'] = {}
        if 'hbv' not in self.config_dict.get('model', {}):
            self.config_dict['model']['hbv'] = {}
        self.config_dict['model']['hbv']['spatial_mode'] = 'lumped'

    def get_algorithm_config(self, algorithm: str, max_eval: int, n_params: int) -> Dict[str, Any]:
        """
        Compute optimal configuration for an algorithm given an evaluation budget.

        Different algorithms interpret 'iterations' differently:
        - DDS: 1 eval per iteration
        - PSO/DE/GA: population_size evals per iteration
        - CMA-ES: population_size evals per generation
        - Adam: (2*n_params + 1) evals per step (finite-diff)
        - Nelder-Mead: ~2 evals per iteration (variable)
        - SA: steps_per_temp evals per iteration
        - SCE-UA: complex-based evaluation pattern

        Args:
            algorithm: Algorithm name
            max_eval: Maximum number of function evaluations
            n_params: Number of parameters being optimized

        Returns:
            Dict with 'iterations', 'population_size', and any algorithm-specific settings
        """
        algorithm_lower = algorithm.lower().replace('-', '_').replace(' ', '_')

        # Default population size heuristics from literature
        default_pop_size = max(20, min(50, 4 + int(3 * np.log(n_params)) * 2))

        config = {
            'iterations': 100,
            'population_size': default_pop_size,
            'extra_settings': {}
        }

        if algorithm_lower == 'dds':
            # DDS: 1 eval per iteration + 1 initial
            # evaluations = iterations + 1
            config['iterations'] = max(1, max_eval - 1)
            # Note: DDS doesn't use population, but validation requires >= 2
            config['population_size'] = 2

        elif algorithm_lower in ['pso', 'de', 'ga']:
            # Population-based: evaluations = (iterations + 1) * population_size
            # Optimal: balance exploration (population) vs exploitation (iterations)
            # Literature suggests pop_size ~ 10*n_params or sqrt(max_eval)
            pop_size = max(10, min(int(np.sqrt(max_eval)), 10 * n_params, 100))
            config['population_size'] = pop_size
            # iterations = max_eval / pop_size - 1 (accounting for initial population)
            config['iterations'] = max(1, max_eval // pop_size - 1)

        elif algorithm_lower == 'cmaes':
            # CMA-ES: evaluations = iterations * population_size
            # Recommended pop_size = 4 + floor(3 * ln(n_params))
            pop_size = 4 + int(3 * np.log(max(1, n_params)))
            # Ensure minimum viable population
            pop_size = max(pop_size, 6)
            config['population_size'] = pop_size
            config['iterations'] = max(1, max_eval // pop_size)

        elif algorithm_lower == 'adam':
            # Adam with finite differences: (2*n_params + 1) evals per step
            evals_per_step = 2 * n_params + 1
            config['iterations'] = max(1, max_eval // evals_per_step)
            # Note: ADAM doesn't use population, but validation requires >= 2
            config['population_size'] = 2
            # Adjust learning rate based on available steps
            if config['iterations'] < 50:
                config['extra_settings']['lr'] = 0.05  # Higher LR for fewer steps
            elif config['iterations'] > 500:
                config['extra_settings']['lr'] = 0.005  # Lower LR for more steps

        elif algorithm_lower == 'nelder_mead':
            # Nelder-Mead: (n_params + 1) initial + ~1-3 evals per iteration
            # Conservatively estimate ~2 evals per iteration on average
            initial_evals = n_params + 1
            remaining = max_eval - initial_evals
            config['iterations'] = max(1, remaining // 2)
            config['population_size'] = n_params + 1  # Simplex size

        elif algorithm_lower in ['simulated_annealing', 'sa']:
            # SA: steps_per_temp * iterations + 1 initial
            # Balance temperature annealing schedule with step count
            steps_per_temp = 10  # Default
            config['iterations'] = max(1, (max_eval - 1) // steps_per_temp)
            # Note: SA doesn't use population, but validation requires >= 2
            config['population_size'] = 2
            config['extra_settings']['steps_per_temp'] = steps_per_temp
            # Adjust cooling rate based on iterations
            if config['iterations'] < 50:
                config['extra_settings']['cooling_rate'] = 0.9  # Faster cooling
            elif config['iterations'] > 200:
                config['extra_settings']['cooling_rate'] = 0.98  # Slower cooling

        elif algorithm_lower == 'sce_ua':
            # SCE-UA: pop_size = n_complexes * (2*n_params + 1)
            # evaluations ~ pop_size * (1 + iterations)
            n_complexes = max(2, n_params // 2)
            n_per_complex = 2 * n_params + 1
            pop_size = n_complexes * n_per_complex
            config['population_size'] = pop_size
            # iterations = max_eval / pop_size - 1
            config['iterations'] = max(1, max_eval // pop_size - 1)

        elif algorithm_lower == 'basin_hopping':
            # Basin hopping: ~iterations evaluations (plus local minimizer evals)
            # Conservatively estimate 5-10 evals per iteration due to local search
            config['iterations'] = max(1, max_eval // 10)
            # Note: Basin-hopping doesn't use population, but validation requires >= 2
            config['population_size'] = 2

        elif algorithm_lower in ['dream', 'glue']:
            # MCMC-based: chain_length * n_chains evaluations
            n_chains = max(3, n_params)
            config['population_size'] = n_chains
            config['iterations'] = max(1, max_eval // n_chains)

        else:
            # Unknown algorithm - use conservative defaults
            # Assume population-based with moderate population
            self.logger.warning(f"Unknown algorithm '{algorithm}', using default config")
            pop_size = max(20, min(50, default_pop_size))
            config['population_size'] = pop_size
            config['iterations'] = max(1, max_eval // pop_size)

        # Log the computed configuration
        self.logger.debug(
            f"Algorithm config for {algorithm}: "
            f"iterations={config['iterations']}, pop_size={config['population_size']}, "
            f"estimated_evals~{config['iterations'] * config['population_size']}"
        )

        return config

    def _create_lumped_shapefile(self) -> Path:
        """
        Create a lumped (single-catchment) shapefile by dissolving distributed catchments.

        Returns:
            Path to lumped catchment shapefile.
        """
        self.logger.info("Creating lumped catchment shapefile from distributed catchments")

        import geopandas as gpd

        # Look for existing distributed catchment shapefiles
        catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
        river_basins_dir = self.project_dir / 'shapefiles' / 'river_basins'

        # Try to find existing catchment shapefile
        source_shp = None
        for pattern in [
            f"{self.domain_name}_HRUs_GRUs.shp",
            f"{self.domain_name}_HRUs*.shp",
            "*HRU*.shp",
        ]:
            matches = list(catchment_dir.glob(pattern)) if catchment_dir.exists() else []
            if matches:
                source_shp = matches[0]
                break

        # Try river_basins directory if no catchment found
        if not source_shp:
            for pattern in [
                f"{self.domain_name}_riverBasins*.shp",
                "*basin*.shp",
            ]:
                matches = list(river_basins_dir.glob(pattern)) if river_basins_dir.exists() else []
                if matches:
                    source_shp = matches[0]
                    break

        if not source_shp:
            raise FileNotFoundError(
                f"No catchment/basin shapefile found in {catchment_dir} or {river_basins_dir}. "
                "Cannot create lumped shapefile."
            )

        self.logger.info(f"Using source shapefile: {source_shp}")

        # Load and dissolve
        gdf = gpd.read_file(source_shp)
        self.logger.info(f"Source has {len(gdf)} features")

        # Dissolve all features into single polygon
        gdf['dissolve_key'] = 1
        lumped_gdf = gdf.dissolve(by='dissolve_key')

        # Calculate area in UTM
        utm_crs = lumped_gdf.estimate_utm_crs()
        lumped_utm = lumped_gdf.to_crs(utm_crs)
        total_area = lumped_utm.geometry.area.iloc[0]

        # Add required fields
        lumped_gdf = lumped_gdf.reset_index(drop=True)
        lumped_gdf['HRU_ID'] = 1
        lumped_gdf['GRU_ID'] = 1
        lumped_gdf['hruId'] = 1
        lumped_gdf['gruId'] = 1
        lumped_gdf['HRU_area'] = total_area
        lumped_gdf['GRU_area'] = total_area

        # Calculate centroid for forcing lookup
        centroid = lumped_gdf.to_crs(epsg=4326).geometry.centroid.iloc[0]
        lumped_gdf['center_lat'] = centroid.y
        lumped_gdf['center_lon'] = centroid.x

        # Save lumped shapefile
        lumped_dir = catchment_dir / 'lumped'
        lumped_dir.mkdir(parents=True, exist_ok=True)
        lumped_shp = lumped_dir / f"{self.domain_name}_lumped_catchment.shp"
        lumped_gdf.to_file(lumped_shp)

        self.logger.info(f"Created lumped shapefile: {lumped_shp}")
        self.logger.info(f"Total basin area: {total_area/1e6:.2f} km²")

        return lumped_shp

    def _get_fresh_config_dict(self) -> dict:
        """Load a fresh config dict from the YAML file to avoid frozen Pydantic object issues."""
        with open(self.config_path) as f:
            config_dict = yaml.safe_load(f)
        # Apply lumped settings
        if 'SUB_GRID_DISCRETIZATION' not in config_dict:
            config_dict['SUB_GRID_DISCRETIZATION'] = 'lumped'
        if 'model' not in config_dict:
            config_dict['model'] = {}
        if 'hbv' not in config_dict.get('model', {}):
            config_dict['model']['hbv'] = {}
        config_dict['model']['hbv']['spatial_mode'] = 'lumped'
        return config_dict

    def _remap_forcing_to_lumped(self, lumped_shp: Path) -> bool:
        """
        Remap raw forcing data to lumped catchment shapefile.

        Args:
            lumped_shp: Path to lumped catchment shapefile.

        Returns:
            True if successful.
        """
        self.logger.info("Remapping forcing data to lumped catchment")

        try:
            from symfluence.data.preprocessing.forcing_resampler import ForcingResampler
            from symfluence.core.config.models import SymfluenceConfig

            # Build override dict for lumped mode
            overrides = {
                'SUB_GRID_DISCRETIZATION': 'lumped',
                'DOMAIN_DEFINITION_METHOD': 'lumped',
            }

            # Copy lumped shapefile to expected location for ForcingResampler
            # The path resolver looks in shapefiles/catchment/ directly
            expected_shp = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_HRUs_lumped.shp"
            if not expected_shp.exists():
                import shutil
                # Copy all shapefile components
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = lumped_shp.with_suffix(ext)
                    if src.exists():
                        dst = expected_shp.with_suffix(ext)
                        shutil.copy2(src, dst)
                self.logger.info(f"Copied lumped shapefile to: {expected_shp}")

            # Override catchment config to use lumped
            overrides['CATCHMENT_SHP_NAME'] = expected_shp.name

            # Set up lumped output directory
            lumped_forcing_dir = self.project_dir / 'forcing' / 'lumped_basin_averaged'
            lumped_forcing_dir.mkdir(parents=True, exist_ok=True)

            # Use from_file factory method which handles config loading properly
            lumped_config = SymfluenceConfig.from_file(self.config_path, overrides=overrides)

            # Create resampler and remap
            resampler = ForcingResampler(lumped_config, self.logger)

            # Check if raw forcing exists
            raw_forcing_dir = self.project_dir / 'forcing' / 'raw_data'
            if not raw_forcing_dir.exists() or not list(raw_forcing_dir.glob('*.nc')):
                self.logger.error(f"No raw forcing data found in {raw_forcing_dir}")
                return False

            # Run remapping
            self.logger.info("Starting forcing remapping (this may take a while for large datasets)")
            success = resampler.remap_forcing()

            if success:
                self.logger.info("Forcing remapping completed successfully")
            else:
                self.logger.error("Forcing remapping failed")

            return success

        except Exception as e:
            self.logger.error(f"Error remapping forcing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def preprocess_for_lumped(self) -> bool:
        """
        Preprocess domain data for lumped HBV calibration.

        Full workflow:
        1. Create lumped catchment shapefile (dissolve distributed catchments)
        2. Remap raw forcing to lumped catchment
        3. Run HBV preprocessor on lumped forcing
        """
        self.logger.info(f"Preprocessing {self.domain_name} for lumped HBV")
        self.logger.info("=" * 50)

        try:
            from symfluence.models.hbv.preprocessor import HBVPreprocessor

            # Step 1: Create lumped shapefile if needed
            lumped_shp_dir = self.project_dir / 'shapefiles' / 'catchment' / 'lumped'
            lumped_shp = lumped_shp_dir / f"{self.domain_name}_lumped_catchment.shp"

            if not lumped_shp.exists():
                self.logger.info("Step 1: Creating lumped catchment shapefile...")
                try:
                    lumped_shp = self._create_lumped_shapefile()
                except Exception as e:
                    self.logger.error(f"Failed to create lumped shapefile: {e}")
                    return False
            else:
                self.logger.info(f"Step 1: Lumped shapefile already exists: {lumped_shp}")

            # Step 2: Check for forcing data (using area-weighted averaging approach)
            basin_avg = self.project_dir / 'forcing' / 'basin_averaged_data'
            if basin_avg.exists() and list(basin_avg.glob('*.nc')):
                self.logger.info("Step 2: Using existing basin-averaged forcing with area-weighted HRU averaging")
                self.logger.info("  (Area-weighted averaging is mathematically equivalent to geospatial remapping)")
            else:
                raw_forcing = self.project_dir / 'forcing' / 'raw_data'
                if raw_forcing.exists() and list(raw_forcing.glob('*.nc')):
                    self.logger.error(
                        "Raw forcing found but basin-averaged forcing missing. "
                        "Please run forcing acquisition first to create basin-averaged data."
                    )
                else:
                    self.logger.error(
                        "No forcing data found. Please run forcing acquisition first or "
                        "ensure basin_averaged_data/ contains forcing files."
                    )
                return False

            # Step 3: Run HBV preprocessor
            self.logger.info("Step 3: Running HBV preprocessor...")

            # Load fresh config dict (HBVPreprocessor accepts dict or config object)
            config_dict = self._get_fresh_config_dict()
            config_dict['SUB_GRID_DISCRETIZATION'] = 'lumped'

            preprocessor = HBVPreprocessor(config_dict, self.logger)
            success = preprocessor.run_preprocessing()

            if success:
                self.logger.info("HBV lumped preprocessing completed successfully")
            else:
                self.logger.error("HBV lumped preprocessing failed")

            return success

        except Exception as e:
            self.logger.error(f"Error in lumped preprocessing: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def check_data_ready(self) -> bool:
        """Check if forcing data is ready for HBV calibration."""
        forcing_dir = self.project_dir / 'forcing' / 'HBV_input'
        forcing_file = forcing_dir / f"{self.domain_name}_hbv_forcing.nc"
        csv_file = forcing_dir / f"{self.domain_name}_hbv_forcing.csv"

        return forcing_file.exists() or csv_file.exists()

    def run(self, preprocess: bool = True) -> CalibrationStudyResult:
        """
        Run the calibration study with all specified algorithms.

        Args:
            preprocess: Whether to run preprocessing if data not ready

        Returns:
            CalibrationStudyResult with all algorithm results
        """
        self.logger.info("=" * 70)
        self.logger.info(f"HBV CALIBRATION STUDY: {self.domain_name}")
        self.logger.info(f"Algorithms: {', '.join(self.algorithms)}")
        self.logger.info(f"Max evaluations: {self.max_evaluations}")
        self.logger.info(f"Number of runs per algorithm: {self.n_runs}")
        self.logger.info(f"Target KGE: {self.target_kge}")
        if self.quick:
            self.logger.info("(Quick mode enabled)")
        self.logger.info("=" * 70)

        # Check/run preprocessing
        if not self.check_data_ready():
            if preprocess:
                self.logger.info("Forcing data not found, running preprocessing...")
                if not self.preprocess_for_lumped():
                    raise RuntimeError(
                        f"Preprocessing failed for {self.domain_name}. "
                        "Check that basin-averaged forcing data exists."
                    )
            else:
                raise FileNotFoundError(
                    f"HBV forcing data not found for {self.domain_name}. "
                    "Run with preprocess=True or manually preprocess first."
                )

        # Initialize result container
        result = CalibrationStudyResult(
            domain_name=self.domain_name,
            model='HBV',
            timestamp=datetime.now().isoformat(),
            target_kge=self.target_kge,
            max_evaluations=self.max_evaluations,
            n_runs=self.n_runs,
        )

        # Run each algorithm with multiple runs
        total_runs = len(self.algorithms) * self.n_runs
        current_run = 0

        for algorithm in self.algorithms:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running: {algorithm.upper()} ({self.n_runs} runs)")
            self.logger.info(f"{'='*50}")

            algo_results = []

            for run_id in range(self.n_runs):
                current_run += 1
                # Generate unique seed for each run
                run_seed = (self.random_seed + run_id * 1000) if self.random_seed else None

                self.logger.info(f"  Run {run_id + 1}/{self.n_runs} (seed={run_seed}) [{current_run}/{total_runs}]")

                try:
                    alg_result = self._run_algorithm(algorithm, run_id=run_id, seed=run_seed)
                    alg_result.compute_derived_metrics(self.target_kge)
                    result.algorithm_results.append(alg_result)
                    algo_results.append(alg_result)

                    self.logger.info(
                        f"    KGE: {alg_result.final_kge:.4f}, "
                        f"NSE: {alg_result.final_nse:.4f}, "
                        f"Time: {alg_result.runtime_seconds:.1f}s"
                    )

                except Exception as e:
                    self.logger.error(f"  Run {run_id + 1} failed: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())

                    # Add failed result
                    result.algorithm_results.append(AlgorithmResult(
                        algorithm=algorithm,
                        run_id=run_id,
                        seed=run_seed or 0,
                        final_kge=-999.0,
                        final_nse=-999.0,
                        best_params={},
                        n_evaluations=0,
                        runtime_seconds=0.0,
                        error=str(e)
                    ))

            # Report algorithm summary
            valid_results = [r for r in algo_results if r.final_kge > -900]
            if valid_results:
                kge_vals = [r.final_kge for r in valid_results]
                self.logger.info(
                    f"  {algorithm.upper()} Summary: "
                    f"KGE = {np.mean(kge_vals):.4f} ± {np.std(kge_vals):.4f} "
                    f"({len(valid_results)}/{self.n_runs} successful)"
                )

        # Compute statistics and summary
        result.compute_statistics()
        result.compute_summary()

        # Run statistical tests if multiple runs
        if self.n_runs > 1:
            self.logger.info("\nRunning statistical tests...")
            result.run_statistical_tests()

            if result.friedman_pvalue is not None:
                sig = "significant" if result.friedman_pvalue < 0.05 else "not significant"
                self.logger.info(
                    f"  Friedman test: χ² = {result.friedman_statistic:.2f}, "
                    f"p = {result.friedman_pvalue:.4f} ({sig})"
                )

            if result.algorithm_ranks:
                self.logger.info("  Algorithm rankings (lower is better):")
                sorted_ranks = sorted(result.algorithm_ranks.items(), key=lambda x: x[1])
                for algo, rank in sorted_ranks:
                    self.logger.info(f"    {algo}: {rank:.2f}")

        # Save results
        results_path = self.output_dir / f"{self.domain_name}_calibration_study.json"
        result.save(results_path)
        self.logger.info(f"\nResults saved to: {results_path}")

        return result

    def _run_algorithm(self, algorithm: str, run_id: int = 0, seed: Optional[int] = None) -> AlgorithmResult:
        """
        Run a single algorithm and collect results.

        Args:
            algorithm: Algorithm name
            run_id: Run number for multi-run support
            seed: Random seed for this specific run

        Returns:
            AlgorithmResult with all metrics
        """
        from symfluence.models.hbv.calibration.optimizer import HBVModelOptimizer

        # Load fresh config to avoid Pydantic frozen object issues
        config_dict = self._get_fresh_config_dict()

        # Get optimal configuration for this algorithm given the evaluation budget
        alg_config = self.get_algorithm_config(algorithm, self.max_evaluations, self.n_params)
        iterations = alg_config['iterations']
        population_size = alg_config['population_size']
        extra_settings = alg_config.get('extra_settings', {})

        if run_id == 0:  # Only log config once per algorithm
            self.logger.info(
                f"  Config: iterations={iterations}, pop_size={population_size}, "
                f"budget={self.max_evaluations} evals"
            )

        # Set optimization parameters
        if 'optimization' not in config_dict:
            config_dict['optimization'] = {}
        config_dict['optimization']['algorithm'] = algorithm
        config_dict['optimization']['iterations'] = iterations
        config_dict['optimization']['population_size'] = population_size
        config_dict['optimization']['metric'] = 'KGE'

        # Apply any algorithm-specific extra settings
        for key, value in extra_settings.items():
            config_key = f"{algorithm.upper().replace('-', '_').replace(' ', '_')}_{key.upper()}"
            config_dict[config_key] = value
            self.logger.debug(f"  Setting {config_key}={value}")

        # Use run-specific seed
        if seed is not None:
            if 'system' not in config_dict:
                config_dict['system'] = {}
            config_dict['system']['random_seed'] = seed

        # Create optimizer
        optimizer = HBVModelOptimizer(config_dict, self.logger)

        # Run optimization
        start_time = time.time()

        try:
            _results_path = optimizer.run_optimization(algorithm)  # noqa: F841
            runtime = time.time() - start_time

            # Extract results from optimizer
            best_result = optimizer.get_best_result()
            best_params = best_result.get('params', {})
            best_score = best_result.get('score', -999.0)

            # Get iteration history (may not be available for all optimizers)
            history = getattr(optimizer, 'iteration_history', None) or []

            # Calculate final metrics
            final_kge = best_score
            final_nse = best_result.get('nse', -999.0)

            # Initialize additional metrics
            final_lognse = None
            final_pbias = None
            final_rmse = None
            final_mae = None
            final_kge_alpha = best_result.get('alpha', None)
            final_kge_beta = best_result.get('beta', None)
            final_kge_r = best_result.get('r', best_result.get('correlation', None))

            # Get all metrics from final evaluation
            if best_params:
                try:
                    optimizer.worker.apply_parameters(best_params, optimizer.hbv_setup_dir)
                    optimizer.worker.run_model(config_dict, optimizer.hbv_setup_dir, optimizer.results_dir)
                    metrics = optimizer.worker.calculate_metrics(None, config_dict)

                    # Extract all available metrics
                    if final_nse == -999.0:
                        final_nse = metrics.get('nse', -999.0)
                    final_lognse = metrics.get('lognse', metrics.get('log_nse', None))
                    final_pbias = metrics.get('pbias', metrics.get('percent_bias', None))
                    final_rmse = metrics.get('rmse', None)
                    final_mae = metrics.get('mae', None)

                    # KGE components if not already set
                    if final_kge_alpha is None:
                        final_kge_alpha = metrics.get('alpha', metrics.get('kge_alpha', None))
                    if final_kge_beta is None:
                        final_kge_beta = metrics.get('beta', metrics.get('kge_beta', None))
                    if final_kge_r is None:
                        final_kge_r = metrics.get('r', metrics.get('correlation', None))

                except Exception as e:
                    self.logger.debug(f"Could not compute additional metrics: {e}")

            # Estimate number of evaluations from history or config
            if history:
                n_evals = sum(h.get('n_evals', population_size) for h in history)
            else:
                # Estimate based on algorithm type
                alg_lower = algorithm.lower().replace('-', '_')
                if alg_lower == 'dds':
                    n_evals = iterations + 1
                elif alg_lower in ['adam', 'nelder_mead', 'simulated_annealing', 'sa']:
                    n_evals = iterations * max(1, population_size)
                else:
                    n_evals = (iterations + 1) * population_size

            return AlgorithmResult(
                algorithm=algorithm,
                run_id=run_id,
                seed=seed or 0,
                final_kge=final_kge,
                final_nse=final_nse,
                final_lognse=final_lognse,
                final_pbias=final_pbias,
                final_rmse=final_rmse,
                final_mae=final_mae,
                final_kge_alpha=final_kge_alpha,
                final_kge_beta=final_kge_beta,
                final_kge_r=final_kge_r,
                best_params=best_params,
                n_evaluations=n_evals,
                runtime_seconds=runtime,
                convergence_history=[
                    {'iteration': h.get('generation', i),
                     'best_score': h.get('best_score', -999),
                     'n_evals': h.get('n_evals', population_size)}
                    for i, h in enumerate(history)
                ] if history else []
            )

        except Exception as e:
            runtime = time.time() - start_time
            self.logger.error(f"Error running {algorithm}: {e}")
            raise

    def print_summary(self, result: CalibrationStudyResult) -> None:
        """Print a summary table of results with multi-run statistics."""
        print("\n" + "=" * 100)
        print(f"CALIBRATION STUDY SUMMARY: {result.domain_name}")
        print(f"Runs per algorithm: {result.n_runs}")
        print("=" * 100)

        # Use multi-run statistics if available
        if result.algorithm_statistics:
            # Header for multi-run summary
            print(f"{'Algorithm':<15} {'KGE (mean±std)':>18} {'NSE (mean±std)':>18} "
                  f"{'95% CI':>16} {'Rank':>6} {'Time(s)':>10}")
            print("-" * 100)

            # Sort by mean KGE descending
            sorted_stats = sorted(
                result.algorithm_statistics.items(),
                key=lambda x: x[1].kge_mean,
                reverse=True
            )

            for algo, stats in sorted_stats:
                kge_str = f"{stats.kge_mean:.4f}±{stats.kge_std:.4f}"
                nse_str = f"{stats.nse_mean:.4f}±{stats.nse_std:.4f}"
                ci_str = f"[{stats.kge_ci_lower:.3f},{stats.kge_ci_upper:.3f}]"
                rank_str = f"{stats.mean_rank:.2f}" if stats.mean_rank else "N/A"
                time_str = f"{stats.runtime_mean:.1f}±{stats.runtime_std:.1f}"

                print(f"{algo:<15} {kge_str:>18} {nse_str:>18} "
                      f"{ci_str:>16} {rank_str:>6} {time_str:>10}")

            # Statistical significance
            print("-" * 100)
            if result.friedman_pvalue is not None:
                sig = "***" if result.friedman_pvalue < 0.001 else (
                    "**" if result.friedman_pvalue < 0.01 else (
                        "*" if result.friedman_pvalue < 0.05 else "n.s."
                    )
                )
                print(f"Friedman test: χ² = {result.friedman_statistic:.2f}, "
                      f"p = {result.friedman_pvalue:.4f} ({sig})")

        else:
            # Fallback to single-run display
            print(f"{'Algorithm':<18} {'KGE':>8} {'NSE':>8} {'Evals':>8} "
                  f"{'Time(s)':>10} {'Evals/s':>10} {'To Target':>10}")
            print("-" * 100)

            sorted_results = sorted(
                [r for r in result.algorithm_results if r.final_kge > -900],
                key=lambda r: r.final_kge,
                reverse=True
            )

            for r in sorted_results:
                evals_s = f"{r.evals_per_second:.1f}" if r.evals_per_second else "N/A"
                to_target = str(r.evals_to_target) if r.evals_to_target else "N/A"

                print(f"{r.algorithm:<18} {r.final_kge:>8.4f} {r.final_nse:>8.4f} "
                      f"{r.n_evaluations:>8} {r.runtime_seconds:>10.1f} "
                      f"{evals_s:>10} {to_target:>10}")

        # Show failed algorithms
        failed = [r for r in result.algorithm_results if r.final_kge <= -900]
        if failed:
            # Group failures by algorithm
            failed_algos = {}
            for r in failed:
                if r.algorithm not in failed_algos:
                    failed_algos[r.algorithm] = []
                failed_algos[r.algorithm].append(r.error or 'Unknown error')

            print("-" * 100)
            print("Failed runs:")
            for algo, errors in failed_algos.items():
                unique_errors = list(set(errors))
                print(f"  {algo}: {len(errors)} failures - {unique_errors[0][:50]}...")

        print("-" * 100)
        if result.best_algorithm:
            print(f"Best Algorithm: {result.best_algorithm} (KGE = {result.best_kge:.4f})")
        if result.fastest_to_target:
            print(f"Fastest to Target (KGE >= {result.target_kge}): {result.fastest_to_target}")
        print("=" * 100)

    def generate_plots(self, result: CalibrationStudyResult, dpi: int = 300) -> Path:
        """
        Generate publication-quality comparison plots.

        Args:
            result: CalibrationStudyResult with all data
            dpi: Resolution for raster elements (default 300 for publication)

        Returns:
            Path to plots directory
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch  # noqa: F401 - used in legend creation
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")
            return self.output_dir

        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Publication-quality settings
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.family': 'sans-serif',
        })

        # Check if we have multi-run statistics
        has_stats = bool(result.algorithm_statistics)

        if has_stats:
            self._generate_multirun_plots(result, plots_dir, dpi)
        else:
            self._generate_single_run_plots(result, plots_dir, dpi)

        # Generate convergence plots (works for both)
        self._generate_convergence_plots(result, plots_dir, dpi)

        self.logger.info(f"Plots saved to: {plots_dir}")
        return plots_dir

    def _generate_multirun_plots(self, result: CalibrationStudyResult, plots_dir: Path, dpi: int) -> None:
        """Generate plots for multi-run results with error bars."""
        import matplotlib.pyplot as plt

        stats = result.algorithm_statistics
        algorithms = sorted(stats.keys(), key=lambda x: stats[x].kge_mean, reverse=True)
        n_algos = len(algorithms)

        # Color palette (colorblind-friendly)
        colors = plt.cm.tab10(np.linspace(0, 1, n_algos))
        algo_colors = {algo: colors[i] for i, algo in enumerate(algorithms)}

        # Figure 1: Performance comparison with error bars (main figure)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # 1a. KGE boxplot
        ax1 = axes[0, 0]
        kge_data = []
        for algo in algorithms:
            kge_vals = [r.final_kge for r in stats[algo].individual_results if r.final_kge > -900]
            kge_data.append(kge_vals)

        bp = ax1.boxplot(kge_data, labels=[a.upper() for a in algorithms], patch_artist=True)
        for patch, algo in zip(bp['boxes'], algorithms):
            patch.set_facecolor(algo_colors[algo])
            patch.set_alpha(0.7)
        ax1.axhline(y=result.target_kge, color='green', linestyle='--', alpha=0.7, label=f'Target={result.target_kge}')
        ax1.set_ylabel('KGE')
        ax1.set_title('(a) KGE Distribution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 1b. NSE boxplot
        ax2 = axes[0, 1]
        nse_data = []
        for algo in algorithms:
            nse_vals = [r.final_nse for r in stats[algo].individual_results if r.final_nse > -900]
            nse_data.append(nse_vals)

        bp2 = ax2.boxplot(nse_data, labels=[a.upper() for a in algorithms], patch_artist=True)
        for patch, algo in zip(bp2['boxes'], algorithms):
            patch.set_facecolor(algo_colors[algo])
            patch.set_alpha(0.7)
        ax2.set_ylabel('NSE')
        ax2.set_title('(b) NSE Distribution')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 1c. Mean KGE with 95% CI error bars
        ax3 = axes[1, 0]
        x = np.arange(n_algos)
        means = [stats[a].kge_mean for a in algorithms]
        ci_lower = [stats[a].kge_mean - stats[a].kge_ci_lower for a in algorithms]
        ci_upper = [stats[a].kge_ci_upper - stats[a].kge_mean for a in algorithms]

        ax3.bar(x, means, yerr=[ci_lower, ci_upper], capsize=4,
                color=[algo_colors[a] for a in algorithms], alpha=0.8,
                error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        ax3.axhline(y=result.target_kge, color='green', linestyle='--', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([a.upper() for a in algorithms], rotation=45, ha='right')
        ax3.set_ylabel('KGE (mean ± 95% CI)')
        ax3.set_title('(c) Mean Performance with Confidence Intervals')
        ax3.grid(True, alpha=0.3, axis='y')

        # 1d. Algorithm rankings
        ax4 = axes[1, 1]
        if result.algorithm_ranks:
            ranks = [result.algorithm_ranks.get(a, n_algos) for a in algorithms]
            _bars = ax4.barh(x, ranks, color=[algo_colors[a] for a in algorithms], alpha=0.8)  # noqa: F841
            ax4.set_yticks(x)
            ax4.set_yticklabels([a.upper() for a in algorithms])
            ax4.set_xlabel('Mean Rank (lower is better)')
            ax4.set_title('(d) Algorithm Rankings')
            ax4.invert_xaxis()  # Lower rank on right
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'Rankings require\nmultiple runs',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('(d) Algorithm Rankings')

        plt.tight_layout()
        plt.savefig(plots_dir / 'algorithm_comparison.pdf', format='pdf', dpi=dpi)
        plt.savefig(plots_dir / 'algorithm_comparison.png', format='png', dpi=dpi)
        plt.close()

        # Figure 2: Additional metrics (if available)
        has_additional = any(
            s.lognse_mean is not None or s.pbias_mean is not None
            for s in stats.values()
        )

        if has_additional:
            fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

            # logNSE
            ax = axes2[0]
            lognse_data = []
            valid_algos = []
            for algo in algorithms:
                vals = [r.final_lognse for r in stats[algo].individual_results
                       if r.final_lognse is not None]
                if vals:
                    lognse_data.append(vals)
                    valid_algos.append(algo)
            if lognse_data:
                bp = ax.boxplot(lognse_data, labels=[a.upper() for a in valid_algos], patch_artist=True)
                for patch, algo in zip(bp['boxes'], valid_algos):
                    patch.set_facecolor(algo_colors[algo])
                    patch.set_alpha(0.7)
            ax.set_ylabel('logNSE')
            ax.set_title('(a) Log-transformed NSE\n(low flow performance)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # PBIAS
            ax = axes2[1]
            pbias_data = []
            valid_algos = []
            for algo in algorithms:
                vals = [r.final_pbias for r in stats[algo].individual_results
                       if r.final_pbias is not None]
                if vals:
                    pbias_data.append(vals)
                    valid_algos.append(algo)
            if pbias_data:
                bp = ax.boxplot(pbias_data, labels=[a.upper() for a in valid_algos], patch_artist=True)
                for patch, algo in zip(bp['boxes'], valid_algos):
                    patch.set_facecolor(algo_colors[algo])
                    patch.set_alpha(0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('PBIAS (%)')
            ax.set_title('(b) Percent Bias')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # Runtime
            ax = axes2[2]
            runtime_means = [stats[a].runtime_mean for a in algorithms]
            runtime_stds = [stats[a].runtime_std for a in algorithms]
            ax.bar(x, runtime_means, yerr=runtime_stds, capsize=3,
                   color=[algo_colors[a] for a in algorithms], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([a.upper() for a in algorithms], rotation=45, ha='right')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title('(c) Computational Cost')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(plots_dir / 'additional_metrics.pdf', format='pdf', dpi=dpi)
            plt.savefig(plots_dir / 'additional_metrics.png', format='png', dpi=dpi)
            plt.close()

    def _generate_single_run_plots(self, result: CalibrationStudyResult, plots_dir: Path, dpi: int) -> None:
        """Generate plots for single-run results (backward compatibility)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        valid_results = [r for r in result.algorithm_results if r.final_kge > -900]
        if not valid_results:
            self.logger.warning("No valid results to plot")
            return

        fig = plt.figure(figsize=(12, 10))

        algorithms = [r.algorithm for r in valid_results]
        kge_vals = [r.final_kge for r in valid_results]
        nse_vals = [r.final_nse for r in valid_results]
        runtimes = [r.runtime_seconds for r in valid_results]

        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

        # 1. Performance comparison
        ax1 = fig.add_subplot(2, 2, 1)
        x = np.arange(len(algorithms))
        width = 0.35
        ax1.bar(x - width/2, kge_vals, width, label='KGE', color='steelblue')
        ax1.bar(x + width/2, nse_vals, width, label='NSE', color='darkorange')
        ax1.set_ylabel('Score')
        ax1.set_title('(a) Final Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a.upper() for a in algorithms], rotation=45, ha='right')
        ax1.legend()
        ax1.axhline(y=result.target_kge, color='green', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Runtime comparison
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.bar(x, runtimes, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels([a.upper() for a in algorithms], rotation=45, ha='right')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('(b) Computational Cost')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. AUCC (Area Under Convergence Curve)
        ax3 = fig.add_subplot(2, 2, 3)
        aucc_vals = [r.area_under_curve if r.area_under_curve else 0 for r in valid_results]
        ax3.bar(x, aucc_vals, color=colors)
        ax3.set_xticks(x)
        ax3.set_xticklabels([a.upper() for a in algorithms], rotation=45, ha='right')
        ax3.set_ylabel('AUCC')
        ax3.set_title('(c) Area Under Convergence Curve')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Evaluations to target
        ax4 = fig.add_subplot(2, 2, 4)
        evals_to_target = [r.evals_to_target if r.evals_to_target else 0 for r in valid_results]
        reached_target = [r.evals_to_target is not None for r in valid_results]
        bar_colors = ['forestgreen' if reached else 'lightgray' for reached in reached_target]
        ax4.bar(x, evals_to_target, color=bar_colors)
        ax4.set_xticks(x)
        ax4.set_xticklabels([a.upper() for a in algorithms], rotation=45, ha='right')
        ax4.set_ylabel(f'Evaluations to KGE ≥ {result.target_kge}')
        ax4.set_title('(d) Convergence Speed')
        ax4.grid(True, alpha=0.3, axis='y')

        legend_elements = [Patch(facecolor='forestgreen', label='Reached Target'),
                         Patch(facecolor='lightgray', label='Did Not Reach')]
        ax4.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(plots_dir / 'algorithm_comparison.pdf', format='pdf', dpi=dpi)
        plt.savefig(plots_dir / 'algorithm_comparison.png', format='png', dpi=dpi)
        plt.close()

    def _generate_convergence_plots(self, result: CalibrationStudyResult, plots_dir: Path, dpi: int) -> None:
        """Generate convergence curves using cumulative evaluations on x-axis."""
        import matplotlib.pyplot as plt

        # Collect all results with convergence history
        results_with_history = [r for r in result.algorithm_results
                               if r.convergence_history and r.final_kge > -900]

        if not results_with_history:
            return

        # Group by algorithm for multi-run
        algo_histories = {}
        for r in results_with_history:
            if r.algorithm not in algo_histories:
                algo_histories[r.algorithm] = []
            algo_histories[r.algorithm].append(r)

        algorithms = sorted(algo_histories.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        algo_colors = {algo: colors[i] for i, algo in enumerate(algorithms)}

        # Figure: Convergence curves with cumulative evaluations
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo in algorithms:
            runs = algo_histories[algo]

            if len(runs) == 1:
                # Single run - simple line
                r = runs[0]
                cum_evals = [h.get('cumulative_evals', i+1) for i, h in enumerate(r.convergence_history)]
                scores = [h.get('best_score', -999) for h in r.convergence_history]
                ax.plot(cum_evals, scores, label=f'{algo.upper()} (KGE={r.final_kge:.3f})',
                       color=algo_colors[algo], linewidth=1.5)
            else:
                # Multiple runs - plot mean with shaded std
                # Interpolate to common x-axis
                max_evals = max(
                    r.convergence_history[-1].get('cumulative_evals', len(r.convergence_history))
                    for r in runs if r.convergence_history
                )
                common_x = np.linspace(0, max_evals, 100)

                all_curves = []
                for r in runs:
                    if not r.convergence_history:
                        continue
                    cum_evals = [h.get('cumulative_evals', i+1) for i, h in enumerate(r.convergence_history)]
                    scores = [h.get('best_score', -999) for h in r.convergence_history]

                    if len(cum_evals) > 1:
                        # Interpolate to common x-axis
                        interp_scores = np.interp(common_x, cum_evals, scores)
                        all_curves.append(interp_scores)

                if all_curves:
                    all_curves = np.array(all_curves)
                    mean_curve = np.mean(all_curves, axis=0)
                    std_curve = np.std(all_curves, axis=0)

                    # Get mean final KGE
                    mean_kge = np.mean([r.final_kge for r in runs if r.final_kge > -900])

                    ax.plot(common_x, mean_curve,
                           label=f'{algo.upper()} (KGE={mean_kge:.3f}±{std_curve[-1]:.3f})',
                           color=algo_colors[algo], linewidth=1.5)
                    ax.fill_between(common_x, mean_curve - std_curve, mean_curve + std_curve,
                                   alpha=0.2, color=algo_colors[algo])

        ax.axhline(y=result.target_kge, color='green', linestyle='--', alpha=0.7,
                  label=f'Target KGE = {result.target_kge}')
        ax.set_xlabel('Cumulative Function Evaluations')
        ax.set_ylabel('Best KGE')
        ax.set_title('Convergence Curves')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        plt.savefig(plots_dir / 'convergence_curves.pdf', format='pdf', dpi=dpi)
        plt.savefig(plots_dir / 'convergence_curves.png', format='png', dpi=dpi)
        plt.close()

    def save_latex_tables(self, result: CalibrationStudyResult) -> Path:
        """
        Save LaTeX tables for publication.

        Args:
            result: CalibrationStudyResult with statistics

        Returns:
            Path to tables directory
        """
        tables_dir = self.output_dir / 'tables'
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Main results table
        latex_main = result.to_latex_table(
            metrics=['kge', 'nse', 'lognse', 'pbias'],
            caption=f'Optimization algorithm comparison for {result.domain_name} '
                    f'({result.n_runs} runs per algorithm, {result.max_evaluations} evaluations)'
        )
        with open(tables_dir / 'results_table.tex', 'w') as f:
            f.write(latex_main)

        # Pairwise significance table
        if result.pairwise_tests:
            latex_pairwise = result.to_latex_pairwise_table()
            with open(tables_dir / 'pairwise_tests.tex', 'w') as f:
                f.write(latex_pairwise)

        self.logger.info(f"LaTeX tables saved to: {tables_dir}")
        return tables_dir


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def find_config_files(config_dir: Path, pattern: str = "config_*.yaml") -> List[Path]:
    """Find all configuration files in a directory."""
    config_dir = Path(config_dir)
    configs = list(config_dir.glob(pattern))

    # Filter out template files
    configs = [c for c in configs if 'template' not in c.name.lower()]

    return sorted(configs)


def run_single_basin(
    config_path: Path,
    algorithms: List[str],
    max_evaluations: int,
    n_runs: int,
    target_kge: float,
    output_base_dir: Path,
    quick: bool = False,
    n_params: int = 10
) -> Optional[CalibrationStudyResult]:
    """Run calibration study for a single basin (used for batch processing)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(f'study_{config_path.stem}')

    try:
        study = HBVCalibrationStudy(
            config_path=config_path,
            algorithms=algorithms,
            max_evaluations=max_evaluations,
            n_runs=n_runs,
            target_kge=target_kge,
            output_dir=output_base_dir / config_path.stem.replace('config_', ''),
            quick=quick,
            logger=logger,
            n_params=n_params,
        )
        return study.run()
    except Exception as e:
        logger.error(f"Failed to run study for {config_path}: {e}")
        return None


def run_batch(
    config_dir: Path,
    algorithms: List[str],
    max_evaluations: int,
    n_runs: int,
    target_kge: float,
    output_dir: Path,
    quick: bool = False,
    max_workers: int = 1,
    n_params: int = 10
) -> List[CalibrationStudyResult]:
    """Run calibration studies for all basins in parallel."""
    config_files = find_config_files(config_dir)
    print(f"Found {len(config_files)} configuration files")

    if not config_files:
        print(f"No config files found in {config_dir}")
        return []

    results = []

    if max_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_basin,
                    cfg, algorithms, max_evaluations, n_runs, target_kge, output_dir, quick, n_params
                ): cfg for cfg in config_files
            }

            for future in as_completed(futures):
                cfg = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"Completed: {cfg.stem} - Best KGE: {result.best_kge:.4f}")
                except Exception as e:
                    print(f"Failed: {cfg.stem} - {e}")
    else:
        # Sequential execution
        for cfg in config_files:
            result = run_single_basin(
                cfg, algorithms, max_evaluations, n_runs, target_kge, output_dir, quick, n_params
            )
            if result:
                results.append(result)
                print(f"Completed: {cfg.stem} - Best KGE: {result.best_kge:.4f}")

    # Generate combined summary
    if results:
        all_results_df = pd.concat([r.to_dataframe() for r in results], ignore_index=True)
        all_results_df.to_csv(output_dir / 'all_basins_summary.csv', index=False)
        print(f"\nCombined results saved to: {output_dir / 'all_basins_summary.csv'}")

        # Also save summary statistics
        summary_dfs = [r.to_summary_dataframe() for r in results if r.algorithm_statistics]
        if summary_dfs:
            summary_df = pd.concat(summary_dfs, ignore_index=True)
            summary_df.to_csv(output_dir / 'all_basins_statistics.csv', index=False)

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HBV Calibration Study - Multi-Algorithm Comparison (Publication-Ready)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single basin with 1000 evaluations, 10 replicate runs (publication-ready)
    python hbv_calibration_study.py --config config.yaml --max-eval 1000 --n-runs 10

    # Quick test with fewer evaluations (single run)
    python hbv_calibration_study.py --config config.yaml --quick

    # Specific algorithms with 2000 evaluations, 20 runs
    python hbv_calibration_study.py --config config.yaml \\
        --algorithms dds pso de adam cmaes --max-eval 2000 --n-runs 20

    # Batch processing (all basins)
    python hbv_calibration_study.py --batch --config-dir 0_config_files/ --max-eval 1000 --n-runs 10

    # Full publication run with LaTeX output
    python hbv_calibration_study.py --config config.yaml --max-eval 2000 --n-runs 30

Output includes:
- Publication-quality figures (PDF at 300 dpi)
- LaTeX tables with mean ± std and 95% CI
- Statistical tests (Friedman, pairwise Wilcoxon)
- Algorithm rankings

Note: The script automatically computes optimal iterations and population_size
for each algorithm based on max_eval, ensuring fair comparison across algorithms
with different evaluation patterns.
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to basin configuration YAML file'
    )
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Run batch processing for all basins'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('0_config_files'),
        help='Directory containing config files (for batch mode)'
    )
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        default=None,
        help='Algorithms to compare (default: dds pso de sce-ua cmaes adam nelder_mead ga simulated_annealing)'
    )
    parser.add_argument(
        '--max-eval', '-e',
        type=int,
        default=1000,
        help='Maximum function evaluations per algorithm (default: 1000). '
             'Optimal iterations/population computed per algorithm for fair comparison.'
    )
    parser.add_argument(
        '--n-runs', '-r',
        type=int,
        default=1,
        help='Number of replicate runs per algorithm (default: 1). '
             'Use 10-30 runs for publication-quality statistical analysis.'
    )
    parser.add_argument(
        '--n-params',
        type=int,
        default=10,
        help='Number of parameters being optimized (default: 10). '
             'Used to compute optimal algorithm configurations.'
    )
    parser.add_argument(
        '--target-kge', '-t',
        type=float,
        default=0.7,
        help='Target KGE for convergence metrics (default: 0.7)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: fewer iterations and algorithms for testing'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of parallel workers for batch mode (default: 1)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('hbv_calibration_study')

    # Validate arguments
    if not args.batch and not args.config:
        parser.error("Either --config or --batch must be specified")

    if args.batch:
        # Batch mode
        output_dir = args.output_dir or Path('results/hbv_calibration_studies')
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nBatch mode: Processing all basins in {args.config_dir}")
        print(f"Runs per algorithm: {args.n_runs}")
        results = run_batch(
            config_dir=args.config_dir,
            algorithms=args.algorithms or HBVCalibrationStudy.DEFAULT_ALGORITHMS,
            max_evaluations=args.max_eval,
            n_runs=args.n_runs,
            target_kge=args.target_kge,
            output_dir=output_dir,
            quick=args.quick,
            max_workers=args.workers,
            n_params=args.n_params
        )
        print(f"\nCompleted {len(results)} basins")

    else:
        # Single basin mode
        study = HBVCalibrationStudy(
            config_path=args.config,
            algorithms=args.algorithms,
            max_evaluations=args.max_eval,
            n_runs=args.n_runs,
            target_kge=args.target_kge,
            output_dir=args.output_dir,
            quick=args.quick,
            logger=logger,
            n_params=args.n_params,
        )

        result = study.run()
        study.print_summary(result)

        if not args.no_plots:
            study.generate_plots(result)

        # Save LaTeX tables if multi-run
        if args.n_runs > 1:
            study.save_latex_tables(result)

        # Save CSV summaries
        df = result.to_dataframe()
        csv_path = study.output_dir / 'results_all_runs.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nAll runs CSV saved to: {csv_path}")

        if result.algorithm_statistics:
            summary_df = result.to_summary_dataframe()
            summary_path = study.output_dir / 'results_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary statistics saved to: {summary_path}")


if __name__ == '__main__':
    main()
