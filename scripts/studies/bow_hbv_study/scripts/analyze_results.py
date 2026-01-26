#!/usr/bin/env python3
"""
Analyze and visualize results from the Bow HBV study.

This script generates comprehensive plots and reports comparing:
1. Daily vs Hourly performance
2. Optimization algorithm comparison
3. Differentiability analysis results
4. Gradient method comparisons

Usage:
    python analyze_results.py
    python analyze_results.py --output-dir ../results
    python analyze_results.py --format html  # Generate interactive HTML report
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150

# Paths
STUDY_DIR = Path(__file__).parent.parent
RESULTS_BASE = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/simulations")
OUTPUT_DIR = STUDY_DIR / "results" / "plots"


class StudyAnalyzer:
    """Analyzer for Bow HBV study results."""

    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = self._load_experiments()
        print(f"Found {len(self.experiments)} experiments")

    def _load_experiments(self) -> Dict[str, Dict]:
        """Load all experiment results."""
        experiments: Dict[str, Dict[str, Any]] = {}

        # Look for experiment directories
        if not self.results_dir.exists():
            print(f"WARNING: Results directory not found: {self.results_dir}")
            return experiments

        for exp_dir in self.results_dir.glob("study_*"):
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name
            exp_data = self._load_experiment(exp_dir)
            if exp_data:
                experiments[exp_name] = exp_data

        return experiments

    def _load_experiment(self, exp_dir: Path) -> Optional[Dict]:
        """Load data for a single experiment."""
        try:
            # Load calibration history
            calib_file = exp_dir / "optimization" / "calibration_history.csv"
            if not calib_file.exists():
                # Try alternate location
                calib_file = exp_dir / "calibration_history.csv"

            if not calib_file.exists():
                print(f"WARNING: Calibration history not found for {exp_dir.name}")
                return None

            history = pd.read_csv(calib_file)

            # Load final results
            results_file = exp_dir / "optimization" / "best_parameters.csv"
            if not results_file.exists():
                results_file = exp_dir / "best_parameters.csv"

            final_results = None
            if results_file.exists():
                final_results = pd.read_csv(results_file)

            # Load observed vs simulated
            sim_file = exp_dir / "model_output" / "streamflow.csv"
            obs_sim_data = None
            if sim_file.exists():
                obs_sim_data = pd.read_csv(sim_file)

            return {
                'dir': exp_dir,
                'name': exp_dir.name,
                'history': history,
                'final_results': final_results,
                'obs_sim': obs_sim_data,
            }
        except Exception as e:
            print(f"ERROR loading {exp_dir.name}: {e}")
            return None

    def plot_convergence_comparison(self, part: str = 'all', save: bool = True):
        """Plot convergence curves for comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Filter experiments by part
        if part == '1':
            # Daily vs Hourly
            experiments = {k: v for k, v in self.experiments.items()
                         if 'daily_dds' in k or 'hourly_dds' in k}
            title = "Convergence: Daily vs Hourly"
        elif part == '2':
            # Optimizer comparison
            experiments = {k: v for k, v in self.experiments.items()
                         if any(opt in k for opt in ['dds', 'pso', 'de', 'ga', 'adam'])
                         and 'daily' in k and 'smooth' not in k}
            title = "Convergence: Optimization Algorithms"
        elif part == '3':
            # Differentiability
            experiments = {k: v for k, v in self.experiments.items()
                         if 'smooth' in k or ('dds' in k and 'daily' in k)}
            title = "Convergence: Smoothing Comparison"
        else:
            experiments = self.experiments
            title = "Convergence: All Experiments"

        # Plot each experiment
        for exp_name, exp_data in experiments.items():
            history = exp_data['history']
            if 'iteration' in history.columns and 'kge' in history.columns:
                label = self._format_label(exp_name)
                ax.plot(history['iteration'], history['kge'],
                       label=label, linewidth=2, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('KGE')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            filename = f"convergence_part_{part}.png"
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()

        plt.close()

    def plot_performance_comparison(self, metric: str = 'kge', save: bool = True):
        """Bar plot comparing final performance."""
        # Extract final performance
        results = []
        for exp_name, exp_data in self.experiments.items():
            history = exp_data['history']
            if metric in history.columns:
                final_value = history[metric].iloc[-1]
                results.append({
                    'experiment': self._format_label(exp_name),
                    'metric': final_value
                })

        if not results:
            print(f"WARNING: No results found for metric {metric}")
            return

        df = pd.DataFrame(results)
        df = df.sort_values('metric', ascending=False)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(df['experiment'], df['metric'], color='steelblue', alpha=0.8)

        # Color code by performance
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=df['metric'].min(), vmax=df['metric'].max())
        for bar, val in zip(bars, df['metric']):
            bar.set_color(cmap(norm(val)))

        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Experiment')
        ax.set_title(f'Final {metric.upper()} Comparison')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        if save:
            filename = f"performance_{metric}.png"
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()

        plt.close()

    def plot_hydrograph_comparison(self, experiments: List[str], save: bool = True):
        """Plot observed vs simulated hydrographs."""
        n_exp = len(experiments)
        fig, axes = plt.subplots(n_exp, 1, figsize=(14, 4*n_exp), sharex=True)
        if n_exp == 1:
            axes = [axes]

        for ax, exp_name in zip(axes, experiments):
            if exp_name not in self.experiments:
                continue

            exp_data = self.experiments[exp_name]
            obs_sim = exp_data.get('obs_sim')

            if obs_sim is None or 'date' not in obs_sim.columns:
                continue

            # Plot
            if 'observed' in obs_sim.columns:
                ax.plot(obs_sim['date'], obs_sim['observed'],
                       label='Observed', color='black', linewidth=1, alpha=0.7)
            if 'simulated' in obs_sim.columns:
                ax.plot(obs_sim['date'], obs_sim['simulated'],
                       label='Simulated', color='steelblue', linewidth=1, alpha=0.7)

            # Format
            ax.set_ylabel('Streamflow (m³/s)')
            ax.set_title(self._format_label(exp_name))
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()

        if save:
            filename = "hydrographs_comparison.png"
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()

        plt.close()

    def plot_scatter_comparison(self, experiments: List[str], save: bool = True):
        """Scatter plots of observed vs simulated."""
        n_exp = len(experiments)
        n_cols = min(3, n_exp)
        n_rows = (n_exp + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = np.atleast_2d(axes).flatten()

        for ax, exp_name in zip(axes, experiments):
            if exp_name not in self.experiments:
                ax.axis('off')
                continue

            exp_data = self.experiments[exp_name]
            obs_sim = exp_data.get('obs_sim')

            if obs_sim is None or 'observed' not in obs_sim.columns:
                ax.axis('off')
                continue

            obs = obs_sim['observed'].values
            sim = obs_sim['simulated'].values

            # Remove NaN
            mask = ~(np.isnan(obs) | np.isnan(sim))
            obs = obs[mask]
            sim = sim[mask]

            # Scatter plot
            ax.scatter(obs, sim, alpha=0.3, s=10, color='steelblue')

            # 1:1 line
            max_val = max(obs.max(), sim.max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 line')

            # Statistics
            from scipy.stats import pearsonr
            r, _ = pearsonr(obs, sim)
            rmse = np.sqrt(np.mean((obs - sim)**2))

            ax.text(0.05, 0.95, f'r = {r:.3f}\nRMSE = {rmse:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Observed (m³/s)')
            ax.set_ylabel('Simulated (m³/s)')
            ax.set_title(self._format_label(exp_name))
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused axes
        for ax in axes[len(experiments):]:
            ax.axis('off')

        plt.tight_layout()

        if save:
            filename = "scatter_comparison.png"
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        else:
            plt.show()

        plt.close()

    def generate_summary_table(self, save: bool = True) -> pd.DataFrame:
        """Generate summary statistics table."""
        summary = []

        for exp_name, exp_data in self.experiments.items():
            history = exp_data['history']

            # Extract final metrics
            row = {'Experiment': self._format_label(exp_name)}

            for metric in ['kge', 'nse', 'rmse', 'bias']:
                if metric in history.columns:
                    row[metric.upper()] = history[metric].iloc[-1]

            # Convergence info
            if 'iteration' in history.columns:
                row['Iterations'] = len(history)

            summary.append(row)

        df = pd.DataFrame(summary)
        df = df.sort_values('KGE', ascending=False)

        if save:
            filename = "summary_table.csv"
            df.to_csv(self.output_dir / filename, index=False)
            print(f"Saved: {filename}")

        return df

    def _format_label(self, exp_name: str) -> str:
        """Format experiment name for display."""
        # Remove 'study_' prefix
        name = exp_name.replace('study_', '')

        # Replace underscores with spaces
        name = name.replace('_', ' ')

        # Capitalize
        name = name.title()

        return name

    def generate_html_report(self, save_path: Optional[Path] = None):
        """Generate interactive HTML report."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("WARNING: plotly not installed. Install with: pip install plotly")
            return

        if save_path is None:
            save_path = self.output_dir / "summary_report.html"

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Convergence Curves', 'Final Performance',
                          'Metric Distribution', 'Efficiency Analysis'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )

        # Add traces (simplified example)
        for exp_name, exp_data in self.experiments.items():
            history = exp_data['history']
            if 'iteration' in history.columns and 'kge' in history.columns:
                fig.add_trace(
                    go.Scatter(x=history['iteration'], y=history['kge'],
                             name=self._format_label(exp_name), mode='lines'),
                    row=1, col=1
                )

        # Update layout
        fig.update_layout(height=900, showlegend=True, title_text="Bow HBV Study Results")
        fig.write_html(str(save_path))
        print(f"Saved HTML report: {save_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze Bow HBV study results')
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=RESULTS_BASE,
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Directory for output plots'
    )
    parser.add_argument(
        '--format',
        choices=['png', 'html', 'both'],
        default='both',
        help='Output format'
    )
    parser.add_argument(
        '--part',
        type=str,
        default='all',
        help='Study part to analyze (1, 2, 3, or all)'
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = StudyAnalyzer(args.results_dir, args.output_dir)

    if len(analyzer.experiments) == 0:
        print("No experiments found. Make sure to run the study first.")
        return

    # Generate plots
    print("\nGenerating plots...")
    print("-" * 60)

    if args.format in ['png', 'both']:
        # Convergence plots
        for part in ['1', '2', '3', 'all']:
            analyzer.plot_convergence_comparison(part=part)

        # Performance comparison
        for metric in ['kge', 'nse']:
            analyzer.plot_performance_comparison(metric=metric)

        # Hydrographs (select representative experiments)
        representative = list(analyzer.experiments.keys())[:4]
        if representative:
            analyzer.plot_hydrograph_comparison(representative)
            analyzer.plot_scatter_comparison(representative)

        # Summary table
        summary_df = analyzer.generate_summary_table()
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))

    if args.format in ['html', 'both']:
        analyzer.generate_html_report()

    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
