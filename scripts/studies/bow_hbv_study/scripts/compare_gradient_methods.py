#!/usr/bin/env python3
"""
Comprehensive gradient method comparison for HBV model.

Compares three gradient computation approaches:
1. Direct AD (JAX): Native automatic differentiation through discrete time-stepping
2. ODE Adjoint: Continuous-time formulation with adjoint sensitivity analysis
3. Finite Differences: Numerical gradient approximation (reference)

This script provides a focused analysis on differentiability and gradient accuracy
for the Bow at Banff HBV setup.

Usage:
    python compare_gradient_methods.py
    python compare_gradient_methods.py --n-days 365 --timestep 24
    python compare_gradient_methods.py --with-smoothing
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add SYMFLUENCE to path if needed
SYMFLUENCE_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE")
if str(SYMFLUENCE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(SYMFLUENCE_DIR / "src"))

try:
    from symfluence.models.hbv import compare_solvers
    HAS_HBV = True
except ImportError:
    HAS_HBV = False
    print("WARNING: Could not import HBV module")

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class GradientComparison:
    """Comprehensive gradient method comparison."""

    def __init__(self, n_days: int = 365, timestep_hours: int = 24, warmup_days: int = 30):
        self.n_days = n_days
        self.timestep_hours = timestep_hours
        self.warmup_days = warmup_days
        self.results: Dict[str, Any] = {}

    def run_comparison(self, with_smoothing: bool = False):
        """Run full gradient comparison."""
        print("\n" + "="*70)
        print("COMPREHENSIVE GRADIENT METHOD COMPARISON")
        print("="*70)
        print("\nConfiguration:")
        print(f"  Simulation length: {self.n_days} days")
        print(f"  Timestep: {self.timestep_hours} hours")
        print(f"  Warmup period: {self.warmup_days} days")
        print(f"  Smoothing: {'Enabled' if with_smoothing else 'Disabled'}")
        print()

        # Run the built-in comparison
        results = compare_solvers.run_comparison(
            n_days=self.n_days,
            timestep_hours=self.timestep_hours,
            warmup_days=self.warmup_days,
            plot=True,
            save_plot=None,
            verbose=True
        )

        self.results = results
        return results

    def plot_gradient_comparison(self, save_dir: Path):
        """Create detailed gradient comparison plots."""
        if not self.results or 'gradients' not in self.results:
            print("No gradient results to plot")
            return

        grad_results = self.results['gradients']

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Gradient magnitude comparison
        ax1 = fig.add_subplot(gs[0, 0])
        params = list(grad_results['discrete_grads'].keys())
        x = np.arange(len(params))
        width = 0.35

        discrete_vals = [abs(grad_results['discrete_grads'][p]) for p in params]
        ode_vals = [abs(grad_results['ode_grads'][p]) for p in params]

        ax1.bar(x - width/2, discrete_vals, width, label='Direct AD (JAX)', alpha=0.7, color='steelblue')
        ax1.bar(x + width/2, ode_vals, width, label='ODE Adjoint', alpha=0.7, color='coral')
        ax1.set_xlabel('Parameter')
        ax1.set_ylabel('|Gradient|')
        ax1.set_title('Gradient Magnitude Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Relative difference
        ax2 = fig.add_subplot(gs[0, 1])
        rel_diffs = [grad_results['relative_diff'][p] * 100 for p in params]
        colors = ['green' if abs(d) < 5 else 'orange' if abs(d) < 15 else 'red' for d in rel_diffs]
        ax2.bar(params, rel_diffs, color=colors, alpha=0.7)
        ax2.axhline(y=5, color='green', linestyle='--', linewidth=1, label='5% threshold')
        ax2.axhline(y=15, color='orange', linestyle='--', linewidth=1, label='15% threshold')
        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('Relative Difference (%)')
        ax2.set_title('Gradient Accuracy (ODE vs Direct AD)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Computation time comparison
        ax3 = fig.add_subplot(gs[0, 2])
        methods = ['Direct AD\n(BPTT)', 'ODE\n(Adjoint)']
        times = [grad_results['discrete_time'] * 1000, grad_results['ode_time'] * 1000]
        colors_time = ['steelblue', 'coral']
        bars = ax3.bar(methods, times, color=colors_time, alpha=0.7)
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('Gradient Computation Time')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f} ms',
                    ha='center', va='bottom')

        # 4. Gradient direction consistency
        ax4 = fig.add_subplot(gs[1, 0])
        discrete_grad_vec = np.array([grad_results['discrete_grads'][p] for p in params])
        ode_grad_vec = np.array([grad_results['ode_grads'][p] for p in params])

        # Normalize
        discrete_norm = discrete_grad_vec / (np.linalg.norm(discrete_grad_vec) + 1e-10)
        ode_norm = ode_grad_vec / (np.linalg.norm(ode_grad_vec) + 1e-10)

        # Cosine similarity
        cos_sim = np.dot(discrete_norm, ode_norm)

        ax4.scatter(discrete_grad_vec, ode_grad_vec, alpha=0.6, s=100, color='purple')

        # Add parameter labels
        for i, param in enumerate(params):
            ax4.annotate(param, (discrete_grad_vec[i], ode_grad_vec[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Diagonal line
        all_grads = np.concatenate([discrete_grad_vec, ode_grad_vec])
        min_val, max_val = all_grads.min(), all_grads.max()
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement')

        ax4.set_xlabel('Direct AD Gradient')
        ax4.set_ylabel('ODE Adjoint Gradient')
        ax4.set_title(f'Gradient Direction Consistency\n(Cosine Similarity: {cos_sim:.4f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Parameter-wise comparison table
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('tight')
        ax5.axis('off')

        table_data = []
        for param in params:
            table_data.append([
                param,
                f"{grad_results['discrete_grads'][param]:.6f}",
                f"{grad_results['ode_grads'][param]:.6f}",
                f"{grad_results['relative_diff'][param]*100:.2f}%"
            ])

        table = ax5.table(
            cellText=table_data,
            colLabels=['Parameter', 'Direct AD', 'ODE Adjoint', 'Rel. Diff'],
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.25, 0.25, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color code relative differences
        for i, row in enumerate(table_data, start=1):
            rel_diff = float(row[3].rstrip('%'))
            if abs(rel_diff) < 5:
                color = 'lightgreen'
            elif abs(rel_diff) < 15:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
            table[(i, 3)].set_facecolor(color)

        ax5.set_title('Detailed Gradient Comparison', pad=20, fontsize=12, fontweight='bold')

        # Save
        save_path = save_dir / "gradient_method_comparison_detailed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDetailed comparison plot saved: {save_path}")
        plt.close()

    def generate_recommendations(self):
        """Generate method recommendations based on results."""
        if not self.results or 'gradients' not in self.results:
            return

        grad_results = self.results['gradients']

        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)

        # Speed comparison
        speedup = grad_results['discrete_time'] / grad_results['ode_time']
        if speedup > 1.2:
            print(f"\n✓ Direct AD is {speedup:.1f}x FASTER than ODE Adjoint")
            print("  Recommendation: Use Direct AD for routine calibration")
        elif speedup < 0.8:
            print(f"\n✓ ODE Adjoint is {1/speedup:.1f}x FASTER than Direct AD")
            print("  Recommendation: Use ODE Adjoint for long simulations")
        else:
            print(f"\n≈ Similar performance (ratio: {speedup:.2f})")
            print("  Recommendation: Choice depends on other factors")

        # Accuracy comparison
        mean_rel_diff = np.mean(list(grad_results['relative_diff'].values()))
        max_rel_diff = np.max(list(grad_results['relative_diff'].values()))

        print("\n✓ Gradient Accuracy:")
        print(f"  Mean relative difference: {mean_rel_diff*100:.2f}%")
        print(f"  Max relative difference:  {max_rel_diff*100:.2f}%")

        if max_rel_diff < 0.05:
            print("  → Excellent agreement - both methods are reliable")
        elif max_rel_diff < 0.15:
            print("  → Good agreement - both methods suitable for optimization")
        else:
            print("  → Moderate differences - verify with finite differences")

        # Use case recommendations
        print("\n✓ Use Case Recommendations:")
        print("\n  1. Direct AD (JAX lax.scan + BPTT):")
        print("     - Short to medium simulations (< 5 years)")
        print("     - Exact discrete model gradients needed")
        print("     - Development and debugging")
        print("     - When smoothing is enabled")

        print("\n  2. ODE Adjoint (diffrax):")
        print("     - Long simulations (> 5 years)")
        print("     - Memory-constrained environments")
        print("     - Physics/solver separation desired")
        print("     - Adaptive time-stepping needed")

        print("\n  3. Finite Differences:")
        print("     - Gradient verification only")
        print("     - Not recommended for optimization (too slow)")

        # Optimization algorithm recommendations
        print("\n✓ Optimization Algorithm Selection:")

        if mean_rel_diff < 0.10:
            print("\n  Gradient quality is GOOD - gradient-based methods recommended:")
            print("  - ADAM: Fast convergence, good for 1000-2000 iterations")
            print("  - L-BFGS: Even faster, good for 100-500 iterations")
            print("  - Consider smoothing for better gradient quality")
        else:
            print("\n  Gradient quality is MODERATE - consider hybrid approach:")
            print("  - Start with DDS/PSO for global search (1000 iterations)")
            print("  - Finish with ADAM for local refinement (500 iterations)")

        print("\n  Gradient-free methods (always robust):")
        print("  - DDS: Best single-solution method, 2000-4000 iterations")
        print("  - PSO/DE/GA: Population diversity, 2000-4000 iterations")

        print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare gradient computation methods for HBV model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--n-days', type=int, default=365,
                       help='Simulation length in days (default: 365)')
    parser.add_argument('--timestep', type=int, default=24,
                       choices=[1, 3, 6, 12, 24],
                       help='Timestep in hours (default: 24)')
    parser.add_argument('--warmup', type=int, default=30,
                       help='Warmup days for loss calculation (default: 30)')
    parser.add_argument('--with-smoothing', action='store_true',
                       help='Enable smooth threshold functions')
    parser.add_argument('--output-dir', type=Path,
                       default=Path(__file__).parent.parent / "results",
                       help='Output directory for plots')

    args = parser.parse_args()

    if not HAS_HBV:
        print("ERROR: Could not import HBV module. Check SYMFLUENCE installation.")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    comparator = GradientComparison(
        n_days=args.n_days,
        timestep_hours=args.timestep,
        warmup_days=args.warmup
    )

    _results = comparator.run_comparison(with_smoothing=args.with_smoothing)

    # Generate detailed plots
    comparator.plot_gradient_comparison(args.output_dir)

    # Generate recommendations
    comparator.generate_recommendations()

    print(f"\nResults saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
