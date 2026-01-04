"""
Optimization visualization plotter.

Handles plotting of optimization progress, parameter evolution, and convergence.
"""

import numpy as np  # type: ignore
from pathlib import Path
from typing import List, Dict, Optional, Any

from symfluence.reporting.core.base_plotter import BasePlotter


class OptimizationPlotter(BasePlotter):
    """
    Plotter for optimization visualizations.

    Handles:
    - Optimization progress over generations
    - Parameter evolution tracking
    - Convergence analysis
    - Best solution highlighting
    """

    def plot_optimization_progress(
        self,
        history: List[Dict],
        output_dir: Path,
        calibration_variable: str,
        metric: str
    ) -> Optional[str]:
        """
        Visualize optimization progress over generations.

        Args:
            history: List of optimization history dictionaries
            output_dir: Directory to save the plot
            calibration_variable: Name of variable being calibrated
            metric: Name of optimization metric

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            # Setup plot directory
            plots_dir = output_dir / "reporting"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Extract progress data
            generations = [h.get('generation', 0) for h in history]
            best_scores = [h['best_score'] for h in history if h.get('best_score') is not None]

            if not best_scores:
                self.logger.warning("No best scores found in history for plotting.")
                return None

            # Create figure
            fig, ax = plt.subplots(
                figsize=self.plot_config.FIGURE_SIZE_MEDIUM
            )

            # Plot progress
            ax.plot(
                generations[:len(best_scores)],
                best_scores,
                'b-o',
                markersize=self.plot_config.MARKER_SIZE_SMALL,
                linewidth=self.plot_config.LINE_WIDTH_DEFAULT
            )

            # Mark best score
            best_idx = np.nanargmax(best_scores)
            ax.plot(
                generations[best_idx],
                best_scores[best_idx],
                'ro',
                markersize=self.plot_config.MARKER_SIZE_LARGE,
                label=f'Best: {best_scores[best_idx]:.4f} at generation {generations[best_idx]}'
            )

            # Styling
            self._apply_standard_styling(
                ax,
                xlabel='Generation',
                ylabel=f'Performance ({metric})',
                title=f'Optimization Progress - {calibration_variable.title()} Calibration',
                legend=True
            )

            plt.tight_layout()

            # Save
            plot_path = plots_dir / "optimization_progress.png"
            return self._save_and_close(fig, plot_path)

        except Exception as e:
            self.logger.error(f"Error creating optimization progress plot: {str(e)}")
            return None

    def plot_depth_parameters(
        self,
        history: List[Dict],
        output_dir: Path
    ) -> Optional[str]:
        """
        Visualize depth parameter evolution over optimization.

        Args:
            history: List of optimization history dictionaries
            output_dir: Directory to save the plot

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            # Setup plot directory
            plots_dir = output_dir / "reporting"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Extract depth parameters
            generations = []
            total_mults = []
            shape_factors = []

            for h in history:
                if h.get('best_params') and 'total_mult' in h['best_params'] and 'shape_factor' in h['best_params']:
                    generations.append(h.get('generation', 0))

                    tm = h['best_params']['total_mult']
                    sf = h['best_params']['shape_factor']

                    tm_val = tm[0] if isinstance(tm, np.ndarray) and len(tm) > 0 else tm
                    sf_val = sf[0] if isinstance(sf, np.ndarray) and len(sf) > 0 else sf

                    total_mults.append(tm_val)
                    shape_factors.append(sf_val)

            if not generations:
                self.logger.warning("No depth parameter data found in history for plotting.")
                return None

            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=self.plot_config.FIGURE_SIZE_MEDIUM_TALL
            )

            # Total multiplier plot
            ax1.plot(
                generations,
                total_mults,
                'g-o',
                markersize=self.plot_config.MARKER_SIZE_SMALL,
                linewidth=self.plot_config.LINE_WIDTH_DEFAULT
            )
            ax1.axhline(
                y=1.0,
                color=self.plot_config.COLOR_VALIDATION,
                linestyle='--',
                alpha=0.5,
                label='No change (1.0)'
            )

            self._apply_standard_styling(
                ax1,
                xlabel='Generation',
                ylabel='Total Depth Multiplier',
                title='Soil Depth Total Multiplier Evolution',
                legend=True
            )

            # Shape factor plot
            ax2.plot(
                generations,
                shape_factors,
                'm-o',
                markersize=self.plot_config.MARKER_SIZE_SMALL,
                linewidth=self.plot_config.LINE_WIDTH_DEFAULT
            )
            ax2.axhline(
                y=1.0,
                color=self.plot_config.COLOR_VALIDATION,
                linestyle='--',
                alpha=0.5,
                label='Uniform scaling (1.0)'
            )

            self._apply_standard_styling(
                ax2,
                xlabel='Generation',
                ylabel='Shape Factor',
                title='Soil Depth Shape Factor Evolution',
                legend=True
            )

            plt.tight_layout()

            # Save
            plot_path = plots_dir / "depth_parameter_evolution.png"
            return self._save_and_close(fig, plot_path)

        except Exception as e:
            self.logger.error(f"Error creating depth parameter plots: {str(e)}")
            return None

    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Main plot method (required by BasePlotter).

        Delegates to plot_optimization_progress() by default.
        """
        if 'history' in kwargs and 'output_dir' in kwargs:
            return self.plot_optimization_progress(
                kwargs['history'],
                kwargs['output_dir'],
                kwargs.get('calibration_variable', 'unknown'),
                kwargs.get('metric', 'performance')
            )
        return None
