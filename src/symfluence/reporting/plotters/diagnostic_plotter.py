"""
Diagnostic visualization tools for data quality and availability assessment.

Provides specialized plotting capabilities for analyzing data distributions,
spatial coverage, temporal availability, and missing data patterns. Used
throughout SYMFLUENCE workflows to validate data quality at each processing stage.
"""

import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from symfluence.reporting.core.base_plotter import BasePlotter


class DiagnosticPlotter(BasePlotter):
    """
    Plotter for diagnostic visualizations.

    Handles:
    - Data distribution plots (histograms, boxplots)
    - Spatial coverage maps for gridded data
    - Time series availability plots
    - Missing data analysis
    """

    def plot_data_distribution(
        self,
        data: Any,
        variable_name: str,
        stage: str,
        output_name: str = 'distribution'
    ) -> Optional[str]:
        """
        Plot distribution of data values.

        Args:
            data: Data to plot (numpy array, Series, or flattened DataFrame)
            variable_name: Name of the variable (e.g., 'precipitation')
            stage: Processing stage (e.g., 'acquisition', 'preprocessing')
            output_name: Base name for output file

        Returns:
            Path to saved plot, or None if failed
        """
        plt, _ = self._setup_matplotlib()

        try:
            # Setup plot directory
            plot_dir = self._ensure_output_dir('diagnostics', stage)
            plot_filename = plot_dir / f'{output_name}_{variable_name}.png'

            # flatten data if needed
            if isinstance(data, pd.DataFrame):
                flat_data = data.values.flatten()
            elif isinstance(data, pd.Series):
                flat_data = data.values
            elif isinstance(data, np.ndarray):
                flat_data = data.flatten()
            else:
                self.logger.warning(f"Unsupported data type for distribution plot: {type(data)}")
                return None

            # Remove NaNs
            flat_data = flat_data[~np.isnan(flat_data)]

            if len(flat_data) == 0:
                self.logger.warning(f"No valid data to plot for {variable_name}")
                return None

            # Create figure with two subplots: histogram and boxplot
            fig, (ax1, ax2) = plt.subplots(
                1, 2,
                figsize=self.plot_config.FIGURE_SIZE_MEDIUM,
                gridspec_kw={'width_ratios': [3, 1]}
            )

            # Histogram
            ax1.hist(
                flat_data,
                bins=50,
                color=self.plot_config.COLOR_SIMULATED_PRIMARY,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )

            # Add mean and median lines
            mean_val = np.mean(flat_data)
            median_val = np.median(flat_data)

            ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax1.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.2f}')

            self._apply_standard_styling(
                ax1,
                xlabel=f'{variable_name} Value',
                ylabel='Frequency',
                title=f'{variable_name} Distribution',
                legend=True
            )

            # Boxplot
            ax2.boxplot(
                flat_data,
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor=self.plot_config.COLOR_SIMULATED_SECONDARY, alpha=0.7),
                medianprops=dict(color='black')
            )

            self._apply_standard_styling(
                ax2,
                ylabel='Value',
                title='Boxplot',
                legend=False,
                add_grid=True
            )
            ax2.set_xticklabels([''])

            plt.suptitle(f'Diagnostic: {variable_name} ({stage})', fontsize=self.plot_config.FONT_SIZE_TITLE)
            plt.tight_layout()

            return self._save_and_close(fig, plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_data_distribution: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def plot_spatial_coverage(
        self,
        raster_path: Path,
        variable_name: str,
        stage: str
    ) -> Optional[str]:
        """
        Plot spatial coverage of a raster file.

        Args:
            raster_path: Path to raster file
            variable_name: Name of variable
            stage: Processing stage

        Returns:
            Path to saved plot, or None
        """
        import rasterio  # type: ignore
        plt, _ = self._setup_matplotlib()

        try:
            if not raster_path.exists():
                self.logger.warning(f"Raster file not found: {raster_path}")
                return None

            plot_dir = self._ensure_output_dir('diagnostics', stage)
            plot_filename = plot_dir / f'spatial_{variable_name}.png'

            with rasterio.open(raster_path) as src:
                data = src.read(1, masked=True)
                extent = src.bounds

                fig, ax = plt.subplots(figsize=self.plot_config.FIGURE_SIZE_MEDIUM)

                # Use a divergent colormap if appropriate, or sequential
                cmap = 'viridis'
                if 'temp' in variable_name.lower():
                    cmap = 'RdYlBu_r'
                elif 'precip' in variable_name.lower():
                    cmap = 'Blues'

                im = ax.imshow(data, cmap=cmap, extent=[extent.left, extent.right, extent.bottom, extent.top])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(variable_name)

                # Basic stats
                valid_data = data.compressed()
                if valid_data.size > 0:
                    stats_text = (
                        f"Min: {valid_data.min():.2f}\n"
                        f"Max: {valid_data.max():.2f}\n"
                        f"Mean: {valid_data.mean():.2f}"
                    )
                    ax.text(
                        0.02, 0.02, stats_text,
                        transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                        verticalalignment='bottom'
                    )

                self._apply_standard_styling(
                    ax,
                    xlabel='Longitude',
                    ylabel='Latitude',
                    title=f'Spatial Coverage: {variable_name}',
                    add_grid=False
                )

                return self._save_and_close(fig, plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_spatial_coverage: {str(e)}")
            return None

    def plot(self, *args, **kwargs) -> Optional[str]:
        """Main plot method."""
        if 'data' in kwargs:
            return self.plot_data_distribution(
                kwargs['data'],
                kwargs.get('variable_name', 'unknown'),
                kwargs.get('stage', 'unknown'),
                kwargs.get('output_name', 'distribution')
            )
        elif 'raster_path' in kwargs:
            return self.plot_spatial_coverage(
                kwargs['raster_path'],
                kwargs.get('variable_name', 'unknown'),
                kwargs.get('stage', 'unknown')
            )
        return None
