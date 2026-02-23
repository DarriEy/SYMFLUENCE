"""
Abstract base class for all plotters.

This module provides common functionality shared across all plotting classes,
eliminating code duplication and providing consistent behavior.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from symfluence.core.constants import ConfigKeys
from symfluence.core.mixins import ConfigMixin
from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG, PlotConfig


class BasePlotter(ConfigMixin, ABC):
    """
    Abstract base class for all plot generators.

    Provides shared functionality for:
    - Matplotlib setup and imports
    - Directory management
    - File saving and cleanup
    - Common plot styling (grids, axes, metrics boxes)
    - Consistent logging

    All concrete plotters should inherit from this class.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        plot_config: Optional[PlotConfig] = None
    ):
        """
        Initialize the base plotter.

        Args:
            config: SYMFLUENCE configuration dictionary
            logger: Logger instance for messaging
            plot_config: Optional PlotConfig instance (uses default if not provided)
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.plot_config = plot_config or DEFAULT_PLOT_CONFIG

        # Lazy-loaded matplotlib modules
        self._plt = None
        self._mdates = None

        # Base project directory
        self.project_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key=ConfigKeys.SYMFLUENCE_DATA_DIR)) / f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}"

    def _setup_matplotlib(self) -> Tuple[Any, Any]:
        """
        Lazy import and setup of matplotlib modules.

        Returns:
            Tuple of (pyplot, dates) modules

        Note:
            This method delays matplotlib import until actually needed,
            improving startup time and allowing for headless operation.
        """
        if self._plt is None or self._mdates is None:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            self._plt = plt
            self._mdates = mdates

        return self._plt, self._mdates

    def _ensure_output_dir(self, *subdirs: str) -> Path:
        """
        Ensure output directory exists and return its path.

        Args:
            *subdirs: Subdirectories under the reporting folder (e.g., 'results', 'domain')

        Returns:
            Path to the output directory

        Note:
            Creates directory structure if it doesn't exist.
        """
        # Use 'reporting' as the standard output directory
        output_dir = self.project_dir / "reporting"

        for subdir in subdirs:
            output_dir = output_dir / subdir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_and_close(
        self,
        fig: Any,
        filepath: Path,
        dpi: Optional[int] = None,
        bbox_inches: Optional[str] = None
    ) -> str:
        """
        Save figure to file and close it.

        Args:
            fig: Matplotlib figure object
            filepath: Path where to save the plot
            dpi: Dots per inch (uses config default if not specified)
            bbox_inches: Bounding box setting (uses config default if not specified)

        Returns:
            String path to the saved file

        Note:
            Eliminates ~30 duplicated save/close patterns.
            Always logs the save operation.
        """
        plt, _ = self._setup_matplotlib()

        dpi = dpi or self.plot_config.DPI_DEFAULT
        bbox_inches = bbox_inches or self.plot_config.BBOX_INCHES

        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
            self.logger.info(f"Plot saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving plot to {filepath}: {str(e)}")
            raise
        finally:
            plt.close(fig)

        return str(filepath)

    def _format_date_axis(
        self,
        ax: Any,
        format_type: str = 'year',
        rotation: int = 0
    ) -> None:
        """
        Apply standard date formatting to an axis.

        Args:
            ax: Matplotlib axis object
            format_type: One of 'year', 'month', 'day', 'full'
            rotation: Rotation angle for tick labels

        Note:
            Eliminates ~20 duplicated date formatting patterns.
        """
        _, mdates = self._setup_matplotlib()

        format_map = {
            'year': (mdates.YearLocator(), self.plot_config.DATE_FORMAT_YEAR),
            'month': (mdates.MonthLocator(), self.plot_config.DATE_FORMAT_MONTH),
            'day': (mdates.DayLocator(interval=7), self.plot_config.DATE_FORMAT_DAY),
            'full': (mdates.AutoDateLocator(), self.plot_config.DATE_FORMAT_FULL),
        }

        if format_type not in format_map:
            self.logger.warning(
                f"Unknown date format_type '{format_type}', using 'year'"
            )
            format_type = 'year'

        locator, fmt = format_map[format_type]
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

        if rotation:
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)

    def _add_grid(
        self,
        ax: Any,
        which: str = 'major',
        alpha: Optional[float] = None,
        linestyle: Optional[str] = None,
        color: Optional[str] = None
    ) -> None:
        """
        Add grid to an axis with standard styling.

        Args:
            ax: Matplotlib axis object
            which: One of 'major', 'minor', 'both'
            alpha: Grid transparency (uses config default if not specified)
            linestyle: Grid line style (uses config default if not specified)
            color: Grid color (uses config default if not specified)
        """
        alpha = alpha or self.plot_config.GRID_ALPHA
        linestyle = linestyle or self.plot_config.GRID_STYLE
        color = color or self.plot_config.GRID_COLOR

        ax.grid(
            True,
            which=which,
            alpha=alpha,
            linestyle=linestyle,
            color=color
        )

    def _add_metrics_text(
        self,
        ax: Any,
        metrics: Dict[str, float],
        position: Tuple[float, float] = (0.02, 0.98),
        label: str = "Metrics",
        fontsize: Optional[int] = None,
        bbox_alpha: Optional[float] = None
    ) -> None:
        """
        Add formatted metrics text box to a plot.

        Args:
            ax: Matplotlib axis object
            metrics: Dictionary of metric names and values
            position: (x, y) position in axis coordinates (0-1)
            label: Label prefix for the metrics box
            fontsize: Font size (uses config default if not specified)
            bbox_alpha: Text box transparency (uses config default if not specified)
        """
        fontsize = fontsize or self.plot_config.METRICS_FONT_SIZE
        bbox_alpha = bbox_alpha or self.plot_config.METRICS_BOX_ALPHA

        if not metrics:
            return

        # Format metrics text
        if label:
            metric_text = f"{label}:\n"
        else:
            metric_text = ""

        metric_text += "\n".join([
            f"{k}: {v:.3f}" if not isinstance(v, str) else f"{k}: {v}"
            for k, v in metrics.items()
        ])

        # Add text box
        ax.text(
            position[0], position[1],
            metric_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=fontsize,
            bbox=dict(
                facecolor=self.plot_config.METRICS_BOX_COLOR,
                alpha=bbox_alpha,
                edgecolor='none',
                pad=3
            )
        )

    def _set_background_color(
        self,
        ax: Any,
        color: Optional[str] = None
    ) -> None:
        """
        Set axis background color.

        Args:
            ax: Matplotlib axis object
            color: Background color (uses config default if not specified)
        """
        color = color or self.plot_config.COLOR_BACKGROUND_LIGHT
        ax.set_facecolor(color)

    def _apply_standard_styling(
        self,
        ax: Any,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        add_grid: bool = True,
        add_background: bool = True,
        legend: bool = True,
        legend_loc: Optional[str] = None
    ) -> None:
        """
        Apply standard styling to an axis.

        Args:
            ax: Matplotlib axis object
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            add_grid: Whether to add grid
            add_background: Whether to set background color
            legend: Whether to add legend
            legend_loc: Legend location (uses config default if not specified)
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.plot_config.FONT_SIZE_MEDIUM)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.plot_config.FONT_SIZE_MEDIUM)

        if title:
            ax.set_title(title, fontsize=self.plot_config.FONT_SIZE_TITLE)

        if add_grid:
            self._add_grid(ax)

        if add_background:
            self._set_background_color(ax)

        if legend:
            legend_loc = legend_loc or self.plot_config.LEGEND_LOCATION
            ax.legend(
                loc=legend_loc,
                fontsize=self.plot_config.LEGEND_FONT_SIZE,
                framealpha=self.plot_config.LEGEND_FRAMEALPHA
            )

    @abstractmethod
    def plot(self, *args, **kwargs) -> Optional[str]:
        """
        Abstract method that all concrete plotters must implement.

        Returns:
            Path to the saved plot, or None if visualization is disabled or failed
        """
        pass
