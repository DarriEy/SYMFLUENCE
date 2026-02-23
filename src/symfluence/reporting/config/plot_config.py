"""
Configuration dataclass for plot styling and defaults.

This module centralizes all hard-coded values related to plotting,
making them easy to maintain and customize.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PlotConfig:
    """Configuration for plot styling, sizes, colors, and defaults."""

    # ============================================================================
    # Figure Sizes (width, height in inches)
    # ============================================================================
    FIGURE_SIZE_SMALL: Tuple[int, int] = (10, 6)
    FIGURE_SIZE_MEDIUM: Tuple[int, int] = (12, 6)
    FIGURE_SIZE_MEDIUM_TALL: Tuple[int, int] = (12, 8)
    FIGURE_SIZE_LARGE: Tuple[int, int] = (14, 10)
    FIGURE_SIZE_XLARGE: Tuple[int, int] = (15, 15)
    FIGURE_SIZE_XLARGE_TALL: Tuple[int, int] = (15, 16)
    FIGURE_SIZE_XXLARGE: Tuple[int, int] = (20, 10)

    # ============================================================================
    # DPI Settings
    # ============================================================================
    DPI_DEFAULT: int = 300
    DPI_HIGH: int = 600

    # ============================================================================
    # Color Palettes
    # ============================================================================
    # Matplotlib default color cycle (tab10)
    COLOR_PALETTE_DEFAULT: List[str] = field(default_factory=lambda: [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ])

    # Specific colors for common uses
    COLOR_OBSERVED: str = '#000000'  # Black
    COLOR_SIMULATED_PRIMARY: str = '#1f77b4'  # Blue
    COLOR_CALIBRATION: str = '#2ca02c'  # Green
    COLOR_VALIDATION: str = '#d62728'  # Red

    # Domain/spatial visualization colors
    COLOR_BOUNDARY: str = '#2c3e50'  # Dark blue-gray
    COLOR_RIVER: str = '#3498db'  # Light blue
    COLOR_POUR_POINT: str = '#e74c3c'  # Red
    COLOR_TABLE_HEADER: str = '#4472C4'  # Blue

    # Background color
    COLOR_BACKGROUND_LIGHT: str = '#f0f0f0'  # Light gray

    # ============================================================================
    # Line and Marker Styles
    # ============================================================================
    LINE_WIDTH_THIN: float = 0.8
    LINE_WIDTH_DEFAULT: float = 1.5
    LINE_WIDTH_THICK: float = 2.0
    LINE_WIDTH_OBSERVED: float = 2.5

    MARKER_SIZE_SMALL: int = 4
    MARKER_SIZE_MEDIUM: int = 6
    MARKER_SIZE_LARGE: int = 8

    ALPHA_DEFAULT: float = 1.0
    ALPHA_LIGHT: float = 0.7
    ALPHA_FAINT: float = 0.3

    # ============================================================================
    # Grid and Axis Styling
    # ============================================================================
    GRID_ALPHA: float = 0.3
    GRID_STYLE: str = '--'
    GRID_COLOR: str = 'gray'

    # ============================================================================
    # Text and Font Settings
    # ============================================================================
    FONT_SIZE_SMALL: int = 8
    FONT_SIZE_MEDIUM: int = 10
    FONT_SIZE_LARGE: int = 12
    FONT_SIZE_TITLE: int = 14

    METRICS_FONT_SIZE: int = 8
    METRICS_BOX_ALPHA: float = 0.7
    METRICS_BOX_COLOR: str = 'white'

    # ============================================================================
    # Legend Settings
    # ============================================================================
    LEGEND_FONT_SIZE: int = 10
    LEGEND_LOCATION: str = 'best'
    LEGEND_FRAMEALPHA: float = 0.9

    # ============================================================================
    # Spinup and Data Processing Defaults
    # ============================================================================
    SPINUP_PERCENT_DEFAULT: float = 10.0  # Percentage of data to skip at beginning
    SPINUP_DAYS_DEFAULT: int = 365  # Days to skip at beginning
    SPINUP_DAYS_SHORT: int = 50  # Shorter spinup for some models
    SPINUP_DAYS_MEDIUM: int = 100  # Medium spinup

    # ============================================================================
    # Plot Layout and Spacing
    # ============================================================================
    BBOX_INCHES: str = 'tight'
    PAD_INCHES: float = 0.1

    # Subplot spacing
    HSPACE: float = 0.3  # Vertical space between subplots
    WSPACE: float = 0.2  # Horizontal space between subplots

    # ============================================================================
    # Date Formatting
    # ============================================================================
    DATE_FORMAT_YEAR: str = '%Y'
    DATE_FORMAT_MONTH: str = '%Y-%m'
    DATE_FORMAT_DAY: str = '%Y-%m-%d'
    DATE_FORMAT_FULL: str = '%Y-%m-%d %H:%M'

    # ============================================================================
    # Validation and Utility Methods
    # ============================================================================

    def get_figure_size(self, size_key: str) -> Tuple[int, int]:
        """
        Get figure size by key.

        Args:
            size_key: One of 'small', 'medium', 'medium_tall', 'large',
                     'xlarge', 'xlarge_tall', 'xxlarge'

        Returns:
            Tuple of (width, height) in inches

        Raises:
            ValueError: If size_key is not recognized
        """
        size_map = {
            'small': self.FIGURE_SIZE_SMALL,
            'medium': self.FIGURE_SIZE_MEDIUM,
            'medium_tall': self.FIGURE_SIZE_MEDIUM_TALL,
            'large': self.FIGURE_SIZE_LARGE,
            'xlarge': self.FIGURE_SIZE_XLARGE,
            'xlarge_tall': self.FIGURE_SIZE_XLARGE_TALL,
            'xxlarge': self.FIGURE_SIZE_XXLARGE,
        }

        if size_key not in size_map:
            raise ValueError(
                f"Unknown size_key '{size_key}'. "
                f"Must be one of: {', '.join(size_map.keys())}"
            )

        return size_map[size_key]

    def get_color_from_palette(self, index: int) -> str:
        """
        Get color from default palette by index (with wrapping).

        Args:
            index: Color index

        Returns:
            Hex color string
        """
        return self.COLOR_PALETTE_DEFAULT[index % len(self.COLOR_PALETTE_DEFAULT)]

    def get_line_style(self, index: int) -> str:
        """
        Get line style by index (cycles through solid, dashed, dotted, dash-dot).

        Args:
            index: Style index

        Returns:
            Line style string
        """
        styles = ['-', '--', ':', '-.']
        return styles[index % len(styles)]


# Default global instance
DEFAULT_PLOT_CONFIG = PlotConfig()
