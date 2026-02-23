"""
Unit tests for PlotConfig dataclass.
"""

import pytest

from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG, PlotConfig


class TestPlotConfig:
    """Test suite for PlotConfig dataclass."""

    def test_default_initialization(self):
        """Test that PlotConfig can be initialized with defaults."""
        config = PlotConfig()
        assert config.DPI_DEFAULT == 300
        assert config.FIGURE_SIZE_SMALL == (10, 6)
        assert config.COLOR_OBSERVED == '#000000'

    def test_custom_initialization(self):
        """Test that PlotConfig can be initialized with custom values."""
        config = PlotConfig(
            DPI_DEFAULT=600,
            FIGURE_SIZE_SMALL=(8, 5)
        )
        assert config.DPI_DEFAULT == 600
        assert config.FIGURE_SIZE_SMALL == (8, 5)

    def test_default_plot_config_exists(self):
        """Test that DEFAULT_PLOT_CONFIG is available."""
        assert DEFAULT_PLOT_CONFIG is not None
        assert isinstance(DEFAULT_PLOT_CONFIG, PlotConfig)

    def test_get_figure_size_valid(self):
        """Test get_figure_size with valid size keys."""
        config = PlotConfig()

        assert config.get_figure_size('small') == (10, 6)
        assert config.get_figure_size('medium') == (12, 6)
        assert config.get_figure_size('medium_tall') == (12, 8)
        assert config.get_figure_size('large') == (14, 10)
        assert config.get_figure_size('xlarge') == (15, 15)
        assert config.get_figure_size('xlarge_tall') == (15, 16)
        assert config.get_figure_size('xxlarge') == (20, 10)

    def test_get_figure_size_invalid(self):
        """Test get_figure_size with invalid size key."""
        config = PlotConfig()

        with pytest.raises(ValueError, match="Unknown size_key"):
            config.get_figure_size('invalid_size')

    def test_get_color_from_palette(self):
        """Test get_color_from_palette with various indices."""
        config = PlotConfig()

        # First color
        assert config.get_color_from_palette(0) == '#1f77b4'

        # Last color in palette
        assert config.get_color_from_palette(9) == '#17becf'

        # Wrapping (index beyond palette length)
        assert config.get_color_from_palette(10) == '#1f77b4'  # wraps to 0
        assert config.get_color_from_palette(11) == '#ff7f0e'  # wraps to 1

    def test_get_line_style(self):
        """Test get_line_style with various indices."""
        config = PlotConfig()

        assert config.get_line_style(0) == '-'
        assert config.get_line_style(1) == '--'
        assert config.get_line_style(2) == ':'
        assert config.get_line_style(3) == '-.'

        # Test wrapping
        assert config.get_line_style(4) == '-'
        assert config.get_line_style(5) == '--'

    def test_color_palette_length(self):
        """Test that color palette has expected length."""
        config = PlotConfig()
        assert len(config.COLOR_PALETTE_DEFAULT) == 10

    def test_color_palette_all_hex(self):
        """Test that all colors in palette are valid hex colors."""
        config = PlotConfig()

        for color in config.COLOR_PALETTE_DEFAULT:
            assert color.startswith('#')
            assert len(color) == 7
            # Validate hex characters
            assert all(c in '0123456789abcdefABCDEF' for c in color[1:])

    def test_dpi_values(self):
        """Test DPI values are reasonable."""
        config = PlotConfig()

        assert config.DPI_DEFAULT >= 72  # Minimum screen DPI
        assert config.DPI_HIGH >= config.DPI_DEFAULT

    def test_line_width_values(self):
        """Test line width values are positive and ordered."""
        config = PlotConfig()

        assert config.LINE_WIDTH_THIN > 0
        assert config.LINE_WIDTH_DEFAULT > config.LINE_WIDTH_THIN
        assert config.LINE_WIDTH_THICK > config.LINE_WIDTH_DEFAULT
        assert config.LINE_WIDTH_OBSERVED > config.LINE_WIDTH_THICK

    def test_alpha_values_in_range(self):
        """Test that alpha values are between 0 and 1."""
        config = PlotConfig()

        assert 0 <= config.ALPHA_FAINT <= 1
        assert 0 <= config.ALPHA_LIGHT <= 1
        assert 0 <= config.ALPHA_DEFAULT <= 1
        assert 0 <= config.GRID_ALPHA <= 1
        assert 0 <= config.METRICS_BOX_ALPHA <= 1

    def test_spinup_defaults(self):
        """Test spinup default values are reasonable."""
        config = PlotConfig()

        assert 0 < config.SPINUP_PERCENT_DEFAULT <= 100
        assert config.SPINUP_DAYS_DEFAULT > 0
        assert config.SPINUP_DAYS_SHORT > 0
        assert config.SPINUP_DAYS_MEDIUM > 0
        assert config.SPINUP_DAYS_SHORT < config.SPINUP_DAYS_MEDIUM < config.SPINUP_DAYS_DEFAULT

    def test_font_size_ordering(self):
        """Test that font sizes are ordered correctly."""
        config = PlotConfig()

        assert config.FONT_SIZE_SMALL < config.FONT_SIZE_MEDIUM
        assert config.FONT_SIZE_MEDIUM < config.FONT_SIZE_LARGE
        assert config.FONT_SIZE_LARGE < config.FONT_SIZE_TITLE
