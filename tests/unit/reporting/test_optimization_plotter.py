"""
Unit tests for OptimizationPlotter.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile

from symfluence.utils.reporting.plotters.optimization_plotter import OptimizationPlotter
from symfluence.utils.reporting.config.plot_config import PlotConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/test',
        'DOMAIN_NAME': 'test_domain'
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock()


@pytest.fixture
def optimization_plotter(mock_config, mock_logger):
    """Create an OptimizationPlotter instance."""
    return OptimizationPlotter(mock_config, mock_logger)


@pytest.fixture
def sample_history():
    """Create sample optimization history."""
    return [
        {'generation': 0, 'best_score': 0.5},
        {'generation': 1, 'best_score': 0.6},
        {'generation': 2, 'best_score': 0.7},
        {'generation': 3, 'best_score': 0.75},
        {'generation': 4, 'best_score': 0.8},
    ]


@pytest.fixture
def sample_depth_history():
    """Create sample history with depth parameters."""
    return [
        {
            'generation': 0,
            'best_params': {
                'total_mult': 1.0,
                'shape_factor': 1.0
            }
        },
        {
            'generation': 1,
            'best_params': {
                'total_mult': 1.2,
                'shape_factor': 1.1
            }
        },
        {
            'generation': 2,
            'best_params': {
                'total_mult': np.array([1.5]),  # Test array handling
                'shape_factor': np.array([1.3])
            }
        },
    ]


class TestOptimizationPlotter:
    """Test suite for OptimizationPlotter."""

    def test_initialization(self, optimization_plotter):
        """Test that OptimizationPlotter initializes correctly."""
        assert optimization_plotter.config is not None
        assert optimization_plotter.logger is not None
        assert optimization_plotter.plot_config is not None

    def test_plot_optimization_progress_success(
        self, optimization_plotter, sample_history
    ):
        """Test successful optimization progress plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create the reporting directory
            (output_dir / "reporting").mkdir(parents=True, exist_ok=True)

            with patch.object(optimization_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(optimization_plotter, '_save_and_close', return_value='/fake/path.png') as mock_save:

                # Mock matplotlib objects
                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax = Mock()
                mock_plt.subplots.return_value = (mock_fig, mock_ax)
                mock_setup.return_value = (mock_plt, None)

                result = optimization_plotter.plot_optimization_progress(
                    sample_history,
                    output_dir,
                    'streamflow',
                    'NSE'
                )

                assert result == '/fake/path.png'
                mock_save.assert_called_once()

    def test_plot_optimization_progress_empty_history(
        self, optimization_plotter
    ):
        """Test plotting with empty history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = optimization_plotter.plot_optimization_progress(
                [],
                output_dir,
                'streamflow',
                'NSE'
            )

            assert result is None

    def test_plot_optimization_progress_no_scores(
        self, optimization_plotter
    ):
        """Test plotting with history but no scores."""
        history = [
            {'generation': 0},
            {'generation': 1, 'best_score': None},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = optimization_plotter.plot_optimization_progress(
                history,
                output_dir,
                'streamflow',
                'NSE'
            )

            assert result is None

    def test_plot_depth_parameters_success(
        self, optimization_plotter, sample_depth_history
    ):
        """Test successful depth parameter plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create the reporting directory
            (output_dir / "reporting").mkdir(parents=True, exist_ok=True)

            with patch.object(optimization_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(optimization_plotter, '_save_and_close', return_value='/fake/path.png') as mock_save:

                # Mock matplotlib objects
                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax1 = Mock()
                mock_ax2 = Mock()
                mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
                mock_setup.return_value = (mock_plt, None)

                result = optimization_plotter.plot_depth_parameters(
                    sample_depth_history,
                    output_dir
                )

                assert result == '/fake/path.png'
                mock_save.assert_called_once()

    def test_plot_depth_parameters_empty_history(
        self, optimization_plotter
    ):
        """Test depth parameters with empty history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = optimization_plotter.plot_depth_parameters(
                [],
                output_dir
            )

            assert result is None

    def test_plot_depth_parameters_no_params(
        self, optimization_plotter
    ):
        """Test depth parameters with history but no params."""
        history = [
            {'generation': 0},
            {'generation': 1, 'best_params': {}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = optimization_plotter.plot_depth_parameters(
                history,
                output_dir
            )

            assert result is None

    def test_plot_depth_parameters_array_handling(
        self, optimization_plotter, sample_depth_history
    ):
        """Test that numpy arrays are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create the reporting directory
            (output_dir / "reporting").mkdir(parents=True, exist_ok=True)

            with patch.object(optimization_plotter, '_setup_matplotlib') as mock_setup, \
                 patch.object(optimization_plotter, '_save_and_close', return_value='/fake/path.png'):

                # Mock matplotlib objects
                mock_plt = Mock()
                mock_fig = Mock()
                mock_ax1 = Mock()
                mock_ax2 = Mock()
                mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
                mock_setup.return_value = (mock_plt, None)

                # Should not raise an error
                result = optimization_plotter.plot_depth_parameters(
                    sample_depth_history,
                    output_dir
                )

                assert result is not None

    def test_plot_method_with_kwargs(
        self, optimization_plotter, sample_history
    ):
        """Test the generic plot method with kwargs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch.object(
                optimization_plotter,
                'plot_optimization_progress',
                return_value='/fake/path.png'
            ) as mock_plot:

                result = optimization_plotter.plot(
                    history=sample_history,
                    output_dir=output_dir,
                    calibration_variable='test',
                    metric='NSE'
                )

                assert result == '/fake/path.png'
                mock_plot.assert_called_once()

    def test_plot_method_without_required_kwargs(
        self, optimization_plotter
    ):
        """Test the generic plot method without required kwargs."""
        result = optimization_plotter.plot()
        assert result is None

    def test_error_handling(
        self, optimization_plotter, sample_history, mock_logger
    ):
        """Test error handling in plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch('matplotlib.pyplot.subplots', side_effect=Exception("Test error")):
                result = optimization_plotter.plot_optimization_progress(
                    sample_history,
                    output_dir,
                    'streamflow',
                    'NSE'
                )

                assert result is None
                mock_logger.error.assert_called()
