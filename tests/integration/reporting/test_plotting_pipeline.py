"""
Integration tests for end-to-end plotting pipeline.

These tests verify that the full plotting pipeline works correctly,
generating actual plot files to disk using real matplotlib.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a logger for tests."""
    logger = logging.getLogger('test_plotting')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_config():
    """Create a test configuration."""
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/test',
        'DOMAIN_NAME': 'integration_test',
        'EXPERIMENT_ID': 'test_exp',
        'RIVER_BASINS_NAME': 'default',
        'RIVER_NETWORK_SHP_NAME': 'default',
        'POUR_POINT_SHP_NAME': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'SIM_REACH_ID': 1,
        'OPTIMIZATION_METRIC': 'KGE',
        'OPTIMIZATION_TARGET': 'streamflow',
        'SPINUP_PERIOD': '1980-01-01,1981-01-01',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'GRUs',
    }


@pytest.fixture
def sample_streamflow_data():
    """Create realistic streamflow data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=730, freq='D')  # 2 years

    # Create seasonal pattern
    day_of_year = dates.dayofyear
    base_flow = 10 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 1, len(dates))
    base_flow = np.maximum(0.5, base_flow)

    obs_series = pd.Series(base_flow, index=dates, name='discharge_cms')

    # Create simulated data with some bias
    sim_summa = base_flow * 0.95 + np.random.normal(0, 0.3, len(dates))
    sim_fuse = base_flow * 1.02 + np.random.normal(0, 0.4, len(dates))
    sim_summa = np.maximum(0.1, sim_summa)
    sim_fuse = np.maximum(0.1, sim_fuse)

    results_df = pd.DataFrame({
        'obs_discharge': base_flow,
        'SUMMA_discharge': sim_summa,
        'FUSE_discharge': sim_fuse,
    }, index=dates)

    return results_df, obs_series


# ============================================================================
# Panel Integration Tests
# ============================================================================

class TestPanelRendering:
    """Test that panel classes render correctly to actual figures."""

    def test_timeseries_panel_creates_plot(self, sample_config, mock_logger, sample_streamflow_data):
        """Test TimeSeriesPanel creates valid plot."""
        import matplotlib.pyplot as plt
        from symfluence.reporting.panels import TimeSeriesPanel
        from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG

        results_df, obs_series = sample_streamflow_data
        model_cols = ['SUMMA_discharge', 'FUSE_discharge']

        fig, ax = plt.subplots(figsize=(10, 6))

        panel = TimeSeriesPanel(DEFAULT_PLOT_CONFIG, mock_logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': model_cols
        })

        # Verify plot has content
        assert len(ax.lines) >= 2  # At least observations and one model
        assert ax.get_ylabel() != ''

        plt.close(fig)

    def test_fdc_panel_creates_plot(self, sample_config, mock_logger, sample_streamflow_data):
        """Test FDCPanel creates valid flow duration curve."""
        import matplotlib.pyplot as plt
        from symfluence.reporting.panels import FDCPanel
        from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG

        results_df, obs_series = sample_streamflow_data
        model_cols = ['SUMMA_discharge', 'FUSE_discharge']

        fig, ax = plt.subplots(figsize=(8, 6))

        panel = FDCPanel(DEFAULT_PLOT_CONFIG, mock_logger)
        panel.render(ax, {
            'results_df': results_df,
            'obs_series': obs_series,
            'model_cols': model_cols
        })

        # FDC should have log y-scale (x-axis is linear exceedance probability)
        assert ax.get_yscale() == 'log'

        plt.close(fig)

    def test_scatter_panel_creates_plot(self, sample_config, mock_logger, sample_streamflow_data):
        """Test ScatterPanel creates valid scatter plot."""
        import matplotlib.pyplot as plt
        from symfluence.reporting.panels import ScatterPanel
        from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG

        results_df, obs_series = sample_streamflow_data

        fig, ax = plt.subplots(figsize=(6, 6))

        panel = ScatterPanel(DEFAULT_PLOT_CONFIG, mock_logger)
        panel.render(ax, {
            'obs_values': obs_series.values,
            'sim_values': results_df['SUMMA_discharge'].values,
            'model_name': 'SUMMA',
            'color_index': 0
        })

        # Should have scatter points
        assert len(ax.collections) > 0 or len(ax.lines) > 0

        plt.close(fig)

    def test_metrics_table_panel(self, sample_config, mock_logger):
        """Test MetricsTablePanel renders metrics table."""
        import matplotlib.pyplot as plt
        from symfluence.reporting.panels import MetricsTablePanel
        from symfluence.reporting.config.plot_config import DEFAULT_PLOT_CONFIG

        metrics_dict = {
            'SUMMA': {'KGE': 0.75, 'NSE': 0.70, 'RMSE': 2.5, 'Bias': -0.5},
            'FUSE': {'KGE': 0.68, 'NSE': 0.65, 'RMSE': 3.1, 'Bias': 0.3},
        }

        fig, ax = plt.subplots(figsize=(4, 4))

        panel = MetricsTablePanel(DEFAULT_PLOT_CONFIG, mock_logger)
        panel.render(ax, {'metrics_dict': metrics_dict})

        # Table should have turned off axes
        assert not ax.axison or ax.get_frame_on() is False or len(ax.tables) > 0

        plt.close(fig)


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestModelComparisonPipeline:
    """Test the full model comparison plotting pipeline."""

    def test_model_comparison_overview_creates_file(
        self, sample_config, mock_logger, sample_streamflow_data
    ):
        """Test that model comparison overview creates actual PNG file."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        from unittest.mock import patch

        results_df, obs_series = sample_streamflow_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = ModelComparisonPlotter(sample_config, mock_logger, None)
            plotter.project_dir = Path(tmpdir)

            with patch.object(plotter, '_collect_model_data',
                            return_value=(results_df, obs_series)):

                result = plotter.plot_model_comparison_overview(
                    experiment_id='integration_test',
                    context='run_model'
                )

                # Verify file was created
                assert result is not None
                assert Path(result).exists()
                assert Path(result).suffix == '.png'

                # Verify file has content
                file_size = Path(result).stat().st_size
                assert file_size > 1000  # Should be at least 1KB

    def test_model_comparison_different_contexts(
        self, sample_config, mock_logger, sample_streamflow_data
    ):
        """Test model comparison works with different contexts."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        from unittest.mock import patch

        results_df, obs_series = sample_streamflow_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = ModelComparisonPlotter(sample_config, mock_logger, None)
            plotter.project_dir = Path(tmpdir)

            with patch.object(plotter, '_collect_model_data',
                            return_value=(results_df, obs_series)):

                # Test both contexts
                for context in ['run_model', 'calibrate_model']:
                    result = plotter.plot_model_comparison_overview(
                        experiment_id=f'test_{context}',
                        context=context
                    )

                    assert result is not None
                    assert Path(result).exists()


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases in plotting pipeline."""

    def test_single_model_comparison(
        self, sample_config, mock_logger, sample_streamflow_data
    ):
        """Test that comparison works with a single model."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        from unittest.mock import patch

        results_df, obs_series = sample_streamflow_data

        # Keep only one model
        single_model_df = results_df[['obs_discharge', 'SUMMA_discharge']].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = ModelComparisonPlotter(sample_config, mock_logger, None)
            plotter.project_dir = Path(tmpdir)

            with patch.object(plotter, '_collect_model_data',
                            return_value=(single_model_df, obs_series)):

                result = plotter.plot_model_comparison_overview(
                    experiment_id='single_model_test'
                )

                assert result is not None
                assert Path(result).exists()

    def test_short_time_series(self, sample_config, mock_logger):
        """Test handling of short time series."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        from unittest.mock import patch

        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        obs_series = pd.Series(np.random.random(30) * 10 + 5, index=dates)
        results_df = pd.DataFrame({
            'obs_discharge': obs_series.values,
            'SUMMA_discharge': obs_series.values * 0.9,
        }, index=dates)

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = ModelComparisonPlotter(sample_config, mock_logger, None)
            plotter.project_dir = Path(tmpdir)

            with patch.object(plotter, '_collect_model_data',
                            return_value=(results_df, obs_series)):

                result = plotter.plot_model_comparison_overview(
                    experiment_id='short_series_test'
                )

                # Should still create plot even with short series
                assert result is not None
                assert Path(result).exists()

    def test_data_with_nan_values(self, sample_config, mock_logger, sample_streamflow_data):
        """Test that NaN values are handled gracefully."""
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        from unittest.mock import patch

        results_df, obs_series = sample_streamflow_data

        # Introduce NaN values
        results_df_with_nan = results_df.copy()
        results_df_with_nan.iloc[10:20, 1] = np.nan  # NaN in SUMMA
        obs_with_nan = obs_series.copy()
        obs_with_nan.iloc[50:60] = np.nan

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = ModelComparisonPlotter(sample_config, mock_logger, None)
            plotter.project_dir = Path(tmpdir)

            with patch.object(plotter, '_collect_model_data',
                            return_value=(results_df_with_nan, obs_with_nan)):

                result = plotter.plot_model_comparison_overview(
                    experiment_id='nan_test'
                )

                # Should handle NaN gracefully
                assert result is not None
                assert Path(result).exists()


# ============================================================================
# Reporting Manager Integration
# ============================================================================

class TestReportingManagerIntegration:
    """Test ReportingManager integration with plotters."""

    def test_model_comparison_plotter_standalone(self, sample_config, mock_logger, sample_streamflow_data):
        """Test ModelComparisonPlotter works independently of ReportingManager.

        Note: Full ReportingManager integration requires SymfluenceConfig object
        and complex domain setup. This test verifies the plotter component works.
        """
        from symfluence.reporting.plotters.model_comparison_plotter import ModelComparisonPlotter
        from unittest.mock import patch

        results_df, obs_series = sample_streamflow_data

        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = ModelComparisonPlotter(sample_config, mock_logger, None)
            plotter.project_dir = Path(tmpdir)

            with patch.object(plotter, '_collect_model_data',
                            return_value=(results_df, obs_series)):

                result = plotter.plot_model_comparison_overview(
                    experiment_id='standalone_test',
                    context='run_model'
                )

                assert result is not None
                assert Path(result).exists()

                # Verify the plot file has reasonable size (actual plot content)
                file_size = Path(result).stat().st_size
                assert file_size > 5000  # Should be more than 5KB with all panels
