import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from symfluence.utils.reporting.reporting_manager import ReportingManager

@pytest.fixture
def mock_config():
    return {
        'SYMFLUENCE_DATA_DIR': '/tmp/symfluence_data',
        'DOMAIN_NAME': 'test_domain'
    }

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def mock_viz_reporter():
    with patch('symfluence.utils.reporting.reporting_manager.VisualizationReporter') as mock:
        yield mock

@pytest.fixture
def mock_ts_visualizer():
    with patch('symfluence.utils.reporting.reporting_manager.TimeseriesVisualizer') as mock:
        yield mock

@pytest.fixture
def mock_benchmark_visualizer():
    with patch('symfluence.utils.reporting.reporting_manager.BenchmarkVizualiser') as mock:
        yield mock

class TestReportingManager:

    def test_init(self, mock_config, mock_logger, mock_viz_reporter, mock_ts_visualizer, mock_benchmark_visualizer):
        """Test initialization of ReportingManager."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        
        assert manager.visualize is True
        assert manager.project_dir == Path('/tmp/symfluence_data/domain_test_domain')
        
        # Verify reporters are initialized
        mock_viz_reporter.assert_called_once_with(mock_config, mock_logger)
        mock_ts_visualizer.assert_called_once_with(mock_config, mock_logger)
        mock_benchmark_visualizer.assert_called_once_with(mock_config, mock_logger)

    def test_visualization_disabled(self, mock_config, mock_logger, mock_viz_reporter):
        """Test that methods return None/Empty when visualization is disabled."""
        manager = ReportingManager(mock_config, mock_logger, visualize=False)
        reporter_instance = mock_viz_reporter.return_value

        # Test various methods
        assert manager.visualize_domain() is None
        assert manager.visualize_discretized_domain('elevation') is None
        assert manager.visualize_model_outputs([], []) is None
        assert manager.visualize_lumped_model_outputs([], []) is None
        assert manager.visualize_fuse_outputs([], []) is None
        assert manager.visualize_benchmarks({}) == []
        assert manager.visualize_snow_comparison([]) == {}
        
        # Ensure underlying methods were NOT called
        reporter_instance.plot_domain.assert_not_called()
        reporter_instance.plot_discretized_domain.assert_not_called()

    def test_visualization_enabled(self, mock_config, mock_logger, mock_viz_reporter):
        """Test that methods call underlying reporters when visualization is enabled."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        reporter_instance = mock_viz_reporter.return_value

        # Setup return values
        reporter_instance.plot_domain.return_value = "plot_path.png"
        
        # Test visualize_domain
        result = manager.visualize_domain()
        assert result == "plot_path.png"
        reporter_instance.plot_domain.assert_called_once()

        # Test visualize_discretized_domain
        manager.visualize_discretized_domain('landclass')
        reporter_instance.plot_discretized_domain.assert_called_once_with('landclass')

    def test_model_output_visualization(self, mock_config, mock_logger, mock_viz_reporter):
        """Test model output visualization delegation."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        reporter_instance = mock_viz_reporter.return_value
        
        models = [('model', 'path')]
        obs = [('obs', 'path')]

        # Standard outputs
        manager.visualize_model_outputs(models, obs)
        reporter_instance.plot_streamflow_simulations_vs_observations.assert_called_once_with(models, obs)

        # Lumped outputs
        manager.visualize_lumped_model_outputs(models, obs)
        reporter_instance.plot_lumped_streamflow_simulations_vs_observations.assert_called_once_with(models, obs)

        # FUSE outputs
        manager.visualize_fuse_outputs(models, obs)
        reporter_instance.plot_fuse_streamflow_simulations_vs_observations.assert_called_once_with(models, obs)

    def test_timeseries_results_visualization(self, mock_config, mock_logger, mock_ts_visualizer):
        """Test timeseries visualization delegation."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        ts_instance = mock_ts_visualizer.return_value
        
        manager.visualize_timeseries_results()
        ts_instance.create_visualizations.assert_called_once()
        
        # Test error handling
        ts_instance.create_visualizations.side_effect = Exception("Viz Error")
        manager.visualize_timeseries_results() # Should not raise exception, but log error
        mock_logger.error.assert_called()

    def test_update_sim_reach_id_always_runs(self, mock_config, mock_logger, mock_viz_reporter):
        """Test that update_sim_reach_id runs regardless of visualization flag."""
        # Case 1: visualize=False
        manager_false = ReportingManager(mock_config, mock_logger, visualize=False)
        manager_false.update_sim_reach_id("config.yaml")
        mock_viz_reporter.return_value.update_sim_reach_id.assert_called_with("config.yaml")

        mock_viz_reporter.reset_mock()

        # Case 2: visualize=True
        manager_true = ReportingManager(mock_config, mock_logger, visualize=True)
        manager_true.update_sim_reach_id("config.yaml")
        mock_viz_reporter.return_value.update_sim_reach_id.assert_called_with("config.yaml")

    def test_visualize_benchmarks(self, mock_config, mock_logger, mock_benchmark_visualizer):
        """Test benchmark visualization delegation."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        bench_instance = mock_benchmark_visualizer.return_value
        
        bench_results = {'score': 100}
        bench_instance.visualize_benchmarks.return_value = ['plot1.png']
        
        result = manager.visualize_benchmarks(bench_results)
        
        assert result == ['plot1.png']
        bench_instance.visualize_benchmarks.assert_called_once_with(bench_results)

    def test_optimization_visualizations(self, mock_config, mock_logger):
        """Test optimization-related visualization methods."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        output_dir = Path('/tmp/opt')
        history = [{'generation': 1, 'best_score': 0.8}]
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            manager.visualize_optimization_progress(history, output_dir, 'flow', 'KGE')
            manager.visualize_optimization_depth_parameters(history, output_dir)
            # We don't verify full matplotlib calls here, just that they don't crash
            # and logic flow is correct (logging, path creation)

    def test_analysis_visualizations(self, mock_config, mock_logger):
        """Test analysis-related visualization methods."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('pandas.read_csv'):
            manager.visualize_sensitivity_analysis(MagicMock(), Path('sens.png'))
            manager.visualize_decision_impacts(Path('res.csv'), Path('out'))
            manager.visualize_drop_analysis([{'threshold': 10, 'mean_drop': 1}], 10, Path('proj'))

    def test_model_specific_visualizations(self, mock_config, mock_logger):
        """Test model-specific visualization methods."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'):
            # LSTM
            manager.visualize_lstm_results(MagicMock(), MagicMock(), MagicMock(), True, Path('out'), 'exp1')
            # HYPE
            manager.visualize_hype_results(MagicMock(), MagicMock(), '1', 'dom', 'exp1', Path('proj'))
            # NGen
            manager.visualize_ngen_results(MagicMock(), None, 'exp1', Path('res'))

    def test_hydrographs_with_highlight(self, mock_config, mock_logger):
        """Test hydrograph visualization with highlight."""
        manager = ReportingManager(mock_config, mock_logger, visualize=True)
        
        with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('pandas.read_csv') as mock_read:
            mock_df = MagicMock()
            mock_df.__getitem__.return_value.quantile.return_value = 0.9
            mock_read.return_value = mock_df
            
            manager.visualize_hydrographs_with_highlight(
                Path('res.csv'), {}, MagicMock(), {}, Path('out')
            )
