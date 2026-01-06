"""
Unit tests for GR preprocessor.

Tests GR-specific preprocessing functionality, including mode detection.
"""

import pytest
from unittest.mock import Mock, patch
from symfluence.core.config.models import SymfluenceConfig

# Patch imports before importing the module under test to handle optional dependencies
with patch.dict('sys.modules', {
    'rpy2': Mock(),
    'rpy2.robjects': Mock(),
    'rpy2.robjects.packages': Mock(),
    'rpy2.robjects.conversion': Mock(),
    'torch': Mock(),
}):
    # Mock HAS_RPY2 in the module
    with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
        from symfluence.models.gr.preprocessor import GRPreProcessor


class TestGRPreProcessorModeDetection:
    """Test GR preprocessor mode detection logic."""

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    @pytest.fixture
    def common_config_setup(self, tmp_path):
        """Setup common config mocks."""
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'FORCING_DATASET': 'test_forcing',
            'FORCING_TIME_STEP_SIZE': 86400,
        }
        
        config = Mock(spec=SymfluenceConfig)
        
        # Initialize nested mocks
        config.system = Mock()
        config.system.data_dir = tmp_path
        
        config.domain = Mock()
        config.domain.name = 'test_domain'
        
        config.forcing = Mock()
        config.forcing.dataset = 'test_forcing'
        config.forcing.time_step_size = 86400
        
        config.model = Mock()
        config.model.gr = Mock()
        
        return config, config_dict

    def test_explicit_config_lumped(self, mock_logger, common_config_setup):
        """Test that explicit configuration for lumped mode is respected."""
        config, config_dict = common_config_setup
        
        config_dict['DOMAIN_DEFINITION_METHOD'] = 'delineate'
        config_dict['GR_SPATIAL_MODE'] = 'lumped'
        
        config.domain.definition_method = 'delineate'
        config.model.gr.spatial_mode = 'lumped'
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == 'lumped'

    def test_explicit_config_distributed(self, mock_logger, common_config_setup):
        """Test that explicit configuration for distributed mode is respected."""
        config, config_dict = common_config_setup
        
        config_dict['DOMAIN_DEFINITION_METHOD'] = 'lumped'
        config_dict['GR_SPATIAL_MODE'] = 'distributed'
        
        config.domain.definition_method = 'lumped'
        config.model.gr.spatial_mode = 'distributed'
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == 'distributed'

    def test_implicit_config_delineate(self, mock_logger, common_config_setup):
        """Test that mode defaults to distributed when delineating."""
        config, config_dict = common_config_setup
        
        config_dict['DOMAIN_DEFINITION_METHOD'] = 'delineate'
        
        config.domain.definition_method = 'delineate'
        config.model.gr.spatial_mode = None
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == 'distributed'

    def test_implicit_config_lumped(self, mock_logger, common_config_setup):
        """Test that mode defaults to lumped when not delineating."""
        config, config_dict = common_config_setup
        
        config_dict['DOMAIN_DEFINITION_METHOD'] = 'lumped'
        
        config.domain.definition_method = 'lumped'
        config.model.gr.spatial_mode = None
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == 'lumped'
        
    def test_missing_gr_config(self, mock_logger, common_config_setup):
        """Test behavior when GR config section is missing/None."""
        config, config_dict = common_config_setup
        
        config_dict['DOMAIN_DEFINITION_METHOD'] = 'delineate'
        
        config.domain.definition_method = 'delineate'
        config.model.gr = None # Simulate missing GR config
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == 'distributed'
