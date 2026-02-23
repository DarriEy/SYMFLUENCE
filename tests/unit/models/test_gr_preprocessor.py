"""
Unit tests for GR preprocessor.

Tests GR-specific preprocessing functionality, including mode detection.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from importlib.machinery import ModuleSpec
from symfluence.core.config.models import SymfluenceConfig

# Pre-import jax at collection time to avoid abseil timezone conflict that
# causes SIGABRT when jaxlib is loaded lazily during test execution (macOS).
# The GRPreProcessor constructor triggers model registry imports which
# transitively load jax through the optimization workers.
try:
    import jax  # noqa: F401
except Exception:
    pass

# Patch rpy2 imports before importing the module under test to handle optional
# dependencies.  We give the rpy2 mock a valid __spec__ so that
# ``importlib.util.find_spec("rpy2")`` succeeds inside the preprocessor module
# (which sets HAS_RPY2 = find_spec("rpy2") is not None).
# NOTE: Do NOT mock ``torch`` at sys.modules level -- scipy's
# ``is_torch_array()`` uses ``issubclass(cls, torch.Tensor)`` which fails when
# ``torch.Tensor`` is a Mock instead of a real class.
_rpy2_mock = MagicMock()
_rpy2_mock.__spec__ = ModuleSpec('rpy2', None)

with patch.dict('sys.modules', {
    'rpy2': _rpy2_mock,
    'rpy2.robjects': MagicMock(),
    'rpy2.robjects.packages': MagicMock(),
    'rpy2.robjects.conversion': MagicMock(),
}):
    from symfluence.models.gr.preprocessor import GRPreProcessor

from symfluence.models.spatial_modes import SpatialMode


class TestGRPreProcessorModeDetection:
    """Test GR preprocessor mode detection logic."""

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    @pytest.fixture
    def common_config_setup(self, tmp_path):
        """Setup common config mocks.

        The Mock(spec=SymfluenceConfig) passes isinstance checks in
        coerce_config(), so the mock is used as-is.  We must explicitly set
        every nested attribute that the constructor accesses -- any unset
        attribute on an unspecced child Mock returns a new Mock, which then
        fails when used in Path operations or string comparisons.
        """
        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'run_1',
            'SUB_GRID_DISCRETIZATION': 'lumped',
            'FORCING_DATASET': 'test_forcing',
            'FORCING_TIME_STEP_SIZE': 86400,
        }

        config = Mock(spec=SymfluenceConfig)

        # Initialize nested mocks with concrete values for all attributes
        # accessed during GRPreProcessor.__init__
        config.system = Mock()
        config.system.data_dir = tmp_path

        config.domain = Mock()
        config.domain.name = 'test_domain'
        config.domain.discretization = 'lumped'
        config.domain.experiment_id = 'run_1'
        config.domain.definition_method = 'lumped'

        config.forcing = Mock()
        config.forcing.dataset = 'test_forcing'
        config.forcing.time_step_size = 86400

        config.model = Mock()
        config.model.gr = Mock()
        config.model.routing_model = 'none'

        config.paths = Mock()
        # _get_catchment_file_path() accesses config.paths.catchment_name
        # (not catchment_shp_name).  Must be 'default' or None so the code
        # constructs a name from domain_name + discretization instead of
        # trying to use a Mock as a path component.
        config.paths.catchment_name = 'default'
        config.paths.catchment_shp_name = 'default'

        # Properly mock to_dict to handle flatten parameter
        config.to_dict = Mock(return_value=config_dict)
        config.to_dict.side_effect = lambda flatten=False: config_dict

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
            assert preprocessor.spatial_mode == SpatialMode.LUMPED

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
            assert preprocessor.spatial_mode == SpatialMode.DISTRIBUTED

    def test_implicit_config_delineate(self, mock_logger, common_config_setup):
        """Test that mode defaults to distributed when delineating."""
        config, config_dict = common_config_setup

        config_dict['DOMAIN_DEFINITION_METHOD'] = 'delineate'

        config.domain.definition_method = 'delineate'
        config.model.gr.spatial_mode = None
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == SpatialMode.DISTRIBUTED

    def test_implicit_config_lumped(self, mock_logger, common_config_setup):
        """Test that mode defaults to lumped when not delineating."""
        config, config_dict = common_config_setup

        config_dict['DOMAIN_DEFINITION_METHOD'] = 'lumped'

        config.domain.definition_method = 'lumped'
        config.model.gr.spatial_mode = None
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == SpatialMode.LUMPED

    def test_missing_gr_config(self, mock_logger, common_config_setup):
        """Test behavior when GR config section is missing/None."""
        config, config_dict = common_config_setup

        config_dict['DOMAIN_DEFINITION_METHOD'] = 'delineate'

        config.domain.definition_method = 'delineate'
        config.model.gr = None  # Simulate missing GR config
        config.to_dict.return_value = config_dict

        with patch('symfluence.models.gr.preprocessor.HAS_RPY2', True):
            preprocessor = GRPreProcessor(config, mock_logger)
            assert preprocessor.spatial_mode == SpatialMode.DISTRIBUTED
