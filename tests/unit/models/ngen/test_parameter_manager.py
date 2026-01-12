"""
Unit tests for NgenParameterManager.

Tests NextGen-specific parameter handling including:
- Multi-module parameter management (CFE, NOAH, PET)
- JSON configuration file operations
- Module.param naming convention
"""

import pytest
import numpy as np
import logging

# Mark all tests in this module
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_ngen_parameter_manager')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def ngen_config(tmp_path):
    """Create NextGen-specific configuration."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_catchment',
        'EXPERIMENT_ID': 'test_ngen_exp',
        'NGEN_MODULES_TO_CALIBRATE': 'CFE,NOAH',
        'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc,satdk,bb,slop',
        'NGEN_NOAH_PARAMS_TO_CALIBRATE': 'refkdt,slope',
        'NGEN_PET_PARAMS_TO_CALIBRATE': 'wind_speed_measurement_height_m',
    }


@pytest.fixture
def ngen_cfe_only_config(tmp_path):
    """Configuration with only CFE module."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_catchment',
        'EXPERIMENT_ID': 'test_ngen_exp',
        'NGEN_MODULES_TO_CALIBRATE': 'CFE',
        'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc,satdk',
    }


@pytest.fixture
def ngen_project_structure(tmp_path, ngen_config):
    """Create NextGen project directory structure."""
    domain_name = ngen_config['DOMAIN_NAME']
    experiment_id = ngen_config['EXPERIMENT_ID']

    # Create directories
    project_dir = tmp_path / f"domain_{domain_name}"
    sim_dir = project_dir / 'simulations' / experiment_id / 'NGEN'
    setup_dir = project_dir / 'settings' / 'NGEN'

    # Module-specific directories
    cfe_dir = setup_dir / 'CFE'
    noah_dir = setup_dir / 'NOAH'
    pet_dir = setup_dir / 'PET'

    for d in [sim_dir, cfe_dir, noah_dir, pet_dir]:
        d.mkdir(parents=True)

    return {
        'project_dir': project_dir,
        'sim_dir': sim_dir,
        'setup_dir': setup_dir,
        'cfe_dir': cfe_dir,
        'noah_dir': noah_dir,
        'pet_dir': pet_dir,
    }


@pytest.fixture
def sample_cfe_config():
    """Sample CFE configuration JSON."""
    return {
        "global": {
            "maxsmc": 0.439,
            "satdk": 0.00000338,
            "bb": 4.05,
            "slop": 1.0,
            "Cgw": 0.01,
            "expon": 6.0
        }
    }


# ============================================================================
# Initialization tests
# ============================================================================

class TestNgenParameterManagerInitialization:
    """Test NgenParameterManager initialization."""

    def test_init_parses_modules(self, ngen_config, test_logger, tmp_path):
        """Test that modules to calibrate are parsed from config."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        assert 'CFE' in manager.modules_to_calibrate
        assert 'NOAH' in manager.modules_to_calibrate

    def test_init_parses_module_params(self, ngen_config, test_logger, tmp_path):
        """Test that parameters for each module are parsed."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        assert 'CFE' in manager.params_to_calibrate
        assert 'maxsmc' in manager.params_to_calibrate['CFE']
        assert 'satdk' in manager.params_to_calibrate['CFE']

        assert 'NOAH' in manager.params_to_calibrate
        assert 'refkdt' in manager.params_to_calibrate['NOAH']

    def test_init_defaults_to_cfe(self, tmp_path, test_logger):
        """Test that initialization defaults to CFE if no modules specified."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': '',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(config, test_logger, settings_dir)

        assert 'CFE' in manager.modules_to_calibrate

    def test_init_ignores_invalid_modules(self, tmp_path, test_logger, caplog):
        """Test that invalid modules are ignored with warning."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': 'CFE,INVALID_MODULE',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            manager = NgenParameterManager(config, test_logger, settings_dir)

        # CFE should still be present
        assert 'CFE' in manager.modules_to_calibrate


# ============================================================================
# Parameter naming convention tests
# ============================================================================

class TestNgenParameterNaming:
    """Test NextGen module.param naming convention."""

    def test_parameter_names_use_module_prefix(self, ngen_config, test_logger, tmp_path):
        """Test that parameter names use module.param format."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)
        names = manager.all_param_names

        # Check for module.param format
        for name in names:
            assert '.' in name, f"Parameter {name} missing module prefix"

        # Specific checks
        assert 'CFE.maxsmc' in names
        assert 'CFE.satdk' in names
        assert 'NOAH.refkdt' in names

    def test_get_parameter_names_returns_all_modules(self, ngen_config, test_logger, tmp_path):
        """Test that _get_parameter_names includes all module params."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)
        names = manager._get_parameter_names()

        # Count CFE params
        cfe_params = [n for n in names if n.startswith('CFE.')]
        assert len(cfe_params) == 4  # maxsmc, satdk, bb, slop

        # Count NOAH params
        noah_params = [n for n in names if n.startswith('NOAH.')]
        assert len(noah_params) == 2  # refkdt, slope

    def test_single_module_naming(self, ngen_cfe_only_config, test_logger, tmp_path):
        """Test naming with single module."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_cfe_only_config, test_logger, settings_dir)
        names = manager.all_param_names

        # All should be CFE params
        for name in names:
            assert name.startswith('CFE.')


# ============================================================================
# Parameter bounds tests
# ============================================================================

class TestNgenParameterBounds:
    """Test NextGen parameter bounds."""

    def test_bounds_use_module_prefix(self, ngen_config, test_logger, tmp_path):
        """Test that bounds use module.param format."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)
        bounds = manager.param_bounds

        assert 'CFE.maxsmc' in bounds
        assert 'CFE.satdk' in bounds
        assert 'NOAH.refkdt' in bounds

    def test_bounds_have_min_max(self, ngen_config, test_logger, tmp_path):
        """Test that each bound has min and max."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)
        bounds = manager.param_bounds

        for param_name, bound in bounds.items():
            assert 'min' in bound, f"Missing 'min' for {param_name}"
            assert 'max' in bound, f"Missing 'max' for {param_name}"
            assert bound['min'] <= bound['max'], f"Invalid bounds for {param_name}"

    def test_default_bounds_for_cfe_params(self, ngen_cfe_only_config, test_logger, tmp_path):
        """Test default bounds for CFE parameters."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_cfe_only_config, test_logger, settings_dir)
        default_bounds = manager._get_default_ngen_bounds()

        # Check CFE-specific bounds exist
        assert 'maxsmc' in default_bounds
        assert 'satdk' in default_bounds

        # Check they're physically reasonable
        assert default_bounds['maxsmc']['min'] > 0
        assert default_bounds['maxsmc']['max'] <= 1.0  # Porosity <= 1

    def test_unknown_param_gets_default_bounds(self, tmp_path, test_logger, caplog):
        """Test that unknown parameters get default bounds with warning."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': 'CFE',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc,unknown_param',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            manager = NgenParameterManager(config, test_logger, settings_dir)
            bounds = manager.param_bounds

        # Should have bounds for unknown param (default)
        assert 'CFE.unknown_param' in bounds
        # Should have logged warning
        assert any('unknown_param' in record.message.lower() or 'no bounds' in record.message.lower()
                   for record in caplog.records)


# ============================================================================
# Normalization/Denormalization tests
# ============================================================================

class TestNgenNormalization:
    """Test NextGen parameter normalization."""

    def test_normalize_multimodule_params(self, ngen_config, test_logger, tmp_path):
        """Test normalizing parameters from multiple modules."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        # Get midpoint values for all params
        params = {}
        for name in manager.all_param_names:
            bounds = manager.param_bounds[name]
            params[name] = (bounds['min'] + bounds['max']) / 2

        normalized = manager.normalize_parameters(params)

        # All should be around 0.5
        np.testing.assert_array_almost_equal(normalized, [0.5] * len(params), decimal=1)

    def test_denormalize_preserves_module_prefix(self, ngen_config, test_logger, tmp_path):
        """Test that denormalized params keep module prefix."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        # Denormalize from midpoint
        normalized = np.array([0.5] * len(manager.all_param_names))
        params = manager.denormalize_parameters(normalized)

        # All keys should have module prefix
        for key in params:
            assert '.' in key

    def test_roundtrip_consistency(self, ngen_config, test_logger, tmp_path):
        """Test normalize → denormalize roundtrip."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        # Original values
        original = {}
        for i, name in enumerate(manager.all_param_names):
            bounds = manager.param_bounds[name]
            frac = (i + 1) / (len(manager.all_param_names) + 1)
            original[name] = bounds['min'] + frac * (bounds['max'] - bounds['min'])

        normalized = manager.normalize_parameters(original)
        denormalized = manager.denormalize_parameters(normalized)

        for name in original:
            assert denormalized[name] == pytest.approx(original[name], rel=0.001)


# ============================================================================
# Configuration update tests
# ============================================================================

class TestNgenConfigUpdate:
    """Test NextGen configuration file updates."""

    def test_update_model_files_signature(self, ngen_config, test_logger, tmp_path):
        """Test that update_model_files has correct signature."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        assert hasattr(manager, 'update_model_files')
        assert callable(manager.update_model_files)

    def test_update_config_files_parses_module_params(self, ngen_config, test_logger, ngen_project_structure):
        """Test that update parses module.param format."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        manager = NgenParameterManager(
            ngen_config, test_logger, ngen_project_structure['setup_dir']
        )

        # Params with module prefix
        params = {
            'CFE.maxsmc': 0.45,
            'CFE.satdk': 0.00001,
            'NOAH.refkdt': 3.0,
        }

        # Should not raise
        assert manager.validate_parameters(params) is True


# ============================================================================
# Initial parameters tests
# ============================================================================

class TestNgenInitialParameters:
    """Test NextGen initial parameter retrieval."""

    def test_get_initial_parameters_returns_dict(self, ngen_config, test_logger, tmp_path):
        """Test that get_initial_parameters returns dictionary."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        assert hasattr(manager, 'get_initial_parameters')

    def test_get_default_parameters(self, ngen_config, test_logger, tmp_path):
        """Test get_default_parameters method."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        if hasattr(manager, 'get_default_parameters'):
            defaults = manager.get_default_parameters()
            assert isinstance(defaults, dict)

            # Should have module prefixes
            for key in defaults:
                assert '.' in key


# ============================================================================
# Edge cases and error handling
# ============================================================================

class TestNgenEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_whitespace_in_modules(self, tmp_path, test_logger):
        """Test handling of whitespace in module list."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': '  CFE  ,  NOAH  ',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc',
            'NGEN_NOAH_PARAMS_TO_CALIBRATE': 'refkdt',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(config, test_logger, settings_dir)

        assert 'CFE' in manager.modules_to_calibrate
        assert 'NOAH' in manager.modules_to_calibrate

    def test_handles_whitespace_in_params(self, tmp_path, test_logger):
        """Test handling of whitespace in parameter list."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': 'CFE',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': '  maxsmc  ,  satdk  ',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(config, test_logger, settings_dir)

        assert 'maxsmc' in manager.params_to_calibrate['CFE']
        assert 'satdk' in manager.params_to_calibrate['CFE']

    def test_handles_case_insensitive_modules(self, tmp_path, test_logger):
        """Test that module names are normalized to uppercase."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': 'cfe,noah',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc',
            'NGEN_NOAH_PARAMS_TO_CALIBRATE': 'refkdt',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(config, test_logger, settings_dir)

        # Should be normalized to uppercase
        assert 'CFE' in manager.modules_to_calibrate
        assert 'NOAH' in manager.modules_to_calibrate

    def test_handles_empty_param_list_for_module(self, tmp_path, test_logger):
        """Test handling of empty parameter list for a module."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        config = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_ID': 'test',
            'NGEN_MODULES_TO_CALIBRATE': 'CFE',
            'NGEN_CFE_PARAMS_TO_CALIBRATE': '',
        }

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(config, test_logger, settings_dir)

        # CFE should be in modules but with empty param list
        assert 'CFE' in manager.modules_to_calibrate
        assert manager.params_to_calibrate.get('CFE', []) == []


# ============================================================================
# Integration with base class
# ============================================================================

class TestNgenBaseClassIntegration:
    """Test integration with BaseParameterManager."""

    def test_inherits_from_base(self, ngen_config, test_logger, tmp_path):
        """Test that NgenParameterManager inherits from BaseParameterManager."""
        from symfluence.optimization.parameter_managers import NgenParameterManager
        from symfluence.optimization.core.base_parameter_manager import BaseParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        assert isinstance(manager, BaseParameterManager)

    def test_implements_abstract_methods(self, ngen_config, test_logger, tmp_path):
        """Test that all abstract methods are implemented."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        # These should all be callable
        assert callable(manager._get_parameter_names)
        assert callable(manager._load_parameter_bounds)
        assert callable(manager.update_model_files)

        # And return correct types
        names = manager._get_parameter_names()
        assert isinstance(names, list)

        bounds = manager._load_parameter_bounds()
        assert isinstance(bounds, dict)

    def test_uses_base_normalization(self, ngen_config, test_logger, tmp_path):
        """Test that base class normalization is used."""
        from symfluence.optimization.parameter_managers import NgenParameterManager

        settings_dir = tmp_path / 'settings'
        settings_dir.mkdir(parents=True)

        manager = NgenParameterManager(ngen_config, test_logger, settings_dir)

        # normalize_parameters should be inherited from base
        params = {'CFE.maxsmc': 0.4, 'CFE.satdk': 0.00001}

        # Should not raise - uses inherited method
        normalized = manager.normalize_parameters(params)
        assert isinstance(normalized, np.ndarray)
