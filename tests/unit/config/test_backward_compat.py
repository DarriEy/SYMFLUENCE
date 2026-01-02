"""
Unit tests for backward compatibility features.

Tests dict-like access methods (get, __getitem__, __contains__) and to_dict()
to ensure the new hierarchical config system maintains compatibility with
existing code expecting flat dictionaries.
"""

import pytest
from pathlib import Path
from symfluence.utils.config.models import SymfluenceConfig, SystemConfig, DomainConfig, ForcingConfig, ModelConfig


class TestDictLikeAccess:
    """Test dict-like access methods for backward compatibility"""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing"""
        return SymfluenceConfig(
            system=SystemConfig(
                SYMFLUENCE_DATA_DIR=Path('/data'),
                SYMFLUENCE_CODE_DIR=Path('/code'),
                MPI_PROCESSES=4,
                DEBUG_MODE=True
            ),
            domain=DomainConfig(
                DOMAIN_NAME='test_basin',
                EXPERIMENT_ID='run_1',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00',
                DOMAIN_DEFINITION_METHOD='lumped',
                DOMAIN_DISCRETIZATION='lumped'
            ),
            forcing=ForcingConfig(
                FORCING_DATASET='ERA5'
            ),
            model=ModelConfig(
                HYDROLOGICAL_MODEL='SUMMA'
            )
        )

    def test_getitem_basic_access(self, sample_config):
        """Test bracket access to config values"""
        assert sample_config['DOMAIN_NAME'] == 'test_basin'
        assert sample_config['EXPERIMENT_ID'] == 'run_1'
        assert sample_config['FORCING_DATASET'] == 'ERA5'
        assert sample_config['HYDROLOGICAL_MODEL'] == 'SUMMA'

    def test_getitem_system_fields(self, sample_config):
        """Test bracket access to system fields"""
        assert str(sample_config['SYMFLUENCE_DATA_DIR']) == '/data'
        assert str(sample_config['SYMFLUENCE_CODE_DIR']) == '/code'
        assert sample_config['MPI_PROCESSES'] == 4
        assert sample_config['DEBUG_MODE'] is True

    def test_getitem_raises_keyerror_for_missing(self, sample_config):
        """Test that bracket access raises KeyError for missing keys"""
        with pytest.raises(KeyError, match="NONEXISTENT_KEY"):
            _ = sample_config['NONEXISTENT_KEY']

    def test_get_with_default(self, sample_config):
        """Test get() method with default value"""
        assert sample_config.get('DOMAIN_NAME') == 'test_basin'
        assert sample_config.get('NONEXISTENT_KEY') is None
        assert sample_config.get('NONEXISTENT_KEY', 'default') == 'default'

    def test_get_without_default(self, sample_config):
        """Test get() method without default value"""
        assert sample_config.get('DOMAIN_NAME') == 'test_basin'
        assert sample_config.get('FORCING_DATASET') == 'ERA5'

    def test_contains(self, sample_config):
        """Test __contains__ (in operator)"""
        assert 'DOMAIN_NAME' in sample_config
        assert 'FORCING_DATASET' in sample_config
        assert 'NONEXISTENT_KEY' not in sample_config

    def test_path_conversion_to_string(self, sample_config):
        """Test that Path objects are converted to strings in dict access"""
        data_dir = sample_config['SYMFLUENCE_DATA_DIR']
        code_dir = sample_config['SYMFLUENCE_CODE_DIR']

        # Should be strings in flat dict
        assert isinstance(data_dir, str)
        assert isinstance(code_dir, str)


class TestToDictMethod:
    """Test to_dict() method for backward compatibility"""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing"""
        return SymfluenceConfig(
            system=SystemConfig(
                SYMFLUENCE_DATA_DIR=Path('/data'),
                SYMFLUENCE_CODE_DIR=Path('/code'),
                MPI_PROCESSES=4
            ),
            domain=DomainConfig(
                DOMAIN_NAME='test_basin',
                EXPERIMENT_ID='run_1',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00',
                DOMAIN_DEFINITION_METHOD='lumped',
                DOMAIN_DISCRETIZATION='lumped'
            ),
            forcing=ForcingConfig(
                FORCING_DATASET='ERA5'
            ),
            model=ModelConfig(
                HYDROLOGICAL_MODEL='SUMMA'
            )
        )

    def test_to_dict_flatten_true(self, sample_config):
        """Test to_dict(flatten=True) returns flat dict with uppercase keys"""
        flat = sample_config.to_dict(flatten=True)

        # Should be a flat dict
        assert isinstance(flat, dict)

        # Should have uppercase keys
        assert 'DOMAIN_NAME' in flat
        assert 'FORCING_DATASET' in flat
        assert 'EXPERIMENT_ID' in flat

        # Should have correct values
        assert flat['DOMAIN_NAME'] == 'test_basin'
        assert flat['FORCING_DATASET'] == 'ERA5'
        assert flat['MPI_PROCESSES'] == 4

        # Paths should be strings
        assert isinstance(flat['SYMFLUENCE_DATA_DIR'], str)
        assert flat['SYMFLUENCE_DATA_DIR'] == '/data'

    def test_to_dict_flatten_false(self, sample_config):
        """Test to_dict(flatten=False) returns nested structure"""
        nested = sample_config.to_dict(flatten=False)

        # Should be nested
        assert 'system' in nested
        assert 'domain' in nested
        assert 'forcing' in nested
        assert 'model' in nested

        # Should have lowercase keys
        assert 'name' in nested['domain']
        assert 'dataset' in nested['forcing']

        # Should have correct values
        assert nested['domain']['name'] == 'test_basin'
        assert nested['forcing']['dataset'] == 'ERA5'

    def test_to_dict_default_flatten(self, sample_config):
        """Test to_dict() defaults to flatten=True"""
        flat = sample_config.to_dict()

        # Should be flat by default
        assert 'DOMAIN_NAME' in flat
        assert flat['DOMAIN_NAME'] == 'test_basin'

    def test_round_trip_compatibility(self, sample_config):
        """Test that dict access and to_dict() return same values"""
        flat_from_to_dict = sample_config.to_dict(flatten=True)

        # Values from bracket access should match to_dict()
        assert sample_config['DOMAIN_NAME'] == flat_from_to_dict['DOMAIN_NAME']
        assert sample_config['FORCING_DATASET'] == flat_from_to_dict['FORCING_DATASET']
        assert sample_config['MPI_PROCESSES'] == flat_from_to_dict['MPI_PROCESSES']


class TestBackwardCompatibilityPatterns:
    """Test common usage patterns from legacy code"""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing"""
        return SymfluenceConfig(
            system=SystemConfig(
                SYMFLUENCE_DATA_DIR=Path('/data'),
                SYMFLUENCE_CODE_DIR=Path('/code'),
            ),
            domain=DomainConfig(
                DOMAIN_NAME='test_basin',
                EXPERIMENT_ID='run_1',
                EXPERIMENT_TIME_START='2020-01-01 00:00',
                EXPERIMENT_TIME_END='2020-12-31 23:00',
                DOMAIN_DEFINITION_METHOD='lumped',
                DOMAIN_DISCRETIZATION='lumped',
                POUR_POINT_COORDS='51.17/-115.57'
            ),
            forcing=ForcingConfig(
                FORCING_DATASET='ERA5'
            ),
            model=ModelConfig(
                HYDROLOGICAL_MODEL='SUMMA'
            )
        )

    def test_pattern_get_with_default(self, sample_config):
        """Test common pattern: config.get('KEY', 'default')"""
        # Existing keys
        domain_name = sample_config.get('DOMAIN_NAME', 'fallback')
        assert domain_name == 'test_basin'

        # Missing keys
        missing = sample_config.get('MISSING_KEY', 'fallback')
        assert missing == 'fallback'

    def test_pattern_bracket_access(self, sample_config):
        """Test common pattern: config['KEY']"""
        assert sample_config['DOMAIN_NAME'] == 'test_basin'
        assert sample_config['EXPERIMENT_ID'] == 'run_1'

    def test_pattern_path_conversion(self, sample_config):
        """Test common pattern: Path(config.get('PATH'))"""
        # Should work even though internal representation is Path
        data_dir = Path(sample_config.get('SYMFLUENCE_DATA_DIR'))
        assert data_dir == Path('/data')

    def test_pattern_conditional_access(self, sample_config):
        """Test common pattern: if 'KEY' in config: value = config['KEY']"""
        if 'POUR_POINT_COORDS' in sample_config:
            coords = sample_config['POUR_POINT_COORDS']
            assert coords == '51.17/-115.57'

        if 'MISSING_KEY' not in sample_config:
            # Should execute
            pass
        else:
            pytest.fail("Should not have found MISSING_KEY")

    def test_pattern_iterate_flat_dict(self, sample_config):
        """Test pattern: for key, value in config.items()"""
        flat = sample_config.to_dict(flatten=True)

        # Should be able to iterate
        assert 'DOMAIN_NAME' in flat.keys()
        assert 'test_basin' in flat.values()
        assert ('DOMAIN_NAME', 'test_basin') in flat.items()
