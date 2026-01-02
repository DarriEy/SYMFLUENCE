"""
Unit tests for configuration transformation utilities.

Tests the flat-to-nested and nested-to-flat transformations that enable
backward compatibility between the old flat config format and the new
hierarchical format.
"""

import pytest
from pathlib import Path
from symfluence.utils.config.transformers import (
    transform_flat_to_nested,
    flatten_nested_config,
    FLAT_TO_NESTED_MAP
)


class TestTransformFlatToNested:
    """Test transformation from flat dict to nested structure"""

    def test_basic_system_config(self):
        """Test basic system configuration transformation"""
        flat = {
            'SYMFLUENCE_DATA_DIR': '/path/to/data',
            'SYMFLUENCE_CODE_DIR': '/path/to/code',
            'MPI_PROCESSES': 4,
            'DEBUG_MODE': True
        }

        nested = transform_flat_to_nested(flat)

        assert nested['system']['data_dir'] == '/path/to/data'
        assert nested['system']['code_dir'] == '/path/to/code'
        assert nested['system']['mpi_processes'] == 4
        assert nested['system']['debug_mode'] is True

    def test_basic_domain_config(self):
        """Test basic domain configuration transformation"""
        flat = {
            'DOMAIN_NAME': 'test_basin',
            'EXPERIMENT_ID': 'run_1',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2020-12-31 23:00',
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'DOMAIN_DISCRETIZATION': 'lumped'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['domain']['name'] == 'test_basin'
        assert nested['domain']['experiment_id'] == 'run_1'
        assert nested['domain']['time_start'] == '2020-01-01 00:00'
        assert nested['domain']['time_end'] == '2020-12-31 23:00'
        assert nested['domain']['definition_method'] == 'lumped'
        assert nested['domain']['discretization'] == 'lumped'

    def test_nested_delineation_config(self):
        """Test nested delineation configuration transformation"""
        flat = {
            'ROUTING_DELINEATION': 'distributed',
            'STREAM_THRESHOLD': 1000.0,
            'DELINEATION_METHOD': 'stream_threshold',
            'USE_DROP_ANALYSIS': True
        }

        nested = transform_flat_to_nested(flat)

        assert nested['domain']['delineation']['routing'] == 'distributed'
        assert nested['domain']['delineation']['stream_threshold'] == 1000.0
        assert nested['domain']['delineation']['method'] == 'stream_threshold'
        assert nested['domain']['delineation']['use_drop_analysis'] is True

    def test_forcing_config(self):
        """Test forcing configuration transformation"""
        flat = {
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'PET_METHOD': 'oudin',
            'SUPPLEMENT_FORCING': False
        }

        nested = transform_flat_to_nested(flat)

        assert nested['forcing']['dataset'] == 'ERA5'
        assert nested['forcing']['time_step_size'] == 3600
        assert nested['forcing']['pet_method'] == 'oudin'
        assert nested['forcing']['supplement'] is False

    def test_nex_forcing_config(self):
        """Test NEX forcing configuration transformation"""
        flat = {
            'NEX_MODELS': ['model1', 'model2'],
            'NEX_SCENARIOS': ['ssp245', 'ssp585'],
            'NEX_VARIABLES': ['pr', 'tas']
        }

        nested = transform_flat_to_nested(flat)

        assert nested['forcing']['nex']['models'] == ['model1', 'model2']
        assert nested['forcing']['nex']['scenarios'] == ['ssp245', 'ssp585']
        assert nested['forcing']['nex']['variables'] == ['pr', 'tas']

    def test_model_config(self):
        """Test model configuration transformation"""
        flat = {
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'ROUTING_MODEL': 'mizuRoute',
            'SUMMA_EXE': 'summa_sundials.exe',
            'SETTINGS_SUMMA_PATH': '/path/to/summa',
            'FUSE_SPATIAL_MODE': 'distributed'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['model']['hydrological_model'] == 'SUMMA'
        assert nested['model']['routing_model'] == 'mizuRoute'
        assert nested['model']['summa']['exe'] == 'summa_sundials.exe'
        assert nested['model']['summa']['settings_path'] == '/path/to/summa'
        assert nested['model']['fuse']['spatial_mode'] == 'distributed'

    def test_optimization_config(self):
        """Test optimization configuration transformation"""
        flat = {
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'PSO',
            'OPTIMIZATION_METRIC': 'KGE',
            'NUMBER_OF_ITERATIONS': 1000,
            'PSO_COGNITIVE_PARAM': 1.5,
            'PSO_SOCIAL_PARAM': 1.5
        }

        nested = transform_flat_to_nested(flat)

        assert nested['optimization']['methods'] == ['iteration']
        assert nested['optimization']['algorithm'] == 'PSO'
        assert nested['optimization']['metric'] == 'KGE'
        assert nested['optimization']['iterations'] == 1000
        assert nested['optimization']['pso']['cognitive_param'] == 1.5
        assert nested['optimization']['pso']['social_param'] == 1.5

    def test_evaluation_config(self):
        """Test evaluation configuration transformation"""
        flat = {
            'EVALUATION_DATA': ['streamflow'],
            'STREAMFLOW_DATA_PROVIDER': 'USGS',
            'DOWNLOAD_USGS_DATA': True,
            'STATION_ID': '12345678'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['evaluation']['evaluation_data'] == ['streamflow']
        assert nested['evaluation']['streamflow']['data_provider'] == 'USGS'
        assert nested['evaluation']['streamflow']['download_usgs'] is True
        assert nested['evaluation']['streamflow']['station_id'] == '12345678'

    def test_paths_config(self):
        """Test paths configuration transformation"""
        flat = {
            'CATCHMENT_PATH': '/path/to/catchment',
            'FORCING_PATH': '/path/to/forcing',
            'DATATOOL_PATH': '/path/to/datatool',
            'RIVER_NETWORK_SHP_SEGID': 'LINKNO'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['paths']['catchment_path'] == '/path/to/catchment'
        assert nested['paths']['forcing_path'] == '/path/to/forcing'
        assert nested['paths']['datatool_path'] == '/path/to/datatool'
        assert nested['paths']['river_network_segid'] == 'LINKNO'

    def test_unknown_keys_stored_in_extra(self):
        """Test that unknown keys are stored in _extra"""
        flat = {
            'DOMAIN_NAME': 'test',
            'UNKNOWN_KEY_1': 'value1',
            'UNKNOWN_KEY_2': 'value2'
        }

        nested = transform_flat_to_nested(flat)

        assert nested['domain']['name'] == 'test'
        assert nested['_extra']['UNKNOWN_KEY_1'] == 'value1'
        assert nested['_extra']['UNKNOWN_KEY_2'] == 'value2'

    def test_empty_dict(self):
        """Test transformation of empty dict"""
        flat = {}
        nested = transform_flat_to_nested(flat)

        # Should have all top-level sections
        assert 'system' in nested
        assert 'domain' in nested
        assert 'forcing' in nested
        assert 'model' in nested
        assert 'optimization' in nested
        assert 'evaluation' in nested
        assert 'paths' in nested

    def test_comprehensive_transformation(self):
        """Test comprehensive transformation with many fields"""
        flat = {
            # System
            'SYMFLUENCE_DATA_DIR': '/data',
            'MPI_PROCESSES': 8,
            # Domain
            'DOMAIN_NAME': 'test',
            'EXPERIMENT_TIME_START': '2020-01-01 00:00',
            'EXPERIMENT_TIME_END': '2021-01-01 00:00',
            'DOMAIN_DEFINITION_METHOD': 'distributed',
            'DOMAIN_DISCRETIZATION': 'GRUs',
            'STREAM_THRESHOLD': 5000.0,
            # Forcing
            'FORCING_DATASET': 'ERA5',
            'PET_METHOD': 'oudin',
            # Model
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'SUMMA_EXE': 'summa.exe',
            # Optimization
            'OPTIMIZATION_METHODS': ['iteration'],
            'NUMBER_OF_ITERATIONS': 500,
            # Paths
            'FORCING_PATH': '/forcing'
        }

        nested = transform_flat_to_nested(flat)

        # Verify structure
        assert isinstance(nested['system'], dict)
        assert isinstance(nested['domain'], dict)
        assert isinstance(nested['forcing'], dict)
        assert isinstance(nested['model'], dict)

        # Verify values
        assert nested['system']['data_dir'] == '/data'
        assert nested['domain']['name'] == 'test'
        assert nested['forcing']['dataset'] == 'ERA5'
        assert nested['model']['hydrological_model'] == 'SUMMA'
        assert nested['optimization']['methods'] == ['iteration']


class TestFlatteningMapping:
    """Test the FLAT_TO_NESTED_MAP completeness"""

    def test_mapping_has_required_fields(self):
        """Test that mapping includes all required fields"""
        required_fields = [
            'SYMFLUENCE_DATA_DIR',
            'SYMFLUENCE_CODE_DIR',
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START',
            'EXPERIMENT_TIME_END',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION',
            'HYDROLOGICAL_MODEL',
            'FORCING_DATASET'
        ]

        for field in required_fields:
            assert field in FLAT_TO_NESTED_MAP, f"Required field {field} missing from mapping"

    def test_mapping_has_no_duplicates(self):
        """Test that mapping has no duplicate target paths"""
        paths = list(FLAT_TO_NESTED_MAP.values())
        unique_paths = set(paths)

        assert len(paths) == len(unique_paths), "Duplicate target paths found in mapping"

    def test_all_paths_are_tuples(self):
        """Test that all mapping values are tuples"""
        for key, path in FLAT_TO_NESTED_MAP.items():
            assert isinstance(path, tuple), f"Path for {key} is not a tuple: {path}"
            assert len(path) >= 2, f"Path for {key} is too short: {path}"

    def test_section_names_are_valid(self):
        """Test that all section names are valid"""
        valid_sections = {'system', 'domain', 'forcing', 'model', 'optimization', 'evaluation', 'paths'}

        for key, path in FLAT_TO_NESTED_MAP.items():
            section = path[0]
            assert section in valid_sections, f"Invalid section '{section}' for key {key}"


# Note: flatten_nested_config tests require actual SymfluenceConfig instances
# These will be tested in test_backward_compat.py
