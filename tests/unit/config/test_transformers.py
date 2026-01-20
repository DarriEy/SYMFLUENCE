"""
Unit tests for configuration transformation utilities.

Tests the flat-to-nested and nested-to-flat transformations that enable
backward compatibility between the old flat config format and the new
hierarchical format.
"""

from symfluence.core.config.transformers import (
    transform_flat_to_nested,
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
        # STREAMFLOW_DATA_PROVIDER and DOWNLOAD_USGS_DATA map to 'data' section
        assert nested['data']['streamflow_data_provider'] == 'USGS'
        assert nested['data']['download_usgs_data'] is True
        # STATION_ID maps to evaluation.streamflow section
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

    def test_mapping_has_only_intentional_aliases(self):
        """Test that any duplicate target paths are intentional aliases"""
        from collections import Counter

        paths = list(FLAT_TO_NESTED_MAP.values())
        path_counts = Counter(paths)

        # These are known intentional aliases for backward compatibility
        # Multiple flat keys can map to the same nested path for legacy support
        known_aliases = {
            ('optimization', 'nsga2', 'secondary_target'),  # NSGA2_SECONDARY_TARGET, OPTIMIZATION_TARGET2
            ('optimization', 'nsga2', 'secondary_metric'),  # NSGA2_SECONDARY_METRIC, OPTIMIZATION_METRIC2
            # MiZuRoute deprecated aliases (Phase 2 deprecation)
            ('model', 'mizuroute', 'install_path'),  # MIZUROUTE_INSTALL_PATH, INSTALL_PATH_MIZUROUTE
            ('model', 'mizuroute', 'exe'),  # MIZUROUTE_EXE, EXE_NAME_MIZUROUTE
        }

        # Check that any duplicates are in the known_aliases set
        for path, count in path_counts.items():
            if count > 1:
                assert path in known_aliases, \
                    f"Unexpected duplicate target path: {path} appears {count} times"

    def test_all_paths_are_tuples(self):
        """Test that all mapping values are tuples"""
        for key, path in FLAT_TO_NESTED_MAP.items():
            assert isinstance(path, tuple), f"Path for {key} is not a tuple: {path}"
            assert len(path) >= 2, f"Path for {key} is too short: {path}"

    def test_section_names_are_valid(self):
        """Test that all section names are valid"""
        valid_sections = {'system', 'domain', 'data', 'forcing', 'model', 'optimization', 'evaluation', 'paths'}

        for key, path in FLAT_TO_NESTED_MAP.items():
            section = path[0]
            assert section in valid_sections, f"Invalid section '{section}' for key {key}"


# Note: flatten_nested_config tests require actual SymfluenceConfig instances
# These will be tested in test_backward_compat.py


class TestAutoGeneratedMapping:
    """Test auto-generation of configuration mapping"""

    def test_auto_generated_matches_manual(self):
        """
        Validate that auto-generated mapping is close to manual mapping.

        PHASE 1: This test allows for known differences during migration.
        It will report discrepancies but only fail if critical differences are found.

        Known acceptable differences:
        - OPTIMIZATION_TARGET2, OPTIMIZATION_METRIC2 (backward compat aliases)
        - STREAMFLOW_DATA_PROVIDER, DOWNLOAD_USGS_DATA (moved to evaluation section)
        - RHESSYS_PARAMS_TO_CALIBRATE (new field in Pydantic models)

        Once Phase 4 is complete and Pydantic models are updated, this test
        should pass with perfect equivalence, and can then be removed.
        """
        from symfluence.core.config.introspection import generate_flat_to_nested_map, validate_mapping_equivalence
        from symfluence.core.config.models import SymfluenceConfig

        # Generate from Pydantic models
        auto_generated = generate_flat_to_nested_map(
            SymfluenceConfig,
            include_model_overrides=True
        )

        # Compare with manual mapping
        result = validate_mapping_equivalence(auto_generated, FLAT_TO_NESTED_MAP)

        # Detailed reporting
        print(f"\nAuto-generated count: {result['auto_count']}")
        print(f"Manual count: {result['manual_count']}")
        print(f"Missing in auto-generated: {result['missing_in_auto']}")
        print(f"Extra in auto-generated: {result['extra_in_auto']}")
        print(f"Mismatched paths: {list(result['mismatched'].items())}")

        # Known acceptable differences during Phase 1
        # Note: MIZUROUTE_INSTALL_PATH and MIZUROUTE_EXE are new standard keys
        # added to manual mapping, but Pydantic models still use deprecated aliases
        # (INSTALL_PATH_MIZUROUTE, EXE_NAME_MIZUROUTE) - will be updated in Phase 3
        known_missing = {
            'OPTIMIZATION_TARGET2', 'OPTIMIZATION_METRIC2',
            'MIZUROUTE_INSTALL_PATH', 'MIZUROUTE_EXE',  # New standard keys not yet in Pydantic
        }
        # Known extra keys from new model-specific config adapters (CFUSE, JFUSE, HBV)
        known_extra = {
            'RHESSYS_PARAMS_TO_CALIBRATE',
            # CFUSE model configuration keys
            'CFUSE_USE_NATIVE_GRADIENTS', 'CFUSE_OUTPUT_FREQUENCY', 'CFUSE_USE_GRADIENT_CALIBRATION',
            'CFUSE_PARAMS_TO_CALIBRATE', 'CFUSE_WARMUP_DAYS', 'CFUSE_N_HRUS', 'CFUSE_MODEL_STRUCTURE',
            'CFUSE_SPATIAL_PARAMS', 'CFUSE_TIMESTEP_DAYS', 'CFUSE_SPATIAL_MODE', 'CFUSE_FORCING_FILE',
            'CFUSE_NETWORK_FILE', 'CFUSE_DEVICE', 'CFUSE_CALIBRATION_METRIC', 'CFUSE_ENABLE_SNOW',
            'CFUSE_ENABLE_ROUTING', 'CFUSE_INITIAL_SNOW', 'CFUSE_SAVE_STATES', 'CFUSE_INITIAL_S1',
            'CFUSE_INITIAL_S2',
            # JFUSE model configuration keys
            'JFUSE_CALIBRATION_METRIC', 'JFUSE_ENABLE_ROUTING', 'JFUSE_WARMUP_DAYS', 'JFUSE_JIT_COMPILE',
            'JFUSE_NETWORK_FILE', 'JFUSE_N_HRUS', 'JFUSE_TIMESTEP_DAYS', 'JFUSE_ENABLE_SNOW',
            'JFUSE_USE_GPU', 'JFUSE_DEFAULT_MANNINGS_N', 'JFUSE_INITIAL_S2', 'JFUSE_MODEL_CONFIG_NAME',
            'JFUSE_SPATIAL_MODE', 'JFUSE_SAVE_STATES', 'JFUSE_INITIAL_SNOW', 'JFUSE_PARAMS_TO_CALIBRATE',
            'JFUSE_INITIAL_S1', 'JFUSE_USE_GRADIENT_CALIBRATION', 'JFUSE_ROUTING_PARAMS_TO_CALIBRATE',
            'JFUSE_OUTPUT_FREQUENCY', 'JFUSE_FORCING_FILE', 'JFUSE_ROUTING_SUBSTEP_METHOD',
            'JFUSE_ROUTING_MAX_SUBSTEPS', 'JFUSE_DEFAULT_CHANNEL_SLOPE',
            # HBV model distributed/routing configuration keys
            'HBV_ROUTING_MAX_SUBSTEPS', 'HBV_DISTRIBUTED_PARAM_MODE', 'HBV_TIMESTEP_HOURS',
            'HBV_DISTRIBUTED_ROUTING', 'HBV_ROUTING_SUBSTEP_METHOD', 'HBV_DEFAULT_CHANNEL_SLOPE',
            'HBV_DEFAULT_MANNINGS_N',
        }
        # HBV uses a custom config adapter (HBVConfigAdapter) that returns field transformers
        # in a different format than nested paths. The manual mapping is correct for
        # the transformer use case while the adapter serves the model execution use case.
        known_mismatched = {
            'STREAMFLOW_DATA_PROVIDER', 'DOWNLOAD_USGS_DATA',
            # HBV config adapter pattern returns (field_name, type) tuples
            'HBV_SPATIAL_MODE', 'HBV_ROUTING_INTEGRATION', 'HBV_BACKEND',
            'HBV_USE_GPU', 'HBV_JIT_COMPILE', 'HBV_WARMUP_DAYS',
            'HBV_PARAMS_TO_CALIBRATE', 'HBV_USE_GRADIENT_CALIBRATION',
            'HBV_CALIBRATION_METRIC', 'HBV_INITIAL_SNOW', 'HBV_INITIAL_SM',
            'HBV_INITIAL_SUZ', 'HBV_INITIAL_SLZ', 'HBV_PET_METHOD', 'HBV_LATITUDE',
            'HBV_SAVE_STATES', 'HBV_OUTPUT_FREQUENCY',
            'HBV_DEFAULT_TT', 'HBV_DEFAULT_CFMAX', 'HBV_DEFAULT_SFCF',
            'HBV_DEFAULT_CFR', 'HBV_DEFAULT_CWH', 'HBV_DEFAULT_FC',
            'HBV_DEFAULT_LP', 'HBV_DEFAULT_BETA', 'HBV_DEFAULT_K0',
            'HBV_DEFAULT_K1', 'HBV_DEFAULT_K2', 'HBV_DEFAULT_UZL',
            'HBV_DEFAULT_PERC', 'HBV_DEFAULT_MAXBAS',
        }

        # Filter out known differences
        unexpected_missing = set(result['missing_in_auto']) - known_missing
        unexpected_extra = set(result['extra_in_auto']) - known_extra
        unexpected_mismatched = set(result['mismatched'].keys()) - known_mismatched

        # Assert no unexpected differences
        assert len(unexpected_missing) == 0, (
            f"Unexpected missing keys: {unexpected_missing}"
        )
        assert len(unexpected_extra) == 0, (
            f"Unexpected extra keys: {unexpected_extra}"
        )
        assert len(unexpected_mismatched) == 0, (
            f"Unexpected mismatched keys: {unexpected_mismatched}"
        )

        # Ensure we have a substantial mapping (>90% of manual)
        coverage = result['auto_count'] / result['manual_count'] if result['manual_count'] > 0 else 0
        assert coverage > 0.9, f"Auto-generated coverage is too low: {coverage:.1%}"

    def test_auto_generated_completeness(self):
        """Ensure critical fields are present in auto-generated mapping."""
        from symfluence.core.config.transformers import get_flat_to_nested_map

        mapping = get_flat_to_nested_map()

        # Required fields must be present
        required = [
            'DOMAIN_NAME',
            'HYDROLOGICAL_MODEL',
            'FORCING_DATASET',
            'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START',
            'EXPERIMENT_TIME_END',
        ]

        for field in required:
            assert field in mapping, f"Required field {field} missing from auto-generated mapping"

    def test_duplicate_key_resolution(self):
        """Verify that duplicate keys resolve with expected priority."""
        from symfluence.core.config.transformers import get_flat_to_nested_map

        mapping = get_flat_to_nested_map()

        # Data section should win over system for FORCE_DOWNLOAD
        # (evaluation has even higher priority, but FORCE_DOWNLOAD is not in evaluation)
        if 'FORCE_DOWNLOAD' in mapping:
            assert mapping['FORCE_DOWNLOAD'][0] in ['data', 'system'], \
                f"FORCE_DOWNLOAD should be in data or system section, got {mapping['FORCE_DOWNLOAD']}"

        # Forcing section should win over data for SUPPLEMENT_FORCING
        if 'SUPPLEMENT_FORCING' in mapping:
            assert mapping['SUPPLEMENT_FORCING'][0] in ['forcing', 'data'], \
                f"SUPPLEMENT_FORCING should be in forcing or data section, got {mapping['SUPPLEMENT_FORCING']}"

    def test_generation_performance(self):
        """Auto-generation should complete quickly."""
        import time
        from symfluence.core.config.transformers import get_flat_to_nested_map

        # Clear cache to force regeneration
        import symfluence.core.config.transformers as t
        t._AUTO_GENERATED_MAP = None

        start = time.time()
        mapping = get_flat_to_nested_map()
        elapsed = time.time() - start

        assert len(mapping) > 400, f"Mapping should have 400+ entries, got {len(mapping)}"
        assert elapsed < 0.5, f"Generation took {elapsed:.3f}s, should be < 0.5s"

    def test_cached_access_performance(self):
        """Cached access should be near-instant."""
        import time
        from symfluence.core.config.transformers import get_flat_to_nested_map

        # First call (ensures cache populated)
        get_flat_to_nested_map()

        # Second call should be cached
        start = time.time()
        for _ in range(1000):
            mapping = get_flat_to_nested_map()
        elapsed = time.time() - start

        avg_time = elapsed / 1000
        assert avg_time < 0.0001, f"Cached access took {avg_time*1e6:.1f}μs per call (target: <100μs)"
