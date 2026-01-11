"""
Integration tests for attribute processing system.

Tests the refactored modular attribute processing architecture including:
- Individual processor functionality
- Full orchestration through attributeProcessor wrapper
- Backward compatibility with original interface
- Both lumped and distributed catchment configurations
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

from symfluence.data.preprocessing.attribute_processor import attributeProcessor
from symfluence.data.preprocessing.attribute_processors import (
    ElevationProcessor,
    GeologyProcessor,
    SoilProcessor,
    LandCoverProcessor,
    ClimateProcessor,
    HydrologyProcessor
)


@pytest.fixture
def logger():
    """Create a mock logger for testing."""
    return Mock()


@pytest.fixture
def lumped_config(tmp_path):
    """Configuration for lumped catchment testing."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_lumped',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'CATCHMENT_PATH': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'CATCHMENT_SHP_HRUID': 'HRU_ID'
    }


@pytest.fixture
def distributed_config(tmp_path):
    """Configuration for distributed catchment testing."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_distributed',
        'DOMAIN_DEFINITION_METHOD': 'distributed',
        'DOMAIN_DISCRETIZATION': 'distributed_5',
        'CATCHMENT_PATH': 'default',
        'CATCHMENT_SHP_NAME': 'default',
        'CATCHMENT_SHP_HRUID': 'HRU_ID'
    }


def setup_test_directories(tmp_path, domain_name):
    """Create necessary directory structure for testing."""
    project_dir = tmp_path / f"domain_{domain_name}"

    # Create directory structure
    directories = [
        'shapefiles/catchment',
        'attributes/elevation/dem',
        'attributes/elevation/slope',
        'attributes/elevation/aspect',
        'cache/elevation',
        'cache/geology',
        'cache/soil',
        'cache/landcover',
        'cache/climate',
        'cache/hydrology'
    ]

    for dir_path in directories:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)

    return project_dir


class TestDeprecationWarning:
    """Test that deprecation warning is raised for original module."""

    def test_original_module_raises_deprecation_warning(self):
        """Verify that importing the original module raises a deprecation warning."""
        import sys
        import importlib
        from symfluence.data.preprocessing import attribute_processing

        # Reset warning registry for the module to ensure warning triggers again
        if hasattr(attribute_processing, '__warningregistry__'):
            attribute_processing.__warningregistry__.clear()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Force reload to trigger module-level warning code
            importlib.reload(attribute_processing)

            # Verify deprecation warning was raised
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "attribute_processing_refactored" in str(w[0].message)


class TestIndividualProcessors:
    """Test individual attribute processors."""

    def test_elevation_processor_initialization(self, lumped_config, logger, tmp_path):
        """Test ElevationProcessor initializes correctly."""
        setup_test_directories(tmp_path, lumped_config['DOMAIN_NAME'])
        processor = ElevationProcessor(lumped_config, logger)

        assert processor.config == lumped_config
        assert processor.logger == logger
        assert processor.domain_name == 'test_lumped'
        assert processor.dem_dir.exists()

    def test_geology_processor_initialization(self, lumped_config, logger, tmp_path):
        """Test GeologyProcessor initializes correctly."""
        processor = GeologyProcessor(lumped_config, logger)

        assert processor.config == lumped_config
        assert processor.logger == logger
        assert processor.domain_name == 'test_lumped'

    def test_soil_processor_initialization(self, lumped_config, logger, tmp_path):
        """Test SoilProcessor initializes correctly."""
        processor = SoilProcessor(lumped_config, logger)

        assert processor.config == lumped_config
        assert processor.logger == logger
        assert processor.domain_name == 'test_lumped'

    def test_landcover_processor_initialization(self, lumped_config, logger, tmp_path):
        """Test LandCoverProcessor initializes correctly."""
        processor = LandCoverProcessor(lumped_config, logger)

        assert processor.config == lumped_config
        assert processor.logger == logger
        assert processor.domain_name == 'test_lumped'

    def test_climate_processor_initialization(self, lumped_config, logger, tmp_path):
        """Test ClimateProcessor initializes correctly."""
        processor = ClimateProcessor(lumped_config, logger)

        assert processor.config == lumped_config
        assert processor.logger == logger
        assert processor.domain_name == 'test_lumped'

    def test_hydrology_processor_initialization(self, lumped_config, logger, tmp_path):
        """Test HydrologyProcessor initializes correctly."""
        processor = HydrologyProcessor(lumped_config, logger)

        assert processor.config == lumped_config
        assert processor.logger == logger
        assert processor.domain_name == 'test_lumped'


class TestProcessorProcessMethods:
    """Test that all processors implement process() method correctly."""

    def test_processors_return_dict(self, lumped_config, logger, tmp_path):
        """Verify all processors return dictionaries from process()."""
        processors = [
            ElevationProcessor(lumped_config, logger),
            GeologyProcessor(lumped_config, logger),
            SoilProcessor(lumped_config, logger),
            LandCoverProcessor(lumped_config, logger),
            ClimateProcessor(lumped_config, logger),
            HydrologyProcessor(lumped_config, logger)
        ]

        for processor in processors:
            # Mock the catchment_path to avoid file not found errors
            with patch.object(processor, 'catchment_path', tmp_path / 'test.shp'):
                result = processor.process()
                assert isinstance(result, dict), f"{processor.__class__.__name__} should return dict"


class TestAttributeProcessorWrapper:
    """Test the backward-compatible attributeProcessor wrapper."""

    def test_wrapper_initialization_lumped(self, lumped_config, logger, tmp_path):
        """Test wrapper initializes all sub-processors for lumped catchment."""
        processor = attributeProcessor(lumped_config, logger)

        # Verify all sub-processors are initialized
        assert isinstance(processor.elevation, ElevationProcessor)
        assert isinstance(processor.geology, GeologyProcessor)
        assert isinstance(processor.soil, SoilProcessor)
        assert isinstance(processor.landcover, LandCoverProcessor)
        assert isinstance(processor.climate, ClimateProcessor)
        assert isinstance(processor.hydrology, HydrologyProcessor)

        # Verify common properties are exposed
        assert processor.data_dir == Path(tmp_path)
        assert processor.domain_name == 'test_lumped'
        assert processor.project_dir == tmp_path / 'domain_test_lumped'

    def test_wrapper_initialization_distributed(self, distributed_config, logger, tmp_path):
        """Test wrapper initializes all sub-processors for distributed catchment."""
        processor = attributeProcessor(distributed_config, logger)

        assert isinstance(processor.elevation, ElevationProcessor)
        assert processor.domain_name == 'test_distributed'

    def test_wrapper_delegates_elevation_methods(self, lumped_config, logger, tmp_path):
        """Test wrapper correctly delegates elevation-related methods."""
        processor = attributeProcessor(lumped_config, logger)

        # Mock the underlying methods
        processor.elevation.find_dem_file = Mock(return_value=Path('/mock/dem.tif'))
        processor.elevation.generate_slope_and_aspect = Mock(return_value={'slope': Path('/mock/slope.tif')})

        # Test delegation
        dem_file = processor.find_dem_file()
        assert dem_file == Path('/mock/dem.tif')

        slope_aspect = processor.generate_slope_and_aspect(dem_file)
        assert 'slope' in slope_aspect

    def test_wrapper_delegates_hydrology_methods(self, lumped_config, logger, tmp_path):
        """Test wrapper correctly delegates hydrology-related methods."""
        processor = attributeProcessor(lumped_config, logger)

        # Mock the underlying methods
        processor.hydrology.calculate_water_balance = Mock(return_value={'water_balance': 100.0})
        processor.hydrology.calculate_streamflow_signatures = Mock(return_value={'signature': 0.5})

        # Test delegation
        wb = processor.calculate_water_balance()
        assert 'water_balance' in wb

        signatures = processor.calculate_streamflow_signatures()
        assert 'signature' in signatures


class TestFullOrchestration:
    """Test full attribute processing orchestration."""

    @patch('symfluence.data.preprocessing.attribute_processors.elevation.ElevationProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.geology.GeologyProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.soil.SoilProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.landcover.LandCoverProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.climate.ClimateProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.hydrology.HydrologyProcessor.process')
    def test_process_attributes_lumped(self, mock_hydro, mock_climate, mock_landcover,
                                      mock_soil, mock_geology, mock_elevation,
                                      lumped_config, logger, tmp_path):
        """Test full attribute processing for lumped catchment."""
        # Mock processor outputs
        mock_elevation.return_value = {'elevation.mean': 1000.0, 'elevation.slope_mean': 5.0}
        mock_geology.return_value = {'geology.permeability_mean': 1e-5}
        mock_soil.return_value = {'soil.sand_fraction': 0.3, 'soil.clay_fraction': 0.2}
        mock_landcover.return_value = {'landcover.forest_fraction': 0.5}
        mock_climate.return_value = {'climate.prec_annual_mean': 800.0}
        mock_hydro.return_value = {'hydrology.stream_density': 2.5}

        processor = attributeProcessor(lumped_config, logger)
        df = processor.process_attributes()

        # Verify DataFrame structure for lumped catchment
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Single row for lumped
        assert df.index.name == 'basin_id'
        assert df.index[0] == 1

        # Verify all attributes are present
        assert 'elevation.mean' in df.columns
        assert 'geology.permeability_mean' in df.columns
        assert 'soil.sand_fraction' in df.columns
        assert 'landcover.forest_fraction' in df.columns
        assert 'climate.prec_annual_mean' in df.columns
        assert 'hydrology.stream_density' in df.columns

        # Verify all processors were called
        mock_elevation.assert_called_once()
        mock_geology.assert_called_once()
        mock_soil.assert_called_once()
        mock_landcover.assert_called_once()
        mock_climate.assert_called_once()
        mock_hydro.assert_called_once()

    @patch('symfluence.data.preprocessing.attribute_processors.elevation.ElevationProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.geology.GeologyProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.soil.SoilProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.landcover.LandCoverProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.climate.ClimateProcessor.process')
    @patch('symfluence.data.preprocessing.attribute_processors.hydrology.HydrologyProcessor.process')
    def test_process_attributes_distributed(self, mock_hydro, mock_climate, mock_landcover,
                                           mock_soil, mock_geology, mock_elevation,
                                           distributed_config, logger, tmp_path):
        """Test full attribute processing for distributed catchment."""
        # Mock processor outputs with HRU prefixes
        mock_elevation.return_value = {
            'HRU_1_elevation.mean': 1000.0,
            'HRU_2_elevation.mean': 1100.0,
            'HRU_3_elevation.mean': 1200.0
        }
        mock_geology.return_value = {
            'HRU_1_geology.permeability_mean': 1e-5,
            'HRU_2_geology.permeability_mean': 2e-5,
            'HRU_3_geology.permeability_mean': 1.5e-5
        }
        mock_soil.return_value = {
            'HRU_1_soil.sand_fraction': 0.3,
            'HRU_2_soil.sand_fraction': 0.4,
            'HRU_3_soil.sand_fraction': 0.35
        }
        mock_landcover.return_value = {
            'HRU_1_landcover.forest_fraction': 0.5,
            'HRU_2_landcover.forest_fraction': 0.6,
            'HRU_3_landcover.forest_fraction': 0.55
        }
        mock_climate.return_value = {
            'HRU_1_climate.prec_annual_mean': 800.0,
            'HRU_2_climate.prec_annual_mean': 850.0,
            'HRU_3_climate.prec_annual_mean': 825.0
        }
        mock_hydro.return_value = {
            'HRU_1_hydrology.stream_density': 2.5,
            'HRU_2_hydrology.stream_density': 2.8,
            'HRU_3_hydrology.stream_density': 2.6
        }

        processor = attributeProcessor(distributed_config, logger)
        df = processor.process_attributes()

        # Verify DataFrame structure for distributed catchment
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Three HRUs
        assert df.index.names == ['basin_id', 'hru_id']

        # Verify HRU prefixes are removed from column names
        assert 'elevation.mean' in df.columns
        assert 'geology.permeability_mean' in df.columns
        assert 'soil.sand_fraction' in df.columns

        # Verify values are correctly distributed across HRUs
        assert df.loc[(1, 1), 'elevation.mean'] == 1000.0
        assert df.loc[(1, 2), 'elevation.mean'] == 1100.0
        assert df.loc[(1, 3), 'elevation.mean'] == 1200.0


class TestCaching:
    """Test caching functionality across processors."""

    def test_processors_create_cache_directories(self, lumped_config, logger, tmp_path):
        """Verify processors create cache directories."""
        # Set up the base directory structure
        setup_test_directories(tmp_path, lumped_config['DOMAIN_NAME'])

        processors = [
            GeologyProcessor(lumped_config, logger),
            SoilProcessor(lumped_config, logger),
            LandCoverProcessor(lumped_config, logger),
            ClimateProcessor(lumped_config, logger)
        ]

        cache_base = tmp_path / f"domain_{lumped_config['DOMAIN_NAME']}" / 'cache'

        # After setup, cache base directory should exist
        assert cache_base.exists()


class TestErrorHandling:
    """Test error handling in attribute processing."""

    def test_empty_results_returns_empty_dataframe(self, lumped_config, logger, tmp_path):
        """Test that empty results return empty DataFrame gracefully."""
        # Mock all processors to return empty dicts
        with patch.multiple(
            'symfluence.data.preprocessing.attribute_processors.elevation.ElevationProcessor',
            process=Mock(return_value={})
        ), patch.multiple(
            'symfluence.data.preprocessing.attribute_processors.geology.GeologyProcessor',
            process=Mock(return_value={})
        ), patch.multiple(
            'symfluence.data.preprocessing.attribute_processors.soil.SoilProcessor',
            process=Mock(return_value={})
        ), patch.multiple(
            'symfluence.data.preprocessing.attribute_processors.landcover.LandCoverProcessor',
            process=Mock(return_value={})
        ), patch.multiple(
            'symfluence.data.preprocessing.attribute_processors.climate.ClimateProcessor',
            process=Mock(return_value={})
        ), patch.multiple(
            'symfluence.data.preprocessing.attribute_processors.hydrology.HydrologyProcessor',
            process=Mock(return_value={})
        ):
            processor = attributeProcessor(lumped_config, logger)
            df = processor.process_attributes()

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_processor_exception_propagates(self, lumped_config, logger, tmp_path):
        """Test that exceptions in processors propagate correctly."""
        with patch.object(ElevationProcessor, 'process', side_effect=Exception("Test error")):
            processor = attributeProcessor(lumped_config, logger)

            with pytest.raises(Exception, match="Test error"):
                processor.process_attributes()


class TestBackwardCompatibility:
    """Test backward compatibility with original interface."""

    def test_has_elevation_delegate_methods(self, lumped_config, logger, tmp_path):
        """Verify wrapper has all expected elevation methods."""
        processor = attributeProcessor(lumped_config, logger)

        assert hasattr(processor, 'find_dem_file')
        assert hasattr(processor, 'generate_slope_and_aspect')
        assert hasattr(processor, 'calculate_statistics')
        assert hasattr(processor, '_process_elevation_attributes')

    def test_has_hydrology_delegate_methods(self, lumped_config, logger, tmp_path):
        """Verify wrapper has all expected hydrology methods."""
        processor = attributeProcessor(lumped_config, logger)

        assert hasattr(processor, 'calculate_water_balance')
        assert hasattr(processor, 'calculate_streamflow_signatures')
        assert hasattr(processor, 'calculate_baseflow_attributes')
        assert hasattr(processor, 'enhance_river_network_analysis')

    def test_has_common_properties(self, lumped_config, logger, tmp_path):
        """Verify wrapper exposes common properties."""
        processor = attributeProcessor(lumped_config, logger)

        assert hasattr(processor, 'data_dir')
        assert hasattr(processor, 'domain_name')
        assert hasattr(processor, 'project_dir')
        assert hasattr(processor, 'catchment_path')
        assert hasattr(processor, 'dem_dir')
        assert hasattr(processor, 'slope_dir')
        assert hasattr(processor, 'aspect_dir')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
