"""
Unit tests for elevation processing methods.

Tests:
- generate_slope_and_aspect() - DEM processing with GDAL
- _process_elevation_attributes() - Statistics calculation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.data.preprocessing.attribute_processors.elevation import ElevationProcessor

pytestmark = [pytest.mark.unit, pytest.mark.quick]


class TestGenerateSlopeAndAspect:
    """Tests for generate_slope_and_aspect method."""

    def test_generate_slope_and_aspect_success(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file, temp_project_dir
    ):
        """Test successful slope and aspect generation from DEM."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock GDAL DEMProcessing
        with patch('osgeo.gdal.DEMProcessing') as mock_dem_proc:
            mock_dem_proc.return_value = MagicMock()  # Simulate successful GDAL operation

            result = processor.generate_slope_and_aspect(mock_dem_file)

        assert isinstance(result, dict)
        assert "slope" in result
        assert "aspect" in result

        # Check that output paths are Path objects
        assert isinstance(result["slope"], Path)
        assert isinstance(result["aspect"], Path)

        # Verify GDAL was called twice (once for slope, once for aspect)
        assert mock_dem_proc.call_count == 2

        # Check file naming convention
        assert "slope" in str(result["slope"])
        assert "aspect" in str(result["aspect"])

    def test_generate_slope_and_aspect_output_paths(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test that output files are created in correct directories."""
        processor = ElevationProcessor(base_config, test_logger)

        with patch('osgeo.gdal.DEMProcessing') as mock_dem_proc:
            mock_dem_proc.return_value = MagicMock()

            result = processor.generate_slope_and_aspect(mock_dem_file)

        # Slope should go to attributes/elevation/slope/
        assert "slope" in str(result["slope"])
        assert result["slope"].parent.name == "slope"

        # Aspect should go to attributes/elevation/aspect/
        assert "aspect" in str(result["aspect"])
        assert result["aspect"].parent.name == "aspect"

    def test_generate_slope_and_aspect_gdal_parameters(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test that GDAL is called with correct parameters."""
        processor = ElevationProcessor(base_config, test_logger)

        with patch('osgeo.gdal.DEMProcessing') as mock_dem_proc:
            mock_dem_proc.return_value = MagicMock()

            processor.generate_slope_and_aspect(mock_dem_file)

        # Verify GDAL was called twice (slope and aspect)
        assert mock_dem_proc.call_count == 2

        # Verify slope processing call (positional args: output, input, 'slope')
        slope_call = mock_dem_proc.call_args_list[0]
        assert 'slope' in slope_call[0]  # 'slope' is the 3rd positional arg

        # Verify aspect processing call
        aspect_call = mock_dem_proc.call_args_list[1]
        assert 'aspect' in aspect_call[0]  # 'aspect' is the 3rd positional arg

    def test_generate_slope_and_aspect_missing_dem(
        self, base_config, test_logger, lumped_catchment_shapefile
    ):
        """Test error handling when DEM file doesn't exist."""
        processor = ElevationProcessor(base_config, test_logger)

        fake_dem = Path("/nonexistent/dem.tif")

        with pytest.raises((FileNotFoundError, Exception)):
            processor.generate_slope_and_aspect(fake_dem)

    def test_generate_slope_and_aspect_gdal_failure(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test handling of GDAL processing failure."""
        processor = ElevationProcessor(base_config, test_logger)

        with patch('osgeo.gdal.DEMProcessing') as mock_dem_proc:
            # Simulate GDAL failure by raising an exception
            mock_dem_proc.side_effect = Exception("GDAL processing failed")

            with pytest.raises(Exception):
                processor.generate_slope_and_aspect(mock_dem_file)

    @pytest.mark.integration
    def test_generate_slope_and_aspect_real_dem(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Integration test with real GDAL (if available)."""
        try:
            from osgeo import gdal
        except ImportError:
            pytest.skip("GDAL not available")

        processor = ElevationProcessor(base_config, test_logger)

        result = processor.generate_slope_and_aspect(mock_dem_file)

        # Check that files were actually created
        assert result["slope"].exists(), "Slope file not created"
        assert result["aspect"].exists(), "Aspect file not created"

        # Verify files are valid rasters
        slope_ds = gdal.Open(str(result["slope"]))
        assert slope_ds is not None, "Slope raster invalid"
        slope_ds = None  # Close dataset

        aspect_ds = gdal.Open(str(result["aspect"]))
        assert aspect_ds is not None, "Aspect raster invalid"
        aspect_ds = None  # Close dataset


class TestProcessElevationAttributes:
    """Tests for _process_elevation_attributes method."""

    def test_process_elevation_attributes_finds_dem(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test that elevation processing can find DEM file."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock find_dem_file to return our mock DEM
        with patch.object(processor, 'find_dem_file', return_value=mock_dem_file):
            with patch.object(processor, 'generate_slope_and_aspect') as mock_gen:
                mock_gen.return_value = {
                    "slope": mock_dem_file.parent / "slope.tif",
                    "aspect": mock_dem_file.parent / "aspect.tif"
                }
                with patch.object(processor, 'calculate_statistics', return_value={"elevation_mean": 150.0}):
                    result = processor.process()

        assert isinstance(result, dict)
        # Should have called generate_slope_and_aspect
        assert mock_gen.called

    def test_process_elevation_attributes_calculates_all_stats(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test that all three attributes (dem, slope, aspect) are calculated."""
        processor = ElevationProcessor(base_config, test_logger)

        mock_stats = {
            "dem_mean": 150.0,
            "dem_std": 25.0
        }

        with patch.object(processor, 'find_dem_file', return_value=mock_dem_file):
            with patch.object(processor, 'generate_slope_and_aspect') as mock_gen:
                # Return all 3 rasters: dem, slope, aspect
                mock_gen.return_value = {
                    "dem": mock_dem_file,
                    "slope": mock_dem_file.parent / "slope.tif",
                    "aspect": mock_dem_file.parent / "aspect.tif"
                }
                with patch.object(processor, 'calculate_statistics', return_value=mock_stats) as mock_calc:
                    result = processor.process()

        # Should call calculate_statistics 3 times (dem, slope, aspect)
        assert mock_calc.call_count == 3

        # Verify it was called with correct attribute names
        call_args = [call[0][1] for call in mock_calc.call_args_list]
        assert "dem" in call_args
        assert "slope" in call_args
        assert "aspect" in call_args

    def test_process_elevation_attributes_no_dem_found(
        self, base_config, test_logger, lumped_catchment_shapefile
    ):
        """Test handling when no DEM file is found."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock find_dem_file to raise FileNotFoundError
        with patch.object(processor, 'find_dem_file', side_effect=FileNotFoundError):
            result = processor.process()

        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)
        assert len(result) == 0 or all(v is None for v in result.values())

    def test_process_elevation_attributes_combines_results(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test that results from all three attributes are combined."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock different stats for each attribute (note: keys use dots in process())
        stats_by_attr = {
            "dem": {"dem_mean": 150.0, "dem_std": 25.0},
            "slope": {"slope_mean": 10.5, "slope_std": 5.2},
            "aspect": {"aspect_circmean": 180.0, "aspect_circstd": 90.0}
        }

        def mock_calc_stats(raster_file, attr_name):
            return stats_by_attr.get(attr_name, {})

        with patch.object(processor, 'find_dem_file', return_value=mock_dem_file):
            with patch.object(processor, 'generate_slope_and_aspect') as mock_gen:
                mock_gen.return_value = {
                    "dem": mock_dem_file,
                    "slope": mock_dem_file.parent / "slope.tif",
                    "aspect": mock_dem_file.parent / "aspect.tif"
                }
                with patch.object(processor, 'calculate_statistics', side_effect=mock_calc_stats):
                    result = processor.process()

        # Result should contain all attributes (with dot notation)
        assert "dem.mean" in result
        assert "slope.mean" in result
        assert "aspect.circmean" in result

        assert result["dem.mean"] == 150.0
        assert result["slope.mean"] == 10.5
        assert result["aspect.circmean"] == 180.0


class TestFindDemFile:
    """Tests for find_dem_file method."""

    def test_find_dem_file_in_dem_dir(
        self, base_config, test_logger, lumped_catchment_shapefile, temp_project_dir
    ):
        """Test finding DEM in the expected directory."""
        # Setup: Create a mock DEM in attributes/elevation/dem
        # Ensure we use the correct domain directory structure
        domain_name = base_config['DOMAIN_NAME']
        dem_dir = temp_project_dir / f"domain_{domain_name}" / "data" / "attributes" / "elevation" / "dem"
        dem_dir.mkdir(parents=True, exist_ok=True)
        mock_dem = dem_dir / "found_me.tif"
        mock_dem.touch()

        # Initialize processor
        # Ensure processor uses the temp_project_dir as data dir
        base_config['SYMFLUENCE_DATA_DIR'] = str(temp_project_dir)
        processor = ElevationProcessor(base_config, test_logger)

        # Action: Find DEM
        found_dem = processor.find_dem_file()

        # Assert: Check if correct file found
        assert found_dem == mock_dem

    def test_find_dem_file_autodiscovery(
        self, base_config, test_logger, lumped_catchment_shapefile, temp_project_dir
    ):
        """Test DEM auto-discovery in attributes/dem directory."""
        # Create a DEM in the expected location
        dem_dir = temp_project_dir / "data" / "attributes" / "dem"
        dem_dir.mkdir(parents=True, exist_ok=True)
        auto_dem = dem_dir / "test_domain_dem.tif"
        auto_dem.touch()  # Create empty file

        base_config["DEM_PATH"] = "default"  # Trigger auto-discovery

        processor = ElevationProcessor(base_config, test_logger)

        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [auto_dem]

            found_dem = processor.find_dem_file()

        assert found_dem == auto_dem

    def test_find_dem_file_not_found(
        self, base_config, test_logger, lumped_catchment_shapefile
    ):
        """Test error when DEM file doesn't exist."""
        base_config["DEM_PATH"] = "/nonexistent/dem.tif"

        processor = ElevationProcessor(base_config, test_logger)

        with pytest.raises((FileNotFoundError, ValueError)):
            processor.find_dem_file()

    def test_find_dem_file_multiple_dems(
        self, base_config, test_logger, lumped_catchment_shapefile, temp_project_dir
    ):
        """Test handling when multiple DEM files exist (should use first or most recent)."""
        dem_dir = temp_project_dir / "data" / "attributes" / "dem"
        dem_dir.mkdir(parents=True, exist_ok=True)

        dem1 = dem_dir / "dem_v1.tif"
        dem2 = dem_dir / "dem_v2.tif"
        dem1.touch()
        dem2.touch()

        base_config["DEM_PATH"] = "default"

        processor = ElevationProcessor(base_config, test_logger)

        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [dem1, dem2]

            found_dem = processor.find_dem_file()

        # Should return one of them (implementation-dependent)
        assert found_dem in [dem1, dem2]
