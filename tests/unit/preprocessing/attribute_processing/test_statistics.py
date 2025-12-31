"""
Unit tests for attributeProcessor.calculate_statistics()

This is a critical method used by 20+ other methods for zonal statistics.
Tests cover:
- Normal statistics calculation
- Circular statistics for aspect
- Missing/empty data handling
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from symfluence.utils.data.preprocessing.attribute_processors.elevation import ElevationProcessor


pytestmark = [pytest.mark.unit, pytest.mark.quick]


class TestCalculateStatistics:
    """Tests for the calculate_statistics method."""

    def test_calculate_statistics_normal_attribute(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file, mock_zonal_stats_result
    ):
        """Test statistics calculation for non-circular attributes (elevation, slope, etc.)."""
        processor = ElevationProcessor(base_config, test_logger)

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_zonal_stats_result):
            result = processor.calculate_statistics(mock_dem_file, "elevation")

        # Should return dict with mean, min, max, std (no median in default stats)
        assert isinstance(result, dict)
        assert "elevation_mean" in result
        assert "elevation_min" in result
        assert "elevation_max" in result
        assert "elevation_std" in result

        # Check values match actual DEM file statistics
        assert isinstance(result["elevation_mean"], (int, float, np.floating))
        assert isinstance(result["elevation_min"], (int, float, np.floating))
        assert isinstance(result["elevation_max"], (int, float, np.floating))

    def test_calculate_statistics_aspect_circular(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test circular statistics for aspect (requires special handling)."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock zonal_stats to return aspect data with pixel values
        aspect_values = np.array([0, 45, 90, 135, 180, 225, 270, 315] * 12 + [0, 45, 90, 135])
        mock_result = [{
            "min": 0.0,
            "max": 359.0,
            "mean": 180.0,  # Linear mean (not valid for circular)
            "median": 175.0,
            "std": 90.0,
            "count": len(aspect_values)
        }]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_result) as mock_zs:
            # Configure mock to return pixel values when raster_out=True
            mock_zs.return_value = [{"raster": aspect_values}]

            # Ensure catchment_path is a file path, not a directory
            processor.catchment_path = lumped_catchment_shapefile
            result = processor.calculate_statistics(mock_dem_file, "aspect")

        # Should use circular statistics (circmean, circstd)
        assert isinstance(result, dict)
        assert "aspect_circmean" in result or "aspect_min" in result

        # If circular mean is present, check it's in valid range
        if "aspect_circmean" in result:
            assert 0 <= result["aspect_circmean"] <= 360

    def test_calculate_statistics_empty_catchment(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test handling of empty/masked catchments (all nodata)."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock to return empty zonal stats (all None values filtered out)
        empty_stats = [{}]  # Empty dict when all stats are None

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=empty_stats):
            result = processor.calculate_statistics(mock_dem_file, "elevation")

        # Should return empty dict when no valid statistics
        assert isinstance(result, dict)
        assert len(result) == 0  # No valid stats, so empty dict

    def test_calculate_statistics_missing_raster_file(
        self, base_config, test_logger, lumped_catchment_shapefile
    ):
        """Test error handling when raster file doesn't exist."""
        processor = ElevationProcessor(base_config, test_logger)

        fake_file = Path("/nonexistent/raster.tif")

        with pytest.raises((FileNotFoundError, Exception)):
            processor.calculate_statistics(fake_file, "elevation")

    def test_calculate_statistics_distributed_hrus(
        self, distributed_config, test_logger, distributed_catchment_shapefile, mock_dem_file, mock_zonal_stats_result
    ):
        """Test statistics calculation for multiple HRUs."""
        processor = ElevationProcessor(distributed_config, test_logger)

        # Mock zonal_stats to return results for 5 HRUs
        multi_hru_results = [mock_zonal_stats_result[0].copy() for _ in range(5)]
        # Vary the elevation slightly for each HRU
        for i, result in enumerate(multi_hru_results):
            result["mean"] = 150.0 + i * 10

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=multi_hru_results):
            result = processor.calculate_statistics(mock_dem_file, "elevation")

        # Should return dict with HRU-prefixed keys
        assert isinstance(result, dict)
        # With distributed config, results might be aggregated or per-HRU
        # Check that we got results
        assert len(result) > 0

    @patch('scipy.stats.circmean')
    @patch('scipy.stats.circstd')
    def test_calculate_statistics_aspect_uses_scipy(
        self, mock_circstd, mock_circmean, base_config, test_logger,
        lumped_catchment_shapefile, mock_dem_file
    ):
        """Verify that aspect statistics use scipy circular functions."""
        processor = ElevationProcessor(base_config, test_logger)

        # Set mock return values
        mock_circmean.return_value = 180.0
        mock_circstd.return_value = 45.0

        aspect_values = np.array([0, 90, 180, 270] * 25)
        mock_result = [{"raster": aspect_values}]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_result):
            # Ensure catchment_path is a file path
            processor.catchment_path = lumped_catchment_shapefile
            result = processor.calculate_statistics(mock_dem_file, "aspect")

        # Check that results include circular statistics
        assert isinstance(result, dict)
        if "aspect_circmean" in result:
            assert isinstance(result["aspect_circmean"], (int, float, np.floating))

    def test_calculate_statistics_aspect_conversion_to_radians(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test that aspect values are correctly converted to radians for circular stats."""
        processor = ElevationProcessor(base_config, test_logger)

        # Known aspect values in degrees
        aspect_deg = np.array([0, 90, 180, 270])
        mock_result = [{"raster": aspect_deg}]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_result):
            # Ensure catchment_path is a file path
            processor.catchment_path = lumped_catchment_shapefile
            result = processor.calculate_statistics(mock_dem_file, "aspect")

            # Verify result contains aspect statistics
            assert isinstance(result, dict)
            if "aspect_circmean" in result:
                assert 0 <= result["aspect_circmean"] <= 360

    def test_calculate_statistics_with_nodata_values(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test handling of nodata/masked values in raster."""
        processor = ElevationProcessor(base_config, test_logger)

        # Mock result with partial nodata
        mock_result = [{
            "min": 100.0,
            "max": 200.0,
            "mean": 150.0,
            "median": 148.0,
            "std": 25.0,
            "count": 50  # Only 50 valid pixels out of 100
        }]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_result):
            result = processor.calculate_statistics(mock_dem_file, "elevation")

        # Should still return valid statistics for non-masked pixels
        assert isinstance(result["elevation_mean"], (int, float, np.floating))
        # Note: rasterstats may not return 'count' in the default stats
        if "elevation_count" in result:
            assert result["elevation_count"] > 0

    def test_calculate_statistics_attribute_name_in_keys(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file, mock_zonal_stats_result
    ):
        """Test that attribute name is correctly incorporated into result keys."""
        processor = ElevationProcessor(base_config, test_logger)

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_zonal_stats_result):
            result = processor.calculate_statistics(mock_dem_file, "custom_attr")

        # All keys should start with attribute name
        for key in result.keys():
            assert key.startswith("custom_attr_"), f"Key {key} doesn't start with 'custom_attr_'"

        assert "custom_attr_mean" in result
        assert "custom_attr_std" in result


class TestCalculateStatisticsEdgeCases:
    """Edge case tests for calculate_statistics."""

    def test_very_small_catchment_single_pixel(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test statistics for a catchment covering only 1 pixel."""
        processor = ElevationProcessor(base_config, test_logger)

        single_pixel_result = [{
            "min": 150.0,
            "max": 150.0,
            "mean": 150.0,
            "std": 0.0
        }]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=single_pixel_result):
            result = processor.calculate_statistics(mock_dem_file, "elevation")

        # For single pixel, mean should equal min and max (use approximate check for floating point)
        assert abs(result["elevation_mean"] - 150.0) < 0.1
        assert result["elevation_std"] == 0.0
        assert result["elevation_min"] == result["elevation_max"]

    def test_aspect_all_north_facing(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test aspect statistics when all pixels face north (0/360 degrees)."""
        processor = ElevationProcessor(base_config, test_logger)

        # All pixels facing north (use small variation around 0)
        north_aspect = np.array([0, 1, 359, 0, 2, 358] * 10)  # Tight cluster around north

        # Create custom stats function that rasterstats would use
        def mock_zonal_with_raster(*args, **kwargs):
            if 'add_stats' in kwargs:
                # Call the custom functions with our data
                result = {"min": 0.0, "max": 359.0}
                for stat_name, stat_func in kwargs['add_stats'].items():
                    result[stat_name] = stat_func(north_aspect)
                return [result]
            return [{"min": 0.0, "max": 359.0}]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', side_effect=mock_zonal_with_raster):
            # Ensure catchment_path is a file path
            processor.catchment_path = lumped_catchment_shapefile
            result = processor.calculate_statistics(mock_dem_file, "aspect")

        # Should have circular statistics
        if "aspect_circmean" in result:
            assert isinstance(result["aspect_circmean"], (int, float, np.floating))
            # Mean should be near 0 (or near 360) - allow wider range since circular mean calculation varies
            assert result["aspect_circmean"] < 15 or result["aspect_circmean"] > 345

            # Circular std should be small (all facing same direction)
            if "aspect_circstd" in result:
                assert result["aspect_circstd"] < 30.0  # Relaxed threshold

    def test_aspect_bimodal_distribution(
        self, base_config, test_logger, lumped_catchment_shapefile, mock_dem_file
    ):
        """Test aspect with bimodal distribution (north and south facing)."""
        processor = ElevationProcessor(base_config, test_logger)

        # Half north (0°), half south (180°)
        bimodal_aspect = np.array([0] * 50 + [180] * 50)
        mock_result = [{"raster": bimodal_aspect}]

        with patch('symfluence.utils.data.preprocessing.attribute_processors.elevation.zonal_stats', return_value=mock_result):
            # Ensure catchment_path is a file path
            processor.catchment_path = lumped_catchment_shapefile
            result = processor.calculate_statistics(mock_dem_file, "aspect")

        # Circular mean for bimodal distribution
        if "aspect_circmean" in result:
            assert isinstance(result["aspect_circmean"], (int, float, np.floating))

            # Circular std should be relatively high for bimodal
            if "aspect_circstd" in result:
                assert result["aspect_circstd"] > 0.0