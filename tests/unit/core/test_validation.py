"""Tests for symfluence.core.validation module."""

import math
from unittest.mock import MagicMock

import pytest

from symfluence.core.exceptions import (
    ConfigurationError,
    FileOperationError,
    GeospatialError,
    ValidationError,
)
from symfluence.core.validation import (
    validate_bounding_box,
    validate_config_keys,
    validate_date_range,
    validate_directory_exists,
    validate_file_exists,
    validate_netcdf_dimensions,
    validate_netcdf_variables,
    validate_numeric_range,
    validate_positive,
)

# =============================================================================
# validate_config_keys
# =============================================================================

class TestValidateConfigKeys:
    """Tests for validate_config_keys."""

    def test_passes_when_all_present(self):
        config = {"A": 1, "B": 2, "C": 3}
        validate_config_keys(config, ["A", "B"])

    def test_raises_on_missing_key(self):
        config = {"A": 1}
        with pytest.raises(ConfigurationError, match="Missing required"):
            validate_config_keys(config, ["A", "B"])

    def test_error_mentions_missing_keys(self):
        config = {"A": 1}
        with pytest.raises(ConfigurationError, match="X.*Y|Y.*X"):
            validate_config_keys(config, ["X", "Y"])

    def test_error_includes_operation(self):
        with pytest.raises(ConfigurationError, match="my operation"):
            validate_config_keys({}, ["A"], operation="my operation")

    def test_empty_required_keys_passes(self):
        validate_config_keys({}, [])


# =============================================================================
# validate_file_exists
# =============================================================================

class TestValidateFileExists:
    """Tests for validate_file_exists."""

    def test_returns_path_for_existing_file(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("data")
        result = validate_file_exists(f)
        assert result == f

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileOperationError, match="not found"):
            validate_file_exists(tmp_path / "missing.txt")

    def test_raises_when_path_is_directory(self, tmp_path):
        with pytest.raises(FileOperationError, match="not a file"):
            validate_file_exists(tmp_path)

    def test_includes_description_in_error(self, tmp_path):
        with pytest.raises(FileOperationError, match="config file"):
            validate_file_exists(tmp_path / "nope", file_description="config file")

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("data")
        result = validate_file_exists(str(f))
        assert result == f


# =============================================================================
# validate_directory_exists
# =============================================================================

class TestValidateDirectoryExists:
    """Tests for validate_directory_exists."""

    def test_returns_path_for_existing_dir(self, tmp_path):
        result = validate_directory_exists(tmp_path)
        assert result == tmp_path

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileOperationError, match="not found"):
            validate_directory_exists(tmp_path / "missing")

    def test_raises_when_path_is_file(self, tmp_path):
        f = tmp_path / "afile"
        f.write_text("data")
        with pytest.raises(FileOperationError, match="not a directory"):
            validate_directory_exists(f)

    def test_includes_description(self, tmp_path):
        with pytest.raises(FileOperationError, match="output dir"):
            validate_directory_exists(tmp_path / "nope", dir_description="output dir")


# =============================================================================
# validate_bounding_box
# =============================================================================

class TestValidateBoundingBox:
    """Tests for validate_bounding_box."""

    def test_valid_bbox_passes(self):
        bbox = {"lat_min": 49.0, "lat_max": 52.0, "lon_min": -115.0, "lon_max": -113.0}
        result = validate_bounding_box(bbox)
        assert result == bbox

    def test_missing_key_raises(self):
        bbox = {"lat_min": 49.0, "lat_max": 52.0}
        with pytest.raises(ValidationError, match="missing keys"):
            validate_bounding_box(bbox)

    def test_none_value_raises(self):
        bbox = {"lat_min": None, "lat_max": 52.0, "lon_min": -115.0, "lon_max": -113.0}
        with pytest.raises(ValidationError, match="None"):
            validate_bounding_box(bbox)

    def test_nan_value_raises(self):
        bbox = {"lat_min": float("nan"), "lat_max": 52.0, "lon_min": -115.0, "lon_max": -113.0}
        with pytest.raises(ValidationError, match="NaN"):
            validate_bounding_box(bbox)

    def test_lat_out_of_range_raises(self):
        bbox = {"lat_min": -100.0, "lat_max": 52.0, "lon_min": -115.0, "lon_max": -113.0}
        with pytest.raises(ValidationError, match="out of range"):
            validate_bounding_box(bbox)

    def test_lat_min_gte_lat_max_raises(self):
        bbox = {"lat_min": 52.0, "lat_max": 49.0, "lon_min": -115.0, "lon_max": -113.0}
        with pytest.raises(ValidationError, match="lat_min"):
            validate_bounding_box(bbox)

    def test_lon_out_of_range_raises(self):
        bbox = {"lat_min": 49.0, "lat_max": 52.0, "lon_min": -200.0, "lon_max": -113.0}
        with pytest.raises(ValidationError, match="out of valid range"):
            validate_bounding_box(bbox)

    def test_small_bbox_warns(self):
        logger = MagicMock()
        bbox = {"lat_min": 49.0, "lat_max": 49.0005, "lon_min": -115.0, "lon_max": -114.9995}
        validate_bounding_box(bbox, logger=logger)
        logger.warning.assert_called()

    def test_large_bbox_warns_when_not_global(self):
        logger = MagicMock()
        bbox = {"lat_min": -80.0, "lat_max": 80.0, "lon_min": -170.0, "lon_max": 170.0}
        validate_bounding_box(bbox, logger=logger)
        logger.warning.assert_called()

    def test_large_bbox_no_warn_when_allow_global(self):
        logger = MagicMock()
        bbox = {"lat_min": -80.0, "lat_max": 80.0, "lon_min": -170.0, "lon_max": 170.0}
        validate_bounding_box(bbox, allow_global=True, logger=logger)
        logger.warning.assert_not_called()


# =============================================================================
# validate_numeric_range
# =============================================================================

class TestValidateNumericRange:
    """Tests for validate_numeric_range."""

    def test_value_in_range(self):
        assert validate_numeric_range(5, min_val=0, max_val=10) == 5

    def test_value_at_min_boundary(self):
        assert validate_numeric_range(0, min_val=0) == 0

    def test_value_at_max_boundary(self):
        assert validate_numeric_range(10, max_val=10) == 10

    def test_below_min_raises(self):
        with pytest.raises(ValidationError, match="below minimum"):
            validate_numeric_range(-1, min_val=0)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_numeric_range(11, max_val=10)

    def test_none_value_raises(self):
        with pytest.raises(ValidationError, match="cannot be None"):
            validate_numeric_range(None, min_val=0)

    def test_nan_value_raises(self):
        with pytest.raises(ValidationError, match="invalid value"):
            validate_numeric_range(float("nan"))

    def test_inf_value_raises(self):
        with pytest.raises(ValidationError, match="invalid value"):
            validate_numeric_range(float("inf"))

    def test_no_bounds_passes(self):
        assert validate_numeric_range(999) == 999


# =============================================================================
# validate_positive
# =============================================================================

class TestValidatePositive:
    """Tests for validate_positive."""

    def test_positive_passes(self):
        assert validate_positive(5) == 5

    def test_zero_raises_by_default(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive(0)

    def test_zero_passes_with_allow_zero(self):
        assert validate_positive(0, allow_zero=True) == 0

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive(-1)

    def test_negative_raises_with_allow_zero(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_positive(-1, allow_zero=True)


# =============================================================================
# validate_date_range
# =============================================================================

class TestValidateDateRange:
    """Tests for validate_date_range."""

    def test_valid_dates_pass(self):
        start, end = validate_date_range("2020-01-01", "2020-12-31")
        assert start.year == 2020
        assert end.month == 12

    def test_start_after_end_raises(self):
        with pytest.raises(ValidationError, match="must be before"):
            validate_date_range("2021-01-01", "2020-01-01")

    def test_equal_dates_raises(self):
        with pytest.raises(ValidationError, match="must be before"):
            validate_date_range("2020-01-01", "2020-01-01")

    def test_none_start_raises(self):
        with pytest.raises(ValidationError, match="required"):
            validate_date_range(None, "2020-01-01")

    def test_none_end_raises(self):
        with pytest.raises(ValidationError, match="required"):
            validate_date_range("2020-01-01", None)

    def test_unparseable_date_raises(self):
        with pytest.raises(ValidationError, match="Could not parse"):
            validate_date_range("not-a-date", "2020-01-01")

    def test_max_span_exceeded_raises(self):
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_date_range("2000-01-01", "2020-01-01", max_span_days=365)

    def test_max_span_within_limit_passes(self):
        start, end = validate_date_range("2020-01-01", "2020-06-01", max_span_days=365)
        assert start is not None

    def test_long_span_warns(self):
        logger = MagicMock()
        validate_date_range("1900-01-01", "2020-01-01", logger=logger)
        logger.warning.assert_called()


# =============================================================================
# validate_netcdf_variables
# =============================================================================

class TestValidateNetcdfVariables:
    """Tests for validate_netcdf_variables."""

    def _make_xarray_ds(self):
        """Create a mock xarray-like dataset."""
        ds = MagicMock()
        ds.data_vars = {"temp", "precip", "wind"}
        return ds

    def _make_netcdf4_ds(self):
        """Create a mock netCDF4-like dataset."""
        ds = MagicMock(spec=[])
        ds.variables = MagicMock()
        ds.variables.keys = MagicMock(return_value=["temp", "precip", "wind"])
        del ds.data_vars  # Ensure xarray path is not taken
        return ds

    def test_all_required_present_xarray(self):
        ds = self._make_xarray_ds()
        present = validate_netcdf_variables(ds, ["temp", "precip"])
        assert set(present) == {"temp", "precip"}

    def test_missing_variable_raises(self):
        ds = self._make_xarray_ds()
        with pytest.raises(ValidationError, match="missing required"):
            validate_netcdf_variables(ds, ["temp", "pressure"])

    def test_any_of_mode_passes_with_one(self):
        ds = self._make_xarray_ds()
        present = validate_netcdf_variables(ds, ["temp", "pressure"], any_of=True)
        assert present == ["temp"]

    def test_any_of_mode_raises_when_none_present(self):
        ds = self._make_xarray_ds()
        with pytest.raises(ValidationError, match="missing all"):
            validate_netcdf_variables(ds, ["pressure", "humidity"], any_of=True)

    def test_netcdf4_interface(self):
        ds = self._make_netcdf4_ds()
        present = validate_netcdf_variables(ds, ["temp"])
        assert present == ["temp"]

    def test_unknown_dataset_type_raises(self):
        ds = object()
        with pytest.raises(ValidationError, match="Unknown dataset"):
            validate_netcdf_variables(ds, ["temp"])


# =============================================================================
# validate_netcdf_dimensions
# =============================================================================

class TestValidateNetcdfDimensions:
    """Tests for validate_netcdf_dimensions."""

    def test_all_dims_present(self):
        ds = MagicMock()
        ds.dims = {"time", "lat", "lon"}
        validate_netcdf_dimensions(ds, ["time", "lat"])

    def test_missing_dim_raises(self):
        ds = MagicMock()
        ds.dims = {"time", "lat"}
        with pytest.raises(ValidationError, match="missing required dimensions"):
            validate_netcdf_dimensions(ds, ["time", "lon"])

    def test_netcdf4_interface(self):
        ds = MagicMock(spec=[])
        ds.dimensions = MagicMock()
        ds.dimensions.keys = MagicMock(return_value=["time", "lat", "lon"])
        del ds.dims
        validate_netcdf_dimensions(ds, ["time"])

    def test_unknown_dataset_raises(self):
        ds = object()
        with pytest.raises(ValidationError, match="Unknown dataset"):
            validate_netcdf_dimensions(ds, ["time"])
