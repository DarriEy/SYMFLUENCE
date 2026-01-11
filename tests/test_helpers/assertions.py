"""
Custom assertion helpers for SYMFLUENCE tests.

Provides domain-specific assertion functions for validating
SYMFLUENCE outputs and behaviors.
"""

from pathlib import Path
import xarray as xr
import numpy as np


def assert_netcdf_has_variables(nc_path: Path, expected_vars: list):
    """
    Assert that a NetCDF file contains all expected variables.

    Args:
        nc_path: Path to NetCDF file
        expected_vars: List of variable names that should be present

    Raises:
        AssertionError: If any expected variables are missing
    """
    with xr.open_dataset(nc_path) as ds:
        found_vars = list(ds.data_vars.keys())
        missing_vars = [v for v in expected_vars if v not in found_vars]
        assert not missing_vars, f"Missing variables in {nc_path}: {missing_vars}"


def assert_netcdf_dimensions(nc_path: Path, expected_dims: dict):
    """
    Assert that a NetCDF file has expected dimensions and sizes.

    Args:
        nc_path: Path to NetCDF file
        expected_dims: Dict mapping dimension names to expected sizes
                      Use None for dimensions that should exist but size is flexible

    Raises:
        AssertionError: If dimensions don't match expectations
    """
    with xr.open_dataset(nc_path) as ds:
        for dim_name, expected_size in expected_dims.items():
            assert dim_name in ds.dims, f"Missing dimension '{dim_name}' in {nc_path}"
            if expected_size is not None:
                actual_size = ds.dims[dim_name]
                assert actual_size == expected_size, (
                    f"Dimension '{dim_name}' size mismatch in {nc_path}: "
                    f"expected {expected_size}, got {actual_size}"
                )


def assert_simulation_outputs_exist(
    sim_dir: Path,
    model: str,
    expected_patterns: list = None
):
    """
    Assert that simulation output files exist for a given model.

    Args:
        sim_dir: Simulation output directory
        model: Model name (SUMMA, FUSE, NGEN, etc.)
        expected_patterns: List of glob patterns for expected output files
                          If None, uses default patterns for the model

    Raises:
        AssertionError: If expected output files don't exist
    """
    if expected_patterns is None:
        # Default patterns for each model
        patterns_by_model = {
            "SUMMA": ["*_timestep.nc", "*_day.nc"],
            "FUSE": ["*_timestep.nc"],
            "NGEN": ["*.csv"],
            "GR": ["*.csv"],
        }
        expected_patterns = patterns_by_model.get(model, ["*"])

    for pattern in expected_patterns:
        files = list(sim_dir.glob(pattern))
        assert files, f"No files matching '{pattern}' found in {sim_dir} for {model}"


def assert_calibration_outputs_exist(calib_dir: Path):
    """
    Assert that calibration output files exist.

    Args:
        calib_dir: Calibration output directory

    Raises:
        AssertionError: If expected calibration files don't exist
    """
    assert calib_dir.exists(), f"Calibration directory doesn't exist: {calib_dir}"

    # Check for typical calibration outputs
    expected_files = [
        "best_parameters.txt",
        "optimization_history.csv",
    ]

    for filename in expected_files:
        filepath = calib_dir / filename
        # Check if any variant exists (some might have prefixes/suffixes)
        matching_files = list(calib_dir.glob(f"*{filename}*"))
        assert matching_files, f"No calibration output matching '{filename}' in {calib_dir}"


def assert_preprocessing_outputs_exist(forcing_dir: Path):
    """
    Assert that preprocessing output files exist.

    Args:
        forcing_dir: Forcing directory with preprocessed data

    Raises:
        AssertionError: If expected preprocessing outputs don't exist
    """
    # Check for basin-averaged data
    basin_avg_dir = forcing_dir / "basin_averaged_data"
    if basin_avg_dir.exists():
        nc_files = list(basin_avg_dir.glob("*.nc"))
        assert nc_files, f"No NetCDF files found in {basin_avg_dir}"

    # Check for model-specific inputs
    summa_input_dir = forcing_dir.parent / "forcing" / "SUMMA_input"
    if summa_input_dir.exists():
        summa_files = list(summa_input_dir.glob("*.nc"))
        assert summa_files, f"No SUMMA input files found in {summa_input_dir}"
