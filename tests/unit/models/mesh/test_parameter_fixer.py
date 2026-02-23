"""
Tests for MESH parameter fixer.

Tests cover run options fixes, GRU count mismatch handling,
DDB operations, CLASS file operations, and safe forcing creation.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forcing_dir(tmp_path):
    """Create a temporary forcing directory."""
    d = tmp_path / "forcing"
    d.mkdir()
    return d


@pytest.fixture
def setup_dir(tmp_path):
    """Create a temporary settings directory."""
    d = tmp_path / "settings"
    d.mkdir()
    return d


@pytest.fixture
def fixer(forcing_dir, setup_dir):
    """Create a MESHParameterFixer with default config."""
    from symfluence.models.mesh.preprocessing.parameter_fixer import MESHParameterFixer

    return MESHParameterFixer(
        forcing_dir=forcing_dir,
        setup_dir=setup_dir,
        config={
            "HYDROLOGICAL_MODEL": "MESH",
            "MESH_SPATIAL_MODE": "distributed",
            "MESH_SPINUP_DAYS": 365,
        },
    )


@pytest.fixture
def run_options_content():
    """Sample MESH_input_run_options.ini content."""
    return """\
##### Global settings #####
 15 # Number of control flags
----# Control flags
BASINFORCINGFLAG      nc_subbasin
SHDFILEFLAG           nc_subbasin pad_outlets
OUTFILESFLAG         daily
OUTFIELDSFLAG        none
STREAMFLOWOUTFLAG     csv
BASINAVGWBFILEFLAG    daily
RUNMODE               runrte
FROZENSOILINFILFLAG   0
FREZTH                0.0
SWELIM                800.0
SNDENLIM              600.0
PBSMFLAG              off
METRICSSPINUP         730
PRINTSIMSTATUS        date_monthly
DIAGNOSEMODE          off
#####
name_var=SWRadAtm
name_var=spechum
name_var=airtemp
name_var=windspd
name_var=pptrate
name_var=airpres
name_var=LWRadAtm
"""


@pytest.fixture
def class_ini_content():
    """Sample MESH_parameters_CLASS.ini content with 2 GRU blocks."""
    return """\
  51.0  -116.0  1.0  1.0  0.1  1.0  0.0  1  2  04 DEGLAT/DEGLON/ZBLDGRD/ZRFHGRD/ZRFMGRD/GCGRD/FAREROT/NL/NM
 0.0  0.0  1.0  0.0  0.0  3.5  1.0  0.0  0.0  1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0  05 5xFCAN/4xLAMX/3xLNZ0/SDEP/XSLP/XDRAINH/MANN/KSAT/MID
 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  06 SANDG/CLAYG/ORGM/...
 0.1  0.2  0.3  0.4  0.5  07 CMIDROT/...
 0.1  0.2  0.3  0.4  08 ROOT/...
 0.0  0.0  1.0  0.0  0.0  2.0  1.0  0.0  0.0  1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0  05 5xFCAN/4xLAMX/3xLNZ0/SDEP/XSLP/XDRAINH/MANN/KSAT/MID
 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0  06 SANDG/CLAYG/ORGM/...
 0.1  0.2  0.3  0.4  0.5  07 CMIDROT/...
 0.1  0.2  0.3  0.4  08 ROOT/...
 0  0  0  0  20 IORGC
 0.0 0.0  21 RSMLNa/RSMLNb
 0.0 0.0  22 INITIALS
"""


@pytest.fixture
def ddb_dataset():
    """Create a minimal DDB xarray Dataset for testing."""
    n_sub = 1
    n_gru = 3
    gru_data = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)

    return xr.Dataset({
        "GRU": (["subbasin", "NGRU"], gru_data),
        "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        "GridArea": (["subbasin"], np.array([1e8])),
        "lat": (["subbasin"], np.array([51.0])),
        "lon": (["subbasin"], np.array([-116.0])),
    })


# ---------------------------------------------------------------------------
# TestRunOptionsVarNames
# ---------------------------------------------------------------------------

class TestRunOptionsVarNames:
    """Test fix_run_options_var_names."""

    def test_replaces_variable_names(self, fixer, run_options_content):
        """Old-style variable names should be replaced with MESH names."""
        fixer.run_options_path.write_text(run_options_content)
        fixer.fix_run_options_var_names()

        content = fixer.run_options_path.read_text()
        assert "name_var=FSIN" in content
        assert "name_var=QA" in content
        assert "name_var=TA" in content
        assert "name_var=UV" in content
        assert "name_var=PRE" in content
        assert "name_var=PRES" in content
        assert "name_var=FLIN" in content
        # Old names should be gone
        assert "name_var=SWRadAtm" not in content
        assert "name_var=spechum" not in content

    def test_idempotent(self, fixer, run_options_content):
        """Running twice should produce the same output."""
        fixer.run_options_path.write_text(run_options_content)
        fixer.fix_run_options_var_names()
        content1 = fixer.run_options_path.read_text()
        fixer.fix_run_options_var_names()
        content2 = fixer.run_options_path.read_text()
        assert content1 == content2

    def test_missing_file_no_error(self, fixer):
        """Should silently return if run_options file doesn't exist."""
        fixer.fix_run_options_var_names()  # No exception


class TestRunOptionsSnowParams:
    """Test fix_run_options_snow_params."""

    def test_single_cell_forces_noroute(self, fixer, run_options_content, ddb_dataset):
        """Single-cell domain should force RUNMODE=noroute."""
        # Create single-cell DDB
        ddb_dataset.to_netcdf(fixer.ddb_path)
        fixer.run_options_path.write_text(run_options_content)

        fixer.fix_run_options_snow_params()

        content = fixer.run_options_path.read_text()
        assert "noroute" in content

    def test_frozen_soil_flag(self, forcing_dir, setup_dir, run_options_content):
        """MESH_ENABLE_FROZEN_SOIL=True should set FROZENSOILINFILFLAG=1."""
        from symfluence.models.mesh.preprocessing.parameter_fixer import MESHParameterFixer

        fixer = MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_ENABLE_FROZEN_SOIL": True,
                "MESH_SPINUP_DAYS": 365,
            },
        )
        fixer.run_options_path.write_text(run_options_content)

        fixer.fix_run_options_snow_params()

        content = fixer.run_options_path.read_text()
        assert "FROZENSOILINFILFLAG   1" in content

    def test_missing_file_no_error(self, fixer):
        """Should silently return if run_options file doesn't exist."""
        fixer.fix_run_options_snow_params()


class TestUpdateControlFlagCount:
    """Test _update_control_flag_count."""

    def test_counts_flags_correctly(self, fixer, run_options_content):
        """Should count non-comment, non-empty lines in the flags section."""
        fixer.run_options_path.write_text(run_options_content)
        fixer._update_control_flag_count()
        content = fixer.run_options_path.read_text()
        # Should have the correct count of flags
        lines = content.split("\n")
        for line in lines:
            if "Number of control flags" in line:
                # Extract the number
                import re
                match = re.search(r"(\d+)", line)
                if match:
                    count = int(match.group(1))
                    assert count > 0
                break


# ---------------------------------------------------------------------------
# TestGRUCountMismatch
# ---------------------------------------------------------------------------

class TestGRUCountMismatch:
    """Test fix_gru_count_mismatch orchestration."""

    def test_collapse_to_single_gru(self, forcing_dir, setup_dir, ddb_dataset, class_ini_content):
        """MESH_FORCE_SINGLE_GRU should collapse to 1 active GRU."""
        from symfluence.models.mesh.preprocessing.parameter_fixer import MESHParameterFixer

        fixer = MESHParameterFixer(
            forcing_dir=forcing_dir,
            setup_dir=setup_dir,
            config={
                "HYDROLOGICAL_MODEL": "MESH",
                "MESH_FORCE_SINGLE_GRU": True,
            },
        )
        ddb_dataset.to_netcdf(fixer.ddb_path)
        fixer.class_file_path.write_text(class_ini_content)

        fixer.fix_gru_count_mismatch()

        with xr.open_dataset(fixer.ddb_path) as ds:
            assert ds.sizes["NGRU"] == 2  # MESH off-by-one: 2 cols → reads 1

    def test_idempotent_when_aligned(self, fixer, class_ini_content):
        """When already aligned, no changes should be made."""
        # Create DDB with 3 GRU cols → MESH reads 2 → needs 2 CLASS blocks
        gru_data = np.array([[0.7, 0.3, 0.0]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
            "GridArea": (["subbasin"], np.array([1e8])),
            "lat": (["subbasin"], np.array([51.0])),
            "lon": (["subbasin"], np.array([-116.0])),
        })
        ds.to_netcdf(fixer.ddb_path)
        fixer.class_file_path.write_text(class_ini_content)

        fixer.fix_gru_count_mismatch()

        # Should not crash, file should still exist
        assert fixer.ddb_path.exists()
        assert fixer.class_file_path.exists()


# ---------------------------------------------------------------------------
# TestDDBOperations
# ---------------------------------------------------------------------------

class TestDDBOperations:
    """Test DDB-related methods on MESHParameterFixer."""

    def test_get_ddb_gru_count(self, fixer, ddb_dataset):
        """Should return NGRU dimension size."""
        ddb_dataset.to_netcdf(fixer.ddb_path)
        assert fixer._get_ddb_gru_count() == 3

    def test_get_ddb_gru_count_missing_file(self, fixer):
        """Should return None when DDB doesn't exist."""
        assert fixer._get_ddb_gru_count() is None

    def test_trim_to_active_grus(self, fixer, ddb_dataset):
        """Should trim DDB to target count and renormalize."""
        ddb_dataset.to_netcdf(fixer.ddb_path)

        fixer._trim_ddb_to_active_grus(2)

        with xr.open_dataset(fixer.ddb_path) as ds:
            assert ds.sizes["NGRU"] == 2
            # Fractions should be renormalized to sum to 1
            gru_sum = ds["GRU"].sum("NGRU").values[0]
            assert gru_sum == pytest.approx(1.0, abs=0.01)

    def test_trim_no_op_when_at_target(self, fixer, ddb_dataset):
        """Should do nothing when already at target count."""
        ddb_dataset.to_netcdf(fixer.ddb_path)

        fixer._trim_ddb_to_active_grus(5)  # target > current (3)

        with xr.open_dataset(fixer.ddb_path) as ds:
            assert ds.sizes["NGRU"] == 3  # Unchanged

    def test_ensure_gru_normalization(self, fixer):
        """Should normalize GRU fractions to sum to 1.0."""
        gru_data = np.array([[0.5, 0.3]], dtype=np.float64)  # sum = 0.8
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._ensure_gru_normalization()

        with xr.open_dataset(fixer.ddb_path) as ds:
            gru_sum = ds["GRU"].sum("NGRU").values[0]
            assert gru_sum == pytest.approx(1.0, abs=0.001)

    def test_renormalize_mesh_active_grus(self, fixer):
        """Should renormalize only the first N active GRU columns."""
        gru_data = np.array([[0.4, 0.3, 0.1]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._renormalize_mesh_active_grus(2)  # Only first 2 cols

        with xr.open_dataset(fixer.ddb_path) as ds:
            gru = ds["GRU"].values[0]
            # First 2 should sum to 1.0
            assert gru[:2].sum() == pytest.approx(1.0, abs=0.001)
            # Third column should be zeroed
            assert gru[2] == pytest.approx(0.0, abs=0.001)

    def test_off_by_one_ngru_count(self, fixer):
        """MESH reads NGRU-1: 3 cols → reads 2 active GRUs."""
        gru_data = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        count = fixer._get_mesh_active_gru_count()
        assert count == 2  # NGRU=3, MESH reads 3-1=2

    def test_get_num_cells(self, fixer, ddb_dataset):
        """Should return number of subbasins in DDB."""
        ddb_dataset.to_netcdf(fixer.ddb_path)
        assert fixer._get_num_cells() == 1

    def test_get_num_cells_missing_file(self, fixer):
        """Should return 1 when DDB doesn't exist."""
        assert fixer._get_num_cells() == 1

    def test_get_spatial_dim(self, fixer):
        """Should detect subbasin or N dimension."""
        ds1 = xr.Dataset({"x": (["subbasin"], [1])})
        assert fixer._get_spatial_dim(ds1) == "subbasin"

        ds2 = xr.Dataset({"x": (["N"], [1])})
        assert fixer._get_spatial_dim(ds2) == "N"

        ds3 = xr.Dataset({"x": (["gridcell"], [1])})
        assert fixer._get_spatial_dim(ds3) is None


# ---------------------------------------------------------------------------
# TestCLASSFileOperations
# ---------------------------------------------------------------------------

class TestCLASSFileOperations:
    """Test CLASS .ini file operations."""

    def test_get_class_block_count(self, fixer, class_ini_content):
        """Should count blocks via XSLP/XDRAINH/MANN/KSAT/MID marker."""
        fixer.class_file_path.write_text(class_ini_content)
        assert fixer._get_class_block_count() == 2

    def test_get_class_block_count_missing_file(self, fixer):
        """Should return None when file doesn't exist."""
        assert fixer._get_class_block_count() is None

    def test_read_nm_from_lines_legacy(self, fixer):
        """Should parse NM from legacy format (9th column of line 04)."""
        lines = [
            "  51.0  -116.0  1.0  1.0  0.1  1.0  0.0  1  3  04 DEGLAT/DEGLON/ZBLDGRD/ZRFHGRD/ZRFMGRD/GCGRD/FAREROT/NL/NM"
        ]
        assert fixer._read_nm_from_lines(lines) == 3

    def test_read_nm_from_lines_ini_style(self, fixer):
        """Should parse NM from ini-style 'NM x' format."""
        lines = ["NM 5    ! number of landcover classes (GRUs)"]
        assert fixer._read_nm_from_lines(lines) == 5

    def test_update_class_nm_legacy(self, fixer, class_ini_content):
        """Should update NM in legacy CLASS format."""
        fixer.class_file_path.write_text(class_ini_content)
        fixer._update_class_nm(5)

        content = fixer.class_file_path.read_text()
        lines = content.split("\n")
        for line in lines:
            if "04 DEGLAT" in line:
                parts = line.split()
                assert parts[8] == "5"
                break

    def test_update_class_nm_ini_style(self, fixer):
        """Should update NM in ini-style format."""
        fixer.class_file_path.write_text("NM 2    ! number of landcover classes (GRUs)\n")
        fixer._update_class_nm(7)

        content = fixer.class_file_path.read_text()
        assert "NM 7" in content

    def test_trim_class_to_count(self, fixer, class_ini_content):
        """Should keep only the first N CLASS blocks."""
        fixer.class_file_path.write_text(class_ini_content)
        fixer._trim_class_to_count(1)

        content = fixer.class_file_path.read_text()
        # Only 1 block marker should remain
        assert content.count("05 5xFCAN/4xLAMX") == 1


# ---------------------------------------------------------------------------
# TestSafeForcing
# ---------------------------------------------------------------------------

class TestSafeForcing:
    """Test create_safe_forcing method."""

    def test_creates_trimmed_forcing(self, fixer):
        """Should create a safe forcing file trimmed to simulation period."""
        from datetime import datetime

        import pandas as pd

        # Create minimal forcing file
        times = pd.date_range("2019-06-01", "2021-06-30", freq="h")
        ds = xr.Dataset(
            {
                "FSIN": (["subbasin", "time"], np.random.rand(1, len(times))),
                "PRE": (["subbasin", "time"], np.random.rand(1, len(times))),
            },
            coords={"time": times, "subbasin": [1]},
        )
        ds["time"].encoding["units"] = "hours since 1900-01-01"
        ds["time"].encoding["calendar"] = "standard"
        forcing_path = fixer.forcing_dir / "MESH_forcing.nc"
        ds.to_netcdf(forcing_path)

        # Return datetime objects, not strings — create_safe_forcing does timedelta arithmetic
        fixer.get_simulation_time_window = lambda: (
            datetime(2020, 1, 1), datetime(2020, 12, 31)
        )
        fixer.create_safe_forcing()

        safe_path = fixer.forcing_dir / "MESH_forcing_safe.nc"
        assert safe_path.exists()

        with xr.open_dataset(safe_path) as ds_safe:
            # Should be trimmed to roughly the simulation period (with spinup)
            assert len(ds_safe.time) < len(times)

    def test_no_crash_missing_forcing(self, fixer):
        """Should not crash when forcing file doesn't exist."""
        fixer.get_simulation_time_window = lambda: ("2020-01-01", "2020-12-31")
        fixer.create_safe_forcing()  # Should log warning and return


# ---------------------------------------------------------------------------
# TestElevationBandBlocks
# ---------------------------------------------------------------------------

class TestElevationBandBlocks:
    """Test create_elevation_band_class_blocks."""

    def test_creates_correct_block_count(self, fixer, class_ini_content, ddb_dataset):
        """Should create one CLASS block per elevation band."""
        fixer.class_file_path.write_text(class_ini_content)
        ddb_dataset.to_netcdf(fixer.ddb_path)

        elevation_info = [
            {"elevation": 1500.0, "fraction": 0.3},
            {"elevation": 2000.0, "fraction": 0.4},
            {"elevation": 2500.0, "fraction": 0.3},
        ]

        fixer.create_elevation_band_class_blocks(elevation_info)

        content = fixer.class_file_path.read_text()
        # Should have exactly 3 blocks
        block_count = content.count("05 5xFCAN/4xLAMX") + content.count("[GRU_")
        assert block_count == 3

    def test_missing_class_file_no_error(self, fixer, ddb_dataset):
        """Should not crash when CLASS file doesn't exist."""
        ddb_dataset.to_netcdf(fixer.ddb_path)
        elevation_info = [{"elevation": 1500.0, "fraction": 1.0}]
        fixer.create_elevation_band_class_blocks(elevation_info)


# ---------------------------------------------------------------------------
# TestRemoveSmallGRUs
# ---------------------------------------------------------------------------

class TestRemoveSmallGRUs:
    """Test _remove_small_grus method."""

    def test_removes_below_threshold(self, fixer, class_ini_content):
        """GRUs below 5% threshold should be removed."""
        gru_data = np.array([[0.7, 0.27, 0.03]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)
        fixer.class_file_path.write_text(class_ini_content)

        fixer._remove_small_grus()

        with xr.open_dataset(fixer.ddb_path) as ds_out:
            # Third GRU (3%) should be removed
            assert ds_out.sizes["NGRU"] == 2

    def test_keeps_all_above_threshold(self, fixer):
        """No GRUs should be removed when all above threshold."""
        gru_data = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._remove_small_grus()

        with xr.open_dataset(fixer.ddb_path) as ds_out:
            assert ds_out.sizes["NGRU"] == 3

    def test_keeps_largest_when_all_below(self, fixer):
        """When all GRUs below threshold, keep the largest."""
        gru_data = np.array([[0.02, 0.04, 0.01]], dtype=np.float64)
        ds = xr.Dataset({
            "GRU": (["subbasin", "NGRU"], gru_data),
            "Rank": (["subbasin"], np.array([1], dtype=np.int32)),
            "Next": (["subbasin"], np.array([0], dtype=np.int32)),
        })
        ds.to_netcdf(fixer.ddb_path)

        fixer._remove_small_grus()

        with xr.open_dataset(fixer.ddb_path) as ds_out:
            # Should keep at least one GRU (the largest)
            assert ds_out.sizes["NGRU"] >= 1
