"""
Unit tests for MESH Parameter Manager

Tests that the MESH parameter manager can:
1. Load parameter bounds from registry
2. Normalize and denormalize parameters correctly
3. Handle parameter initialization
"""

import logging
import tempfile
from pathlib import Path

import pytest

from symfluence.optimization.parameter_managers import MESHParameterManager


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_logger')


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mesh_config(temp_dir):
    """Create a basic MESH configuration."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'MESH_PARAMS_TO_CALIBRATE': 'ZSNL,MANN,RCHARG',
    }


class TestMESHParameterManager:
    """Tests for MESH Parameter Manager."""

    def test_can_instantiate(self, mesh_config, logger, temp_dir):
        """Test that MESHParameterManager can be instantiated."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        assert manager is not None
        assert manager.domain_name == 'test_domain'
        assert manager.experiment_id == 'test_exp'

    def test_parameter_names(self, mesh_config, logger, temp_dir):
        """Test that parameter names are parsed correctly."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        param_names = manager._get_parameter_names()
        assert param_names == ['ZSNL', 'MANN', 'RCHARG']

    def test_load_bounds(self, mesh_config, logger, temp_dir):
        """Test that parameter bounds can be loaded from registry."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        bounds = manager._load_parameter_bounds()

        assert 'ZSNL' in bounds
        assert 'MANN' in bounds
        assert 'RCHARG' in bounds

        # Check specific bounds
        assert bounds['ZSNL']['min'] == 0.001
        assert bounds['ZSNL']['max'] == 0.1
        assert bounds['MANN']['min'] == 0.01
        assert bounds['MANN']['max'] == 0.3

    def test_normalize_denormalize(self, mesh_config, logger, temp_dir):
        """Test parameter normalization and denormalization roundtrip."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)

        # Test parameters
        params = {'ZSNL': 0.05, 'MANN': 0.15, 'RCHARG': 0.5}

        # Normalize
        normalized = manager.normalize_parameters(params)
        assert len(normalized) == 3

        # All normalized values should be between 0 and 1
        for val in normalized:
            assert 0 <= val <= 1

        # Denormalize
        denorm = manager.denormalize_parameters(normalized)

        # Check roundtrip accuracy
        for param_name in params:
            assert abs(denorm[param_name] - params[param_name]) < 1e-6

    def test_get_default_initial_values(self, mesh_config, logger, temp_dir):
        """Test that default initial values are midpoint of bounds."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)
        defaults = manager._get_default_initial_values()

        # ZSNL bounds: 0.001 to 0.1, midpoint should be ~0.0505
        assert abs(defaults['ZSNL'] - 0.0505) < 1e-6

        # MANN bounds: 0.01 to 0.3, midpoint should be 0.155
        assert abs(defaults['MANN'] - 0.155) < 1e-6

        # RCHARG bounds: 0.0 to 1.0, midpoint should be 0.5
        assert abs(defaults['RCHARG'] - 0.5) < 1e-6

    def test_param_file_mapping(self, mesh_config, logger, temp_dir):
        """Test that parameters are mapped to correct files."""
        manager = MESHParameterManager(mesh_config, logger, temp_dir)

        assert manager.param_file_map['ZSNL'] == 'hydrology'
        assert manager.param_file_map['MANN'] == 'hydrology'
        assert manager.param_file_map['FRZTH'] == 'hydrology'
        assert manager.param_file_map['RCHARG'] == 'hydrology'
        assert manager.param_file_map['DTMINUSR'] == 'routing'


# -----------------------------------------------------------------------
# Tests for parameter update behaviour (Bug fixes: static KGE during
# calibration caused by parameters not being written)
# -----------------------------------------------------------------------

@pytest.fixture
def class_ini_content():
    """Standard meshflow-generated CLASS.ini file (22 lines)."""
    return (
        "  MESH Model\n"
        "  MESHFlow\n"
        "  University of Calgary, Canada\n"
        "  51.40  -115.60      10.0      2.0      50.0   -1.0    1    1    1\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100   0.100  05 5xFCAN/4xLAMX\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100   0.100  06 5xLNZ0/4xLAMN\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100   0.100  07 5xALVC/4xCMAS\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100   0.100  08 5xALIC/4xROOT\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100  09 4xRSMN/4xQA50\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100  10 4xVPDA/4xVPDB\n"
        "   0.000   0.000   0.000   0.000   0.100   0.100   0.100   0.100  11 4xPSGA/4xPSGB\n"
        "   1.000   2.500   1.000  50.000  12 DRN/SDEP/FARE/DD\n"
        "   0.030   0.350   0.100   0.050   100 Temp_sub-_gras  13 XSLP/XDRAINH/MANN/KSAT/MID\n"
        "  50.000  30.000  20.000  14 3xSAND\n"
        "  20.000  25.000  30.000  15 3xCLAY\n"
        "   5.000   3.000   1.000  16 3xORGM\n"
        "   4.000   2.000   1.000   2.000   0.000   4.000  17 3xTBAR/TCAN/TSNO/TPND\n"
        "   0.200   0.200   0.200   0.000   0.000   0.000   0.000  18 3xTHLQ/3xTHIC/ZPND\n"
        "   0.000   0.000 100.000   0.750 250.000   1.000  19 RCAN/SCAN/SNO/ALBS/RHOS/GRO\n"
        "   0   0   0   0  20\n"
        "   0   0   0   0  21\n"
        "   0   0   0   0  22 IHOUR/IMINS/IJDAY/IYEAR\n"
    )


@pytest.fixture
def hydro_ini_content():
    """Standard meshflow-generated hydrology.ini file."""
    return (
        "2.0: MESH Hydrology parameters input file (Version 2.0)\n"
        "\n"
        "##### Option Flags #####\n"
        "----#\n"
        "    0\n"
        "\n"
        "##### Channel routing parameters per river class #####\n"
        "-------#\n"
        "       4\n"
        "R2N    0.400\n"
        "R1N    0.020\n"
        "PWR    2.370\n"
        "FLZ    0.001\n"
        "\n"
        "##### GRU class independent hydrologic parameters #####\n"
        "-------#\n"
        "       0\n"
        "\n"
        "##### GRU class dependent hydrologic parameters #####\n"
        "-------#\n"
        "       3\n"
        "!     10\n"
        "ZSNL  0.050\n"
        "ZPLS  0.060\n"
        "ZPLG  0.200\n"
    )


class TestMESHParameterUpdates:
    """Tests that parameter files are correctly written during calibration."""

    def test_update_class_file_applies_params(
        self, mesh_config, logger, temp_dir, class_ini_content
    ):
        """CLASS parameters are written to correct positions."""
        class_file = temp_dir / 'MESH_parameters_CLASS.ini'
        class_file.write_text(class_ini_content)

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'KSAT,DRN,SDEP,XSLP'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        params = {'KSAT': 5.5, 'DRN': 2.0, 'SDEP': 1.8, 'XSLP': 0.015}
        result = mgr._update_class_file(class_file, params)
        assert result is True

        lines = class_file.read_text().splitlines()
        # Line 11 (0-indexed) should now have DRN=2.0, SDEP=1.8
        drn_sdep_line = lines[11].split()
        assert float(drn_sdep_line[0]) == pytest.approx(2.0, abs=0.01)
        assert float(drn_sdep_line[1]) == pytest.approx(1.8, abs=0.01)

        # Line 12 (0-indexed) should now have XSLP=0.015, KSAT=5.5
        xslp_line = lines[12].split()
        assert float(xslp_line[0]) == pytest.approx(0.015, abs=0.001)
        assert float(xslp_line[3]) == pytest.approx(5.5, abs=0.1)

    def test_update_class_file_returns_false_when_no_updates(
        self, mesh_config, logger, temp_dir
    ):
        """_update_class_file returns False when nothing was updated."""
        class_file = temp_dir / 'MESH_parameters_CLASS.ini'
        # File with fewer lines than expected
        class_file.write_text("line 0\nline 1\n")

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'KSAT'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        result = mgr._update_class_file(class_file, {'KSAT': 10.0})
        assert result is False

    def test_update_ini_file_applies_existing_params(
        self, mesh_config, logger, temp_dir, hydro_ini_content
    ):
        """Existing hydrology parameters are updated via regex."""
        hydro_file = temp_dir / 'MESH_parameters_hydrology.ini'
        hydro_file.write_text(hydro_ini_content)

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'FLZ,PWR,ZSNL'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        params = {'FLZ': 0.005, 'PWR': 2.8, 'ZSNL': 0.030}
        result = mgr._update_ini_file(hydro_file, params)
        assert result is True

        content = hydro_file.read_text()
        # Verify values were updated
        assert 'FLZ 0.005000' in content
        assert 'PWR 2.800000' in content
        assert 'ZSNL 0.030000' in content

    def test_update_ini_file_injects_missing_params(
        self, mesh_config, logger, temp_dir
    ):
        """Missing parameters are injected at end of file."""
        hydro_file = temp_dir / 'MESH_parameters_hydrology.ini'
        # File without FLZ or PWR
        hydro_file.write_text(
            "2.0: MESH Hydrology\n"
            "ZSNL  0.050\n"
            "ZPLS  0.060\n"
        )

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'FLZ,PWR,ZSNL'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        params = {'FLZ': 0.003, 'PWR': 2.5, 'ZSNL': 0.040}
        result = mgr._update_ini_file(hydro_file, params)
        assert result is True

        content = hydro_file.read_text()
        # ZSNL was already there — should be updated
        assert 'ZSNL 0.040000' in content
        # FLZ and PWR were missing — should be injected
        assert 'FLZ  0.003000' in content
        assert 'PWR  2.500000' in content

    def test_update_ini_file_injected_params_updatable_next_call(
        self, mesh_config, logger, temp_dir
    ):
        """Once injected, parameters can be updated by a second call."""
        hydro_file = temp_dir / 'MESH_parameters_hydrology.ini'
        hydro_file.write_text("2.0: header\nZSNL  0.050\n")

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'FLZ'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        # First call injects FLZ
        mgr._update_ini_file(hydro_file, {'FLZ': 0.002})
        assert 'FLZ  0.002000' in hydro_file.read_text()

        # Second call updates the injected value
        result = mgr._update_ini_file(hydro_file, {'FLZ': 0.007})
        assert result is True
        content = hydro_file.read_text()
        assert 'FLZ 0.007000' in content
        # Old value should be gone
        assert '0.002000' not in content

    def test_update_ini_file_returns_false_when_no_updates(
        self, mesh_config, logger, temp_dir
    ):
        """_update_ini_file returns False when nothing was updated or injected."""
        hydro_file = temp_dir / 'MESH_parameters_hydrology.ini'
        hydro_file.write_text("2.0: header\n")

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'WF_R2'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        # WF_R2 is an array param that cannot be injected
        result = mgr._update_ini_file(hydro_file, {'WF_R2': 0.3})
        assert result is False


class TestMESHClassPositionDetection:
    """Tests for dynamic CLASS.ini parameter position detection."""

    def test_detects_positions_from_markers(
        self, mesh_config, logger, temp_dir, class_ini_content
    ):
        """Marker comments are used to find the correct line indices."""
        class_file = temp_dir / 'MESH_parameters_CLASS.ini'
        class_file.write_text(class_ini_content)

        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'KSAT,DRN'}
        mgr = MESHParameterManager(config, logger, temp_dir)

        positions = mgr.class_param_positions
        # DRN is on the line containing "DRN/SDEP"
        assert 'DRN' in positions
        drn_line, drn_pos, _ = positions['DRN']
        # Verify the marker-detected line actually has the right value
        lines = class_ini_content.splitlines()
        parts = lines[drn_line].split()
        assert float(parts[drn_pos]) == pytest.approx(1.0)  # Original DRN value

    def test_falls_back_to_defaults_without_file(
        self, mesh_config, logger, temp_dir
    ):
        """Without CLASS.ini file, defaults are returned."""
        config = {**mesh_config, 'MESH_PARAMS_TO_CALIBRATE': 'KSAT'}
        mgr = MESHParameterManager(config, logger, temp_dir)
        # Should use default positions without error
        assert mgr.class_param_positions['KSAT'] == (13, 3, 5)
