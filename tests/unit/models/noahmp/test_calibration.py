"""Tests for Noah-MP calibration components."""
from __future__ import annotations

import pytest

from symfluence.core.registries import R


class TestNoahMPCalibrationRegistration:
    """Tests for Noah-MP calibration component registration."""

    def test_optimizer_registered(self):
        from symfluence.models.noahmp.calibration import NoahMPModelOptimizer  # noqa: F401
        assert 'NOAHMP' in R.optimizers

    def test_optimizer_model_name(self):
        from symfluence.models.noahmp.calibration.optimizer import NoahMPModelOptimizer
        # Verify the class method returns 'NOAHMP'
        assert NoahMPModelOptimizer._get_model_name(None) == 'NOAHMP'

    def test_worker_registered(self):
        from symfluence.models.noahmp.calibration import NoahMPWorker  # noqa: F401
        assert 'NOAHMP' in R.workers

    def test_parameter_manager_registered(self):
        from symfluence.models.noahmp.calibration import NoahMPParameterManager  # noqa: F401
        assert 'NOAHMP' in R.parameter_managers


class TestNoahMPWorkerAttributes:
    """Tests for Noah-MP worker class attributes."""

    def test_namelist_params(self):
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        assert 'rain_snow_thresh' in NoahMPWorker.NAMELIST_PARAMS
        assert 'ZREF' in NoahMPWorker.NAMELIST_PARAMS
        assert 'refkdt' in NoahMPWorker.NAMELIST_PARAMS

    def test_soilparm_columns(self):
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        assert NoahMPWorker.SOILPARM_COLUMNS['bexp'] == 1
        assert NoahMPWorker.SOILPARM_COLUMNS['smcmax'] == 4
        assert NoahMPWorker.SOILPARM_COLUMNS['smcref'] == 5
        assert NoahMPWorker.SOILPARM_COLUMNS['psisat'] == 6
        assert NoahMPWorker.SOILPARM_COLUMNS['dksat'] == 7
        assert NoahMPWorker.SOILPARM_COLUMNS['smcwlt'] == 9

    def test_apply_namelist_params(self, tmp_path):
        """Test namelist parameter application to a file."""
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker

        # Write a minimal namelist.input
        nl_content = """\
&timing
  dt                 = 3600
  startdate          = "200001010000"
  enddate            = "200112310000"
  forcing_filename   = "forcing.txt"
  output_filename    = "output.nc"
/

&forcing
  ZREF               =  10.0
/

&model_options
  dynamic_veg_option         = 1
  runoff_option              = 8
  drainage_option            = 8
  refkdt                     = 3.0
/

&structure
  isltyp             = 4
  nsoil              = 4
  nsnow              = 3
  nveg               = 20
  vegtyp             = 1
  dzsnso             = 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 1.0
/
"""
        nl_path = tmp_path / 'namelist.input'
        nl_path.write_text(nl_content)

        worker = NoahMPWorker()
        success = worker._apply_namelist_params(
            {'ZREF': 5.0, 'refkdt': 2.5},
            tmp_path,
        )
        assert success

        result = nl_path.read_text()
        assert '5' in result  # ZREF updated
        assert '2.5' in result  # refkdt updated

    def test_apply_soilparm_params(self, tmp_path):
        """Test SOILPARM.TBL parameter application."""
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker

        # Write a minimal SOILPARM.TBL (STAS classification)
        # Format: soil_type, bexp, ... (comma-separated)
        soilparm_content = """\
Soil Parameters
STAS
19, 1
1, 2.790000, 0.000000, 0.000000, 0.000000, 0.339000, 0.236000, 0.069000, 0.000034, 0.000000, 0.010000, 0.000000
2, 4.260000, 0.000000, 0.000000, 0.000000, 0.421000, 0.383000, 0.141000, 0.000018, 0.000000, 0.028000, 0.000000
3, 4.740000, 0.000000, 0.000000, 0.000000, 0.434000, 0.383000, 0.218000, 0.000045, 0.000000, 0.047000, 0.000000
4, 5.330000, 0.000000, 0.000000, 0.000000, 0.476000, 0.360000, 0.254000, 0.000021, 0.000000, 0.084000, 0.000000
5, 5.330000, 0.000000, 0.000000, 0.000000, 0.476000, 0.383000, 0.254000, 0.000021, 0.000000, 0.084000, 0.000000
"""
        soilparm_path = tmp_path / 'SOILPARM.TBL'
        soilparm_path.write_text(soilparm_content)

        # Write a matching namelist
        nl_content = """\
&structure
  isltyp             = 4
/
"""
        (tmp_path / 'namelist.input').write_text(nl_content)

        worker = NoahMPWorker()
        success = worker._update_soilparm_tbl(
            {'bexp': 6.0, 'smcmax': 0.5},
            tmp_path,
        )
        assert success

        result = soilparm_path.read_text()
        lines = result.strip().splitlines()
        # Line index 6 corresponds to soil type 4 (data_start=3, isltyp=4, so 3+3=6)
        soil4_line = lines[6]
        parts = [p.strip() for p in soil4_line.split(',')]
        assert float(parts[1]) == pytest.approx(6.0)
        assert float(parts[4]) == pytest.approx(0.5)


class TestNoahMPParameterBounds:
    """Tests for Noah-MP parameter bounds in registry."""

    def test_bounds_loaded(self):
        from symfluence.core.calibration.parameters.parameter_bounds_registry import get_noahmp_bounds
        bounds = get_noahmp_bounds()
        assert len(bounds) > 0

    def test_expected_params(self):
        from symfluence.core.calibration.parameters.parameter_bounds_registry import get_noahmp_bounds
        bounds = get_noahmp_bounds()
        expected = ['slope', 'dksat', 'psisat', 'bexp', 'smcmax', 'smcwlt',
                    'smcref', 'refkdt', 'noah_czil', 'rain_snow_thresh', 'ZREF']
        for param in expected:
            assert param in bounds, f"Missing expected parameter: {param}"

    def test_min_less_than_max(self):
        from symfluence.core.calibration.parameters.parameter_bounds_registry import get_noahmp_bounds
        bounds = get_noahmp_bounds()
        for name, b in bounds.items():
            assert b['min'] < b['max'], f"{name}: min ({b['min']}) >= max ({b['max']})"

    def test_dksat_log_transform(self):
        from symfluence.core.calibration.parameters.parameter_bounds_registry import get_noahmp_bounds
        bounds = get_noahmp_bounds()
        assert bounds['dksat']['transform'] == 'log'
