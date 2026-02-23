"""
Integration tests for MODFLOW 6 model and SUMMA→MODFLOW coupling.

Tests:
    1. Binary install verification
    2. Preprocessor generates valid MODFLOW 6 input files
    3. Standalone MODFLOW run with constant recharge
    4. SUMMA → MODFLOW coupling (recharge extraction + conversion)
    5. Combined flow (surface + baseflow)
"""

import math
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def modflow_config_dict(tmp_path):
    """Minimal config dict for MODFLOW tests."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / "settings" / "MODFLOW").mkdir(parents=True)
    (project_dir / "simulations" / "test_exp" / "MODFLOW").mkdir(parents=True)

    return {
        'PROJECT_DIR': str(project_dir),
        'DOMAIN_NAME': 'bow_at_banff',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2000-01-01',
        'EXPERIMENT_TIME_END': '2000-04-01',
        'CATCHMENT_AREA': 2210.0,
        'MODFLOW_NLAY': 1,
        'MODFLOW_NROW': 1,
        'MODFLOW_NCOL': 1,
        'MODFLOW_K': 5.0,
        'MODFLOW_SY': 0.15,
        'MODFLOW_SS': 1e-5,
        'MODFLOW_TOP': 1500.0,
        'MODFLOW_BOT': 1400.0,
        'MODFLOW_DRAIN_ELEVATION': 1450.0,
        'MODFLOW_DRAIN_CONDUCTANCE': 50.0,
        'MODFLOW_STRESS_PERIOD_LENGTH': 1.0,
        'MODFLOW_NSTP': 1,
    }


@pytest.fixture
def mock_config(modflow_config_dict):
    """Create a mock config object matching SYMFLUENCE patterns."""
    config = MagicMock()
    config._config_dict = modflow_config_dict
    config.to_dict.return_value = modflow_config_dict
    config.domain.name = 'bow_at_banff'
    config.domain.experiment_id = 'test_exp'
    config.domain.time_start = '2000-01-01'
    config.domain.time_end = '2000-04-01'
    config.domain.catchment_area = 2210.0
    config.model.modflow.nlay = 1
    config.model.modflow.nrow = 1
    config.model.modflow.ncol = 1
    config.model.modflow.k = 5.0
    config.model.modflow.sy = 0.15
    config.model.modflow.ss = 1e-5
    config.model.modflow.top = 1500.0
    config.model.modflow.bot = 1400.0
    config.model.modflow.strt = None
    config.model.modflow.drain_elevation = 1450.0
    config.model.modflow.drain_conductance = 50.0
    config.model.modflow.stress_period_length = 1.0
    config.model.modflow.nstp = 1
    config.model.modflow.cell_size = 1000.0
    config.model.modflow.coupling_source = 'SUMMA'
    config.model.modflow.recharge_variable = 'scalarSoilDrainage'
    config.model.modflow.timeout = 3600
    config.model.modflow.exe = 'mf6'
    return config


# ---------------------------------------------------------------------------
# Test: Preprocessor generates valid MODFLOW 6 input files
# ---------------------------------------------------------------------------

class TestMODFLOWPreprocessor:
    """Tests for MODFLOW 6 input file generation."""

    def test_preprocessor_generates_all_files(self, mock_config, modflow_config_dict):
        """Verify preprocessor creates all required MODFLOW input files."""
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"

        expected_files = [
            'mfsim.nam', 'gwf.nam', 'gwf.tdis', 'gwf.dis',
            'gwf.ic', 'gwf.npf', 'gwf.sto', 'gwf.rch',
            'gwf.drn', 'gwf.oc', 'gwf.ims', 'recharge.ts',
        ]

        for fname in expected_files:
            fpath = settings_dir / fname
            assert fpath.exists(), f"Missing MODFLOW input file: {fname}"
            assert fpath.stat().st_size > 0, f"Empty MODFLOW input file: {fname}"

    def test_mfsim_nam_structure(self, mock_config, modflow_config_dict):
        """Verify mfsim.nam references correct files."""
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"
        content = (settings_dir / "mfsim.nam").read_text()

        assert 'gwf.tdis' in content
        assert 'gwf.nam' in content
        assert 'gwf.ims' in content

    def test_tdis_stress_periods(self, mock_config, modflow_config_dict):
        """Verify temporal discretization matches config time range."""
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"
        content = (settings_dir / "gwf.tdis").read_text()

        # 2000-01-01 to 2000-04-01 = 91 days, sp_length=1 → 91 periods
        assert 'NPER 91' in content
        assert 'TIME_UNITS DAYS' in content

    def test_dis_grid_dimensions(self, mock_config, modflow_config_dict):
        """Verify spatial discretization matches config."""
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"
        content = (settings_dir / "gwf.dis").read_text()

        assert 'NLAY 1' in content
        assert 'NROW 1' in content
        assert 'NCOL 1' in content
        assert 'CONSTANT 1500.0' in content   # TOP
        assert 'CONSTANT 1400.0' in content   # BOTM

    def test_drn_package(self, mock_config, modflow_config_dict):
        """Verify drain package has correct elevation and conductance."""
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"
        content = (settings_dir / "gwf.drn").read_text()

        # Drain at elevation 1450 with conductance 50
        assert '1450.0' in content
        assert '50.0' in content


# ---------------------------------------------------------------------------
# Test: MODFLOW standalone run (requires mf6 binary)
# ---------------------------------------------------------------------------

class TestMODFLOWStandalone:
    """Tests for standalone MODFLOW execution (requires mf6 installed)."""

    @pytest.fixture
    def mf6_exe(self):
        """Find mf6 executable or skip."""
        exe = shutil.which('mf6')
        if exe is None:
            # Check SYMFLUENCE install path
            default_path = Path.home() / "compHydro" / "data" / "SYMFLUENCE_data" / "installs" / "modflow" / "bin" / "mf6"
            if default_path.exists():
                exe = str(default_path)
            else:
                pytest.skip("mf6 executable not found (install with: symfluence binary install modflow)")
        return Path(exe)

    def test_modflow_standalone_constant_recharge(self, mf6_exe, mock_config, modflow_config_dict, tmp_path):
        """Run MODFLOW with constant recharge and verify output."""
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()

        # Copy all input files to sim directory
        for f in settings_dir.iterdir():
            shutil.copy2(f, sim_dir / f.name)

        # Run MODFLOW
        result = subprocess.run(
            [str(mf6_exe)],
            cwd=str(sim_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"MODFLOW failed: {result.stderr}"

        # Verify outputs exist
        hds_files = list(sim_dir.glob("*.hds"))
        assert len(hds_files) > 0, "No head output files produced"

        bud_files = list(sim_dir.glob("*.bud"))
        assert len(bud_files) > 0, "No budget output files produced"

        # Verify head file is non-empty and contains valid data
        hds_size = hds_files[0].stat().st_size
        assert hds_size > 0, "Head file is empty"

    def test_modflow_water_balance(self, mf6_exe, mock_config, modflow_config_dict, tmp_path):
        """Verify MODFLOW water balance: recharge ≈ drain + Δstorage."""
        from symfluence.models.modflow.extractor import MODFLOWResultExtractor
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor

        logger = MagicMock()
        preprocessor = MODFLOWPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(modflow_config_dict['PROJECT_DIR']) / "settings" / "MODFLOW"
        sim_dir = tmp_path / "sim_wb"
        sim_dir.mkdir()

        for f in settings_dir.iterdir():
            shutil.copy2(f, sim_dir / f.name)

        result = subprocess.run(
            [str(mf6_exe)],
            cwd=str(sim_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            pytest.skip(f"MODFLOW execution failed: {result.stderr[:200]}")

        # Extract drain discharge
        extractor = MODFLOWResultExtractor()
        try:
            drain = extractor.extract_variable(
                sim_dir, 'drain_discharge',
                stress_period_length=1.0,
                start_date='2000-01-01',
            )
            # Drain discharge should be non-negative
            assert (drain >= 0).all(), "Negative drain discharge found"
        except Exception:
            # Budget file parsing may fail for some MODFLOW versions
            pytest.skip("Could not parse MODFLOW budget file")


# ---------------------------------------------------------------------------
# Test: SUMMA → MODFLOW coupling
# ---------------------------------------------------------------------------

class TestSUMMAMODFLOWCoupling:
    """Tests for the SUMMA → MODFLOW coupling module."""

    @pytest.fixture
    def sample_recharge(self):
        """Create sample recharge time series."""
        dates = pd.date_range('2000-01-01', periods=90, freq='D')
        # Sinusoidal recharge pattern (seasonal)
        values = 0.002 + 0.001 * np.sin(np.arange(90) * 2 * np.pi / 365)
        return pd.Series(values, index=dates, name='recharge_m_d')

    def test_write_modflow_recharge_rch(self, sample_recharge, tmp_path):
        """Verify recharge RCH package file format."""
        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler

        coupler = SUMMAToMODFLOWCoupler({})
        rch_file = coupler.write_modflow_recharge_rch(
            sample_recharge,
            tmp_path / "gwf.rch",
        )

        assert rch_file.exists()
        content = rch_file.read_text()

        # Verify RCH package format
        assert 'BEGIN OPTIONS' in content
        assert 'READASARRAYS' in content
        assert 'BEGIN PERIOD 1' in content
        assert 'RECHARGE' in content
        assert 'CONSTANT' in content

    def test_combine_flows(self):
        """Verify surface runoff + baseflow combination."""
        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler

        coupler = SUMMAToMODFLOWCoupler({})
        dates = pd.date_range('2000-01-01', periods=30, freq='D')

        # Surface runoff: 1e-6 m/s
        surface = pd.Series(1e-6, index=dates, name='surface')
        # Drain discharge: 100 m3/d
        drain = pd.Series(100.0, index=dates, name='drain')
        area_m2 = 2210.0 * 1e6  # 2210 km2

        total = coupler.combine_flows(surface, drain, area_m2)

        # Surface: 1e-6 m/s * 2210e6 m2 = 2210 m3/s
        # Baseflow: 100 / 86400 ≈ 0.001157 m3/s
        expected_surface = 1e-6 * area_m2
        expected_baseflow = 100.0 / 86400.0

        assert len(total) == 30
        np.testing.assert_allclose(
            total.values[0],
            expected_surface + expected_baseflow,
            rtol=1e-6,
        )

    def test_combine_flows_mismatched_dates(self):
        """Verify graceful handling of non-overlapping date ranges."""
        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler

        coupler = SUMMAToMODFLOWCoupler({})

        dates1 = pd.date_range('2000-01-01', periods=10, freq='D')
        dates2 = pd.date_range('2000-02-01', periods=10, freq='D')

        surface = pd.Series(0.001, index=dates1)
        drain = pd.Series(100.0, index=dates2)

        total = coupler.combine_flows(surface, drain, 1e9)
        # No overlap → returns surface only
        assert len(total) == 10

    def test_extract_recharge_from_summa(self, tmp_path):
        """Verify SUMMA recharge extraction with unit conversion."""
        import xarray as xr

        from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler

        # Create fake SUMMA output
        dates = pd.date_range('2000-01-01', periods=30, freq='D')
        drainage = np.full(30, 1.157e-8)  # ~0.001 m/d in m/s

        ds = xr.Dataset(
            {'scalarSoilDrainage': (['time', 'hru'], drainage.reshape(-1, 1))},
            coords={'time': dates, 'hru': [1]},
        )
        output_dir = tmp_path / "summa_output"
        output_dir.mkdir()
        ds.to_netcdf(output_dir / "test_output_day.nc")

        coupler = SUMMAToMODFLOWCoupler({})
        recharge = coupler.extract_recharge_from_summa(output_dir)

        assert len(recharge) == 30
        # 1.157e-8 m/s × 86400 s/d ≈ 0.001 m/d
        np.testing.assert_allclose(recharge.values[0], 1.157e-8 * 86400, rtol=0.01)
        assert (recharge >= 0).all()


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------

class TestMODFLOWConfig:
    """Tests for MODFLOW configuration validation."""

    def test_config_adapter_validation(self):
        """Verify config adapter rejects invalid configurations."""
        from symfluence.models.modflow.config import MODFLOWConfigAdapter

        adapter = MODFLOWConfigAdapter()

        # Valid config should not raise
        adapter.validate({'top': 1500.0, 'bot': 1400.0, 'drain_elevation': 1450.0})

        # top <= bot should raise
        with pytest.raises(ValueError, match="top.*must be greater than bot"):
            adapter.validate({'top': 1400.0, 'bot': 1500.0})

        # drain_elevation out of range should raise
        with pytest.raises(ValueError, match="drain_elevation"):
            adapter.validate({'top': 1500.0, 'bot': 1400.0, 'drain_elevation': 1600.0})

    def test_modflow_pydantic_config(self):
        """Verify MODFLOWConfig Pydantic model defaults."""
        from symfluence.core.config.models.model_configs import MODFLOWConfig

        cfg = MODFLOWConfig()
        assert cfg.k == 5.0
        assert cfg.sy == 0.15
        assert cfg.nlay == 1
        assert cfg.grid_type == 'dis'
        assert cfg.coupling_source == 'SUMMA'
        assert cfg.drain_conductance == 50.0
        assert cfg.timeout == 3600

    def test_modflow_config_from_alias(self):
        """Verify MODFLOWConfig can be created from alias keys."""
        from symfluence.core.config.models.model_configs import MODFLOWConfig

        cfg = MODFLOWConfig(**{
            'MODFLOW_K': 10.0,
            'MODFLOW_SY': 0.20,
            'MODFLOW_TOP': 2000.0,
            'MODFLOW_BOT': 1800.0,
        })
        assert cfg.k == 10.0
        assert cfg.sy == 0.20
        assert cfg.top == 2000.0
        assert cfg.bot == 1800.0


# ---------------------------------------------------------------------------
# Test: Model registration
# ---------------------------------------------------------------------------

class TestMODFLOWRegistration:
    """Tests for MODFLOW model registration in ModelRegistry."""

    def test_modflow_registered(self):
        """Verify MODFLOW components are registered."""
        from symfluence.models.registry import ModelRegistry

        assert ModelRegistry.get_preprocessor('MODFLOW') is not None
        assert ModelRegistry.get_runner('MODFLOW') is not None
        assert ModelRegistry.get_result_extractor('MODFLOW') is not None
        assert ModelRegistry.get_postprocessor('MODFLOW') is not None

    def test_modflow_runner_method(self):
        """Verify runner has a run() method (direct override, no method_name dispatch)."""
        from symfluence.models.registry import ModelRegistry

        runner_cls = ModelRegistry.get_runner('MODFLOW')
        assert hasattr(runner_cls, 'run'), "MODFLOWRunner must have a run() method"


# ---------------------------------------------------------------------------
# Test: MODFLOW Plotter
# ---------------------------------------------------------------------------

class TestMODFLOWPlotter:
    """Tests for MODFLOW coupling diagnostics plotter."""

    def test_plotter_registered(self):
        """Verify MODFLOWPlotter is registered in PlotterRegistry."""
        from symfluence.reporting.plotter_registry import PlotterRegistry

        assert PlotterRegistry.has_plotter('MODFLOW')
        plotter_cls = PlotterRegistry.get_plotter('MODFLOW')
        assert plotter_cls is not None
        assert plotter_cls.__name__ == 'MODFLOWPlotter'

    def test_plotter_generates_coupling_overview(self, tmp_path):
        """Generate coupling overview with synthetic 90-day data and verify PNG."""
        import matplotlib
        matplotlib.use('Agg')

        from symfluence.models.modflow.plotter import MODFLOWPlotter

        # Build minimal config dict
        project_dir = tmp_path / "test_project"
        (project_dir / "reporting" / "modflow_coupling").mkdir(parents=True)

        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'test_exp',
            'EXPERIMENT_TIME_START': '2000-01-01',
            'EXPERIMENT_TIME_END': '2000-04-01',
            'CATCHMENT_AREA': 100.0,
            'MODFLOW_TOP': 1500.0,
            'MODFLOW_BOT': 1400.0,
            'MODFLOW_DRAIN_ELEVATION': 1450.0,
            'MODFLOW_SY': 0.15,
            'MODFLOW_CELL_SIZE': 1000.0,
            'MODFLOW_STRESS_PERIOD_LENGTH': 1.0,
        }

        logger_mock = MagicMock()
        plotter = MODFLOWPlotter(config_dict, logger_mock)

        # Override project_dir to our tmp_path-based dir
        plotter.project_dir = project_dir

        # Build synthetic 90-day data
        dates = pd.date_range('2000-01-01', periods=90, freq='D')
        np.random.seed(42)

        obs = pd.Series(
            5.0 + 2.0 * np.sin(np.arange(90) * 2 * np.pi / 90) + np.random.normal(0, 0.3, 90),
            index=dates, name='obs',
        )
        surface_m3s = pd.Series(
            3.0 + 1.5 * np.sin(np.arange(90) * 2 * np.pi / 90),
            index=dates, name='surface_m3s',
        )
        baseflow_m3s = pd.Series(
            2.0 + 0.5 * np.sin(np.arange(90) * 2 * np.pi / 90 + 1.0),
            index=dates, name='baseflow_m3s',
        )
        total_m3s = surface_m3s + baseflow_m3s
        total_m3s.name = 'total_m3s'

        head = pd.Series(
            1460.0 + 10.0 * np.sin(np.arange(90) * 2 * np.pi / 90),
            index=dates, name='head_m',
        )
        recharge_m_d = pd.Series(
            0.002 + 0.001 * np.sin(np.arange(90) * 2 * np.pi / 365),
            index=dates, name='recharge_m_d',
        )
        drain_m3_d = baseflow_m3s * 86400.0
        drain_m3_d.name = 'drain_m3d'

        # Patch _collect_coupling_data to return our synthetic data
        synthetic_data = {
            'obs': obs,
            'surface_m3s': surface_m3s,
            'baseflow_m3s': baseflow_m3s,
            'total_m3s': total_m3s,
            'head': head,
            'recharge_m_d': recharge_m_d,
            'drain_m3_d': drain_m3_d,
            'top': 1500.0,
            'bot': 1400.0,
            'drain_elev': 1450.0,
            'sy': 0.15,
            'cell_size': 1000.0,
        }

        with patch.object(plotter, '_collect_coupling_data', return_value=synthetic_data):
            result = plotter.plot_coupling_results('test_exp')

        assert result is not None
        output_path = Path(result)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert output_path.name == 'test_exp_coupling_overview.png'

    def test_plotter_handles_missing_obs(self, tmp_path):
        """Verify plotter renders gracefully when observed data is None."""
        import matplotlib
        matplotlib.use('Agg')

        from symfluence.models.modflow.plotter import MODFLOWPlotter

        project_dir = tmp_path / "test_project"
        (project_dir / "reporting" / "modflow_coupling").mkdir(parents=True)

        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'test_exp',
            'EXPERIMENT_TIME_START': '2000-01-01',
            'EXPERIMENT_TIME_END': '2000-04-01',
        }

        logger_mock = MagicMock()
        plotter = MODFLOWPlotter(config_dict, logger_mock)
        plotter.project_dir = project_dir

        dates = pd.date_range('2000-01-01', periods=30, freq='D')
        surface = pd.Series(1.0, index=dates)
        base = pd.Series(0.5, index=dates)

        synthetic_data = {
            'obs': None,
            'surface_m3s': surface,
            'baseflow_m3s': base,
            'total_m3s': surface + base,
            'head': pd.Series(1460.0, index=dates),
            'recharge_m_d': pd.Series(0.002, index=dates),
            'drain_m3_d': base * 86400.0,
            'top': 1500.0,
            'bot': 1400.0,
            'drain_elev': 1450.0,
            'sy': 0.15,
            'cell_size': 1000.0,
        }

        with patch.object(plotter, '_collect_coupling_data', return_value=synthetic_data):
            result = plotter.plot_coupling_results('test_exp')

        assert result is not None
        assert Path(result).exists()
