"""
Integration tests for ParFlow and SUMMA->ParFlow coupling.

Tests:
- ParFlow preprocessor generates valid .pfidb file
- ParFlow standalone execution (requires parflow binary, skipped otherwise)
- SUMMA -> ParFlow coupling: unit conversions and flow combination
- ParFlow Pydantic config validation
- ParFlow model registration in ModelRegistry
- ParFlow plotter registration and PNG generation
"""
import shutil
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
def parflow_config_dict(tmp_path):
    """Create a minimal config dict for ParFlow tests."""
    project_dir = tmp_path / "test_domain"
    project_dir.mkdir()
    (project_dir / "settings" / "PARFLOW").mkdir(parents=True)
    (project_dir / "simulations" / "test_exp" / "PARFLOW").mkdir(parents=True)
    (project_dir / "results").mkdir(parents=True)

    return {
        'PROJECT_DIR': str(project_dir),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2000-01-01',
        'EXPERIMENT_TIME_END': '2000-04-01',
        'CATCHMENT_AREA': 2210.0,
        'PARFLOW_INSTALL_PATH': 'default',
        'PARFLOW_EXE': 'parflow',
        'PARFLOW_DIR': 'default',
        'PARFLOW_NX': 1,
        'PARFLOW_NY': 1,
        'PARFLOW_NZ': 1,
        'PARFLOW_DX': 1000.0,
        'PARFLOW_DY': 1000.0,
        'PARFLOW_DZ': 100.0,
        'PARFLOW_TOP': 1500.0,
        'PARFLOW_BOT': 1400.0,
        'PARFLOW_K_SAT': 5.0,
        'PARFLOW_POROSITY': 0.4,
        'PARFLOW_VG_ALPHA': 1.0,
        'PARFLOW_VG_N': 2.0,
        'PARFLOW_S_RES': 0.1,
        'PARFLOW_S_SAT': 1.0,
        'PARFLOW_SS': 1e-5,
        'PARFLOW_MANNINGS_N': 0.03,
        'PARFLOW_INITIAL_PRESSURE': None,
        'PARFLOW_COUPLING_SOURCE': 'SUMMA',
        'PARFLOW_RECHARGE_VARIABLE': 'scalarSoilDrainage',
        'PARFLOW_SOLVER': 'Richards',
        'PARFLOW_TIMESTEP_HOURS': 1.0,
        'PARFLOW_NUM_PROCS': 1,
        'PARFLOW_TIMEOUT': 3600,
    }


@pytest.fixture
def mock_config(parflow_config_dict):
    """Create a mock config object matching SYMFLUENCE patterns."""
    config = MagicMock()
    config._config_dict = parflow_config_dict
    config.to_dict.return_value = parflow_config_dict
    config.domain.name = 'test_domain'
    config.domain.experiment_id = 'test_exp'
    config.domain.time_start = '2000-01-01'
    config.domain.time_end = '2000-04-01'
    config.domain.catchment_area = 2210.0
    config.model.parflow.nx = 1
    config.model.parflow.ny = 1
    config.model.parflow.nz = 1
    config.model.parflow.dx = 1000.0
    config.model.parflow.dy = 1000.0
    config.model.parflow.dz = 100.0
    config.model.parflow.k_sat = 5.0
    config.model.parflow.porosity = 0.4
    config.model.parflow.vg_alpha = 1.0
    config.model.parflow.vg_n = 2.0
    config.model.parflow.s_res = 0.1
    config.model.parflow.s_sat = 1.0
    config.model.parflow.specific_storage = 1e-5
    config.model.parflow.top = 1500.0
    config.model.parflow.bot = 1400.0
    config.model.parflow.mannings_n = 0.03
    config.model.parflow.initial_pressure = None
    config.model.parflow.coupling_source = 'SUMMA'
    config.model.parflow.recharge_variable = 'scalarSoilDrainage'
    config.model.parflow.solver = 'Richards'
    config.model.parflow.timestep_hours = 1.0
    config.model.parflow.num_procs = 1
    config.model.parflow.timeout = 3600
    config.model.parflow.exe = 'parflow'
    config.model.parflow.parflow_dir = 'default'
    return config


# ---------------------------------------------------------------------------
# Test: Preprocessor generates valid ParFlow input
# ---------------------------------------------------------------------------

class TestParFlowPreprocessor:
    """Tests for ParFlow input file generation."""

    def test_preprocessor_generates_pfidb(self, mock_config, parflow_config_dict):
        """Verify preprocessor creates a .pfidb file."""
        from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

        logger = MagicMock()
        preprocessor = ParFlowPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(parflow_config_dict['PROJECT_DIR']) / "settings" / "PARFLOW"

        # Should have a .pfidb file
        pfidb_files = list(settings_dir.glob("*.pfidb"))
        assert len(pfidb_files) > 0, "No .pfidb file generated"
        assert pfidb_files[0].stat().st_size > 0, "Empty .pfidb file"

    def test_pfidb_contains_grid_dims(self, mock_config, parflow_config_dict):
        """Verify .pfidb contains correct grid dimensions."""
        from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

        logger = MagicMock()
        preprocessor = ParFlowPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(parflow_config_dict['PROJECT_DIR']) / "settings" / "PARFLOW"
        pfidb_files = list(settings_dir.glob("*.pfidb"))
        content = pfidb_files[0].read_text()

        # Check grid dimensions present
        assert 'ComputationalGrid.NX' in content
        assert 'ComputationalGrid.NY' in content
        assert 'ComputationalGrid.NZ' in content

    def test_pfidb_contains_subsurface_properties(self, mock_config, parflow_config_dict):
        """Verify .pfidb contains van Genuchten and permeability parameters."""
        from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

        logger = MagicMock()
        preprocessor = ParFlowPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(parflow_config_dict['PROJECT_DIR']) / "settings" / "PARFLOW"
        pfidb_files = list(settings_dir.glob("*.pfidb"))
        content = pfidb_files[0].read_text()

        assert 'Perm.Value' in content
        assert 'VanGenuchten' in content
        assert 'Porosity.Value' in content

    def test_pfidb_contains_timing(self, mock_config, parflow_config_dict):
        """Verify .pfidb contains timing information."""
        from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

        logger = MagicMock()
        preprocessor = ParFlowPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(parflow_config_dict['PROJECT_DIR']) / "settings" / "PARFLOW"
        pfidb_files = list(settings_dir.glob("*.pfidb"))
        content = pfidb_files[0].read_text()

        assert 'TimingInfo.StartTime' in content
        assert 'TimingInfo.StopTime' in content
        assert 'TimingInfo.DumpInterval' in content

    def test_pfidb_contains_overland_flow_bc(self, mock_config, parflow_config_dict):
        """Verify .pfidb contains overland flow boundary condition at surface."""
        from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

        logger = MagicMock()
        preprocessor = ParFlowPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = Path(parflow_config_dict['PROJECT_DIR']) / "settings" / "PARFLOW"
        pfidb_files = list(settings_dir.glob("*.pfidb"))
        content = pfidb_files[0].read_text()

        assert 'OverlandFlow' in content
        assert 'Mannings' in content


# ---------------------------------------------------------------------------
# Test: ParFlow standalone run (requires parflow binary)
# ---------------------------------------------------------------------------

class TestParFlowStandalone:
    """Tests for standalone ParFlow execution (requires parflow installed)."""

    @pytest.fixture
    def parflow_exe(self):
        """Find parflow executable or skip."""
        exe = shutil.which('parflow')
        if exe is None:
            # Check common install locations
            for subdir in ("code", "data"):
                candidate = (
                    Path.home() / "compHydro" / subdir / "SYMFLUENCE_data"
                    / "installs" / "parflow" / "bin" / "parflow"
                )
                if candidate.exists():
                    exe = str(candidate)
                    break
            if exe is None:
                pytest.skip(
                    "parflow executable not found "
                    "(install with: symfluence binary install parflow)"
                )
        return Path(exe)

    def test_parflow_standalone_run(
        self, parflow_exe, mock_config, parflow_config_dict, tmp_path
    ):
        """Run ParFlow with default config and verify .pfb outputs."""
        from symfluence.models.parflow.preprocessor import ParFlowPreProcessor

        logger = MagicMock()
        preprocessor = ParFlowPreProcessor(mock_config, logger)
        preprocessor.run_preprocessing()

        settings_dir = (
            Path(parflow_config_dict['PROJECT_DIR']) / "settings" / "PARFLOW"
        )
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()

        # Copy all input files to sim directory
        for f in settings_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, sim_dir / f.name)

        # Get run name
        runname = 'test_domain'
        runname_file = sim_dir / 'runname.txt'
        if runname_file.exists():
            runname = runname_file.read_text().strip()

        # Run ParFlow
        result = subprocess.run(
            [str(parflow_exe), runname],
            cwd=str(sim_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"ParFlow failed: {result.stderr}"

        # Verify pressure outputs exist
        press_files = list(sim_dir.glob("*.out.press.*.pfb"))
        assert len(press_files) > 0, "No pressure output files produced"


# ---------------------------------------------------------------------------
# Test: SUMMA -> ParFlow coupling
# ---------------------------------------------------------------------------

class TestSUMMAParFlowCoupling:
    """Tests for the SUMMA -> ParFlow coupling module."""

    @pytest.fixture
    def sample_recharge(self):
        """Create sample recharge time series."""
        dates = pd.date_range('2000-01-01', periods=720, freq='h')  # 30 days
        # Sinusoidal recharge pattern
        values = 0.002 + 0.001 * np.sin(np.arange(720) * 2 * np.pi / (24 * 365))
        return pd.Series(values, index=dates, name='recharge_m_hr')

    def test_write_parflow_recharge(self, sample_recharge, tmp_path):
        """Verify recharge file format."""
        from symfluence.models.parflow.coupling import SUMMAToParFlowCoupler

        coupler = SUMMAToParFlowCoupler({})
        rch_file = coupler.write_parflow_recharge(
            sample_recharge,
            tmp_path / "recharge.csv",
        )

        assert rch_file.exists()
        content = rch_file.read_text()

        # Verify header
        assert 'ParFlow recharge flux' in content
        assert 'sim_hour' in content

        # Verify data rows
        lines = [ln for ln in content.splitlines() if not ln.startswith('#')]
        assert len(lines) >= 720

    def test_combine_flows(self):
        """Verify surface runoff + subsurface flow combination."""
        from symfluence.models.parflow.coupling import SUMMAToParFlowCoupler

        coupler = SUMMAToParFlowCoupler({})
        dates = pd.date_range('2000-01-01', periods=720, freq='h')

        # Surface runoff: 0.001 kg/m2/s
        surface = pd.Series(0.001, index=dates, name='surface')
        # Subsurface drainage: 100 m3/hr
        subsurface = pd.Series(100.0, index=dates, name='subsurface')
        area_m2 = 2210.0 * 1e6  # 2210 km2

        total = coupler.combine_flows(surface, subsurface, area_m2)

        # Surface: 0.001 * 2210e6 / 1000 = 2210 m3/s
        # Subsurface: 100 / 3600 ~= 0.02778 m3/s
        expected_surface = 0.001 * area_m2 / 1000.0
        expected_subsurface = 100.0 / 3600.0

        assert len(total) == 720
        np.testing.assert_allclose(
            total.values[0],
            expected_surface + expected_subsurface,
            rtol=1e-6,
        )

    def test_combine_flows_mismatched_dates(self):
        """Verify graceful handling of non-overlapping date ranges."""
        from symfluence.models.parflow.coupling import SUMMAToParFlowCoupler

        coupler = SUMMAToParFlowCoupler({})

        dates1 = pd.date_range('2000-01-01', periods=24, freq='h')
        dates2 = pd.date_range('2000-02-01', periods=24, freq='h')

        surface = pd.Series(0.001, index=dates1)
        subsurface = pd.Series(100.0, index=dates2)

        total = coupler.combine_flows(surface, subsurface, 1e9)
        # No overlap -> returns surface only
        assert len(total) == 24

    def test_extract_recharge_from_summa(self, tmp_path):
        """Verify SUMMA recharge extraction with unit conversion to m/hr."""
        import xarray as xr
        from symfluence.models.parflow.coupling import SUMMAToParFlowCoupler

        # Create fake SUMMA output
        dates = pd.date_range('2000-01-01', periods=720, freq='h')
        drainage = np.full(720, 1.157e-5)  # ~1 mm/d in kg/m2/s

        ds = xr.Dataset(
            {'scalarSoilDrainage': (['time', 'hru'], drainage.reshape(-1, 1))},
            coords={'time': dates, 'hru': [1]},
        )
        output_dir = tmp_path / "summa_output"
        output_dir.mkdir()
        ds.to_netcdf(output_dir / "test_output_day.nc")

        coupler = SUMMAToParFlowCoupler({})
        recharge = coupler.extract_recharge_from_summa(output_dir)

        assert len(recharge) == 720
        # 1.157e-5 kg/m2/s x 3.6 ~= 4.165e-5 m/hr
        np.testing.assert_allclose(recharge.values[0], 1.157e-5 * 3.6, rtol=0.01)
        assert (recharge >= 0).all()

    def test_unit_conversion_constants(self):
        """Verify key unit conversion constants."""
        from symfluence.models.parflow.coupling import SUMMAToParFlowCoupler

        # kg/m2/s -> m/hr: 3600/1000 = 3.6
        assert SUMMAToParFlowCoupler.KG_M2_S_TO_M_HR == 3.6


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------

class TestParFlowConfig:
    """Tests for ParFlow configuration validation."""

    def test_config_adapter_validation(self):
        """Verify config adapter rejects invalid configurations."""
        from symfluence.models.parflow.config import ParFlowConfigAdapter

        adapter = ParFlowConfigAdapter()

        # Valid config should not raise
        adapter.validate({
            'top': 1500.0, 'bot': 1400.0,
            'vg_alpha': 1.0, 'vg_n': 2.0,
            's_res': 0.1, 's_sat': 1.0,
        })

        # top <= bot should raise
        with pytest.raises(ValueError, match="top.*must be greater than bot"):
            adapter.validate({'top': 1400.0, 'bot': 1500.0})

        # vg_alpha <= 0 should raise
        with pytest.raises(ValueError, match="vg_alpha"):
            adapter.validate({'top': 1500.0, 'bot': 1400.0, 'vg_alpha': -1.0})

        # vg_n <= 1.0 should raise
        with pytest.raises(ValueError, match="vg_n"):
            adapter.validate({
                'top': 1500.0, 'bot': 1400.0,
                'vg_alpha': 1.0, 'vg_n': 0.5,
            })

        # s_res >= s_sat should raise
        with pytest.raises(ValueError, match="s_res"):
            adapter.validate({
                'top': 1500.0, 'bot': 1400.0,
                'vg_alpha': 1.0, 'vg_n': 2.0,
                's_res': 1.0, 's_sat': 0.5,
            })

    def test_parflow_pydantic_config(self):
        """Verify ParFlowConfig Pydantic model defaults."""
        from symfluence.core.config.models.model_configs import ParFlowConfig

        cfg = ParFlowConfig()
        assert cfg.k_sat == 5.0
        assert cfg.porosity == 0.4
        assert cfg.vg_alpha == 1.0
        assert cfg.vg_n == 2.0
        assert cfg.nx == 1
        assert cfg.solver == 'Richards'
        assert cfg.coupling_source == 'SUMMA'
        assert cfg.mannings_n == 0.03
        assert cfg.timeout == 3600

    def test_parflow_config_from_alias(self):
        """Verify ParFlowConfig can be created from alias keys."""
        from symfluence.core.config.models.model_configs import ParFlowConfig

        cfg = ParFlowConfig(**{
            'PARFLOW_K_SAT': 10.0,
            'PARFLOW_POROSITY': 0.35,
            'PARFLOW_TOP': 2000.0,
            'PARFLOW_BOT': 1800.0,
            'PARFLOW_VG_N': 3.0,
        })
        assert cfg.k_sat == 10.0
        assert cfg.porosity == 0.35
        assert cfg.top == 2000.0
        assert cfg.bot == 1800.0
        assert cfg.vg_n == 3.0


# ---------------------------------------------------------------------------
# Test: Model registration
# ---------------------------------------------------------------------------

class TestParFlowRegistration:
    """Tests for ParFlow model registration in ModelRegistry."""

    def test_parflow_registered(self):
        """Verify ParFlow components are registered."""
        from symfluence.models.registry import ModelRegistry

        assert ModelRegistry.get_preprocessor('PARFLOW') is not None
        assert ModelRegistry.get_runner('PARFLOW') is not None
        assert ModelRegistry.get_result_extractor('PARFLOW') is not None
        assert ModelRegistry.get_postprocessor('PARFLOW') is not None

    def test_parflow_runner_method(self):
        """Verify runner method name is registered."""
        from symfluence.models.registry import ModelRegistry

        method = ModelRegistry.get_runner_method('PARFLOW')
        assert method == 'run_parflow'


# ---------------------------------------------------------------------------
# Test: ParFlow Plotter
# ---------------------------------------------------------------------------

class TestParFlowPlotter:
    """Tests for ParFlow coupling diagnostics plotter."""

    def test_plotter_registered(self):
        """Verify ParFlowPlotter is registered in PlotterRegistry."""
        from symfluence.reporting.plotter_registry import PlotterRegistry

        assert PlotterRegistry.has_plotter('PARFLOW')
        plotter_cls = PlotterRegistry.get_plotter('PARFLOW')
        assert plotter_cls is not None
        assert plotter_cls.__name__ == 'ParFlowPlotter'

    def test_plotter_generates_coupling_overview(self, tmp_path):
        """Generate coupling overview with synthetic data and verify PNG."""
        import matplotlib
        matplotlib.use('Agg')

        from symfluence.models.parflow.plotter import ParFlowPlotter

        project_dir = tmp_path / "test_project"
        (project_dir / "reporting" / "parflow_coupling").mkdir(parents=True)

        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'test_exp',
            'EXPERIMENT_TIME_START': '2000-01-01',
            'EXPERIMENT_TIME_END': '2000-04-01',
            'CATCHMENT_AREA': 100.0,
            'PARFLOW_TOP': 1500.0,
            'PARFLOW_BOT': 1400.0,
            'PARFLOW_POROSITY': 0.4,
            'PARFLOW_DX': 1000.0,
            'PARFLOW_DY': 1000.0,
            'PARFLOW_TIMESTEP_HOURS': 1.0,
        }

        logger_mock = MagicMock()
        plotter = ParFlowPlotter(config_dict, logger_mock)
        plotter.project_dir = project_dir

        # Build synthetic data
        dates = pd.date_range('2000-01-01', periods=2160, freq='h')  # 90 days
        np.random.seed(42)

        obs = pd.Series(
            5.0 + 2.0 * np.sin(np.arange(2160) * 2 * np.pi / 2160)
            + np.random.normal(0, 0.3, 2160),
            index=dates, name='obs',
        )
        surface_m3s = pd.Series(
            3.0 + 1.5 * np.sin(np.arange(2160) * 2 * np.pi / 2160),
            index=dates, name='surface_m3s',
        )
        subsurface_m3s = pd.Series(
            2.0 + 0.5 * np.sin(np.arange(2160) * 2 * np.pi / 2160 + 1.0),
            index=dates, name='subsurface_m3s',
        )
        total_m3s = surface_m3s + subsurface_m3s
        total_m3s.name = 'total_m3s'

        pressure = pd.Series(
            50.0 + 10.0 * np.sin(np.arange(2160) * 2 * np.pi / 2160),
            index=dates, name='pressure_head_m',
        )
        recharge_m_hr = pd.Series(
            0.002 + 0.001 * np.sin(np.arange(2160) * 2 * np.pi / (24 * 365)),
            index=dates, name='recharge_m_hr',
        )
        subsurface_m3hr = subsurface_m3s * 3600.0
        subsurface_m3hr.name = 'subsurface_m3hr'

        synthetic_data = {
            'obs': obs,
            'surface_m3s': surface_m3s,
            'subsurface_m3s': subsurface_m3s,
            'total_m3s': total_m3s,
            'pressure': pressure,
            'recharge_m_hr': recharge_m_hr,
            'subsurface_m3hr': subsurface_m3hr,
            'top': 1500.0,
            'bot': 1400.0,
            'porosity': 0.4,
            'dx': 1000.0,
            'dy': 1000.0,
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

        from symfluence.models.parflow.plotter import ParFlowPlotter

        project_dir = tmp_path / "test_project"
        (project_dir / "reporting" / "parflow_coupling").mkdir(parents=True)

        config_dict = {
            'SYMFLUENCE_DATA_DIR': str(tmp_path),
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'test_exp',
            'EXPERIMENT_TIME_START': '2000-01-01',
            'EXPERIMENT_TIME_END': '2000-04-01',
        }

        logger_mock = MagicMock()
        plotter = ParFlowPlotter(config_dict, logger_mock)
        plotter.project_dir = project_dir

        dates = pd.date_range('2000-01-01', periods=720, freq='h')
        surface = pd.Series(1.0, index=dates)
        sub = pd.Series(0.5, index=dates)

        synthetic_data = {
            'obs': None,
            'surface_m3s': surface,
            'subsurface_m3s': sub,
            'total_m3s': surface + sub,
            'pressure': pd.Series(50.0, index=dates),
            'recharge_m_hr': pd.Series(0.002, index=dates),
            'subsurface_m3hr': sub * 3600.0,
            'top': 1500.0,
            'bot': 1400.0,
            'porosity': 0.4,
            'dx': 1000.0,
            'dy': 1000.0,
        }

        with patch.object(plotter, '_collect_coupling_data', return_value=synthetic_data):
            result = plotter.plot_coupling_results('test_exp')

        assert result is not None
        assert Path(result).exists()
