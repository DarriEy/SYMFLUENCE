"""Wflow Worker."""
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import xarray as xr

from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker


@OptimizerRegistry.register_worker('WFLOW')
class WflowWorker(BaseWorker):
    """Worker for Wflow model calibration."""

    def __init__(self, config=None, logger=None):
        super().__init__(config, logger)

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params, settings_dir, **kwargs):
        try:
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings = data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW'
            staticmaps_name = config.get('WFLOW_STATICMAPS_FILE', 'wflow_staticmaps.nc')
            original_file = original_settings / staticmaps_name
            target_file = Path(settings_dir) / staticmaps_name
            if original_file.exists() and original_file != target_file:
                shutil.copy2(original_file, target_file)
            from .parameter_manager import WflowParameterManager
            pm = WflowParameterManager(config, self.logger, Path(settings_dir))
            return pm.update_model_files(params, Path(settings_dir))
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to apply Wflow parameters: {e}")
            return False

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        try:
            install_path = config.get('WFLOW_INSTALL_PATH', 'default')
            code_dir = config.get('SYMFLUENCE_CODE_DIR', '.')
            exe_path = Path(code_dir) / 'installs' / 'wflow' / 'bin' / 'wflow_cli' if install_path == 'default' else Path(install_path) / 'wflow_cli'
            config_file = config.get('WFLOW_CONFIG_FILE', 'wflow_sbm.toml')
            timeout = int(config.get('WFLOW_TIMEOUT', 7200))
            if not exe_path.exists():
                self.logger.error(f"wflow_cli not found at {exe_path}")
                return False
            result = subprocess.run(
                [str(exe_path), str(Path(settings_dir) / config_file)],
                cwd=str(settings_dir), capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Wflow execution failed: {e}")
            return False

    def calculate_metrics(self, config, output_dir, **kwargs):
        try:
            output_file = config.get('WFLOW_OUTPUT_FILE', 'output.nc')
            output_path = Path(output_dir) / output_file
            if not output_path.exists():
                matches = list(Path(output_dir).glob('output*.nc'))
                output_path = matches[0] if matches else output_path
            if not output_path.exists():
                return {'KGE': -999.0, 'NSE': -999.0}
            ds = xr.open_dataset(output_path)
            q_var = None
            for var in ['Q', 'Q_av', 'q_av']:
                if var in ds.data_vars:
                    q_var = ds[var]
                    break
            if q_var is None:
                ds.close()
                return {'KGE': -999.0, 'NSE': -999.0}
            spatial_dims = [d for d in q_var.dims if d not in ['time']]
            sim = q_var.max(dim=spatial_dims).to_series() if spatial_dims else q_var.to_series()
            ds.close()
            obs = self._load_observations(config)
            if obs is None:
                return {'KGE': -999.0, 'NSE': -999.0}
            sim, obs = sim.align(obs, join='inner')
            common_idx = sim.dropna().index.intersection(obs.dropna().index)
            sim, obs = sim.loc[common_idx], obs.loc[common_idx]
            if len(sim) < 30:
                return {'KGE': -999.0, 'NSE': -999.0}
            return self._streamflow_metrics.compute_all(obs.values, sim.values)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Metric calculation failed: {e}")
            return {'KGE': -999.0, 'NSE': -999.0}

    def _load_observations(self, config):
        try:
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            obs_dir = data_dir / f'domain_{domain_name}' / 'observations' / 'streamflow' / 'preprocessed'
            if not obs_dir.exists():
                return None
            obs_files = list(obs_dir.glob('*.csv'))
            if not obs_files:
                return None
            df = pd.read_csv(obs_files[0], parse_dates=True, index_col=0)
            return df['discharge_cms'] if 'discharge_cms' in df.columns else df.iloc[:, 0]
        except Exception:  # noqa: BLE001
            return None
