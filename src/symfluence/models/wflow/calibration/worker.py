# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow Worker."""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
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
        self._routing_params = None  # Populated per iteration by apply_parameters

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params, settings_dir, **kwargs):
        try:
            # Extract routing params (not written to staticmaps, applied post-hoc)
            self._routing_params = {k: v for k, v in params.items() if k.startswith('ROUTE_')}
            wflow_params = {k: v for k, v in params.items() if not k.startswith('ROUTE_')}
            params = wflow_params

            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_settings = data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW'
            settings_dir = Path(settings_dir)
            staticmaps_name = config.get('WFLOW_STATICMAPS_FILE', 'wflow_staticmaps.nc')

            # Copy staticmaps from original settings to process directory
            original_file = original_settings / staticmaps_name
            target_file = settings_dir / staticmaps_name
            if original_file.exists() and original_file != target_file:
                shutil.copy2(original_file, target_file)

            # Copy and patch the TOML config for process-specific paths
            config_file = config.get('WFLOW_CONFIG_FILE', 'wflow_sbm.toml')
            original_toml = original_settings / config_file
            target_toml = settings_dir / config_file
            if original_toml.exists() and original_toml != target_toml:
                self._patch_toml(original_toml, target_toml, settings_dir, kwargs.get('output_dir') or kwargs.get('proc_output_dir'))

            from .parameter_manager import WflowParameterManager
            pm = WflowParameterManager(config, self.logger, settings_dir)
            return pm.update_model_files(params, settings_dir)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to apply Wflow parameters: {e}")
            return False

    def _patch_toml(self, source_toml, target_toml, settings_dir, output_dir):
        """Copy TOML and patch path_static and dir_output for process isolation."""
        content = source_toml.read_text()
        # Point path_static to the process-specific staticmaps
        content = re.sub(
            r'(path_static\s*=\s*)"[^"]*"',
            lambda m: f'{m.group(1)}"{settings_dir / "wflow_staticmaps.nc"}"',
            content,
        )
        # Redirect output to process-specific dir if available
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            content = re.sub(
                r'(dir_output\s*=\s*)"[^"]*"',
                lambda m: f'{m.group(1)}"{output_dir}"',
                content,
            )
            content = re.sub(
                r'(path\s*=\s*)"[^"]*/output\.csv"',
                lambda m: f'{m.group(1)}"{output_dir / "output.csv"}"',
                content,
            )
            # Remove state I/O paths for calibration (cold start each iteration)
            # Keep [state] and [state.variables] sections — Wflow requires them
            content = re.sub(r'path_input\s*=\s*"[^"]*"\n?', '', content)
            content = re.sub(r'path_output\s*=\s*"[^"]*"\n?', '', content)
        target_toml.write_text(content)

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        try:
            install_path = config.get('WFLOW_INSTALL_PATH', 'default')
            data_dir = config.get('SYMFLUENCE_DATA_DIR', '.')
            exe_path = Path(data_dir) / 'installs' / 'wflow' / 'bin' / 'wflow_cli' if install_path == 'default' else Path(install_path) / 'wflow_cli'
            config_file = config.get('WFLOW_CONFIG_FILE', 'wflow_sbm.toml')
            timeout = int(config.get('WFLOW_TIMEOUT', 7200))
            if not exe_path.exists():
                self.logger.error(f"wflow_cli not found at {exe_path}")
                return False
            result = subprocess.run(
                [str(exe_path), str(Path(settings_dir) / config_file)],
                cwd=str(settings_dir), capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0 and result.stderr:
                self.logger.warning(f"Wflow stderr: {result.stderr[-500:]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Wflow execution failed: {e}")
            return False

    def calculate_metrics(self, output_dir, config, **kwargs):
        # Note: base_worker calls calculate_metrics(task.output_dir, task.config)
        try:
            sim = self._load_simulated_streamflow(
                Path(output_dir), config, self._routing_params or None
            )
            if sim is None:
                self.logger.warning(f"No sim data found in {output_dir}")
                return {'KGE': -999.0, 'NSE': -999.0}
            obs = self._load_observations(config)
            if obs is None:
                self.logger.warning("No observation data found")
                return {'KGE': -999.0, 'NSE': -999.0}
            # Ensure datetime index and resample sub-daily to daily
            if not isinstance(sim.index, pd.DatetimeIndex):
                sim.index = pd.to_datetime(sim.index)
            if not isinstance(obs.index, pd.DatetimeIndex):
                obs.index = pd.to_datetime(obs.index)
            if len(sim) > 1:
                median_dt = pd.Series(sim.index).diff().median()
                if median_dt < pd.Timedelta(days=1):
                    sim = sim.resample('D').mean()
            # Filter to calibration period (excludes cold-start spinup)
            cal_period = config.get('CALIBRATION_PERIOD', '') if isinstance(config, dict) else ''
            if cal_period:
                parts = [p.strip() for p in cal_period.split(',')]
                if len(parts) == 2:
                    sim = sim.loc[parts[0]:parts[1]]  # type: ignore[misc]
                    obs = obs.loc[parts[0]:parts[1]]  # type: ignore[misc]
            sim, obs = sim.align(obs, join='inner')
            common_idx = sim.dropna().index.intersection(obs.dropna().index)
            sim, obs = sim.loc[common_idx], obs.loc[common_idx]
            if len(sim) < 30:
                self.logger.warning(f"Insufficient data: n={len(sim)} < 30")
                return {'KGE': -999.0, 'NSE': -999.0}
            self.logger.info(
                f"Metrics: n={len(sim)}, sim_mean={sim.mean():.2f}, obs_mean={obs.mean():.2f}, "
                f"output_dir={output_dir}, cal_period={cal_period}"
            )
            metrics = self._streamflow_metrics.calculate_metrics(obs.values, sim.values)
            self.logger.info(f"Computed metrics: {metrics}")
            return metrics
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Metric calculation failed: {e}")
            return {'KGE': -999.0, 'NSE': -999.0}

    def _load_simulated_streamflow(self, output_dir: Path, config: dict = None, routing_params: dict = None):
        """Load Q from Wflow CSV or netCDF output.

        For lumped models using soil_surface_water__net_runoff_volume_flux,
        converts from mm/dt to m³/s using basin area and timestep, then
        applies linear reservoir routing to smooth the instantaneous flux.
        """
        q = None
        # Try CSV first (primary format from [output.csv] TOML section)
        csv_matches = list(output_dir.glob('output*.csv'))
        if csv_matches:
            df = pd.read_csv(csv_matches[0], parse_dates=[0], index_col=0, on_bad_lines='skip')
            for col in ['Q', 'Q_av', 'q_av']:
                if col in df.columns:
                    q = df[col]
                    break
            if q is None and len(df.columns) == 1:
                q = df.iloc[:, 0]
        # Fallback: netCDF
        if q is None:
            nc_matches = list(output_dir.glob('output*.nc'))
            if nc_matches:
                ds = xr.open_dataset(nc_matches[0])
                for var in ['Q', 'Q_av', 'q_av']:
                    if var in ds.data_vars:
                        q_var = ds[var]
                        spatial_dims = [d for d in q_var.dims if d not in ['time']]
                        q = q_var.max(dim=spatial_dims).to_series() if spatial_dims else q_var.to_series()
                        break
                ds.close()
        if q is None:
            return None
        # Convert mm/dt → m³/s if output is from net_runoff_volume_flux (lumped)
        if self._is_runoff_flux_output(output_dir, config) and config:
            area, dt = self._get_basin_area_and_timestep(config)
            if area > 0:
                q = q * area / (dt * 1000.0)
                self.logger.info(f"Converted Q from mm/dt to m³/s (area={area:.0f} m², dt={dt}s)")
        # Apply post-hoc linear reservoir routing for lumped mode
        # (net_runoff_volume_flux is instantaneous — needs smoothing)
        if self._is_runoff_flux_output(output_dir, config) and routing_params:
            q = self._apply_routing(q, routing_params)
        return q

    def _is_runoff_flux_output(self, output_dir: Path, config: dict = None):
        """Check if the TOML output uses net_runoff_volume_flux (mm/dt units)."""
        toml_name = config.get('WFLOW_CONFIG_FILE', 'wflow_sbm.toml') if config else 'wflow_sbm.toml'
        # Search multiple possible locations for the TOML
        search_dirs = [output_dir.parent / 'settings', output_dir.parent]
        # Also check the original domain settings (works during DDS calibration)
        if config:
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = config.get('DOMAIN_NAME', '')
            search_dirs.append(data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW')
        for settings_dir in search_dirs:
            toml_path = settings_dir / toml_name
            if toml_path.exists():
                content = toml_path.read_text()
                if 'net_runoff_volume_flux' in content:
                    return True
                if 'volume_flow_rate' in content:
                    return False
        return False

    def _get_basin_area_and_timestep(self, config):
        """Read basin area from staticmaps and timestep from config."""
        area = 0.0
        dt = int(config.get('WFLOW_TIMESTEP', 3600))
        try:
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            staticmaps_name = config.get('WFLOW_STATICMAPS_FILE', 'wflow_staticmaps.nc')
            staticmaps_path = data_dir / f'domain_{domain_name}' / 'settings' / 'WFLOW' / staticmaps_name
            if staticmaps_path.exists():
                ds = xr.open_dataset(staticmaps_path)
                if 'wflow_cellarea' in ds:
                    area = float(np.nansum(ds['wflow_cellarea'].values))
                ds.close()
        except Exception:  # noqa: BLE001
            pass
        return area, dt

    @staticmethod
    def _apply_routing(q, routing_params):
        """Two-store linear reservoir routing (fast + slow) with baseflow offset.

        Splits instantaneous runoff into fast (surface) and slow (baseflow)
        components, each routed through an exponential reservoir, plus a
        constant baseflow term for irreducible glacial/deep-GW contribution:
            S_fast(t) = alpha * S_fast(t-1) + split * Q_in(t)
            S_slow(t) = beta  * S_slow(t-1) + (1-split) * Q_in(t)
            Q_out(t)  = (1-alpha) * S_fast(t) + (1-beta) * S_slow(t) + baseflow

        Parameters (from routing_params dict):
            ROUTE_ALPHA:    fast reservoir retention [0, 0.95]
            ROUTE_BETA:     slow reservoir retention [0.9, 0.9999]
            ROUTE_SPLIT:    fraction to fast store [0.1, 0.9]
            ROUTE_BASEFLOW: constant baseflow offset in m³/s [0, 15]
        """
        alpha = routing_params.get('ROUTE_ALPHA', 0.5)
        beta = routing_params.get('ROUTE_BETA', 0.98)
        split = routing_params.get('ROUTE_SPLIT', 0.5)
        baseflow = routing_params.get('ROUTE_BASEFLOW', 0.0)
        vals = q.values.astype(float)
        n = len(vals)
        s_fast = 0.0
        s_slow = 0.0
        out = np.empty(n)
        for i in range(n):
            q_in = max(vals[i], 0.0)
            s_fast = alpha * s_fast + split * q_in
            s_slow = beta * s_slow + (1.0 - split) * q_in
            out[i] = (1.0 - alpha) * s_fast + (1.0 - beta) * s_slow + baseflow
        return pd.Series(out, index=q.index, name=q.name)

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
