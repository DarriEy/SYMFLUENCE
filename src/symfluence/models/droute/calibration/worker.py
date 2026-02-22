"""
dRoute Calibration Worker.

Worker implementation for dRoute routing parameter optimization with support for
both evolutionary and gradient-based calibration via automatic differentiation.
"""

import os
import sys
import signal
import random
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.metrics import kge, nse
from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins.project import resolve_data_subdir

# Try to import dRoute
try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False
    droute = None


@OptimizerRegistry.register_worker('DROUTE')
class DRouteWorker(BaseWorker):
    """
    Worker for dRoute routing parameter calibration.

    Supports:
    - Standard evolutionary optimization (evaluate -> route -> metrics)
    - Gradient-based optimization with AD when dRoute is compiled with CoDiPack/Enzyme
    - Efficient in-memory routing (no file I/O during calibration)

    Calibration Parameters:
        Common routing parameters that can be calibrated:
        - velocity: Base flow velocity (m/s)
        - diffusivity: Diffusion coefficient for diffusive wave routing
        - muskingum_k: Muskingum storage constant
        - muskingum_x: Muskingum weighting factor
        - manning_n: Manning's roughness coefficient
    """

    # Default parameter bounds for dRoute calibration
    PARAM_BOUNDS = {
        'velocity': (0.1, 5.0),         # m/s
        'diffusivity': (100.0, 5000.0), # m^2/s
        'muskingum_k': (0.1, 24.0),     # hours
        'muskingum_x': (0.0, 0.5),      # dimensionless
        'manning_n': (0.01, 0.1),       # dimensionless
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dRoute worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Lazy-loaded components
        self._network_config = None
        self._runoff_data = None
        self._observations = None
        self._time_index = None

        # Routing configuration
        self.routing_method = 'muskingum_cunge'
        self.routing_dt = 3600  # seconds

        if config:
            self.routing_method = config.get('DROUTE_ROUTING_METHOD', 'muskingum_cunge')
            self.routing_dt = config.get('DROUTE_ROUTING_DT', 3600)

    def supports_native_gradients(self) -> bool:
        """
        Check if native gradient computation is available.

        dRoute supports native gradients via CoDiPack/Enzyme AD when compiled
        with DROUTE_ENABLE_AD=ON.

        Returns:
            True if dRoute has AD support and is available.
        """
        if not HAS_DROUTE:
            return False

        # Check if dRoute was built with AD
        return hasattr(droute, 'gradient') or hasattr(droute, 'compute_gradients')

    def _load_data(self) -> bool:
        """
        Load runoff and observation data for calibration.

        Returns:
            True if data loaded successfully.
        """
        if self._runoff_data is not None:
            return True

        try:
            if self.config is None:
                self.logger.error("Config not set for dRoute worker")
                return False

            data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = self.config.get('DOMAIN_NAME', 'unknown')
            experiment_id = self.config.get('EXPERIMENT_ID', 'default')
            project_dir = data_dir / f"domain_{domain_name}"

            # Determine source model
            hydro_model = self.config.get('HYDROLOGICAL_MODEL', 'SUMMA')
            if ',' in str(hydro_model):
                from_model = str(hydro_model).split(',')[0].strip().upper()
            else:
                from_model = str(hydro_model).strip().upper()

            # Load runoff from source model
            runoff_dir = project_dir / f"simulations/{experiment_id}" / from_model
            runoff_file = self._find_runoff_file(runoff_dir)

            if runoff_file is None:
                self.logger.error(f"No runoff file found in {runoff_dir}")
                return False

            import xarray as xr
            ds = xr.open_dataset(runoff_file)

            # Find runoff variable
            runoff_var = None
            for var in ['averageRoutedRunoff', 'scalarTotalRunoff', 'runoff']:
                if var in ds:
                    runoff_var = var
                    break

            if runoff_var is None:
                for var in ds.data_vars:
                    if 'runoff' in var.lower():
                        runoff_var = var
                        break

            if runoff_var is None:
                self.logger.error("No runoff variable found")
                ds.close()
                return False

            self._runoff_data = ds[runoff_var].values
            if self._runoff_data.ndim == 1:
                self._runoff_data = self._runoff_data.reshape(-1, 1)

            self._time_index = pd.DatetimeIndex(ds.time.values)
            ds.close()

            # Load network configuration
            settings_dir = project_dir / 'settings' / 'dRoute'
            network_config_path = settings_dir / 'droute_network.yaml'

            if network_config_path.exists():
                import yaml
                with open(network_config_path, encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self._network_config = {
                    'n_segments': config['network']['n_segments'],
                    'downstream_idx': config['network']['downstream_idx'],
                    'outlet_indices': config['network']['outlet_indices'],
                    'routing_order': config['network']['routing_order'],
                    'slopes': config['geometry']['slopes'],
                    'lengths': config['geometry']['lengths'],
                    'widths': config['geometry']['widths'],
                    'hru_to_seg_idx': config['hru_mapping']['hru_to_seg_idx'],
                }
            else:
                self.logger.warning("Network config not found, trying mizuRoute topology")
                # Try mizuRoute topology
                from symfluence.models.droute.network_adapter import DRouteNetworkAdapter
                adapter = DRouteNetworkAdapter(self.logger)
                mizu_topo = project_dir / 'settings' / 'mizuRoute' / 'mizuRoute_topology.nc'
                if mizu_topo.exists():
                    topology = adapter.load_topology(mizu_topo, format='netcdf')
                    self._network_config = adapter.to_droute_format(
                        topology,
                        routing_method=self.routing_method,
                        routing_dt=self.routing_dt
                    )
                else:
                    self.logger.error("No network configuration found")
                    return False

            # Load observations
            obs_file = resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv"
            if obs_file.exists():
                obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
                obs_cms = obs_df.iloc[:, 0]

                # Resample to daily if needed
                obs_freq = pd.infer_freq(obs_cms.index[:100])
                if obs_freq and ('H' in str(obs_freq) or 'h' in str(obs_freq)):
                    obs_daily = obs_cms.resample('D').mean()
                else:
                    obs_daily = obs_cms

                # Align with runoff time
                assert self._time_index is not None
                forcing_dates = pd.to_datetime(self._time_index).normalize()
                self._observations = obs_daily.reindex(forcing_dates).values
            else:
                self.logger.warning("Observations not found")
                self._observations = None

            self.logger.info(f"Loaded dRoute data: {len(self._runoff_data)} timesteps")
            return True

        except Exception as e:
            self.logger.error(f"Error loading dRoute data: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _find_runoff_file(self, runoff_dir: Path) -> Optional[Path]:
        """Find runoff file in directory."""
        if not runoff_dir.exists():
            return None

        patterns = ['*_timestep.nc', '*_output.nc', '*_runs_def.nc', '*.nc']
        for pattern in patterns:
            files = list(runoff_dir.glob(pattern))
            if files:
                return files[0]
        return None

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply routing parameters.

        For dRoute, parameters are passed directly to routing function,
        no file writing needed.

        Args:
            params: Parameter values
            settings_dir: Settings directory (unused for dRoute)
            **kwargs: Additional arguments

        Returns:
            True (always succeeds for dRoute)
        """
        self._current_params = params
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run dRoute routing.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if routing succeeded.
        """
        try:
            if not self._load_data():
                return False

            params = getattr(self, '_current_params', {})

            # Run routing
            routed = self._route_with_params(params)
            if routed is None:
                return False

            self._last_routed = routed

            # Save output if requested
            if kwargs.get('save_output', False) and output_dir:
                self._save_output(routed, output_dir, config)

            return True

        except Exception as e:
            self.logger.error(f"Error running dRoute: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _route_with_params(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Execute routing with given parameters.

        Args:
            params: Routing parameters

        Returns:
            Routed streamflow array, or None on error
        """
        if self._runoff_data is None or self._network_config is None:
            return None

        n_time, n_hru = self._runoff_data.shape
        n_segments = self._network_config['n_segments']
        hru_to_seg = self._network_config['hru_to_seg_idx']
        downstream = self._network_config['downstream_idx']
        routing_order = self._network_config['routing_order']
        lengths = np.array(self._network_config['lengths'])
        slopes = np.array(self._network_config['slopes'])

        # Aggregate runoff to segments
        segment_runoff = np.zeros((n_time, n_segments))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                segment_runoff[:, seg_idx] += self._runoff_data[:, hru_idx]

        # Get velocity parameter (or use slope-based estimate)
        velocity = params.get('velocity', 1.0)
        if isinstance(velocity, (int, float)):
            velocities = np.full(n_segments, velocity)
        else:
            velocities = velocity * np.sqrt(slopes)

        velocities = np.clip(velocities, 0.1, 5.0)

        # Muskingum parameters
        musk_x = params.get('muskingum_x', 0.2)

        # Travel time and K
        K = lengths / velocities / 3600.0  # hours

        # Route
        routed = np.zeros((n_time, n_segments))
        dt_hours = self.routing_dt / 3600.0

        for t in range(n_time):
            for seg_idx in routing_order:
                Q_local = segment_runoff[t, seg_idx]

                # Upstream inflow
                Q_upstream = 0.0
                for up_idx in range(n_segments):
                    if downstream[up_idx] == seg_idx:
                        Q_upstream += routed[t, up_idx] if t > 0 else 0.0

                Q_prev = routed[t-1, seg_idx] if t > 0 else 0.0
                I = Q_local + Q_upstream
                I_prev = segment_runoff[t-1, seg_idx] + (
                    sum(routed[t-1, up_idx] for up_idx in range(n_segments)
                        if downstream[up_idx] == seg_idx) if t > 0 else 0.0
                )

                k = max(K[seg_idx], 0.1)
                denom = 2*k*(1-musk_x) + dt_hours
                C1 = (dt_hours - 2*k*musk_x) / denom
                C2 = (dt_hours + 2*k*musk_x) / denom
                C3 = (2*k*(1-musk_x) - dt_hours) / denom

                routed[t, seg_idx] = max(0, C1*I + C2*I_prev + C3*Q_prev)

        return routed

    def _save_output(self, routed: np.ndarray, output_dir: Path, config: Dict[str, Any]):
        """Save routed output to file."""
        import xarray as xr

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert self._network_config is not None
        outlet_indices = self._network_config['outlet_indices']
        outlet_idx = outlet_indices[0] if outlet_indices else 0

        ds = xr.Dataset(
            {
                'routedRunoff': (['time', 'seg'], routed.astype(np.float32)),
                'outletStreamflow': (['time'], routed[:, outlet_idx].astype(np.float32)),
            },
            coords={
                'time': self._time_index,
                'seg': np.arange(routed.shape[1]),
            }
        )

        domain_name = config.get('DOMAIN_NAME', 'unknown')
        output_file = output_dir / f"{domain_name}_droute_calibration.nc"
        ds.to_netcdf(output_file)
        ds.close()

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from routing output.

        Args:
            output_dir: Output directory (unused, uses in-memory results)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values.
        """
        try:
            routed = getattr(self, '_last_routed', None)
            if routed is None:
                return {'kge': self.penalty_score, 'error': 'No routing results'}

            if self._observations is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            # Get outlet streamflow
            outlet_indices = self._network_config['outlet_indices']
            outlet_idx = outlet_indices[0] if outlet_indices else 0
            sim = routed[:, outlet_idx]

            obs = self._observations

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]

            # Remove NaN
            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim = sim[valid_mask]
            obs = obs[valid_mask]

            if len(sim) < 10:
                return {'kge': self.penalty_score, 'error': 'Insufficient data'}

            kge_val = kge(obs, sim, transfo=1)
            nse_val = nse(obs, sim, transfo=1)

            if np.isnan(kge_val):
                kge_val = self.penalty_score
            if np.isnan(nse_val):
                nse_val = self.penalty_score

            return {
                'kge': float(kge_val),
                'nse': float(nse_val),
                'n_points': len(sim)
            }

        except Exception as e:
            self.logger.error(f"Error calculating dRoute metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss with respect to routing parameters.

        Uses dRoute's AD capabilities when available.

        Args:
            params: Current parameter values
            metric: Metric to compute gradient for ('kge' or 'nse')

        Returns:
            Dictionary of parameter gradients, or None if AD unavailable.
        """
        if not HAS_DROUTE:
            self.logger.warning("dRoute not available for gradient computation")
            return None

        if not self.supports_native_gradients():
            self.logger.warning("dRoute not compiled with AD support")
            return None

        if not self._load_data():
            return None

        try:
            # Use dRoute's gradient computation
            gradients = droute.compute_gradients(
                runoff=self._runoff_data,
                network=self._network_config,
                params=params,
                observations=self._observations,
                metric=metric
            )
            return gradients

        except Exception as e:
            self.logger.error(f"Error computing dRoute gradient: {e}")
            return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> tuple:
        """
        Evaluate loss and compute gradient in single pass.

        Args:
            params: Parameter values
            metric: Metric to evaluate

        Returns:
            Tuple of (loss_value, gradient_dict)
        """
        if not self.supports_native_gradients():
            # Fallback to separate evaluation
            self._current_params = params
            if self.run_model(self.config or {}, Path('.'), Path('.'), save_output=False):
                metrics = self.calculate_metrics(Path('.'), self.config or {})
                loss = -metrics.get('kge', self.penalty_score)
                return loss, None
            return self.penalty_score, None

        # Use AD for efficient value+gradient
        if not self._load_data():
            return self.penalty_score, None

        try:
            loss, gradients = droute.value_and_gradient(
                runoff=self._runoff_data,
                network=self._network_config,
                params=params,
                observations=self._observations,
                metric=metric
            )
            return float(loss), gradients

        except Exception as e:
            self.logger.error(f"Error in evaluate_with_gradient: {e}")
            return self.penalty_score, None

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_droute_parameters_worker(task_data)


def _evaluate_droute_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary containing params, config, etc.

    Returns:
        Result dictionary with score and metrics.
    """
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    })

    # Small random delay
    time.sleep(random.uniform(0.05, 0.2))  # nosec B311

    try:
        worker = DRouteWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'dRoute worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }


__all__ = ['DRouteWorker']
