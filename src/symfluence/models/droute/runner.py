"""
dRoute Model Runner.

Manages the execution of the dRoute routing model.
Supports both Python API mode (preferred) and subprocess fallback.
"""

import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelRunner
from symfluence.models.droute.mixins import DRouteConfigMixin
from symfluence.models.droute.network_adapter import DRouteNetworkAdapter
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

# Try to import dRoute Python bindings
try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False
    droute = None


@ModelRegistry.register_runner('DROUTE', method_name='run_droute')
class DRouteRunner(BaseModelRunner, DRouteConfigMixin):  # type: ignore[misc]
    """
    Runner for the dRoute routing model.

    Supports two execution modes:
    1. Python API (default): Direct calls to dRoute Python bindings
       - Faster execution (no subprocess overhead)
       - Enables native gradient computation via AD
       - Requires dRoute compiled with Python bindings

    2. Subprocess mode: Executes dRoute as external command
       - Fallback when Python bindings unavailable
       - Works with any dRoute installation
       - No gradient support

    The runner handles:
    - Loading runoff data from source hydrological models
    - Configuring routing parameters
    - Executing routing computation
    - Writing routed streamflow output

    Attributes:
        config: Configuration settings
        logger: Logger instance
        execution_mode: 'python' or 'subprocess'
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize the dRoute runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig
            logger: Logger instance
            reporting_manager: Optional reporting manager for progress tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Determine execution mode
        preferred_mode = self.droute_execution_mode
        if preferred_mode == 'python' and not HAS_DROUTE:
            self.logger.warning(
                "dRoute Python bindings not available, falling back to subprocess mode. "
                "Install dRoute with: pip install droute"
            )
            self.execution_mode = 'subprocess'
        else:
            self.execution_mode = preferred_mode

        self.logger.debug(f"dRoute execution mode: {self.execution_mode}")

        # Initialize network adapter
        self.network_adapter = DRouteNetworkAdapter(logger)

        # Cached network and routing function
        self._network_config = None
        self._routing_fn = None

    def _get_model_name(self) -> str:
        """Return model name for dRoute."""
        return "dRoute"

    def _should_create_output_dir(self) -> bool:
        """dRoute creates directories on-demand."""
        return False

    def run_droute(self):
        """
        Run the dRoute routing model.

        Orchestrates the complete routing workflow:
        1. Load runoff from source hydrological model
        2. Load network topology
        3. Execute routing
        4. Write output

        Returns:
            Path to output directory
        """
        # Emit experimental warning on first use
        from symfluence.models.droute import _warn_experimental
        _warn_experimental()

        self.logger.info("Starting dRoute run")

        with symfluence_error_handler(
            "dRoute model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            if self.execution_mode == 'python':
                return self._run_python_mode()
            else:
                return self._run_subprocess_mode()

    def _run_python_mode(self):
        """
        Run dRoute using Python API.

        This is the preferred mode as it:
        - Avoids subprocess overhead
        - Enables native gradient computation
        - Provides better error handling
        """
        self.logger.info("Running dRoute in Python API mode")

        # Load network configuration
        network_config = self._load_network_config()
        if network_config is None:
            raise ModelExecutionError("Failed to load network configuration")

        # Load runoff input from source model
        runoff_data, time_index = self._load_runoff_input()
        if runoff_data is None:
            raise ModelExecutionError("Failed to load runoff data")

        self.logger.info(
            f"Loaded runoff data: {runoff_data.shape[0]} timesteps, "
            f"{runoff_data.shape[1]} HRUs"
        )

        # Get routing parameters
        routing_method = self.droute_routing_method
        routing_dt = self.droute_routing_dt

        # Execute routing
        try:
            if HAS_DROUTE:
                # Use dRoute Python bindings
                routed_flow = self._route_with_droute(
                    runoff_data,
                    network_config,
                    routing_method,
                    routing_dt
                )
            else:
                # Fallback: use numpy-based routing
                routed_flow = self._route_numpy_fallback(
                    runoff_data,
                    network_config,
                    routing_method,
                    routing_dt
                )

        except Exception as e:
            self.logger.error(f"Routing computation failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise ModelExecutionError(f"dRoute routing failed: {e}")

        # Save output
        output_dir = self._get_output_dir()
        self._save_output(routed_flow, time_index, network_config, output_dir)

        self.logger.info(f"dRoute completed. Output: {output_dir}")
        return output_dir

    def _run_subprocess_mode(self):
        """
        Run dRoute as external subprocess.

        Fallback mode when Python bindings are unavailable.
        """
        self.logger.info("Running dRoute in subprocess mode")

        # Get executable path
        droute_exe = self.get_model_executable(
            install_path_key='DROUTE_INSTALL_PATH',
            default_install_subpath='installs/droute/bin',
            exe_name_key='DROUTE_EXE',
            default_exe_name='droute',
            must_exist=True
        )

        # Get config file path
        settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
        config_file = settings_path / self.droute_config_file

        if not config_file.exists():
            raise ModelExecutionError(f"dRoute config file not found: {config_file}")

        # Setup output directory
        output_dir = self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup log directory
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'droute_log.txt'

        # Build command
        command = [str(droute_exe), str(config_file)]

        self.logger.debug(f"Running dRoute command: {' '.join(command)}")

        # Execute
        self.execute_subprocess(
            command,
            log_file,
            success_message="dRoute run completed successfully"
        )

        return output_dir

    def _load_network_config(self) -> Optional[Dict[str, Any]]:
        """Load network configuration from preprocessor output."""
        settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
        network_config_path = settings_path / 'droute_network.yaml'

        if network_config_path.exists():
            try:
                import yaml
                with open(network_config_path, encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # Flatten nested structure for routing
                network_config = {
                    'n_segments': config['network']['n_segments'],
                    'downstream_idx': config['network']['downstream_idx'],
                    'outlet_indices': config['network']['outlet_indices'],
                    'routing_order': config['network']['routing_order'],
                    'slopes': config['geometry']['slopes'],
                    'lengths': config['geometry']['lengths'],
                    'widths': config['geometry']['widths'],
                    'hru_to_seg_idx': config['hru_mapping']['hru_to_seg_idx'],
                }
                return network_config

            except Exception as e:
                self.logger.error(f"Error loading network config: {e}")
                return None

        # Try to load from topology file directly
        topology_path = self._find_topology_file()
        if topology_path:
            topology = self.network_adapter.load_topology(
                topology_path,
                format=self.droute_topology_format
            )
            return self.network_adapter.to_droute_format(
                topology,
                routing_method=self.droute_routing_method,
                routing_dt=self.droute_routing_dt
            )

        return None

    def _find_topology_file(self) -> Optional[Path]:
        """Find topology file in standard locations."""
        # Check dRoute settings
        settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
        droute_topo = settings_path / self.droute_topology_file
        if droute_topo.exists():
            return droute_topo

        # Check mizuRoute settings
        mizu_topo = self.project_dir / 'settings' / 'mizuRoute' / 'mizuRoute_topology.nc'
        if mizu_topo.exists():
            return mizu_topo

        return None

    def _load_runoff_input(self):
        """
        Load runoff data from source hydrological model.

        Returns:
            Tuple of (runoff_array, time_index) where:
            - runoff_array: 2D array [time, hru] of runoff values
            - time_index: DatetimeIndex of timestamps
        """
        import xarray as xr
        import pandas as pd

        input_dir = self._get_input_dir()
        self.logger.debug(f"Looking for runoff data in: {input_dir}")

        # Find runoff file
        runoff_file = self._find_runoff_file(input_dir)
        if runoff_file is None:
            self.logger.error(f"No runoff file found in {input_dir}")
            return None, None

        self.logger.info(f"Loading runoff from: {runoff_file}")

        try:
            ds = xr.open_dataset(runoff_file)

            # Find runoff variable
            runoff_vars = ['averageRoutedRunoff', 'scalarTotalRunoff', 'runoff', 'q_runoff']
            runoff_var = None
            for var in runoff_vars:
                if var in ds:
                    runoff_var = var
                    break

            if runoff_var is None:
                # Try to find any variable with 'runoff' in name
                for var in ds.data_vars:
                    if 'runoff' in var.lower():
                        runoff_var = var
                        break

            if runoff_var is None:
                self.logger.error(f"No runoff variable found in {runoff_file}")
                ds.close()
                return None, None

            self.logger.debug(f"Using runoff variable: {runoff_var}")

            # Extract data
            runoff = ds[runoff_var].values

            # Handle dimensions - ensure [time, hru] shape
            if runoff.ndim == 1:
                runoff = runoff.reshape(-1, 1)
            elif runoff.ndim == 3:
                # Possibly [time, gru, hru] - sum over HRUs within GRUs
                runoff = runoff.sum(axis=-1)

            # Get time coordinate
            time_index = pd.DatetimeIndex(ds.time.values)

            ds.close()
            return runoff, time_index

        except Exception as e:
            self.logger.error(f"Error loading runoff: {e}")
            self.logger.debug(traceback.format_exc())
            return None, None

    def _find_runoff_file(self, input_dir: Path) -> Optional[Path]:
        """Find runoff NetCDF file in input directory."""
        if not input_dir.exists():
            return None

        # Common runoff file patterns
        patterns = [
            '*_timestep.nc',
            '*_output.nc',
            '*_runs_def.nc',
            '*_runoff.nc',
            '*.nc',
        ]

        for pattern in patterns:
            files = list(input_dir.glob(pattern))
            if files:
                return files[0]

        return None

    def _route_with_droute(
        self,
        runoff: np.ndarray,
        network: Dict[str, Any],
        method: str,
        dt: int
    ) -> np.ndarray:
        """
        Route runoff using dRoute Python bindings.

        Args:
            runoff: 2D array [time, hru] of runoff values
            network: Network configuration
            method: Routing method name
            dt: Routing timestep in seconds

        Returns:
            2D array [time, segment] of routed streamflow
        """
        if not HAS_DROUTE:
            raise ImportError("dRoute Python bindings not available")

        # Map HRU runoff to segments
        n_time, n_hru = runoff.shape
        n_segments = network['n_segments']
        hru_to_seg = network['hru_to_seg_idx']

        # Aggregate runoff to segments
        segment_runoff = np.zeros((n_time, n_segments))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                segment_runoff[:, seg_idx] += runoff[:, hru_idx]

        # Create dRoute network object
        net = droute.Network(
            n_segments=n_segments,
            downstream=network['downstream_idx'],
            lengths=network['lengths'],
            slopes=network['slopes'],
            widths=network['widths']
        )

        # Create router with specified method
        method_map = {
            'muskingum_cunge': droute.RoutingMethod.MUSKINGUM_CUNGE,
            'irf': droute.RoutingMethod.IRF,
            'lag': droute.RoutingMethod.LAG,
            'diffusive_wave': droute.RoutingMethod.DIFFUSIVE_WAVE,
            'kwt': droute.RoutingMethod.KWT,
        }

        if method not in method_map:
            self.logger.warning(f"Unknown method '{method}', using muskingum_cunge")
            method = 'muskingum_cunge'

        router = droute.Router(net, method=method_map[method], dt=dt)

        # Route flow
        routed_flow = router.route(segment_runoff)

        return routed_flow

    def _route_numpy_fallback(
        self,
        runoff: np.ndarray,
        network: Dict[str, Any],
        method: str,
        dt: int
    ) -> np.ndarray:
        """
        Simple numpy-based routing fallback.

        Implements basic Muskingum-Cunge routing when dRoute unavailable.
        """
        self.logger.warning("Using numpy fallback routing (dRoute bindings unavailable)")

        n_time, n_hru = runoff.shape
        n_segments = network['n_segments']
        hru_to_seg = network['hru_to_seg_idx']
        downstream = network['downstream_idx']
        routing_order = network['routing_order']
        lengths = np.array(network['lengths'])
        slopes = np.array(network['slopes'])

        # Aggregate runoff to segments
        segment_runoff = np.zeros((n_time, n_segments))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                segment_runoff[:, seg_idx] += runoff[:, hru_idx]

        # Simple Muskingum routing
        # Q_out = C1*I + C2*I_prev + C3*Q_prev
        routed = np.zeros((n_time, n_segments))

        # Estimate velocity from slope (Manning-like)
        velocity = 1.0 * np.sqrt(slopes)  # m/s
        velocity = np.clip(velocity, 0.1, 5.0)

        # Travel time
        travel_time = lengths / velocity / 3600.0  # hours

        # Muskingum parameters
        K = travel_time  # hours
        x = 0.2  # weighting factor

        # Routing coefficients (simplified)
        for t in range(n_time):
            for seg_idx in routing_order:
                # Local runoff
                Q_local = segment_runoff[t, seg_idx]

                # Upstream inflow
                Q_upstream = 0.0
                for up_idx in range(n_segments):
                    if downstream[up_idx] == seg_idx:
                        Q_upstream += routed[t, up_idx] if t > 0 else 0.0

                # Previous timestep
                Q_prev = routed[t-1, seg_idx] if t > 0 else 0.0
                I_prev = segment_runoff[t-1, seg_idx] + (
                    sum(routed[t-1, up_idx] for up_idx in range(n_segments)
                        if downstream[up_idx] == seg_idx) if t > 0 else 0.0
                )

                # Total inflow
                I = Q_local + Q_upstream

                # Muskingum routing (explicit scheme)
                k = max(K[seg_idx], 0.1)
                denom = 2*k*(1-x) + dt/3600
                C1 = (dt/3600 - 2*k*x) / denom
                C2 = (dt/3600 + 2*k*x) / denom
                C3 = (2*k*(1-x) - dt/3600) / denom

                routed[t, seg_idx] = max(0, C1*I + C2*I_prev + C3*Q_prev)

        return routed

    def _get_input_dir(self) -> Path:
        """Get input directory for runoff data from source model."""
        from_model = self.droute_from_model.upper()
        if from_model == 'DEFAULT':
            hydro_model = self.config_dict.get('HYDROLOGICAL_MODEL', 'SUMMA')
            if ',' in str(hydro_model):
                from_model = str(hydro_model).split(',')[0].strip().upper()
            else:
                from_model = str(hydro_model).strip().upper()

        experiment_output = self.config_dict.get(f'EXPERIMENT_OUTPUT_{from_model}')
        if experiment_output and experiment_output != 'default':
            return Path(experiment_output)

        return self.project_dir / f"simulations/{self.experiment_id}" / from_model

    def _get_output_dir(self) -> Path:
        """Get output directory for routed streamflow."""
        droute_output = self.droute_experiment_output
        if droute_output and droute_output != 'default':
            output_dir = Path(droute_output)
        else:
            output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'dRoute'

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_output(
        self,
        routed_flow: np.ndarray,
        time_index,
        network: Dict[str, Any],
        output_dir: Path
    ):
        """Save routed streamflow to NetCDF file."""
        import xarray as xr

        n_time, n_segments = routed_flow.shape
        outlet_indices = network['outlet_indices']

        # Create dataset
        ds = xr.Dataset(
            {
                'routedRunoff': (['time', 'seg'], routed_flow.astype(np.float32)),
            },
            coords={
                'time': time_index,
                'seg': np.arange(n_segments),
            }
        )

        ds['routedRunoff'].attrs = {
            'units': 'm3/s',
            'long_name': 'Routed streamflow',
        }

        # Add outlet streamflow as separate variable
        if outlet_indices:
            outlet_idx = outlet_indices[0]
            ds['outletStreamflow'] = (['time'], routed_flow[:, outlet_idx])
            ds['outletStreamflow'].attrs = {
                'units': 'm3/s',
                'long_name': 'Streamflow at outlet',
            }

        # Add metadata
        ds.attrs['routing_method'] = self.droute_routing_method
        ds.attrs['routing_dt'] = self.droute_routing_dt
        ds.attrs['n_outlets'] = len(outlet_indices)

        # Save
        output_file = output_dir / f"{self.experiment_id}_droute_output.nc"
        ds.to_netcdf(output_file)
        ds.close()

        self.logger.info(f"Saved dRoute output to {output_file}")

    def compute_gradients(
        self,
        params: Dict[str, float],
        runoff: np.ndarray,
        network: Dict[str, Any],
        obs: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradients using dRoute's AD capabilities.

        Args:
            params: Routing parameters (e.g., velocity, diffusivity)
            runoff: Input runoff array
            network: Network configuration
            obs: Observed streamflow for loss calculation

        Returns:
            Dictionary of parameter gradients, or None if AD unavailable
        """
        if not HAS_DROUTE:
            self.logger.warning("dRoute bindings not available for gradient computation")
            return None

        if not self.droute_enable_gradients:
            self.logger.warning("Gradient computation not enabled in config")
            return None

        try:
            # Check if AD is enabled in dRoute build
            if not hasattr(droute, 'gradient'):
                self.logger.warning("dRoute not compiled with AD support")
                return None

            # Create AD-enabled router
            net = droute.Network(
                n_segments=network['n_segments'],
                downstream=network['downstream_idx'],
                lengths=network['lengths'],
                slopes=network['slopes'],
                widths=network['widths']
            )

            router = droute.Router(net, enable_ad=True)

            # Compute gradients
            gradients = router.compute_gradients(
                runoff=runoff,
                params=params,
                observations=obs,
                metric='kge'
            )

            return gradients

        except Exception as e:
            self.logger.error(f"Error computing gradients: {e}")
            self.logger.debug(traceback.format_exc())
            return None


__all__ = ['DRouteRunner']
