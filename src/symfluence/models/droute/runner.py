"""
dRoute Model Runner.

Handles execution of the dRoute routing model.
Supports various routing methods (MC, Lag, IRF, etc.) and integrates with Symfluence workflow.
"""

import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xarray as xr

from ..registry import ModelRegistry
from ..base import BaseModelRunner
from ..execution import SpatialOrchestrator
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler

try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False


@ModelRegistry.register_runner('DROUTE', method_name='run_droute')
class DRouteRunner(BaseModelRunner, SpatialOrchestrator):
    """
    Runner for the dRoute routing model.
    """

    def _get_model_name(self) -> str:
        return "dRoute"

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        super().__init__(config, logger, reporting_manager=reporting_manager)
        self.setup_dir = self.project_dir / "settings" / "dRoute"
        
        # Load routing configuration
        self.routing_method = self.config_dict.get('DROUTE_METHOD', 'mc').lower()
        self.dt = float(self.config_dict.get('SETTINGS_MIZU_ROUTING_DT', 3600))
        
        # Output paths
        self.experiment_id = self.config_dict.get('EXPERIMENT_ID')
        self.output_dir = self.project_dir / 'simulations' / self.experiment_id / 'dRoute'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_droute(self) -> Optional[Path]:
        """Run dRoute routing workflow."""
        self.logger.info(f"Starting dRoute routing using method: {self.routing_method}")

        if not HAS_DROUTE:
            self.logger.error("dRoute not found. Please install it to use dRoute routing.")
            return None

        with symfluence_error_handler(
            "dRoute model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # 1. Load the pickled network
            network_data_path = self.setup_dir / 'dRoute_network.pkl'
            if not network_data_path.exists():
                self.logger.info("dRoute network pickle not found, running preprocessing")
                from .preprocessor import DRoutePreProcessor
                pre = DRoutePreProcessor(self.config_dict, self.logger)
                pre.run_preprocessing()
            
            with open(network_data_path, 'rb') as f:
                network_data = pickle.load(f)
            
            network = network_data['network']
            seg_areas = network_data['seg_areas']
            outlet_idx = network_data['outlet_idx']
            hru_to_seg_idx = network_data['hru_to_seg_idx']
            hru_ids_topo = network_data['hru_ids']
            
            # 2. Identify input runoff file
            # We look for the runoff file produced by the source model
            from_model = self.config_dict.get('MIZU_FROM_MODEL', '').upper()
            if not from_model:
                # Infer from configuration
                models = self.config_dict.get('HYDROLOGICAL_MODEL', '').split(',')
                for m in models:
                    m = m.strip().upper()
                    if m != 'DROUTE' and m != 'MIZUROUTE':
                        from_model = m
                        break
            
            # Find input file based on model convention
            input_dir = self.project_dir / 'simulations' / self.experiment_id / from_model
            
            # Special case for GR
            if from_model == 'GR':
                input_file = input_dir / f"{self.domain_name}_{self.experiment_id}_runs_def.nc"
            else:
                input_file = input_dir / f"{self.experiment_id}_timestep.nc"
            
            if not input_file.exists():
                # Try generic naming
                input_file = input_dir / f"{from_model}_output.nc"
                if not input_file.exists():
                    self.logger.error(f"Runoff input file not found: {input_file}")
                    return None

            # 3. Load and prepare runoff data
            self.logger.info(f"Loading runoff from {input_file}")
            with xr.open_dataset(input_file) as ds:
                # Find runoff variable
                routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
                if routing_var not in ds.data_vars:
                    # Fallback to common runoff names
                    for var in ['q_routed', 'q_instnt', 'basin__TotalRunoff', 'qsim', 'runoff']:
                        if var in ds.data_vars:
                            routing_var = var
                            break
                
                if routing_var not in ds.data_vars:
                    self.logger.error(f"Runoff variable {routing_var} not found in {input_file}")
                    return None
                
                runoff = ds[routing_var].values  # (time, gru)
                gru_ids = ds['gruId'].values if 'gruId' in ds else ds['hruId'].values
                times = ds['time'].values
                
                # Normalize time units to datetime if needed
                if not np.issubdtype(times.dtype, np.datetime64):
                    time_units = ds['time'].attrs.get('units', 'seconds since 1970-01-01')
                    times = pd.to_datetime(times, unit='s', origin='1970-01-01' if '1970' in time_units else '1990-01-01')
                else:
                    times = pd.to_datetime(times)

            # Reorder and convert runoff to CMS
            n_timesteps = len(times)
            n_reaches = network.num_reaches()
            runoff_reordered = np.zeros((n_timesteps, n_reaches))
            
            units = self.config_dict.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
            
            for i, gru_id in enumerate(gru_ids):
                gru_id_int = int(gru_id)
                if gru_id_int in hru_to_seg_idx:
                    seg_idx = hru_to_seg_idx[gru_id_int]
                    # Convert to CMS if needed (SUMMA/FUSE/GR often output m/s or mm/d)
                    if units == 'm/s':
                        runoff_reordered[:, seg_idx] = runoff[:, i] * seg_areas[seg_idx]
                    elif units == 'mm/d':
                        # 1 mm/day = 1 / (1000 * 86400) m/s
                        runoff_reordered[:, seg_idx] = (runoff[:, i] / (1000.0 * 86400.0)) * seg_areas[seg_idx]
                    else:
                        # Assume CMS
                        runoff_reordered[:, seg_idx] = runoff[:, i]

            # 4. Configure and run dRoute
            self.logger.info(f"Running dRoute simulation for {n_timesteps} timesteps")
            
            # Setup router
            config = droute.RouterConfig()
            config.dt = self.dt
            config.enable_gradients = False
            
            router_classes = {
                'mc': droute.MuskingumCungeRouter,
                'lag': droute.LagRouter,
                'irf': droute.IRFRouter,
                'kwt': droute.SoftGatedKWT,
                'diffusive': droute.DiffusiveWaveIFT,
            }
            
            RouterClass = router_classes.get(self.routing_method, droute.MuskingumCungeRouter)
            router = RouterClass(network, config)
            
            # Execute routing
            outlet_Q = np.zeros(n_timesteps)
            start_time = time.time()
            
            for t in range(n_timesteps):
                # Set lateral inflows
                for r in range(n_reaches):
                    router.set_lateral_inflow(r, float(runoff_reordered[t, r]))
                
                # Route
                router.route_timestep()
                
                # Get outlet discharge
                outlet_Q[t] = router.get_discharge(outlet_idx)
            
            elapsed = time.time() - start_time
            self.logger.info(f"dRoute simulation completed in {elapsed:.2f}s")

            # 5. Save results
            output_file = self.output_dir / f"{self.experiment_id}_dRoute_output.nc"
            
            ds_out = xr.Dataset(
                data_vars={
                    "routed_discharge": (["time"], outlet_Q)
                },
                coords={
                    "time": times
                },
                attrs={
                    "model": "dRoute",
                    "routing_method": self.routing_method,
                    "units": "m3 s-1"
                }
            )
            ds_out.to_netcdf(output_file)
            
            # Also save to standardized CSV for metrics
            self._save_to_standard_csv(times, outlet_Q)
            
            return output_file

    def _save_to_standard_csv(self, times, discharge):
        """Save results to standardized CSV format expected by Symfluence."""
        csv_path = self.output_dir / f"{self.experiment_id}_dRoute_discharge.csv"
        df = pd.DataFrame({
            'datetime': times,
            'dRoute_discharge_cms': discharge
        })
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Standardized discharge saved to {csv_path}")
