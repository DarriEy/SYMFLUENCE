"""
FUSE Worker

Worker implementation for FUSE model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry


logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('FUSE')
class FUSEWorker(BaseWorker):
    """
    Worker for FUSE model calibration.

    Handles parameter application to NetCDF files, FUSE execution,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to FUSE parameter NetCDF file.

        Args:
            params: Parameter values to apply
            settings_dir: FUSE settings directory (not used directly, path from config)
            **kwargs: Must include 'config' for path resolution

        Returns:
            True if successful
        """
        try:
            import netCDF4 as nc

            config = kwargs.get('config', self.config)

            # Construct parameter file path
            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            fuse_id = config.get('FUSE_FILE_ID', experiment_id)
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            param_file_path = (data_dir / f"domain_{domain_name}" / 'simulations' /
                              experiment_id / 'FUSE' / f"{domain_name}_{fuse_id}_para_def.nc")

            if not param_file_path.exists():
                self.logger.error(f"FUSE parameter file not found: {param_file_path}")
                return False

            # Update parameters in NetCDF file
            with nc.Dataset(param_file_path, 'r+') as ds:
                for param_name, value in params.items():
                    if param_name in ds.variables:
                        ds.variables[param_name][0] = value
                    else:
                        self.logger.warning(f"Parameter {param_name} not found in NetCDF file")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error applying FUSE parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run FUSE model.

        Args:
            config: Configuration dictionary
            settings_dir: FUSE settings directory
            output_dir: Output directory (not used directly by FUSE)
            **kwargs: Additional arguments including 'mode'

        Returns:
            True if model ran successfully
        """
        try:
            import subprocess

            mode = kwargs.get('mode', 'run_def')

            # Get FUSE executable path
            fuse_install = config.get('FUSE_INSTALL_PATH', 'default')
            if fuse_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                fuse_exe = data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
            else:
                fuse_exe = Path(fuse_install) / 'fuse.exe'

            # Get file manager path
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            filemanager_path = (data_dir / f"domain_{domain_name}" / 'settings' /
                               'FUSE' / 'fm_catch.txt')

            if not fuse_exe.exists():
                self.logger.error(f"FUSE executable not found: {fuse_exe}")
                return False

            if not filemanager_path.exists():
                self.logger.error(f"FUSE file manager not found: {filemanager_path}")
                return False

            # Execute FUSE
            cmd = [str(fuse_exe), str(filemanager_path), mode]
            fuse_settings_dir = filemanager_path.parent

            result = subprocess.run(
                cmd,
                cwd=fuse_settings_dir,
                capture_output=True,
                text=True,
                timeout=config.get('FUSE_TIMEOUT', 300)
            )

            if result.returncode != 0:
                self.logger.error(f"FUSE failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("FUSE execution timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running FUSE: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from FUSE output.

        Args:
            output_dir: Directory containing model outputs (not used directly)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            import xarray as xr
            import pandas as pd
            from symfluence.utils.common.metrics import get_KGE, get_NSE, get_RMSE, get_MAE

            # Get paths
            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            # Read observed streamflow
            obs_file_path = config.get('OBSERVATIONS_PATH', 'default')
            if obs_file_path == 'default':
                obs_file_path = (project_dir / 'observations' / 'streamflow' / 'preprocessed' /
                                f"{domain_name}_streamflow_processed.csv")
            else:
                obs_file_path = Path(obs_file_path)

            if not obs_file_path.exists():
                self.logger.error(f"Observation file not found: {obs_file_path}")
                return {'kge': self.penalty_score}

            # Read observations
            df_obs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            observed_streamflow = df_obs['discharge_cms'].resample('D').mean()

            # Read FUSE simulation output
            sim_file_path = (project_dir / 'simulations' / experiment_id / 'FUSE' /
                            f"{domain_name}_{experiment_id}_runs_def.nc")

            if not sim_file_path.exists():
                self.logger.error(f"Simulation file not found: {sim_file_path}")
                return {'kge': self.penalty_score}

            # Read simulations
            with xr.open_dataset(sim_file_path) as ds:
                if 'q_routed' in ds.variables:
                    simulated = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0)
                elif 'q_instnt' in ds.variables:
                    simulated = ds['q_instnt'].isel(param_set=0, latitude=0, longitude=0)
                else:
                    self.logger.error("No runoff variable found in FUSE output")
                    return {'kge': self.penalty_score}

                simulated_streamflow = simulated.to_pandas()

            # Get catchment area for unit conversion
            area_km2 = self._get_catchment_area(config, project_dir)

            # Convert FUSE output from mm/day to cms
            # Q(cms) = Q(mm/day) * Area(km2) / 86.4
            simulated_streamflow = simulated_streamflow * area_km2 / 86.4

            # Align time series
            common_index = observed_streamflow.index.intersection(simulated_streamflow.index)
            if len(common_index) == 0:
                self.logger.error("No overlapping time period")
                return {'kge': self.penalty_score}

            obs_aligned = observed_streamflow.loc[common_index].dropna()
            sim_aligned = simulated_streamflow.loc[common_index].dropna()

            common_index = obs_aligned.index.intersection(sim_aligned.index)
            obs_values = obs_aligned.loc[common_index].values
            sim_values = sim_aligned.loc[common_index].values

            if len(obs_values) == 0:
                self.logger.error("No valid data points")
                return {'kge': self.penalty_score}

            # Calculate metrics
            metrics = {
                'kge': float(get_KGE(obs_values, sim_values, transfo=1)),
                'nse': float(get_NSE(obs_values, sim_values, transfo=1)),
                'rmse': float(get_RMSE(obs_values, sim_values, transfo=1)),
                'mae': float(get_MAE(obs_values, sim_values, transfo=1)),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}

    def _get_catchment_area(self, config: Dict[str, Any], project_dir: Path) -> float:
        """
        Get catchment area for FUSE unit conversion.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path

        Returns:
            Catchment area in km2
        """
        try:
            import geopandas as gpd

            # Get catchment shapefile path
            catchment_path = config.get('CATCHMENT_PATH', 'default')
            catchment_name = config.get('CATCHMENT_SHP_NAME', 'default')

            if catchment_path == 'default':
                catchment_path = project_dir / 'shapefiles' / 'catchment'
            else:
                catchment_path = Path(catchment_path)

            if catchment_name == 'default':
                domain_name = config.get('DOMAIN_NAME')
                discretization = config.get('DOMAIN_DISCRETIZATION', 'elevation')
                catchment_name = f"{domain_name}_HRUs_{discretization}.shp"

            catchment_file = catchment_path / catchment_name

            if not catchment_file.exists():
                self.logger.warning(f"Catchment file not found: {catchment_file}")
                return 1000.0  # Default value

            # Read catchment and calculate area
            gdf = gpd.read_file(catchment_file)

            if 'GRU_area' in gdf.columns:
                total_area_m2 = gdf['GRU_area'].sum()
                return total_area_m2 / 1e6

            # Calculate from geometry
            if gdf.crs and not gdf.crs.is_geographic:
                total_area_m2 = gdf.geometry.area.sum()
            else:
                gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                total_area_m2 = gdf_utm.geometry.area.sum()

            return total_area_m2 / 1e6

        except Exception as e:
            self.logger.warning(f"Error calculating catchment area: {e}")
            return 1000.0

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        worker = FUSEWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
