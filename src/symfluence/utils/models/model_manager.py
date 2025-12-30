# src/symfluence/utils/models/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any
import pandas as pd 

# Registry
from symfluence.utils.models.registry import ModelRegistry

# Visualization (keeping these as they are not model-specific in the same way)
from symfluence.utils.reporting.reporting_utils import VisualizationReporter # type: ignore
from symfluence.utils.reporting.result_vizualisation_utils import TimeseriesVisualizer # type: ignore

# Data management
from symfluence.utils.data.utilities.archive_utils import tar_directory # type: ignore

class ModelManager:
    """
    Manages all hydrological model operations within the SYMFLUENCE framework.
    Uses a registry-based system for easy extension with new models.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Model Manager.
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
    def preprocess_models(self):
        """
        Process the forcing data into model-specific formats.
        """
        self.logger.info("Starting model-specific preprocessing")

        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')

        for model in models:
            model = model.strip()
            try:
                # Create model input directory
                model_input_dir = self.project_dir / "forcing" / f"{model}_input"
                model_input_dir.mkdir(parents=True, exist_ok=True)

                # Select preprocessor for this model from registry
                preprocessor_class = ModelRegistry.get_preprocessor(model)

                if preprocessor_class is None:
                    # Models that truly don't need preprocessing (e.g., FLASH)
                    if model in ['FLASH']:
                        self.logger.info(f"Model {model} doesn't require preprocessing")
                    else:
                        self.logger.warning(f"Unsupported model or no preprocessor: {model}.")
                    continue

                # Run model-specific preprocessing
                preprocessor = preprocessor_class(self.config, self.logger)
                preprocessor.run_preprocessing()

                # ----- Routing preprocessing hooks -----
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)

                # MizuRoute is special as it's often a secondary step
                if needs_mizuroute:
                    mizu_preproc_class = ModelRegistry.get_preprocessor('MIZUROUTE')
                    if mizu_preproc_class:
                        if model == 'SUMMA':
                            self.logger.info("Initializing mizuRoute preprocessor for SUMMA")
                            self.config['MIZU_FROM_MODEL'] = 'SUMMA'
                            mp = mizu_preproc_class(self.config, self.logger)
                            mp.run_preprocessing()
                        elif model == 'FUSE':
                            self.logger.info("Initializing mizuRoute preprocessor for FUSE")
                            self.config['MIZU_FROM_MODEL'] = 'FUSE'
                            mp = mizu_preproc_class(self.config, self.logger)
                            mp.run_preprocessing()

            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        self.logger.info("Model-specific preprocessing completed")

    def run_models(self):
        """Execute model runs using the registry."""
        self.logger.info("Starting model runs")
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            try:
                self.logger.info(f"Running model: {model}")
                runner_class = ModelRegistry.get_runner(model)
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model or no runner registered: {model}")
                    continue

                runner = runner_class(self.config, self.logger)
                method_name = ModelRegistry.get_runner_method(model)
                if method_name and hasattr(runner, method_name):
                    getattr(runner, method_name)()
                else:
                    self.logger.error(f"Runner method '{method_name}' not found for model: {model}")
                    continue

                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)

                if needs_mizuroute:
                    mizuroute_runner_class = ModelRegistry.get_runner('MIZUROUTE')
                    if mizuroute_runner_class:
                        if model == 'FUSE':
                            try:
                                fuse_spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
                                if fuse_spatial_mode in ['semi_distributed', 'distributed']:
                                    self.logger.info("Converting distributed FUSE output to mizuRoute format (gru/gruId)")
                                    # self._convert_fuse_distributed_to_mizuroute_format()
                            except Exception as e:
                                self.logger.error(f"FUSE→mizuRoute distributed conversion failed: {e}")
                                raise
                            mr = mizuroute_runner_class(self.config, self.logger)
                            mr.run_mizuroute()
                        elif model == 'SUMMA':
                            if domain_method == 'lumped' and routing_delineation == 'river_network':
                                self.logger.info("Converting lumped SUMMA output for distributed routing")
                                self._convert_lumped_to_distributed_routing()
                            mr = mizuroute_runner_class(self.config, self.logger)
                            mr.run_mizuroute()

            except Exception as e:
                self.logger.error(f"Error running model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def postprocess_results(self):
        """Post-process model results using the registry."""
        self.logger.info("Starting model post-processing")
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            try:
                # Get postprocessor class from registry
                postprocessor_class = ModelRegistry.get_postprocessor(model)
                
                if postprocessor_class is None:
                    self.logger.info(f"No postprocessor defined for model: {model}")
                    continue
                
                # Create postprocessor instance
                postprocessor = postprocessor_class(self.config, self.logger)
                
                # Run postprocessing
                if hasattr(postprocessor, 'extract_streamflow'):
                    postprocessor.extract_streamflow()
                elif hasattr(postprocessor, 'extract_results'):
                    postprocessor.extract_results()
                else:
                    self.logger.warning(f"No extraction method found for {model} postprocessor")
                    continue
                
                self.logger.info(f"Post-processing completed for {model}")
                
            except Exception as e:
                self.logger.error(f"Error post-processing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Create final visualizations
        try:
            tv = TimeseriesVisualizer(self.config, self.logger)
            tv.create_visualizations()
            self.logger.info("Time series visualizations created")
        except Exception as e:
            self.logger.error(f"Error creating time series visualizations: {str(e)}")

    def _needs_mizuroute_routing(self, domain_method: str, routing_delineation: str) -> bool:
        """Helper to determine if mizuRoute is needed."""
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        if 'FUSE' in [m.strip() for m in models]:
            fuse_spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            fuse_routing = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')
            if fuse_routing == 'mizuRoute':
                if fuse_spatial_mode in ['semi_distributed', 'distributed']:
                    return True
                elif fuse_spatial_mode == 'lumped' and routing_delineation == 'river_network':
                    return True
        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        return False

    def visualize_outputs(self):
        """Visualize model outputs."""
        self.logger.info('Starting model output visualisation')
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        for model in models:
            model = model.strip()
            visualizer = VisualizationReporter(self.config, self.logger)
            if model == 'SUMMA':
                obs_files = [('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"))]
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', '')
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                if self._needs_mizuroute_routing(domain_method, routing_delineation):
                    visualizer.update_sim_reach_id()
                    model_outputs = [(model, str(self.project_dir / "simulations" / self.experiment_id / "mizuRoute" / f"{self.experiment_id}*.nc"))]
                    visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
                else:
                    summa_output_file = str(self.project_dir / "simulations" / self.experiment_id / "SUMMA" / f"{self.experiment_id}_timestep.nc")
                    model_outputs = [(model, summa_output_file)]
                    visualizer.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)
            elif model == 'FUSE':
                model_outputs = [("FUSE", str(self.project_dir / "simulations" / self.experiment_id / "FUSE" / f"{self.domain_name}_{self.experiment_id}_runs_best.nc"))]
                obs_files = [('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"))]
                visualizer.plot_fuse_streamflow_simulations_vs_observations(model_outputs, obs_files)
            else:
                self.logger.info(f"Visualization for {model} not yet implemented")

    def _convert_lumped_to_distributed_routing(self):
        """Ported conversion logic for lumped SUMMA to distributed routing."""
        import xarray as xr
        import numpy as np
        import netCDF4 as nc4
        import shutil
        import tempfile
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_output_dir = self.project_dir / "simulations" / experiment_id / "SUMMA"
        mizuroute_settings_dir = self.project_dir / "settings" / "mizuRoute"
        summa_timestep_file = summa_output_dir / f"{experiment_id}_timestep.nc"
        
        if not summa_timestep_file.exists():
            return

        topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
        if not topology_file.exists():
            return

        with xr.open_dataset(topology_file) as mizuTopology:
            hru_id = 1 
        
        routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
        summa_output = xr.open_dataset(summa_timestep_file, decode_times=False)
        
        try:
            mizuForcing = xr.Dataset()
            original_time = summa_output['time']
            mizuForcing['time'] = xr.DataArray(original_time.values, dims=('time',), attrs=dict(original_time.attrs))
            if 'units' in mizuForcing['time'].attrs:
                mizuForcing['time'].attrs['units'] = mizuForcing['time'].attrs['units'].replace('T', ' ')
            
            mizuForcing['gru'] = xr.DataArray([hru_id], dims=('gru',), attrs={'long_name': 'Index of GRU', 'units': '-'})
            mizuForcing['gruId'] = xr.DataArray([hru_id], dims=('gru',), attrs={'long_name': 'ID of grouped response unit', 'units': '-'})
            mizuForcing.attrs.update(summa_output.attrs)
            
            source_var = None
            for var in [routing_var, 'averageRoutedRunoff', 'basin__TotalRunoff']:
                if var in summa_output:
                    source_var = var
                    break
            
            if source_var:
                lumped_runoff = summa_output[source_var].values
                if len(lumped_runoff.shape) == 2:
                    lumped_runoff = lumped_runoff[:, 0]
                mizuForcing[routing_var] = xr.DataArray(lumped_runoff[:, np.newaxis], dims=('time', 'gru'), attrs={'units': 'm/s'})
                summa_output.close()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=summa_output_dir) as tmp_file:
                    temp_path = tmp_file.name
                mizuForcing.to_netcdf(temp_path, format='NETCDF4')
                mizuForcing.close()
                shutil.move(temp_path, summa_timestep_file)
        finally:
            summa_output.close()
