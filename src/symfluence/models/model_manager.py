# src/symfluence/models/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING

import pandas as pd

from symfluence.core.base_manager import BaseManager
from symfluence.models.registry import ModelRegistry
from symfluence.optimization.workers.utilities.routing_decider import RoutingDecider

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class ModelManager(BaseManager):
    """
    Manages all hydrological model operations within the SYMFLUENCE framework.

    Uses a registry-based system for easy extension with new models.
    Inherits from BaseManager for standardized initialization and common patterns.
    """

    # Shared routing decision logic (class-level for efficiency)
    _routing_decider = RoutingDecider()

    def _resolve_model_workflow(self) -> List[str]:
        """
        Resolve the order of models to run, including implicit dependencies.

        Returns:
            List of model names to execute in order.
        """
        models_str = self.config.model.hydrological_model or ''
        configured_models = [m.strip() for m in str(models_str).split(',') if m.strip()]
        execution_list = []

        # Models that support routing via mizuRoute
        # Note: MESH, HYPE, and NGEN have internal routing, so don't need mizuRoute
        routable_models = {'SUMMA', 'FUSE', 'GR'}

        for model in configured_models:
            if model not in execution_list:
                execution_list.append(model)

            # Check implicit dependencies (e.g. mizuRoute) using shared routing decider
            if model in routable_models:
                if self._routing_decider.needs_routing(self.config_dict, model):
                    self._ensure_mizuroute_in_workflow(execution_list, source_model=model)

        return execution_list

    def _ensure_mizuroute_in_workflow(self, execution_list: List[str], source_model: str):
        """Helper to add mizuRoute to workflow and set context."""
        if 'MIZUROUTE' not in execution_list:
            execution_list.append('MIZUROUTE')
            self.logger.info(f"Automatically adding MIZUROUTE to workflow (dependency of {source_model})")

        # Check if MIZU_FROM_MODEL is set in config
        mizu_from = self._get_config_value(
            lambda: self.config.model.mizuroute.from_model if self.config.model.mizuroute else None,
            default=None
        )
        if not mizu_from:
            # Log the source model (config is immutable, so we can't update it)
            self.logger.info(f"MIZU_FROM_MODEL not set, using {source_model} as source")

    def preprocess_models(self, params: Optional[Dict[str, Any]] = None):
        """
        Process the forcing data into model-specific formats.

        Args:
            params: Optional dictionary of parameter values (for calibration)
        """
        self.logger.debug("Starting model-specific preprocessing")

        workflow = self._resolve_model_workflow()
        self.logger.debug(f"Preprocessing workflow order: {workflow}")

        for model in workflow:
            try:
                # Create model input directory
                model_input_dir = self.project_dir / "forcing" / f"{model}_input"
                model_input_dir.mkdir(parents=True, exist_ok=True)

                # Select preprocessor for this model from registry
                preprocessor_class = ModelRegistry.get_preprocessor(model)

                if preprocessor_class is None:
                    # Models that truly don't need preprocessing (e.g., LSTM)
                    if model in ['LSTM']:
                        self.logger.debug(f"Model {model} doesn't require preprocessing")
                    else:
                        # Only warn if it's a primary model, not a utility like MIZUROUTE which definitely has one
                        self.logger.debug(f"No preprocessor registered for {model} (or not required).")
                    continue

                # Run model-specific preprocessing
                self.logger.debug(f"Running preprocessor for {model}")

                # Check if preprocessor accepts params
                import inspect
                sig = inspect.signature(preprocessor_class.__init__)
                if 'params' in sig.parameters:
                    preprocessor = preprocessor_class(self.config, self.logger, params=params)
                else:
                    preprocessor = preprocessor_class(self.config, self.logger)

                preprocessor.run_preprocessing()

            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        self.logger.info("Model-specific preprocessing completed")

        

    def run_models(self):
        """Execute model runs using the registry."""
        self.logger.info("Starting model runs")

        workflow = self._resolve_model_workflow()
        self.logger.info(f"Execution workflow order: {workflow}")

        for model in workflow:
            try:
                self.logger.info(f"Running model: {model}")
                runner_class = ModelRegistry.get_runner(model)
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model or no runner registered: {model}")
                    continue

                runner = runner_class(self.config, self.logger, reporting_manager=self.reporting_manager)
                method_name = ModelRegistry.get_runner_method(model)
                if method_name and hasattr(runner, method_name):
                    getattr(runner, method_name)()
                else:
                    self.logger.error(f"Runner method '{method_name}' not found for model: {model}")
                    continue

            except Exception as e:
                self.logger.error(f"Error running model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def postprocess_results(self):
        """Post-process model results using the registry."""
        self.logger.info("Starting model post-processing")

        workflow = self._resolve_model_workflow()

        for model in workflow:
            try:
                # Get postprocessor class from registry
                postprocessor_class = ModelRegistry.get_postprocessor(model)

                if postprocessor_class is None:
                    continue

                self.logger.info(f"Post-processing {model}")
                # Create postprocessor instance
                postprocessor = postprocessor_class(self.config, self.logger, reporting_manager=self.reporting_manager)

                # Run postprocessing
                # Standardized interface: extract_streamflow is the main entry point
                if hasattr(postprocessor, 'extract_streamflow'):
                    postprocessor.extract_streamflow()
                elif hasattr(postprocessor, 'extract_results'):
                    # Legacy support for models that might still use extract_results (e.g. HYPE if not updated)
                    postprocessor.extract_results()
                else:
                    self.logger.warning(f"No extraction method found for {model} postprocessor")
                    continue

            except Exception as e:
                self.logger.error(f"Error post-processing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Note: visualize_timeseries_results is now triggered automatically by extract_streamflow/save_streamflow_to_results

        # Log baseline performance metrics after postprocessing
        self.log_baseline_performance()

    def log_baseline_performance(self):
        """
        Log baseline model performance metrics (KGE, NSE) after model runs.

        This diagnostic helps users understand initial model performance before
        calibration and identify potential issues with model setup.
        """
        try:
            import numpy as np
            from symfluence.evaluation.metrics import kge, nse, kge_prime

            # Get results file path
            results_file = self.project_dir / "results" / f"{self.experiment_id}_results.csv"

            if not results_file.exists():
                self.logger.debug("Results file not found - skipping baseline performance logging")
                return

            # Load results
            results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)

            # Find observation column in results, or load from observations directory
            obs_col = None
            obs_series = None
            for col in results_df.columns:
                if 'obs' in col.lower() or 'observed' in col.lower():
                    obs_col = col
                    break

            if obs_col is None:
                # Try to load observations from standard location
                obs_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
                domain_name = self.config.domain.name
                obs_files = list(obs_dir.glob(f"{domain_name}*_streamflow*.csv")) if obs_dir.exists() else []

                if obs_files:
                    try:
                        obs_df = pd.read_csv(obs_files[0])
                        # Find datetime and discharge columns
                        datetime_col = None
                        discharge_col = None
                        for col in obs_df.columns:
                            if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                                datetime_col = col
                            if 'discharge' in col.lower() or 'flow' in col.lower() or col.lower() == 'q':
                                discharge_col = col

                        if datetime_col and discharge_col:
                            obs_df[datetime_col] = pd.to_datetime(obs_df[datetime_col])
                            obs_series = obs_df.set_index(datetime_col)[discharge_col]
                            obs_series = obs_series.resample('D').mean()  # Resample to daily
                            self.logger.debug(f"Loaded observations from {obs_files[0].name}")
                    except Exception as e:
                        self.logger.debug(f"Could not load observations: {e}")

                if obs_series is None:
                    self.logger.debug("No observation data found - skipping baseline metrics")
                    return

            # Find simulation columns (model outputs)
            sim_cols = [c for c in results_df.columns if 'discharge' in c.lower()]

            if not sim_cols:
                self.logger.debug("No simulation columns found in results")
                return

            # Log header
            self.logger.info("=" * 60)
            self.logger.info("BASELINE MODEL PERFORMANCE (before calibration)")
            self.logger.info("=" * 60)

            for sim_col in sim_cols:
                sim_series = results_df[sim_col]

                # Get observations - either from results file column or externally loaded
                if obs_col is not None:
                    obs_aligned = results_df[obs_col]
                    sim_aligned = sim_series
                elif obs_series is not None:
                    # Align observations with simulation by index (datetime)
                    common_idx = sim_series.index.intersection(obs_series.index)
                    if len(common_idx) == 0:
                        self.logger.warning(f"  {sim_col}: No overlapping dates with observations")
                        continue
                    obs_aligned = obs_series.loc[common_idx]
                    sim_aligned = sim_series.loc[common_idx]
                else:
                    continue

                obs = obs_aligned.values
                sim = sim_aligned.values

                # Remove NaN pairs
                valid_mask = ~(np.isnan(obs) | np.isnan(sim))
                obs_clean = obs[valid_mask]
                sim_clean = sim[valid_mask]

                if len(obs_clean) < 10:
                    self.logger.warning(f"  {sim_col}: Insufficient valid data ({len(obs_clean)} points)")
                    continue

                # Calculate metrics
                kge_val = kge(obs_clean, sim_clean, transfo=1)
                kgep_val = kge_prime(obs_clean, sim_clean, transfo=1)
                nse_val = nse(obs_clean, sim_clean, transfo=1)

                # Calculate bias
                mean_obs = np.mean(obs_clean)
                mean_sim = np.mean(sim_clean)
                bias_pct = ((mean_sim - mean_obs) / mean_obs) * 100 if mean_obs != 0 else np.nan

                # Determine model name from column
                model_name = sim_col.replace('_discharge_cms', '').replace('_discharge', '')

                # Log metrics
                self.logger.info(f"  {model_name}:")
                self.logger.info(f"    KGE  = {kge_val:.4f}")
                self.logger.info(f"    KGE' = {kgep_val:.4f}")
                self.logger.info(f"    NSE  = {nse_val:.4f}")
                self.logger.info(f"    Bias = {bias_pct:+.1f}%")
                self.logger.info(f"    Valid data points: {len(obs_clean)}")

                # Provide interpretation
                if kge_val < 0:
                    self.logger.warning(f"    Note: KGE < 0 indicates model performs worse than mean observed flow")
                elif kge_val < 0.5:
                    self.logger.info(f"    Note: KGE < 0.5 suggests calibration may significantly improve results")
                elif kge_val >= 0.7:
                    self.logger.info(f"    Note: KGE >= 0.7 indicates reasonable baseline performance")

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.debug(f"Could not calculate baseline metrics: {e}")

        

        

    def visualize_outputs(self):
        """Visualize model outputs using registered model visualizers."""
        self.logger.info('Starting model output visualisation')

        if not self.reporting_manager:
            self.logger.info("Visualization disabled or reporting manager not available.")
            return

        workflow = self._resolve_model_workflow()
        # Primary models from configuration
        models_str = self.config.model.hydrological_model or ''
        models = [m.strip() for m in str(models_str).split(',') if m.strip()]

        for model in models:
            visualizer = ModelRegistry.get_visualizer(model)
            if visualizer:
                try:
                    self.logger.info(f"Using registered visualizer for {model}")
                    visualizer(
                        self.reporting_manager,
                        self.config_dict,  # Visualizer expects flat dict
                        self.project_dir,
                        self.experiment_id,
                        workflow
                    )
                except Exception as e:
                    self.logger.error(f"Error during {model} visualization: {str(e)}")
            else:
                self.logger.info(f"Visualization for {model} not yet implemented or registered")

        
        
    