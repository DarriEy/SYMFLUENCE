"""
NextGen (NGEN) Worker

Worker implementation for NextGen model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from .utilities.streamflow_metrics import StreamflowMetrics


logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('NGEN')
class NgenWorker(BaseWorker):
    """
    Worker for NextGen (ngen) model calibration.

    Handles parameter application to JSON config files, ngen execution,
    and metric calculation for streamflow calibration.
    """

    # Shared streamflow metrics utility
    _streamflow_metrics = StreamflowMetrics()

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ngen worker.

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
        Apply parameters to ngen BMI text or JSON configuration files.

        Parameters use MODULE.param naming convention (e.g., CFE.Kn).

        Args:
            params: Parameter values to apply (MODULE.param format)
            settings_dir: Ngen settings directory (isolated for parallel workers)
            **kwargs: Additional arguments including 'config'

        Returns:
            True if successful
        """
        try:
            ngen_setup_dir = Path(settings_dir) if not isinstance(settings_dir, Path) else settings_dir

            self.logger.debug(f"Applying parameters to {ngen_setup_dir}")

            # Group parameters by module
            module_params: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                if '.' in param_name:
                    module, param = param_name.split('.', 1)
                    if module not in module_params:
                        module_params[module] = {}
                    module_params[module][param] = value
                else:
                    self.logger.warning(f"Parameter {param_name} missing module prefix")

            # Update each module's config file
            for module, module_param_dict in module_params.items():
                module_upper = module.upper()
                module_dir = ngen_setup_dir / module_upper

                if module_upper == 'CFE':
                    self._update_cfe_config(module_dir, module_param_dict)
                elif module_upper == 'NOAH':
                    self._update_noah_config(module_dir, module_param_dict)
                elif module_upper == 'PET':
                    self._update_pet_config(module_dir, module_param_dict)
                else:
                    self.logger.warning(f"Unknown module {module_upper}")

            return True

        except Exception as e:
            error_msg = f"Error applying ngen parameters: {e}"
            self.logger.error(error_msg)
            import traceback
            self._last_error = error_msg + "\n" + traceback.format_exc()
            return False

    def _update_cfe_config(self, cfe_dir: Path, params: Dict[str, float]) -> bool:
        """Update CFE configuration from BMI text files."""
        try:
            # Find BMI text config files matching pattern cat-*_bmi_config_cfe_*.txt
            candidates = list(cfe_dir.glob("cat-*_bmi_config_cfe_*.txt"))

            if not candidates:
                # Try any .txt file as fallback
                candidates = list(cfe_dir.glob("*.txt"))

            if not candidates:
                error_msg = f"CFE BMI config not found in {cfe_dir}"
                self.logger.error(error_msg)
                self._last_error = error_msg
                return False

            if len(candidates) > 1:
                self.logger.warning(f"Multiple CFE BMI files found in {cfe_dir}, using first: {candidates[0]}")

            config_file = candidates[0]
            self.logger.debug(f"Updating CFE config: {config_file}")

            # Read BMI text config
            lines = config_file.read_text().splitlines()

            # Parameter key mapping for BMI text format
            keymap = {
                # Soil parameters
                "bb": "soil_params.b",
                "satdk": "soil_params.satdk",
                "slop": "soil_params.slop",
                "maxsmc": "soil_params.smcmax",
                "smcmax": "soil_params.smcmax",
                "wltsmc": "soil_params.wltsmc",
                "satpsi": "soil_params.satpsi",
                "expon": "soil_params.expon",
                # Groundwater parameters
                "Cgw": "Cgw",
                "max_gw_storage": "max_gw_storage",
                # Routing parameters
                "K_nash": "K_nash",
                "K_lf": "K_lf",
                "Kn": "K_nash",
                "Klf": "K_lf",
                # Other CFE parameters
                "alpha_fc": "alpha_fc",
                "refkdt": "refkdt",
            }

            # Determine num_timesteps from config
            import pandas as pd
            start_time = self.config.get('EXPERIMENT_TIME_START')
            end_time = self.config.get('EXPERIMENT_TIME_END')
            if start_time and end_time:
                try:
                    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
                    num_steps = int(duration.total_seconds() / 3600)
                except:
                    num_steps = 1
            else:
                num_steps = 1

            updated_count = 0
            for i, line in enumerate(lines):
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, rhs = line.split("=", 1)
                k = k.strip()
                rhs_keep = rhs.strip()

                # Enforce working settings (no spaces around = for CFE)
                if k == "num_timesteps":
                    lines[i] = f"num_timesteps={num_steps}"
                    updated_count += 1
                    continue
                if k == "surface_water_partitioning_scheme":
                    lines[i] = "surface_water_partitioning_scheme=Schaake"
                    updated_count += 1
                    continue
                if k == "surface_runoff_scheme":
                    lines[i] = "surface_runoff_scheme=GIUH"
                    updated_count += 1
                    continue
                if k == "giuh_ordinates":
                    lines[i] = "giuh_ordinates=0.06,0.51,0.28,0.12,0.03"
                    updated_count += 1
                    continue
                if k == "nash_storage":
                    lines[i] = "nash_storage=0.0,0.0"
                    updated_count += 1
                    continue

                for param_name, config_key in keymap.items():
                    if param_name in params and k == config_key:
                        # Extract units if present (format: value[unit])
                        parts = rhs_keep.split('[')
                        units = f"[{parts[1]}" if len(parts) > 1 else ""
                        lines[i] = f"{config_key}={params[param_name]}{units}"
                        updated_count += 1
                        self.logger.debug(f"Updated CFE.{param_name} = {params[param_name]}")

            # Write updated config
            if updated_count > 0:
                config_file.write_text('\n'.join(lines) + '\n')
                self.logger.debug(f"Updated {updated_count} CFE parameters")

            return True

        except Exception as e:
            self.logger.error(f"Error updating CFE config: {e}")
            return False

    def _update_noah_config(self, noah_dir: Path, params: Dict[str, float]) -> bool:
        """Update NOAH configuration."""
        try:
            # NOAH config files: cat-*_*.input or noah_config.json
            json_config = noah_dir / 'noah_config.json'

            if json_config.exists():
                with open(json_config, 'r') as f:
                    cfg = json.load(f)
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
                        self.logger.debug(f"Updated NOAH.{k} = {v}")
                with open(json_config, 'w') as f:
                    json.dump(cfg, f, indent=2)
                return True

            # Fallback: update BMI text files
            candidates = list(noah_dir.glob("cat-*.input"))
            if candidates:
                config_file = candidates[0]
                lines = config_file.read_text().splitlines()

                for i, line in enumerate(lines):
                    for param_name, value in params.items():
                        pattern = rf"^{re.escape(param_name)}\s*="
                        if re.match(pattern, line.strip()):
                            parts = line.split('[')
                            units = f" [{parts[1]}" if len(parts) > 1 else ""
                            lines[i] = f"{param_name} = {value}{units}"
                            self.logger.debug(f"Updated NOAH.{param_name} = {value}")

                config_file.write_text('\n'.join(lines) + '\n')
                return True

            self.logger.warning(f"No NOAH config found in {noah_dir}")
            return True  # Don't fail if NOAH not being used

        except Exception as e:
            self.logger.error(f"Error updating NOAH config: {e}")
            return False

    def _update_pet_config(self, pet_dir: Path, params: Dict[str, float]) -> bool:
        """Update PET configuration."""
        try:
            json_config = pet_dir / 'pet_config.json'

            if json_config.exists():
                with open(json_config, 'r') as f:
                    cfg = json.load(f)
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
                        self.logger.debug(f"Updated PET.{k} = {v}")
                with open(json_config, 'w') as f:
                    json.dump(cfg, f, indent=2)
                return True

            # Fallback: update BMI text files
            candidates = list(pet_dir.glob("cat-*_pet_config.txt"))
            if candidates:
                config_file = candidates[0]
                lines = config_file.read_text().splitlines()

                # Determine num_timesteps from config
                import pandas as pd
                start_time = self.config.get('EXPERIMENT_TIME_START')
                end_time = self.config.get('EXPERIMENT_TIME_END')
                if start_time and end_time:
                    try:
                        duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
                        num_steps = int(duration.total_seconds() / 3600)
                    except:
                        num_steps = 1
                else:
                    num_steps = 1

                for i, line in enumerate(lines):
                    if "=" not in line or line.strip().startswith("#"):
                        continue
                    k, rhs = line.split("=", 1)
                    k = k.strip()
                    rhs_keep = rhs.strip()

                    # PET configs also should not have spaces around =
                    if k == "num_timesteps":
                        lines[i] = f"num_timesteps={num_steps}"
                        continue

                    for param_name, value in params.items():
                        if k == param_name:
                            lines[i] = f"{param_name}={value}"
                            self.logger.debug(f"Updated PET.{param_name} = {value}")

                config_file.write_text('\n'.join(lines) + '\n')
                return True

            self.logger.warning(f"No PET config found in {pet_dir}")
            return True  # Don't fail if PET not being used

        except Exception as e:
            self.logger.error(f"Error updating PET config: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run ngen model.

        Supports both serial and parallel execution modes.

        Args:
            config: Configuration dictionary
            settings_dir: Ngen settings directory
            output_dir: Output directory
            **kwargs: Additional arguments including parallel config keys

        Returns:
            True if model ran successfully
        """
        try:
            # Use a dictionary for local modifications to avoid SymfluenceConfig immutability/subscriptability issues
            if hasattr(config, 'to_dict'):
                parallel_config = config.to_dict(flatten=True)
            else:
                parallel_config = config.copy()

            # Ensure runner uses isolated directories
            parallel_config['_ngen_output_dir'] = str(output_dir)
            parallel_config['_ngen_settings_dir'] = str(settings_dir)
            
            # Ensure NGEN_INSTALL_PATH is present
            if 'NGEN_INSTALL_PATH' not in parallel_config:
                parallel_config['NGEN_INSTALL_PATH'] = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/installs/ngen/cmake_build"

            # Import NgenRunner
            from symfluence.models.ngen import NgenRunner

            experiment_id = parallel_config.get('EXPERIMENT_ID')

            # Initialize and run
            runner = NgenRunner(parallel_config, self.logger)
            
            # Pass GeoJSON fallback preference if detected
            if hasattr(self, '_use_geojson_catchments'):
                runner._use_geojson_catchments = self._use_geojson_catchments
                
            success = runner.run_ngen(experiment_id)

            return success

        except FileNotFoundError as e:
            error_msg = f"Required ngen input file not found: {e}"
            self.logger.error(error_msg)
            self._last_error = error_msg
            return False
        except Exception as e:
            error_msg = f"Error running ngen: {e}"
            self.logger.error(error_msg)
            import traceback
            self._last_error = error_msg + "\n" + traceback.format_exc()
            return False

    def _patch_realization_config(self, settings_dir: Path, output_dir: Path) -> bool:
        """
        DEPRECATED: Now handled by NgenRunner.
        """
        return True

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from ngen output.

        Args:
            output_dir: Directory containing model outputs (isolated)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Try to use calibration target
            from ..calibration_targets import NgenStreamflowTarget

            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            self.logger.debug(f"Calculating metrics from {output_dir}")
            self.logger.debug(f"Project dir: {project_dir}, Domain: {domain_name}")

            # Create calibration target
            target = NgenStreamflowTarget(config, project_dir, self.logger)

            # Calculate metrics using isolated output_dir
            # NgenStreamflowTarget needs to be aware of the isolated directory
            metrics = target.calculate_metrics(experiment_id=experiment_id, output_dir=output_dir)

            if metrics is None:
                self.logger.warning(f"Metrics calculation returned None for {output_dir}")
                return {'kge': self.penalty_score}

            self.logger.debug(f"Calculated metrics: {metrics}")

            # Normalize metric keys to lowercase
            result = {k.lower(): float(v) for k, v in metrics.items()}
            self.logger.debug(f"Normalized metrics: {result}")
            return result

        except ImportError as e:
            # Fallback: Calculate metrics directly
            self.logger.debug(f"Calibration target import failed, falling back to direct calculation: {e}")
            return self._calculate_metrics_direct(output_dir, config)

        except Exception as e:
            error_msg = f"Error calculating ngen metrics: {e}"
            self.logger.error(error_msg)
            import traceback
            self._last_error = error_msg + "\n" + traceback.format_exc()
            return {'kge': self.penalty_score}

    def _calculate_metrics_direct(
        self,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate metrics directly from ngen output files.

        Args:
            output_dir: Output directory (isolated)
            config: Configuration dictionary

        Returns:
            Dictionary of metrics
        """
        try:
            import pandas as pd
            from symfluence.evaluation.metrics import kge, nse

            domain_name = config.get('DOMAIN_NAME')
            
            # Find ngen output in isolated output_dir
            output_files = list(output_dir.glob('*.csv')) + list(output_dir.glob('*.nc'))

            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files found'}

            # Read simulation
            if output_files[0].suffix == '.csv':
                sim_df = pd.read_csv(output_files[0], index_col=0, parse_dates=True)
                if 'q_cms' in sim_df.columns:
                    sim = sim_df['q_cms'].values
                else:
                    sim = sim_df.iloc[:, 0].values
            else:
                import xarray as xr
                with xr.open_dataset(output_files[0]) as ds:
                    # Generic extraction - pick first data variable
                    var = next(iter(ds.data_vars))
                    sim = ds[var].values.flatten()

            # Load observations using shared utility
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq=None
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            # Simple alignment using shared utility
            obs_series = pd.Series(obs_values, index=obs_index)
            # Create simulation series (use same index length as simple fallback)
            sim_series = pd.Series(sim, index=obs_index[:len(sim)] if len(sim) <= len(obs_index) else obs_index)

            try:
                obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(sim_series, obs_series)
                return self._streamflow_metrics.calculate_metrics(obs_aligned, sim_aligned, metrics=['kge', 'nse'])
            except ValueError as e:
                # Fallback to simple length-based alignment
                min_len = min(len(sim), len(obs_values))
                return self._streamflow_metrics.calculate_metrics(
                    obs_values[:min_len], sim[:min_len], metrics=['kge', 'nse']
                )

        except Exception as e:
            self.logger.error(f"Error in direct ngen metrics calculation: {e}")
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_ngen_parameters_worker(task_data)


def _evaluate_ngen_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = NgenWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
