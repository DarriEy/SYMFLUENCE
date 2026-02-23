"""
MESH Worker

Worker implementation for MESH model optimization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.evaluation.metrics import kge, nse
from symfluence.models.mesh.runner import MESHRunner
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask


@OptimizerRegistry.register_worker('MESH')
class MESHWorker(BaseWorker):
    """
    Worker for MESH model calibration.

    Handles parameter application, MESH execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize MESH worker."""
        super().__init__(config, logger)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to MESH configuration files.

        Args:
            params: Parameter values to apply
            settings_dir: MESH settings directory
            **kwargs: Additional arguments (including proc_forcing_dir where MESH runs)

        Returns:
            True if successful
        """
        try:
            config = kwargs.get('config', self.config)

            # Use MESHParameterManager from registry
            from symfluence.optimization.registry import OptimizerRegistry
            param_manager_cls = OptimizerRegistry.get_parameter_manager('MESH')

            if param_manager_cls is None:
                self.logger.error("MESHParameterManager not found in registry")
                return False

            # MESH runs from proc_forcing_dir (where CLASS.ini etc. are located)
            # Not from settings_dir. Use proc_forcing_dir if available.
            proc_forcing_dir = kwargs.get('proc_forcing_dir')
            if proc_forcing_dir:
                param_dir = Path(proc_forcing_dir)
                self.logger.debug(f"Using proc_forcing_dir for MESH params: {param_dir}")
            else:
                param_dir = settings_dir
                self.logger.debug(f"Using settings_dir for MESH params (no proc_forcing_dir): {param_dir}")

            param_manager = param_manager_cls(config, self.logger, param_dir)

            success = param_manager.update_model_files(params)

            if not success:
                self.logger.error(f"Failed to update MESH parameter files in {param_dir}")
            else:
                self.logger.debug(
                    f"Applied {len(params)} MESH parameters to {param_dir}: "
                    f"{', '.join(f'{k}={v:.4g}' for k, v in params.items())}"
                )

            return success

        except (FileNotFoundError, OSError) as e:
            self.logger.error(f"File error applying MESH parameters: {e}")
            return False
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error applying MESH parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run MESH model.

        Args:
            config: Configuration dictionary
            settings_dir: MESH settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:
            # Initialize MESH runner
            runner = MESHRunner(config, self.logger)

            # Determine where to run from (isolated or global)
            proc_forcing_dir = kwargs.get('proc_forcing_dir')

            if proc_forcing_dir:
                proc_forcing_path = Path(proc_forcing_dir)
                self.logger.debug(f"Running MESH worker in isolated dir: {proc_forcing_path}")
                runner.set_process_directories(proc_forcing_path, output_dir)
            elif settings_dir and (settings_dir / 'MESH_input_run_options.ini').exists():
                self.logger.debug(f"Running MESH worker using settings_dir: {settings_dir}")
                runner.set_process_directories(settings_dir, output_dir)
            else:
                # Fallback to standard paths
                # Handle both flat (DOMAIN_NAME) and nested (domain.name) config formats
                domain_name = config.get('DOMAIN_NAME')
                if domain_name is None and 'domain' in config:
                    domain_name = config['domain'].get('name')

                data_dir = config.get('SYMFLUENCE_DATA_DIR')
                if data_dir is None and 'system' in config:
                    data_dir = config['system'].get('data_dir')

                if data_dir is None:
                    raise ValueError("SYMFLUENCE_DATA_DIR or system.data_dir is required")

                data_dir = Path(data_dir)
                project_dir = data_dir / f"domain_{domain_name}"
                runner.forcing_mesh_path = resolve_data_subdir(project_dir, 'forcing') / 'MESH_input'
                runner.output_dir = output_dir

            # Run MESH
            result_path = runner.run_mesh()

            return result_path is not None

        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for MESH: {e}")
            return False
        except (OSError, IOError) as e:
            self.logger.error(f"I/O error running MESH: {e}")
            return False
        except (RuntimeError, ValueError) as e:
            # Model execution or configuration errors
            self.logger.error(f"Error running MESH: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from MESH output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        from datetime import datetime, timedelta

        from symfluence.models.mesh.extractor import MESHResultExtractor

        try:
            # MESH output file for streamflow or runoff.
            # Basin_average is preferred for lumped domains (uses RFF + LKG,
            # avoids self-referential routing artifacts in single-cell run_def).
            sim_file_candidates = [
                # Primary: Basin average water balance (daily, RFF + LKG)
                output_dir / 'Basin_average_water_balance.csv',
                output_dir / 'results' / 'Basin_average_water_balance.csv',
                # Fallback: GRU water balance (hourly, contains ROF)
                output_dir / 'GRU_water_balance.csv',
                # Routed output (multi-cell domains or verification)
                output_dir / 'MESH_output_streamflow.csv',
                output_dir / 'results' / 'MESH_output_streamflow.csv',
                output_dir / 'streamflow.csv',
            ]

            sim_file = None
            for candidate in sim_file_candidates:
                if candidate.exists():
                    # Check if file has data (not just headers)
                    try:
                        with open(candidate, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > 1:  # More than just header
                                sim_file = candidate
                                break
                    except (IOError, UnicodeDecodeError):
                        continue

            if sim_file is None:
                self.logger.error(f"MESH output not found in {output_dir}")
                return {'kge': self.penalty_score, 'error': 'MESH output not found'}

            self.logger.debug(f"Using MESH output file: {sim_file}")

            # Use extractor for water balance files (lumped mode)
            is_basin_wb = 'Basin_average_water_balance' in sim_file.name
            is_gru_wb = 'GRU_water_balance' in sim_file.name
            if is_basin_wb or is_gru_wb:
                extractor = MESHResultExtractor('MESH')

                # Get simulation start date from config or run options
                start_date = kwargs.get('start_date')
                if start_date is None:
                    # Try to parse from run options if available
                    run_opts = output_dir / 'MESH_input_run_options.ini'
                    if run_opts.exists():
                        start_date = self._parse_start_date_from_run_options(run_opts)
                    else:
                        start_date = datetime(2001, 1, 1)

                # Extract daily runoff using extractor
                sim_series = extractor.extract_variable(
                    sim_file,
                    'runoff',
                    start_date=start_date,
                    aggregate='daily'
                )

                # Get basin area for unit conversion (mm/day -> m³/s)
                basin_area_m2 = kwargs.get('basin_area_m2')
                if basin_area_m2 is None:
                    # Try to get from drainage database
                    basin_area_m2 = self._get_basin_area(output_dir, config)

                if basin_area_m2 is not None and basin_area_m2 > 0:
                    # Convert: Q (m³/s) = ROF (mm/day) × Area (m²) × 0.001 / 86400
                    conversion_factor = basin_area_m2 * 0.001 / 86400
                    sim_series = sim_series * conversion_factor
                    self.logger.debug(
                        f"Converted runoff to discharge: area={basin_area_m2/1e6:.1f} km², "
                        f"factor={conversion_factor:.4f}"
                    )

                sim_df = sim_series.to_frame(name='runoff')

            elif 'streamflow' in sim_file.name.lower():
                # Read standard streamflow file
                sim_df = pd.read_csv(sim_file, skipinitialspace=True)
                sim_df.columns = sim_df.columns.str.strip()

                # Check for QOSIM columns (routed streamflow)
                qosim_cols = [c for c in sim_df.columns if c.startswith('QOSIM')]
                if qosim_cols:
                    # Convert YEAR and JDAY/DAY to datetime
                    day_col = 'JDAY' if 'JDAY' in sim_df.columns else 'DAY'
                    sim_df['time'] = sim_df.apply(
                        lambda row: datetime(int(row['YEAR']), 1, 1) +
                                   timedelta(days=int(row[day_col]) - 1),
                        axis=1
                    )
                    sim_df = sim_df.set_index('time')
                    sim_df = sim_df.rename(columns={qosim_cols[0]: 'runoff'})
                else:
                    # Try to parse with generic time column
                    sim_df = pd.read_csv(sim_file, parse_dates=['time'])
                    sim_df = sim_df.set_index('time')
                    for col in ['streamflow', 'discharge', 'flow']:
                        if col in sim_df.columns:
                            sim_df = sim_df.rename(columns={col: 'runoff'})
                            break
            else:
                self.logger.error(f"Unknown MESH output format: {sim_file}")
                return {'kge': self.penalty_score, 'error': 'Unknown output format'}

            if 'runoff' not in sim_df.columns:
                self.logger.error(f"Runoff column not found in {sim_file}")
                return {'kge': self.penalty_score, 'error': 'Runoff column not found'}

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            if domain_name is None and 'domain' in config:
                domain_name = config['domain'].get('name')

            data_dir = config.get('SYMFLUENCE_DATA_DIR', '.')
            if data_dir == '.' and 'system' in config:
                data_dir = config['system'].get('data_dir', '.')

            data_dir = Path(data_dir)
            project_dir = data_dir / f'domain_{domain_name}'
            obs_dir = resolve_data_subdir(project_dir, 'observations')
            obs_file = (obs_dir / 'streamflow' / 'preprocessed' /
                       f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                self.logger.error(f"Observations not found: {obs_file}")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Get observation column (usually 'discharge_cms' or 'discharge')
            obs_col = None
            for col in ['discharge_cms', 'discharge', 'flow', 'streamflow']:
                if col in obs_df.columns:
                    obs_col = col
                    break
            if obs_col is None:
                obs_col = obs_df.columns[0]

            # Ensure indices are DatetimeIndex
            sim_df.index = pd.DatetimeIndex(sim_df.index)
            obs_df.index = pd.DatetimeIndex(obs_df.index)

            # MESH water balance outputs are daily (Basin_average) or hourly
            # (GRU, aggregated to daily by the extractor). Observations may be
            # sub-daily (e.g., hourly). Detect and resample so each daily MESH
            # value is compared against the daily-mean observation, not just a
            # single sub-daily timestamp.
            if is_basin_wb or is_gru_wb:
                # Check if obs are sub-daily (more records than unique days)
                n_obs_days = len(obs_df.index.normalize().unique())
                if len(obs_df) > n_obs_days * 1.5:
                    obs_daily = obs_df[obs_col].resample('D').mean()
                    self.logger.debug(
                        f"Aggregated {len(obs_df)} sub-daily obs to "
                        f"{len(obs_daily)} daily values"
                    )
                else:
                    obs_daily = obs_df[obs_col]
            else:
                obs_daily = obs_df[obs_col]

            # Drop warm-up from simulation before alignment.
            # Prefer the actual spinup written to run options by the parameter
            # fixer (METRICSSPINUP), which accounts for forcing-data limits.
            # Fall back to config value only when run options are unavailable.
            run_opts = output_dir / 'MESH_input_run_options.ini'
            actual_spinup = 0
            if run_opts.exists():
                actual_spinup = self._parse_spinup_from_run_options(run_opts)
            warmup_days = actual_spinup if actual_spinup > 0 else int(
                config.get('MESH_SPINUP_DAYS', 365)
            )
            if not sim_df.empty:
                sim_start = sim_df.index.min()
                warmup_cutoff = sim_start + pd.Timedelta(days=warmup_days)
                sim_warm = sim_df.loc[sim_df.index >= warmup_cutoff]
                if sim_warm.empty:
                    self.logger.warning(
                        "Warm-up trimming removed all simulation data; "
                        "model likely crashed before completing spinup. "
                        f"Sim range: {sim_start.date()} to {sim_df.index.max().date()}, "
                        f"warmup cutoff: {warmup_cutoff.date()}"
                    )
                    return {'kge': self.penalty_score, 'error': 'Simulation shorter than spinup'}
                else:
                    sim_df = sim_warm
                    self.logger.debug(
                        f"Trimmed {warmup_days} warm-up days "
                        f"(cutoff={warmup_cutoff.date()}, "
                        f"remaining={len(sim_df)} rows)"
                    )

            # Filter to requested period.  The caller can pass an explicit
            # ``period`` kwarg (e.g. the evaluation period); otherwise fall
            # back to CALIBRATION_PERIOD from the config dict.
            period = kwargs.get('period', '')
            if not period:
                period = config.get('CALIBRATION_PERIOD', '')
            if period and ',' in str(period):
                try:
                    start_str, end_str = [s.strip() for s in str(period).split(',')]
                    period_start = pd.Timestamp(start_str)
                    period_end = pd.Timestamp(end_str)
                    sim_df = sim_df.loc[
                        (sim_df.index >= period_start) & (sim_df.index <= period_end)
                    ]
                    obs_daily = obs_daily.loc[
                        (obs_daily.index >= period_start) & (obs_daily.index <= period_end)
                    ]
                    self.logger.debug(
                        f"Applied period filter: {period_start.date()} to {period_end.date()}, "
                        f"sim={len(sim_df)} obs={len(obs_daily)} rows"
                    )
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not parse period '{period}': {e}")

            # Align simulation and observations
            common_idx = sim_df.index.intersection(obs_daily.index)
            if len(common_idx) == 0:
                self.logger.error("No common dates between simulation and observations")
                if not sim_df.empty:
                    self.logger.debug(f"Sim dates: {sim_df.index.min()} to {sim_df.index.max()}")
                if not obs_daily.empty:
                    self.logger.debug(f"Obs dates: {obs_daily.index.min()} to {obs_daily.index.max()}")
                return {'kge': self.penalty_score, 'error': 'No common dates'}

            common_idx = common_idx.sort_values()

            obs_aligned = obs_daily.loc[common_idx].values
            sim_aligned = sim_df.loc[common_idx, 'runoff'].values

            # Calculate metrics with KGE decomposition for diagnostics
            kge_result = kge(obs_aligned, sim_aligned, transfo=1, return_components=True)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            # Log-transformed KGE for composite objective support
            kge_log_val = float('nan')
            try:
                epsilon = 0.01 * np.mean(obs_aligned[obs_aligned > 0]) if np.any(obs_aligned > 0) else 0.01
                obs_log = np.log(obs_aligned + epsilon)
                sim_log = np.log(np.maximum(sim_aligned, 0) + epsilon)
                valid_log = np.isfinite(obs_log) & np.isfinite(sim_log)
                if np.sum(valid_log) > 10:
                    kge_log_val = float(kge(obs_log[valid_log], sim_log[valid_log], transfo=1))
            except Exception:
                pass

            self.logger.debug(
                f"MESH metrics (n={len(obs_aligned)}): "
                f"KGE={kge_result['KGE']:.4f} "  # type: ignore[index]
                f"(r={kge_result['r']:.3f}, "  # type: ignore[index]
                f"alpha={kge_result['alpha']:.3f}, "  # type: ignore[index]
                f"beta={kge_result['beta']:.3f}), "  # type: ignore[index]
                f"NSE={nse_val:.4f}, KGE_log={kge_log_val:.4f}"
            )

            return {
                'kge': float(kge_result['KGE']),  # type: ignore[index]
                'nse': float(nse_val),
                'r': float(kge_result['r']),  # type: ignore[index]
                'alpha': float(kge_result['alpha']),  # type: ignore[index]
                'beta': float(kge_result['beta']),  # type: ignore[index]
                'kge_log': kge_log_val,
            }

        except FileNotFoundError as e:
            self.logger.error(f"Output or observation file not found: {e}")
            return {'kge': self.penalty_score}
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error calculating MESH metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score}
        except (OSError, pd.errors.ParserError) as e:
            self.logger.error(f"Error calculating MESH metrics: {e}")
            return {'kge': self.penalty_score}

    def _parse_start_date_from_run_options(self, run_opts_path: Path):
        """Parse simulation start date from MESH run options file."""
        from datetime import datetime

        try:
            with open(run_opts_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for simulation start line (format: YYYY JJJ HH MM)
            import re
            # The start date line typically looks like: 2001 001   1   0
            match = re.search(r'(\d{4})\s+(\d{1,3})\s+\d+\s+\d+\s*$', content, re.MULTILINE)
            if match:
                year = int(match.group(1))
                jday = int(match.group(2))
                from datetime import timedelta
                return datetime(year, 1, 1) + timedelta(days=jday - 1)
        except (IOError, ValueError):
            pass

        return datetime(2001, 1, 1)  # Default

    def _parse_spinup_from_run_options(self, run_opts_path: Path) -> int:
        """Parse actual METRICSSPINUP days from MESH run options file.

        The parameter fixer writes the actual (possibly limited) spinup days
        to the run options as METRICSSPINUP. This is more reliable than
        the config value which may reflect the originally requested spinup
        before it was clamped to available forcing data.

        Returns:
            Actual spinup days from run options, or 0 if not found.
        """
        import re

        try:
            with open(run_opts_path, 'r', encoding='utf-8') as f:
                content = f.read()

            match = re.search(r'METRICSSPINUP\s+(\d+)', content)
            if match:
                return int(match.group(1))
        except (IOError, ValueError):
            pass

        return 0

    def _get_basin_area(self, output_dir: Path, config: Dict[str, Any]) -> Optional[float]:
        """Get basin area in m² from drainage database or config."""
        import xarray as xr

        # Try to find drainage database
        drainage_db_candidates = [
            output_dir / 'MESH_drainage_database.nc',
            output_dir.parent / 'MESH_drainage_database.nc',
        ]

        # Also try forcing directory from config
        domain_name = config.get('DOMAIN_NAME')
        if domain_name is None and 'domain' in config:
            domain_name = config['domain'].get('name')

        data_dir = config.get('SYMFLUENCE_DATA_DIR', '.')
        if data_dir == '.' and 'system' in config:
            data_dir = config['system'].get('data_dir', '.')

        if domain_name:
            data_dir = Path(data_dir)
            project_dir = data_dir / f'domain_{domain_name}'
            forcing_dir = resolve_data_subdir(project_dir, 'forcing')
            drainage_db_candidates.append(
                forcing_dir / 'MESH_input' / 'MESH_drainage_database.nc'
            )

        for db_path in drainage_db_candidates:
            if db_path.exists():
                try:
                    with xr.open_dataset(db_path) as ds:
                        if 'GridArea' in ds:
                            # GridArea is the actual area of each subbasin in m²
                            # DA (drainage area) can be wrong for lumped single-subbasin
                            # setups where the outlet points to itself, doubling the area
                            return float(ds['GridArea'].values.sum())
                        elif 'DA' in ds:
                            return float(ds['DA'].values.sum())
                except Exception as e:
                    self.logger.debug(f"Could not read basin area from {db_path}: {e}")

        # Fallback: try config
        basin_area = config.get('basin_area_m2')
        if basin_area is None and 'domain' in config:
            basin_area = config['domain'].get('area_m2')

        return basin_area

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_mesh_parameters_worker(task_data)


def _evaluate_mesh_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = MESHWorker()
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
