"""
Data loading layer for GUI results visualization.

ResultsLoader reads project output files (CSVs, NetCDF, JSON) into pandas
objects for display by the interactive results viewer.  All methods return
None on failure and never raise.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResultsLoader:
    """Load project outputs into pandas objects for GUI visualization.

    Args:
        project_dir: Resolved project directory path.
        config: SymfluenceConfig instance (used for domain/experiment metadata).
    """

    def __init__(self, project_dir, config=None):
        self.project_dir = Path(project_dir) if project_dir else None
        self.config = config
        self._cache = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cached(self, key):
        return self._cache.get(key)

    def _store(self, key, value):
        self._cache[key] = value
        return value

    def clear_cache(self):
        """Invalidate all cached results."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Observed streamflow
    # ------------------------------------------------------------------

    def load_observed_streamflow(self):
        """Load preprocessed observed streamflow as a pd.Series.

        Searches ``observations/streamflow/preprocessed/`` for a CSV whose
        name contains ``_streamflow_processed``.

        Returns:
            pd.Series with DatetimeIndex, or None.
        """
        key = ('observed_streamflow',)
        cached = self._cached(key)
        if cached is not None:
            return cached

        if not self.project_dir:
            return None

        search_dir = self.project_observations_dir / 'streamflow' / 'preprocessed'
        if not search_dir.is_dir():
            return None

        try:
            candidates = list(search_dir.glob('*streamflow_processed*.csv'))
            if not candidates:
                return None

            df = pd.read_csv(candidates[0], parse_dates=True, index_col=0)
            # Use first numeric column as flow values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            series = df[numeric_cols[0]].dropna()
            series.index = pd.to_datetime(series.index)
            series.name = 'observed'
            return self._store(key, series)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            logger.debug(f"Failed to load observed streamflow: {exc}")
            return None

    # ------------------------------------------------------------------
    # Simulated streamflow
    # ------------------------------------------------------------------

    def load_simulated_streamflow(self, experiment_id=None):
        """Load post-processed simulated streamflow as a pd.Series.

        Looks for ``simulations/{experiment_id}_postprocessed.nc`` or falls
        back to any ``*_postprocessed.nc`` in the simulations directory.

        Returns:
            pd.Series with DatetimeIndex, or None.
        """
        experiment_id = experiment_id or self.get_experiment_id()
        key = ('simulated_streamflow', experiment_id)
        cached = self._cached(key)
        if cached is not None:
            return cached

        if not self.project_dir:
            return None

        sim_dir = self.project_dir / 'simulations'
        if not sim_dir.is_dir():
            return None

        try:
            import xarray as xr

            # Try exact match first, then fallback to any postprocessed file
            candidates = []
            if experiment_id:
                candidates = list(sim_dir.glob(f'{experiment_id}_postprocessed.nc'))
            if not candidates:
                candidates = list(sim_dir.glob('*_postprocessed.nc'))
            if not candidates:
                # Also check for *_timestep.nc or similar common patterns
                candidates = list(sim_dir.glob('*_timestep.nc'))
            if not candidates:
                return None

            ds = xr.open_dataset(candidates[0])
            # Look for common streamflow variable names
            flow_vars = [v for v in ds.data_vars
                         if any(kw in v.lower() for kw in
                                ['flow', 'discharge', 'runoff', 'streamflow', 'q_sim', 'scalarTotalRunoff'])]
            if not flow_vars:
                # Fall back to first data variable
                flow_vars = list(ds.data_vars)

            if not flow_vars:
                ds.close()
                return None

            da = ds[flow_vars[0]]
            # Squeeze out singleton dimensions (e.g. hru)
            if da.ndim > 1:
                da = da.squeeze()
            series = da.to_series().dropna()
            series.name = 'simulated'
            ds.close()
            return self._store(key, series)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            logger.debug(f"Failed to load simulated streamflow: {exc}")
            return None

    # ------------------------------------------------------------------
    # Optimization history
    # ------------------------------------------------------------------

    def load_optimization_history(self, experiment_id=None):
        """Load calibration iteration results as a DataFrame.

        Searches ``optimization/{experiment_id}_parallel_iteration_results.csv``.

        Returns:
            pd.DataFrame with columns normalised to [iteration, score, ...params],
            or None.
        """
        experiment_id = experiment_id or self.get_experiment_id()
        key = ('optimization_history', experiment_id)
        cached = self._cached(key)
        if cached is not None:
            return cached

        if not self.project_dir:
            return None

        opt_dir = self.project_dir / 'optimization'
        if not opt_dir.is_dir():
            return None

        try:
            candidates = []
            if experiment_id:
                candidates = list(opt_dir.glob(f'{experiment_id}_parallel_iteration_results.csv'))
            if not candidates:
                candidates = list(opt_dir.glob('*_parallel_iteration_results.csv'))
            if not candidates:
                return None

            df = pd.read_csv(candidates[0])

            # Normalize column names — look for iteration/score columns
            col_lower = {c: c.lower() for c in df.columns}
            rename: dict[str, str] = {}
            for orig, low in col_lower.items():
                if 'iter' in low and 'iteration' not in rename.values():
                    rename[orig] = 'iteration'
                elif any(kw in low for kw in ['score', 'obj', 'metric', 'fitness', 'kge', 'nse']):
                    if 'score' not in rename.values():
                        rename[orig] = 'score'
            df = df.rename(columns=rename)

            if 'iteration' not in df.columns:
                df['iteration'] = range(len(df))
            df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')

            return self._store(key, df)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            logger.debug(f"Failed to load optimization history: {exc}")
            return None

    # ------------------------------------------------------------------
    # Best parameters
    # ------------------------------------------------------------------

    def load_best_params(self, experiment_id=None):
        """Load best calibration parameters from JSON.

        Returns:
            dict of parameter values, or None.
        """
        experiment_id = experiment_id or self.get_experiment_id()
        key = ('best_params', experiment_id)
        cached = self._cached(key)
        if cached is not None:
            return cached

        if not self.project_dir:
            return None

        opt_dir = self.project_dir / 'optimization'
        if not opt_dir.is_dir():
            return None

        try:
            candidates = []
            if experiment_id:
                candidates = list(opt_dir.glob(f'{experiment_id}_*_best_params.json'))
            if not candidates:
                candidates = list(opt_dir.glob('*_best_params.json'))
            if not candidates:
                return None

            with open(candidates[0], encoding='utf-8') as f:
                data = json.load(f)
            return self._store(key, data)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            logger.debug(f"Failed to load best params: {exc}")
            return None

    # ------------------------------------------------------------------
    # Benchmark scores
    # ------------------------------------------------------------------

    def load_benchmark_scores(self):
        """Load benchmark evaluation scores.

        Returns:
            pd.DataFrame or None.
        """
        key = ('benchmark_scores',)
        cached = self._cached(key)
        if cached is not None:
            return cached

        if not self.project_dir:
            return None

        path = self.project_dir / 'evaluation' / 'benchmark_scores.csv'
        if not path.is_file():
            return None

        try:
            df = pd.read_csv(path)
            return self._store(key, df)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            logger.debug(f"Failed to load benchmark scores: {exc}")
            return None

    # ------------------------------------------------------------------
    # Metrics calculation (delegates to evaluation module)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(obs, sim):
        """Calculate all standard performance metrics.

        Returns:
            dict of {metric_name: value}, or empty dict on failure.
        """
        try:
            from symfluence.evaluation.metrics import calculate_all_metrics
            return calculate_all_metrics(obs, sim)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            logger.debug(f"Metrics calculation failed: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Flow duration curve
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_fdc(data):
        """Calculate flow duration curve (exceedance probability vs flows).

        Returns:
            Tuple of (exceedance, sorted_flows), or (None, None) on failure.
        """
        try:
            from symfluence.reporting.core.plot_utils import calculate_flow_duration_curve
            values = np.asarray(data.dropna()) if hasattr(data, 'dropna') else np.asarray(data)
            exc, flows = calculate_flow_duration_curve(values)
            return exc, flows
        except Exception as exc_err:  # noqa: BLE001 — UI resilience
            logger.debug(f"FDC calculation failed: {exc_err}")
            return None, None

    # ------------------------------------------------------------------
    # Experiment listing
    # ------------------------------------------------------------------

    def list_experiments(self):
        """Scan optimization directory for experiment IDs.

        Returns:
            List of experiment ID strings, or [].
        """
        if not self.project_dir:
            return []

        opt_dir = self.project_dir / 'optimization'
        if not opt_dir.is_dir():
            return []

        try:
            ids = []
            for f in opt_dir.glob('*_parallel_iteration_results.csv'):
                # Filename pattern: {experiment_id}_parallel_iteration_results.csv
                name = f.stem.replace('_parallel_iteration_results', '')
                if name:
                    ids.append(name)
            return sorted(set(ids))
        except Exception:  # noqa: BLE001 — UI resilience
            return []

    # ------------------------------------------------------------------
    # Current experiment ID
    # ------------------------------------------------------------------

    def get_experiment_id(self):
        """Return current experiment_id from config, or None."""
        if self.config is None:
            return None
        try:
            return self.config.domain.experiment_id
        except AttributeError:
            return None
