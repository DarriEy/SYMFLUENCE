"""
Wrapper around gui.data.ResultsLoader for TUI calibration views.

Provides experiment listing, optimization history, best parameters,
and metric calculation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CalibrationDataService:
    """Load calibration and results data for a project directory."""

    def __init__(self, project_dir: str):
        self._project_dir = Path(project_dir)
        self._loader = None

    @property
    def loader(self):
        """Lazily create the ResultsLoader."""
        if self._loader is None:
            try:
                from symfluence.gui.data import ResultsLoader
                self._loader = ResultsLoader(self._project_dir)
            except ImportError:
                logger.warning("ResultsLoader not available (gui package not installed)")
                return None
        return self._loader

    def list_experiments(self) -> List[str]:
        """Return list of experiment IDs found in optimization directory."""
        if self.loader is None:
            return self._list_experiments_fallback()
        return self.loader.list_experiments()

    def load_optimization_history(self, experiment_id: str) -> Optional[Any]:
        """Load iteration results DataFrame for an experiment.

        Returns pd.DataFrame or None.
        """
        if self.loader is None:
            return None
        return self.loader.load_optimization_history(experiment_id)

    def load_best_params(self, experiment_id: str) -> Optional[Dict]:
        """Load best calibration parameters dict for an experiment."""
        if self.loader is None:
            return None
        return self.loader.load_best_params(experiment_id)

    def load_observed(self) -> Optional[Any]:
        """Load observed streamflow as pd.Series."""
        if self.loader is None:
            return None
        return self.loader.load_observed_streamflow()

    def load_simulated(self, experiment_id: str) -> Optional[Any]:
        """Load simulated streamflow as pd.Series."""
        if self.loader is None:
            return None
        return self.loader.load_simulated_streamflow(experiment_id)

    def calculate_metrics(self, experiment_id: str) -> Dict[str, float]:
        """Calculate performance metrics for an experiment.

        Returns dict of metric_name -> value, or empty dict.
        """
        obs = self.load_observed()
        sim = self.load_simulated(experiment_id)
        if obs is None or sim is None:
            return {}

        if self.loader is not None:
            return self.loader.calculate_metrics(obs, sim)
        return {}

    def load_benchmark_scores(self) -> Optional[Any]:
        """Load benchmark evaluation scores DataFrame."""
        if self.loader is None:
            return None
        return self.loader.load_benchmark_scores()

    def clear_cache(self) -> None:
        """Clear the ResultsLoader cache."""
        if self._loader is not None:
            self._loader.clear_cache()

    def _list_experiments_fallback(self) -> List[str]:
        """Fallback experiment listing without ResultsLoader."""
        opt_dir = self._project_dir / "optimization"
        if not opt_dir.is_dir():
            return []
        ids = []
        for f in opt_dir.glob("*_parallel_iteration_results.csv"):
            name = f.stem.replace("_parallel_iteration_results", "")
            if name:
                ids.append(name)
        return sorted(set(ids))
