"""
Results Tracking Mixin

Provides results persistence and tracking for optimization runs.
Handles iteration history, best solution tracking, and results file I/O.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResultsTrackingMixin:
    """
    Mixin class providing results tracking and persistence for optimizers.

    Requires the following attributes on the class using this mixin:
    - self.config: Dict[str, Any]
    - self.logger: logging.Logger
    - self.results_dir: Path

    Provides:
    - Iteration history recording
    - Best parameter tracking
    - Results file I/O (CSV, JSON)
    - Pareto front storage for multi-objective optimization
    """

    def __init_results_tracking__(self):
        """Initialize results tracking state. Call in optimizer __init__."""
        self._iteration_history: List[Dict[str, Any]] = []
        self._best_score: float = float('-inf')
        self._best_params: Optional[Dict[str, float]] = None
        self._best_iteration: int = -1
        self._pareto_front: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None

    # =========================================================================
    # Iteration tracking
    # =========================================================================

    def record_iteration(
        self,
        iteration: int,
        score: float,
        params: Dict[str, float],
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record results from an optimization iteration.

        Args:
            iteration: Iteration number
            score: Fitness/objective score
            params: Parameter values used
            additional_metrics: Optional additional metrics (NSE, RMSE, etc.)
        """
        record = {
            'iteration': iteration,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            **params,
        }

        if additional_metrics:
            record.update(additional_metrics)

        self._iteration_history.append(record)

        self.logger.debug(
            f"Iteration {iteration}: score={score:.4f}, params={params}"
        )

    def update_best(
        self,
        score: float,
        params: Dict[str, float],
        iteration: int
    ) -> bool:
        """
        Update best solution if the new score is better.

        Args:
            score: New fitness score
            params: Parameter values
            iteration: Iteration number

        Returns:
            True if the best was updated, False otherwise
        """
        # Handle invalid scores
        if score is None or np.isnan(score) or score <= -900:
            return False

        # Check if better (for maximization problems like KGE)
        if score > self._best_score:
            self._best_score = score
            self._best_params = params.copy()
            self._best_iteration = iteration

            self.logger.info(
                f"New best at iteration {iteration}: score={score:.4f}"
            )
            return True

        return False

    def get_best_result(self) -> Dict[str, Any]:
        """
        Get the best result found so far.

        Returns:
            Dictionary with best score, params, and iteration
        """
        return {
            'score': self._best_score,
            'params': self._best_params,
            'iteration': self._best_iteration,
        }

    def get_iteration_history(self) -> pd.DataFrame:
        """
        Get iteration history as a DataFrame.

        Returns:
            DataFrame with all recorded iterations
        """
        if not self._iteration_history:
            return pd.DataFrame()

        return pd.DataFrame(self._iteration_history)

    # =========================================================================
    # Results persistence
    # =========================================================================

    def save_results(
        self,
        algorithm: str,
        metric_name: str = 'KGE',
        experiment_id: Optional[str] = None,
        standard_filename: bool = False
    ) -> Path:
        """
        Save optimization results to a CSV file.

        Args:
            algorithm: Algorithm name (e.g., 'PSO', 'DDS')
            metric_name: Name of the optimization metric
            experiment_id: Optional experiment identifier
            standard_filename: If True, uses the standard SYMFLUENCE naming convention 
                              ({experiment_id}_parallel_iteration_results.csv)

        Returns:
            Path to the saved results file
        """
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'optimization')

        # Create results dataframe
        df = self.get_iteration_history()

        if df.empty:
            self.logger.warning("No results to save")
            return None

        # Generate filename
        if standard_filename:
            filename = f"{experiment_id}_parallel_iteration_results.csv"
        else:
            filename = f"{experiment_id}_{algorithm.lower()}_results.csv"
            
        results_path = self.results_dir / filename

        # Save to CSV
        df.to_csv(results_path, index=False)

        self.logger.info(f"Saved optimization results to {results_path}")

        return results_path

    def save_best_params(
        self,
        algorithm: str,
        experiment_id: Optional[str] = None
    ) -> Path:
        """
        Save best parameters to a JSON file.

        Args:
            algorithm: Algorithm name
            experiment_id: Optional experiment identifier

        Returns:
            Path to the saved parameters file
        """
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'optimization')

        if self._best_params is None:
            self.logger.warning("No best parameters to save")
            return None

        # Convert numpy types to JSON-serializable types
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types."""
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(val) for key, val in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # Prepare output
        output = {
            'algorithm': algorithm,
            'experiment_id': experiment_id,
            'best_score': float(self._best_score) if self._best_score is not None else None,
            'best_iteration': int(self._best_iteration) if self._best_iteration is not None else None,
            'best_params': convert_to_serializable(self._best_params),
            'timestamp': datetime.now().isoformat(),
        }

        # Generate filename
        filename = f"{experiment_id}_{algorithm.lower()}_best_params.json"
        params_path = self.results_dir / filename

        # Save to JSON
        with open(params_path, 'w') as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Saved best parameters to {params_path}")

        return params_path

    def load_results(self, results_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load results from a CSV file.

        Args:
            results_path: Path to results file

        Returns:
            DataFrame with loaded results
        """
        results_path = Path(results_path)

        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        return pd.read_csv(results_path)

    def load_best_params(self, params_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load best parameters from a JSON file.

        Args:
            params_path: Path to parameters file

        Returns:
            Dictionary with best parameters
        """
        params_path = Path(params_path)

        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

        with open(params_path, 'r') as f:
            return json.load(f)

    # =========================================================================
    # Multi-objective support
    # =========================================================================

    def record_pareto_solution(
        self,
        objectives: List[float],
        params: Dict[str, float],
        dominated: bool = False
    ) -> None:
        """
        Record a solution on the Pareto front.

        Args:
            objectives: List of objective values
            params: Parameter values
            dominated: Whether this solution is dominated
        """
        record = {
            'objectives': objectives,
            'params': params,
            'dominated': dominated,
            'timestamp': datetime.now().isoformat(),
        }

        self._pareto_front.append(record)

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """
        Get non-dominated solutions from the Pareto front.

        Returns:
            List of non-dominated solutions
        """
        return [s for s in self._pareto_front if not s.get('dominated', False)]

    def save_pareto_front(
        self,
        algorithm: str,
        experiment_id: Optional[str] = None
    ) -> Path:
        """
        Save Pareto front to a CSV file.

        Args:
            algorithm: Algorithm name
            experiment_id: Optional experiment identifier

        Returns:
            Path to the saved Pareto front file
        """
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'optimization')

        pareto_solutions = self.get_pareto_front()

        if not pareto_solutions:
            self.logger.warning("No Pareto front to save")
            return None

        # Convert to DataFrame
        records = []
        for sol in pareto_solutions:
            record = {
                f'obj_{i}': obj for i, obj in enumerate(sol['objectives'])
            }
            record.update(sol['params'])
            records.append(record)

        df = pd.DataFrame(records)

        # Generate filename
        filename = f"{experiment_id}_{algorithm.lower()}_pareto_front.csv"
        pareto_path = self.results_dir / filename

        # Save
        df.to_csv(pareto_path, index=False)

        self.logger.info(f"Saved Pareto front to {pareto_path}")

        return pareto_path

    # =========================================================================
    # Timing
    # =========================================================================

    def start_timing(self) -> None:
        """Start timing the optimization run."""
        self._start_time = datetime.now()

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since start.

        Returns:
            Elapsed time in seconds
        """
        if self._start_time is None:
            return 0.0

        return (datetime.now() - self._start_time).total_seconds()

    def format_elapsed_time(self) -> str:
        """
        Get formatted elapsed time string.

        Returns:
            Formatted time string (e.g., "1h 23m 45s")
        """
        elapsed = self.get_elapsed_time()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
