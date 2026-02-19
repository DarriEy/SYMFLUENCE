"""
Evaluation Metrics Tracker

Tracks crash rates and logs optimization progress in a consistent format.
"""

import logging
from typing import Dict, Any, Optional, Callable

from symfluence.core.constants import ModelDefaults


class EvaluationMetricsTracker:
    """Tracks evaluation crash rates and logs iteration progress.

    Separated from BaseModelOptimizer to isolate metrics bookkeeping
    from algorithm execution logic.

    Args:
        max_iterations: Total iterations for progress reporting
        logger: Logger instance
        elapsed_time_fn: Callable returning formatted elapsed time string
    """

    PENALTY_SCORE = ModelDefaults.PENALTY_SCORE

    def __init__(
        self,
        max_iterations: int,
        logger: logging.Logger,
        elapsed_time_fn: Callable[[], str]
    ):
        self.max_iterations = max_iterations
        self.logger = logger
        self._elapsed_time_fn = elapsed_time_fn

        # Crash-rate counters
        self._total_evaluations: int = 0
        self._crash_count: int = 0
        self._last_crash_warning: int = 0

    # ------------------------------------------------------------------
    # Crash tracking
    # ------------------------------------------------------------------

    def track_evaluation(self, score: float) -> None:
        """Record an evaluation result, incrementing crash counter if penalty.

        Args:
            score: Fitness score from evaluation
        """
        self._total_evaluations += 1
        if score <= self.PENALTY_SCORE:
            self._crash_count += 1

        # Warn every 50 evaluations when crash rate exceeds 10%
        if (self._total_evaluations % 50 == 0
                and self._total_evaluations > self._last_crash_warning):
            crash_rate = self._crash_count / self._total_evaluations
            if crash_rate > 0.10:
                self.logger.warning(
                    f"High crash rate: {self._crash_count}/{self._total_evaluations} "
                    f"({crash_rate:.1%}) evaluations returned penalty score"
                )
                self._last_crash_warning = self._total_evaluations

    def get_crash_stats(self) -> Dict[str, Any]:
        """Return crash rate statistics.

        Returns:
            Dictionary with 'crash_count', 'total_evaluations', 'crash_rate'.
        """
        rate = (self._crash_count / self._total_evaluations
                if self._total_evaluations > 0 else 0.0)
        return {
            'crash_count': self._crash_count,
            'total_evaluations': self._total_evaluations,
            'crash_rate': rate,
        }

    # ------------------------------------------------------------------
    # Progress logging
    # ------------------------------------------------------------------

    def log_iteration_progress(
        self,
        algorithm_name: str,
        iteration: int,
        best_score: float,
        secondary_score: Optional[float] = None,
        secondary_label: Optional[str] = None,
        n_improved: Optional[int] = None,
        population_size: Optional[int] = None,
        crash_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log optimization progress in consistent format.

        Format: "{ALG} {iter}/{max} ({%}) | Best: {score} | ... | Elapsed: {time}"
        """
        progress_pct = (iteration / self.max_iterations) * 100
        elapsed = self._elapsed_time_fn()

        msg_parts = [
            f"{algorithm_name} {iteration}/{self.max_iterations} ({progress_pct:.0f}%)",
            f"Best: {best_score:.4f}"
        ]

        if secondary_score is not None:
            label = secondary_label or "Secondary"
            msg_parts.append(f"{label}: {secondary_score:.4f}")

        if n_improved is not None and population_size is not None:
            msg_parts.append(f"Improved: {n_improved}/{population_size}")

        if crash_stats and crash_stats.get('total_evaluations', 0) > 0:
            msg_parts.append(
                f"Crashes: {crash_stats['crash_count']}/{crash_stats['total_evaluations']} "
                f"({crash_stats['crash_rate']:.1%})"
            )

        msg_parts.append(f"Elapsed: {elapsed}")

        self.logger.info(" | ".join(msg_parts))

    def log_initial_population(
        self,
        algorithm_name: str,
        population_size: int,
        best_score: float
    ) -> None:
        """Log initial population evaluation completion.

        Args:
            algorithm_name: Algorithm name
            population_size: Population size
            best_score: Best score from initial evaluation
        """
        self.logger.info(
            f"{algorithm_name} initial population ({population_size} individuals) "
            f"complete | Best score: {best_score:.4f}"
        )
