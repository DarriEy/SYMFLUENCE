# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Multi-criteria objective function for multi-variable model calibration.

Combines multiple variables and metrics into a single weighted scalar
objective for multi-criteria optimization.

This module implements the MULTIVARIATE objective that allows simultaneous
calibration against multiple data streams (streamflow, snow, evapotranspiration, etc.)
with user-defined weights to reflect relative importance.
"""

from typing import Dict

from .base import BaseObjective
from .registry import ObjectiveRegistry


@ObjectiveRegistry.register('MULTIVARIATE')
class MultivariateObjective(BaseObjective):
    """Weighted multi-variable objective function.

    Combines multiple variables and metrics into a single scalar objective
    for multi-criteria calibration. Supports flexible metric selection per
    variable and user-defined weighting for multi-criteria optimization.

    This objective is the primary method for calibrating models against
    multiple observational datasets simultaneously (e.g., streamflow, snow cover,
    evapotranspiration, groundwater anomalies).

    Configuration Requirements:
        OBJECTIVE_WEIGHTS (dict): Variable-specific weights. Example:
            {'STREAMFLOW': 0.7, 'SWE': 0.2, 'ET': 0.1}
            Weights are automatically normalized to sum to 1.0.

        OBJECTIVE_METRICS (dict, optional): Primary metric per variable.
            Defaults: {'STREAMFLOW': 'kge', 'TWS': 'nse', 'SCA': 'corr', 'ET': 'corr'}
            Valid metrics depend on evaluator but commonly: 'kge', 'nse', 'kgeprime',
            'rmse', 'pbias', 'correlation', 'r'.

    Algorithm:
        1. Retrieve weights from config (default: STREAMFLOW=1.0)
        2. Normalize weights so they sum to 1.0
        3. For each variable in weights:
           - Look up primary metric value (with case-insensitive fallback)
           - If metric not found, assign penalty value (-10.0)
           - If variable not in results, assign penalty (2.0)
        4. Sum weighted transformed scores: sum(w_i * (1 - metric_i))
        5. Return composite score (to be minimized)

    Metric Value Resolution (robust lookup):
        The method handles varying metric case conventions by trying:
        1. Exact match (e.g., 'kge')
        2. Uppercase match (e.g., 'KGE')
        3. Lowercase match
        4. Aliases (e.g., 'corr' -> 'correlation' or 'r')

    Score Transformation:
        Most metrics (KGE, NSE) are naturally maximized (range 0-1).
        This objective transforms them for minimization: score = 1 - metric
        This makes optimization algorithms (which minimize) move toward 1.0 metric values.

    Example Configuration:
        OBJECTIVE_FUNCTION: MULTIVARIATE
        OBJECTIVE_WEIGHTS:
            STREAMFLOW: 0.7
            SWE: 0.2
            ET: 0.1
        OBJECTIVE_METRICS:
            STREAMFLOW: kge
            SWE: nse
            ET: correlation
    """
    def calculate(self, evaluation_results: Dict[str, Dict[str, float]]) -> float:
        """Calculate weighted multi-variable objective score.

        Transforms multi-dimensional evaluation results into a scalar objective
        by: (1) weighting each variable by user-specified importance, (2) selecting
        the primary metric for each variable, (3) transforming metrics to minimize,
        and (4) summing weighted contributions.

        Args:
            evaluation_results: Nested dict structure {variable: {metric: value}}.
                Example: {'STREAMFLOW': {'kge': 0.85, 'nse': 0.82}, 'SWE': {'nse': 0.72}}

        Returns:
            float: Composite objective score. Lower values indicate better overall
            model performance. Typical range 0.0 (perfect) to 2.0+ (poor).

        Note:
            - Variables in evaluation_results but not in weights are ignored
            - Variables in weights but missing from evaluation_results incur 2.0 penalty
            - Missing metrics for a variable incur -10.0 penalty (high cost to encourage
              fixing evaluation issues)
            - Metric values are transformed with (1 - value) assuming they are maximized
              scores (KGE, NSE, correlation range ~0-1)
        """
        # Get weights from config (default: streamflow only if not specified)
        weights = self._get_config_value(
            lambda: self.config.optimization.objective_weights,
            default={'STREAMFLOW': 1.0},
            dict_key='OBJECTIVE_WEIGHTS'
        )

        # Normalize weights to sum to 1.0. This ensures the objective value is
        # scale-independent: changing all weights by a constant factor doesn't
        # change optimization behavior.
        total_weight = sum(weights.values())
        norm_weights = {k.upper(): v/total_weight for k, v in weights.items()}

        # Primary metric per variable from config. These define which metric
        # is most important for each variable (e.g., KGE for streamflow, NSE for snow)
        default_metrics = {
            'STREAMFLOW': 'kge',
            'TWS': 'nse',
            'SCA': 'corr',
            'ET': 'corr'
        }
        metrics = self._get_config_value(
            lambda: self.config.optimization.objective_metrics,
            default=default_metrics,
            dict_key='OBJECTIVE_METRICS'
        )

        composite_score = 0.0

        for var, weight in norm_weights.items():
            if var in evaluation_results:
                metric_name = metrics.get(var, 'kge')

                # Robust lookup for metric value. Different evaluators may use
                # different case conventions (kge vs KGE vs KGE_prime), so we
                # search systematically: exact -> upper -> lower -> aliases
                val = None

                # 1. Try exact match (most common convention)
                if metric_name in evaluation_results[var]:
                    val = evaluation_results[var][metric_name]
                # 2. Try uppercase (common in evaluators: KGE, NSE)
                elif metric_name.upper() in evaluation_results[var]:
                    val = evaluation_results[var][metric_name.upper()]
                # 3. Try lowercase
                elif metric_name.lower() in evaluation_results[var]:
                    val = evaluation_results[var][metric_name.lower()]
                # 4. Handle common aliases (correlation vs corr vs r)
                elif metric_name.lower() == 'corr':
                    # Try 'correlation' and 'r'
                    if 'correlation' in evaluation_results[var]:
                        val = evaluation_results[var]['correlation']
                    elif 'r' in evaluation_results[var]:
                        val = evaluation_results[var]['r']

                if val is None:
                    # Metric not found for this variable. This indicates a problem
                    # (data issues, mismatched evaluator output). Use high penalty
                    # to encourage fixing the underlying evaluation issue.
                    score = -10.0
                else:
                    score = val

                # Transform metrics for minimization: 1 - metric_value
                # This assumes metrics are in range [0, 1] where 1 is perfect.
                # For metrics like KGE, NSE, correlation: higher is better,
                # so (1 - metric) makes lower scores better for minimization.
                composite_score += weight * (1.0 - score)
            else:
                # Variable not in evaluation results (missing observations for this stream).
                # Assign moderate penalty to reduce contribution without completely
                # killing convergence (unlike -10.0 penalty for missing metrics).
                composite_score += weight * 2.0

        return composite_score
