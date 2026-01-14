"""
Base Objective for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseObjective(ABC):
    """
    Abstract base class for optimization objective functions in SYMFLUENCE.

    This class defines the interface that all objective function implementations must follow.
    Objective functions transform multi-dimensional evaluation results (multiple variables
    and metrics) into a single scalar value that optimization algorithms can minimize.

    Role in Calibration Workflow:
        1. **Evaluation Phase**: Model runs produce output for multiple variables (streamflow,
           snow, ET, TWS, etc.)
        2. **Metric Calculation**: Evaluators compute performance metrics (KGE, NSE, RMSE,
           correlation, etc.) for each variable by comparing simulated vs observed data
        3. **Objective Calculation**: This class transforms the multi-dimensional metric
           results into a single scalar objective value
        4. **Optimization**: Optimizer algorithms (DDS, DE, PSO, NSGA-II) use the scalar
           objective to guide parameter search toward better solutions

    Contract for Implementations:
        Concrete subclasses must implement the `calculate()` method to:
        - Accept evaluation results as nested dictionary: {variable: {metric: value}}
        - Transform multi-dimensional results into a single scalar value
        - Return a value to be MINIMIZED (lower is better)
        - Handle missing data gracefully (penalties or default values)
        - Apply weighting schemes for multi-criteria optimization

    Input Specification:
        The `evaluation_results` parameter structure::

            {
                'STREAMFLOW': {
                    'kge': 0.85,
                    'nse': 0.82,
                    'pbias': -5.2,
                    'rmse': 12.3
                },
                'SWE': {
                    'kge': 0.72,
                    'pbias': 8.1
                },
                'ET': {
                    'correlation': 0.68,
                    'rmse': 15.4
                }
            }

    Output Specification:
        - **Type**: float
        - **Convention**: MINIMIZATION (lower values = better performance)
        - **Transformation**: Many metrics (KGE, NSE, correlation) are maximized in
          isolation but must be transformed for minimization (e.g., 1 - KGE)
        - **Typical Range**: 0.0 (perfect) to 2.0+ (poor), though unbounded
        - **Penalty Values**: Large positive values (e.g., 10.0) for missing data or
          failed simulations

    Common Objective Formulations:
        Single-Variable:
            objective = 1 - KGE_streamflow

        Multi-Variable Weighted:
            objective = w1*(1 - KGE_streamflow) + w2*(1 - NSE_swe) + w3*(1 - corr_et)
            where w1 + w2 + w3 = 1

        Multi-Metric:
            objective = (1 - KGE) + 0.1*|PBIAS|/100 + 0.01*RMSE/sigma

    Registry Pattern:
        Objective classes are registered using the ObjectiveRegistry decorator::

            @ObjectiveRegistry.register('MULTIVARIATE')
            class MultivariateObjective(BaseObjective):
                def calculate(self, evaluation_results):
                    # Implementation here
                    pass

        This enables runtime selection of objective functions via configuration::

            OBJECTIVE_FUNCTION: MULTIVARIATE
            OBJECTIVE_WEIGHTS:
                STREAMFLOW: 0.7
                SWE: 0.3

    Configuration Integration:
        Implementations should access configuration via `self.config`:
        - OBJECTIVE_FUNCTION: Name of objective class (registry key)
        - OBJECTIVE_WEIGHTS: Variable-specific weights for multi-criteria optimization
        - OBJECTIVE_METRICS: Primary metric per variable (e.g., {'STREAMFLOW': 'kge'})
        - OBJECTIVE_PENALTY: Penalty value for missing/failed evaluations

    Error Handling:
        Implementations should handle:
        - Missing variables (not all expected variables in evaluation_results)
        - Missing metrics (expected metric not calculated for a variable)
        - Empty dictionaries (variable present but no metrics calculated)
        - NaN or infinite values in evaluation results
        - Data quality issues (insufficient overlap, all-zero observations)

    Thread Safety:
        Objective instances may be used across multiple parallel evaluations.
        Implementations should:
        - Avoid modifying instance state in calculate()
        - Use only method-local variables or read-only config access
        - Be stateless with respect to evaluation results

    Example Implementation:
        >>> from symfluence.optimization.objectives.base import BaseObjective
        >>> from symfluence.optimization.objectives.registry import ObjectiveRegistry
        >>>
        >>> @ObjectiveRegistry.register('SIMPLE_KGE')
        >>> class SimpleKGEObjective(BaseObjective):
        ...     '''Minimize 1 - KGE for streamflow only'''
        ...
        ...     def calculate(self, evaluation_results):
        ...         if 'STREAMFLOW' not in evaluation_results:
        ...             return 10.0  # Penalty for missing data
        ...
        ...         kge = evaluation_results['STREAMFLOW'].get('kge')
        ...         if kge is None:
        ...             return 10.0  # Penalty for missing metric
        ...
        ...         return 1.0 - kge  # Minimize (1 - KGE)
        >>>
        >>> # Usage in calibration
        >>> config = {'OBJECTIVE_FUNCTION': 'SIMPLE_KGE'}
        >>> objective = ObjectiveRegistry.get_objective(config, logger)
        >>> eval_results = {'STREAMFLOW': {'kge': 0.85, 'nse': 0.82}}
        >>> score = objective.calculate(eval_results)
        >>> print(score)  # 0.15 (lower is better)
        0.15

    Notes:
        - The calculate() method is called once per parameter set during optimization
        - Computational efficiency is important for large-scale calibrations
        - Avoid expensive operations (file I/O, logging) inside calculate()
        - Consider using cached/pre-computed values when possible
        - The objective value directly influences optimization convergence speed

    See Also:
        - optimization.objectives.multivariate.MultivariateObjective: Multi-criteria implementation
        - optimization.objectives.registry.ObjectiveRegistry: Objective factory
        - optimization.calibration_targets.base.BaseCalibrationTarget: Uses objectives
        - evaluation.evaluators.base.BaseEvaluator: Produces evaluation_results input
        - optimization.optimizers.base_model_optimizer.BaseModelOptimizer: Consumes objectives
    """
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def calculate(self, evaluation_results: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate a scalar objective value from evaluation results.

        Args:
            evaluation_results: Nested dict of {variable: {metric: value}}

        Returns:
            Scalar objective value (to be minimized)
        """
        pass
