"""
Multi-criteria objective function for multi-variable model calibration.

Combines multiple variables and metrics into a single weighted scalar
objective for multi-criteria optimization.
"""

from typing import Dict, Any
from .base import BaseObjective
from .registry import ObjectiveRegistry

@ObjectiveRegistry.register('MULTIVARIATE')
class MultivariateObjective(BaseObjective):
    """
    Combines multiple variables and metrics into a single scalar objective.
    Used for multi-criteria calibration.
    """
    def calculate(self, evaluation_results: Dict[str, Dict[str, float]]) -> float:
        # Get weights from config
        weights = self.config.get('OBJECTIVE_WEIGHTS', {'STREAMFLOW': 1.0})
        # Normalize weights
        total_weight = sum(weights.values())
        norm_weights = {k.upper(): v/total_weight for k, v in weights.items()}
        
        # Primary metric per variable from config
        metrics = self.config.get('OBJECTIVE_METRICS', {
            'STREAMFLOW': 'kge',
            'TWS': 'nse',
            'SCA': 'corr',
            'ET': 'corr'
        })
        
        composite_score = 0.0
        
        for var, weight in norm_weights.items():
            if var in evaluation_results:
                metric_name = metrics.get(var, 'kge').lower()
                score = evaluation_results[var].get(metric_name, -10.0)
                
                # Transform to 0-1 range where 1 is best if needed, or keep as is for KGE/NSE
                # For calibration we want to MINIMIZE, so we'll use (1 - score) for KGE/NSE
                composite_score += weight * (1.0 - score)
            else:
                # Penalty for missing data
                composite_score += weight * 2.0
                
        return composite_score
