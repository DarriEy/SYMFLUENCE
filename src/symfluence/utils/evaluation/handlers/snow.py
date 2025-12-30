import numpy as np
import pandas as pd
from typing import Dict, Any
from ..base import BaseEvaluator
from ..registry import EvaluationRegistry

@EvaluationRegistry.register('SCA')
class SnowEvaluator(BaseEvaluator):
    def calculate_metrics(self, sim: pd.Series, obs: pd.Series) -> Dict[str, float]:
        # sim should be scalarGroundSnowFraction or normalized scalarSWE
        # obs should be snow_cover_ratio
        s, o = self.align_series(sim, obs)
        if len(s) < 30:
            return {'corr': np.nan, 'rmse': np.nan}
            
        return {
            'corr': s.corr(o),
            'rmse': np.sqrt(np.mean((s - o) ** 2))
        }
