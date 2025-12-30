import numpy as np
import pandas as pd
from typing import Dict, Any
from ..base import BaseEvaluator
from ..registry import EvaluationRegistry

@EvaluationRegistry.register('ET')
class ETEvaluator(BaseEvaluator):
    def calculate_metrics(self, sim: pd.Series, obs: pd.Series) -> Dict[str, float]:
        # sim should be sum of ET components, mm/day
        # obs should be 8-day ET / 8, mm/day
        s, o = self.align_series(sim, obs)
        if len(s) < 10:
            return {'corr': np.nan, 'rmse': np.nan}
            
        return {
            'corr': s.corr(o),
            'rmse': np.sqrt(np.mean((s - o) ** 2))
        }
