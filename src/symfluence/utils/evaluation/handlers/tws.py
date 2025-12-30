import numpy as np
import pandas as pd
from typing import Dict, Any
from ..base import BaseEvaluator
from ..registry import EvaluationRegistry

@EvaluationRegistry.register('TWS')
class TWSEvaluator(BaseEvaluator):
    def calculate_metrics(self, sim: pd.Series, obs: pd.Series) -> Dict[str, float]:
        # TWS typically monthly, anomalies already calculated in observation handler
        s, o = self.align_series(sim, obs)
        if len(s) < 12:
            return {'corr': np.nan, 'rmse': np.nan}
            
        return {
            'corr': s.corr(o),
            'rmse': np.sqrt(np.mean((s - o) ** 2)),
            'nse': 1 - (np.sum((o - s)**2) / np.sum((o - o.mean())**2))
        }
