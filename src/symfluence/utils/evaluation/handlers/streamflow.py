import numpy as np
import pandas as pd
from typing import Dict, Any
from ..base import BaseEvaluator
from ..registry import EvaluationRegistry
from symfluence.utils.evaluation.calculate_sim_stats import get_KGE, get_NSE, get_RMSE, get_MAE

@EvaluationRegistry.register('STREAMFLOW')
class StreamflowEvaluator(BaseEvaluator):
    def calculate_metrics(self, sim: pd.Series, obs: pd.Series) -> Dict[str, float]:
        s, o = self.align_series(sim, obs)
        if len(s) < 30:
            return {'kge': np.nan, 'nse': np.nan}
            
        return {
            'kge': get_KGE(s.values, o.values),
            'nse': get_NSE(s.values, o.values),
            'rmse': get_RMSE(s.values, o.values),
            'mae': get_MAE(s.values, o.values)
        }
