"""
Base Evaluator for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

class BaseEvaluator(ABC):
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.spinup_years = config.get('EVALUATION_SPINUP_YEARS', 1)
        self.start_date = pd.to_datetime(config.get('EXPERIMENT_TIME_START'))
        self.eval_start = self.start_date + pd.DateOffset(years=self.spinup_years)

    @abstractmethod
    def calculate_metrics(self, sim: pd.Series, obs: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics between simulated and observed series."""
        pass

    def align_series(self, sim: pd.Series, obs: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align time series and remove spinup period."""
        # Ensure indices are datetime
        sim.index = pd.to_datetime(sim.index)
        obs.index = pd.to_datetime(obs.index)
        
        # Trim to evaluation period
        sim_eval = sim[sim.index >= self.eval_start]
        
        # Find common indices
        common = sim_eval.index.intersection(obs.index)
        if len(common) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)
            
        return sim_eval.loc[common], obs.loc[common]