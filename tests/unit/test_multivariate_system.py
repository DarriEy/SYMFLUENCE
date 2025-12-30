import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from symfluence.utils.data.observation.registry import ObservationRegistry
from symfluence.utils.evaluation.registry import EvaluationRegistry
from symfluence.utils.optimization.objective_registry import ObjectiveRegistry

def test_observation_registry():
    """Verify that observations are correctly registered and retrievable."""
    datasets = ObservationRegistry.list_observations()
    assert 'GRACE' in datasets
    assert 'MODIS_SNOW' in datasets
    assert 'MODIS_ET' in datasets

def test_evaluation_registry():
    """Verify that evaluators are correctly registered."""
    evaluators = EvaluationRegistry.list_evaluators()
    assert 'STREAMFLOW' in evaluators
    assert 'TWS' in evaluators
    assert 'SCA' in evaluators
    assert 'ET' in evaluators

def test_multivariate_objective_calculation():
    """Test the composite score calculation logic."""
    config = {
        'OBJECTIVE_WEIGHTS': {'STREAMFLOW': 0.7, 'TWS': 0.3},
        'OBJECTIVE_METRICS': {'STREAMFLOW': 'kge', 'TWS': 'nse'}
    }
    
    # Mock evaluation results
    eval_results = {
        'STREAMFLOW': {'kge': 0.8, 'nse': 0.75},
        'TWS': {'nse': 0.6, 'corr': 0.9}
    }
    
    handler = ObjectiveRegistry.get_objective('MULTIVARIATE', config, None)
    assert handler is not None
    
    # Expected: 0.7 * (1.0 - 0.8) + 0.3 * (1.0 - 0.6)
    # = 0.7 * 0.2 + 0.3 * 0.4
    # = 0.14 + 0.12 = 0.26
    score = handler.calculate(eval_results)
    assert pytest.approx(score) == 0.26

def test_evaluator_alignment():
    """Test the base evaluator's time series alignment logic."""
    from symfluence.utils.evaluation.base import BaseEvaluator
    
    class MockEvaluator(BaseEvaluator):
        def calculate_metrics(self, sim, obs): return {}
        
    config = {
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EVALUATION_SPINUP_YEARS': 1
    }
    
    evaluator = MockEvaluator(config, None)
    
    # Create test data
    times = pd.date_range('2020-01-01', periods=24, freq='MS')
    sim = pd.Series(np.random.rand(24), index=times)
    obs = pd.Series(np.random.rand(24), index=times)
    
    s_aligned, o_aligned = evaluator.align_series(sim, obs)
    
    # Should start from 2021-01-01
    assert s_aligned.index[0] == pd.Timestamp('2021-01-01')
    assert len(s_aligned) == 12
