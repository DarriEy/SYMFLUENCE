#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SYMFLUENCE Calibration Targets

This module provides calibration targets for different hydrologic variables.
These targets handle data loading, processing, and metric calculation for specific variables
during the optimization/calibration process.

Note: This module has been refactored to use the centralized evaluators in 
symfluence.evaluation.evaluators. The classes here are aliases for backward compatibility.
"""

import logging

import pandas as pd

from pathlib import Path

from typing import Dict, List, Optional, Any



from symfluence.evaluation.evaluators import (

    ModelEvaluator as CalibrationTarget,

    ETEvaluator as ETTarget,

        StreamflowEvaluator as StreamflowTarget,

        GRStreamflowEvaluator as GRStreamflowTarget,

        HYPEStreamflowEvaluator as HYPEStreamflowTarget,

        SoilMoistureEvaluator as SoilMoistureTarget,



    SnowEvaluator as SnowTarget,

    GroundwaterEvaluator as GroundwaterTarget,

    TWSEvaluator as TWSTarget

)



class MultivariateTarget(CalibrationTarget):



    """



    Multivariate calibration target that combines multiple variables.



    Delegates scoring to the AnalysisManager and MultivariateObjective.



    """



    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):



        super().__init__(config, project_dir, logger)



        from symfluence.evaluation.analysis_manager import AnalysisManager



        from ..objectives import ObjectiveRegistry



        



        self.analysis_manager = AnalysisManager(config, logger)



        self.objective_handler = ObjectiveRegistry.get_objective('MULTIVARIATE', config, logger)



        



        # Get requested variables from weights/metrics config



        self.variables = list(config.get('OBJECTIVE_WEIGHTS', {'STREAMFLOW': 1.0}).keys())







    def get_simulation_files(self, sim_dir: Path) -> List[Path]:



        # Multivariate needs both daily and hourly possibly, return all



        return list(sim_dir.glob("*.nc"))







    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:



        """Not directly used by MultivariateTarget as it delegates to component evaluators."""



        raise NotImplementedError("MultivariateTarget delegates extraction to individual evaluators.")







    def get_observed_data_path(self) -> Path:



        """Not directly used by MultivariateTarget."""



        return self.project_dir / "observations"







    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:



        """Not directly used by MultivariateTarget."""



        return None







    def needs_routing(self) -> bool:



        """Multivariate target might need routing if streamflow is one of the components."""



        return 'STREAMFLOW' in [v.upper() for v in self.variables]







    def evaluate(self, sim_dir: Path, **kwargs) -> float:





        """

        Evaluate multiple variables and return a composite score.

        """

        # 1. Extract simulated data for all requested variables

        sim_results = {}

        for var in self.variables:

            evaluator = self.analysis_manager.EvaluationRegistry.get_evaluator(

                var, self.config, self.logger, self.project_dir, target=var

            )

            if evaluator:

                sim_files = evaluator.get_simulation_files(sim_dir)

                if sim_files:

                    sim_results[var] = evaluator.extract_simulated_data(sim_files)



        # 2. Run multivariate evaluation

        eval_results = self.analysis_manager.run_multivariate_evaluation(sim_results)

        

        # 3. Calculate scalar objective

        return self.objective_handler.calculate(eval_results)



# Re-export for backward compatibility

__all__ = [

    'CalibrationTarget',

        'ETTarget',

        'StreamflowTarget',

        'GRStreamflowTarget',

        'SoilMoistureTarget',

    

    'SnowTarget',

    'GroundwaterTarget',

    'TWSTarget',

    'MultivariateTarget'

]
