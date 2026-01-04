"""
Base Objective for SYMFLUENCE
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class BaseObjective(ABC):
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
