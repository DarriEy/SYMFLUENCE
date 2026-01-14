"""
Base Parameter Manager for Model Calibration.

Provides shared normalization/denormalization algorithms and validation logic
for all model-specific parameter managers. Subclasses implement model-specific
bounds sources and file I/O through abstract methods.

Architecture:
    - Shared implementation: normalize, denormalize, validate
    - Abstract methods: bounds loading, parameter names, file I/O
    - Hooks: parameter formatting, scalar extraction

Usage:
    class MyModelParameterManager(BaseParameterManager):
        def _get_parameter_names(self) -> List[str]:
            return ['param1', 'param2']

        def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
            return {'param1': {'min': 0.0, 'max': 10.0}, ...}

        def update_model_files(self, params: Dict) -> bool:
            # Write params to model config files
            pass

        def get_initial_parameters(self) -> Dict:
            # Read initial params from files or defaults
            pass
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import logging


class BaseParameterManager(ABC):
    """
    Abstract base class for model parameter managers.

    Provides shared normalization/denormalization algorithms and validation logic.
    Subclasses implement model-specific bounds sources and file I/O.

    Attributes:
        config: Configuration dictionary for the model domain and experiment
        logger: Logger instance for diagnostic messages
        settings_dir: Path to model settings directory
        _param_names: Cached list of parameter names (lazy loaded)
        _param_bounds: Cached parameter bounds dictionary (lazy loaded)
    """

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path):
        """
        Initialize base parameter manager.

        Args:
            config: Configuration dictionary containing domain and experiment settings
            logger: Logger instance for diagnostic output
            settings_dir: Path to model-specific settings directory
        """
        self.config = config
        self.logger = logger
        self.settings_dir = settings_dir

        # Lazy-loaded caches (populated by subclass implementations)
        self._param_names: List[str] = []
        self._param_bounds: Dict[str, Dict[str, float]] = {}

    # ========================================================================
    # ABSTRACT METHODS (Model-specific - must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    def _get_parameter_names(self) -> List[str]:
        """
        Get ordered list of parameter names to calibrate.

        Returns:
            List of parameter names in calibration order.

        Implementation notes:
            - SUMMA: Combine local_params + basin_params + depth_params + mizu_params
            - FUSE: Return config-specified FUSE params
            - Ngen: Return module.param format (e.g., 'CFE.maxsmc', 'NOAH.refkdt')

        Example:
            return ['param1', 'param2', 'param3']
        """
        pass

    @abstractmethod
    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Load parameter bounds from model-specific source.

        Returns:
            Dictionary mapping param_name -> {'min': float, 'max': float}

        Implementation notes:
            - SUMMA: Parse from localParamInfo.txt/basinParamInfo.txt files
            - FUSE: Return hardcoded dictionary of FUSE parameter bounds
            - Ngen: Return hardcoded dictionary of CFE/NOAH/PET bounds

        Example:
            return {
                'param1': {'min': 0.0, 'max': 10.0},
                'param2': {'min': -5.0, 'max': 5.0}
            }
        """
        pass

    @abstractmethod
    def update_model_files(self, params: Dict[str, Any]) -> bool:
        """
        Write denormalized parameters to model configuration files.

        Args:
            params: Denormalized parameter dictionary (model-native format)

        Returns:
            True if successful, False otherwise

        Implementation notes:
            - SUMMA: Generate trialParams.nc, update coldState.nc, update param.nml.default
            - FUSE: Update NetCDF para_def.nc file
            - Ngen: Update JSON/BMI text files for CFE/NOAH/PET modules

        Example:
            # Write params to model config file
            config_path = self.settings_dir / 'model_config.txt'
            with open(config_path, 'w') as f:
                for param_name, value in params.items():
                    f.write(f"{param_name} = {value}\\n")
            return True
        """
        pass

    @abstractmethod
    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Get initial parameter values from model defaults or existing files.

        Returns:
            Dictionary of parameter values in model-native format, or None if not found.

        Implementation notes:
            - SUMMA: Extract from existing trialParams.nc or parse defaults
            - FUSE: Read from para_def.nc or use bounds midpoint
            - Ngen: Read from config files or use bounds midpoint

        Example:
            return {
                'param1': 5.0,
                'param2': 0.0
            }
        """
        pass

    # ========================================================================
    # PROPERTIES (Lazy-loaded caches)
    # ========================================================================

    @property
    def all_param_names(self) -> List[str]:
        """
        Get list of all parameter names (lazy loaded).

        Returns:
            Ordered list of parameter names for calibration.
        """
        if not self._param_names:
            self._param_names = self._get_parameter_names()
        return self._param_names

    @property
    def param_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter bounds dictionary (lazy loaded).

        Returns:
            Dictionary mapping param_name -> {'min': float, 'max': float}
        """
        if not self._param_bounds:
            self._param_bounds = self._load_parameter_bounds()
        return self._param_bounds

    # ========================================================================
    # SHARED IMPLEMENTATION (DRY - eliminates duplication across managers)
    # ========================================================================

    def normalize_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range for optimization.

        This is the SHARED ALGORITHM - identical across all managers.
        Eliminates ~50 lines of duplication per manager.

        Args:
            params: Dictionary of denormalized parameter values (model-native format).
                   May contain floats, ints, or numpy arrays.

        Returns:
            Normalized array in [0, 1] range, one value per parameter.

        Algorithm:
            normalized[i] = (value - min) / (max - min)
            clipped to [0, 1] for safety

        Missing parameters default to 0.5 with a warning.
        """
        normalized = np.zeros(len(self.all_param_names))

        for i, param_name in enumerate(self.all_param_names):
            # Handle missing parameters
            if param_name not in params or param_name not in self.param_bounds:
                self.logger.warning(f"Parameter {param_name} missing, using 0.5")
                normalized[i] = 0.5
                continue

            bounds = self.param_bounds[param_name]
            value = self._extract_scalar_value(params[param_name])

            # Normalize to [0, 1]: (value - min) / (max - min)
            range_size = bounds['max'] - bounds['min']
            if range_size == 0:
                # Handle edge case: min == max (constant parameter)
                self.logger.warning(f"Parameter {param_name} has zero range, setting to 0.5")
                normalized[i] = 0.5
            else:
                normalized[i] = (value - bounds['min']) / range_size

        # Clip to [0, 1] for safety (handles out-of-bounds values)
        return np.clip(normalized, 0.0, 1.0)

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        """
        Denormalize parameters from [0, 1] range to actual values.

        This is the SHARED ALGORITHM - identical across all managers.
        Eliminates ~40 lines of duplication per manager.

        Args:
            normalized_array: Normalized parameter array in [0, 1] range.

        Returns:
            Dictionary of denormalized parameter values in model-native format.

        Algorithm:
            denorm = min + normalized * (max - min)
            clipped to [min, max] for safety

        Calls _format_parameter_value hook for model-specific formatting
        (e.g., SUMMA returns np.ndarray, others return float).
        """
        params = {}

        for i, param_name in enumerate(self.all_param_names):
            if param_name not in self.param_bounds:
                self.logger.warning(f"No bounds for {param_name}, skipping")
                continue

            bounds = self.param_bounds[param_name]

            # Denormalize: min + normalized * (max - min)
            denorm_value = bounds['min'] + normalized_array[i] * (bounds['max'] - bounds['min'])

            # Clip to bounds for safety
            denorm_value = np.clip(denorm_value, bounds['min'], bounds['max'])

            # Format based on model-specific needs (hook for subclasses)
            params[param_name] = self._format_parameter_value(param_name, denorm_value)

        return params

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate that all parameters are within bounds.

        This is SHARED VALIDATION LOGIC (currently duplicated in FUSE/Ngen).
        Eliminates ~20 lines of duplication.

        Args:
            params: Parameter dictionary to validate.

        Returns:
            True if all parameters are within bounds, False otherwise.

        Logs warnings for out-of-bounds parameters.
        Ignores parameters not in bounds dictionary.
        """
        for param_name, value in params.items():
            if param_name not in self.param_bounds:
                continue

            scalar_value = self._extract_scalar_value(value)
            bounds = self.param_bounds[param_name]

            if not (bounds['min'] <= scalar_value <= bounds['max']):
                self.logger.warning(
                    f"Parameter {param_name}={scalar_value} outside bounds "
                    f"[{bounds['min']}, {bounds['max']}]"
                )
                return False

        return True

    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter bounds (convenience method).

        Returns:
            Copy of parameter bounds dictionary.
        """
        return self.param_bounds.copy()

    # ========================================================================
    # PROTECTED HELPERS (Subclasses can override for model-specific behavior)
    # ========================================================================

    def _extract_scalar_value(self, value: Any) -> float:
        """
        Extract scalar from parameter value (may be array, scalar, etc.).

        Handles multiple value types:
            - float/int: return as float
            - numpy scalar: return as float
            - single-element array: return element
            - multi-element array: return mean

        Args:
            value: Parameter value in any format

        Returns:
            Scalar float value

        Override this for custom extraction logic (e.g., median instead of mean).
        """
        if isinstance(value, np.ndarray):
            if len(value) == 1:
                return float(value[0])
            else:
                # Use mean for multi-element arrays
                return float(np.mean(value))
        return float(value)

    def _format_parameter_value(self, param_name: str, value: float) -> Any:
        """
        Format denormalized value for model-specific needs.

        Default: return as float.
        Subclasses override for arrays, special formatting, etc.

        Args:
            param_name: Name of parameter
            value: Denormalized scalar value

        Returns:
            Formatted value (float by default, np.ndarray for SUMMA, etc.)

        Override examples:
            # SUMMA - return arrays
            def _format_parameter_value(self, param_name, value):
                if param_name in self.local_params:
                    return self._expand_to_hru_count(value)
                return np.array([value])

            # Default - return float
            def _format_parameter_value(self, param_name, value):
                return float(value)
        """
        return float(value)
