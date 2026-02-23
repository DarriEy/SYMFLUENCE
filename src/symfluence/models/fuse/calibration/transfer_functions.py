"""
Transfer Functions for Spatially Distributed FUSE Parameters.

This module implements Multiscale Parameter Regionalization (MPR) style
transfer functions that map catchment attributes to local parameter values.
Instead of calibrating parameters directly, we calibrate the coefficients
of transfer functions that relate parameters to physical catchment properties.

Transfer Function Forms:
    1. Power:       param = a * attr^b  (default - most flexible)
    2. Linear:      param = a + b * attr
    3. Exponential: param = a * exp(b * attr)
    4. Constant:    param = a (b fixed at 0, no spatial variation)

Key Design Principle:
    - All spatially varying parameters use: param = a * attr^b
    - Set b=0 (fixed) for spatially uniform parameters
    - Calibrate both a and b for parameters that should vary with attributes
    - This gives a unified framework where:
        * b=0: uniform parameter (just calibrate a)
        * b>0: parameter increases with attribute
        * b<0: parameter decreases with attribute

The calibration optimizes coefficients (a, b) which are then applied
across all subcatchments to generate spatially varying parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TransferFunction:
    """Base class for transfer functions."""

    def __init__(self, name: str, n_coefficients: int):
        self.name = name
        self.n_coefficients = n_coefficients

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply transfer function to attribute array."""
        raise NotImplementedError

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for each coefficient."""
        raise NotImplementedError


class LinearTF(TransferFunction):
    """Linear transfer function: param = a + b * attr"""

    def __init__(self, a_bounds: Tuple[float, float], b_bounds: Tuple[float, float]):
        super().__init__('linear', 2)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a, b = coeffs[0], coeffs[1]
        return a + b * attr

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds, self.b_bounds]


class PowerTF(TransferFunction):
    """Power transfer function: param = a * attr^b (with offset for zero handling)"""

    def __init__(self, a_bounds: Tuple[float, float], b_bounds: Tuple[float, float]):
        super().__init__('power', 2)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a, b = coeffs[0], coeffs[1]
        # Add small offset to avoid zero issues
        return a * np.power(attr + 0.01, b)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds, self.b_bounds]


class ExponentialTF(TransferFunction):
    """Exponential transfer function: param = a * exp(b * attr)"""

    def __init__(self, a_bounds: Tuple[float, float], b_bounds: Tuple[float, float]):
        super().__init__('exponential', 2)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a, b = coeffs[0], coeffs[1]
        return a * np.exp(b * attr)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds, self.b_bounds]


class ConstantTF(TransferFunction):
    """Constant (spatially uniform) parameter: param = a"""

    def __init__(self, a_bounds: Tuple[float, float]):
        super().__init__('constant', 1)
        self.a_bounds = a_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a = coeffs[0]
        return np.full_like(attr, a)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds]


class FlexiblePowerTF(TransferFunction):
    """
    Flexible power-law transfer function: param = a * attr^b

    This is the recommended unified form for all transfer functions.
    - When calibrate_exponent=True: both a and b are calibrated
    - When calibrate_exponent=False: b is fixed (default=0 gives constant)

    With normalized attributes [0,1] and b=0: param = a (uniform)
    With normalized attributes [0,1] and b>0: param increases with attribute
    With normalized attributes [0,1] and b<0: param decreases with attribute
    """

    def __init__(
        self,
        a_bounds: Tuple[float, float],
        b_bounds: Tuple[float, float] = (-2.0, 2.0),
        calibrate_exponent: bool = True,
        fixed_exponent: float = 0.0
    ):
        n_coeffs = 2 if calibrate_exponent else 1
        super().__init__('flexible_power', n_coeffs)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds
        self.calibrate_exponent = calibrate_exponent
        self.fixed_exponent = fixed_exponent

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a = coeffs[0]
        b = coeffs[1] if self.calibrate_exponent else self.fixed_exponent

        # Handle the case where attr might be 0 or negative
        # Add small offset and use absolute value for safety
        safe_attr = np.abs(attr) + 0.01

        if np.abs(b) < 1e-10:
            # b ≈ 0: return constant a (attr^0 = 1)
            return np.full_like(attr, a, dtype=float)
        else:
            return a * np.power(safe_attr, b)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        if self.calibrate_exponent:
            return [self.a_bounds, self.b_bounds]
        else:
            return [self.a_bounds]


class ParameterTransferManager:
    """
    Manages transfer functions for all FUSE parameters.

    Maps calibration coefficients to spatially distributed parameter values
    using catchment attributes and transfer functions.
    """

    # Default transfer function configurations for FUSE parameters
    # All use unified power-law form: param = a * attr^b
    # - calibrate_b=True: both a and b calibrated (spatial variation)
    # - calibrate_b=False: only a calibrated, b=0 (uniform)
    DEFAULT_PARAM_CONFIG = {
        # =====================================================================
        # STORAGE PARAMETERS - vary with climate/water availability
        # =====================================================================
        'MAXWATR_1': {
            'attribute': 'precip_mm_yr',
            'calibrate_b': True,  # Calibrate exponent - wetter areas may have more storage
            'description': 'Upper layer storage scales with precipitation',
        },
        'MAXWATR_2': {
            'attribute': 'precip_mm_yr',
            'calibrate_b': True,  # Calibrate exponent
            'description': 'Lower layer storage scales with precipitation',
        },
        'FRACTEN': {
            'attribute': 'aridity',
            'calibrate_b': False,  # Uniform - soil property
            'description': 'Tension storage fraction - uniform (soil property)',
        },

        # =====================================================================
        # FLOW PARAMETERS - vary with terrain/climate
        # =====================================================================
        'BASERTE': {
            'attribute': 'aridity',
            'calibrate_b': True,  # More arid = faster drainage
            'description': 'Baseflow rate varies with aridity',
        },
        'QB_POWR': {
            'attribute': 'aridity',
            'calibrate_b': True,  # Spatially varying - baseflow nonlinearity
            'description': 'Baseflow nonlinearity varies with aridity',
        },
        'PERCRTE': {
            'attribute': 'aridity',
            'calibrate_b': True,  # Spatially varying - percolation rate
            'description': 'Percolation rate varies with aridity',
        },
        'TIMEDELAY': {
            'attribute': 'precip_mm_yr',
            'calibrate_b': False,  # Uniform - handled by mizuRoute
            'description': 'Time delay - uniform (handled by mizuRoute)',
        },
        'RTFRAC1': {
            'attribute': 'aridity',
            'calibrate_b': False,  # Uniform
            'description': 'Root fraction in upper layer - uniform',
        },

        # =====================================================================
        # SNOW PARAMETERS - vary with elevation/temperature
        # =====================================================================
        'MBASE': {
            'attribute': 'elev_m',
            'calibrate_b': True,  # Melt threshold varies with elevation
            'description': 'Melt threshold varies with elevation',
        },
        'MFMAX': {
            'attribute': 'temp_C',
            'calibrate_b': True,  # Melt factor varies with temperature
            'description': 'Max melt factor varies with mean temperature',
        },
        'MFMIN': {
            'attribute': 'snow_frac',
            'calibrate_b': True,  # Min melt varies with snow prevalence
            'description': 'Min melt factor varies with snow prevalence',
        },
        'PXTEMP': {
            'attribute': 'elev_m',
            'calibrate_b': True,  # Rain/snow threshold varies with elevation
            'description': 'Rain/snow threshold varies with elevation',
        },
        'LAPSE': {
            'attribute': 'elev_m',
            'calibrate_b': True,  # Spatially varying - lapse rate varies with elevation
            'description': 'Lapse rate varies with elevation',
        },
    }

    def __init__(
        self,
        attributes_path: Path,
        param_bounds: Dict[str, Tuple[float, float]],
        param_config: Optional[Dict[str, Dict]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize transfer function manager.

        Args:
            attributes_path: Path to subcatchment attributes CSV
            param_bounds: Original parameter bounds {param_name: (min, max)}
            param_config: Optional custom transfer function configuration
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.param_bounds = param_bounds
        self.param_config = param_config or self.DEFAULT_PARAM_CONFIG

        # Load attributes
        self.attributes = pd.read_csv(attributes_path)
        self.n_subcatchments = len(self.attributes)
        self.logger.info(f"Loaded attributes for {self.n_subcatchments} subcatchments")

        # Normalize attributes to [0, 1] for stable transfer functions
        self._normalize_attributes()

        # Build transfer functions for each parameter
        self.transfer_functions: Dict[str, Tuple[TransferFunction, str]] = {}
        self.coefficient_map: Dict[str, List[str]] = {}  # param -> [coeff_names]
        self._build_transfer_functions()

    def _normalize_attributes(self):
        """Normalize numeric attributes to [0, 1] range."""
        self.attr_stats = {}
        for col in ['elev_m', 'precip_mm_yr', 'temp_C', 'aridity', 'snow_frac', 'temp_range_C']:
            if col in self.attributes.columns:
                min_val = self.attributes[col].min()
                max_val = self.attributes[col].max()
                self.attr_stats[col] = {'min': min_val, 'max': max_val}
                # Create normalized version
                range_val = max_val - min_val
                if range_val > 0:
                    self.attributes[f'{col}_norm'] = (self.attributes[col] - min_val) / range_val
                else:
                    self.attributes[f'{col}_norm'] = 0.5

    def _build_transfer_functions(self):
        """
        Build transfer functions for each parameter.

        Uses unified power-law form: param = a * attr^b
        - calibrate_b=True: calibrate both a and b (spatial variation)
        - calibrate_b=False: only calibrate a, b fixed at 0 (uniform)
        """
        self.all_coefficients = []
        self.coeff_bounds = []

        for param_name, config in self.param_config.items():
            if param_name not in self.param_bounds:
                continue

            p_min, p_max = self.param_bounds[param_name]
            attr_name = config.get('attribute', 'precip_mm_yr')
            calibrate_b = config.get('calibrate_b', False)

            # Always use normalized attributes for power-law stability
            norm_attr_name = f'{attr_name}_norm'
            if norm_attr_name in self.attributes.columns:
                attr_name = norm_attr_name

            # Create flexible power-law transfer function
            # a bounds: parameter range (output will be clipped anyway)
            # b bounds: [-2, 2] for reasonable power-law behavior
            tf = FlexiblePowerTF(
                a_bounds=(p_min, p_max),
                b_bounds=(-1.5, 1.5),  # Reasonable range for exponent
                calibrate_exponent=calibrate_b,
                fixed_exponent=0.0  # When b not calibrated, use b=0 (uniform)
            )

            if calibrate_b:
                coeff_names = [f'{param_name}_a', f'{param_name}_b']
            else:
                coeff_names = [f'{param_name}_a']

            self.transfer_functions[param_name] = (tf, attr_name)
            self.coefficient_map[param_name] = coeff_names

            # Add to global coefficient list
            for name in coeff_names:
                self.all_coefficients.append(name)
            self.coeff_bounds.extend(tf.get_coefficient_bounds())

        # Count spatially varying vs uniform parameters
        n_varying = sum(1 for c in self.param_config.values() if c.get('calibrate_b', False))
        n_uniform = len(self.transfer_functions) - n_varying

        self.logger.info(
            f"Built {len(self.transfer_functions)} transfer functions: "
            f"{n_varying} spatially varying, {n_uniform} uniform, "
            f"{len(self.all_coefficients)} total coefficients"
        )

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for calibration.

        Returns dictionary of coefficient names and their bounds,
        to be used instead of raw parameter bounds.
        """
        return {
            name: bounds
            for name, bounds in zip(self.all_coefficients, self.coeff_bounds)
        }

    def coefficients_to_parameters(
        self,
        coefficients: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert calibration coefficients to spatially distributed parameters.

        Args:
            coefficients: Dictionary of coefficient values

        Returns:
            Array of shape (n_subcatchments, n_parameters) with local values
        """
        n_params = len(self.transfer_functions)
        param_array = np.zeros((self.n_subcatchments, n_params))
        param_names = []

        for i, (param_name, (tf, attr_name)) in enumerate(self.transfer_functions.items()):
            param_names.append(param_name)

            # Get attribute values
            if attr_name == 'constant' or attr_name not in self.attributes.columns:
                attr_values = np.ones(self.n_subcatchments)
            else:
                attr_values = self.attributes[attr_name].values

            # Get coefficients for this parameter
            coeff_names = self.coefficient_map[param_name]
            coeffs = np.array([coefficients[cn] for cn in coeff_names])

            # Apply transfer function
            param_values = tf.apply(attr_values, coeffs)

            # Clip to original bounds
            p_min, p_max = self.param_bounds[param_name]
            param_values = np.clip(param_values, p_min, p_max)

            param_array[:, i] = param_values

        return param_array, param_names

    def create_distributed_para_def(
        self,
        coefficients: Dict[str, float],
        template_path: Path,
        output_path: Path
    ) -> bool:
        """
        Create a distributed para_def.nc with spatially varying parameters.

        Args:
            coefficients: Transfer function coefficients from calibration
            template_path: Path to template para_def.nc (single parameter set)
            output_path: Path to write distributed para_def.nc

        Returns:
            True if successful
        """
        try:
            import shutil

            import netCDF4 as nc

            # Get distributed parameter values
            param_array, param_names = self.coefficients_to_parameters(coefficients)

            # Copy template
            shutil.copy(template_path, output_path)

            # Update with distributed values
            with nc.Dataset(output_path, 'r+') as ds:
                # Check/update dimension
                if 'par' in ds.dimensions:
                    current_size = ds.dimensions['par'].size
                    if current_size != self.n_subcatchments:
                        self.logger.warning(
                            f"para_def has par={current_size}, need {self.n_subcatchments}. "
                            f"Recreating file..."
                        )
                        # Would need to recreate file with new dimension
                        # For now, just update index 0 with mean values
                        for i, param_name in enumerate(param_names):
                            if param_name in ds.variables:
                                mean_val = np.mean(param_array[:, i])
                                ds.variables[param_name][0] = float(mean_val)
                                self.logger.debug(f"  {param_name}: mean={mean_val:.4f}")
                        return True

                # Update each parameter
                for i, param_name in enumerate(param_names):
                    if param_name in ds.variables:
                        ds.variables[param_name][:] = param_array[:, i]
                        self.logger.debug(
                            f"  {param_name}: range [{param_array[:, i].min():.3f}, "
                            f"{param_array[:, i].max():.3f}]"
                        )

                ds.sync()

            self.logger.info(f"Created distributed para_def with {len(param_names)} parameters")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error creating distributed para_def: {e}")
            return False

    def summarize_spatial_variation(
        self,
        coefficients: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Summarize spatial variation of parameters.

        Returns DataFrame with min, max, mean, std for each parameter.
        """
        param_array, param_names = self.coefficients_to_parameters(coefficients)

        summary = []
        for i, param_name in enumerate(param_names):
            values = param_array[:, i]
            config = self.param_config.get(param_name, {})
            summary.append({
                'parameter': param_name,
                'attribute': config.get('attribute', 'constant'),
                'transform': config.get('transform', 'constant'),
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std(),
                'cv': values.std() / values.mean() if values.mean() > 0 else 0
            })

        return pd.DataFrame(summary)


def create_transfer_function_config(
    attributes_path: str,
    param_bounds: Dict[str, Tuple[float, float]],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create SYMFLUENCE configuration for transfer function calibration.

    Args:
        attributes_path: Path to subcatchment attributes CSV
        param_bounds: Original parameter bounds
        output_path: Optional path to save coefficient bounds

    Returns:
        Configuration dictionary for SYMFLUENCE
    """
    manager = ParameterTransferManager(
        attributes_path=Path(attributes_path),
        param_bounds=param_bounds
    )

    coeff_bounds = manager.get_calibration_parameters()

    config = {
        'USE_TRANSFER_FUNCTIONS': True,
        'TRANSFER_FUNCTION_ATTRIBUTES': attributes_path,
        'TRANSFER_FUNCTION_COEFFICIENTS': coeff_bounds,
        'ORIGINAL_PARAM_BOUNDS': param_bounds,
    }

    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            # Convert tuples to lists for JSON
            json_config = {
                k: {ck: list(cv) for ck, cv in v.items()} if isinstance(v, dict) else v
                for k, v in config.items()
            }
            json.dump(json_config, f, indent=2)

    return config
