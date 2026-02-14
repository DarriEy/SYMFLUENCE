"""
Parameter Regionalization for Distributed Hydrological Models.

Provides multiple strategies for handling spatially distributed parameters:
- lumped: Single parameter set applied uniformly across all subcatchments
- transfer_function: Parameters derived from catchment attributes via power-law functions
- distributed: Independent parameters for each subcatchment (requires regularization)
- zones: Group subcatchments into zones with shared parameters

Configuration:
    PARAMETER_REGIONALIZATION: lumped | transfer_function | zones | distributed
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
import pandas as pd


class ParameterRegionalization(ABC):
    """Abstract base class for parameter regionalization strategies."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_subcatchments: int,
        logger: Optional[logging.Logger] = None
    ):
        self.param_bounds = param_bounds
        self.n_subcatchments = n_subcatchments
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the parameters/coefficients to be calibrated.

        Returns:
            Dictionary of {param_name: (min_bound, max_bound)}
        """
        pass

    @abstractmethod
    def to_distributed(
        self,
        calibration_params: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert calibration parameters to distributed parameter values.

        Args:
            calibration_params: Values from the optimizer

        Returns:
            Tuple of (param_array [n_subcatchments, n_params], param_names)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass


class LumpedRegionalization(ParameterRegionalization):
    """
    Lumped parameter regionalization.

    All subcatchments share the same parameter values.
    This is the simplest approach with fewest degrees of freedom.
    """

    @property
    def name(self) -> str:
        return "lumped"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Return original parameter bounds - one value per parameter."""
        return self.param_bounds.copy()

    def to_distributed(
        self,
        calibration_params: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """Replicate single values across all subcatchments."""
        param_names = list(calibration_params.keys())
        n_params = len(param_names)

        param_array = np.zeros((self.n_subcatchments, n_params))
        for i, name in enumerate(param_names):
            param_array[:, i] = calibration_params[name]

        return param_array, param_names


class TransferFunctionRegionalization(ParameterRegionalization):
    """
    Transfer function parameter regionalization (MPR-style).

    Standard approach using linear transfer functions with log-transformed attributes:
        param = a + b * attr_norm

    Where:
        - 'a' is the base value (intercept)
        - 'b' is the slope (sensitivity to attribute)
        - attr_norm is the normalized (and optionally log-transformed) attribute

    For skewed attributes (precipitation, etc.), log-transform is applied before
    normalization to compress the range and improve numerical stability.

    Set calibrate_b=False to make parameter spatially uniform (b=0).
    """

    # Attributes that should be log-transformed before normalization
    # (skewed distributions benefit from log-transform)
    LOG_TRANSFORM_ATTRS = {'precip_mm_yr', 'aridity'}

    # Default configuration for which parameters vary spatially
    DEFAULT_PARAM_CONFIG = {
        'MAXWATR_1': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
        'MAXWATR_2': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
        'FRACTEN': {'attribute': 'aridity', 'calibrate_b': False},
        'BASERTE': {'attribute': 'aridity', 'calibrate_b': True},
        'QB_POWR': {'attribute': 'aridity', 'calibrate_b': True},
        'PERCRTE': {'attribute': 'aridity', 'calibrate_b': True},
        'TIMEDELAY': {'attribute': 'precip_mm_yr', 'calibrate_b': False},
        'RTFRAC1': {'attribute': 'aridity', 'calibrate_b': False},
        'MBASE': {'attribute': 'elev_m', 'calibrate_b': True},
        'MFMAX': {'attribute': 'temp_C', 'calibrate_b': True},
        'MFMIN': {'attribute': 'snow_frac', 'calibrate_b': True},
        'PXTEMP': {'attribute': 'elev_m', 'calibrate_b': True},
        'LAPSE': {'attribute': 'elev_m', 'calibrate_b': True},
    }

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_subcatchments: int,
        attributes: pd.DataFrame,
        param_config: Optional[Dict[str, Dict]] = None,
        b_bounds: Tuple[float, float] = (-1.5, 1.5),
        transfer_function_type: str = 'linear',
        log_transform_attrs: Optional[set] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(param_bounds, n_subcatchments, logger)
        self.attributes = attributes.copy()  # Don't modify original
        self.param_config = param_config or self.DEFAULT_PARAM_CONFIG
        self.b_bounds = b_bounds
        self.transfer_function_type = transfer_function_type
        self.log_transform_attrs = log_transform_attrs or self.LOG_TRANSFORM_ATTRS

        # Normalize attributes to [0, 1] with optional log-transform
        self._normalize_attributes()

        # Build coefficient mapping
        self._build_coefficient_map()

    def _normalize_attributes(self):
        """
        Normalize numeric attributes to [0, 1] range.

        For skewed attributes (in LOG_TRANSFORM_ATTRS), applies log-transform
        before normalization to compress the range and improve stability.
        """
        self.attr_stats = {}

        for col in ['elev_m', 'precip_mm_yr', 'temp_C', 'aridity', 'snow_frac']:
            if col not in self.attributes.columns:
                continue

            values = self.attributes[col].values.copy()

            # Apply log-transform for skewed attributes
            if col in self.log_transform_attrs:
                # log(1 + x) to handle zeros safely
                values = np.log1p(np.maximum(values, 0))
                transform = 'log1p'
            else:
                transform = 'none'

            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val

            self.attr_stats[col] = {
                'min': min_val,
                'max': max_val,
                'transform': transform
            }

            if range_val > 0:
                self.attributes[f'{col}_norm'] = (values - min_val) / range_val
            else:
                self.attributes[f'{col}_norm'] = 0.5

    def _build_coefficient_map(self):
        """Build mapping from coefficients to parameters."""
        self.coeff_to_param = {}  # {coeff_name: (param_name, is_slope)}
        self.param_to_coeffs = {}  # {param_name: [coeff_names]}
        self.param_to_attr = {}  # {param_name: attr_name}

        for param_name, config in self.param_config.items():
            if param_name not in self.param_bounds:
                continue

            attr = config.get('attribute', 'precip_mm_yr')
            calibrate_b = config.get('calibrate_b', False)

            # Use normalized attribute
            attr_norm = f'{attr}_norm'
            if attr_norm in self.attributes.columns:
                self.param_to_attr[param_name] = attr_norm
            else:
                self.param_to_attr[param_name] = attr

            coeff_names = [f'{param_name}_a']
            self.coeff_to_param[f'{param_name}_a'] = (param_name, False)

            if calibrate_b:
                coeff_names.append(f'{param_name}_b')
                self.coeff_to_param[f'{param_name}_b'] = (param_name, True)

            self.param_to_coeffs[param_name] = coeff_names

    @property
    def name(self) -> str:
        return "transfer_function"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Return coefficient bounds for calibration.

        For linear transfer function (param = a + b * attr_norm):
        - 'a' bounds: parameter bounds (base value when attr_norm=0)
        - 'b' bounds: slope bounds (change from attr_norm=0 to attr_norm=1)

        With attr_norm in [0,1], the parameter range is [a, a+b].
        To allow full parameter range, b_bounds should be [-(p_max-p_min), +(p_max-p_min)].
        """
        bounds = {}

        for param_name, coeff_names in self.param_to_coeffs.items():
            p_min, p_max = self.param_bounds[param_name]
            p_range = p_max - p_min

            for coeff_name in coeff_names:
                if coeff_name.endswith('_a'):
                    # Base value: allow full parameter range
                    bounds[coeff_name] = (p_min, p_max)
                elif coeff_name.endswith('_b'):
                    # Slope: scale by parameter range for physically meaningful bounds
                    # b_bounds are relative multipliers of parameter range
                    b_min = self.b_bounds[0] * p_range
                    b_max = self.b_bounds[1] * p_range
                    bounds[coeff_name] = (b_min, b_max)

        return bounds

    def to_distributed(
        self,
        calibration_params: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert coefficients to distributed parameters using linear transfer function.

        Linear form: param = a + b * attr_norm

        Where attr_norm is in [0, 1] (with optional log-transform applied).
        Result is clipped to original parameter bounds.
        """
        param_names = list(self.param_to_coeffs.keys())
        n_params = len(param_names)

        param_array = np.zeros((self.n_subcatchments, n_params))

        for i, param_name in enumerate(param_names):
            attr_name = self.param_to_attr[param_name]

            # Get coefficient values
            a = calibration_params.get(f'{param_name}_a', 1.0)
            b = calibration_params.get(f'{param_name}_b', 0.0)

            # Get normalized attribute values [0, 1]
            if attr_name in self.attributes.columns:
                attr_vals = self.attributes[attr_name].values
            else:
                attr_vals = np.full(self.n_subcatchments, 0.5)

            # Apply linear transfer function: param = a + b * attr_norm
            values = a + b * attr_vals

            # Clip to original bounds
            p_min, p_max = self.param_bounds[param_name]
            values = np.clip(values, p_min, p_max)

            param_array[:, i] = values

        return param_array, param_names


class ZoneRegionalization(ParameterRegionalization):
    """
    Zone-based parameter regionalization.

    Subcatchments are grouped into zones (e.g., by elevation, climate, geology).
    Each zone has its own parameter set.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_subcatchments: int,
        zone_assignments: np.ndarray,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(param_bounds, n_subcatchments, logger)
        self.zone_assignments = zone_assignments
        self.n_zones = len(np.unique(zone_assignments))
        self.logger.info(f"Zone regionalization: {self.n_zones} zones")

    @property
    def name(self) -> str:
        return "zones"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for each parameter in each zone."""
        bounds = {}

        for param_name, (p_min, p_max) in self.param_bounds.items():
            for zone in range(self.n_zones):
                bounds[f'{param_name}_z{zone}'] = (p_min, p_max)

        return bounds

    def to_distributed(
        self,
        calibration_params: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """Map zone parameters to subcatchments."""
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)

        param_array = np.zeros((self.n_subcatchments, n_params))

        for i, param_name in enumerate(param_names):
            for zone in range(self.n_zones):
                coeff_name = f'{param_name}_z{zone}'
                value = calibration_params.get(coeff_name, 0.0)

                # Assign to subcatchments in this zone
                mask = self.zone_assignments == zone
                param_array[mask, i] = value

        return param_array, param_names


class DistributedRegionalization(ParameterRegionalization):
    """
    Fully distributed parameter regionalization.

    Each subcatchment has independent parameters.
    WARNING: This leads to many parameters and requires strong regularization.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_subcatchments: int,
        regularization: str = 'spatial_smoothing',
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(param_bounds, n_subcatchments, logger)
        self.regularization = regularization

        n_params = len(param_bounds) * n_subcatchments
        self.logger.warning(
            f"Distributed regionalization: {n_params} parameters! "
            f"Consider using transfer_function or zones instead."
        )

    @property
    def name(self) -> str:
        return "distributed"

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for each parameter at each subcatchment."""
        bounds = {}

        for param_name, (p_min, p_max) in self.param_bounds.items():
            for sub in range(self.n_subcatchments):
                bounds[f'{param_name}_s{sub}'] = (p_min, p_max)

        return bounds

    def to_distributed(
        self,
        calibration_params: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        """Direct mapping - parameters are already distributed."""
        param_names = list(self.param_bounds.keys())
        n_params = len(param_names)

        param_array = np.zeros((self.n_subcatchments, n_params))

        for i, param_name in enumerate(param_names):
            for sub in range(self.n_subcatchments):
                coeff_name = f'{param_name}_s{sub}'
                param_array[sub, i] = calibration_params.get(coeff_name, 0.0)

        return param_array, param_names


class RegionalizationFactory:
    """Factory for creating parameter regionalization strategies."""

    @staticmethod
    def create(
        method: str,
        param_bounds: Dict[str, Tuple[float, float]],
        n_subcatchments: int,
        config: Optional[Dict[str, Any]] = None,
        attributes: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None
    ) -> ParameterRegionalization:
        """
        Create a parameter regionalization strategy.

        Args:
            method: One of 'lumped', 'transfer_function', 'zones', 'distributed'
            param_bounds: Original parameter bounds
            n_subcatchments: Number of subcatchments
            config: Additional configuration options
            attributes: Subcatchment attributes DataFrame (for transfer_function)
            logger: Logger instance

        Returns:
            ParameterRegionalization instance
        """
        config = config or {}
        logger = logger or logging.getLogger(__name__)

        method = method.lower().replace('-', '_')

        if method == 'lumped':
            return LumpedRegionalization(
                param_bounds=param_bounds,
                n_subcatchments=n_subcatchments,
                logger=logger
            )

        elif method == 'transfer_function':
            if attributes is None:
                raise ValueError(
                    "transfer_function regionalization requires 'attributes' DataFrame"
                )
            return TransferFunctionRegionalization(
                param_bounds=param_bounds,
                n_subcatchments=n_subcatchments,
                attributes=attributes,
                param_config=config.get('TRANSFER_FUNCTION_PARAM_CONFIG'),
                b_bounds=config.get('TRANSFER_FUNCTION_B_BOUNDS', (-1.0, 1.0)),
                transfer_function_type=config.get('TRANSFER_FUNCTION_TYPE', 'linear'),
                log_transform_attrs=config.get('TRANSFER_FUNCTION_LOG_ATTRS'),
                logger=logger
            )

        elif method == 'zones':
            zone_assignments = config.get('zone_assignments')
            if zone_assignments is None:
                raise ValueError(
                    "zones regionalization requires 'zone_assignments' in config"
                )
            return ZoneRegionalization(
                param_bounds=param_bounds,
                n_subcatchments=n_subcatchments,
                zone_assignments=zone_assignments,
                logger=logger
            )

        elif method == 'distributed':
            return DistributedRegionalization(
                param_bounds=param_bounds,
                n_subcatchments=n_subcatchments,
                regularization=config.get('regularization', 'spatial_smoothing'),
                logger=logger
            )

        else:
            raise ValueError(
                f"Unknown regionalization method: {method}. "
                f"Choose from: lumped, transfer_function, zones, distributed"
            )


def get_regionalization_info() -> Dict[str, str]:
    """Get descriptions of available regionalization methods."""
    return {
        'lumped': (
            "Single parameter set for all subcatchments. "
            "Simplest approach, fewest parameters to calibrate."
        ),
        'transfer_function': (
            "MPR-style transfer functions: param = a + b * attr_norm. "
            "Log-transforms skewed attributes, normalizes to [0,1], then applies linear function. "
            "Calibrates coefficients (a=intercept, b=slope) that map attributes to parameters."
        ),
        'zones': (
            "Subcatchments grouped into zones with shared parameters. "
            "Moderate complexity, requires zone definition."
        ),
        'distributed': (
            "Independent parameters for each subcatchment. "
            "Most flexible but requires many parameters and regularization."
        ),
    }
