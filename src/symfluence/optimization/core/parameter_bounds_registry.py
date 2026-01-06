"""
Parameter Bounds Registry - Centralized parameter bounds definitions.

This module provides a single source of truth for hydrological parameter bounds
used across different models (SUMMA, FUSE, NGEN). Benefits:
- Eliminates duplication between model-specific parameter managers
- Provides consistent bounds for shared parameters (e.g., soil properties)
- Documents parameter meanings and units
- Allows easy modification of bounds without editing multiple files

Usage:
    from symfluence.optimization.core.parameter_bounds_registry import (
        ParameterBoundsRegistry, get_fuse_bounds, get_ngen_bounds
    )

    # Get all bounds for a model
    fuse_bounds = get_fuse_bounds()

    # Get specific parameter bounds
    registry = ParameterBoundsRegistry()
    mbase_bounds = registry.get_bounds('MBASE')

    # Get bounds for a list of parameters
    bounds = registry.get_bounds_for_params(['MBASE', 'MFMAX', 'maxsmc'])
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ParameterInfo:
    """Information about a hydrological parameter."""
    min: float
    max: float
    units: str = ""
    description: str = ""
    category: str = "other"


class ParameterBoundsRegistry:
    """
    Central registry for hydrological parameter bounds.

    Organizes parameters by category (snow, soil, baseflow, routing, ET)
    and provides lookups by parameter name or model type.
    """

    # ========================================================================
    # SNOW PARAMETERS
    # ========================================================================
    SNOW_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE snow parameters
        'MBASE': ParameterInfo(-5.0, 5.0, '°C', 'Base melt temperature', 'snow'),
        'MFMAX': ParameterInfo(1.0, 10.0, 'mm/(°C·day)', 'Maximum melt factor', 'snow'),
        'MFMIN': ParameterInfo(0.5, 5.0, 'mm/(°C·day)', 'Minimum melt factor', 'snow'),
        'PXTEMP': ParameterInfo(-2.0, 2.0, '°C', 'Rain-snow partition temperature', 'snow'),
        'LAPSE': ParameterInfo(3.0, 10.0, '°C/km', 'Temperature lapse rate', 'snow'),

        # NGEN snow parameters
        'rain_snow_thresh': ParameterInfo(-2.0, 2.0, '°C', 'Rain-snow temperature threshold', 'snow'),
    }

    # ========================================================================
    # SOIL PARAMETERS
    # ========================================================================
    SOIL_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE soil parameters
        'MAXWATR_1': ParameterInfo(50.0, 1000.0, 'mm', 'Maximum storage upper layer', 'soil'),
        'MAXWATR_2': ParameterInfo(100.0, 2000.0, 'mm', 'Maximum storage lower layer', 'soil'),
        'FRACTEN': ParameterInfo(0.1, 0.9, '-', 'Fraction tension storage', 'soil'),
        'PERCRTE': ParameterInfo(0.01, 100.0, 'mm/day', 'Percolation rate', 'soil'),
        'PERCEXP': ParameterInfo(1.0, 20.0, '-', 'Percolation exponent', 'soil'),

        # NGEN CFE soil parameters
        'maxsmc': ParameterInfo(0.3, 0.6, 'fraction', 'Maximum soil moisture content', 'soil'),
        'wltsmc': ParameterInfo(0.02, 0.15, 'fraction', 'Wilting point soil moisture', 'soil'),
        'satdk': ParameterInfo(1e-6, 5e-5, 'm/s', 'Saturated hydraulic conductivity', 'soil'),
        'satpsi': ParameterInfo(0.05, 0.5, 'm', 'Saturated soil potential', 'soil'),
        'bb': ParameterInfo(3.0, 12.0, '-', 'Pore size distribution index', 'soil'),
        'smcmax': ParameterInfo(0.3, 0.55, 'm³/m³', 'Maximum soil moisture', 'soil'),
        'alpha_fc': ParameterInfo(0.3, 0.8, '-', 'Field capacity coefficient', 'soil'),
        'expon': ParameterInfo(1.0, 6.0, '-', 'Exponent parameter', 'soil'),
        'mult': ParameterInfo(500.0, 2000.0, 'mm', 'Multiplier parameter', 'soil'),
        'slop': ParameterInfo(0.01, 0.5, '-', 'TOPMODEL slope parameter', 'soil'),

        # NGEN NOAH-OWP soil parameters
        'slope': ParameterInfo(0.1, 1.0, '-', 'NOAH slope parameter', 'soil'),
        'dksat': ParameterInfo(1e-7, 1e-4, 'm/s', 'NOAH saturated conductivity', 'soil'),
        'psisat': ParameterInfo(0.01, 1.0, 'm', 'NOAH saturated potential', 'soil'),
        'bexp': ParameterInfo(2.0, 14.0, '-', 'NOAH b exponent', 'soil'),
        'smcwlt': ParameterInfo(0.01, 0.3, 'm³/m³', 'NOAH wilting point', 'soil'),
        'smcref': ParameterInfo(0.1, 0.5, 'm³/m³', 'NOAH reference moisture', 'soil'),
        'noah_refdk': ParameterInfo(1e-7, 1e-3, 'm/s', 'NOAH reference conductivity', 'soil'),
        'noah_refkdt': ParameterInfo(0.5, 5.0, '-', 'NOAH reference KDT', 'soil'),
        'noah_czil': ParameterInfo(0.02, 0.2, '-', 'NOAH Zilitinkevich coefficient', 'soil'),
        'noah_z0': ParameterInfo(0.001, 1.0, 'm', 'NOAH roughness length', 'soil'),
        'noah_frzk': ParameterInfo(0.0, 10.0, '-', 'NOAH frozen ground parameter', 'soil'),
        'noah_salp': ParameterInfo(-2.0, 2.0, '-', 'NOAH shape parameter', 'soil'),
        'refkdt': ParameterInfo(0.5, 3.0, '-', 'Reference surface runoff parameter', 'soil'),
    }

    # ========================================================================
    # BASEFLOW / GROUNDWATER PARAMETERS
    # ========================================================================
    BASEFLOW_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE baseflow parameters
        'BASERTE': ParameterInfo(0.001, 1.0, 'mm/day', 'Baseflow rate', 'baseflow'),
        'QB_POWR': ParameterInfo(1.0, 10.0, '-', 'Baseflow exponent', 'baseflow'),
        'QBRATE_2A': ParameterInfo(0.001, 0.1, '1/day', 'Primary baseflow depletion', 'baseflow'),
        'QBRATE_2B': ParameterInfo(0.0001, 0.01, '1/day', 'Secondary baseflow depletion', 'baseflow'),

        # NGEN CFE groundwater parameters
        'Cgw': ParameterInfo(0.0001, 0.005, 'm/h', 'Groundwater coefficient', 'baseflow'),
        'max_gw_storage': ParameterInfo(0.01, 0.3, 'm', 'Maximum groundwater storage', 'baseflow'),
    }

    # ========================================================================
    # ROUTING PARAMETERS
    # ========================================================================
    ROUTING_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE routing parameters
        'TIMEDELAY': ParameterInfo(0.0, 10.0, 'days', 'Time delay in routing', 'routing'),

        # NGEN CFE routing parameters
        'K_lf': ParameterInfo(0.01, 0.5, '1/h', 'Lateral flow coefficient', 'routing'),
        'K_nash': ParameterInfo(0.01, 0.4, '1/h', 'Nash cascade coefficient', 'routing'),
        'Klf': ParameterInfo(0.01, 0.5, '1/h', 'Lateral flow coefficient (alias)', 'routing'),
        'Kn': ParameterInfo(0.01, 0.4, '1/h', 'Nash cascade coefficient (alias)', 'routing'),

        # mizuRoute parameters (SUMMA)
        'velo': ParameterInfo(0.1, 5.0, 'm/s', 'Flow velocity', 'routing'),
        'diff': ParameterInfo(100.0, 5000.0, 'm²/s', 'Diffusion coefficient', 'routing'),
        'mann_n': ParameterInfo(0.01, 0.1, '-', 'Manning roughness coefficient', 'routing'),
        'wscale': ParameterInfo(0.0001, 0.01, '-', 'Width scale parameter', 'routing'),
        'fshape': ParameterInfo(1.0, 5.0, '-', 'Shape parameter', 'routing'),
        'tscale': ParameterInfo(3600, 172800, 's', 'Time scale parameter', 'routing'),
    }

    # ========================================================================
    # EVAPOTRANSPIRATION PARAMETERS
    # ========================================================================
    ET_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE ET parameters
        'RTFRAC1': ParameterInfo(0.1, 0.9, '-', 'Fraction roots upper layer', 'et'),
        'RTFRAC2': ParameterInfo(0.1, 0.9, '-', 'Fraction roots lower layer', 'et'),

        # NGEN PET parameters
        'wind_speed_measurement_height_m': ParameterInfo(2.0, 10.0, 'm', 'Wind measurement height', 'et'),
        'humidity_measurement_height_m': ParameterInfo(2.0, 10.0, 'm', 'Humidity measurement height', 'et'),
        'pet_albedo': ParameterInfo(0.05, 0.5, '-', 'PET albedo', 'et'),
        'pet_z0_mom': ParameterInfo(0.001, 1.0, 'm', 'PET momentum roughness', 'et'),
        'pet_z0_heat': ParameterInfo(0.0001, 0.1, 'm', 'PET heat roughness', 'et'),
        'pet_veg_h': ParameterInfo(0.1, 30.0, 'm', 'PET vegetation height', 'et'),
        'pet_d0': ParameterInfo(0.0, 20.0, 'm', 'PET zero plane displacement', 'et'),

        # NGEN NOAH reference height
        'ZREF': ParameterInfo(2.0, 10.0, 'm', 'Reference height for measurements', 'et'),
    }

    # ========================================================================
    # DEPTH PARAMETERS (SUMMA-specific)
    # ========================================================================
    DEPTH_PARAMS: Dict[str, ParameterInfo] = {
        'total_mult': ParameterInfo(0.1, 5.0, '-', 'Total soil depth multiplier', 'depth'),
        'total_soil_depth_multiplier': ParameterInfo(0.1, 5.0, '-', 'Total soil depth multiplier (alias)', 'depth'),
        'shape_factor': ParameterInfo(0.1, 3.0, '-', 'Soil depth shape factor', 'depth'),
    }

    # ========================================================================
    # HYPE PARAMETERS
    # ========================================================================
    HYPE_PARAMS: Dict[str, ParameterInfo] = {
        # Snow parameters
        'ttmp': ParameterInfo(-3.0, 3.0, '°C', 'Snowmelt threshold temperature', 'snow'),
        'cmlt': ParameterInfo(1.0, 15.0, 'mm/°C/day', 'Snowmelt degree-day coefficient', 'snow'),
        'ttpi': ParameterInfo(0.5, 3.0, '°C', 'Temperature interval for mixed precipitation', 'snow'),
        'cmrefr': ParameterInfo(0.0, 0.5, '-', 'Snow refreeze capacity', 'snow'),

        # Evapotranspiration parameters
        'cevp': ParameterInfo(0.0, 1.0, '-', 'Evaporation coefficient', 'et'),
        'lp': ParameterInfo(0.3, 1.0, '-', 'Threshold for ET reduction', 'et'),
        'epotdist': ParameterInfo(1.0, 10.0, '-', 'PET depth dependency coefficient', 'et'),

        # Soil hydraulic parameters
        'rrcs1': ParameterInfo(0.01, 1.0, '1/day', 'Recession coefficient upper layer', 'soil'),
        'rrcs2': ParameterInfo(0.001, 0.5, '1/day', 'Recession coefficient lower layer', 'soil'),
        'rrcs3': ParameterInfo(0.0, 0.2, '1/°', 'Recession slope dependence', 'soil'),
        'wcwp': ParameterInfo(0.01, 0.3, '-', 'Wilting point water content', 'soil'),
        'wcfc': ParameterInfo(0.1, 0.5, '-', 'Field capacity', 'soil'),
        'wcep': ParameterInfo(0.1, 0.6, '-', 'Effective porosity', 'soil'),
        'srrcs': ParameterInfo(0.0, 0.5, '1/day', 'Surface runoff coefficient', 'soil'),

        # Routing parameters
        'rivvel': ParameterInfo(0.5, 20.0, 'm/s', 'River flow velocity', 'routing'),
        'damp': ParameterInfo(0.0, 1.0, '-', 'River damping fraction', 'routing'),
        'qmean': ParameterInfo(50.0, 500.0, 'mm/yr', 'Initial mean flow', 'routing'),

        # Lake parameters
        'ilratk': ParameterInfo(0.1, 500.0, '-', 'Internal lake rating curve coefficient', 'routing'),
        'ilratp': ParameterInfo(1.0, 10.0, '-', 'Internal lake rating curve exponent', 'routing'),
    }

    # ========================================================================
    # MESH PARAMETERS
    # ========================================================================
    MESH_PARAMS: Dict[str, ParameterInfo] = {
        # CLASS land surface parameters
        'ZSNL': ParameterInfo(0.001, 0.1, 'm', 'Limiting snow depth', 'snow'),
        'ZPLG': ParameterInfo(0.0, 0.5, 'm', 'Maximum ponding depth (ground)', 'soil'),
        'ZPLS': ParameterInfo(0.0, 0.5, 'm', 'Maximum ponding depth (snow)', 'snow'),
        'FRZTH': ParameterInfo(0.0, 5.0, 'm', 'Frozen soil infiltration threshold', 'soil'),
        'MANN': ParameterInfo(0.01, 0.3, '-', 'Manning roughness coefficient', 'routing'),

        # Hydrology parameters
        'RCHARG': ParameterInfo(0.0, 1.0, '-', 'Recharge fraction to groundwater', 'baseflow'),
        'DRAINFRAC': ParameterInfo(0.0, 1.0, '-', 'Drainage fraction', 'soil'),
        'BASEFLW': ParameterInfo(0.001, 0.1, 'm/day', 'Baseflow rate', 'baseflow'),

        # Routing parameters
        'DTMINUSR': ParameterInfo(60.0, 600.0, 's', 'Routing time-step', 'routing'),
    }

    # ========================================================================
    # GR PARAMETERS
    # ========================================================================
    GR_PARAMS: Dict[str, ParameterInfo] = {
        # GR4J parameters
        'X1': ParameterInfo(1.0, 2000.0, 'mm', 'Production store capacity', 'soil'),
        'X2': ParameterInfo(-10.0, 10.0, 'mm/day', 'Groundwater exchange coefficient', 'baseflow'),
        'X3': ParameterInfo(1.0, 300.0, 'mm', 'Routing store capacity', 'soil'),
        'X4': ParameterInfo(0.5, 4.0, 'days', 'Unit hydrograph time constant', 'routing'),

        # CemaNeige parameters
        'CTG': ParameterInfo(0.0, 1.0, '-', 'Snow process parameter', 'snow'),
        'Kf': ParameterInfo(0.0, 10.0, 'mm/°C/day', 'Melt factor', 'snow'),
        'Gratio': ParameterInfo(0.01, 0.5, '-', 'Thermal coefficient of snow pack', 'snow'),
        'Albedo_diff': ParameterInfo(0.1, 1.0, '-', 'Albedo diffusion coefficient', 'snow'),
    }

    def __init__(self):
        """Initialize registry with all parameter categories combined."""
        self._all_params: Dict[str, ParameterInfo] = {}
        self._all_params.update(self.SNOW_PARAMS)
        self._all_params.update(self.SOIL_PARAMS)
        self._all_params.update(self.BASEFLOW_PARAMS)
        self._all_params.update(self.ROUTING_PARAMS)
        self._all_params.update(self.ET_PARAMS)
        self._all_params.update(self.DEPTH_PARAMS)
        self._all_params.update(self.HYPE_PARAMS)
        self._all_params.update(self.MESH_PARAMS)
        self._all_params.update(self.GR_PARAMS)

    def get_bounds(self, param_name: str) -> Optional[Dict[str, float]]:
        """
        Get bounds for a single parameter.

        Args:
            param_name: Parameter name

        Returns:
            Dictionary with 'min' and 'max' keys, or None if not found
        """
        info = self._all_params.get(param_name)
        if info:
            return {'min': info.min, 'max': info.max}
        return None

    def get_info(self, param_name: str) -> Optional[ParameterInfo]:
        """
        Get full parameter info including description and units.

        Args:
            param_name: Parameter name

        Returns:
            ParameterInfo object or None if not found
        """
        return self._all_params.get(param_name)

    def get_bounds_for_params(self, param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get bounds for multiple parameters.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping param_name -> {'min': float, 'max': float}
        """
        bounds = {}
        for name in param_names:
            b = self.get_bounds(name)
            if b:
                bounds[name] = b
        return bounds

    def get_params_by_category(self, category: str) -> Dict[str, Dict[str, float]]:
        """
        Get all parameter bounds for a category.

        Args:
            category: One of 'snow', 'soil', 'baseflow', 'routing', 'et', 'depth'

        Returns:
            Dictionary of parameter bounds
        """
        return {
            name: {'min': info.min, 'max': info.max}
            for name, info in self._all_params.items()
            if info.category == category
        }

    @property
    def all_param_names(self) -> List[str]:
        """Get list of all registered parameter names."""
        return list(self._all_params.keys())


# ============================================================================
# CONVENIENCE FUNCTIONS FOR MODEL-SPECIFIC BOUNDS
# ============================================================================

# Singleton registry instance
_registry: Optional[ParameterBoundsRegistry] = None


def get_registry() -> ParameterBoundsRegistry:
    """Get singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ParameterBoundsRegistry()
    return _registry


def get_fuse_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all FUSE parameter bounds.

    Returns:
        Dictionary mapping FUSE param_name -> {'min': float, 'max': float}
    """
    fuse_params = [
        # Snow
        'MBASE', 'MFMAX', 'MFMIN', 'PXTEMP', 'LAPSE',
        # Soil
        'MAXWATR_1', 'MAXWATR_2', 'FRACTEN', 'PERCRTE', 'PERCEXP',
        # Baseflow
        'BASERTE', 'QB_POWR', 'QBRATE_2A', 'QBRATE_2B',
        # Routing
        'TIMEDELAY',
        # ET
        'RTFRAC1', 'RTFRAC2',
    ]
    return get_registry().get_bounds_for_params(fuse_params)


def get_ngen_cfe_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get CFE module parameter bounds.

    Returns:
        Dictionary mapping CFE param_name -> {'min': float, 'max': float}
    """
    cfe_params = [
        'maxsmc', 'wltsmc', 'satdk', 'satpsi', 'bb', 'mult', 'slop',
        'smcmax', 'alpha_fc', 'expon', 'K_lf', 'K_nash', 'Klf', 'Kn',
        'Cgw', 'max_gw_storage', 'refkdt',
    ]
    return get_registry().get_bounds_for_params(cfe_params)


def get_ngen_noah_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get NOAH-OWP module parameter bounds.

    Returns:
        Dictionary mapping NOAH param_name -> {'min': float, 'max': float}
    """
    noah_params = [
        'slope', 'dksat', 'psisat', 'bexp', 'smcwlt', 'smcref',
        'noah_refdk', 'noah_refkdt', 'noah_czil', 'noah_z0',
        'noah_frzk', 'noah_salp', 'rain_snow_thresh', 'ZREF',
    ]
    return get_registry().get_bounds_for_params(noah_params)


def get_ngen_pet_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get PET module parameter bounds.

    Returns:
        Dictionary mapping PET param_name -> {'min': float, 'max': float}
    """
    pet_params = [
        'wind_speed_measurement_height_m', 'humidity_measurement_height_m',
        'pet_albedo', 'pet_z0_mom', 'pet_z0_heat', 'pet_veg_h', 'pet_d0',
    ]
    return get_registry().get_bounds_for_params(pet_params)


def get_ngen_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all NGEN parameter bounds (CFE + NOAH + PET).

    Returns:
        Dictionary mapping param_name -> {'min': float, 'max': float}
    """
    bounds = {}
    bounds.update(get_ngen_cfe_bounds())
    bounds.update(get_ngen_noah_bounds())
    bounds.update(get_ngen_pet_bounds())
    return bounds


def get_mizuroute_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get mizuRoute parameter bounds.

    Returns:
        Dictionary mapping param_name -> {'min': float, 'max': float}
    """
    mizu_params = ['velo', 'diff', 'mann_n', 'wscale', 'fshape', 'tscale']
    return get_registry().get_bounds_for_params(mizu_params)


def get_depth_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get soil depth calibration parameter bounds.

    Returns:
        Dictionary mapping param_name -> {'min': float, 'max': float}
    """
    depth_params = ['total_mult', 'total_soil_depth_multiplier', 'shape_factor']
    return get_registry().get_bounds_for_params(depth_params)


def get_hype_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all HYPE parameter bounds.

    Returns:
        Dictionary mapping HYPE param_name -> {'min': float, 'max': float}
    """
    hype_params = [
        'ttmp', 'cmlt', 'ttpi', 'cmrefr',  # Snow
        'cevp', 'lp', 'epotdist',  # ET
        'rrcs1', 'rrcs2', 'rrcs3', 'wcwp', 'wcfc', 'wcep', 'srrcs',  # Soil
        'rivvel', 'damp', 'qmean',  # Routing
        'ilratk', 'ilratp',  # Lakes
    ]
    return get_registry().get_bounds_for_params(hype_params)


def get_mesh_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all MESH parameter bounds.

    Returns:
        Dictionary mapping MESH param_name -> {'min': float, 'max': float}
    """
    mesh_params = [
        'ZSNL', 'ZPLG', 'ZPLS', 'FRZTH', 'MANN',  # CLASS
        'RCHARG', 'DRAINFRAC', 'BASEFLW',  # Hydrology
        'DTMINUSR',  # Routing
    ]
    return get_registry().get_bounds_for_params(mesh_params)


def get_gr_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all GR parameter bounds.

    Returns:
        Dictionary mapping GR param_name -> {'min': float, 'max': float}
    """
    gr_params = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf', 'Gratio', 'Albedo_diff']
    return get_registry().get_bounds_for_params(gr_params)
