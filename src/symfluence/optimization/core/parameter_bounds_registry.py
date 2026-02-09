"""
Parameter Bounds Registry - Centralized parameter bounds definitions.

This module provides a single source of truth for hydrological parameter bounds
used across different models (SUMMA, FUSE, NGEN). Benefits:
- Eliminates duplication between model-specific parameter managers
- Provides consistent bounds for shared parameters (e.g., soil properties)
- Documents parameter meanings and units
- Allows easy modification of bounds without editing multiple files

Architecture Decision:
    This module intentionally contains model-specific functions (get_fuse_bounds,
    get_ngen_bounds, etc.) despite the general pattern of moving model-specific
    code to respective model packages (models/fuse/, models/ngen/, etc.).

    Rationale for centralization:
    - Single source of truth: All parameter bounds in one place for easy comparison
    - Cross-model consistency: Ensures shared parameters use consistent bounds
    - Easier maintenance: Modifying bounds doesn't require editing 11 model packages
    - Better overview: Developers can see all parameter bounds at a glance
    - Scientific documentation: Bounds are documented with units and descriptions

    Alternative considered:
    - Splitting bounds into models/{model}/calibration/parameter_bounds.py
    - Rejected due to increased fragmentation and harder cross-model validation

    Decision affirmed during pre-migration refactoring (January 2026) as part of
    the effort to consolidate model-specific code before the main migration.

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
    """Information about a hydrological parameter.

    Attributes:
        min: Minimum bound for the parameter.
        max: Maximum bound for the parameter.
        units: Physical units string (e.g., 'm/day', '°C').
        description: Human-readable description of the parameter.
        category: Parameter category ('snow', 'soil', 'baseflow', etc.).
        transform: Normalization transform type. 'linear' (default) maps
            uniformly between min and max. 'log' maps uniformly in
            log-space, which is appropriate for parameters spanning
            multiple orders of magnitude (e.g., conductivities, loss
            coefficients). Log transform requires min > 0.
    """
    min: float
    max: float
    units: str = ""
    description: str = ""
    category: str = "other"
    transform: str = "linear"


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
        'satdk': ParameterInfo(1e-7, 1e-4, 'm/s', 'Saturated hydraulic conductivity (expanded bounds)', 'soil', 'log'),
        'satpsi': ParameterInfo(0.05, 0.5, 'm', 'Saturated soil potential', 'soil'),
        'bb': ParameterInfo(2.0, 8.0, '-', 'Pore size distribution index', 'soil'),
        # Note: smcmax defined in NOAH section below with bounds (0.3, 0.6)
        'alpha_fc': ParameterInfo(0.3, 0.8, '-', 'Field capacity coefficient', 'soil'),
        'expon': ParameterInfo(1.0, 6.0, '-', 'Exponent parameter', 'soil'),
        'mult': ParameterInfo(500.0, 2000.0, 'mm', 'Multiplier parameter', 'soil'),
        'slop': ParameterInfo(0.01, 0.5, '-', 'TOPMODEL slope parameter', 'soil'),
        'soil_depth': ParameterInfo(1.0, 10.0, 'm', 'CFE soil depth for mountainous catchments', 'soil'),

        # NGEN NOAH-OWP soil parameters
        'slope': ParameterInfo(0.1, 1.0, '-', 'NOAH slope parameter', 'soil'),
        'dksat': ParameterInfo(1e-7, 1e-4, 'm/s', 'NOAH saturated conductivity', 'soil', 'log'),
        'psisat': ParameterInfo(0.01, 1.0, 'm', 'NOAH saturated potential', 'soil'),
        'bexp': ParameterInfo(2.0, 14.0, '-', 'NOAH b exponent', 'soil'),
        'smcmax': ParameterInfo(0.3, 0.6, 'm³/m³', 'NOAH maximum soil moisture (should match CFE)', 'soil'),
        'smcwlt': ParameterInfo(0.01, 0.3, 'm³/m³', 'NOAH wilting point', 'soil'),
        'smcref': ParameterInfo(0.1, 0.5, 'm³/m³', 'NOAH reference moisture', 'soil'),
        'noah_refdk': ParameterInfo(1e-7, 1e-3, 'm/s', 'NOAH reference conductivity', 'soil', 'log'),
        'noah_refkdt': ParameterInfo(0.5, 5.0, '-', 'NOAH reference KDT', 'soil'),
        'noah_czil': ParameterInfo(0.02, 0.2, '-', 'NOAH Zilitinkevich coefficient', 'soil'),
        'noah_z0': ParameterInfo(0.001, 1.0, 'm', 'NOAH roughness length', 'soil'),
        'noah_frzk': ParameterInfo(0.0, 10.0, '-', 'NOAH frozen ground parameter', 'soil'),
        'noah_salp': ParameterInfo(-2.0, 2.0, '-', 'NOAH shape parameter', 'soil'),
        'refkdt': ParameterInfo(0.5, 5.0, '-', 'Reference surface runoff parameter (expanded for infiltration control)', 'soil'),
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
        'Cgw': ParameterInfo(1e-5, 0.01, 'm/h', 'Groundwater coefficient', 'baseflow', 'log'),
        'max_gw_storage': ParameterInfo(0.05, 2.0, 'm', 'Maximum groundwater storage (expanded for mountain catchments)', 'baseflow'),
    }

    # ========================================================================
    # ROUTING PARAMETERS
    # ========================================================================
    ROUTING_PARAMS: Dict[str, ParameterInfo] = {
        # FUSE routing parameters
        'TIMEDELAY': ParameterInfo(0.0, 10.0, 'days', 'Time delay in routing', 'routing'),

        # NGEN CFE routing parameters
        'K_lf': ParameterInfo(0.01, 0.8, '1/h', 'Lateral flow coefficient (expanded upper bound)', 'routing'),
        'K_nash': ParameterInfo(0.01, 0.8, '1/h', 'Nash cascade coefficient (expanded upper bound)', 'routing'),
        'Klf': ParameterInfo(0.01, 0.8, '1/h', 'Lateral flow coefficient (alias)', 'routing'),
        'Kn': ParameterInfo(0.01, 0.8, '1/h', 'Nash cascade coefficient (alias)', 'routing'),

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

        # NGEN PET parameters (BMI config file key names)
        'vegetation_height_m': ParameterInfo(0.1, 30.0, 'm', 'Vegetation height', 'et'),
        'zero_plane_displacement_height_m': ParameterInfo(0.0, 20.0, 'm', 'Zero plane displacement height', 'et'),
        'momentum_transfer_roughness_length': ParameterInfo(0.001, 1.0, 'm', 'Momentum transfer roughness length', 'et'),
        'heat_transfer_roughness_length_m': ParameterInfo(0.0001, 0.1, 'm', 'Heat transfer roughness length', 'et'),
        'surface_shortwave_albedo': ParameterInfo(0.05, 0.5, '-', 'Surface shortwave albedo', 'et'),
        'surface_longwave_emissivity': ParameterInfo(0.9, 1.0, '-', 'Surface longwave emissivity', 'et'),
        'wind_speed_measurement_height_m': ParameterInfo(2.0, 10.0, 'm', 'Wind measurement height', 'et'),
        'humidity_measurement_height_m': ParameterInfo(2.0, 10.0, 'm', 'Humidity measurement height', 'et'),

        # NGEN PET parameters (legacy/alias names)
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
        # ==== SNOW PARAMETERS ====
        # Threshold temperature for snowmelt - critical for timing of spring melt
        'ttmp': ParameterInfo(-5.0, 5.0, '°C', 'Snowmelt threshold temperature', 'snow'),
        # Degree-day melt factor - controls snowmelt rate; expanded for alpine basins
        'cmlt': ParameterInfo(0.5, 20.0, 'mm/°C/day', 'Snowmelt degree-day coefficient', 'snow'),
        # Temperature interval for rain/snow partition
        'ttpi': ParameterInfo(0.5, 4.0, '°C', 'Temperature interval for mixed precipitation', 'snow'),
        # Snow refreeze capacity (fraction of melt factor)
        'cmrefr': ParameterInfo(0.0, 0.5, '-', 'Snow refreeze capacity', 'snow'),
        # Fresh snow density - affects snow accumulation and SWE
        'sdnsnew': ParameterInfo(0.05, 0.25, 'kg/dm³', 'Fresh snow density', 'snow'),
        # Snow densification rate
        'snowdensdt': ParameterInfo(0.0005, 0.005, '1/day', 'Snow densification parameter', 'snow'),
        # Fractional snow cover efficiency for reducing melt/evap
        'fsceff': ParameterInfo(0.5, 1.0, '-', 'Fractional snow cover efficiency', 'snow'),

        # ==== EVAPOTRANSPIRATION PARAMETERS ====
        # ET coefficient - CRITICAL: expanded to allow higher ET for water balance
        'cevp': ParameterInfo(0.1, 2.0, '-', 'Evapotranspiration coefficient (expanded for alpine)', 'et'),
        # Soil moisture threshold for ET reduction
        'lp': ParameterInfo(0.3, 1.0, '-', 'Threshold for ET reduction', 'et'),
        # PET depth dependency - controls root water uptake distribution
        'epotdist': ParameterInfo(1.0, 15.0, '-', 'PET depth dependency coefficient', 'et'),
        # Fraction of PET used for snow sublimation - important in alpine/cold regions
        'fepotsnow': ParameterInfo(0.0, 1.0, '-', 'Fraction of PET for snow sublimation', 'et'),
        # Soil temperature threshold for transpiration
        'ttrig': ParameterInfo(-5.0, 5.0, '°C', 'Soil temperature threshold for transpiration', 'et'),
        # Soil temperature response function coefficients
        'treda': ParameterInfo(0.5, 1.0, '-', 'Soil temp response coefficient A', 'et'),
        'tredb': ParameterInfo(0.1, 0.8, '-', 'Soil temp response coefficient B', 'et'),

        # ==== SOIL HYDRAULIC PARAMETERS ====
        # Recession coefficient upper soil layer - controls fast response
        'rrcs1': ParameterInfo(0.001, 1.0, '1/day', 'Recession coefficient upper layer', 'soil'),
        # Recession coefficient lower soil layer - controls slow response
        'rrcs2': ParameterInfo(0.0001, 0.5, '1/day', 'Recession coefficient lower layer', 'soil'),
        # Recession slope dependence
        'rrcs3': ParameterInfo(0.0, 0.3, '1/°', 'Recession slope dependence', 'soil'),
        # Wilting point - minimum soil water content for ET
        'wcwp': ParameterInfo(0.01, 0.3, '-', 'Wilting point water content', 'soil'),
        # Field capacity - soil water holding capacity
        'wcfc': ParameterInfo(0.1, 0.6, '-', 'Field capacity', 'soil'),
        # Effective porosity - maximum soil water storage
        'wcep': ParameterInfo(0.2, 0.7, '-', 'Effective porosity', 'soil'),
        # Surface runoff coefficient
        'srrcs': ParameterInfo(0.0, 0.5, '1/day', 'Surface runoff coefficient', 'soil'),
        # Frozen soil infiltration parameter
        'bfroznsoil': ParameterInfo(1.0, 10.0, '-', 'Frozen soil infiltration parameter', 'soil'),
        # Saturated matric potential (log scale)
        'logsatmp': ParameterInfo(0.5, 3.0, 'log(cm)', 'Saturated matric potential', 'soil'),
        # Cosby B parameter for soil water retention
        'bcosby': ParameterInfo(4.0, 15.0, '-', 'Cosby B parameter', 'soil'),
        # Frost depth parameter
        'sfrost': ParameterInfo(0.5, 3.0, 'cm/°C', 'Frost depth parameter', 'soil'),

        # ==== GROUNDWATER PARAMETERS ====
        # Regional GW recession - CRITICAL for baseflow and water balance
        'rcgrw': ParameterInfo(0.00001, 1.0, '1/day', 'Regional groundwater recession coefficient', 'baseflow'),
        # Deep groundwater loss coefficient (if model supports it)
        'deepperc': ParameterInfo(0.0, 0.5, 'mm/day', 'Deep percolation loss rate', 'baseflow'),

        # ==== SOIL TEMPERATURE PARAMETERS ====
        # Deep soil temperature memory
        'deepmem': ParameterInfo(100.0, 2000.0, 'days', 'Deep soil temperature memory', 'soil'),
        # Upper soil temperature memory
        'surfmem': ParameterInfo(5.0, 50.0, 'days', 'Upper soil temperature memory', 'soil'),
        # Depth relation for soil temp memory
        'depthrel': ParameterInfo(0.5, 3.0, '-', 'Depth relation for soil temperature', 'soil'),

        # ==== ROUTING PARAMETERS ====
        # River flow velocity
        'rivvel': ParameterInfo(0.2, 30.0, 'm/s', 'River flow velocity', 'routing'),
        # River damping fraction
        'damp': ParameterInfo(0.0, 1.0, '-', 'River damping fraction', 'routing'),
        # Initial mean flow estimate
        'qmean': ParameterInfo(10.0, 1000.0, 'mm/yr', 'Initial mean flow', 'routing'),

        # ==== LAKE PARAMETERS ====
        # Internal lake rating curve coefficient
        'ilratk': ParameterInfo(0.1, 1000.0, '-', 'Internal lake rating curve coefficient', 'routing'),
        # Internal lake rating curve exponent
        'ilratp': ParameterInfo(1.0, 10.0, '-', 'Internal lake rating curve exponent', 'routing'),
        # Internal lake depth
        'illdepth': ParameterInfo(0.1, 2.0, 'm', 'Internal lake depth', 'routing'),
    }

    # ========================================================================
    # MESH PARAMETERS
    # ========================================================================
    MESH_PARAMS: Dict[str, ParameterInfo] = {
        # CLASS.ini parameters - control runoff generation (most impactful)
        'KSAT': ParameterInfo(0.1, 500.0, 'mm/hr', 'Saturated hydraulic conductivity', 'soil', 'log'),
        'DRN': ParameterInfo(0.1, 10.0, '-', 'Drainage parameter', 'soil'),
        'SDEP': ParameterInfo(0.5, 5.0, 'm', 'Soil depth', 'soil'),
        'XSLP': ParameterInfo(0.005, 0.3, '-', 'Slope for overland flow', 'surface'),
        'XDRAINH': ParameterInfo(0.01, 1.0, '-', 'Horizontal drainage coefficient', 'soil'),
        'MANN_CLASS': ParameterInfo(0.01, 0.5, '-', 'Manning coefficient for overland flow', 'surface'),

        # CLASS.ini vegetation parameters (control ET partitioning)
        'LAMX': ParameterInfo(0.3, 6.0, 'm²/m²', 'Maximum LAI for primary vegetation class', 'et'),
        'ROOT': ParameterInfo(0.1, 2.0, 'm', 'Root depth for primary vegetation class', 'et'),
        'RSMIN': ParameterInfo(100.0, 800.0, 's/m', 'Minimum stomatal resistance (controls max transpiration rate)', 'et'),

        # Hydrology.ini parameters (snow/ponding)
        'ZSNL': ParameterInfo(0.001, 0.1, 'm', 'Limiting snow depth', 'snow'),
        'ZPLG': ParameterInfo(0.0, 0.5, 'm', 'Maximum ponding depth (ground)', 'soil'),
        'ZPLS': ParameterInfo(0.0, 0.5, 'm', 'Maximum ponding depth (snow)', 'snow'),
        'FRZTH': ParameterInfo(0.0, 5.0, 'm', 'Frozen soil infiltration threshold', 'soil'),
        'MANN': ParameterInfo(0.01, 0.3, '-', 'Manning roughness coefficient', 'routing'),
        'R2N': ParameterInfo(0.01, 0.5, '-', 'Overland routing roughness (Manning n)', 'routing'),
        'R1N': ParameterInfo(0.0, 2.0, '-', 'River routing parameter', 'routing'),

        # Baseflow parameters (hydrology.ini)
        'FLZ': ParameterInfo(0.0001, 0.1, '-', 'Baseflow recession coefficient', 'baseflow', 'log'),
        'PWR': ParameterInfo(1.0, 5.0, '-', 'Baseflow power exponent', 'baseflow'),

        # Legacy hydrology parameters
        'RCHARG': ParameterInfo(0.0, 1.0, '-', 'Recharge fraction to groundwater', 'baseflow'),
        'DRAINFRAC': ParameterInfo(0.0, 1.0, '-', 'Drainage fraction', 'soil'),
        'BASEFLW': ParameterInfo(0.001, 0.1, 'm/day', 'Baseflow rate', 'baseflow'),

        # Routing parameters (meshflow-generated files)
        'WF_R2': ParameterInfo(0.1, 0.5, '-', 'Channel roughness coefficient for WATFLOOD routing', 'routing'),
        'DTMINUSR': ParameterInfo(60.0, 600.0, 's', 'Routing time-step', 'routing'),
    }

    # ========================================================================
    # RHESSYS PARAMETERS
    # ========================================================================
    RHESSYS_PARAMS: Dict[str, ParameterInfo] = {
        # Groundwater/baseflow parameters (basin.def and soil.def)
        # Log-space transform for parameters spanning orders of magnitude
        'sat_to_gw_coeff': ParameterInfo(0.0001, 0.1, '1/day', 'Saturation to groundwater coefficient', 'baseflow', 'log'),
        'gw_loss_coeff': ParameterInfo(0.001, 0.5, '-', 'Groundwater loss coefficient (controls slow baseflow)', 'baseflow', 'log'),
        'gw_loss_fast_coeff': ParameterInfo(0.01, 1.0, '-', 'Fast groundwater loss coefficient', 'baseflow', 'log'),
        'gw_loss_fast_threshold': ParameterInfo(0.05, 0.5, 'm', 'GW storage threshold for fast flow activation', 'baseflow'),

        # Soil hydraulic parameters (soil.def)
        'psi_air_entry': ParameterInfo(-10.0, -1.0, 'kPa', 'Air entry pressure (negative)', 'soil'),
        'pore_size_index': ParameterInfo(0.05, 0.4, '-', 'Pore size distribution index', 'soil'),
        'porosity_0': ParameterInfo(0.3, 0.6, 'm³/m³', 'Surface porosity', 'soil'),
        'porosity_decay': ParameterInfo(0.1, 0.8, 'm³/m³', 'Porosity decay with depth', 'soil'),
        # Ksat_0 in m/day: 0.0001-0.1 m/day = 0.1-100 mm/day
        'Ksat_0': ParameterInfo(0.0001, 0.1, 'm/day', 'Surface saturated conductivity (lateral)', 'soil', 'log'),
        # Ksat_0_v: Should be similar magnitude to Ksat_0 (ratio typically 1-10x)
        'Ksat_0_v': ParameterInfo(0.0001, 0.5, 'm/day', 'Vertical saturated conductivity', 'soil', 'log'),
        'm': ParameterInfo(0.5, 5.0, '-', 'Lateral decay of Ksat with depth', 'soil'),
        'm_z': ParameterInfo(0.2, 3.0, '-', 'Vertical decay of Ksat with depth', 'soil'),
        'soil_depth': ParameterInfo(2.0, 15.0, 'm', 'Total soil depth', 'soil'),
        'active_zone_z': ParameterInfo(0.5, 3.0, 'm', 'Active zone depth', 'soil'),

        # Subgrid variability parameters (soil.def) — critical for lumped mode peak flows
        'theta_mean_std_p1': ParameterInfo(0.01, 0.5, '-', 'Std dev of saturation deficit (controls partial saturation area)', 'soil'),
        'theta_mean_std_p2': ParameterInfo(0.0, 0.3, '-', 'Second parameter for saturation deficit variance', 'soil'),

        # Snow parameters (soil.def for snow_melt_Tcoef, zone.def for temps)
        'max_snow_temp': ParameterInfo(-2.0, 2.0, '°C', 'Max temp for snow (rain/snow threshold)', 'snow'),
        'min_rain_temp': ParameterInfo(-6.0, 0.0, '°C', 'Min temp for rain (all snow below this)', 'snow'),
        'snow_melt_Tcoef': ParameterInfo(0.5, 8.0, 'mm/°C/day', 'Snow melt temperature coefficient', 'snow'),
        'snow_water_capacity': ParameterInfo(0.1, 1.5, '-', 'Snow water holding capacity coefficient', 'snow'),
        'maximum_snow_energy_deficit': ParameterInfo(-1500.0, -100.0, 'kJ/m²', 'Maximum snow energy deficit (must be negative)', 'snow'),

        # Vegetation parameters (stratum.def)
        'epc.max_lai': ParameterInfo(0.5, 8.0, 'm²/m²', 'Maximum LAI', 'et'),
        'epc.gl_smax': ParameterInfo(0.001, 0.02, 'm/s', 'Maximum stomatal conductance', 'et', 'log'),
        'epc.gl_c': ParameterInfo(0.00001, 0.001, 'm/s', 'Cuticular conductance', 'et', 'log'),
        'epc.vpd_open': ParameterInfo(0.1, 2.0, 'kPa', 'VPD at stomatal opening', 'et'),
        'epc.vpd_close': ParameterInfo(2.0, 6.0, 'kPa', 'VPD at stomatal closure', 'et'),

        # Routing parameters (basin.def)
        'n_routing_power': ParameterInfo(0.1, 1.0, '-', 'Routing power exponent', 'routing'),

        # Forcing correction parameters (worldfile)
        'precip_lapse_rate': ParameterInfo(0.5, 1.5, '-', 'Precipitation multiplier (corrects forcing bias)', 'forcing'),
    }

    # ========================================================================
    # GR PARAMETERS
    # ========================================================================
    GR_PARAMS: Dict[str, ParameterInfo] = {
        # GR4J parameters (bounds based on airGR defaults)
        'X1': ParameterInfo(1.0, 5000.0, 'mm', 'Production store capacity', 'soil'),
        'X2': ParameterInfo(-10.0, 10.0, 'mm/day', 'Groundwater exchange coefficient', 'baseflow'),
        'X3': ParameterInfo(1.0, 500.0, 'mm', 'Routing store capacity', 'soil'),
        'X4': ParameterInfo(0.5, 5.0, 'days', 'Unit hydrograph time constant', 'routing'),

        # CemaNeige parameters (bounds based on airGR defaults)
        'CTG': ParameterInfo(0.0, 1.0, '-', 'Snow process parameter', 'snow'),
        'Kf': ParameterInfo(0.0, 20.0, 'mm/°C/day', 'Melt factor', 'snow'),
        'Gratio': ParameterInfo(0.01, 200.0, '-', 'Thermal coefficient for snow pack thermal state', 'snow'),
        'Albedo_diff': ParameterInfo(0.001, 1.0, '-', 'Albedo diffusion coefficient', 'snow'),
    }

    # ========================================================================
    # VIC PARAMETERS
    # ========================================================================
    VIC_PARAMS: Dict[str, ParameterInfo] = {
        # Variable infiltration curve parameter - controls infiltration nonlinearity
        'infilt': ParameterInfo(0.001, 0.9, '-', 'Variable infiltration curve parameter', 'soil'),
        # Baseflow parameters
        'Ds': ParameterInfo(0.0, 1.0, '-', 'Fraction of Dsmax where nonlinear baseflow begins', 'baseflow'),
        'Dsmax': ParameterInfo(0.1, 30.0, 'mm/day', 'Maximum baseflow velocity', 'baseflow'),
        'Ws': ParameterInfo(0.1, 1.0, '-', 'Fraction of max soil moisture for nonlinear baseflow', 'baseflow'),
        'c': ParameterInfo(1.0, 4.0, '-', 'Exponent in baseflow curve', 'baseflow'),
        # Soil layer depths
        'depth1': ParameterInfo(0.05, 0.5, 'm', 'Soil layer 1 depth', 'soil'),
        'depth2': ParameterInfo(0.1, 1.5, 'm', 'Soil layer 2 depth', 'soil'),
        'depth3': ParameterInfo(0.1, 2.0, 'm', 'Soil layer 3 depth', 'soil'),
        # Soil hydraulic parameters
        'Ksat_vic': ParameterInfo(1.0, 5000.0, 'mm/day', 'VIC saturated hydraulic conductivity', 'soil'),
        'expt_vic': ParameterInfo(4.0, 30.0, '-', 'VIC soil layer exponent', 'soil'),
        # Bulk density
        'bulk_density': ParameterInfo(1200.0, 1800.0, 'kg/m³', 'Soil bulk density', 'soil'),
        # Snow parameters
        'snow_rough': ParameterInfo(0.0001, 0.01, 'm', 'Snow surface roughness', 'snow'),
    }

    # ========================================================================
    # HBV-96 PARAMETERS
    # ========================================================================
    HBV_PARAMS: Dict[str, ParameterInfo] = {
        # Snow parameters
        'tt': ParameterInfo(-3.0, 3.0, '°C', 'Threshold temperature for snow/rain', 'snow'),
        'cfmax': ParameterInfo(1.0, 10.0, 'mm/°C/day', 'Degree-day factor for snowmelt', 'snow'),
        'sfcf': ParameterInfo(0.5, 1.5, '-', 'Snowfall correction factor', 'snow'),
        'cfr': ParameterInfo(0.0, 0.1, '-', 'Refreezing coefficient', 'snow'),
        'cwh': ParameterInfo(0.0, 0.2, '-', 'Snow water holding capacity', 'snow'),

        # Soil parameters
        'fc': ParameterInfo(50.0, 700.0, 'mm', 'Field capacity / max soil moisture', 'soil'),
        'lp': ParameterInfo(0.3, 1.0, '-', 'ET reduction threshold (fraction of FC)', 'soil'),
        'beta': ParameterInfo(1.0, 6.0, '-', 'Shape coefficient for soil routine', 'soil'),

        # Response/baseflow parameters
        'k0': ParameterInfo(0.05, 0.99, '1/day', 'Fast recession coefficient', 'baseflow'),
        'k1': ParameterInfo(0.01, 0.5, '1/day', 'Slow recession coefficient', 'baseflow'),
        'k2': ParameterInfo(0.0001, 0.1, '1/day', 'Baseflow recession coefficient', 'baseflow'),
        'uzl': ParameterInfo(0.0, 100.0, 'mm', 'Upper zone threshold for fast flow', 'baseflow'),
        'perc': ParameterInfo(0.0, 10.0, 'mm/day', 'Maximum percolation rate', 'baseflow'),

        # Routing parameters
        'maxbas': ParameterInfo(1.0, 7.0, 'days', 'Triangular routing function length', 'routing'),

        # Numerical parameters
        'smoothing': ParameterInfo(1.0, 50.0, '-', 'Smoothing factor for thresholds', 'numerical'),
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
        self._all_params.update(self.RHESSYS_PARAMS)
        self._all_params.update(self.GR_PARAMS)
        self._all_params.update(self.VIC_PARAMS)
        self._all_params.update(self.HBV_PARAMS)

    def get_bounds(self, param_name: str) -> Optional[Dict]:
        """
        Get bounds for a single parameter.

        Args:
            param_name: Parameter name

        Returns:
            Dictionary with 'min', 'max', and 'transform' keys, or None if not found
        """
        info = self._all_params.get(param_name)
        if info:
            return {'min': info.min, 'max': info.max, 'transform': info.transform}
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

    def get_bounds_for_params(self, param_names: List[str]) -> Dict[str, Dict]:
        """
        Get bounds for multiple parameters.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping param_name -> {'min': float, 'max': float, 'transform': str}
        """
        bounds = {}
        for name in param_names:
            b = self.get_bounds(name)
            if b:
                bounds[name] = b
        return bounds

    def get_params_by_category(self, category: str) -> Dict[str, Dict]:
        """
        Get all parameter bounds for a category.

        Args:
            category: One of 'snow', 'soil', 'baseflow', 'routing', 'et', 'depth'

        Returns:
            Dictionary of parameter bounds
        """
        return {
            name: {'min': info.min, 'max': info.max, 'transform': info.transform}
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
        'Cgw', 'max_gw_storage', 'refkdt', 'soil_depth',
    ]
    return get_registry().get_bounds_for_params(cfe_params)


def get_ngen_noah_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get NOAH-OWP module parameter bounds.

    Returns:
        Dictionary mapping NOAH param_name -> {'min': float, 'max': float}
    """
    noah_params = [
        'slope', 'dksat', 'psisat', 'bexp', 'smcmax', 'smcwlt', 'smcref',
        'noah_refdk', 'noah_refkdt', 'noah_czil', 'noah_z0',
        'noah_frzk', 'noah_salp', 'rain_snow_thresh', 'ZREF', 'refkdt',
    ]
    return get_registry().get_bounds_for_params(noah_params)


def get_ngen_pet_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get PET module parameter bounds.

    Returns:
        Dictionary mapping PET param_name -> {'min': float, 'max': float}
    """
    pet_params = [
        # BMI config file key names (primary)
        'vegetation_height_m', 'zero_plane_displacement_height_m',
        'momentum_transfer_roughness_length', 'heat_transfer_roughness_length_m',
        'surface_shortwave_albedo', 'surface_longwave_emissivity',
        'wind_speed_measurement_height_m', 'humidity_measurement_height_m',
        # Legacy/alias names
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
        # Snow parameters
        'ttmp', 'cmlt', 'ttpi', 'cmrefr', 'sdnsnew', 'snowdensdt', 'fsceff',
        # Evapotranspiration parameters
        'cevp', 'lp', 'epotdist', 'fepotsnow', 'ttrig', 'treda', 'tredb',
        # Soil hydraulic parameters
        'rrcs1', 'rrcs2', 'rrcs3', 'wcwp', 'wcfc', 'wcep', 'srrcs',
        'bfroznsoil', 'logsatmp', 'bcosby', 'sfrost',
        # Groundwater parameters
        'rcgrw', 'deepperc',
        # Soil temperature parameters
        'deepmem', 'surfmem', 'depthrel',
        # Routing parameters
        'rivvel', 'damp', 'qmean',
        # Lake parameters
        'ilratk', 'ilratp', 'illdepth',
    ]
    return get_registry().get_bounds_for_params(hype_params)


def get_mesh_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all MESH parameter bounds.

    Returns:
        Dictionary mapping MESH param_name -> {'min': float, 'max': float}
    """
    mesh_params = [
        # CLASS.ini parameters (runoff generation)
        'KSAT', 'DRN', 'SDEP', 'XSLP', 'XDRAINH', 'MANN_CLASS',
        # CLASS.ini vegetation parameters (ET control)
        'LAMX', 'ROOT', 'RSMIN',
        # Hydrology.ini parameters (snow/ponding)
        'ZSNL', 'ZPLG', 'ZPLS', 'FRZTH', 'MANN', 'R2N', 'R1N',
        # Baseflow parameters
        'FLZ', 'PWR',
        # Legacy hydrology
        'RCHARG', 'DRAINFRAC', 'BASEFLW',
        # Routing
        'WF_R2', 'DTMINUSR',
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


def get_rhessys_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all RHESSys parameter bounds.

    Returns:
        Dictionary mapping RHESSys param_name -> {'min': float, 'max': float}
    """
    rhessys_params = [
        # Groundwater/baseflow
        'sat_to_gw_coeff', 'gw_loss_coeff', 'gw_loss_fast_coeff', 'gw_loss_fast_threshold',
        # Soil
        'psi_air_entry', 'pore_size_index', 'porosity_0', 'porosity_decay',
        'Ksat_0', 'Ksat_0_v', 'm', 'm_z', 'soil_depth', 'active_zone_z',
        # Subgrid variability (critical for lumped mode peak flows)
        'theta_mean_std_p1', 'theta_mean_std_p2',
        # Snow
        'max_snow_temp', 'min_rain_temp', 'snow_melt_Tcoef', 'snow_water_capacity', 'maximum_snow_energy_deficit',
        # Vegetation/ET
        'epc.max_lai', 'epc.gl_smax', 'epc.gl_c', 'epc.vpd_open', 'epc.vpd_close',
        # Routing
        'n_routing_power',
        # Forcing correction
        'precip_lapse_rate',
    ]
    return get_registry().get_bounds_for_params(rhessys_params)


def get_hbv_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all HBV-96 parameter bounds.

    Returns:
        Dictionary mapping HBV param_name -> {'min': float, 'max': float}
    """
    hbv_params = [
        # Snow
        'tt', 'cfmax', 'sfcf', 'cfr', 'cwh',
        # Soil
        'fc', 'lp', 'beta',
        # Response/baseflow
        'k0', 'k1', 'k2', 'uzl', 'perc',
        # Routing
        'maxbas',
        # Numerical
        'smoothing',
    ]
    return get_registry().get_bounds_for_params(hbv_params)


def get_vic_bounds() -> Dict[str, Dict[str, float]]:
    """
    Get all VIC parameter bounds.

    Returns:
        Dictionary mapping VIC param_name -> {'min': float, 'max': float}
    """
    vic_params = [
        # Infiltration
        'infilt',
        # Baseflow parameters
        'Ds', 'Dsmax', 'Ws', 'c',
        # Soil layer depths
        'depth1', 'depth2', 'depth3',
        # Soil hydraulic parameters
        'Ksat_vic', 'expt_vic',
        # Other
        'bulk_density', 'snow_rough',
    ]
    return get_registry().get_bounds_for_params(vic_params)
