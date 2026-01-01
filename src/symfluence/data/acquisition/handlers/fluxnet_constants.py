"""
FLUXNET/AmeriFlux Constants and Utilities.

This module provides shared constants, variable mappings, and utility functions
for both FLUXNET acquisition and observation handlers.

References:
- FLUXNET2015: https://fluxnet.org/data/fluxnet2015-dataset/
- AmeriFlux: https://ameriflux.lbl.gov/
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

# Physical constants for flux conversions
LE_TO_ET_FACTOR = 0.0353
"""
Conversion factor from latent heat flux (W/m²) to evapotranspiration (mm/day).

Derivation: ET = LE / (rho_w * lambda) * 86400
where:
    - rho_w = 1000 kg/m³ (water density)
    - lambda = 2.45e6 J/kg (latent heat of vaporization at ~20°C)
    - 86400 = seconds per day

Simplified: ET ≈ LE * 0.0353 mm/day
Note: This is an approximation. More accurate conversions should account for
temperature-dependent latent heat of vaporization.
"""

LATENT_HEAT_VAPORIZATION = 2.45e6
"""
Latent heat of vaporization at ~20°C (J/kg).

Varies with temperature:
- At 0°C: ~2.5e6 J/kg
- At 20°C: ~2.45e6 J/kg
- At 40°C: ~2.4e6 J/kg
"""

WATER_DENSITY = 1000.0
"""Water density (kg/m³) at standard conditions."""

SECONDS_PER_DAY = 86400.0
"""Seconds per day for unit conversions."""

# Standard FLUXNET variable mappings
# Includes both FLUXNET2015 format (e.g., LE_F_MDS) and
# AmeriFlux BASE format (e.g., LE_PI_F_1_1_1, LE_1_1_1)
FLUXNET_VARIABLE_MAPPING = {
    # Latent heat / ET
    'LE': ['LE_F_MDS', 'LE_PI_F_1_1_1', 'LE_CORR', 'LE_1_1_1', 'LE', 'LE_F'],
    'ET': ['ET', 'ET_F_MDS', 'LE_F_MDS'],  # LE will be converted

    # Sensible heat
    'H': ['H_F_MDS', 'H_PI_F_1_1_1', 'H_CORR', 'H_1_1_1', 'H', 'H_F'],

    # Radiation
    'Rn': ['NETRAD', 'NETRAD_PI_F_1_1_1', 'NETRAD_1_1_1', 'NETRAD_F', 'Rn'],
    'SW_IN': ['SW_IN_F_MDS', 'SW_IN_PI_F_1_1_1', 'SW_IN_1_1_1', 'SW_IN', 'SW_IN_F'],
    'LW_IN': ['LW_IN_F_MDS', 'LW_IN_PI_F_1_1_1', 'LW_IN_1_1_1', 'LW_IN', 'LW_IN_F'],

    # Ground heat
    'G': ['G_F_MDS', 'G_PI_F_1_1_1', 'G_1_1_1', 'G', 'G_F'],

    # Meteorological
    'TA': ['TA_F_MDS', 'TA_PI_F_1_1_1', 'TA_1_1_1', 'TA', 'TA_F'],
    'VPD': ['VPD_F_MDS', 'VPD_PI_F_1_1_1', 'VPD_1_1_1', 'VPD', 'VPD_F'],
    'P': ['P_F', 'P_PI_F_1_1_1', 'P_1_1_1', 'P', 'PREC'],

    # Quality control
    'LE_QC': ['LE_F_MDS_QC', 'LE_QC'],
    'H_QC': ['H_F_MDS_QC', 'H_QC'],
}

# FLUXNET missing value indicators
FLUXNET_MISSING_VALUES = [-9999, -9999.0, '-9999', '-9999.0', '-9999.00']

# Quality control thresholds
# QC values: 0=measured, 1=good gap-fill, 2=medium, 3=poor
QC_MEASURED = 0
QC_GOOD_GAPFILL = 1
QC_MEDIUM_GAPFILL = 2
QC_POOR_GAPFILL = 3


def convert_le_to_et(
    le: Union[float, np.ndarray, pd.Series],
    temperature_celsius: Optional[Union[float, np.ndarray, pd.Series]] = None
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert latent heat flux (W/m²) to evapotranspiration (mm/day).

    Args:
        le: Latent heat flux in W/m²
        temperature_celsius: Optional temperature for temperature-dependent
                            latent heat of vaporization. If None, uses 20°C.

    Returns:
        Evapotranspiration in mm/day
    """
    if temperature_celsius is not None:
        # Temperature-dependent latent heat of vaporization (Henderson-Sellers, 1984)
        # lambda = 2.501e6 - 2370 * T (J/kg, T in °C)
        lambda_v = 2.501e6 - 2370.0 * temperature_celsius
    else:
        lambda_v = LATENT_HEAT_VAPORIZATION

    # ET = LE / (rho_w * lambda) * seconds_per_day
    # In mm/day (1 kg/m² = 1 mm water depth)
    et = (le / (WATER_DENSITY * lambda_v)) * SECONDS_PER_DAY

    # Handle negative values (can occur with quality issues)
    if isinstance(et, (np.ndarray, pd.Series)):
        et = np.where(et < 0, np.nan, et)
    elif et < 0:
        et = np.nan

    return et


def convert_et_to_le(
    et: Union[float, np.ndarray, pd.Series],
    temperature_celsius: Optional[Union[float, np.ndarray, pd.Series]] = None
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert evapotranspiration (mm/day) to latent heat flux (W/m²).

    Args:
        et: Evapotranspiration in mm/day
        temperature_celsius: Optional temperature for temperature-dependent
                            latent heat of vaporization. If None, uses 20°C.

    Returns:
        Latent heat flux in W/m²
    """
    if temperature_celsius is not None:
        lambda_v = 2.501e6 - 2370.0 * temperature_celsius
    else:
        lambda_v = LATENT_HEAT_VAPORIZATION

    # LE = ET * rho_w * lambda / seconds_per_day
    le = (et * WATER_DENSITY * lambda_v) / SECONDS_PER_DAY

    return le


def find_variable_in_dataframe(
    df: pd.DataFrame,
    target_var: str,
    mapping: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """
    Find a FLUXNET variable in a DataFrame using the standard variable mapping.

    Args:
        df: DataFrame to search
        target_var: Target variable name (e.g., 'LE', 'ET', 'H')
        mapping: Custom variable mapping dict. If None, uses FLUXNET_VARIABLE_MAPPING.

    Returns:
        Column name if found, None otherwise
    """
    if mapping is None:
        mapping = FLUXNET_VARIABLE_MAPPING

    # First try direct match
    if target_var in df.columns:
        return target_var

    # Then try mapping
    source_vars = mapping.get(target_var, [])
    for src in source_vars:
        if src in df.columns:
            return src

    # Case-insensitive fallback
    lower_cols = {c.lower(): c for c in df.columns}
    if target_var.lower() in lower_cols:
        return lower_cols[target_var.lower()]

    return None


def standardize_fluxnet_columns(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, List[str]]] = None,
    convert_le_to_et_flag: bool = True
) -> pd.DataFrame:
    """
    Standardize FLUXNET DataFrame column names.

    Args:
        df: Input DataFrame
        mapping: Variable mapping to use
        convert_le_to_et_flag: If True and LE exists but ET doesn't,
                               convert LE to ET

    Returns:
        DataFrame with standardized column names
    """
    if mapping is None:
        mapping = FLUXNET_VARIABLE_MAPPING

    result = df.copy()

    # Rename columns to standard names
    for target_var, source_vars in mapping.items():
        for src in source_vars:
            if src in result.columns and target_var not in result.columns:
                result = result.rename(columns={src: target_var})
                break

    # Convert LE to ET if needed
    if convert_le_to_et_flag and 'LE' in result.columns and 'ET' not in result.columns:
        result['ET'] = convert_le_to_et(result['LE'])
        result['ET_from_LE_mm_per_day'] = result['ET']

    return result
