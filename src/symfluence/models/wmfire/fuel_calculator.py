"""
WMFire Fuel Calculator Module

Handles conversion of RHESSys litter carbon pools to fuel loads
and implements the Nelson (2000) dead fuel moisture model.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FuelStats:
    """Statistics for fuel load calculations."""
    mean: float
    std: float
    min: float
    max: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'total': self.total,
        }


class FuelCalculator:
    """
    Converts RHESSys litter carbon pools to fuel loads for fire modeling.

    RHESSys tracks four litter carbon pools representing different
    decomposition stages:
    - litr1c: Labile litter (fast decomposing, leaves)
    - litr2c: Cellulose litter (medium decomposing)
    - litr3c: Lignin litter (slow decomposing, woody)
    - litr4c: Recalcitrant litter (very slow decomposing)

    Each pool contributes differently to fire fuel based on its
    combustion characteristics.

    Reference:
        Tague, C.L., Band, L.E. 2004. RHESSys: Regional Hydro-Ecologic
        Simulation System. Computer and Geosciences 30(3): 303-317.
    """

    # Default pool weights based on combustion characteristics
    # Higher weights for more flammable (less decomposed) material
    POOL_WEIGHTS: Dict[str, float] = {
        'litr1c': 0.35,   # Labile - highly flammable
        'litr2c': 0.30,   # Cellulose - moderate
        'litr3c': 0.25,   # Lignin - slower burning
        'litr4c': 0.10,   # Recalcitrant - difficult to burn
    }

    # Typical bulk density ranges by fuel type (kg/m³)
    BULK_DENSITY: Dict[str, float] = {
        'light': 2.0,     # Grass, fine litter
        'medium': 4.0,    # Mixed litter
        'heavy': 8.0,     # Woody debris
    }

    def __init__(
        self,
        carbon_to_fuel_ratio: float = 2.0,
        pool_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the FuelCalculator.

        Args:
            carbon_to_fuel_ratio: Conversion factor from kg C to kg fuel
                (default 2.0, assuming ~50% carbon content in biomass)
            pool_weights: Optional custom weights for litter pools
        """
        self.carbon_to_fuel_ratio = carbon_to_fuel_ratio
        self.pool_weights = pool_weights or self.POOL_WEIGHTS.copy()

    def calculate_fuel_load(
        self,
        litter_pools: Dict[str, Union[float, np.ndarray]],
        cell_area_m2: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate total fuel load from litter carbon pools.

        Args:
            litter_pools: Dictionary of litter pool values (kg C/m²)
                Keys should be 'litr1c', 'litr2c', 'litr3c', 'litr4c'
            cell_area_m2: Optional cell area for total mass calculation

        Returns:
            Fuel load in kg/m² (or kg if cell_area_m2 provided)
        """
        total_fuel = 0.0

        for pool_name, weight in self.pool_weights.items():
            if pool_name in litter_pools:
                carbon = litter_pools[pool_name]
                fuel = carbon * self.carbon_to_fuel_ratio * weight
                total_fuel = total_fuel + fuel

        if cell_area_m2 is not None:
            total_fuel = total_fuel * cell_area_m2

        return total_fuel

    def calculate_fuel_load_grid(
        self,
        litter_grids: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate spatially variable fuel load from litter grids.

        Args:
            litter_grids: Dictionary of 2D arrays for each litter pool

        Returns:
            2D array of fuel loads (kg/m²)
        """
        # Get shape from first available grid
        shape = None
        for grid in litter_grids.values():
            shape = grid.shape
            break

        if shape is None:
            raise ValueError("No litter grids provided")

        fuel_grid = np.zeros(shape, dtype='float32')

        for pool_name, weight in self.pool_weights.items():
            if pool_name in litter_grids:
                fuel_grid += litter_grids[pool_name] * self.carbon_to_fuel_ratio * weight

        return fuel_grid

    def calculate_load_coefficients(
        self,
        fuel_load: Union[float, np.ndarray]
    ) -> Tuple[float, float]:
        """
        Calculate WMFire load coefficients (k1, k2) from fuel load statistics.

        The load coefficients control the probability of fire spread
        based on fuel availability. Higher k1 increases base spread rate,
        while k2 controls the sensitivity to fuel load variation.

        Args:
            fuel_load: Fuel load array or scalar (kg/m²)

        Returns:
            Tuple of (load_k1, load_k2) coefficients
        """
        if isinstance(fuel_load, np.ndarray):
            mean_load = np.nanmean(fuel_load)
            std_load = np.nanstd(fuel_load)
        else:
            mean_load = fuel_load
            std_load = 0.0

        # Default coefficients from RHESSys fire.def
        default_k1 = 3.9
        default_k2 = 0.07

        # Scale k1 based on mean fuel load
        # Higher fuel loads increase base spread probability
        # Typical range: 0.5-5.0 kg/m² for forest litter
        if mean_load > 0:
            load_factor = np.clip(mean_load / 2.0, 0.5, 2.0)
            k1 = default_k1 * load_factor
        else:
            k1 = default_k1

        # Scale k2 based on fuel load variability
        # Higher variability increases sensitivity
        if std_load > 0 and mean_load > 0:
            cv = std_load / mean_load  # Coefficient of variation
            k2 = default_k2 * (1.0 + cv)
        else:
            k2 = default_k2

        return k1, k2

    def get_fuel_stats(self, fuel_load: np.ndarray) -> FuelStats:
        """
        Calculate statistics for fuel load array.

        Args:
            fuel_load: 2D array of fuel loads

        Returns:
            FuelStats object with summary statistics
        """
        valid_data = fuel_load[~np.isnan(fuel_load)]

        if len(valid_data) == 0:
            return FuelStats(mean=0.0, std=0.0, min=0.0, max=0.0, total=0.0)

        return FuelStats(
            mean=float(np.mean(valid_data)),
            std=float(np.std(valid_data)),
            min=float(np.min(valid_data)),
            max=float(np.max(valid_data)),
            total=float(np.sum(valid_data))
        )


class FuelMoistureModel:
    """
    Nelson (2000) dead fuel moisture model for fire modeling.

    Implements the equilibrium moisture content (EMC) model and
    time-lag response for dead fuel moisture dynamics.

    Reference:
        Nelson, R.M. 2000. Prediction of diurnal change in 10-h fuel
        stick moisture content. Canadian Journal of Forest Research
        30(7): 1071-1087.
    """

    # Time-lag constants (hours) for different fuel size classes
    TIMELAGS: Dict[str, float] = {
        '1hr': 1.0,      # Fine fuels (< 6mm diameter)
        '10hr': 10.0,    # Small fuels (6-25mm)
        '100hr': 100.0,  # Medium fuels (25-75mm)
        '1000hr': 1000.0  # Large fuels (> 75mm)
    }

    # Fiber saturation point (fraction)
    FIBER_SATURATION = 0.35

    def __init__(self, fuel_class: str = '10hr'):
        """
        Initialize the FuelMoistureModel.

        Args:
            fuel_class: Fuel size class ('1hr', '10hr', '100hr', '1000hr')
        """
        self.fuel_class = fuel_class
        self.timelag = self.TIMELAGS.get(fuel_class, 10.0)

    def equilibrium_moisture(
        self,
        rh: Union[float, np.ndarray],
        temp_c: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate equilibrium moisture content (EMC).

        Uses the Nelson (2000) EMC equations which account for
        temperature effects on the sorption isotherm.

        Args:
            rh: Relative humidity (0-1 or 0-100)
            temp_c: Air temperature in Celsius

        Returns:
            Equilibrium moisture content (fraction, 0-1)
        """
        # Ensure RH is in fraction form
        if isinstance(rh, np.ndarray):
            rh = np.where(rh > 1, rh / 100.0, rh)
        elif rh > 1:
            rh = rh / 100.0

        # Clamp values
        rh = np.clip(rh, 0.01, 0.99)
        temp_c = np.clip(temp_c, -40, 50)

        # Nelson (2000) EMC calculation
        # Modified Simard equation with temperature correction
        if isinstance(rh, np.ndarray) or isinstance(temp_c, np.ndarray):
            # Ensure arrays for vectorized computation
            rh = np.atleast_1d(rh)
            temp_c = np.atleast_1d(temp_c)

        # Temperature correction factor
        # Higher temperatures reduce EMC
        temp_factor = 1.0 - 0.0025 * (temp_c - 20.0)

        # Adsorption curve (rh < 0.5)
        # Desorption curve (rh >= 0.5)
        emc = np.where(
            rh < 0.5,
            # Adsorption
            0.03 + 0.2 * rh * temp_factor,
            # Desorption (higher moisture retention)
            0.08 + 0.16 * rh * temp_factor
        )

        # Apply fiber saturation limit
        emc = np.clip(emc, 0.02, self.FIBER_SATURATION)

        return emc

    def update_moisture(
        self,
        current_mc: Union[float, np.ndarray],
        emc: Union[float, np.ndarray],
        timestep_hours: float,
        timelag: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Update fuel moisture content based on time-lag response.

        Implements exponential decay toward equilibrium moisture.

        Args:
            current_mc: Current moisture content (fraction)
            emc: Equilibrium moisture content (fraction)
            timestep_hours: Time step in hours
            timelag: Optional custom time lag (hours)

        Returns:
            Updated moisture content (fraction)
        """
        if timelag is None:
            timelag = self.timelag

        # Exponential decay toward EMC
        # mc(t+dt) = emc + (mc(t) - emc) * exp(-dt/timelag)
        decay = np.exp(-timestep_hours / timelag)
        new_mc = emc + (current_mc - emc) * decay

        # Clamp to valid range
        new_mc = np.clip(new_mc, 0.02, 0.50)

        return new_mc

    def calculate_moisture_coefficients(
        self,
        moisture: Union[float, np.ndarray]
    ) -> Tuple[float, float]:
        """
        Calculate WMFire moisture coefficients (k1, k2) from moisture data.

        The moisture coefficients control fire spread probability
        based on fuel moisture content.

        Args:
            moisture: Moisture content array or scalar (fraction)

        Returns:
            Tuple of (moisture_k1, moisture_k2) coefficients
        """
        if isinstance(moisture, np.ndarray):
            mean_mc = np.nanmean(moisture)
            std_mc = np.nanstd(moisture)
        else:
            mean_mc = moisture
            std_mc = 0.0

        # Default coefficients from RHESSys fire.def
        default_k1 = 3.8
        default_k2 = 0.27

        # Scale k1 inversely with moisture
        # Drier conditions (lower MC) increase spread probability
        # Typical dead fuel MC range: 5-35%
        if mean_mc > 0:
            mc_factor = np.clip(0.15 / mean_mc, 0.5, 2.0)
            k1 = default_k1 * mc_factor
        else:
            k1 = default_k1

        # Scale k2 based on moisture variability
        if std_mc > 0 and mean_mc > 0:
            cv = std_mc / mean_mc
            k2 = default_k2 * (1.0 + 0.5 * cv)
        else:
            k2 = default_k2

        return k1, k2

    def critical_moisture(self, fuel_type: str = 'forest') -> float:
        """
        Get critical moisture content for fire spread.

        Fire spread is typically inhibited above this threshold.

        Args:
            fuel_type: Type of fuel ('grass', 'shrub', 'forest')

        Returns:
            Critical moisture content (fraction)
        """
        critical_mc = {
            'grass': 0.15,    # 15% - grass fires extinguish easily
            'shrub': 0.20,    # 20% - brush/chaparral
            'forest': 0.25,   # 25% - forest litter
            'slash': 0.30,    # 30% - logging slash
        }
        return critical_mc.get(fuel_type, 0.25)


def estimate_initial_moisture(
    month: int,
    latitude: float,
    climate: str = 'temperate'
) -> float:
    """
    Estimate initial fuel moisture based on season and location.

    Args:
        month: Month of year (1-12)
        latitude: Latitude in degrees
        climate: Climate type ('arid', 'mediterranean', 'temperate', 'boreal')

    Returns:
        Estimated moisture content (fraction)
    """
    # Base seasonal pattern (Northern Hemisphere)
    # Driest in late summer, wettest in winter/spring
    seasonal = {
        1: 0.25, 2: 0.25, 3: 0.22, 4: 0.18,
        5: 0.15, 6: 0.12, 7: 0.10, 8: 0.08,
        9: 0.10, 10: 0.15, 11: 0.20, 12: 0.25
    }

    base_mc = seasonal.get(month, 0.15)

    # Flip pattern for Southern Hemisphere
    if latitude < 0:
        shifted_month = ((month + 5) % 12) + 1
        base_mc = seasonal.get(shifted_month, 0.15)

    # Climate adjustments
    climate_factor = {
        'arid': 0.6,         # Drier
        'mediterranean': 0.8,  # Dry summers
        'temperate': 1.0,     # Baseline
        'boreal': 1.2,        # Wetter
        'tropical': 1.1,      # High humidity
    }

    factor = climate_factor.get(climate, 1.0)
    mc = base_mc * factor

    return np.clip(mc, 0.05, 0.35)
