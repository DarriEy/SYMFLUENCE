"""
Synthetic hydrograph generation for FUSE model.

This module contains the FuseSyntheticDataGenerator class which creates
realistic synthetic hydrographs when observations are not available,
particularly useful for snow-dominated catchments during model calibration.
"""

import numpy as np


class FuseSyntheticDataGenerator:
    """
    Generator for synthetic hydrographs when observations are not available.

    This class handles:
    - Realistic synthetic hydrograph generation from precipitation and temperature
    - Snow accumulation and melt modeling
    - Distributed hydrograph generation for multiple subcatchments
    - Temporal variation and baseflow simulation

    Attributes:
        logger: Logger instance
    """

    def __init__(self, logger):
        """
        Initialize the synthetic data generator.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    def generate_synthetic_hydrograph(
        self,
        ds,
        area_km2: float,
        mean_temp_threshold: float = 0.0
    ) -> np.ndarray:
        """
        Generate a realistic synthetic hydrograph for snow optimization cases.

        Creates a hydrograph based on precipitation and temperature data,
        simulating snow accumulation, melt, and baseflow processes.

        Args:
            ds: xarray dataset with precipitation ('pr') and temperature ('temp')
            area_km2: catchment area in km²
            mean_temp_threshold: temperature threshold for snow/rain (°C)

        Returns:
            np.ndarray: synthetic streamflow in mm/day
        """
        self.logger.info("Generating synthetic hydrograph for optimization")

        # Extract data — handle both 'pr' and 'precip' variable names
        pr_var = 'pr' if 'pr' in ds else 'precip'
        precip = ds[pr_var].values  # mm/day
        temp = ds['temp'].values  # °C
        if precip.ndim > 1:
            precip = precip.mean(axis=tuple(range(1, precip.ndim)))
        if temp.ndim > 1:
            temp = temp.mean(axis=tuple(range(1, temp.ndim)))

        n_timesteps = len(precip)

        # Initialize arrays
        snow_storage = np.zeros(n_timesteps)
        melt = np.zeros(n_timesteps)
        runoff = np.zeros(n_timesteps)

        # Parameters for synthetic hydrograph
        snow_init = 0.0  # Initial snow storage (mm)
        melt_factor = 3.0  # Degree-day melt factor (mm/°C/day)
        runoff_fraction = 0.3  # Fraction of precip+melt that becomes runoff
        baseflow_fraction = 0.1  # Baseflow as fraction of mean precip

        # Calculate baseflow
        mean_precip = np.mean(precip)
        baseflow = mean_precip * baseflow_fraction

        # Simulate snow accumulation and melt
        current_snow = snow_init

        for t in range(n_timesteps):
            # Determine snow vs rain
            if temp[t] < mean_temp_threshold:
                # Precipitation falls as snow
                current_snow += precip[t]
                snow_storage[t] = current_snow
                melt[t] = 0.0
            else:
                # Precipitation falls as rain
                # Calculate snowmelt
                if current_snow > 0 and temp[t] > mean_temp_threshold:
                    potential_melt = melt_factor * (temp[t] - mean_temp_threshold)
                    actual_melt = min(potential_melt, current_snow)
                    current_snow -= actual_melt
                    melt[t] = actual_melt
                else:
                    melt[t] = 0.0

                snow_storage[t] = current_snow

                # Calculate runoff from rain + melt
                total_water = precip[t] + melt[t]
                runoff[t] = total_water * runoff_fraction + baseflow

        # Add some realistic variability
        noise = np.random.normal(0, 0.05 * np.mean(runoff[runoff > 0]), n_timesteps)
        synthetic_q = np.maximum(runoff + noise, 0)  # Ensure non-negative

        self.logger.info(f"Generated synthetic hydrograph: "
                        f"mean={np.mean(synthetic_q):.2f} mm/day, "
                        f"max={np.max(synthetic_q):.2f} mm/day")

        return synthetic_q

    def generate_distributed_synthetic_hydrograph(
        self,
        ds,
        n_subcatchments: int,
        time_length: int
    ) -> np.ndarray:
        """
        Generate synthetic hydrograph for each subcatchment.

        Creates spatially distributed hydrographs with realistic temporal
        and spatial variability.

        Args:
            ds: xarray dataset with base hydrograph data
            n_subcatchments: number of subcatchments
            time_length: number of time steps

        Returns:
            np.ndarray: synthetic streamflow (time, subcatchments) in mm/day
        """
        self.logger.info(f"Generating distributed synthetic hydrograph for {n_subcatchments} subcatchments")

        # Create base synthetic hydrograph
        if ('pr' in ds or 'precip' in ds) and 'temp' in ds:
            # Use precipitation and temperature to create base hydrograph
            base_hydrograph = self.generate_synthetic_hydrograph(ds, area_km2=100.0)
        else:
            # Create simple synthetic pattern
            time = np.arange(time_length)
            base_hydrograph = (
                5.0 +  # Baseflow
                10.0 * np.sin(2 * np.pi * time / 365) +  # Annual cycle
                3.0 * np.sin(2 * np.pi * time / 30) +  # Monthly variation
                np.random.normal(0, 1, time_length)  # Random noise
            )
            base_hydrograph = np.maximum(base_hydrograph, 0.1)  # Ensure positive

        # Add spatial variability for each subcatchment
        variations = np.random.uniform(0.8, 1.2, n_subcatchments)  # ±20% variation
        distributed_q = np.outer(base_hydrograph, variations)  # (time, subcatchments)

        self.logger.info(f"Generated distributed hydrograph with shape: {distributed_q.shape}")

        return distributed_q
