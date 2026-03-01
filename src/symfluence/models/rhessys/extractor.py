# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Result Extractor.

Handles extraction of simulation results from RHESSys (Regional Hydro-Ecologic
Simulation System) model outputs. RHESSys outputs are primarily in text format
(.daily files) rather than NetCDF.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class RHESSysResultExtractor(ModelResultExtractor):
    """RHESSys-specific result extraction.

    Handles RHESSys model's unique output characteristics:
    - File formats: Text (.daily files) rather than NetCDF
    - Variable naming: streamflow, evaporation, transpiration, etc.
    - File patterns: *_basin.daily, *_patch.daily, *_hillslope.daily
    - Spatial levels: Basin, hillslope, patch, zone, canopy, stratum
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for RHESSys outputs."""
        return {
            'streamflow': [
                'rhessys_results.csv',  # Pre-processed by runner
                'rhessys_basin.daily',  # Standard basin output
                '*_basin.daily',        # Any basin output
            ],
            'et': [
                'rhessys_results.csv',
                'rhessys_basin.daily',
                '*_basin.daily',
            ],
            'snow': [
                'rhessys_results.csv',
                'rhessys_basin.daily',
                '*_basin.daily',
            ],
            'soil_moisture': [
                'rhessys_basin.daily',
                '*_basin.daily',
                '*_patch.daily',
            ],
            'litter': [
                '*_patch.daily',           # Patch-level litter pools
                '*_grow_patch.daily',      # Growth patch output
            ],
            'vegetation': [
                '*_stratum.daily',         # Stratum-level vegetation
                '*_grow_stratum.daily',    # Growth stratum output
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get RHESSys variable names for different types.

        RHESSys basin.daily files contain many variables. Common ones:
        - streamflow: basin output in mm/day (needs conversion)
        - evaporation: evap
        - transpiration: trans
        - snow: snowpack
        - litter: litr1c, litr2c, litr3c, litr4c carbon pools
        """
        variable_mapping = {
            'streamflow': [
                'streamflow_cms',  # In pre-processed CSV
                'streamflow',
                'Q',
                'discharge',
            ],
            'et': [
                'evaporation',
                'evap',
                'transpiration',
                'trans',
            ],
            'snow': [
                'snowpack',
                'swe',
                'snow_water_equivalent',
            ],
            'soil_moisture': [
                'soil.water',
                'soil_moisture',
                'unsat_storage',
                'rz_storage',
            ],
            'litter': [
                'litr1c',  # Labile litter carbon
                'litr2c',  # Cellulose litter carbon
                'litr3c',  # Lignin litter carbon
                'litr4c',  # Recalcitrant litter carbon
            ],
            'litter_total': [
                'litrc',   # Total litter carbon (if available)
                'litter_carbon',
            ],
            'vegetation': [
                'leafc',   # Leaf carbon
                'stemc',   # Stem carbon
                'rootc',   # Root carbon
                'lai',     # Leaf area index
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from RHESSys output.

        Args:
            output_file: Path to RHESSys output file (.daily or .csv)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - catchment_area: Catchment area in m² for unit conversion
                - date_column: Name of date column (default: varies by file)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        var_names = self.get_variable_names(variable_type)

        # Handle CSV files (pre-processed results)
        if output_file.suffix == '.csv':
            return self._extract_from_csv(output_file, var_names)

        # Handle .daily files (RHESSys native format)
        if output_file.suffix == '.daily':
            return self._extract_from_daily(
                output_file,
                var_names,
                variable_type,
                kwargs.get('catchment_area')
            )

        raise ValueError(
            f"Unsupported file format for RHESSys: {output_file.suffix}. "
            f"Expected .csv or .daily"
        )

    def _extract_from_csv(
        self,
        csv_file: Path,
        var_names: List[str]
    ) -> pd.Series:
        """Extract variable from pre-processed CSV file.

        Args:
            csv_file: Path to CSV file
            var_names: List of possible variable names

        Returns:
            Time series of variable

        Raises:
            ValueError: If no suitable variable found
        """
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        for var_name in var_names:
            if var_name in df.columns:
                return df[var_name]

        raise ValueError(
            f"No suitable variable found in {csv_file}. "
            f"Tried: {var_names}. Available: {df.columns.tolist()}"
        )

    def _extract_from_daily(
        self,
        daily_file: Path,
        var_names: List[str],
        variable_type: str,
        catchment_area: Optional[float] = None
    ) -> pd.Series:
        """Extract variable from RHESSys .daily file.

        RHESSys .daily files are space-separated with first row as header.
        Basin daily files contain basin-aggregated outputs.

        Args:
            daily_file: Path to .daily file
            var_names: List of possible variable names
            variable_type: Type of variable
            catchment_area: Catchment area for unit conversion

        Returns:
            Time series of variable

        Raises:
            ValueError: If variable cannot be extracted
        """
        try:
            # Read space-separated file
            df = pd.read_csv(daily_file, sep=r'\s+', skipinitialspace=True)

            # RHESSys date columns: day, month, year
            if 'day' in df.columns and 'month' in df.columns and 'year' in df.columns:
                df['date'] = pd.to_datetime(
                    df[['year', 'month', 'day']].rename(
                        columns={'day': 'day', 'month': 'month', 'year': 'year'}
                    )
                )
                df = df.set_index('date')

            # Try to find variable
            for var_name in var_names:
                if var_name in df.columns:
                    series = df[var_name]

                    # Apply unit conversion if needed
                    if variable_type == 'streamflow' and catchment_area:
                        # Convert from mm/day to m³/s
                        # mm/day = 0.001 m/day
                        # m³/s = (0.001 m/day * catchment_area_m²) / 86400 s/day
                        series = (series * 0.001 * catchment_area) / 86400

                    return series

            raise ValueError(
                f"No suitable variable found in {daily_file}. "
                f"Tried: {var_names}. Available: {df.columns.tolist()}"
            )

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(
                f"Error reading RHESSys daily file {daily_file}: {str(e)}"
            ) from e

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        RHESSys outputs streamflow in mm/day, requiring conversion to m³/s.
        """
        return variable_type == 'streamflow'

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method.

        RHESSys basin.daily files are already basin-aggregated.
        """
        return None  # Basin outputs are pre-aggregated

    def extract_litter_pools(
        self,
        output_file: Path,
        pool_names: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Extract litter carbon pools from RHESSys patch output.

        RHESSys outputs litter carbon in four pools representing
        different decomposition stages:
        - litr1c: Labile litter (fast decomposing)
        - litr2c: Cellulose litter (medium)
        - litr3c: Lignin litter (slow)
        - litr4c: Recalcitrant litter (very slow)

        Args:
            output_file: Path to patch.daily or grow_patch.daily file
            pool_names: Optional list of specific pools to extract

        Returns:
            Dict mapping pool names to DataFrames with time series
            by patch_id
        """
        if pool_names is None:
            pool_names = ['litr1c', 'litr2c', 'litr3c', 'litr4c']

        try:
            # Read patch daily file
            df = pd.read_csv(output_file, sep=r'\s+', skipinitialspace=True)

            # Construct date index
            if all(col in df.columns for col in ['year', 'month', 'day']):
                df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

            # Find patch ID column
            patch_col = None
            for col in ['patchID', 'patch_ID', 'patch', 'ID']:
                if col in df.columns:
                    patch_col = col
                    break

            results = {}
            for pool in pool_names:
                if pool in df.columns:
                    if patch_col and 'date' in df.columns:
                        # Pivot to get time series by patch
                        pivot_df = df.pivot_table(
                            index='date',
                            columns=patch_col,
                            values=pool,
                            aggfunc='mean'
                        )
                        results[pool] = pivot_df
                    elif 'date' in df.columns:
                        # No patch column, return as series
                        results[pool] = df.set_index('date')[pool]
                    else:
                        results[pool] = df[pool]

            return results

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Error extracting litter pools from {output_file}: {e}") from e

    def aggregate_litter_to_basin(
        self,
        litter_pools: Dict[str, pd.DataFrame],
        patch_areas: Optional[Dict[int, float]] = None
    ) -> Dict[str, pd.Series]:
        """Aggregate patch-level litter pools to basin level.

        Args:
            litter_pools: Dict from extract_litter_pools()
            patch_areas: Optional dict mapping patch_id to area (m²)
                for area-weighted averaging

        Returns:
            Dict mapping pool names to basin-averaged time series
        """
        basin_pools = {}

        for pool_name, pool_data in litter_pools.items():
            if isinstance(pool_data, pd.DataFrame):
                if patch_areas:
                    # Area-weighted average
                    total_area = sum(
                        patch_areas.get(int(pid), 1.0)
                        for pid in pool_data.columns
                    )
                    weighted_sum = sum(
                        pool_data[pid] * patch_areas.get(int(pid), 1.0)
                        for pid in pool_data.columns
                    )
                    basin_pools[pool_name] = weighted_sum / total_area
                else:
                    # Simple mean across patches
                    basin_pools[pool_name] = pool_data.mean(axis=1)
            else:
                # Already a series
                basin_pools[pool_name] = pool_data

        return basin_pools  # type: ignore[return-value]
