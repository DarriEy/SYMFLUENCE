#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics Calculation for SUMMA Workers

This module contains functions for calculating calibration metrics
in worker processes, supporting multi-target optimization.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def resample_to_timestep(data: pd.Series, target_timestep: str, logger) -> pd.Series:
    """
    Resample time series data to target timestep

    Args:
        data: Time series data with DatetimeIndex
        target_timestep: Target timestep ('native', 'hourly', or 'daily')
        logger: Logger instance

    Returns:
        Resampled time series
    """
    if target_timestep == 'native' or data is None or len(data) == 0:
        return data

    try:
        # Determine current timestep
        time_diff = data.index[1] - data.index[0] if len(data) > 1 else pd.Timedelta(hours=1)

        # Check if already at target timestep
        if target_timestep == 'hourly' and pd.Timedelta(minutes=45) <= time_diff <= pd.Timedelta(minutes=75):
            logger.debug("Data already at hourly timestep")
            return data
        elif target_timestep == 'daily' and pd.Timedelta(hours=20) <= time_diff <= pd.Timedelta(hours=28):
            logger.debug("Data already at daily timestep")
            return data

        # Perform resampling
        if target_timestep == 'hourly':
            if time_diff < pd.Timedelta(hours=1):
                # Upsampling: sub-hourly to hourly (mean aggregation)
                logger.debug(f"Aggregating {time_diff} data to hourly using mean")
                resampled = data.resample('h').mean()
            elif time_diff > pd.Timedelta(hours=1):
                # Downsampling: daily/coarser to hourly (interpolation)
                logger.debug(f"Interpolating {time_diff} data to hourly")
                resampled = data.resample('h').asfreq()
                resampled = resampled.interpolate(method='time', limit_direction='both')
            else:
                resampled = data

        elif target_timestep == 'daily':
            if time_diff < pd.Timedelta(days=1):
                # Upsampling: hourly/sub-daily to daily (mean aggregation)
                logger.debug(f"Aggregating {time_diff} data to daily using mean")
                resampled = data.resample('D').mean()
            elif time_diff > pd.Timedelta(days=1):
                # Downsampling: weekly/monthly to daily (interpolation)
                logger.debug(f"Interpolating {time_diff} data to daily")
                resampled = data.resample('D').asfreq()
                resampled = resampled.interpolate(method='time', limit_direction='both')
            else:
                resampled = data
        else:
            resampled = data

        # Remove any NaN values introduced by resampling at edges
        resampled = resampled.dropna()

        logger.debug(f"Resampled from {len(data)} to {len(resampled)} points (target: {target_timestep})")

        return resampled

    except Exception as e:
        logger.debug(f"Error resampling to {target_timestep}: {str(e)}")
        logger.debug("Returning original data without resampling")
        return data


def _get_catchment_area_worker(config: Dict, logger) -> float:
    """Get actual catchment area for unit conversion (worker version)"""
    try:
        # Priority 0: Manual override
        fixed_area = config.get('FIXED_CATCHMENT_AREA')
        if fixed_area:
            logger.info(f"Using fixed catchment area from config: {fixed_area} m²")
            return float(fixed_area)

        domain_name = config.get('DOMAIN_NAME')
        project_dir = Path(config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}"

        # Priority 1: Try SUMMA attributes file first (most reliable)
        attrs_file = project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
        if attrs_file.exists():
            try:
                import xarray as xr
                with xr.open_dataset(attrs_file) as attrs:
                    if 'HRUarea' in attrs.data_vars:
                        catchment_area_m2 = float(attrs['HRUarea'].values.sum())
                        if 0 < catchment_area_m2 < 1e12:  # Reasonable area check
                            logger.warning(f"WORKER DEBUG: Using catchment area from SUMMA attributes: {catchment_area_m2:.0f} m²")
                            return catchment_area_m2
                    else:
                        logger.debug("HRUarea not found in SUMMA attributes file")
            except Exception as e:
                logger.debug(f"Error reading SUMMA attributes file: {str(e)}")
        else:
            logger.debug(f"SUMMA attributes file not found: {attrs_file}")

        # Priority 2: Try basin shapefile
        basin_path = project_dir / "shapefiles" / "river_basins"
        basin_files = list(basin_path.glob("*.shp"))

        if basin_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(basin_files[0])
                area_col = config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')

                logger.debug(f"Found basin shapefile: {basin_files[0]}")
                logger.debug(f"Looking for area column: {area_col}")
                logger.debug(f"Available columns: {list(gdf.columns)}")

                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    logger.debug(f"Total area from column: {total_area}")

                    if 0 < total_area < 1e12:  # Reasonable area check
                        logger.info(f"Using catchment area from shapefile: {total_area:.0f} m²")
                        return total_area

                # Fallback: calculate from geometry
                if gdf.crs and gdf.crs.is_geographic:
                    # Reproject to UTM for area calculation
                    centroid = gdf.dissolve().centroid.iloc[0]
                    utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +north +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)

                geom_area = gdf.geometry.area.sum()
                logger.info(f"Using catchment area from geometry: {geom_area:.0f} m²")
                return geom_area

            except ImportError:
                logger.warning("geopandas not available for area calculation")
            except Exception as e:
                logger.warning(f"Error reading basin shapefile: {str(e)}")

        # Priority 3: Try catchment shapefile
        catchment_path = project_dir / "shapefiles" / "catchment"
        catchment_files = list(catchment_path.glob("*.shp"))

        if catchment_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(catchment_files[0])
                area_col = config.get('CATCHMENT_SHP_AREA', 'HRU_area')

                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:
                        logger.info(f"Using catchment area from catchment shapefile: {total_area:.0f} m²")
                        return total_area

            except Exception as e:
                logger.warning(f"Error reading catchment shapefile: {str(e)}")

    except Exception as e:
        logger.warning(f"Could not calculate catchment area: {str(e)}")

    # Fallback
    logger.warning("Using default catchment area: 1,000,000 m²")
    return 1e6


def _calculate_metrics_with_target(summa_dir: Path, mizuroute_dir: Path, config: Dict, logger) -> Dict:
    """
    Calculate metrics using proper CalibrationTarget classes.

    This replaces the inline streamflow-only calculation to support all calibration targets
    including streamflow, SWE, SCA, ET, soil moisture, etc.
    """
    try:
        from ...calibration_targets import (
            StreamflowTarget, SnowTarget, GroundwaterTarget, ETTarget,
            SoilMoistureTarget, TWSTarget
        )
        from pathlib import Path as PathType

        # Get the project directory from config
        project_dir = PathType(config.get('SYMFLUENCE_DATA_DIR', '.')) / f"domain_{config.get('DOMAIN_NAME')}"

        # Determine the calibration target type
        calibration_var = config.get('CALIBRATION_VARIABLE', 'streamflow')
        optimization_target = config.get('OPTIMIZATION_TARGET', calibration_var).lower()

        logger.debug(f"Creating calibration target for: {optimization_target}")

        # Create the appropriate calibration target
        if optimization_target in ['streamflow', 'flow', 'discharge']:
            target = StreamflowTarget(config, project_dir, logger)
        elif optimization_target in ['swe', 'sca', 'snow_depth', 'snow']:
            target = SnowTarget(config, project_dir, logger)
        elif optimization_target in ['gw_depth', 'gw_grace', 'groundwater', 'gw']:
            target = GroundwaterTarget(config, project_dir, logger)
        elif optimization_target in ['et', 'latent_heat', 'evapotranspiration']:
            target = ETTarget(config, project_dir, logger)
        elif optimization_target in ['sm_point', 'sm_smap', 'sm_esa', 'sm_ismn', 'soil_moisture', 'sm']:
            target = SoilMoistureTarget(config, project_dir, logger)
        elif optimization_target in ['tws', 'grace', 'grace_tws', 'total_storage']:
            target = TWSTarget(config, project_dir, logger)
        else:
            # Default to streamflow
            logger.warning(f"Unknown optimization target '{optimization_target}', defaulting to streamflow")
            target = StreamflowTarget(config, project_dir, logger)

        # Calculate metrics using the target
        logger.debug(f"Calculating metrics from {summa_dir}")

        # DIAGNOSTIC: Check what files exist and their SWE values
        import glob
        day_files = list(Path(summa_dir).glob("*_day.nc"))
        if day_files:
            import xarray as xr
            with xr.open_dataset(day_files[0]) as ds:
                if 'scalarSWE' in ds:
                    swe_raw = ds['scalarSWE']
                    logger.debug(
                        "WORKER DIAG SWE in file: min=%.3f, max=%.3f kg/m²",
                        float(swe_raw.min()),
                        float(swe_raw.max()),
                    )

        metrics = target.calculate_metrics(
            summa_dir,
            mizuroute_dir=mizuroute_dir,
            calibration_only=True
        )

        if metrics:
            logger.debug(f"Metrics calculated successfully: {list(metrics.keys())}")
            return metrics
        else:
            logger.warning("Calibration target returned empty metrics")
            return None

    except Exception as e:
        import traceback
        logger.error(f"Error calculating metrics with calibration target: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def _calculate_metrics_inline_worker(summa_dir: Path, mizuroute_dir: Path, config: Dict, logger) -> Dict:
    """
    Calculate metrics inline without using CalibrationTarget classes (STREAMFLOW ONLY).

    DEPRECATED: This function is kept for backward compatibility but only supports streamflow.
    Use _calculate_metrics_with_target instead for multi-target support.
    """
    try:
        import xarray as xr

        logger.warning("WORKER DEBUG: Starting inline metrics calculation")
        logger.debug("Starting inline metrics calculation")
        logger.warning(f"WORKER DEBUG: SUMMA dir: {summa_dir}")
        logger.warning(f"WORKER DEBUG: mizuRoute dir: {mizuroute_dir}")
        logger.warning(f"WORKER DEBUG: SUMMA dir exists: {summa_dir.exists()}")
        logger.warning(f"WORKER DEBUG: mizuRoute dir exists: {mizuroute_dir.exists() if mizuroute_dir else 'None'}")

        # Priority 1: Look for mizuRoute output files first (already in m³/s)
        sim_files = []
        use_mizuroute = False
        catchment_area = None

        if mizuroute_dir and mizuroute_dir.exists():
            mizu_files = list(mizuroute_dir.glob("*.nc"))
            logger.debug(f"Found {len(mizu_files)} mizuRoute .nc files")
            for f in mizu_files[:3]:  # Show first 3
                logger.debug(f"mizuRoute file: {f.name}")

            if mizu_files:
                sim_files = mizu_files
                use_mizuroute = True
                logger.debug("Using mizuRoute files (already in m³/s)")

        # Priority 2: If no mizuRoute files, look for SUMMA files (need m/s to m³/s conversion)
        # Only allow fallback to SUMMA if BOTH domain and routing are lumped
        if not sim_files:
            domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

            if domain_method == 'lumped' and routing_delineation == 'lumped':
                summa_files = list(summa_dir.glob("*timestep.nc"))
                logger.debug(f"Found {len(summa_files)} SUMMA timestep files")

                if summa_files:
                    sim_files = summa_files
                    use_mizuroute = False
                    logger.debug("Using SUMMA files (lumped domain and routing, need m/s to m³/s conversion)")

                    # Get the ACTUAL catchment area for unit conversion
                    try:
                        catchment_area = _get_catchment_area_worker(config, logger)
                        logger.warning(f"WORKER DEBUG: Got catchment area = {catchment_area:.2e} m²")
                    except Exception as e:
                        logger.debug(f"Error getting catchment area: {str(e)}")
                        catchment_area = 1e6  # Default fallback
            else:
                logger.error(f"No mizuRoute output found and domain/routing not both lumped (Domain: {domain_method}, Routing: {routing_delineation}). Cannot fall back to SUMMA runoff.")
                return None

        # Check if we found any simulation files
        if not sim_files:
            logger.warning("WORKER DEBUG: No simulation files found")
            logger.debug("No simulation files found")
            return None

        logger.warning(f"WORKER DEBUG: Found {len(sim_files)} simulation files, use_mizuroute={use_mizuroute}")

        sim_file = sim_files[0]
        logger.debug(f"Using simulation file: {sim_file}")

        # Extract simulated streamflow
        logger.debug(f"Extracting simulated streamflow (use_mizuroute={use_mizuroute})...")

        try:
            with xr.open_dataset(sim_file) as ds:
                if use_mizuroute:
                    # mizuRoute output - select segment with highest average runoff (outlet)
                    routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
                    routing_var = None

                    for var_name in routing_vars:
                        if var_name in ds.variables:
                            routing_var = var_name
                            break

                    if routing_var is None:
                        return None

                    var = ds[routing_var]

                    try:
                        # Calculate average runoff for each segment to find outlet
                        if 'seg' in var.dims:
                            segment_means = var.mean(dim='time').values
                            outlet_seg_idx = np.argmax(segment_means)
                            sim_data = var.isel(seg=outlet_seg_idx).to_pandas()
                        elif 'reachID' in var.dims:
                            reach_means = var.mean(dim='time').values
                            outlet_reach_idx = np.argmax(reach_means)
                            sim_data = var.isel(reachID=outlet_reach_idx).to_pandas()
                        else:
                            return None

                        logger.warning(f"WORKER DEBUG: Extracted {routing_var} (mizuRoute), mean = {float(sim_data.mean()):.2f} m³/s")

                    except Exception as e:
                        logger.debug(f"Error extracting outlet segment from {routing_var}: {str(e)}")
                        return None

                else:
                    # SUMMA output - convert from m/s to m³/s using ACTUAL area
                    summa_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
                    sim_data = None

                    for var_name in summa_vars:
                        if var_name in ds.variables:
                            var = ds[var_name]

                            units = var.attrs.get('units', 'unknown')
                            logger.debug(f"Worker found streamflow variable {var_name} with units: '{units}'")

                            # Unit conversion: Mass flux (kg m-2 s-1) to Volume flux (m s-1)
                            is_mass_flux = False
                            if 'units' in var.attrs and 'kg' in var.attrs['units'] and 's-1' in var.attrs['units']:
                                is_mass_flux = True
                            elif float(var.mean()) > 1e-6:
                                logger.debug(f"Worker: Variable {var_name} mean ({float(var.mean()):.2e}) is unreasonably high. Assuming mislabeled mass flux.")
                                is_mass_flux = True

                            if is_mass_flux:
                                logger.debug(f"Worker: Converting {var_name} from mass flux to volume flux (dividing by 1000)")
                                var = var / 1000.0

                            try:
                                # Attempt area-weighted aggregation first
                                aggregated = False
                                if len(var.shape) > 1:
                                    try:
                                        # Look for attributes.nc in common locations
                                        possible_attr_paths = [
                                            summa_dir.parent.parent.parent / 'settings' / 'SUMMA' / 'attributes.nc',
                                            Path(config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'attributes.nc'
                                        ]

                                        attrs_file = None
                                        for p in possible_attr_paths:
                                            if p.exists():
                                                attrs_file = p
                                                break

                                        if attrs_file:
                                            with xr.open_dataset(attrs_file) as attrs:
                                                # Handle HRU dimension
                                                if 'hru' in var.dims and 'HRUarea' in attrs:
                                                    areas = attrs['HRUarea']
                                                    if areas.sizes['hru'] == var.sizes['hru']:
                                                        total_area = float(areas.values.sum())
                                                        logger.debug(f"Worker: Performing area-weighted aggregation for {var_name} (HRU). Total area: {total_area:.1f} m²")
                                                        sim_data = (var * areas).sum(dim='hru').to_pandas()
                                                        aggregated = True

                                                # Handle GRU dimension
                                                elif 'gru' in var.dims and 'GRUarea' in attrs:
                                                    areas = attrs['GRUarea']
                                                    if areas.sizes['gru'] == var.sizes['gru']:
                                                        total_area = float(areas.values.sum())
                                                        logger.debug(f"Worker: Performing area-weighted aggregation for {var_name} (GRU). Total area: {total_area:.1f} m²")
                                                        sim_data = (var * areas).sum(dim='gru').to_pandas()
                                                        aggregated = True

                                                # Handle GRU dimension with HRUarea fallback
                                                elif 'gru' in var.dims and 'HRUarea' in attrs:
                                                    if attrs.sizes['hru'] == var.sizes['gru']:
                                                        areas = attrs['HRUarea']
                                                        total_area = float(areas.values.sum())
                                                        logger.debug(f"Worker: Performing area-weighted aggregation for {var_name} (GRU fallback). Total area: {total_area:.1f} m²")
                                                        dim_name = 'hru' if 'hru' in areas.dims else 'gru'
                                                        sim_data = (var * areas).sum(dim=dim_name).to_pandas()
                                                        aggregated = True
                                    except Exception as e:
                                        logger.debug(f"Aggregation failed, falling back: {e}")

                                if not aggregated:
                                    if len(var.shape) > 1:
                                        if 'hru' in var.dims:
                                            sim_data = var.isel(hru=0).to_pandas()
                                        elif 'gru' in var.dims:
                                            sim_data = var.isel(gru=0).to_pandas()
                                        else:
                                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                                            if non_time_dims:
                                                sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                                            else:
                                                sim_data = var.to_pandas()
                                    else:
                                        sim_data = var.to_pandas()

                                    # Convert units for SUMMA (m/s to m³/s) using ACTUAL catchment area
                                    logger.warning(f"WORKER DEBUG: Converting {var_name} using catchment area = {catchment_area:.2e} m²")
                                    logger.warning(f"WORKER DEBUG: Pre-conversion mean = {sim_data.mean():.2e}")
                                    sim_data = sim_data * catchment_area
                                    logger.warning(f"WORKER DEBUG: Post-conversion mean = {float(sim_data.mean()):.2f} m³/s")

                                break
                            except Exception as e:
                                continue

                    if sim_data is None:
                        logger.warning("WORKER DEBUG: sim_data is None after trying all SUMMA variables")
                        return None

        except Exception as e:
            logger.warning(f"WORKER DEBUG: Exception extracting simulated streamflow: {str(e)}")
            import traceback
            logger.warning(f"WORKER DEBUG: Traceback: {traceback.format_exc()}")
            return None

        # Load observed data
        try:
            domain_name = config.get('DOMAIN_NAME')
            project_dir = Path(config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}"
            obs_path = project_dir / "observations" / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"

            logger.warning(f"WORKER DEBUG: Looking for observations at: {obs_path}")
            if not obs_path.exists():
                logger.warning("WORKER DEBUG: Observation file does not exist")
                return None

            obs_df = pd.read_csv(obs_path)
            date_col = next((col for col in obs_df.columns if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)
            flow_col = next((col for col in obs_df.columns if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow'])), None)

            logger.warning(f"WORKER DEBUG: Found date_col={date_col}, flow_col={flow_col}")
            if not date_col or not flow_col:
                logger.warning("WORKER DEBUG: Missing date or flow column in observations")
                return None

            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            obs_data = obs_df[flow_col]
            logger.warning(f"WORKER DEBUG: Loaded {len(obs_data)} observation points")
        except Exception as e:
            logger.warning(f"WORKER DEBUG: Exception loading observations: {str(e)}")
            return None

        # Filter to calibration period
        cal_period = config.get('CALIBRATION_PERIOD', '')
        if cal_period:
            try:
                dates = [d.strip() for d in cal_period.split(',')]
                if len(dates) >= 2:
                    start_date = pd.Timestamp(dates[0])
                    end_date = pd.Timestamp(dates[1])

                    obs_mask = (obs_data.index >= start_date) & (obs_data.index <= end_date)
                    obs_period = obs_data[obs_mask]

                    # Round sim times if needed
                    sim_time_diff = sim_data.index[1] - sim_data.index[0] if len(sim_data) > 1 else pd.Timedelta(hours=1)
                    if pd.Timedelta(minutes=45) <= sim_time_diff <= pd.Timedelta(minutes=75):
                        sim_data.index = sim_data.index.round('h')

                    sim_mask = (sim_data.index >= start_date) & (sim_data.index <= end_date)
                    sim_period = sim_data[sim_mask]
                    logger.warning(f"WORKER DEBUG: After period filtering, sim_period mean = {float(sim_period.mean()):.2f}")
                else:
                    obs_period = obs_data
                    sim_period = sim_data
            except Exception as e:
                obs_period = obs_data
                sim_period = sim_data
        else:
            obs_period = obs_data
            sim_period = sim_data

        # Timezone alignment
        if obs_period.index.tz is not None and sim_period.index.tz is None:
            sim_period.index = sim_period.index.tz_localize(obs_period.index.tz)
        elif obs_period.index.tz is None and sim_period.index.tz is not None:
            obs_period.index = obs_period.index.tz_localize(sim_period.index.tz)

        # Resampling
        calibration_timestep = config.get('CALIBRATION_TIMESTEP', 'native').lower()
        if calibration_timestep != 'native':
            obs_period = resample_to_timestep(obs_period, calibration_timestep, logger)
            sim_period = resample_to_timestep(sim_period, calibration_timestep, logger)
            logger.warning(f"WORKER DEBUG: After resampling, sim_period mean = {float(sim_period.mean()):.2f}")

        # Alignment
        common_idx = obs_period.index.intersection(sim_period.index)
        logger.warning(f"WORKER DEBUG: obs_period has {len(obs_period)} points, sim_period has {len(sim_period)} points")
        logger.warning(f"WORKER DEBUG: obs_period type: {type(obs_period)}, sim_period type: {type(sim_period)}")
        logger.warning(f"WORKER DEBUG: Common index has {len(common_idx)} points")
        if len(common_idx) == 0:
            logger.warning("WORKER DEBUG: No common timesteps between sim and obs")
            return None

        # Ensure we're working with Series
        if isinstance(obs_period, pd.DataFrame):
            obs_period = obs_period.squeeze()
        if isinstance(sim_period, pd.DataFrame):
            sim_period = sim_period.squeeze()

        obs_common = pd.to_numeric(obs_period.loc[common_idx], errors='coerce')
        sim_common = pd.to_numeric(sim_period.loc[common_idx], errors='coerce')
        logger.warning(f"WORKER DEBUG: After intersection, sim_common mean = {float(sim_common.mean()):.2f}")

        # Final cleaning
        valid = ~(obs_common.isna() | sim_common.isna() | (obs_common < -900) | (sim_common < -900))
        obs_valid = obs_common[valid]
        sim_valid = sim_common[valid]

        if len(obs_valid) < 10:
            return None

        # Calculate metrics
        try:
            mean_obs = obs_valid.mean()
            nse_num = ((obs_valid - sim_valid) ** 2).sum()
            nse_den = ((obs_valid - mean_obs) ** 2).sum()
            nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan

            r = obs_valid.corr(sim_valid)
            alpha = sim_valid.std() / obs_valid.std() if obs_valid.std() != 0 else np.nan
            beta = sim_valid.mean() / mean_obs if mean_obs != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

            rmse = np.sqrt(((obs_valid - sim_valid) ** 2).mean())
            mae = (obs_valid - sim_valid).abs().mean()
            pbias = 100 * (sim_valid.sum() - obs_valid.sum()) / obs_valid.sum() if obs_valid.sum() != 0 else np.nan

            logger.warning(f"WORKER DEBUG: Final KGE = {kge:.4f} (obs_mean={mean_obs:.2f}, sim_mean={float(sim_valid.mean()):.2f})")

            return {
                'Calib_NSE': nse, 'Calib_KGE': kge, 'Calib_RMSE': rmse, 'Calib_MAE': mae, 'Calib_PBIAS': pbias,
                'Calib_r': r, 'Calib_alpha': alpha, 'Calib_beta': beta, 'Calib_correlation': r,
                'NSE': nse, 'KGE': kge, 'RMSE': rmse, 'MAE': mae, 'PBIAS': pbias, 'correlation': r
            }
        except Exception as e:
            return None

    except ImportError as e:
        logger.warning(f"WORKER DEBUG: Import error: {str(e)}")
        return None
    except FileNotFoundError as e:
        logger.warning(f"WORKER DEBUG: File not found error: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"WORKER DEBUG: Error in inline metrics calculation: {str(e)}")
        import traceback
        logger.warning(f"WORKER DEBUG: Full traceback: {traceback.format_exc()}")
        return None


def _calculate_multitarget_objectives(task: Dict, summa_dir: str, mizuroute_dir: str,
                                       config: Dict, project_dir: str, logger) -> List[float]:
    """
    Calculate objectives for multi-target optimization in worker process.

    This function should be called from _evaluate_parameters_worker when
    multi_target_mode is True.

    Parameters
    ----------
    task : Dict
        Task dictionary containing:
        - multi_target_mode: bool
        - primary_target_type: str
        - secondary_target_type: str
        - primary_metric: str
        - secondary_metric: str
    summa_dir : str
        Path to SUMMA simulation directory
    mizuroute_dir : str
        Path to mizuRoute simulation directory
    config : Dict
        Configuration dictionary
    project_dir : str
        Project directory path
    logger : Logger
        Logger instance

    Returns
    -------
    List[float]
        [objective1, objective2] values
    """
    from ...calibration_targets import (
        StreamflowTarget, SnowTarget, GroundwaterTarget, ETTarget, SoilMoistureTarget, TWSTarget
    )
    from pathlib import Path

    # Convert to Path objects for safety
    summa_dir = Path(summa_dir)
    mizuroute_dir = Path(mizuroute_dir) if mizuroute_dir else None

    project_path = Path(project_dir)

    def create_target(target_type: str):
        """Create calibration target by type name."""
        target_type = target_type.lower()

        if target_type in ['streamflow', 'flow', 'discharge']:
            return StreamflowTarget(config, project_path, logger)
        elif target_type in ['swe', 'sca', 'snow_depth', 'snow']:
            return SnowTarget(config, project_path, logger)
        elif target_type in ['gw_depth', 'gw_grace', 'groundwater', 'gw']:
            return GroundwaterTarget(config, project_path, logger)
        elif target_type in ['et', 'latent_heat', 'evapotranspiration']:
            return ETTarget(config, project_path, logger)
        elif target_type in ['sm_point', 'sm_smap', 'sm_esa', 'sm_ismn', 'soil_moisture', 'sm']:
            return SoilMoistureTarget(config, project_path, logger)
        elif target_type in ['tws', 'grace', 'grace_tws', 'total_storage', 'stor_grace']:
            return TWSTarget(config, project_path, logger)
        else:
            # Default to streamflow
            return StreamflowTarget(config, project_path, logger)

    def extract_metric(metrics: Dict, metric_name: str) -> float:
        """Extract specific metric from metrics dictionary."""
        if not metrics:
            return -1.0

        # Try exact match
        if metric_name in metrics:
            val = metrics[metric_name]
            return val if val is not None and not np.isnan(val) else -1.0

        # Try with Calib_ prefix
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics:
            val = metrics[calib_key]
            return val if val is not None and not np.isnan(val) else -1.0

        # Try suffix match
        for key, value in metrics.items():
            if key.endswith(f"_{metric_name}"):
                return value if value is not None and not np.isnan(value) else -1.0

        return -1.0

    try:
        if task.get('multi_target_mode', False):
            # Multi-target mode: use two different calibration targets
            primary_target_type = task.get('primary_target_type', 'streamflow')
            secondary_target_type = task.get('secondary_target_type', 'gw_depth')
            primary_metric = task.get('primary_metric', 'KGE')
            secondary_metric = task.get('secondary_metric', 'KGE')

            # Create targets
            primary_target = create_target(primary_target_type)
            secondary_target = create_target(secondary_target_type)

            # Debug: Log target types being used
            logger.debug(f"[MULTI-TARGET DEBUG] Primary target: {primary_target_type} -> {type(primary_target).__name__}")
            logger.debug(f"[MULTI-TARGET DEBUG] Secondary target: {secondary_target_type} -> {type(secondary_target).__name__}")
            logger.debug(f"[MULTI-TARGET DEBUG] Secondary has calculate_metrics: {hasattr(secondary_target, 'calculate_metrics')}")

            # Check if TWSEvaluator's calculate_metrics is being used
            if 'tws' in secondary_target_type.lower() or 'grace' in secondary_target_type.lower() or 'stor_grace' in secondary_target_type.lower():
                from symfluence.evaluation.evaluators.tws import TWSEvaluator
                is_tws = isinstance(secondary_target, TWSEvaluator)
                logger.debug(f"[MULTI-TARGET DEBUG] Secondary is TWSEvaluator instance: {is_tws}")
                logger.debug(f"[MULTI-TARGET DEBUG] Secondary MRO: {[c.__name__ for c in type(secondary_target).__mro__]}")

            # Calculate metrics
            primary_metrics = primary_target.calculate_metrics(
                summa_dir,
                mizuroute_dir=mizuroute_dir
            )
            secondary_metrics = secondary_target.calculate_metrics(
                summa_dir,
                mizuroute_dir=mizuroute_dir
            )

            obj1 = extract_metric(primary_metrics, primary_metric)
            obj2 = extract_metric(secondary_metrics, secondary_metric)

        else:
            # Default mode: NSE and KGE from same target
            target_type = task.get('calibration_variable', 'streamflow')
            target = create_target(target_type)

            metrics = target.calculate_metrics(
                summa_dir,
                mizuroute_dir=mizuroute_dir
            )

            obj1 = extract_metric(metrics, 'NSE')
            obj2 = extract_metric(metrics, 'KGE')

        return [obj1, obj2]

    except Exception as e:
        logger.warning(f"Multi-target objective calculation failed: {e}")
        return [-1.0, -1.0]
