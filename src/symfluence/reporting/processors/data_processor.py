"""
Processor for loading and preparing data for visualization.

This module centralizes data loading logic for observations and model outputs,
handling various file formats and data transformations.
"""

import pandas as pd  # type: ignore
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from symfluence.reporting.core.shapefile_helper import ShapefileHelper


class DataProcessor:
    """
    Handles loading and preparation of observation and simulation data.

    Supports:
    - Loading streamflow observations from CSV
    - Loading model outputs from NetCDF (xarray)
    - Unit conversions (m/s to m³/s)
    - Data alignment and resampling
    - Snow data loading
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the data processor.

        Args:
            config: SYMFLUENCE configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self._shapefile_helper = ShapefileHelper(config, logger, self.project_dir)

    def load_streamflow_observations(
        self,
        obs_files: List[Tuple[str, str]],
        resample_freq: Optional[str] = 'h'
    ) -> List[Tuple[str, pd.Series]]:
        """
        Load streamflow observations from CSV files.

        Args:
            obs_files: List of (name, filepath) tuples
            resample_freq: Frequency to resample to ('h' for hourly, 'D' for daily, etc.)

        Returns:
            List of (name, Series) tuples

        Note:
            Expected CSV format:
            - 'datetime' column with timestamps
            - 'discharge_cms' column with flow values in m³/s
        """
        obs_data = []

        for obs_name, obs_file in obs_files:
            try:
                df = pd.read_csv(obs_file, parse_dates=['datetime'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)

                # Extract discharge column
                if 'discharge_cms' in df.columns:
                    series = df['discharge_cms']
                else:
                    self.logger.warning(
                        f"Column 'discharge_cms' not found in {obs_file}. "
                        f"Available columns: {df.columns.tolist()}"
                    )
                    continue

                # Resample if requested
                if resample_freq:
                    series = series.resample(resample_freq).mean()

                obs_data.append((obs_name, series))
                self.logger.info(f"Loaded observation: {obs_name} ({len(series)} timesteps)")

            except Exception as e:
                self.logger.warning(f"Could not read observation file {obs_file}: {str(e)}")
                continue

        if not obs_data:
            self.logger.error("No observation data could be loaded")

        return obs_data

    def load_lumped_model_outputs(
        self,
        model_outputs: List[Tuple[str, str]],
        variable_name: str = 'averageRoutedRunoff',
        convert_units: bool = True
    ) -> List[Tuple[str, pd.Series]]:
        """
        Load outputs from lumped watershed models (e.g., SUMMA lumped).

        Args:
            model_outputs: List of (model_name, filepath) tuples
            variable_name: NetCDF variable to extract (default: 'averageRoutedRunoff')
            convert_units: Whether to convert from m/s to m³/s using basin area

        Returns:
            List of (model_name, Series) tuples
        """
        import xarray as xr  # type: ignore
        sim_data = []

        for sim_name, sim_file in model_outputs:
            try:
                ds = xr.open_dataset(sim_file)

                # Extract variable
                if variable_name not in ds:
                    self.logger.error(f"Variable '{variable_name}' not found in {sim_file}")
                    continue

                runoff_series = ds[variable_name].to_pandas()

                # Handle different xarray/pandas output formats
                if isinstance(runoff_series, pd.DataFrame):
                    # If it's a DataFrame with gru/hru as columns, use the first one
                    if 'gru' in ds[variable_name].dims or 'hru' in ds[variable_name].dims:
                        runoff_series = runoff_series.iloc[:, 0]
                    else:
                        runoff_series = runoff_series.iloc[:, 0]

                # Convert units if requested
                if convert_units:
                    area_m2 = self._get_basin_area()
                    if area_m2 is not None:
                        runoff_series = runoff_series * area_m2
                        self.logger.info(
                            f"Converted runoff from m/s to m³/s using basin area: {area_m2} m²"
                        )

                sim_data.append((sim_name, runoff_series))
                self.logger.info(f"Loaded simulation: {sim_name} ({len(runoff_series)} timesteps)")

            except Exception as e:
                self.logger.warning(f"Could not read simulation file {sim_file}: {str(e)}")
                continue

        if not sim_data:
            self.logger.error("No simulation data could be loaded")

        return sim_data

    def load_distributed_model_outputs(
        self,
        model_file: str,
        variable_name: str = 'streamflow',
        reach_id: Optional[int] = None
    ) -> Optional[pd.Series]:
        """
        Load outputs from distributed/routing models (e.g., MizuRoute, T-Route).

        Args:
            model_file: Path to NetCDF file
            variable_name: Variable to extract (default: 'streamflow')
            reach_id: Reach/segment ID to extract (if None, uses first reach)

        Returns:
            Time series for the specified reach, or None if loading fails
        """
        import xarray as xr  # type: ignore
        try:
            ds = xr.open_dataset(model_file)

            if variable_name not in ds:
                self.logger.error(f"Variable '{variable_name}' not found in {model_file}")
                return None

            data = ds[variable_name]

            # Handle different dimension structures
            if 'seg' in data.dims:
                # MizuRoute format
                if reach_id is not None:
                    series = data.sel(seg=reach_id).to_pandas()
                else:
                    series = data.isel(seg=0).to_pandas()
            elif 'feature_id' in data.dims:
                # T-Route format
                if reach_id is not None:
                    series = data.sel(feature_id=reach_id).to_pandas()
                else:
                    series = data.isel(feature_id=0).to_pandas()
            else:
                # Unknown format, try to extract as-is
                series = data.to_pandas()
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]

            self.logger.info(f"Loaded distributed model output ({len(series)} timesteps)")
            return series

        except Exception as e:
            self.logger.error(f"Could not read distributed model file {model_file}: {str(e)}")
            return None

    def load_snow_data(
        self,
        snow_file: str,
        variable_name: str = 'scalarSWE',
        hru_index: int = 0
    ) -> Optional[pd.Series]:
        """
        Load snow water equivalent data from model output.

        Args:
            snow_file: Path to NetCDF file
            variable_name: Variable name (default: 'scalarSWE' for SUMMA)
            hru_index: HRU index to extract (for distributed models)

        Returns:
            Time series of snow data, or None if loading fails
        """
        import xarray as xr  # type: ignore
        try:
            ds = xr.open_dataset(snow_file)

            if variable_name not in ds:
                self.logger.warning(f"Variable '{variable_name}' not found in {snow_file}")
                return None

            data = ds[variable_name]

            # Extract for specific HRU if multi-dimensional
            if 'hru' in data.dims or 'gru' in data.dims:
                series = data.isel(hru=hru_index).to_pandas() if 'hru' in data.dims else data.isel(gru=hru_index).to_pandas()
            else:
                series = data.to_pandas()

            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            self.logger.info(f"Loaded snow data ({len(series)} timesteps)")
            return series

        except Exception as e:
            self.logger.error(f"Could not read snow file {snow_file}: {str(e)}")
            return None

    def determine_common_time_range(
        self,
        datasets: List[Tuple[str, pd.Series]]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Determine overlapping time range across multiple datasets.

        Args:
            datasets: List of (name, Series) tuples

        Returns:
            Tuple of (start_date, end_date)

        Raises:
            ValueError: If no datasets provided or no overlap exists
        """
        if not datasets:
            raise ValueError("No datasets provided")

        start_date = max([data.index.min() for _, data in datasets])
        end_date = min([data.index.max() for _, data in datasets])

        if start_date >= end_date:
            raise ValueError("No overlapping time range found across datasets")

        self.logger.info(f"Common time range: {start_date} to {end_date}")
        return start_date, end_date

    def _get_basin_area(self) -> Optional[float]:
        """
        Get total basin area from river basin shapefile.

        Returns:
            Basin area in m², or None if not available
        """
        return self._shapefile_helper.get_basin_area()

    def align_multiple_datasets(
        self,
        datasets: List[Tuple[str, pd.Series]],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Align multiple time series datasets to common time range.

        Args:
            datasets: List of (name, Series) tuples
            start_date: Optional start date for alignment
            end_date: Optional end date for alignment

        Returns:
            DataFrame with aligned datasets as columns

        Example:
            >>> processor = DataProcessor(config, logger)
            >>> datasets = [('Obs', obs_series), ('Sim1', sim1_series)]
            >>> aligned = processor.align_multiple_datasets(datasets)  # doctest: +SKIP
        """
        if not datasets:
            return pd.DataFrame()

        # Create DataFrame with all datasets
        df = pd.concat(
            [series for _, series in datasets],
            axis=1,
            keys=[name for name, _ in datasets]
        )

        # Filter by date range if specified
        if start_date is not None:
            df = df[df.index >= start_date]

        if end_date is not None:
            df = df[df.index <= end_date]

        self.logger.info(f"Aligned {len(datasets)} datasets ({len(df)} timesteps)")
        return df

    def read_results_file(self) -> pd.DataFrame:
        """
        Read simulation results and observed streamflow from standard results file.
        
        Returns:
            DataFrame containing aligned simulation and observation data.
            
        Raises:
            FileNotFoundError: If results file or observations file is missing.
            ValueError: If data is empty or invalid.
        """
        try:
            # Read simulation results
            results_file = self.project_dir / "results" / f"{self.config.get('EXPERIMENT_ID')}_results.csv"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")
            
            # Read the CSV file
            sim_df = pd.read_csv(results_file)
            
            # Check if the DataFrame is empty
            if sim_df.empty:
                self.logger.error("Results file is empty")
                raise ValueError("Results file contains no data")
            
            # Convert time column to datetime and set as index
            if 'time' in sim_df.columns:
                sim_df['time'] = pd.to_datetime(sim_df['time'])
                sim_df.set_index('time', inplace=True)
            elif 'datetime' in sim_df.columns:
                sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
                sim_df.set_index('datetime', inplace=True)
            elif 'Unnamed: 0' in sim_df.columns:
                # Handle case where dates are in 'Unnamed: 0' column
                sim_df['datetime'] = pd.to_datetime(sim_df['Unnamed: 0'])
                sim_df.set_index('datetime', inplace=True)
                sim_df.drop('Unnamed: 0', axis=1, errors='ignore')
            elif sim_df.index.name is None and isinstance(sim_df.index, pd.Index):
                # Try to convert the unnamed index to datetime
                try:
                    sim_df.index = pd.to_datetime(sim_df.index)
                except (ValueError, TypeError):
                    raise ValueError("Index cannot be converted to datetime format")
            else:
                raise ValueError("No time or datetime column found in results file")
            
            # Read observations
            obs_file_path = self.config.get('OBSERVATIONS_PATH')
            if obs_file_path == 'default' or obs_file_path is None:
                obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"
            else:
                obs_file_path = Path(obs_file_path)

            if not obs_file_path.exists():
                self.logger.warning(f"Observations file not found: {obs_file_path}")
                # If observations are missing, just return simulations
                return sim_df

            obs_df = pd.read_csv(obs_file_path)
            
            if obs_df.empty:
                self.logger.warning("Observations file is empty")
                return sim_df
            
            # Process observations
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'], format='mixed', dayfirst=True)
            obs_df.set_index('datetime', inplace=True)
            
            # Resample to daily if needed (assuming results are daily based on original logic)
            # Original logic resampled observations to daily. Let's keep that default but be mindful.
            obs_series = obs_df['discharge_cms'].resample('D').mean()
            
            # Combine into single dataframe
            results_df = sim_df.copy()
            results_df['Observed'] = obs_series
            
            # Skip first year (spin-up period) - preserving original logic
            if len(results_df.index) > 0:
                start_time = results_df.index[0]
                end_time = results_df.index[-1]
                duration_days = (end_time - start_time).days
                
                if duration_days > 365:
                    first_year = results_df.index[0].year
                    start_date = pd.Timestamp(year=first_year + 1, month=1, day=1)
                    results_df = results_df[results_df.index >= start_date]
                else:
                    self.logger.warning(f"Simulation duration ({duration_days} days) is short. Skipping first year removal to ensure data availability.")
            
            self.logger.info(f"Data period: {results_df.index[0]} to {results_df.index[-1]}")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error reading results: {str(e)}")
            raise
