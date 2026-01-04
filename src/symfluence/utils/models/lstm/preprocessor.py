"""
LSTM Model Preprocessor.

Handles data loading, cleaning, normalization, and tensor conversion for the LSTM model.
"""

import glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import xarray as xr
import torch
from sklearn.preprocessing import StandardScaler

class LSTMPreprocessor:
    """
    Handles data preprocessing for the LSTM model.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger (Any): Logger instance.
        project_dir (Path): Project directory path.
        lookback (int): Number of time steps to look back.
        device (torch.device): Device to use for tensors.
        feature_scaler (StandardScaler): Scaler for input features.
        target_scaler (StandardScaler): Scaler for target variables.
        output_size (int): Number of output variables.
        target_names (List[str]): Names of target variables.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, project_dir: Path, device: torch.device):
        self.config = config
        self.config_dict = config # Alias for compatibility
        self.logger = logger
        self.project_dir = project_dir
        self.device = device
        self.lookback = config.get('LSTM_LOOKBACK', config.get('FLASH_LOOKBACK', 30))
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.output_size = 1
        self.target_names = ['streamflow']

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load forcing, streamflow, and snow data from disk.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Forcing, streamflow, and snow dataframes.
        """
        self.logger.info("Loading data for LSTM model")

        # Load forcing data
        forcing_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.logger.info(f"Looking for forcing files in: {forcing_path}")
        
        # Check if directory exists
        if not forcing_path.exists():
            self.logger.error(f"Forcing path does not exist: {forcing_path}")
        else:
            self.logger.info(f"Directory exists. Contents: {list(forcing_path.glob('*'))}")

        forcing_files = glob.glob(str(forcing_path / '*.nc'))
        self.logger.info(f"Found forcing files: {forcing_files}")

        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")

        forcing_files.sort()
        datasets = [xr.open_dataset(file) for file in forcing_files]
        combined_ds = xr.concat(datasets, dim='time', data_vars='all')
        forcing_df = combined_ds.to_dataframe().reset_index()

        required_vars = ['hruId', 'time', 'pptrate', 'SWRadAtm', 'LWRadAtm', 'airpres', 'airtemp', 'spechum', 'windspd']
        missing_vars = [var for var in required_vars if var not in forcing_df.columns]
        if missing_vars:
            raise ValueError(f"Missing required variables in forcing data: {missing_vars}")

        forcing_df['time'] = pd.to_datetime(forcing_df['time'])
        forcing_df = forcing_df.set_index(['time', 'hruId']).sort_index()

        # Load streamflow data
        streamflow_path = (
            self.project_dir / 'observations' / 'streamflow' / 'preprocessed' /
            f"{self.config_dict.get('DOMAIN_NAME')}_streamflow_processed.csv"
        )
        
        if not streamflow_path.exists():
            # Fallback for legacy naming
            legacy_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / "Bow_at_Banff_lumped_streamflow_processed.csv"
            if legacy_path.exists():
                streamflow_path = legacy_path
                self.logger.info(f"Using legacy streamflow path: {streamflow_path.name}")

        streamflow_df = pd.read_csv(streamflow_path, parse_dates=['datetime'], dayfirst=True)
        streamflow_df = streamflow_df.set_index('datetime').rename(columns={'discharge_cms': 'streamflow'})
        streamflow_df.index = pd.to_datetime(streamflow_df.index)

        # Load snow data
        snow_path = self.project_dir / 'observations' / 'snow' / 'preprocessed'
        snow_files = glob.glob(str(snow_path / f"{self.config_dict.get('DOMAIN_NAME')}_filtered_snow_observations.csv"))
        
        if snow_files:
            snow_df = pd.concat([pd.read_csv(file, parse_dates=['datetime'], dayfirst=True) for file in snow_files])
            # Aggregate snow data across all stations
            snow_df = snow_df.groupby('datetime')['snw'].mean().reset_index()
            snow_df['datetime'] = pd.to_datetime(snow_df['datetime'])
            snow_df = snow_df.set_index('datetime')
        else:
            self.logger.warning(f"No snow observation files found in {snow_path}. Using empty DataFrame.")
            snow_df = pd.DataFrame() # Return empty DF if not found, to handle gracefully

        # Ensure all datasets cover the same time period
        if not snow_df.empty:
            start_date = max(
                forcing_df.index.get_level_values('time').min(),
                streamflow_df.index.min(),
                snow_df.index.min()
            )
            end_date = min(
                forcing_df.index.get_level_values('time').max(),
                streamflow_df.index.max(),
                snow_df.index.max()
            )
            snow_df = snow_df.loc[start_date:end_date]
            snow_df = snow_df.resample('h').interpolate(method='linear')
        else:
            start_date = max(
                forcing_df.index.get_level_values('time').min(),
                streamflow_df.index.min()
            )
            end_date = min(
                forcing_df.index.get_level_values('time').max(),
                streamflow_df.index.max()
            )

        forcing_df = forcing_df.loc[pd.IndexSlice[start_date:end_date, :], :]
        streamflow_df = streamflow_df.loc[start_date:end_date]
        
        self.logger.info(f"Loaded forcing data with shape: {forcing_df.shape}")
        self.logger.info(f"Loaded streamflow data with shape: {streamflow_df.shape}")
        if not snow_df.empty:
            self.logger.info(f"Loaded snow data with shape: {snow_df.shape}")

        return forcing_df, streamflow_df, snow_df

    def process_data(
        self,
        forcing_df: pd.DataFrame,
        streamflow_df: pd.DataFrame,
        snow_df: Optional[pd.DataFrame] = None,
        fit_scalers: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, pd.DataFrame]:
        """
        Preprocess data for LSTM model (clean, scale, sequence).
        
        Args:
            forcing_df: DataFrame containing forcing data.
            streamflow_df: DataFrame containing streamflow data.
            snow_df: Optional DataFrame containing snow data.
            fit_scalers: Whether to fit new scalers or use existing ones.
            
        Returns:
            Tuple containing:
                - X (torch.Tensor): Input sequences.
                - y (torch.Tensor): Target values.
                - common_dates (pd.DatetimeIndex): Dates corresponding to the data.
                - features_avg (pd.DataFrame): Averaged features dataframe.
        """
        self.logger.info(f"Preprocessing data (fit_scalers={fit_scalers})")

        # Align the data
        common_dates = forcing_df.index.get_level_values('time').intersection(streamflow_df.index)
        if snow_df is not None and not snow_df.empty:
            common_dates = common_dates.intersection(snow_df.index)

        forcing_df = forcing_df.loc[pd.IndexSlice[common_dates, :], :]
        streamflow_df = streamflow_df.loc[common_dates]
        if snow_df is not None and not snow_df.empty:
            snow_df = snow_df.loc[common_dates]

        # Prepare features (forcing data)
        features = forcing_df.reset_index()
        feature_columns = features.columns.drop(
            ['time', 'hruId', 'hru', 'latitude', 'longitude']
            if 'time' in features.columns and 'hruId' in features.columns else []
        )

        # Average features across all HRUs for each timestep
        features_avg = forcing_df.groupby('time')[feature_columns].mean()

        # Scale features
        if fit_scalers:
            scaled_features = self.feature_scaler.fit_transform(features_avg)
        else:
            scaled_features = self.feature_scaler.transform(features_avg)
            
        scaled_features = np.clip(scaled_features, -10, 10)

        # Prepare targets (streamflow and optionally snow)
        if snow_df is not None and not snow_df.empty:
            targets = pd.concat([streamflow_df['streamflow'], snow_df['snw']], axis=1)
            targets.columns = ['streamflow', 'SWE']
            if fit_scalers:
                self.output_size = 2
                self.target_names = ['streamflow', 'SWE']
        else:
            targets = pd.DataFrame(streamflow_df['streamflow'], columns=['streamflow'])
            if fit_scalers:
                self.output_size = 1
                self.target_names = ['streamflow']

        # Scale targets
        if fit_scalers:
            scaled_targets = self.target_scaler.fit_transform(targets)
        else:
            scaled_targets = self.target_scaler.transform(targets)
            
        scaled_targets = np.clip(scaled_targets, -10, 10)

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.lookback):
            X.append(scaled_features[i:(i + self.lookback)])
            y.append(scaled_targets[i + self.lookback])

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        y_tensor = torch.FloatTensor(np.array(y)).to(self.device)

        self.logger.info(f"Preprocessed data shape: X: {X_tensor.shape}, y: {y_tensor.shape}")
        return X_tensor, y_tensor, pd.DatetimeIndex(common_dates), features_avg

    def set_scalers(self, feature_scaler, target_scaler, output_size, target_names):
        """Set scalers and metadata from a checkpoint."""
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.output_size = output_size
        self.target_names = target_names