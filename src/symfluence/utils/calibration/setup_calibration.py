#!/usr/bin/env python3
"""
Setup SUMMA calibration with observational discharge data.

This script prepares a domain for calibration by:
1. Loading observational discharge data
2. Creating calibration configuration
3. Setting up parameter bounds
4. Preparing optimization framework
"""

import pandas as pd
import yaml
import xarray as xr
from pathlib import Path
from datetime import datetime
import numpy as np


class CalibrationSetup:
    """Setup calibration for a SUMMA domain."""

    def __init__(self, config_path):
        """Initialize calibration setup."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.domain_name = self.config.get('DOMAIN_NAME')
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.domain_name}"
        self.obs_dir = self.data_dir / 'observations'

    def load_observations(self):
        """Load observational discharge data."""
        # Find discharge CSV file
        discharge_files = list((self.obs_dir / 'discharge' / 'raw').glob('*.csv'))

        if not discharge_files:
            raise FileNotFoundError(f"No discharge observations found in {self.obs_dir / 'discharge' / 'raw'}")

        obs_file = discharge_files[0]
        print(f"Loading observations from: {obs_file}")

        # Load data
        df = pd.read_csv(obs_file, sep=';')

        # Debug: print columns
        print(f"Columns found: {list(df.columns)}")

        # Create datetime index - handle both possible column naming conventions
        if 'YYYY' in df.columns:
            df['date'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].rename(columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
        elif 'year' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        else:
            raise ValueError(f"Could not find date columns in {obs_file}")
        df = df.set_index('date')

        # Filter by quality if needed (for LamaH-Ice, qc_flag 40 is good)
        if 'qc_flag' in df.columns:
            df_good = df[df['qc_flag'] == 40]  # Only use good quality data
            print(f"Filtered from {len(df)} to {len(df_good)} records (quality code 40)")
            df = df_good

        print(f"Observation period: {df.index.min()} to {df.index.max()}")
        print(f"Mean discharge: {df['qobs'].mean():.2f} m³/s")
        print(f"Number of records: {len(df)}")

        return df

    def create_calibration_config(self, obs_df, calib_period=None, valid_period=None):
        """
        Create calibration configuration file.

        Parameters:
        -----------
        obs_df : pd.DataFrame
            Observational discharge data
        calib_period : tuple
            (start_date, end_date) for calibration period
        valid_period : tuple
            (start_date, end_date) for validation period
        """
        # Default periods if not specified
        if calib_period is None:
            # Use first 70% of data for calibration
            n_calib = int(len(obs_df) * 0.7)
            calib_start = obs_df.index[0]
            calib_end = obs_df.index[n_calib]
        else:
            calib_start, calib_end = calib_period

        if valid_period is None:
            # Use last 30% for validation
            n_calib = int(len(obs_df) * 0.7)
            valid_start = obs_df.index[n_calib + 1]
            valid_end = obs_df.index[-1]
        else:
            valid_start, valid_end = valid_period

        calib_config = {
            'domain': self.domain_name,
            'observations': {
                'variable': 'discharge',
                'units': 'm3/s',
                'file': str((self.obs_dir / 'discharge' / 'raw').resolve()),
                'quality_filter': {
                    'column': 'qc_flag',
                    'value': 40,
                    'description': 'Only use good quality observations'
                }
            },
            'periods': {
                'calibration': {
                    'start': calib_start.strftime('%Y-%m-%d'),
                    'end': calib_end.strftime('%Y-%m-%d'),
                    'n_days': (calib_end - calib_start).days
                },
                'validation': {
                    'start': valid_start.strftime('%Y-%m-%d'),
                    'end': valid_end.strftime('%Y-%m-%d'),
                    'n_days': (valid_end - valid_start).days
                },
                'warmup_days': 365  # 1 year warmup before calibration
            },
            'parameters': self.get_default_parameters(),
            'objective_function': {
                'primary': 'NSE',  # Nash-Sutcliffe Efficiency
                'secondary': ['RMSE', 'KGE', 'PBIAS'],  # Additional metrics to track
                'aggregation': 'daily',  # Daily aggregation for comparison
                'log_transform': False  # Set to True for log-NSE (better for low flows)
            },
            'optimization': {
                'algorithm': 'DDS',  # Dynamically Dimensioned Search
                'max_iterations': 1000,
                'seed': 42,
                'parallel': {
                    'enabled': True,
                    'n_workers': 4
                }
            },
            'output': {
                'results_dir': str(self.data_dir / 'calibration' / 'results'),
                'save_best_params': True,
                'save_all_trials': True,
                'plot_hydrograph': True,
                'plot_parameters': True
            }
        }

        # Save configuration
        calib_dir = self.data_dir / 'calibration'
        calib_dir.mkdir(parents=True, exist_ok=True)

        config_file = calib_dir / f'calibration_config_{self.domain_name}.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(calib_config, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Calibration configuration saved: {config_file}")
        print(f"  Calibration period: {calib_start.strftime('%Y-%m-%d')} to {calib_end.strftime('%Y-%m-%d')}")
        print(f"  Validation period: {valid_start.strftime('%Y-%m-%d')} to {valid_end.strftime('%Y-%m-%d')}")

        return calib_config, config_file

    def get_default_parameters(self):
        """
        Define default parameter bounds for SUMMA calibration.

        Returns commonly calibrated SUMMA parameters with physically reasonable bounds.
        """
        parameters = {
            'theta_sat': {
                'description': 'Soil porosity (volumetric)',
                'min': 0.3,
                'max': 0.6,
                'default': 0.45,
                'unit': 'm3/m3'
            },
            'theta_res': {
                'description': 'Residual volumetric water content',
                'min': 0.01,
                'max': 0.1,
                'default': 0.05,
                'unit': 'm3/m3'
            },
            'vGn_alpha': {
                'description': 'van Genuchten alpha parameter',
                'min': -10.0,
                'max': -0.5,
                'default': -5.0,
                'unit': 'm-1',
                'log_scale': True
            },
            'vGn_n': {
                'description': 'van Genuchten n parameter',
                'min': 1.05,
                'max': 3.0,
                'default': 1.5,
                'unit': '-'
            },
            'k_soil': {
                'description': 'Saturated hydraulic conductivity',
                'min': -8.0,
                'max': -2.0,
                'default': -5.0,
                'unit': 'm/s',
                'log_scale': True
            },
            'fieldCapacity': {
                'description': 'Soil field capacity',
                'min': 0.1,
                'max': 0.5,
                'default': 0.3,
                'unit': 'm3/m3'
            },
            'wiltingPoint': {
                'description': 'Wilting point',
                'min': 0.01,
                'max': 0.2,
                'default': 0.1,
                'unit': 'm3/m3'
            },
            'routingGammaShape': {
                'description': 'Shape parameter for gamma routing',
                'min': 1.5,
                'max': 5.0,
                'default': 2.5,
                'unit': '-'
            },
            'routingGammaScale': {
                'description': 'Scale parameter for gamma routing',
                'min': 1.0,
                'max': 100.0,
                'default': 20.0,
                'unit': 'hours'
            },
            'summerLAI': {
                'description': 'Summer leaf area index',
                'min': 0.5,
                'max': 7.0,
                'default': 3.5,
                'unit': 'm2/m2'
            },
            'heightCanopyTop': {
                'description': 'Height of canopy top',
                'min': 0.1,
                'max': 30.0,
                'default': 10.0,
                'unit': 'm'
            },
            'heightCanopyBottom': {
                'description': 'Height of canopy bottom',
                'min': 0.0,
                'max': 5.0,
                'default': 0.5,
                'unit': 'm'
            }
        }

        return parameters

    def prepare_observation_file(self, obs_df, output_file=None):
        """
        Prepare observations in format for calibration comparison.

        Saves as NetCDF for easy comparison with SUMMA output.
        """
        if output_file is None:
            output_file = self.data_dir / 'calibration' / 'observations_discharge.nc'

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to xarray Dataset
        ds = xr.Dataset({
            'discharge_obs': (['time'], obs_df['qobs'].values),
            'quality_flag': (['time'], obs_df['qc_flag'].values)
        }, coords={
            'time': obs_df.index.values
        })

        # Add attributes
        ds['discharge_obs'].attrs = {
            'long_name': 'Observed discharge',
            'units': 'm3/s',
            'standard_name': 'water_volume_transport_in_river_channel'
        }

        ds.attrs = {
            'title': f'{self.domain_name} discharge observations',
            'source': 'Observational data',
            'created': datetime.now().isoformat(),
            'domain': self.domain_name
        }

        # Save
        ds.to_netcdf(output_file)
        print(f"✓ Prepared observation file: {output_file}")

        return output_file


def main():
    """Main function for setting up calibration."""
    import argparse

    parser = argparse.ArgumentParser(description='Setup SUMMA calibration')
    parser.add_argument('--config', type=str, required=True, help='SYMFLUENCE config file')
    parser.add_argument('--calib-start', type=str, help='Calibration start date (YYYY-MM-DD)')
    parser.add_argument('--calib-end', type=str, help='Calibration end date (YYYY-MM-DD)')
    parser.add_argument('--valid-start', type=str, help='Validation start date (YYYY-MM-DD)')
    parser.add_argument('--valid-end', type=str, help='Validation end date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Initialize setup
    setup = CalibrationSetup(args.config)

    # Load observations
    print("\n" + "="*80)
    print("Loading Observational Data")
    print("="*80)
    obs_df = setup.load_observations()

    # Define periods
    calib_period = None
    valid_period = None

    if args.calib_start and args.calib_end:
        calib_period = (pd.to_datetime(args.calib_start), pd.to_datetime(args.calib_end))

    if args.valid_start and args.valid_end:
        valid_period = (pd.to_datetime(args.valid_start), pd.to_datetime(args.valid_end))

    # Create calibration configuration
    print("\n" + "="*80)
    print("Creating Calibration Configuration")
    print("="*80)
    calib_config, config_file = setup.create_calibration_config(obs_df, calib_period, valid_period)

    # Prepare observation file
    print("\n" + "="*80)
    print("Preparing Observation File")
    print("="*80)
    obs_file = setup.prepare_observation_file(obs_df)

    print("\n" + "="*80)
    print("✓ Calibration Setup Complete!")
    print("="*80)
    print(f"Configuration: {config_file}")
    print(f"Observations: {obs_file}")
    print(f"\nNext steps:")
    print(f"  1. Review calibration configuration in {config_file}")
    print(f"  2. Adjust parameter bounds if needed")
    print(f"  3. Run calibration using ostrich or similar optimizer")


if __name__ == "__main__":
    main()
