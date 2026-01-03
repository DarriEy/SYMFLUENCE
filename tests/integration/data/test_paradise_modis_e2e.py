import pytest
import os
import yaml
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from symfluence.core import SYMFLUENCE
from symfluence.utils.data.data_manager import DataManager

@pytest.mark.integration
def test_paradise_modis_full_e2e(tmp_path):
    """
    E2E test for Paradise point-scale using REAL data pipeline.
    Downloads ERA5 and SNOTEL data via live APIs.
    Calibrates against both SWE and SCA (multivariate).
    """
    # Configuration for Paradise point-scale - Live Mode
    config_data = {
        # Global
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(Path(__file__).parents[3]),
        'DOMAIN_NAME': 'paradise_e2e',
        'EXPERIMENT_ID': 'multivariate_test',
        'EXPERIMENT_TIME_START': '2024-01-01 00:00',
        'EXPERIMENT_TIME_END': '2024-01-31 23:00',
        'CALIBRATION_PERIOD': '2024-01-01, 2024-01-15',
        'EVALUATION_PERIOD': '2024-01-16, 2024-01-31',
        'SPINUP_PERIOD': '2024-01-01, 2024-01-01',
        'MPI_PROCESSES': 1,
        'FORCE_RUN_ALL_STEPS': False,
        
        # Domain (Paradise SNOTEL coordinates)
        'POUR_POINT_COORDS': '46.78/-121.75',
        'BOUNDING_BOX_COORDS': '46.781/-121.751/46.779/-121.749',
        'DOMAIN_DEFINITION_METHOD': 'point',
        'ROUTING_DELINEATION': 'lumped',
        'DOMAIN_DISCRETIZATION': 'GRUs',
        'GEOFABRIC_TYPE': 'na',
        'LUMPED_WATERSHED_METHOD': 'TauDEM',
        
        # Forcing
        'DATA_ACCESS': 'cloud',
        'FORCING_DATASET': 'ERA5',
        'FORCING_TIME_STEP_SIZE': 3600,
        
        # Model
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'ROUTING_MODEL': 'none',
        
        # MODIS Snow Settings
        'ADDITIONAL_OBSERVATIONS': 'MODIS_SNOW, SNOTEL',
        'DOWNLOAD_MODIS_SNOW': True,
        'MODIS_SNOW_PRODUCT': 'MOD10A1.006',
        
        # SNOTEL Settings
        'DOWNLOAD_SNOTEL': True,
        'SNOTEL_STATION': '679', # Paradise station ID
        'SNOTEL_STATE': 'WA',
        
        # Disable others
        'DOWNLOAD_USGS_DATA': False,
        'DOWNLOAD_WSC_DATA': False,
        'DOWNLOAD_USGS_GW': False,
        'SUPPLEMENT_FORCING': False,
        
        # Optimization
        'OPTIMIZATION_METHODS': ['iteration'],
        'OPTIMIZATION_TARGET': 'multivariate',
        'OBJECTIVE_WEIGHTS': {
            'SWE': 0.5,
            'SCA': 0.5
        },
        'OBJECTIVE_METRICS': {
            'SWE': 'kge',
            'SCA': 'corr'
        },
        'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
        'NUMBER_OF_ITERATIONS': 1,
        'POPULATION_SIZE': 1,
        'OPTIMIZATION_METRIC': 'RMSE',
        'PARAMS_TO_CALIBRATE': 'albedoMax'
    }
    
    config_file = tmp_path / "paradise_live_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    # 1. Initialize SYMFLUENCE
    sym = SYMFLUENCE(config_path=config_file)
    
    # 2. Setup Project and Domain
    sym.managers['project'].setup_project()
    sym.managers['domain'].define_domain()
    
    # Mocking DEM/Soil/Landcover for discretization to avoid multi-GB downloads
    dem_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}" / "attributes" / "elevation" / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)
    mock_dem = dem_dir / f"domain_{config_data['DOMAIN_NAME']}_elv.tif"
    
    import rasterio
    import numpy as np
    from rasterio.transform import from_origin
    transform = from_origin(-121.751, 46.781, 0.001, 0.001)
    with rasterio.open(
        mock_dem, 'w', driver='GTiff', height=10, width=10, count=1,
        dtype='float32', crs='EPSG:4326', transform=transform
    ) as dst:
        dst.write(np.full((10, 10), 1600.0, dtype='float32'), 1)

    for attr in ['soilclass', 'landclass']:
        attr_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}" / "attributes" / attr
        attr_dir.mkdir(parents=True, exist_ok=True)
        fname = f"domain_{config_data['DOMAIN_NAME']}_{'soil_classes' if attr == 'soilclass' else 'land_classes'}.tif"
        with rasterio.open(
            attr_dir / fname, 'w', driver='GTiff', height=10, width=10, count=1,
            dtype='uint8', crs='EPSG:4326', transform=transform
        ) as dst:
            dst.write(np.full((10, 10), 1, dtype='uint8'), 1)

    sym.managers['domain'].discretize_domain()
    
    # 3. Full Data Pipeline (REAL ERA5 and SNOTEL, Mocked MODIS download)
    with patch('symfluence.utils.data.acquisition.handlers.modis.MODISSnowAcquirer.download') as mock_modis:
        # Create a real-looking MODIS NetCDF for the handler to process
        snow_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}" / "observations" / "snow"
        snow_dir.mkdir(parents=True, exist_ok=True)
        mock_nc = snow_dir / f"{config_data['DOMAIN_NAME']}_{config_data['MODIS_SNOW_PRODUCT']}_raw.nc"
        
        import xarray as xr
        times = pd.date_range(config_data['EXPERIMENT_TIME_START'], config_data['EXPERIMENT_TIME_END'], freq='D')
        ds = xr.Dataset(
            data_vars={'NDSI_Snow_Cover': (('time', 'lat', 'lon'), np.random.rand(len(times), 1, 1))},
            coords={'time': times, 'lat': [46.78], 'lon': [-121.75]}
        )
        ds.to_netcdf(mock_nc)
        mock_modis.return_value = mock_nc
        
        sym.managers['data'].acquire_observations()
        sym.managers['data'].process_observed_data()
    
    sym.managers['data'].acquire_forcings()
    # Run forcing remapping (using the new point-scale bypass)
    sym.managers['data'].run_model_agnostic_preprocessing()
    
    # Verify remapped forcing has all variables
    remapped_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}" / "forcing" / "basin_averaged_data"
    remapped_file = list(remapped_dir.glob("*.nc"))[0]
    import xarray as xr
    with xr.open_dataset(remapped_file) as ds:
        required = ['airtemp', 'airpres', 'pptrate', 'SWRadAtm', 'LWRadAtm', 'windspd', 'spechum']
        for var in required:
            assert var in ds.data_vars, f"Missing required variable {var} in remapped forcing"
        assert ds['airtemp'].shape == (len(ds.time), 1), f"Unexpected shape for airtemp: {ds['airtemp'].shape}"

    # Verify results
    obs_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}" / "observations"
    snotel_file = obs_dir / "snow" / "swe" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_swe_processed.csv"
    modis_file = obs_dir / "snow" / "preprocessed" / f"{config_data['DOMAIN_NAME']}_modis_snow_processed.csv"
    
    assert snotel_file.exists(), "REAL SNOTEL data acquisition failed"
    assert modis_file.exists(), "MODIS processing failed"
    
    # 4. Run Model Preprocessing
    sym.managers['model'].preprocess_models()

    # 5. Run Calibration (Mocking SUMMA execution)
    with patch('symfluence.utils.models.summa.runner.SummaRunner.run_summa', return_value=True):
        sim_dir = Path(config_data['SYMFLUENCE_DATA_DIR']) / f"domain_{config_data['DOMAIN_NAME']}" / "simulations" / config_data['EXPERIMENT_ID'] / "SUMMA"
        sim_dir.mkdir(parents=True, exist_ok=True)
        sim_nc = sim_dir / f"{config_data['EXPERIMENT_ID']}_day.nc"
        
        import xarray as xr
        times = pd.date_range(config_data['EXPERIMENT_TIME_START'], config_data['EXPERIMENT_TIME_END'], freq='D')
        s_ds = xr.Dataset(
            data_vars={
                'scalarSWE': (('time', 'hru'), np.random.rand(len(times), 1) * 100),
                'scalarGroundSnowFraction': (('time', 'hru'), np.random.rand(len(times), 1))
            },
            coords={'time': times, 'hru': [1]}
        )
        s_ds.to_netcdf(sim_nc)
        
        results_file = sym.managers['optimization'].calibrate_model()
        
    assert results_file is not None, "Multivariate calibration should produce results"
    print("\nE2E Multivariate Pipeline passed successfully!")
