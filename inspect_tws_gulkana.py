import xarray as xr
import os
import numpy as np

file_path = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Gulkana/simulations/run_nsga-ii/process_0/simulations/dual_calibration/SUMMA/proc_00_dual_calibration_day.nc"

# Find a successful file if the first one is empty
if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
    # Try final evaluation dir
    file_path = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Gulkana/optimization/nsga-ii_dual_calibration/final_evaluation/glacier_pipeline_test_day.nc"

if os.path.exists(file_path):
    try:
        ds = xr.open_dataset(file_path)
        if ds.sizes['time'] == 0:
            print(f"File {file_path} has zero time steps.")
            # Search for ANY non-empty .nc file in simulations
            import glob
            nc_files = glob.glob("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Gulkana/simulations/run_nsga-ii/**/proc*_day.nc", recursive=True)
            for f in nc_files:
                test_ds = xr.open_dataset(f)
                if test_ds.sizes['time'] > 0:
                    ds = test_ds
                    file_path = f
                    print(f"Found non-empty file: {f}")
                    break
        
        if ds.sizes['time'] == 0:
            print("Could not find any non-empty output files.")
            exit(0)

        storage_vars = ['scalarSWE', 'scalarCanopyWat', 'scalarTotalSoilWat', 'scalarAquiferStorage', 'glacMass4AreaChange']
        
        print(f"File: {os.path.basename(file_path)}")
        # Calculate Total Storage (Simulated TWS)
        total_storage = None
        available_vars = []
        
        for var in storage_vars:
            if var in ds:
                data = ds[var].values
                if data.size > 0:
                    print(f"\nVariable: {var}")
                    print(f"  Units: {ds[var].attrs.get('units', 'unknown')}")
                    print(f"  Mean: {np.nanmean(data):.4f}")
                    print(f"  Range: {np.nanmax(data) - np.nanmin(data):.4f}")
                    
                    val = data.copy()
                    if 'aquifer' in var.lower():
                        val = val * 1000.0 # m to mm
                    
                    if val.ndim > 1:
                        # Mean across HRUs
                        val = np.nanmean(val, axis=tuple(range(1, val.ndim)))
                    
                    if total_storage is None:
                        total_storage = val
                    else:
                        total_storage += val
                    available_vars.append(var)
        
        if total_storage is not None:
            print(f"\nTotal Storage (Simulated TWS using {', '.join(available_vars)}):")
            print(f"  Mean: {np.nanmean(total_storage):.4f} mm")
            print(f"  Std:  {np.nanstd(total_storage):.4f} mm")
            print(f"  Range: {np.nanmax(total_storage) - np.nanmin(total_storage):.4f} mm")
            
            # Anomaly
            anomaly = total_storage - np.nanmean(total_storage)
            print(f"  Anomaly Range: {np.nanmax(anomaly) - np.nanmin(anomaly):.4f} mm")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {file_path}")