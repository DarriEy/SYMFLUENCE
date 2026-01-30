
import xarray as xr
import numpy as np

file_path = '/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/forcing/FUSE_input/Bow_at_Banff_lumped_era5_elev_bands.nc'
ds = xr.open_dataset(file_path)

print(ds)
print("\nDimensions:", ds.dims)
print("Area Frac values:", ds['area_frac'].values)
print("Prec Frac values:", ds['prec_frac'].values)
print("Mean Elev values:", ds['mean_elev'].values)

# Check for non-1 values
if np.any(ds['area_frac'].values != 1.0):
    print("WARNING: Area fraction is not 1.0 everywhere!")
if np.any(ds['prec_frac'].values != 1.0):
    print("WARNING: Precip fraction is not 1.0 everywhere!")
