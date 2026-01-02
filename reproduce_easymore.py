import os
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import easymore

def _create_easymore_instance():
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")

# Dummy data for reproduction
data_dir = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped")
forcing_file = data_dir / "forcing/raw_data/domain_Bow_at_Banff_lumped_ERA5_merged_200401.nc"
target_shp = data_dir / "shapefiles/catchment/Bow_at_Banff_lumped_HRUs_GRUs.shp"
temp_dir = Path("./test_easymore_temp")
temp_dir.mkdir(exist_ok=True)

esmr = _create_easymore_instance()
esmr.case_name = "Bow_at_Banff_lumped_ERA5"
esmr.temp_dir = str(temp_dir.absolute()) + "/"

esmr.source_nc = str(forcing_file)
esmr.var_names = ['airtemp']
esmr.var_lat = 'latitude'
esmr.var_lon = 'longitude'
esmr.target_shp = str(target_shp)
esmr.target_shp_ID = 'HRU_ID'

esmr.only_create_remap_csv = True
esmr.save_csv = False

print("Running easymore...")
esmr.nc_remapper()

print("Contents of temp_dir:")
for f in temp_dir.glob("*"):
    print(f"  {f.name}")