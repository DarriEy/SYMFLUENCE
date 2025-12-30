"""
Minimal test to check CERRA forecast product download.
"""
import cdsapi
import xarray as xr
from pathlib import Path

# Initialize CDS API
c = cdsapi.Client()

# Test download - minimal request
test_file = Path("/tmp/cerra_forecast_test.nc")

request = {
    "level_type": "surface_or_atmosphere",
    "data_type": "reanalysis",  # Missing parameter!
    "product_type": "forecast",
    "variable": [
        "total_precipitation",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards"
    ],
    "year": "2010",
    "month": "01",
    "day": "01",
    "time": ["00:00", "03:00", "06:00"],
    "leadtime_hour": ["1"],
    "data_format": "netcdf",
}

print("Requesting CERRA forecast product...")
print(f"Variables: {request['variable']}")
print(f"Times: {request['time']}")
print(f"Leadtime: {request['leadtime_hour']}")

c.retrieve(
    "reanalysis-cerra-single-levels",
    request,
    str(test_file)
)

print(f"\n✓ Download complete: {test_file}")
print(f"File size: {test_file.stat().st_size / 1024 / 1024:.1f} MB")

# Inspect the file
print("\n=== Inspecting downloaded file ===")
with xr.open_dataset(test_file) as ds:
    print(f"\nDimensions: {dict(ds.dims)}")
    print(f"\nCoordinates:")
    for coord in ds.coords:
        print(f"  {coord}: {ds[coord].values}")

    print(f"\nData variables: {list(ds.data_vars)}")

    for var in ds.data_vars:
        if var not in ['latitude', 'longitude', 'expver', 'number']:
            data = ds[var].values
            print(f"\n{var}:")
            print(f"  Shape: {data.shape}")
            print(f"  Min: {data.min()}")
            print(f"  Max: {data.max()}")
            print(f"  Mean: {data.mean()}")
            print(f"  NaN count: {(data != data).sum()} / {data.size}")
            if (data == data).any():  # Has non-NaN values
                print(f"  Sample values: {data[data == data][:5]}")

print(f"\nTest file saved to: {test_file}")
