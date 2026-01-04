import cdsapi
import logging

logging.basicConfig(level=logging.INFO)

c = cdsapi.Client()

print("Testing CDS API connectivity with FULL request...")

years = [str(y) for y in range(2015, 2020)]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
hours = [f"{h:02d}:00" for h in range(0, 24, 3)]

request = {
    "level_type": "surface_or_atmosphere",
    "product_type": "analysis",
    "variable": [
        "2m_temperature",
        "2m_relative_humidity",
        "surface_pressure",
        "10m_wind_speed"
    ],
    "year": years,
    "month": months,
    "day": days,
    "time": hours,
    "data_format": "netcdf",
    "data_type": "reanalysis"
}

print(f"Requesting CERRA for years: {years}")

try:
    c.retrieve(
        'reanalysis-cerra-single-levels',
        request,
        'test_cerra_full.nc'
    )
    print("\nSUCCESS: Access to full CERRA dataset is verified.")
    import os
    os.remove('test_cerra_full.nc')

except Exception as e:
    print(f"\nFAILURE: {e}")