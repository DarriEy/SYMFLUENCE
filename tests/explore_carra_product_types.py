"""
Explore CARRA product types to find precipitation and radiation variables.

CARRA provides different product types:
- analysis: Assimilated atmospheric state
- forecast: Model forecasts (may include accumulated variables like precip/radiation)
"""

import cdsapi
import xarray as xr
from pathlib import Path
import tempfile

c = cdsapi.Client()

# Test configuration
test_config = {
    "domain": "west_domain",
    "year": "2010",
    "month": "01",
    "day": "01",
    "time": ["00:00", "01:00"],
}

# Product type combinations to test
test_cases = [
    {
        "name": "Analysis - surface",
        "config": {
            "product_type": "analysis",
            "level_type": "surface_or_atmosphere",
        },
        "variables": ["total_precipitation", "surface_solar_radiation_downwards"],
    },
    {
        "name": "Forecast - surface",
        "config": {
            "product_type": "forecast",
            "level_type": "surface_or_atmosphere",
            "leadtime_hour": ["1", "2", "3"],  # Forecast lead times
        },
        "variables": ["total_precipitation", "surface_solar_radiation_downwards"],
    },
    {
        "name": "Forecast - surface - precip only",
        "config": {
            "product_type": "forecast",
            "level_type": "surface_or_atmosphere",
            "leadtime_hour": ["1"],
        },
        "variables": ["total_precipitation"],
    },
    {
        "name": "Forecast - surface - radiation only",
        "config": {
            "product_type": "forecast",
            "level_type": "surface_or_atmosphere",
            "leadtime_hour": ["1"],
        },
        "variables": ["surface_solar_radiation_downwards"],
    },
]


def test_case(case):
    """Test a specific product type configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {case['name']}")
    print(f"Config: {case['config']}")
    print(f"Variables: {case['variables']}")
    print('='*80)

    request = {**test_config, **case['config'], "variable": case['variables'], "data_format": "netcdf"}

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        output_file = Path(tmp.name)

    try:
        print(f"Submitting request to CDS API...")
        c.retrieve("reanalysis-carra-single-levels", request, str(output_file))

        print(f"✓ Download successful: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

        with xr.open_dataset(output_file) as ds:
            print(f"✓ Dataset opened successfully")
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Data variables: {list(ds.data_vars.keys())}")

            # Check variable details
            print(f"\n  Variable details:")
            for var in ds.data_vars:
                if hasattr(ds[var], 'long_name'):
                    print(f"    {var:20s} -> {ds[var].long_name}")
                if hasattr(ds[var], 'units'):
                    print(f"    {' '*20}    units: {ds[var].units}")

        output_file.unlink()
        return True, list(ds.data_vars.keys())

    except Exception as e:
        print(f"✗ Failed: {e}")
        if output_file.exists():
            output_file.unlink()
        return False, str(e)


def main():
    """Run systematic exploration of CARRA product types."""

    print("="*80)
    print("CARRA Product Type Explorer - Precipitation & Radiation")
    print("="*80)

    results = {}
    for case in test_cases:
        success, result = test_case(case)
        results[case['name']] = {"success": success, "result": result, "config": case['config']}

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, outcome in results.items():
        status = "✓ SUCCESS" if outcome["success"] else "✗ FAILED"
        print(f"{status:12s} {name:40s}")
        if outcome["success"]:
            print(f"             Variables: {outcome['result']}")
            print(f"             Config: {outcome['config']}")
        else:
            print(f"             Error: {outcome['result'][:100]}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR CARRA ACQUISITION")
    print("="*80)

    successful = {name: r for name, r in results.items() if r["success"]}

    if successful:
        print("✓ Working configurations found:")
        for name, result in successful.items():
            print(f"\n  {name}:")
            print(f"    Product type: {result['config']['product_type']}")
            print(f"    Level type: {result['config']['level_type']}")
            if 'leadtime_hour' in result['config']:
                print(f"    Lead times: {result['config']['leadtime_hour']}")
            print(f"    Variables: {result['result']}")
    else:
        print("✗ No working configurations found for precipitation/radiation")
        print("  CARRA may not provide these variables, or they require different parameters")
        print("\n  Alternative approaches:")
        print("  1. Use analysis product for met variables (temp, pressure, wind, humidity)")
        print("  2. Source precipitation/radiation from alternative dataset (ERA5, local obs)")
        print("  3. Check CARRA documentation for accumulated variable naming conventions")


if __name__ == "__main__":
    main()
