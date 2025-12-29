"""
Explore CARRA CDS API to understand variable availability.

This script tests different CARRA variable requests to determine:
1. Which variables are actually available
2. Correct variable naming
3. Product type restrictions
4. Level type requirements
"""

import cdsapi
import xarray as xr
from pathlib import Path
import tempfile

# Initialize CDS client
c = cdsapi.Client()

# Test configuration - tiny request for fast iteration
test_config = {
    "domain": "west_domain",
    "product_type": "analysis",
    "level_type": "surface_or_atmosphere",
    "year": "2010",
    "month": "01",
    "day": "01",
    "time": ["00:00", "01:00"],  # Just 2 hours
}

# CARRA variable groups to test
variable_groups = {
    "basic_meteorology": [
        "2m_temperature",
        "2m_relative_humidity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
    ],

    "precipitation": [
        "total_precipitation",
    ],

    "radiation_shortwave": [
        "surface_solar_radiation_downwards",
    ],

    "radiation_longwave": [
        "surface_thermal_radiation_downwards",
    ],

    "all_hydrology": [
        "2m_temperature",
        "2m_relative_humidity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
    ],
}

def test_variable_group(group_name, variables):
    """Test if a group of variables can be downloaded together."""
    print(f"\n{'='*80}")
    print(f"Testing: {group_name}")
    print(f"Variables: {variables}")
    print('='*80)

    request = {**test_config, "variable": variables, "data_format": "netcdf"}

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        output_file = Path(tmp.name)

    try:
        print(f"Submitting request to CDS API...")
        c.retrieve("reanalysis-carra-single-levels", request, str(output_file))

        # Check what we got
        print(f"✓ Download successful: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

        with xr.open_dataset(output_file) as ds:
            print(f"✓ Dataset opened successfully")
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Data variables: {list(ds.data_vars.keys())}")
            print(f"  Coordinates: {list(ds.coords.keys())}")

            # Check variable mapping
            var_mapping = {}
            for var in ds.data_vars:
                if hasattr(ds[var], 'long_name'):
                    var_mapping[var] = ds[var].long_name

            print(f"\n  Variable details:")
            for short_name, long_name in var_mapping.items():
                print(f"    {short_name:15s} -> {long_name}")

        output_file.unlink()  # Clean up
        return True, list(ds.data_vars.keys())

    except Exception as e:
        print(f"✗ Failed: {e}")
        if output_file.exists():
            output_file.unlink()
        return False, str(e)


def main():
    """Run systematic exploration of CARRA variables."""

    print("="*80)
    print("CARRA CDS API Variable Availability Explorer")
    print("="*80)
    print(f"\nDataset: reanalysis-carra-single-levels")
    print(f"Domain: {test_config['domain']}")
    print(f"Product type: {test_config['product_type']}")
    print(f"Level type: {test_config['level_type']}")
    print(f"Test period: {test_config['year']}-{test_config['month']}-{test_config['day']} {test_config['time'][0]}-{test_config['time'][1]}")

    results = {}

    # Test each group
    for group_name, variables in variable_groups.items():
        success, result = test_variable_group(group_name, variables)
        results[group_name] = {"success": success, "result": result}

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for group_name, outcome in results.items():
        status = "✓ SUCCESS" if outcome["success"] else "✗ FAILED"
        print(f"{status:12s} {group_name:25s}")
        if outcome["success"]:
            print(f"             Variables returned: {outcome['result']}")
        else:
            print(f"             Error: {outcome['result'][:100]}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if results["all_hydrology"]["success"]:
        print("✓ All hydrology variables are available - use full set")
    else:
        print("⚠ Full hydrology set not available - use subset:")
        available_groups = [name for name, r in results.items() if r["success"] and name != "all_hydrology"]
        print(f"  Available groups: {available_groups}")


if __name__ == "__main__":
    main()
