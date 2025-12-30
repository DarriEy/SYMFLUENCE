"""
Explore CERRA CDS API to understand variable availability.

CERRA is the European equivalent of CARRA.
Let's see if it has the same product type structure or if variables are available differently.
"""

import cdsapi
import xarray as xr
from pathlib import Path
import tempfile

c = cdsapi.Client()

# Test configuration - tiny request for fast iteration
test_config = {
    "level_type": "surface_or_atmosphere",
    "data_type": "reanalysis",  # Required parameter for CERRA
    "year": "2010",
    "month": "01",
    "day": "01",
    "time": ["00:00", "03:00"],  # CERRA is 3-hourly
    # NOTE: No 'area' parameter - MARS cannot crop Lambert Conformal grids
    # Must download full domain and subset locally (same issue as CARRA)
}

# Test cases to explore CERRA
test_cases = [
    {
        "name": "Analysis - all variables",
        "config": {
            "product_type": "analysis",
        },
        "variables": [
            "2m_temperature",
            "2m_relative_humidity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
        ],
    },
    {
        "name": "Analysis - meteorology only",
        "config": {
            "product_type": "analysis",
        },
        "variables": [
            "2m_temperature",
            "2m_relative_humidity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
        ],
    },
    {
        "name": "Forecast - precipitation",
        "config": {
            "product_type": "forecast",
            "leadtime_hour": ["1"],
        },
        "variables": [
            "total_precipitation",
        ],
    },
    {
        "name": "Forecast - radiation",
        "config": {
            "product_type": "forecast",
            "leadtime_hour": ["1"],
        },
        "variables": [
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
        ],
    },
]


def test_case(case):
    """Test a specific CERRA configuration."""
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
        c.retrieve("reanalysis-cerra-single-levels", request, str(output_file))

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
    """Run systematic exploration of CERRA variables."""

    print("="*80)
    print("CERRA CDS API Variable Availability Explorer")
    print("="*80)
    print(f"\nDataset: reanalysis-cerra-single-levels")
    print(f"Test area: Full CERRA domain (subsetting done locally)")
    print(f"Test period: 2010-01-01 (2 timesteps)")

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
    print("RECOMMENDATIONS FOR CERRA ACQUISITION")
    print("="*80)

    successful = {name: r for name, r in results.items() if r["success"]}

    if successful:
        print("✓ Working configurations found:")
        for name, result in successful.items():
            print(f"\n  {name}:")
            print(f"    Product type: {result['config']['product_type']}")
            if 'leadtime_hour' in result['config']:
                print(f"    Lead times: {result['config']['leadtime_hour']}")
            print(f"    Variables: {result['result']}")

        # Determine strategy
        print("\n" + "="*80)
        print("RECOMMENDED STRATEGY:")
        print("="*80)

        if results.get("Analysis - all variables", {}).get("success"):
            print("✓ Use ANALYSIS product only (all variables available)")
            print("  Single download provides complete forcing set")
        elif results.get("Analysis - meteorology only", {}).get("success"):
            if any(r["success"] for k, r in results.items() if "Forecast" in k):
                print("✓ Use DUAL-PRODUCT strategy (like CARRA):")
                print("  - Analysis: meteorology (t2m, r2, u10, v10, sp)")
                print("  - Forecast: fluxes (tp, ssrd, strd)")
            else:
                print("⚠ Limited variable availability")
                print("  May need to calculate missing variables or use hybrid approach")
    else:
        print("✗ No working configurations found")


if __name__ == "__main__":
    main()
