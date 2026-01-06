#!/usr/bin/env python3
"""
Test config loading to debug ERA5_USE_CDS issue.
"""
import yaml
from pathlib import Path

config_path = Path('/Users/darrieythorsson/compHydro/code/SYMFLUENCE/tests/configs/test_paradise_summa_optimization.yaml')

print("="*70)
print("CONFIG LOADING TEST")
print("="*70)

# Load YAML directly
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"\nConfig type: {type(config)}")
print(f"\nTotal keys: {len(config)}")

# Check ERA5-related keys
era5_keys = {k: v for k, v in config.items() if 'ERA5' in k.upper()}
print(f"\nERA5-related keys: {era5_keys}")

# Check specific key
print(f"\nconfig.get('ERA5_USE_CDS'): {config.get('ERA5_USE_CDS')}")
print(f"Type: {type(config.get('ERA5_USE_CDS'))}")

# Check if it's in the dict directly
print(f"\n'ERA5_USE_CDS' in config: {'ERA5_USE_CDS' in config}")

print("\n" + "="*70)
