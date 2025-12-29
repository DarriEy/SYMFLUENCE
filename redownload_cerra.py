"""
Re-download CERRA forcing with the fixed time-shift code.
"""
import yaml
from pathlib import Path
from symfluence import SYMFLUENCE

# Load config
config_path = Path('test_quick_summa_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("Initializing SYMFLUENCE...")
sym = SYMFLUENCE(config_path)

print("\nDownloading CERRA forcing data with time-shift fix...")
sym.managers['data'].acquire_forcings()

print("\n✓ CERRA download complete!")
