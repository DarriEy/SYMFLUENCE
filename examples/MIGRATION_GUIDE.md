# Example Notebooks Migration Guide (v0.8.0)

## Overview

SYMFLUENCE v0.8.0 reorganized the repository structure. Configuration templates are now distributed as package data instead of in the `0_config_files/` directory.

## What Changed

**Before (v0.7.0):**
- Templates in `0_config_files/` at repository root
- Notebooks searched for `0_config_files/` to find repo root
- Configs saved to `0_config_files/`

**After (v0.8.0):**
- Templates in `src/symfluence/data/config_templates/` (distributed with package)
- Tutorial configs in `examples/configs/`
- User configs can be anywhere (e.g., current directory)

## Required Updates for Notebooks

### 1. Remove Root-Finding Logic

**Old pattern (DELETE THIS):**
```python
# Find SYMFLUENCE root by searching upward for 0_config_files directory
def find_symfluence_root():
    current = Path.cwd().resolve()
    for i in range(30):
        if '.ipynb_checkpoints' in str(current):
            if current.parent == current:
                break
            current = current.parent
            continue

        if (current / '0_config_files').exists():
            if '.ipynb_checkpoints' not in str(current):
                print(f"Found SYMFLUENCE root: {current}")
                return current
            else:
                current = current.parent
                continue

        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError(f"Could not find SYMFLUENCE root...")

SYMFLUENCE_CODE_DIR = find_symfluence_root()
```

**New pattern (USE THIS):**
```python
# No root-finding needed - use package resources directly
from pathlib import Path
```

### 2. Load Templates from Package Data

**Old pattern:**
```python
config_template = SYMFLUENCE_CODE_DIR / '0_config_files' / 'config_template.yaml'
with open(config_template, 'r') as f:
    config = yaml.safe_load(f)
```

**New pattern:**
```python
from symfluence.resources import get_config_template

# Get template from package
config_template = get_config_template()
with open(config_template, 'r') as f:
    config = yaml.safe_load(f)
```

### 3. Save Configs to Current Directory

**Old pattern:**
```python
config_path = SYMFLUENCE_CODE_DIR / '0_config_files' / 'config_basin_lumped.yaml'
```

**New pattern:**
```python
# Save to current directory or examples/configs/
config_path = Path('./config_basin_lumped.yaml')
# Or use examples/configs for tutorial configs
# config_path = Path('../configs/config_basin_lumped.yaml')
```

### 4. Remove sys.path Manipulation (if present)

**Old pattern (DELETE THIS):**
```python
sys.path.append(str(SYMFLUENCE_CODE_DIR))
```

**New pattern:**
```python
# Not needed - package is already installed
```

## Files to Update

All 7 notebooks need these changes:
- `01_point_vertical_flux_estimation/01a_point_scale_snotel.ipynb`
- `01_point_vertical_flux_estimation/01b_point_scale_fluxnet.ipynb`
- `02_watershed_modelling/02a_basin_lumped.ipynb`
- `02_watershed_modelling/02b_basin_semi_distributed.ipynb`
- `02_watershed_modelling/02c_basin_distributed.ipynb`
- `03_large_domain_simulation/03a_domain_regional.ipynb`
- `03_large_domain_simulation/03b_domain_continental.ipynb`

## Complete Example

**Full cell replacement for Step 1 (Configuration):**

```python
# Step 1 — Create configuration

from pathlib import Path
import yaml
from symfluence import SYMFLUENCE
from symfluence.resources import get_config_template

# Load template from package data
config_template = get_config_template()
with open(config_template, 'r') as f:
    config = yaml.safe_load(f)

# Modify configuration
config['DOMAIN_NAME'] = 'Bow_at_Banff_lumped'
config['EXPERIMENT_ID'] = 'run_1'
config['POUR_POINT_COORDS'] = '51.1722/-115.5717'
config['DOMAIN_DEFINITION_METHOD'] = 'lumped'
# ... other config modifications ...

# Save configuration (current directory or examples/configs/)
config_path = Path('./config_basin_lumped.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Configuration saved: {config_path}")

# Initialize SYMFLUENCE
symfluence = SYMFLUENCE(config_path)
```

## Benefits of New Approach

- ✅ **Works anywhere**: No need to be in specific directory
- ✅ **Cleaner code**: No complex root-finding logic
- ✅ **Package distribution**: Templates distributed with pip install
- ✅ **Flexibility**: Configs can be anywhere, not just `0_config_files/`

## Need Help?

- See `examples/configs/` for updated tutorial configurations
- Check `0_config_files/README.md` for workspace usage
- Use `symfluence config list` to see available templates
