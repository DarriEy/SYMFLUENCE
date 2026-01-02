# SYMFLUENCE Configuration Guide

## Overview

SYMFLUENCE uses a hierarchical, type-safe configuration system that organizes 346+ parameters into logical sections. The new system provides three ways to create configurations:

1. **Presets** - Pre-configured templates for common use cases
2. **Minimal** - Quick setup with smart defaults
3. **YAML File** - Full customization and reproducibility

## Quick Start

### Option 1: Using Presets (Recommended for beginners)

```python
from symfluence.utils.config import SymfluenceConfig

# Create configuration from preset
config = SymfluenceConfig.from_preset(
    'summa-basic',
    DOMAIN_NAME='my_watershed',
    POUR_POINT_COORDS='51.17/-115.57'
)
```

**Available Presets:**
- `summa-basic` - SUMMA with basic forcing
- `fuse-basic` - FUSE with basic forcing
- `summa-carra` - SUMMA with CARRA regional reanalysis

### Option 2: Minimal Configuration (Fastest setup)

```python
# Minimal required parameters with intelligent defaults
config = SymfluenceConfig.from_minimal(
    domain_name='bow_at_banff',
    model='SUMMA',
    EXPERIMENT_TIME_START='2020-01-01 00:00',
    EXPERIMENT_TIME_END='2020-12-31 23:00',
    POUR_POINT_COORDS='51.17/-115.57'
)
```

### Option 3: YAML Configuration File (Full control)

```python
# Load from YAML with optional overrides
config = SymfluenceConfig.from_file(
    'config.yaml',
    overrides={'DOMAIN_NAME': 'updated_name'}
)
```

## Configuration Structure

The configuration is organized into 7 logical sections:

```python
config.system       # System paths, MPI, debugging
config.domain       # Domain definition, timing, coordinates
config.forcing      # Meteorological forcing data
config.model        # Hydrological model configurations
config.optimization # Calibration settings
config.evaluation   # Evaluation data
config.paths        # File paths and directories
```

## Type-Safe Access (Recommended)

The new hierarchical structure provides autocomplete and type safety:

```python
# Access nested configuration with autocomplete
domain_name = config.domain.name
start_time = config.domain.time_start
summa_exe = config.model.summa.exe

# Type-safe model-specific config
if config.model.summa:
    decisions = config.model.summa.decisions_file
    output_freq = config.model.summa.output_frequency
```

## Backward Compatible Access

Legacy code using dict-style access continues to work:

```python
# Dict-style access (legacy)
domain_name = config['DOMAIN_NAME']
start_time = config.get('EXPERIMENT_TIME_START')

# Convert to flat dict
flat_config = config.to_dict(flatten=True)
```

## Common Configuration Patterns

### 1. Domain Setup

```python
config = SymfluenceConfig.from_minimal(
    domain_name='fraser_river',
    model='SUMMA',
    EXPERIMENT_TIME_START='2019-01-01 00:00',
    EXPERIMENT_TIME_END='2021-12-31 23:00',
    POUR_POINT_COORDS='50.75/-121.50',
    DOMAIN_DEFINITION_METHOD='delineate'  # or 'subset', 'lumped'
)
```

### 2. Multi-Model Setup

```python
config = SymfluenceConfig.from_minimal(
    domain_name='athabasca',
    model='SUMMA,mizuRoute',  # SUMMA with routing
    EXPERIMENT_TIME_START='2015-01-01 00:00',
    EXPERIMENT_TIME_END='2020-12-31 23:00'
)
```

### 3. Calibration Setup

```python
config = SymfluenceConfig.from_file('base_config.yaml')

# Enable calibration
config_dict = config.to_dict()
config_dict['RUN_CALIBRATION'] = True
config_dict['CALIB_PERIOD'] = '2015-01-01, 2018-12-31'
config_dict['EVAL_PERIOD'] = '2019-01-01, 2020-12-31'
```

## Configuration Sections Reference

### System Configuration

Controls SYMFLUENCE runtime environment:

```python
config.system.data_dir          # Data directory
config.system.code_dir          # Code directory
config.system.mpi_processes     # Parallel processes
config.system.debug_mode        # Debug logging
config.system.log_level         # Log verbosity
```

### Domain Configuration

Defines the modeling domain:

```python
config.domain.name              # Domain identifier
config.domain.experiment_id     # Experiment identifier
config.domain.time_start        # Simulation start
config.domain.time_end          # Simulation end
config.domain.definition_method # delineate/subset/lumped
config.domain.discretization    # elevation/soilclass/radiation/etc
config.domain.pour_point_coords # lat/lon coordinates
```

### Forcing Configuration

Meteorological forcing data:

```python
config.forcing.dataset          # ERA5, RDRS, CARRA, etc
config.forcing.variables        # Variables to download
config.forcing.time_step        # Temporal resolution
config.forcing.bounds           # lat_min, lat_max, lon_min, lon_max
```

### Model Configuration

Model-specific settings:

```python
# SUMMA configuration
config.model.summa.exe                  # Executable path
config.model.summa.file_manager         # File manager template
config.model.summa.decisions_file       # Model decisions
config.model.summa.output_frequency     # Output timestep
config.model.summa.output_control_file  # Output control

# FUSE configuration
config.model.fuse.spatial_mode          # lumped/distributed
config.model.fuse.model_id              # FUSE model ID (900-960)
config.model.fuse.pet_method            # PET calculation

# mizuRoute configuration
config.model.mizuroute.from_model       # Source model (SUMMA/FUSE)
config.model.mizuroute.routing_method   # IRF or KWT
```

### Optimization Configuration

Calibration and optimization:

```python
config.optimization.enabled             # Enable calibration
config.optimization.algorithm           # DDS, SCE-UA, MOCOM
config.optimization.max_iterations      # Maximum iterations
config.optimization.objective_function  # NSE, KGE, RMSE
config.optimization.calib_period        # Calibration period
config.optimization.eval_period         # Evaluation period
```

## Environment Variables

Configuration values can be set via environment variables:

```bash
export SYMFLUENCE_DATA_DIR=/path/to/data
export SYMFLUENCE_CODE_DIR=/path/to/code
export MPI_PROCESSES=8
export DEBUG_MODE=true
```

## Configuration Validation

All configurations are automatically validated:

```python
try:
    config = SymfluenceConfig.from_file('config.yaml')
except ValidationError as e:
    print(f"Configuration error: {e}")
    # Detailed error messages with field names and requirements
```

## Best Practices

1. **Use presets for standard workflows** - Less error-prone
2. **Use from_minimal for experiments** - Quick iteration
3. **Use YAML files for production** - Reproducible and version-controlled
4. **Prefer typed access in new code** - `config.domain.name` over `config['DOMAIN_NAME']`
5. **Keep configurations immutable** - Create new instances for variations

## Migration from Legacy Config

If you have existing code using dict-style config:

```python
# Old way (still works)
domain_name = config['DOMAIN_NAME']
start_time = config.get('EXPERIMENT_TIME_START')

# New way (recommended)
domain_name = config.domain.name
start_time = config.domain.time_start

# Both work! No need to update everything at once.
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

## Examples

### Complete SUMMA Workflow

```python
from symfluence.utils.config import SymfluenceConfig
from symfluence.core import SYMFLUENCE

# Create configuration
config = SymfluenceConfig.from_preset(
    'summa-basic',
    DOMAIN_NAME='bow_at_banff',
    EXPERIMENT_ID='test_run',
    POUR_POINT_COORDS='51.17/-115.57',
    EXPERIMENT_TIME_START='2020-01-01 00:00',
    EXPERIMENT_TIME_END='2020-12-31 23:00'
)

# Save configuration for reproducibility
config.to_dict(flatten=True)  # Can be saved to YAML

# Initialize SYMFLUENCE
sf = SYMFLUENCE('config.yaml')

# Run workflow
sf.run_complete_workflow()
```

### Custom Model Configuration

```python
config = SymfluenceConfig.from_minimal(
    domain_name='custom_basin',
    model='FUSE',
    EXPERIMENT_TIME_START='2018-01-01 00:00',
    EXPERIMENT_TIME_END='2020-12-31 23:00',
    POUR_POINT_COORDS='45.5/-73.5',

    # FUSE-specific
    FUSE_MODEL_ID=902,
    FUSE_SPATIAL_MODE='distributed',
    FUSE_PET_METHOD='priestley-taylor'
)
```

## Troubleshooting

### Missing Required Fields

```python
# Error: Missing EXPERIMENT_TIME_START
config = SymfluenceConfig.from_minimal('test', 'SUMMA')

# Solution: Provide required fields
config = SymfluenceConfig.from_minimal(
    'test',
    'SUMMA',
    EXPERIMENT_TIME_START='2020-01-01 00:00',
    EXPERIMENT_TIME_END='2020-12-31 23:00'
)
```

### Invalid Values

```python
# Error: Invalid model name
config = SymfluenceConfig.from_minimal(
    'test',
    'INVALID_MODEL',  # Not recognized
    ...
)

# Solution: Use valid model name
# Valid: SUMMA, FUSE, HYPE, GR4J, MESH, NGEN, mizuRoute, tRoute
```

## Further Reading

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migrating from legacy config
- [CONFIG_ARCHITECTURE.md](CONFIG_ARCHITECTURE.md) - Technical architecture details
- [README.md](../README.md) - Main SYMFLUENCE documentation
