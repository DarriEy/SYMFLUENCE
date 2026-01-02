# SYMFLUENCE Configuration Migration Guide

## Overview

This guide helps you migrate from the legacy flat configuration system to the new hierarchical typed configuration system. **The good news**: you don't have to migrate all at once—both systems work side-by-side!

## What Changed?

### Before (Legacy System)
```python
from symfluence.utils.config import load_config

# Load flat dictionary
config = load_config('config.yaml')

# Dict-style access everywhere
domain_name = config['DOMAIN_NAME']
start_time = config.get('EXPERIMENT_TIME_START')
```

### After (New System)
```python
from symfluence.utils.config import SymfluenceConfig

# Load hierarchical config
config = SymfluenceConfig.from_file('config.yaml')

# Type-safe access (recommended)
domain_name = config.domain.name
start_time = config.domain.time_start

# Dict-style still works! (backward compatible)
domain_name = config['DOMAIN_NAME']
start_time = config.get('EXPERIMENT_TIME_START')
```

## Migration Strategies

### Strategy 1: No Migration (Keep Using Legacy)

**When**: You have working code and don't want to change it.

**Action**: Nothing! Legacy `load_config()` still works.

```python
# This still works perfectly
from symfluence.utils.config import load_config
config = load_config('config.yaml')
```

### Strategy 2: Gradual Migration (Recommended)

**When**: You want to modernize but have a large codebase.

**Action**: Use new `SymfluenceConfig` but keep dict-style access initially.

```python
# Step 1: Change the loader only
from symfluence.utils.config import SymfluenceConfig
config = SymfluenceConfig.from_file('config.yaml')

# Step 2: All existing dict-style code still works
domain_name = config['DOMAIN_NAME']
start_time = config.get('EXPERIMENT_TIME_START')

# Step 3: Gradually update new code to use typed access
if self.typed_config:
    domain_name = self.typed_config.domain.name
else:
    domain_name = self.config.get('DOMAIN_NAME')
```

### Strategy 3: Full Migration (For New Projects)

**When**: Starting a new project or major refactor.

**Action**: Use typed configuration throughout.

```python
from symfluence.utils.config import SymfluenceConfig

config = SymfluenceConfig.from_minimal(
    domain_name='my_basin',
    model='SUMMA',
    EXPERIMENT_TIME_START='2020-01-01 00:00',
    EXPERIMENT_TIME_END='2020-12-31 23:00'
)

# Type-safe access everywhere
domain = config.domain.name
model = config.model.hydrological_model
```

## Mapping Guide: Old Keys → New Paths

### Domain Settings
| Legacy Key | New Path | Example |
|------------|----------|---------|
| `DOMAIN_NAME` | `config.domain.name` | `'bow_at_banff'` |
| `EXPERIMENT_ID` | `config.domain.experiment_id` | `'calibration_v1'` |
| `EXPERIMENT_TIME_START` | `config.domain.time_start` | `'2020-01-01 00:00'` |
| `EXPERIMENT_TIME_END` | `config.domain.time_end` | `'2020-12-31 23:00'` |
| `DOMAIN_DEFINITION_METHOD` | `config.domain.definition_method` | `'delineate'` |
| `DOMAIN_DISCRETIZATION` | `config.domain.discretization` | `'elevation'` |
| `POUR_POINT_COORDS` | `config.domain.pour_point_coords` | `'51.17/-115.57'` |

### System Settings
| Legacy Key | New Path | Example |
|------------|----------|---------|
| `SYMFLUENCE_DATA_DIR` | `config.system.data_dir` | `Path('/data')` |
| `SYMFLUENCE_CODE_DIR` | `config.system.code_dir` | `Path('/code')` |
| `MPI_PROCESSES` | `config.system.mpi_processes` | `8` |
| `DEBUG_MODE` | `config.system.debug_mode` | `True` |

### Model Settings
| Legacy Key | New Path | Example |
|------------|----------|---------|
| `HYDROLOGICAL_MODEL` | `config.model.hydrological_model` | `'SUMMA'` |
| `SUMMA_EXE` | `config.model.summa.exe` | `'summa.exe'` |
| `SUMMA_FILE_MANAGER` | `config.model.summa.file_manager` | `'file_manager.txt'` |
| `FUSE_SPATIAL_MODE` | `config.model.fuse.spatial_mode` | `'lumped'` |
| `FUSE_MODEL_ID` | `config.model.fuse.model_id` | `902` |

### Forcing Settings
| Legacy Key | New Path | Example |
|------------|----------|---------|
| `FORCING_DATASET` | `config.forcing.dataset` | `'ERA5'` |
| `FORCING_TIME_STEP_SIZE` | `config.forcing.time_step` | `'1H'` |
| `FORCING_VARIABLES` | `config.forcing.variables` | `['T2M', 'TP']` |

## Code Migration Examples

### Example 1: Simple Preprocessor

**Before:**
```python
class MyPreProcessor:
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.domain_name = config.get('DOMAIN_NAME')
        self.start_time = config.get('EXPERIMENT_TIME_START')
```

**After (Gradual):**
```python
class MyPreProcessor:
    def __init__(self, config: Union[Dict, SymfluenceConfig], logger):
        if isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict()  # Backward compat
        else:
            self.config = config
            self.typed_config = None

        # Prefer typed access with fallback
        if self.typed_config:
            self.domain_name = self.typed_config.domain.name
            self.start_time = self.typed_config.domain.time_start
        else:
            self.domain_name = self.config.get('DOMAIN_NAME')
            self.start_time = self.config.get('EXPERIMENT_TIME_START')
```

**After (Full):**
```python
class MyPreProcessor:
    def __init__(self, config: SymfluenceConfig, logger):
        self.config = config
        self.domain_name = config.domain.name
        self.start_time = config.domain.time_start
```

### Example 2: Model Runner

**Before:**
```python
def run_model(self):
    setup_dir = Path(self.config.get('PROJECT_DIR')) / 'settings' / 'SUMMA'
    exe = Path(self.config.get('SUMMA_INSTALL_PATH')) / self.config.get('SUMMA_EXE')
```

**After (Gradual):**
```python
def run_model(self):
    if self.typed_config:
        setup_dir = self.typed_config.paths.project_dir / 'settings' / 'SUMMA'
        exe = self.typed_config.model.summa.install_path / self.typed_config.model.summa.exe
    else:
        setup_dir = Path(self.config.get('PROJECT_DIR')) / 'settings' / 'SUMMA'
        exe = Path(self.config.get('SUMMA_INSTALL_PATH')) / self.config.get('SUMMA_EXE')
```

### Example 3: Workflow Orchestrator

**Before:**
```python
def setup_workflow(self):
    models = self.config.get('HYDROLOGICAL_MODEL').split(',')
    for model in models:
        if model == 'SUMMA':
            self.run_summa()
```

**After:**
```python
def setup_workflow(self):
    models = self.config.model.hydrological_model
    if isinstance(models, str):
        models = [m.strip() for m in models.split(',')]

    for model in models:
        if model == 'SUMMA':
            self.run_summa()
```

## Updating Base Classes

All SYMFLUENCE base classes (`BaseModelPreProcessor`, `BaseModelRunner`) have been updated to accept both config types:

```python
class BaseModelPreProcessor:
    def __init__(self, config: Union[Dict[str, Any], SymfluenceConfig], logger: Any):
        if isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.config = config
            self.typed_config = None
```

**Your model-specific code should follow the same pattern.**

## Testing Your Migration

### 1. Unit Tests

```python
def test_my_preprocessor_with_both_configs():
    # Test with dict config (legacy)
    dict_config = {'DOMAIN_NAME': 'test', 'EXPERIMENT_ID': 'exp1'}
    prep1 = MyPreProcessor(dict_config, logger)
    assert prep1.domain_name == 'test'

    # Test with typed config (new)
    typed_config = SymfluenceConfig.from_minimal('test', 'SUMMA', ...)
    prep2 = MyPreProcessor(typed_config, logger)
    assert prep2.domain_name == 'test'
```

### 2. Integration Tests

```python
def test_workflow_with_new_config():
    config = SymfluenceConfig.from_preset('summa-basic')
    sf = SYMFLUENCE(config_path='config.yaml')
    sf.run_complete_workflow()
    # Verify outputs
```

## Common Migration Issues

### Issue 1: Type Mismatches

**Problem:**
```python
# Old: Always returns string
model = config.get('HYDROLOGICAL_MODEL')  # 'SUMMA,mizuRoute'

# New: Can be string or list
model = config.model.hydrological_model  # Could be str or List[str]
```

**Solution:**
```python
# Normalize to list
models = config.model.hydrological_model
if isinstance(models, str):
    models = [m.strip() for m in models.split(',')]
```

### Issue 2: None vs Default Values

**Problem:**
```python
# Old: get() returns None if not found
value = config.get('OPTIONAL_PARAM')  # None

# New: Field might have default
value = config.model.summa.optional_param  # Could be default value
```

**Solution:**
```python
# Use explicit None checks or hasattr()
if config.model.summa and config.model.summa.optional_param:
    use_param()
```

### Issue 3: Path Objects

**Problem:**
```python
# Old: Everything was strings
data_dir = config.get('SYMFLUENCE_DATA_DIR')  # str

# New: Paths are Path objects
data_dir = config.system.data_dir  # Path
```

**Solution:**
```python
# Path objects work with / operator
file_path = config.system.data_dir / 'subdir' / 'file.txt'

# Convert to string if needed
str_path = str(config.system.data_dir)
```

## Migration Checklist

- [ ] Update imports: `from symfluence.utils.config import SymfluenceConfig`
- [ ] Update config loading: `SymfluenceConfig.from_file()` or `from_preset()`
- [ ] Add dual-mode support to custom classes
- [ ] Update config access to use typed paths (gradual)
- [ ] Test with both dict and typed configs
- [ ] Update documentation and examples
- [ ] Run integration tests

## Performance Benefits

The new system includes performance optimizations:

```python
# Cached flattened dict for legacy access
config.get('DOMAIN_NAME')  # Fast! Cached lookup
config['EXPERIMENT_ID']    # Fast! Cached lookup

# Direct attribute access is even faster
config.domain.name         # Fastest! Direct attribute access
```

## Getting Help

If you encounter issues during migration:

1. Check this guide for common issues
2. Review [CONFIGURATION.md](CONFIGURATION.md) for examples
3. Look at updated model code in `src/symfluence/utils/models/`
4. File an issue on GitHub with migration questions

## Example: Complete Migration

Here's a complete before/after example for a custom model:

<details>
<summary>Click to expand complete example</summary>

**Before (Legacy):**
```python
from typing import Dict, Any

class CustomModelPreProcessor:
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = Path(config.get('PROJECT_DIR'))
        self.setup_dir = self.project_dir / 'settings' / 'CustomModel'

    def run_preprocessing(self):
        start = self.config.get('EXPERIMENT_TIME_START')
        end = self.config.get('EXPERIMENT_TIME_END')
        self.logger.info(f"Processing {self.domain_name} from {start} to {end}")
```

**After (Migrated):**
```python
from typing import Union, Dict, Any
from symfluence.utils.config.models import SymfluenceConfig

class CustomModelPreProcessor:
    def __init__(self, config: Union[Dict[str, Any], SymfluenceConfig], logger):
        # Dual-mode support
        if isinstance(config, SymfluenceConfig):
            self.typed_config = config
            self.config = config.to_dict(flatten=True)
        else:
            self.config = config
            self.typed_config = None

        self.logger = logger

        # Prefer typed access with fallback
        if self.typed_config:
            self.domain_name = self.typed_config.domain.name
            self.project_dir = self.typed_config.paths.project_dir
        else:
            self.domain_name = self.config.get('DOMAIN_NAME')
            self.project_dir = Path(self.config.get('PROJECT_DIR'))

        self.setup_dir = self.project_dir / 'settings' / 'CustomModel'

    def run_preprocessing(self):
        if self.typed_config:
            start = self.typed_config.domain.time_start
            end = self.typed_config.domain.time_end
        else:
            start = self.config.get('EXPERIMENT_TIME_START')
            end = self.config.get('EXPERIMENT_TIME_END')

        self.logger.info(f"Processing {self.domain_name} from {start} to {end}")
```

</details>

## Summary

- **No forced migration** - Both systems coexist
- **Gradual adoption** - Migrate at your own pace
- **Full backward compatibility** - Existing code keeps working
- **Performance gains** - Caching for legacy access
- **Type safety** - Better IDE support and error catching

Start with Strategy 2 (Gradual Migration) for existing projects, and use Strategy 3 (Full Migration) for new development!
