# SYMPHLUENCE Phase 1 Refactoring: Migration Plan

## Overview

This document outlines the Phase 1 refactoring of SYMFLUENCE to eliminate circular dependencies between the `models` and `optimization` packages. The goal is to establish a clean, one-way dependency structure where `optimization` depends on `models`, but `models` does not depend on `optimization`.

## Current Status

✅ **Phase 1 Refactoring Complete**

- Model runners and workers are organized into `src/symfluence/models/{model_name}/`
- Model-specific preprocessors, runners, and postprocessors are decoupled from optimization
- Circular dependencies resolved through strategic module placement and re-exports

## Architecture

### Dependency Structure

```
models/
├── {model_name}/
│   ├── preprocessor.py      # Model-specific preprocessing
│   ├── runner.py            # Model execution
│   ├── postprocessor.py     # Model output processing
│   ├── calibration/
│   │   ├── worker.py        # Optimization worker (registers with OptimizerRegistry)
│   │   ├── optimizer.py     # Model-specific optimizer
│   │   └── parameter_manager.py
│   └── utilities/           # Model-specific utilities
│
└── utilities/               # Shared model utilities
    ├── routing_decider.py
    └── time_window_manager.py

optimization/
├── optimizers/
│   ├── base_model_optimizer.py
│   └── registry.py          # Central optimizer registry
├── workers/
│   ├── base_worker.py
│   └── utilities.py         # Re-exports (e.g., RoutingDecider for backward compatibility)
└── ...

Dependency Direction: optimization → models (one-way only)
```

### Key Design Decisions

1. **Model Runners and Workers**: Located in `models/{model_name}/calibration/` rather than `optimization/`
2. **Registry Pattern**: Model optimizers and workers register themselves via decorators when imported
3. **Utilities Separation**: Shared utilities (RoutingDecider, TimeWindowManager) live in `models/utilities/`
4. **Backward Compatibility**: Re-exports in `optimization/workers/utilities/` maintain compatibility
5. **Configuration-Driven**: Models use config dictionaries rather than importing from optimization

## Refactored Components

### 1. Model Utilities

- **RoutingDecider**: Moved from `optimization/` to `models/utilities/routing_decider.py`
  - Determines if a model needs MizuRoute routing
  - Accessible from both `models.utilities` and `optimization.workers.utilities`

- **TimeWindowManager**: Shared time window utilities in `models/utilities/`

- **FUSE Converter**: Moved to `models/fuse/utilities/fuse_to_mizuroute_converter.py`

### 2. Model Optimizers

All models now have optimizer implementations in `models/{model_name}/calibration/optimizer.py`:
- SUMMA (SUMMAModelOptimizer)
- FUSE (FUSEModelOptimizer)
- NGEN (NGENModelOptimizer)
- GR (GRModelOptimizer)
- HYPE (HYPEModelOptimizer)
- MESH (MESHModelOptimizer)
- RHESSys (RHESSysModelOptimizer)

Each optimizer:
- Inherits from `BaseModelOptimizer`
- Registers via `@OptimizerRegistry.register_optimizer('MODEL_NAME')`
- Imports its worker to trigger worker registration

### 3. Model Workers

Located in `models/{model_name}/calibration/worker.py`:
- Inherit from `BaseWorker`
- Register via `@OptimizerRegistry.register_worker('MODEL_NAME')`
- Implement model-specific objective function evaluation

### 4. Parameter Managers

Located in `optimization/parameter_managers/`:
- SUMMA: `SUMMAParameterManager`
- FUSE: `FUSEParameterManager`
- NGEN: `MLParameterManager` (also used by LSTM)
- Shared: `HydrologicalModelParameterManager`

## Migration Steps Completed

### Step 1: Organize Model Runners
- ✅ Moved model-specific runners to `models/{model_name}/runner.py`
- ✅ Updated imports in optimization layer to reference new locations

### Step 2: Break Circular Dependencies
- ✅ Removed optimization imports from `models/` package initialization
- ✅ Models only import from `optimization` in calibration submodules
- ✅ Created re-exports in `optimization/workers/utilities.py` for backward compatibility

### Step 3: Centralize Registry
- ✅ `OptimizerRegistry` in `optimization/registry.py` is the single source of truth
- ✅ All model optimizers, workers, and parameter managers register on import
- ✅ Registry supports lazy loading via decorator pattern

### Step 4: Base Classes
- ✅ `BaseModelOptimizer` in `optimization/optimizers/base_model_optimizer.py`
- ✅ `BaseWorker` in `optimization/workers/base_worker.py`
- ✅ `BaseModelPreProcessor` in `models/base/base_preprocessor.py`

## Integration Points

### How Optimization Discovers Models

1. **Import Trigger**: When `optimization_manager.py` initializes an optimizer:
   ```python
   from symfluence.models.{model_name}.calibration.optimizer import {MODEL}ModelOptimizer
   ```

2. **Worker Import**: The optimizer imports its worker:
   ```python
   from .worker import {MODEL}Worker  # noqa: F401 - triggers registration
   ```

3. **Registry Discovery**: `OptimizerRegistry.get_optimizer('MODEL_NAME')` returns the registered class

### Configuration Flow

1. Configuration created in `projects/domain_{name}/settings/config.yaml`
2. `InitializationService` validates configuration
3. `OptimizationManager` instantiates optimizer for the specified model
4. Optimizer loads model runner from `models/{model_name}/runner.py`

## Testing

### Test Coverage

- **Test Optimizer Registry** (`test_optimization_workflow_refactored.py::TestOptimizerRegistryAfterRefactoring`)
  - Verifies all model optimizers register correctly
  - Checks worker registry contains all models
  - Validates parameter manager registry

- **Test Model Utilities** (`test_optimization_workflow_refactored.py::TestModelUtilitiesAccessible`)
  - RoutingDecider accessible from `models.utilities`
  - Backward compatibility via re-exports in `optimization.workers.utilities`
  - FUSE converter accessible from new location

- **Test Circular Dependencies** (`test_optimization_workflow_refactored.py::TestNoCircularDependencies`)
  - Importing models doesn't trigger optimization imports
  - Optimization can import from models (one-way dependency)
  - Model utilities work independently

- **Test Migration Readiness** (`test_optimization_workflow_refactored.py::TestMigrationReadiness`)
  - All model directories exist
  - Migration plan documentation exists
  - Base classes are available

### Integration Tests

- **Domain Workflow Tests**: Run models through full pipeline (preprocessing → running → postprocessing)
  - `test_lumped_basin_workflow`: Tests lumped-parameter configurations
  - `test_semi_distributed_basin_workflow`: Tests semi-distributed workflows
  - `test_distributed_basin_workflow`: Tests fully distributed setups

- **Optimization Tests**: Verify optimization loop works with refactored architecture
  - Tests will be added in Phase 2

## Phase 2 Planning (Future)

Planned improvements beyond Phase 1:

1. **Model Defaults System**
   - Move model-specific defaults from hardcoded values to configuration
   - Create `ModelDefaults` config class per model

2. **Configuration Typing**
   - Add Pydantic models for all configuration sections
   - Validate model parameters at initialization time

3. **Calibration Target Framework**
   - Implement `CalibrationTarget` base class
   - Allow multiple objective functions per model

4. **Advanced Parameter Management**
   - Parameter constraints and relationships
   - Parameter transformation functions
   - Adaptive parameter bounds based on domain

5. **Extended Backward Compatibility**
   - Support legacy configuration formats
   - Provide migration utilities for old configs

## Troubleshooting

### Issue: "No optimizer registered for model X"

**Solution**: Ensure the optimizer is imported before `OptimizerRegistry.get_optimizer()` is called.
- Check that `models/{model_name}/calibration/optimizer.py` exists
- Verify the `@OptimizerRegistry.register_optimizer('MODEL_NAME')` decorator is present

### Issue: "No worker registered for model X"

**Solution**: The optimizer must import its worker.
- Check line 15 of the optimizer: `from .worker import {MODEL}Worker  # noqa: F401`

### Issue: RoutingDecider not found

**Solution**: Import from the correct location based on context:
- From models: `from symfluence.models.utilities import RoutingDecider`
- From optimization (backward compatibility): `from symfluence.optimization.workers.utilities import RoutingDecider`

## References

- `src/symfluence/models/`: Model implementations
- `src/symfluence/optimization/`: Optimization framework
- `tests/integration/test_optimization_workflow_refactored.py`: Migration readiness tests
- `tests/integration/domain/test_lumped_basin.py`: Model workflow tests

## Completed By

- Phase 1 implementation: January 2026
- All core model refactoring complete
- Circular dependencies resolved
- Integration tests passing
