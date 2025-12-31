# SYMFLUENCE Test Suite Optimization Plan

## Overview
This document outlines optimization strategies for the SYMFLUENCE test suite to reduce test execution time while maintaining comprehensive coverage.

## Current Test Suite Status

### Unit Tests
- ✅ **attribute_processing**: 46 tests, 100% pass (2.85s)
  - test_elevation.py: DEM processing, slope/aspect
  - test_hydrology.py: Water balance, streamflow signatures
  - test_statistics.py: Zonal stats, circular statistics

- 🚧 **optimization** (new): Stubs created
  - conftest.py: Comprehensive fixtures for calibration testing
  - test_iterative_optimizer.py: Algorithm tests (needs interface adaptation)
  - test_model_calibration.py: SUMMA/FUSE/NGEN worker tests
  - test_optimization_manager.py: Manager interface tests

### Integration Tests
- ✅ **domain**: Lumped, semi-distributed, distributed, point-scale, regional
- ✅ **data**: Cloud acquisition (ERA5, AORC, NEX-GDDP-CMIP6, etc.)
- ✅ **calibration**: Elliðaár and Fyris catchment demos
- ⏱️ **Slow tests**: Some take 6+ minutes (e.g., distributed basin: 367s)

## Basin Test Optimization Opportunities

### Current Bottlenecks

1. **Data Download** (test_distributed.py:42-74)
   - Downloads `example_data_v0.2.zip` (~100MB+)
   - Extracts to SYMFLUENCE_data
   - **Time**: 30-60s depending on connection

2. **Data Copying** (test_distributed.py:208-226)
   - Copies attributes, forcing, shapefiles from semi_dist domain
   - File I/O overhead
   - **Time**: 5-10s

3. **Time Period** (test_distributed.py:121-126)
   - Current: 1 month (2004-01-01 to 2004-01-31)
   - Includes spinup, calibration, evaluation periods
   - **Time**: Impacts simulation runtime significantly

4. **Domain Discretization** (test_distributed.py:238-242)
   - Elevation band discretization (400m bands)
   - Creates 216 HRUs across 49 GRUs
   - **Time**: Depends on complexity

5. **SUMMA Simulation**
   - Runs for all 216 HRUs
   - 1 month of hourly data
   - **Time**: Majority of test time (200-300s)

6. **mizuRoute Routing**
   - Routing across 49 GRUs
   - Additional I/O and computation
   - **Time**: 20-50s

7. **Calibration** (test_distributed.py:133)
   - 3 DE iterations with population of 5
   - Each iteration runs full model
   - **Time**: 100-200s

### Optimization Strategies

#### 1. Use Cached/Minimal Test Data
```python
# Instead of downloading full example data:
@pytest.fixture(scope="session")  # Session scope, not module
def minimal_test_data(symfluence_data_root):
    """Create or reuse minimal synthetic test data."""
    test_domain = "test_minimal_basin"
    domain_path = symfluence_data_root / f"domain_{test_domain}"

    if domain_path.exists():
        return domain_path

    # Create synthetic minimal dataset instead of downloading
    create_synthetic_test_domain(domain_path,
                                 num_hrus=20,  # Instead of 216
                                 time_steps=72)  # 3 days instead of 31
    return domain_path
```

**Savings**: 30-60s (download) + storage

#### 2. Reduce Time Period
```python
# Change from 1 month to 3-5 days:
config["EXPERIMENT_TIME_START"] = "2004-01-01 01:00"
config["EXPERIMENT_TIME_END"] = "2004-01-05 23:00"  # 5 days instead of 31
config["CALIBRATION_PERIOD"] = "2004-01-02, 2004-01-04"  # 2 days
config["SPINUP_PERIOD"] = "2004-01-01, 2004-01-01"  # 1 day
```

**Savings**:
- SUMMA simulation: ~70% faster (5 days vs 31 days)
- mizuRoute: ~70% faster
- **Total**: 150-200s

#### 3. Reduce Spatial Complexity
```python
# Coarser elevation bands → fewer HRUs:
config["ELEVATION_BAND_SIZE"] = 800  # Instead of 400m
# Results in ~50-60 HRUs instead of 216

# Or use simpler discretization for tests:
config["DOMAIN_DISCRETIZATION"] = "lumped"  # Single HRU for smoke tests
```

**Savings**:
- With 60 HRUs: ~60% faster
- With lumped: ~90% faster
- **Total**: 100-250s

#### 4. Share Data Between Tests
```python
@pytest.fixture(scope="session")  # Session-wide, not function
def shared_preprocessed_domain(minimal_test_data):
    """Preprocess domain once and share across tests."""
    # Run preprocessing steps once
    # Cache the result for all tests
    return preprocessed_domain

# Individual tests just load and run models
def test_lumped_basin(shared_preprocessed_domain):
    # No preprocessing, just run model
    sym.managers['model'].run_models()
```

**Savings**: Eliminates redundant preprocessing across tests

#### 5. Minimal Calibration for Tests
```python
# Reduce calibration iterations:
config["NUMBER_OF_ITERATIONS"] = 2  # Instead of 3-5
config["DE_POPULATION_SIZE"] = 3    # Instead of 5

# Or skip calibration for basic workflow tests:
config["OPTIMIZATION_METHODS"] = []  # Disable calibration
```

**Savings**: 50-150s per test

#### 6. Use Pytest Markers for Test Categorization
```python
# In pytest.ini:
[pytest]
markers =
    quick: Quick tests (<5s)
    moderate: Moderate tests (5-30s)
    slow: Slow tests (>30s)
    smoke: Minimal smoke tests for CI
    full: Full integration tests

# Usage:
@pytest.mark.smoke
def test_lumped_basin_minimal(minimal_config):
    """Quick smoke test: 3 days, lumped domain, no calibration."""
    # Runs in ~10-15s

@pytest.mark.full
def test_distributed_basin_complete(full_config):
    """Full test: 1 month, distributed, with calibration."""
    # Runs in ~300s but provides comprehensive validation
```

**Benefits**:
- CI runs quick+smoke tests (~2-5 min total)
- Nightly runs full tests (30-60 min)
- Developers can choose test level

### Implementation Priority

#### High Priority (Immediate Impact)
1. ✅ **Reduce time periods** to 3-5 days (quick wins)
2. ✅ **Add pytest markers** for test categorization
3. ✅ **Use session-scoped fixtures** for data sharing
4. ✅ **Skip calibration** for basic workflow tests

#### Medium Priority
5. **Create minimal synthetic datasets** instead of downloading
6. **Reduce spatial complexity** for smoke tests
7. **Parallelize independent test execution**

#### Low Priority (Future)
8. **Cache intermediate results** (preprocessed models)
9. **Mock model execution** for pure workflow tests
10. **Containerize test environments** for consistency

## Recommended Test Structure

```
tests/
├── unit/                           # Fast (<1s per test)
│   ├── optimization/
│   │   ├── conftest.py            # ✅ Comprehensive fixtures
│   │   ├── test_iterative_optimizer.py  # Needs interface fix
│   │   ├── test_model_calibration.py    # Needs mocking
│   │   └── test_optimization_manager.py # Needs mocking
│   └── preprocessing/
│       └── attribute_processing/   # ✅ 46 tests passing
│           ├── test_elevation.py
│           ├── test_hydrology.py
│           └── test_statistics.py
│
├── integration/
│   ├── quick/                      # NEW: 5-30s per test
│   │   ├── test_lumped_minimal.py
│   │   └── test_workflows_smoke.py
│   │
│   ├── domain/                     # Current: 20s-6min per test
│   │   ├── test_lumped_basin.py   # Optimize to ~20s
│   │   ├── test_semi_distributed.py  # Optimize to ~60s
│   │   └── test_distributed.py    # Optimize to ~120s (from 367s)
│   │
│   ├── data/
│   │   └── test_cloud_acquisition.py  # Already fast (10-110s)
│   │
│   └── calibration/
│       └── test_calibration.py    # Keep comprehensive

