# 4.11 Parallel Scaling Study

Systematic evaluation of SYMFLUENCE's parallel execution capabilities across
shared-memory (ProcessPool), distributed-memory (MPI), algorithm-level (Async-DDS),
and accelerated computing (JAX/GPU) paradigms.

## Study Overview

This experiment characterises the scaling behaviour of SYMFLUENCE's three-tier
parallel architecture on both commodity hardware (laptop/workstation) and
high-performance computing (HPC) clusters. Six sub-experiments progressively
explore different parallelism dimensions, using the same Bow River at Banff testbed
as Sections 4.2--4.7.

### Sub-experiments

| Section | Experiment | Parallelism Type | Variable |
|---------|-----------|-----------------|----------|
| 4.11.1 | Strong scaling (ProcessPool) | Shared-memory multiprocessing | Worker count (1--64) |
| 4.11.2 | Strong scaling (MPI) | Distributed-memory communication | MPI ranks (1--128) |
| 4.11.3 | Async vs Sync DDS | Algorithm-level batch parallelism | Batch strategy & pool size |
| 4.11.4 | JAX acceleration | JIT compilation & GPU offloading | Backend (NumPy/JAX CPU/GPU) |
| 4.11.5 | Weak scaling | Problem-size scaling | Domain complexity (1--2335 HRUs) |
| 4.11.6 | Task-level parallelism | Embarrassingly parallel ensembles | Ensemble execution mode |

## Common Experimental Settings

| Parameter | Value |
|-----------|-------|
| **Domain** | Bow River at Banff, Alberta (2,210 km^2) |
| **Pour Point** | 51.1722 N, -115.5717 W |
| **Forcing** | ERA5 |
| **Calibration Period** | 2004-01-01 to 2007-12-31 |
| **Evaluation Period** | 2008-01-01 to 2009-12-31 |
| **Spinup Period** | 2002-01-01 to 2003-12-31 |
| **Streamflow Station** | WSC 05BB001 |
| **Calibration Metric** | KGE |
| **Random Seed** | 42 |

## Study Structure

```
12. Parallel scaling/
├── configs/
│   ├── base_bow_hbv_dds.yaml                  # Base config (template)
│   ├── strong_processpool_np{1,2,4,8,...}.yaml # 4.11.1 ProcessPool scaling
│   ├── strong_mpi_np{1,2,4,8,...}.yaml         # 4.11.2 MPI scaling
│   ├── async_dds_pool{5,10,20}_batch{4,8,16}.yaml # 4.11.3 Async-DDS
│   ├── sync_dds_np{4,16,64}.yaml               # 4.11.3 Sync baseline
│   ├── jax_numpy.yaml                          # 4.11.4 NumPy backend
│   ├── jax_cpu_nojit.yaml                      # 4.11.4 JAX CPU no JIT
│   ├── jax_cpu_jit.yaml                        # 4.11.4 JAX CPU + JIT
│   ├── jax_gpu_jit.yaml                        # 4.11.4 JAX GPU + JIT
│   ├── weak_lumped.yaml                        # 4.11.5 Lumped (1 HRU)
│   ├── weak_elevation.yaml                     # 4.11.5 Elevation bands (12 HRUs)
│   ├── weak_semidist.yaml                      # 4.11.5 Semi-distributed (379 HRUs)
│   ├── weak_distributed.yaml                   # 4.11.5 Distributed (2335 cells)
│   ├── ensemble_sequential.yaml                # 4.11.6 Sequential baseline
│   └── ensemble_parallel_{model}.yaml          # 4.11.6 Per-model configs
├── scripts/
│   ├── generate_configs.py           # Configuration file generator
│   ├── run_scaling_study.py          # Main experiment orchestrator
│   ├── analyze_scaling.py            # Metrics computation & figures
│   └── create_publication_figures.py # Journal-ready figures
├── slurm/
│   ├── submit_strong_processpool.sh  # SLURM job for 4.11.1
│   ├── submit_strong_mpi.sh          # SLURM job for 4.11.2
│   ├── submit_async_dds.sh           # SLURM job for 4.11.3
│   ├── submit_jax_gpu.sh             # SLURM job for 4.11.4
│   └── submit_all.sh                 # Submit all HPC experiments
├── results/
│   ├── plots/                        # Generated figures
│   ├── scaling_metrics.csv           # Speedup and efficiency data
│   └── timing_raw.csv               # Raw wall-clock measurements
├── analysis/                         # Post-processing outputs
└── README.md                         # This file
```

## Prerequisites

1. **SYMFLUENCE Installation**
   ```bash
   pip install -e ".[hbv,optimization]"
   ```

2. **Domain Setup** (from Section 4.1)
   - Bow at Banff lumped domain: `$SYMFLUENCE_DATA_DIR/Bow_at_Banff_lumped_era5`
   - Bow semi-distributed domain: `$SYMFLUENCE_DATA_DIR/Bow_at_Banff_semidist_era5`
   - Bow distributed domain: `$SYMFLUENCE_DATA_DIR/Bow_at_Banff_distributed_era5`
   - ERA5 forcing data acquired
   - WSC streamflow observations for station 05BB001

3. **MPI (for 4.11.2)**
   ```bash
   pip install mpi4py
   # Verify: mpirun --version
   ```

4. **JAX with GPU (for 4.11.4 on HPC)**
   ```bash
   pip install jax[cuda12]
   ```

## Quick Start

### Step 1: Generate Configuration Files
```bash
cd scripts
python generate_configs.py                        # All configs
python generate_configs.py --experiment 1         # ProcessPool only
python generate_configs.py --hpc                  # Include HPC-scale configs
```

### Step 2: Run Experiments

**Laptop (all sub-experiments at laptop scale):**
```bash
python run_scaling_study.py --part all --platform laptop
```

**Individual sub-experiments:**
```bash
python run_scaling_study.py --part 1              # ProcessPool strong scaling
python run_scaling_study.py --part 2              # MPI strong scaling
python run_scaling_study.py --part 3              # Async vs Sync DDS
python run_scaling_study.py --part 4              # JAX acceleration
python run_scaling_study.py --part 5              # Weak scaling
python run_scaling_study.py --part 6              # Ensemble parallelism
```

**HPC (via SLURM):**
```bash
cd ../slurm
sbatch submit_all.sh                              # Submit all HPC jobs
```

**Dry run:**
```bash
python run_scaling_study.py --part all --dry-run
```

### Step 3: Analyze Results
```bash
python analyze_scaling.py --output-dir ../results
```

### Step 4: Generate Publication Figures
```bash
python create_publication_figures.py --format pdf
```

## Sub-experiment Details

### 4.11.1 Strong Scaling (ProcessPool)

Fixed problem: HBV + DDS (1,000 iterations). Vary `NUM_PROCESSES`.

| Platform | Workers | Expected |
|----------|---------|----------|
| Laptop | 1, 2, 4, 8 | Near-linear to ~4 cores |
| HPC (single node) | 1, 2, 4, 8, 16, 32, 64 | Diminishing returns past ~16 |

**Metrics:** Wall-clock time, speedup (T1/Tn), parallel efficiency (speedup/n)

### 4.11.2 Strong Scaling (MPI)

Same problem, using `mpirun -n N` with MPI execution strategy.

| Platform | Ranks | Nodes |
|----------|-------|-------|
| Laptop | 1, 2, 4, 8 | 1 |
| HPC | 1, 2, 4, 8, 16, 32, 64, 128 | 1, 2, 4, 8 |

**Comparison overlay:** MPI vs ProcessPool at identical core counts.

### 4.11.3 Async vs Synchronous DDS

Compare synchronous batch DDS with Async-DDS across batch configurations.

| Config | Pool Size | Batch Size | Workers |
|--------|-----------|------------|---------|
| sync_baseline | -- | -- | 4 (laptop) / 16 (HPC) |
| async_small | 5 | 4 | 4 / 16 |
| async_medium | 10 | 8 | 8 / 32 |
| async_large | 20 | 16 | 16 / 64 |

**Metrics:** Worker utilization, convergence trajectory, final KGE.

### 4.11.4 JAX Acceleration

Single-evaluation latency and full-calibration timing across backends.

| Backend | JIT | GPU | Platform |
|---------|-----|-----|----------|
| NumPy | -- | -- | Both |
| JAX CPU | No | No | Both |
| JAX CPU | Yes | No | Both |
| JAX GPU | Yes | Yes | HPC only |

### 4.11.5 Weak Scaling

Fix workers at 8, increase spatial complexity.

| Domain | GRUs | HRUs | Platform |
|--------|------|------|----------|
| Lumped | 1 | 1 | Both |
| Elevation bands | 1 | 12 | Both |
| Semi-distributed | 49 | 379 | Both |
| Distributed | 2,335 | 2,335 | HPC only |

**Metrics:** Time per evaluation, I/O volume, memory footprint.

### 4.11.6 Task-Level Ensemble Parallelism

Multi-model ensemble (HBV, GR4J, FUSE, HYPE), each with DDS 500 iterations.

| Mode | Models | Workers/Model | Total Cores |
|------|--------|---------------|-------------|
| Sequential | 4 serial | 4 | 4 |
| Parallel | 4 concurrent | 1 | 4 |
| Hybrid (HPC) | 4 concurrent | 8 | 32 |

## Expected Figures

| Figure | Description |
|--------|-------------|
| fig_strong_scaling | Speedup curves: ProcessPool vs MPI vs ideal (panels a, b) |
| fig_async_dds | Convergence by strategy (a), worker utilization heatmap (b) |
| fig_jax_acceleration | Single-eval latency bars (a), calibration wall-clock (b) |
| fig_weak_ensemble | Time per eval vs domain size (a), ensemble Gantt chart (b) |

## Platform Reporting

All runs record:
- CPU: model, core count, clock speed
- Memory: total RAM
- Storage: type (SSD/NVMe/HDD), measured I/O throughput
- Software: Python version, OS, MPI implementation, JAX version, CUDA version
- SYMFLUENCE: version, git commit hash

## References

- Tolson, B.A. & Shoemaker, C.A. (2007). Dynamically dimensioned search. WRR 43(1).
- Amdahl, G.M. (1967). Validity of the single processor approach. AFIPS.
- Gustafson, J.L. (1988). Reevaluating Amdahl's law. CACM 31(5).
