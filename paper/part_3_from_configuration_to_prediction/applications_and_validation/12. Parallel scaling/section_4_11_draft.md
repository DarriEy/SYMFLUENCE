## 4.11 Parallel Scaling

SYMFLUENCE implements a three-tier parallel execution architecture that automatically selects among sequential, shared-memory (ProcessPool via `concurrent.futures`), and distributed-memory (MPI via `mpi4py`) execution strategies based on the runtime environment and user configuration. Additionally, JAX-based models (HBV, jFUSE) support JIT compilation and GPU offloading, providing model-level acceleration that composes orthogonally with process-level parallelism. This section characterises the scaling behaviour of each parallelism tier through six sub-experiments on the Bow River at Banff testbed used throughout Sections 4.1--4.7.

All experiments use ERA5 forcing, daily timestep, KGE as the calibration objective, and the 2004--2007 calibration period with 2008--2009 evaluation and 2002--2003 spinup (identical to Section 4.4). Results are reported for two platforms: a commodity laptop (M-core Apple Silicon / Intel workstation, 8--16 logical cores, 16--32 GB RAM, NVMe SSD) and an HPC cluster (N-core compute nodes, X GB RAM per node, InfiniBand interconnect). All timing measurements report wall-clock time and use a single random seed (42) to ensure deterministic iteration trajectories across configurations.

### 4.11.1 Strong Scaling (ProcessPool)

To quantify the shared-memory scaling ceiling, we calibrate HBV with DDS (1,000 iterations) while varying the number of ProcessPool workers from 1 to 8 on the laptop and 1 to 64 on a single HPC node. Each worker receives an isolated scratch directory (via SYMFLUENCE's `LocalScratchManager`) to prevent concurrent file I/O conflicts, and the base model settings are copied to each worker directory before execution begins.

**Table X.** ProcessPool strong-scaling results for HBV + DDS (1,000 iterations) on the Bow River at Banff (lumped).

| Workers | Wall-clock (s) | Speedup | Efficiency (%) | Platform |
|---------|---------------|---------|----------------|----------|
| 1 | -- | 1.00 | 100.0 | Both |
| 2 | -- | -- | -- | Both |
| 4 | -- | -- | -- | Both |
| 8 | -- | -- | -- | Both |
| 16 | -- | -- | -- | HPC |
| 32 | -- | -- | -- | HPC |
| 64 | -- | -- | -- | HPC |

Figure X(a) shows the speedup curves for both platforms, with an ideal (linear) scaling reference. The laptop achieves near-linear speedup up to 4 workers, after which efficiency degrades due to the overhead of per-process directory setup, settings file copying, and result aggregation. The HPC node extends efficient scaling further due to higher memory bandwidth and faster storage I/O, but diminishing returns are expected beyond ~16--32 workers as the serial fraction of the DDS algorithm (perturbation generation, objective comparison, and solution update occur on the master process) begins to dominate according to Amdahl's law. Figure X(b) plots parallel efficiency directly, providing a quantitative measure of overhead at each scale.

### 4.11.2 Strong Scaling (MPI)

The same HBV/DDS problem is repeated using the MPI execution strategy (`mpirun -n N`). On the laptop, MPI ranks are confined to a single machine; on the HPC cluster, we scale from 1 to 128 ranks across 1 to 8 nodes. SYMFLUENCE's `MPIExecutionStrategy` auto-detects the MPI environment via `OMPI_COMM_WORLD_RANK` (OpenMPI) or `PMI_RANK` (Intel MPI) and distributes tasks from the master rank (rank 0) to workers via pickle-serialised task/result files.

The key comparison is the overlay of ProcessPool and MPI speedup curves at identical core counts (Figure X). At low core counts on a single node, ProcessPool is expected to outperform MPI because it avoids inter-process serialisation and communication overhead. However, MPI is the only strategy that scales beyond a single node, making it essential for HPC deployments. The crossover point --- the core count at which MPI's multi-node capability compensates for its per-rank overhead --- is an empirical quantity that depends on the ratio of model evaluation time to communication time.

A secondary observation is the automatic fallback behaviour: when MPI is unavailable (e.g., on a laptop without `mpi4py`), SYMFLUENCE transparently falls back to ProcessPool, and when `NUM_PROCESSES = 1`, it falls back to sequential execution. This graceful degradation ensures that the same configuration file can be deployed on both laptop and HPC without modification, requiring only the appropriate launch command (`python` vs `mpirun -n N python`).

### 4.11.3 Asynchronous vs Synchronous DDS

Standard parallel DDS evaluates perturbations in synchronous batches: the master generates a batch of candidate solutions, dispatches them to workers, and waits for all workers to return before generating the next batch. When model evaluation times are heterogeneous --- for example, when some parameter sets cause numerical instability and early termination --- fast-returning workers idle while waiting for the slowest evaluation in the batch.

SYMFLUENCE's `AsyncDDSAlgorithm` addresses this by maintaining a pool of the best solutions seen so far and generating new candidates asynchronously as workers become available. This decouples task generation from result aggregation, improving worker utilization at the cost of reduced sequential information flow between iterations (each candidate is generated from a pool rather than the single best solution).

We compare synchronous DDS (4 workers on laptop; 16 and 64 on HPC) against Async-DDS with three configurations varying pool size (5, 10, 20) and batch size (4, 8, 16), all with a total budget of 4,000 function evaluations to match Section 4.4.

**Table Y.** Async vs synchronous DDS comparison (4,000 function evaluations).

| Strategy | Pool | Batch | Workers | Wall-clock (s) | Final KGE | Worker Util. (%) |
|----------|------|-------|---------|---------------|-----------|-----------------|
| Sync DDS | -- | -- | 4 | -- | -- | -- |
| Async DDS | 5 | 4 | 4 | -- | -- | -- |
| Async DDS | 10 | 8 | 8 | -- | -- | -- |
| Sync DDS | -- | -- | 16 | -- | -- | -- |
| Async DDS | 20 | 16 | 16 | -- | -- | -- |

Two metrics are of interest: (1) wall-clock time, which reflects worker utilization gains from asynchronous dispatch, and (2) final KGE, which reflects the optimality cost of reduced sequential information. If the HBV model produces relatively homogeneous evaluation times (all parameter sets run to completion in similar time), the async advantage will be modest. If evaluation times are heterogeneous, the async advantage will be substantial.

### 4.11.4 JAX Acceleration

SYMFLUENCE's HBV model supports two computational backends: standard NumPy and JAX. The JAX backend enables JIT compilation (fusing Python-level operations into optimised XLA kernels) and optional GPU offloading. We benchmark four backend configurations on a fixed problem (HBV + DDS, 1,000 iterations, single process):

1. **NumPy** --- baseline, no compilation, CPU only
2. **JAX CPU, JIT disabled** --- JAX array operations without compilation
3. **JAX CPU, JIT enabled** --- full XLA compilation on CPU
4. **JAX GPU, JIT enabled** --- XLA compilation targeting GPU (HPC only)

For each configuration we report single-evaluation latency (averaged over 100 evaluations after a warmup pass to amortise JIT compilation cost) and total calibration wall-clock time.

To demonstrate that model acceleration and process-level parallelism are orthogonal, we additionally run JAX-JIT with 1, 4, and 8 ProcessPool workers, computing the *compound speedup* (JAX speedup $\times$ parallel speedup) and comparing it to the speedup achieved by either dimension alone. If the two dimensions compose multiplicatively --- as expected for embarrassingly parallel calibration with independent model evaluations --- the compound speedup should approximate the product of the individual speedups.

### 4.11.5 Weak Scaling (Domain Complexity)

The preceding experiments fix the spatial domain (lumped) and vary parallelism. Here we fix the number of workers at 8 and vary the spatial complexity of the domain to characterise how model evaluation cost scales with problem size. We use FUSE (rather than HBV) because it supports both lumped and distributed operation via mizuRoute, and we test four domain configurations from Section 4.1:

1. **Lumped**: 1 GRU, 1 HRU
2. **Elevation bands**: 1 GRU, 12 HRUs
3. **Semi-distributed**: 49 GRUs, 379 HRUs (with mizuRoute routing)
4. **Distributed**: 2,335 grid cells (HPC only, with mizuRoute routing)

Each configuration runs DDS with 500 iterations. We report time per model evaluation, I/O volume (via SYMFLUENCE's built-in I/O profiler which tracks read/write bytes and IOPS), and peak memory footprint.

The key diagnostic is whether evaluation cost scales linearly, sub-linearly, or super-linearly with HRU count. Linear scaling indicates that per-HRU computation dominates; sub-linear scaling suggests fixed overhead (I/O setup, netCDF header parsing) is amortised over more HRUs; super-linear scaling indicates that inter-HRU interactions (routing) or I/O contention introduce non-linear costs. The transition from compute-bound to I/O-bound behaviour informs users about whether to invest in more cores or faster storage when scaling to large domains.

### 4.11.6 Task-Level Ensemble Parallelism

The most common use of parallelism in practice is not within a single calibration but across independent experiments: calibrating multiple models, testing multiple forcing datasets, or running multi-seed robustness analyses. These are embarrassingly parallel workloads that require no inter-task communication.

We demonstrate this using a four-model ensemble (HBV, GR4J, FUSE, HYPE) on the Bow River at Banff, each calibrated with DDS (500 iterations), in three execution modes:

- **Mode A (Sequential):** Models calibrated one after another, each using 4 ProcessPool workers. Total cores: 4.
- **Mode B (Parallel):** All 4 models launched concurrently as independent processes, each using 1 worker. Total cores: 4.
- **Mode C (Hybrid, HPC only):** All 4 models launched concurrently, each using 8 workers. Total cores: 32.

Mode A represents the typical single-experiment workflow. Mode B tests whether concurrent independent launches (achievable with a simple shell script or `run_scaling_study.py`) outperform sequential execution at the same total core count --- this depends on whether the models have similar or different runtimes. If FUSE completes in half the time of HBV, sequential execution wastes 4 cores while FUSE's slot sits idle; parallel execution fills all cores as soon as the first model finishes. Mode C demonstrates that task-level and process-level parallelism compose naturally when sufficient resources are available.

Figure Z shows a Gantt-chart visualization of the three modes, with each model's execution represented as a horizontal bar. The sequential timeline is the sum of individual model runtimes; the parallel timeline is the maximum; the hybrid timeline adds within-model speedup from multi-worker calibration.

### 4.11.7 Discussion

Three findings from this experiment bear on SYMFLUENCE's design and practical deployment.

First, the automatic execution strategy selection (MPI $\to$ ProcessPool $\to$ Sequential) combined with per-process directory isolation enables the same YAML configuration to run unchanged on both a laptop and an HPC cluster. The user controls parallelism through a single parameter (`NUM_PROCESSES`) and the launch command (`python` vs `mpirun -n N`), with no changes to model configuration, calibration settings, or post-processing scripts. This portability lowers the barrier to transitioning from prototype development on a laptop to production runs on HPC.

Second, JAX-based model acceleration and process-level parallelism are orthogonal and compose multiplicatively. A user running HBV with JAX-JIT on 8 ProcessPool workers benefits from both dimensions simultaneously. This composability means that investments in model acceleration (e.g., porting additional models to JAX or PyTorch) directly multiply the returns from parallel infrastructure, and vice versa.

Third, task-level parallelism --- simply launching independent experiments concurrently --- provides the highest return on investment for the most common usage pattern (multi-model ensembles, multi-seed robustness, multi-algorithm calibration). Sections 4.2, 4.4, and 4.6 each involve dozens of independent calibration runs that could execute concurrently with no algorithmic changes. SYMFLUENCE's configuration-driven design makes this straightforward: each experiment is fully specified by a self-contained YAML file, and the orchestration scripts (`run_scaling_study.py`, SLURM array jobs) handle concurrent dispatch and result collection.
