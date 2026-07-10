# Experiment 12 — Parallel Calibration Scaling (§5, Fig 11)

35 configs measure how SUMMA calibration throughput scales with worker count,
comparing ProcessPool and MPI execution backends on the Bow at Banff lumped
domain (ERA5 forcing). Fig 11's top row is a laptop-scale sweep (~100 model
evaluations, 1–10 workers); the bottom row is an HPC sweep (20,000-iteration
DDS, 10–100 cores). All configs set `force_run_all_steps: true` so timing
runs re-execute rather than skip steps; `random_seed: 42` keeps trajectories
comparable across worker counts.

## Configs

**`exp2_calibration/` — 14 laptop configs (Fig 11, top row)**

- `calib_summa_pp_100evals_np{1,2,3,4,6,8,10}.yaml` — ProcessPool
- `calib_summa_mpi_100evals_np{1,2,3,4,6,8,10}.yaml` — MPI

ASYNC-DDS with iterations set to ≈100/N (np1: 100 … np4: 25 … np10: 10), so
every run costs ≈100 total model evaluations regardless of worker count;
domain `Bow_at_Banff_lumped_era5`, `system.num_processes` = N.

**`exp2_calibration/` — 20 HPC configs (Fig 11, bottom row) — cluster only**

- `calib_summa_pp_20000iter_np{10,20,...,100}.yaml` — ProcessPool
- `calib_summa_mpi_20000iter_np{10,20,...,100}.yaml` — MPI

DDS, 20,000 iterations at 10–100 processes. **These require an HPC cluster
(the paper runs used DRAC); they are not runnable on a laptop.**

**`configs/` — 1 base config**

- `config_calibration_scaling.yaml` — the base calibration-scaling experiment
  (DDS, 20,000 iterations, parallel evaluation, domain `Bow_at_Banff_scaling`).

Note: each `pp`/`mpi` config pair is identical except for its
`experiment_id` (which separates the result sets). The backend is chosen at
runtime: SYMFLUENCE uses a persistent MPI worker pool when MPI + `mpi4py`
are available in the environment and falls back to a ProcessPool otherwise —
so run the `mpi` variants inside an MPI-enabled environment and the `pp`
variants without one.

## Run

```bash
CFG=examples/paper_case_studies/configs/12_parallel_scaling

# One laptop point:
symfluence workflow run --config $CFG/exp2_calibration/calib_summa_pp_100evals_np4.yaml

# Full laptop sweep (Fig 11 top row):
for f in $CFG/exp2_calibration/calib_summa_*_100evals_np*.yaml; do
    symfluence workflow run --config "$f"
done

# HPC sweep (Fig 11 bottom row) — submit on a cluster:
for f in $CFG/exp2_calibration/calib_summa_*_20000iter_np*.yaml; do
    symfluence workflow run --config "$f"
done
```

SUMMA must be installed (`symfluence binary install summa sundials`).

## Outputs

Under `SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/`:

- `optimization/` — per-run iteration history
  (`<experiment_id>_parallel_iteration_results.csv`); wall-clock timing for
  the speedup curves comes from the run logs in `_workLog_*/`

## Verify (Fig 11 reference values)

- Laptop: ProcessPool speedup plateaus at **2.4×** around 4–6 processes;
  MPI peaks at **1.4×** (per-batch startup overhead dominates at this scale).
- Cluster: asynchronous DDS reaches **9.7×** at 100 cores.

## Runtime

Laptop configs: minutes each once the shared domain exists (~100 SUMMA runs).
HPC configs: hours each at 20,000 evaluations, depending on core count.
