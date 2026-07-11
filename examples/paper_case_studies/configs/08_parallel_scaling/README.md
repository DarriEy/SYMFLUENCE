# 08 — Parallel scaling (§5, Fig 11)

Reproduces the **top row of Fig 11**: DDS calibration of lumped SUMMA on Bow
at Banff (~100 total evaluations, ~15 s each) at 7 process counts
(np = 1, 2, 3, 4, 6, 8, 10) for two execution backends. One config per
plotted point:

| Configs | Backend |
|---|---|
| `calib_summa_pp_100evals_np{1,2,3,4,6,8,10}.yaml` | ProcessPool (shared memory) |
| `calib_summa_mpi_100evals_np{1,2,3,4,6,8,10}.yaml` | MPI (distributed memory) |

The pp/mpi pairs differ only in `experiment_id` (separate result
directories); there is no MPI switch in the config — the backend is selected
at runtime. Run the `mpi_*` configs in an MPI-enabled environment
(`mpi4py` installed, launched under `mpirun`); without MPI they fall back to
ProcessPool.

```bash
CFG=examples/paper_case_studies/configs/08_parallel_scaling
for f in $CFG/calib_summa_pp_100evals_np*.yaml; do
    symfluence workflow run --config "$f"
done
```

Reference values (manuscript): ProcessPool speedup plateaus at 2.4× with 4–6
processes; MPI peaks at 1.4× (3 processes) and degrades below serial at 10.

The **bottom row of Fig 11** (Async-DDS vs differential evolution, 10–100
cores, ~60 s per evaluation, catchment CAN_01AD003) was run on the DRAC Fir
cluster and is not reproducible on a laptop; no configs are shipped for it.
