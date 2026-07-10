# Experiment 04 — Calibration Algorithm Comparison (§4.2.3, Fig 9)

130 calibration runs cross 8 hydrological models with up to 17 optimization
algorithms on the identical problem: Bow River at Banff (lumped domain,
`Bow_at_Banff_lumped_calibration_ensemble`), RDRS forcing, KGE objective,
calibration 2004–2007, evaluation 2008–2009, `random_seed: 42`. Each Fig 9
heatmap cell is one of these runs. Gradient-based algorithms (ADAM, L-BFGS)
are only defined for the five JAX-based models and use
`gradient_mode: finite_difference`; the three Fortran-based models get the 15
derivative-free algorithms (17 × 5 + 15 × 3 = 130).

## Configs (one directory per model: `<model>/config_bow_<model>_<algorithm>.yaml`)

| Directory | Model | Algorithms |
|---|---|---|
| `hbv/` | HBV (JAX) | 17 |
| `hechms/` | HEC-HMS (JAX) | 17 |
| `sacsma/` | SAC-SMA (JAX) | 17 |
| `topmodel/` | TOPMODEL (JAX) | 17 |
| `xinanjiang/` | Xinanjiang (JAX) | 17 |
| `fuse/` | FUSE (Fortran) | 15 (no ADAM/L-BFGS) |
| `summa/` | SUMMA (Fortran) | 15 |
| `hype/` | HYPE (Fortran) | 15 |

Algorithms: DDS, SCE-UA, PSO, DE, CMA-ES, GA, SA, Nelder-Mead, Basin-Hopping,
Bayesian Opt, DREAM, GLUE, ABC, NSGA-II, MOEA/D (+ ADAM, L-BFGS on JAX
models).

Evaluation budgets are matched at ≈1,000 model evaluations for the
derivative-free methods (e.g. DDS/Nelder-Mead 1,000 iterations; DE/PSO/
CMA-ES/GA 50 generations × 20; NSGA-II/MOEA/D 20 × 50; GLUE/ABC 10 × 100).
The gradient methods use fewer optimizer steps (ADAM 500, L-BFGS 125), each
step costing several finite-difference model evaluations. Warm starting is
disabled (`skip_warm_start: true`) for a fair comparison.

## Run

```bash
CFG=examples/paper_case_studies/configs/04_calibration_ensemble

# One run (one heatmap cell):
symfluence workflow run --config $CFG/hbv/config_bow_hbv_dds.yaml

# All 130:
for f in $CFG/*/config_bow_*.yaml; do
    symfluence workflow run --config "$f"
done
```

All configs share one domain; the first run acquires RDRS + WSC streamflow
(station 05BB001), the rest reuse it via stage markers. FUSE/SUMMA/HYPE need
their executables installed; the five JAX models ship with
`pip install -e ".[jax]"`.

## Outputs

Under `SYMFLUENCE_data/domain_Bow_at_Banff_lumped_calibration_ensemble/`:

- `optimization/` — iteration history and best parameters per run
  (`experiment_id: cal_ensemble_<model>_<algorithm>`)
- `simulations/cal_ensemble_<model>_<algorithm>/` — model output
- `evaluation/` — calibration/evaluation KGE per run

## Verify (Fig 9 reference values)

- Top tier (DE, CMA-ES, DDS, ADAM): mean calibration KGE **0.867–0.872**.
- Generalization: ADAM mean evaluation KGE **0.769** vs DDS **0.705** on the
  JAX models.
- Bayesian Optimization fails on SAC-SMA's 26-parameter space
  (KGE ≈ **−0.03**) — expected, not a setup error.

Runs are seeded; differences below ~0.02 KGE are within cross-platform noise.

## Runtime

10 min – 2 h per run: minutes for JAX models, the long end for SUMMA/FUSE/
HYPE. The full 130-run sweep is dominated by the three Fortran models.
