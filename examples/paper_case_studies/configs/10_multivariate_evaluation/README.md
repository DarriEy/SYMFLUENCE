# Experiment 10 — Multivariate Evaluation (§4.2.4, Fig 10)

SUMMA at Bow River at Banff (domain `Bow_at_Banff_multivar`, RDRS forcing) is
calibrated four ways against two observation types — WSC streamflow
(05BB001) and GRACE total water storage (JPL RL06 mascon) — to test whether
joint calibration buys TWS realism without sacrificing streamflow skill
(Fig 10). Extended periods: calibration 2004–2010, evaluation 2011–2017.

## Configs

| Config | Strategy | Optimizer / budget |
|---|---|---|
| `config_streamflow_only.yaml` | Streamflow KGE only | DDS, 2,000 iterations |
| `config_tws_only.yaml` | GRACE TWS correlation only | DDS, 2,000 iterations |
| `config_joint.yaml` | Joint Q + TWS (multi-objective) | NSGA-II, 50 gen × 40 pop = 2,000 evals |
| `config_moead_joint.yaml` | Joint Q + TWS (multi-objective) | MOEA/D, 50 gen × 40 pop = 2,000 evals |

All four calibrate the same 17 SUMMA parameters (snow, soil, groundwater +
routing) and run the full pipeline through `calibrate_model`.

## Credentials

GRACE TWS: the shipped configs use the JPL mascon solution the paper used,
which needs Earthdata credentials in `~/.netrc` (see top-level README §1.4).
The CSR and GSFC solutions are public — set `evaluation.grace.product`
accordingly to run without credentials (results will differ slightly from
the paper's JPL-based values). RDRS and WSC data are public.

## Run

```bash
CFG=examples/paper_case_studies/configs/10_multivariate_evaluation

# One strategy:
symfluence workflow run --config $CFG/config_streamflow_only.yaml

# All four:
for f in $CFG/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

The four configs share the domain, so forcing/observation acquisition happens
once.

## Outputs

Under `SYMFLUENCE_data/domain_Bow_at_Banff_multivar/`:

- `optimization/` — iteration history, best parameters, and (for NSGA-II /
  MOEA/D) the Pareto front per experiment (`bow_exp1_streamflow_only` …
  `bow_exp4_moead_joint`)
- `simulations/<experiment_id>/` — SUMMA output
- `evaluation/` — streamflow KGE and TWS correlation per run

## Verify (Fig 10 reference values, evaluation period)

| Strategy | Streamflow KGE | TWS r |
|---|---|---|
| Streamflow-only | 0.88 | 0.75 |
| TWS-only | 0.57 | 0.87 |
| Joint (NSGA-II) | 0.87 | 0.84 |
| Joint (MOEA/D) | 0.87 | 0.85 |

The headline result: joint calibration recovers nearly all single-objective
skill on both variables simultaneously.

## Runtime

2–4 h per config (2,000 SUMMA evaluations over a 16-year period). SUMMA must
be installed (`symfluence binary install summa sundials`).
