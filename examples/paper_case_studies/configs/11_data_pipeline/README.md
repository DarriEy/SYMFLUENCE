# Experiment 11 — Data Processing Pipeline (§2.2, Fig 4)

Three configs exercise the data pipeline end to end — attribute acquisition,
domain definition/discretization, observation processing, forcing
acquisition, model-agnostic preprocessing, and the model-ready store — at
three scales, **without running any hydrological model**. Each config's
`workflow_steps` list stops at `build_model_ready_store`. Fig 4 characterizes
the resulting data volumes and remapping workloads.

## Configs (`configs/`)

| Config | Domain | Forcing | Observations |
|---|---|---|---|
| `config_paradise.yaml` | `paradise_snotel_wa` — point scale (WA) | ERA5 | SNOTEL |
| `config_bow.yaml` | `Bow_at_Banff_lumped` — lumped catchment (AB) | ERA5 | WSC |

All data sources are public (ERA5 via the ARCO cloud mirror); SNOTEL and
WSC are public.

## Run

```bash
CFG=examples/paper_case_studies/configs/11_data_pipeline

# One scale:
symfluence workflow run --config $CFG/configs/config_bow.yaml

# All three:
for f in $CFG/configs/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

## Outputs

Under `SYMFLUENCE_data/domain_<name>/`:

- `shapefiles/` — delineated/discretized domain geometry
- `data/forcing/` — raw and basin-averaged (EASYMORE-remapped) forcing
- `data/observations/` — processed streamflow/SWE observations
- `data/attributes/` — DEM, soil, land-cover attributes
- the model-ready store produced by `build_model_ready_store`

No `simulations/` or `optimization/` output is produced.

## Verify (Fig 4 reference values)

| Domain | Model-ready data volume |
|---|---|
| Paradise (point) | 12.3 MB |
| Bow at Banff (lumped) | 132.9 MB |
| Iceland (distributed) | 4.3 GB |

Bow intermediate checkpoints: **20 ERA5 grid cells** intersect the basin,
producing **102 EASYMORE catchment–cell intersections** during remapping.

## Runtime

0.5–2 h per config, dominated by forcing download (the
long end and needs ~5 GB of disk). No model executables required.
