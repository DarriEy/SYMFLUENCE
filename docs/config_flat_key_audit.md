# Flat config key audit (`RECOGNIZED_FLAT_KEYS`)

**Date:** 2026-06-10 · **Scope:** the 125 keys that were in
`core/config/legacy_aliases.py::RECOGNIZED_FLAT_KEYS` — real, consumed-in-code
flat keys with no nested Pydantic field, riding in the config `_extra`
passthrough. This audit is the pre-1.0 step of pinning the Pydantic schema as
the canonical config contract (every key should either have a typed field or
be an explicitly blessed extra).

Method: automated reader census over `src/symfluence` for every key
(`config.get('KEY')` flat reads, `_get_config_value(..., dict_key='KEY')`
hybrid reads, and other occurrences — alias declarations, key-name strings,
docstrings), followed by manual classification. All 125 keys have at least
one in-tree reader; none are dead.

## Headline findings

1. **24 keys were schema-complete but disconnected (fixed in this change).**
   `StateConfig`, `EnKFConfig` and `DataAssimilationConfig`
   (`core/config/models/state_config.py`) define typed, aliased fields for all
   `STATE_*` (8), `ENKF_*` (15) and `DA_METHOD` keys and are wired into the
   root config — but the introspection walker
   (`introspection.generate_flat_to_nested_map`) hard-codes the root sections
   it visits, and `state` / `data_assimilation` were missing. Consequence:
   flat configs never populated those sections, and
   `data_assimilation/da_manager.py` (which reads
   `config.data_assimilation.enkf.*` typed) silently ran on **defaults**,
   ignoring the user's EnKF settings. Fixed by walking both sections (and
   mirroring them in `flattening.py` so `config.get('ENKF_*')` round-trips);
   the 24 keys were removed from `RECOGNIZED_FLAT_KEYS`. Regression tests in
   `tests/unit/config/test_state_da_flat_keys.py`, including a guard that
   walks *all* root-model sections so a future section can't detach again.

2. **Divergent duplicate schemas exist for some models.** The registered
   schema in `R.config_schemas` is not always the complete one:
   - `IGNACIO`: `core/config/models/model_configs_ml_fire.IGNACIOConfig`
     (registered) is a subset; the full schema with all 20 `IGNACIO_*` aliases
     lives in `models/ignacio/config.py` and is validated directly from the
     flat dict by the model.
   - `GNN` / `LSTM`: `GNN_OUTPUT_SIZE`, `GNN_USE_SNOW`,
     `LSTM_PARAMETER_BOUNDS`, `LSTM_PARAMS_TO_CALIBRATE` are declared in the
     legacy `models/config/model_config_schema.py` ConfigKey tables but not in
     the registered Pydantic configs.
   Reconciling each model to a single registered schema removes those keys
   from the recognized set for free.

3. **The remaining 101 keys fall into three dispositions** (table below):
   promote into an existing typed section, reconcile a divergent schema, or
   bless as a documented extra. None block 1.0 — recognition + strict
   validation already covers them — but each promotion shrinks the
   non-canonical surface.

## Disposition table (101 remaining keys)

Reader pattern legend: *flat* = `config.get('KEY')`; *hybrid* =
`_get_config_value(..., dict_key='KEY')`; *validate* = consumed by
`Model.model_validate(flat_dict)` against field aliases.

| Family (keys) | Readers | Pattern | Disposition |
|---|---|---|---|
| `MULTI_GAUGE_*` (10) | fuse/hype/summa calibration workers, `data_manager`, `observations_builder` | flat + hybrid | **Promote** to an `optimization.multi_gauge` sub-model. Highest-value promotion: documented public feature (CLAUDE.md), 3 model workers read the same keys. Migrate readers in the same change. |
| `IGNACIO_*` (20) | `models/ignacio/config.py` | validate | **Reconcile schemas**: register the full `models/ignacio/config.py::IGNACIOConfig` (or merge fields into the core one). Keys then map automatically via the `R.config_schemas` walk. |
| `HYPE_*` process options (9: DEEP_GROUND, FROZEN_SOIL_MODEL, INFILTRATION_MODEL, PARAM_BOUNDS, PET_MODEL, SNOW_EVAPORATION, SOIL_INIT_WET, SOIL_LAYER_DEPTHS, SURFACE_RUNOFF) | `hype/config_manager.py`, `geodata_manager.py`, parameter manager | hybrid | **Promote** into `HYPEConfig` (registered, already walked). |
| Param bounds & initial params (7: `PARAMETER_BOUNDS`, `INITIAL_PARAMETERS`, `CLM_/MESH_/PARFLOW_/RHESSYS_/VIC_PARAM_BOUNDS`) | per-model parameter managers | hybrid | **Promote** per-model (dict-valued fields in each model config); `PARAMETER_BOUNDS`/`INITIAL_PARAMETERS` → `optimization`. |
| Optimizer/evaluator odds (8: `ADAM_STEPS`, `DDS_STAGNATION_THRESHOLD`, `SKIP_WARM_START`, `OPTIMIZATION_MAX_ITERATIONS`, `LIKELIHOOD_FUNCTION`, `MODEL_ERROR_BASE/FRACTION`, `TRANSFER_FUNCTION_TYPE/B_BOUNDS`) | base optimizer, DDS, evaluators/base, regionalization | hybrid/flat | **Promote** into `optimization` / `evaluation` section models. |
| NGEN family (6: `CALIBRATION_NEXUS_ID`, `CALIBRATION_WARMUP_DAYS`, `SETTINGS_NGEN_REALIZATION`, `EXPERIMENT_OUTPUT_NGEN`, `NWS_HYDROFABRIC_VERSION`, + deprecated `ENABLE_NOAH/PET/SLOTH` (3)) | ngen targets/extractor/optimizer, mizuroute control writer | mixed | **Promote** into `NGENConfig`, except `ENABLE_*` which are already-deprecated spellings handled by NGEN transformers (`model_configs_hydrology.py:253`) — keep recognized until removal at 2.0. |
| GNN/LSTM emulators (5: `GNN_OUTPUT_SIZE`, `GNN_USE_SNOW`, `LSTM`, `LSTM_PARAMETER_BOUNDS`, `LSTM_PARAMS_TO_CALIBRATE`) | gnn/lstm runners + optimizers | hybrid | **Reconcile schemas** (add fields to registered `GNNConfig`/`LSTMConfig`); audit the bare `LSTM` key individually (name collides with the model identifier everywhere). |
| Data-acquisition / observation handler keys (~25: `CANSWE_*` + `DOWNLOAD_CANSWE` (4), `ESA_CCI_SM_*` (6), `GLEAM_ET_*` + `ET_UNIT_CONVERSION` (3), `SNOTEL_STATE`, `SMAP_LAYER`, `GRACE_SUBSET`, `TDX_SOURCE`, `HYDROSHEDS_LEVEL`, `CARRA_DOMAIN`, `EM_EARTH`, `MODIS_SNOW`, `USGS_GW`, `GW_AUTO_ALIGN`, `GW_BASE_DEPTH`, …) | one handler each (1–3 sites) | hybrid | **Bless as extras for 1.0; promote opportunistically.** Each key has exactly one owner; the natural long-term home is per-handler declared key sets (the same mechanism external plugins use), not one giant `DataConfig`. |
| Model misc (8: `FUSE_RUN_MODE`, `FUSE_TEMPLATE_PATH`, `GR_MODEL_TYPE`, `MIZUROUTE_NUM_THREADS`, `CATCHMENT_SHP_PATH`, `CATCHMENT_SHP_SLOPE_UNITS`, `FORCING_RAW_PATH`, `DECISION_OPTIONS`) | respective model code | mixed | **Promote** into the owning model configs. Note `DECISION_OPTIONS` is the model-agnostic spelling; `SUMMA_DECISION_OPTIONS`/`FUSE_DECISION_OPTIONS` already have typed fields — decide alias-vs-promote. |
| Cross-cutting (2: `GAUGE_SEGMENT_MAPPING`, `STATE…` n/a) | multi-model workers | flat | **Promote** alongside the multi-gauge sub-model (same consumers). |

## Recommended sequencing

1. ~~Wire `state` + `data_assimilation` (this change — fixes a real bug).~~
2. Reconcile divergent schemas (IGNACIO, GNN, LSTM) — pure wiring, ~30 keys.
3. Promote the multi-gauge family (one sub-model, three workers to migrate).
4. Promote optimization/evaluation odds and HYPE/NGEN/model-misc families.
5. Leave data-handler keys as blessed extras for 1.0; revisit with per-handler
   key declaration (mirrors the external-plugin mechanism, ADR-0002).

Rule of thumb established by the `state` fix: **promote key and reader
together** — a key given a nested path leaves `_extra`, so any remaining flat
reader must be checked (the flat view still serves promoted keys via
`flatten_nested_config`, but only when flattening covers the section).
