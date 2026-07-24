# Root-cause notes: held-out `eval_score` wrong/blank for FUSE, HECHMS, CRHM, SWAT

`backfill_final_eval.py` REPAIRS the symptom (recomputes the held-out KGE from
existing model output and writes/repairs `<run>_<algo>_final_evaluation.json`).
This note records the two ROOT CAUSES so FUTURE runs are correct. The code
changes are left for review rather than applied blindly — each is small but
touches the live optimization path.

---

## Root cause (i): CRHM & SWAT write `KGE_Eval == KGE_Calib` (eval period never sliced)

**Symptom.** The JSON exists but `evaluation_metrics.KGE_Eval` equals
`calibration_metrics.KGE_Calib` exactly (both are the calibration best score).

**Exact locations.**

* `src/symfluence/models/crhm/calibration/optimizer.py`, `run_final_evaluation`,
  lines ~153-157:
  ```python
  metrics['kge'] = kge                       # kge = optimization best score
  calib_metrics = {"KGE": kge, "KGE_Calib": kge}
  eval_metrics  = {"KGE": kge, "KGE_Eval":  kge}   # <-- same value, no eval slice
  ```
* `src/symfluence/models/swat/calibration/optimizer.py`, same method,
  lines ~156-157:
  ```python
  calib_metrics = {"KGE_Calib": metrics.get('kge', -999)}
  eval_metrics  = {"KGE_Eval":  metrics.get('kge', -999)}   # <-- same value
  ```

**Why.** Both optimizers take the single scalar `kge` (the calibration optimum)
and stamp it into BOTH the calibration and evaluation dicts. There is no second
metric computation over the evaluation window. Note also that the underlying
`worker.calculate_metrics` (CRHM `worker.py` ~L307, SWAT `worker.py` ~L570)
does **not** accept or apply a period — it calls
`StreamflowMetrics.align_timeseries(sim, obs)` with no `calibration_period`, so
it always scores the full obs/sim overlap. So simply setting
`config['CALIBRATION_PERIOD']=eval_period` before calling it (the approach that
was attempted) has no effect: the worker ignores it.

**Proposed fix (for review).** Compute the evaluation metric over the held-out
window explicitly. Two viable shapes:

1. Give `worker.calculate_metrics` an optional `period: Optional[Tuple[str,str]]`
   and, when set, pass it through to `align_timeseries(sim, obs,
   calibration_period=period)`. Then in `run_final_evaluation` call it twice —
   once with the calibration window, once with the evaluation window — and use
   each result for the respective dict. Keep the existing "authoritative
   optimization best" logic for `KGE_Calib` only; `KGE_Eval` must come from the
   held-out recompute.
2. Or reuse the exact logic this backfill uses: build the daily sim/obs series
   and call `models/fuse/.../metrics_calculation.align_and_filter` twice with
   `CALIBRATION_PERIOD` set to each window (it is model-agnostic — takes plain
   `pd.Series`). This is what `backfill_final_eval.py` does and is verified.

The other PRMS-family per-model optimizers (`prms`, `gsflow`, `clm`,
`clmparflow`, `cwatm`, `gr`, `hype`, ... — see
`grep -rl 'KGE_Eval' src/symfluence/models/*/calibration/optimizer.py`) share
this pattern and should be audited with the same fix.

---

## Root cause (ii): FUSE (and the generic path) never saves `*_final_evaluation.json`

**Symptom.** FUSE run dirs contain only `*_best_params.json`,
`*_parallel_iteration_results.csv`, and the `final_evaluation/` output dir
(`*_runs_def.nc`) — the top-level `<run>_<algo>_final_evaluation.json` is absent,
so the collector reports a blank `eval_score`. (HECHMS in the observed data root
*does* get a JSON written, but with EMPTY `calibration_metrics` /
`evaluation_metrics` — same net effect: blank `eval_score`.)

**Exact locations.**

* The writer that SHOULD produce the file:
  `src/symfluence/optimization/optimizers/final_evaluation/results_saver.py`,
  `FinalResultsSaver.save_results` (L74), filename built at L91:
  `f'{experiment_id}_{safe_algorithm}_final_evaluation.json'`. It serializes
  `final_result['calibration_metrics']` and `['evaluation_metrics']`.
* FUSE optimizer: `src/symfluence/models/fuse/calibration/optimizer.py`
  (`class FUSEModelOptimizer(BaseModelOptimizer)`). It implements
  `_run_model_for_final_evaluation` (L735) which produces the `.nc`, but it does
  **not** implement a `run_final_evaluation` that returns a dict with populated
  `calibration_metrics`/`evaluation_metrics`, and the JSON never gets saved for
  FUSE. HECHMS reaches `save_results` but hands it empty metric dicts.

**Why.** `save_results` only writes what it is given. When the model's
final-evaluation step returns no metrics (FUSE: not wired to call the saver at
all; HECHMS: calls it with `{}`), the eval score is blank downstream.

**Proposed fix (for review).** In each generic-path model's
`run_final_evaluation`, after the final model re-run:
1. read the simulated daily series from the produced `final_evaluation/` output,
2. load the observed daily series,
3. compute KGE over BOTH the calibration and evaluation windows (same
   `align_and_filter`-twice recipe as `backfill_final_eval.py`),
4. return `{'calibration_metrics': {...}, 'evaluation_metrics': {...},
   'best_params': ...}` and ensure `FinalResultsSaver.save_results(...)` is
   invoked. FUSE already has the readers in
   `models/fuse/calibration/metrics_calculation.py`; the backfill tool is a
   drop-in reference implementation.

---

## Verification of the backfill (data root: `SYMFLUENCE_data`, 2026-07-24)

Dry-run, symfluence env:

* FUSE `abc_cal_ensemble_fuse_abc`: recomputed **calib KGE = 0.87053**, which
  matches the recorded `best_score = 0.8705254...` in `*_best_params.json` to
  5 decimals — proving the recompute reproduces the calibration exactly (same
  catchment area, obs, and slice). **eval KGE = 0.83473** genuinely differs
  (delta -0.036), confirming a real held-out slice rather than an echo.
* All 16 FUSE runs and 18 HECHMS runs compute distinct calib vs eval KGE.
* CRHM `dds_run_1` is correctly SKIPPED — its `final_evaluation/` dir is empty
  (no `CRHM_output_*.txt`), so there is nothing to recompute; no penalty/echo is
  written.
* No SWAT `optimization/SWAT` run dirs exist in the current data root, so the
  SWAT reader path is implemented but UNVERIFIED end-to-end (see the TODO in
  `read_swat_sim`: `output.rch` carries no absolute dates, so the daily index is
  reconstructed from `EXPERIMENT_TIME_START + SWAT_WARMUP_YEARS`; a run is
  skipped rather than guessed when the start date is unavailable).
