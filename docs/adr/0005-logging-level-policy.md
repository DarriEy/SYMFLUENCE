# ADR-0005: Logging-level policy — ERROR is the operational ceiling

- **Status:** Accepted
- **Date:** 2026-06-05

## Context

Across the source tree, nothing is logged at the `CRITICAL` level. Without a
written policy, a contributor cannot tell whether that is deliberate — the
project has no events it considers critical — or whether the level has simply
not been adopted, and therefore cannot know whether to reach for it. Either
position is fine; what was missing is the written rule.

## Decision

`ERROR` is the **operational ceiling** for routine logging. It is the level used
when an operation fails — a model run crashes, a data acquisition fails, a
metric cannot be computed — and the framework recovers, skips, or surfaces the
failure while continuing. This is consistent with the framework's resilience
posture (`except Exception as e: # noqa: BLE001`), where individual failures are
logged at `ERROR` and the pipeline continues.

`CRITICAL` is **reserved** for the single orchestrator-boundary case where the
*entire run* is being aborted — i.e. the framework cannot continue at all. It is
deliberately rare; there are zero `.critical()` calls in the tree today, and a
contributor should reach for it only at that top-level abort boundary, not for
an individual component failure (which is `ERROR`).

Level guidance, summarized:

| Level | Use for |
|-------|---------|
| `DEBUG` | Diagnostic detail; off by default, surfaced with `--debug` |
| `INFO` | Normal pipeline progress and milestones |
| `WARNING` | Recoverable anomaly; degraded but functioning |
| `ERROR` | An operation failed; framework recovers/skips and continues |
| `CRITICAL` | The whole run is aborting; framework cannot continue (rare) |

## Protocol conventions

The logging-protocol overhaul (2026-07) turned the level table above into a set
of concrete, enforced conventions. The shared implementation lives in
`src/symfluence/core/logging_utils.py`.

### One-time semantics for recurring conditions

A condition that recurs inside a hot loop (per-parameter bounds warnings,
per-evaluation missing-forcing errors, repeated worker exceptions) must not be
re-emitted at full level every iteration. Use:

```python
from symfluence.core.logging_utils import log_once
log_once(logger, logging.WARNING, key, message)
```

The first occurrence of `key` (process-wide, thread-safe) is emitted at the
requested level; every subsequent call with the same key is demoted to `DEBUG`.
This preserves the level policy — the anomaly is still a `WARNING` — without
letting it dominate the log.

### Loop rule: per-item DEBUG, one INFO summary

Inside a loop over files/chunks/parameters, per-item lines go to `DEBUG`. The
loop emits exactly one `INFO` summary, e.g.:

```
89/96 chunks cached, downloading 7
```

`INFO` is for milestones; iterating is not a milestone.

### No whole-object dumps above DEBUG

Dumps of lists, array shapes, dtypes, DataFrame heads, and similar
introspection output are diagnostic detail: `DEBUG` only.

### External-program output goes to a sidecar file

stdout/stderr of an external model or tool is redirected to a sidecar log file
(e.g. `fuse_distributed_run.log` in the run's output directory), and a single
log line references that path. Raw subprocess output is never streamed into
the SYMFLUENCE log at `INFO`.

### Calibration progress schema

All optimization algorithms report progress through
`EvaluationMetricsTracker.log_iteration_progress()`
(`src/symfluence/optimization/optimizers/metrics_tracker.py`), which emits one
fixed schema — every field always present:

```
{ALG} {i}/{max} {unit} ({pct}%) | Best: {score} | Improved: {a}/{b} | Crashes: {c}/{d} | Elapsed: {t}
```

- `unit` is one of `evals`, `gens`, `epochs`, `loops`.
- An optional `[P##]` worker tag prefixes the line when a parallel worker
  reports.
- Throttling is owned by the tracker, not the algorithm: a line is emitted
  every `log_interval` iterations (default 10), and the final iteration is
  always emitted. Algorithms must not add their own bespoke progress lines.

### Worker-subprocess logging

Spawned worker processes obtain their logger via
`symfluence.core.logging_utils.get_worker_logger(worker_id, individual_id)`,
which roots them under the `symfluence.worker.P##` hierarchy and bootstraps a
compact handler only when no configured root exists. Do not call
`logging.basicConfig()` in workers.

### Third-party suppression

Noisy third-party loggers are capped via `silence_third_party()`, which
applies the single central table `THIRD_PARTY_LOGGER_LEVELS` in
`core/logging_utils.py`. Add new noisy libraries to that table; do not
scatter `logging.getLogger('lib').setLevel(...)` calls.

### Completion honesty

Step-completion lines — `✓ Completed` / `✗ Failed` — must reflect the actual
step outcome. A step that failed, or whose command exited non-zero, must not
log a success marker. Run summaries (`run_summary_*.json`, schema version 2)
report error/warning totals counted from actual log records via
`CountingHandler`, not hard-coded zeros.

### File line format

The canonical log-file record format is defined once in `core/logging_utils.py`:

```
FILE_FORMAT = '%(asctime)s %(levelname)-7s [%(sname)s] %(message)s'
```

`%(sname)s` is the compact logger name (leading `symfluence.` stripped),
supplied by `ShortNameFilter`. Debug mode uses `FILE_FORMAT_DEBUG`, which adds
the emitting `module.funcName`. Log files are named
`symfluence_{domain}_{experiment_id}_{timestamp}.log` under
`_workLog_{domain}/`.

## Consequences

- Contributors have an explicit rule: a failed model run or data step is
  `ERROR`, not `CRITICAL`. `CRITICAL` is not "a bad ERROR"; it marks total
  abort.
- The current absence of `.critical()` calls is correct under this policy, not
  an oversight — the codebase's failure modes are designed to be recoverable
  and logged at `ERROR`.
- If a genuine whole-run abort boundary is added later (e.g. a top-level
  orchestrator catch that re-raises after cleanup), that is the appropriate and
  expected place for a single `CRITICAL` line.

## References

- CLAUDE.md — resilience logging convention (`# noqa: BLE001`)
- `--debug` / `--quiet` flags — three console states: quiet (WARNING+),
  normal (INFO+), debug (DEBUG+); the file log always captures everything
- `src/symfluence/core/logging_utils.py` — shared protocol implementation
  (`FILE_FORMAT`, `log_once`, `silence_third_party`, `get_worker_logger`)
- `src/symfluence/project/logging_manager.py` — idempotent setup,
  `CountingHandler`, run summaries
- CONTRIBUTING.md §5 — logging guidance for contributors
