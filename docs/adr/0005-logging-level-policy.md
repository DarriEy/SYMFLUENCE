# ADR-0005: Logging-level policy — ERROR is the operational ceiling

- **Status:** Accepted
- **Date:** 2026-06-05
- **Resolves:** Independent Architectural Review (2026-05-29), open question Q9

## Context

Across the source tree, nothing is logged at the `CRITICAL` level. The review
asked whether this is deliberate — the project has no events it considers
critical — or whether the level has simply not been adopted. Either position is
fine; what is missing is a written policy so future contributors know whether to
reach for the level.

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

- Review Q9 (CRITICAL-level logging)
- CLAUDE.md — resilience logging convention (`# noqa: BLE001`)
- `--debug` flag (Tier 1 items 4/5; raises surfaced log level)
