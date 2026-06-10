# ADR-0008: Test-coverage policy — ratcheted global gate plus a core-spine gate

- **Status:** Accepted
- **Date:** 2026-06-05 (updated 2026-06-10: core-spine gate landed)

## Context

The project enforces a flat coverage floor in CI. A single project-wide number
lets strong coverage in one area average out weak coverage elsewhere, which can
hide a regression in the part of the codebase that matters most (`core/`). The
floor also sat at 15% while measured unit-test coverage was ~29%, so the gate
was not doing real work.

Roughly 700 modules (model adapters, data handlers, geospatial processors)
need external model binaries or live services to exercise and cannot run in CI
at all, which is why a project-wide number is structurally low and why the
testable core deserves its own bar.

## Decision

Coverage is enforced by **two gates, both one-way ratchets by policy** (raise
as coverage improves, never lower):

1. **Global gate:** `--cov-fail-under` / `fail_under` raised from 15 to **25**
   in `ci.yml`, `cross-platform.yml`, and `pyproject.toml` together, set ~4
   points below measured reality (~29%) so the gate binds without flaking on
   parallel-run variance.
2. **Core-spine gate:** the CI-testable core (config parsing, registry, path
   resolution, mixins) gets its own stricter bar — `coverage report
   --include='src/symfluence/core/*' --fail-under=80`, with
   instrumentation-only modules (profiling, npm packaging) omitted. This is
   the per-package target that prevents weak coverage elsewhere from being
   masked by, or masking, the core.

Whether to additionally *enforce* the ratchet mechanically (a CI step diffing
`coverage.json` against a committed baseline) remains deferred; the manual
floors are the 1.0 mechanism.

## Consequences

- Contributions cannot lower either floor, which is the property that matters
  (preventing silent regression), and the part of the codebase every model
  depends on is held to a meaningfully higher bar than the binary-gated long
  tail.
- Both gates start at today's reality, not an aspiration, and are tightened
  over time as coverage grows. Raising a floor is a routine change; lowering
  one requires revisiting this ADR.
- Extending per-package gates beyond `core/` (e.g. `optimization/`,
  `data/`) is available follow-on work using the same pattern.

## References

- `pyproject.toml` (`fail_under = 25`)
- `.github/workflows/ci.yml`, `.github/workflows/cross-platform.yml` —
  `--cov-fail-under=25` and the "Core coverage gate" step (`--fail-under=80`)
