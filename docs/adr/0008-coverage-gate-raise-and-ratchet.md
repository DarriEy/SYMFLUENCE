# ADR-0008: Test-coverage policy — raise the global gate and add a ratchet

- **Status:** Accepted — gate raised; automated ratchet + per-package deferred
- **Date:** 2026-06-05
- **Resolves:** Independent Architectural Review (2026-05-29), open question Q7 / Theme C

## Context

The project enforces a coverage floor of 15% (`--cov-fail-under=15` in
`ci.yml` and `cross-platform.yml`; `fail_under = 15` in `pyproject.toml`). The
review's characterization that the threshold is "not enforced" is stale — it is
enforced — but it is **flat and low**. A single project-wide number lets strong
coverage in one area average out weak coverage elsewhere, which can hide a
regression in the part of the codebase that matters most (`core/`).

The review suggested per-package targets (e.g. ~25-30% project-wide, ~80% on
`core/`). Per-package enforcement is the most faithful answer but the most CI
plumbing, and likely requires near-term test-writing to clear a high `core/`
bar.

## Decision

For 1.0, **raise the global gate and treat it as a one-way ratchet**, deferring
per-package targets to a later iteration:

- Measured actual unit-test coverage at decision time was **~29%** (line), well
  above both the 15% gate and a stale "~17%" note in `pyproject.toml`.
- **Done:** raised the flat `--cov-fail-under` / `fail_under` from 15 to **25**
  in `ci.yml`, `cross-platform.yml`, and `pyproject.toml` together (the
  manifest-consistency discipline from item 16 applies), leaving ~4 points of
  slack below the true 29% for parallel/flaky variance. The stale comment was
  corrected to ~29%.
- The floor is a **never-decrease ratchet by policy**: raise it as coverage
  improves, never lower it. Whether to additionally *enforce* the ratchet
  mechanically (a CI step diffing `coverage.json` against a committed baseline)
  is deferred; the manual floor is the 1.0 mechanism.
- Per-package targets (notably a high `core/` floor) are a **follow-on**, not a
  1.0 blocker.

## Consequences

- Coverage becomes a one-way ratchet: contributions cannot lower the floor,
  which is the property the review actually wants (preventing silent
  regression), without the upfront cost of clearing an 80% `core/` bar.
- The raise from 15 → 25 was validated against measured actual coverage (~29%)
  so the gate sits just below the present number — a ratchet starts at today's
  reality, not an aspiration — and is tightened over time as coverage grows.
- Remaining **follow-ons** (not 1.0 blockers): optional mechanical ratchet
  enforcement, and per-package targets — the latter, when adopted, will be a
  superseding ADR.

## References

- `pyproject.toml` (`fail_under`), `.github/workflows/ci.yml`,
  `.github/workflows/cross-platform.yml` (`--cov-fail-under`)
- Review Q7 / Theme C; manifest-consistency discipline from item 16
