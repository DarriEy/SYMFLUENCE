# ADR-0001: Legacy registry shim classes are removed before 1.0

- **Status:** Accepted
- **Date:** 2026-06-05
- **Related:** GOVERNANCE.md §4

## Context

SYMFLUENCE historically exposed per-domain registry classes
(`ModelRegistry`, `OptimizerRegistry`, `ComponentRegistry`, `ConfigRegistry`,
`ResultExtractorRegistry`, `PlotterRegistry`) and registration decorators such
as `@ModelRegistry.register(...)`. These were superseded by a single unified
registry facade, `R` (e.g. `R.runners.add(...)`, `R.config_schemas.get(...)`).

During the migration the old class names were briefly retained as thin
forwarding shims so that in-tree and external code could be ported
incrementally. Before 1.0 the project had to commit one way or the other: are
these shims removed in a named release, or kept indefinitely? The concern is
that "indefinitely" promotes the legacy class names to permanent public API,
after which semantic versioning forbids removing them without a major-version
break.

## Decision

The legacy forwarding shims are **removed before the 1.0 release**, not kept.
The unified `R` facade and the `model_manifest()` entry point are the only
supported registration surface at 1.0.

This decision is already implemented: the forwarding shim classes were deleted
in PR #138. Names that still end in `*Registry` in the source tree fall into
two categories that this decision does **not** remove:

1. **First-class domain registries** with their own purpose-built API
   (e.g. `ParameterBoundsRegistry`, `ObjectiveRegistry`, `ForcingAdapterRegistry`,
   the `data/` `BaseRegistry` hierarchy). These were never compatibility shims.
2. **Thin documented facades over `R`** that exist for ergonomics, not
   backward compatibility — e.g. `BuildInstructionsRegistry.register(...)` is
   the supported public spelling of `R.build_instructions.add(...)` and is used
   by infrastructure tooling. These forward to the unified registry by design
   and remain supported.

## Consequences

- The 1.0 public API does not include the legacy `*Registry.register`
  decorators. Plugins and core code register through `model_manifest()` and
  `R.*`. Removing the shims pre-1.0 keeps them out of the SemVer surface.
- Any remaining external code using the old decorators must migrate before
  upgrading to 1.0. The migration is mechanical (decorator rename); the JAX
  plugin packages were ported as reference examples.
- This is consistent with GOVERNANCE.md §4.2: a breaking interface change made
  *before* 1.0 is permitted under pre-1.0 SemVer without a deprecation cycle.

## References

- PR #138 — registry migration Phase A (shim deletion)
- GOVERNANCE.md §4 — Interface Stewardship
