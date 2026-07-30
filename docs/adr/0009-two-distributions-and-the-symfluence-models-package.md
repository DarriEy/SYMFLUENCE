# ADR-0009: Two distributions, and the models package is top-level `symfluence_models`

- **Status:** Accepted
- **Date:** 2026-07-27 (recorded 2026-07-29)

## Context

The service-decomposition campaign splits the model adapters out of this
repository so they can version and release independently of the framework. Two
questions had to be settled before any file moved, and both had been decided in
discussion without being written down — which is what this ADR corrects.

**How many distributions?** The obvious decomposition is four (`symfluence-core`,
`symfluence-capability`, `symfluence-interfaces`, `symfluence-models`), mirroring
the four-tier architecture. But the tiers are enforced already, by
`scripts/check_core_layering.py` and the registry contracts; splitting them into
separate distributions buys no additional enforcement and multiplies the release
matrix, the cross-version compatibility surface, and the number of ways a user
can assemble an installation that has never been tested.

**What does the models distribution import as?** The natural answer is to keep
`symfluence.models`, so nothing downstream changes. That does not work, for two
independent reasons:

1. `src/symfluence/__init__.py` does substantial work at import time — it
   registers conda's `Library\bin` through `os.add_dll_directory()` on Windows,
   defaults `JAX_ENABLE_X64`, sets `GDAL_DATA`/`PROJ_DATA`, pins
   `KMP_DUPLICATE_LIB_OK`, adds torch's lib directory, configures HDF5 safety,
   and re-exports `SymfluenceConfig`. A PEP 420 namespace package must have **no**
   `__init__.py` at all. Deleting this one is not an option: it is what makes
   SYMFLUENCE importable on a conda Windows install.
2. Even setting that aside, the development layout breaks it. Under
   `pip install -e .` the framework imports from `src/symfluence/`, while a second
   distribution installing into `site-packages/symfluence/models/` lands in a
   *different directory*. There is no merge; the import simply fails. The failure
   appears only in editable installs — that is, only for developers — which is the
   worst possible place for it.

Every plugin already in the ecosystem resolves this the same way: `jhbv`,
`droute`, and the community data services (`cfs`, `cas`, `csfs`, `cos`) each ship
their own top-level package name and register through the registry rather than by
living inside the `symfluence` namespace.

## Decision

**Two distributions, not four.**

- `symfluence` remains the framework: core, capability and interface tiers
  together. Core is **not** split out in this campaign.
- `symfluence-models` ships the model adapters and pins `symfluence>=1.x,<2`.

**The models distribution ships a top-level `symfluence_models` package**, not
`symfluence.models`.

- `symfluence.models` becomes a lazy alias that forwards to `symfluence_models`,
  so existing imports and configs keep working.
- The alias is removed at 2.0.

## Consequences

- The editable-install failure above cannot happen: two distributions, two
  top-level names, no namespace merge to get wrong.
- `symfluence/__init__.py` keeps doing its platform setup, which every install on
  conda Windows depends on.
- Downstream code importing `symfluence.models.summa...` keeps working until 2.0
  via the alias, so the extraction is not a breaking change on its own schedule.
- The release matrix stays at two artifacts. The cost is that core cannot be
  installed without the capability and interface tiers; nobody has asked for
  that, and it can be revisited without undoing this decision.
- Consistency with `jhbv`, `droute` and the community services: every extension
  is a top-level distribution registering through `R`, with no special case for
  the first-party models.
- The framework must remain fully functional with the models distribution absent.
  That is enforced twice: `tests/conformance/test_models_absent.py` simulates
  absence in-tree on every platform, and the `Models-Absent Contract (stripped
  wheel)` CI job installs the wheel, physically deletes `symfluence/models`, and
  runs the framework against what is left.

## References

- `src/symfluence/__init__.py` — the import-time platform setup that rules out a
  PEP 420 namespace package
- `scripts/check_core_layering.py` — the tier enforcement that makes a four-way
  distribution split unnecessary
- `scripts/check_models_absent.py`, `tests/conformance/test_models_absent.py` —
  the models-absent contract
- `src/symfluence/testing/` — the public test-support surface the extracted
  repository tests against (ADR-0008 governs its coverage)
- ADR-0008 — coverage policy, including the per-package gates now covering
  `core/calibration` and `core/modeling`, the extraction contract surface
