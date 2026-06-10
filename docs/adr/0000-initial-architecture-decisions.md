# ADR-0000: Initial architecture decisions

- **Status:** Accepted (recorded retroactively)
- **Date:** 2026-06-10

## Context

SYMFLUENCE grew to roughly 1,100 source files before the project adopted an
ADR practice, so the decisions that shape the codebase were made — and proved
out — without being written down as decisions. This record captures the six
of them that everything else builds on, so they are legible to contributors
without reverse-engineering the source. Each is stated with its upside and
the cost it knowingly carries.

## Decisions

### 1. Typed configuration is the contract

A Pydantic model hierarchy under `core/config/models/` is the single
agreement between the user and the framework: a run is valid if and only if
its configuration validates. Typos and type mismatches are caught by the
validator before any code runs; the configuration is self-documenting through
its schema; the contract is testable in isolation. The cost is that the
schema must be kept open to plugins rather than hardcoded — addressed by
[ADR-0002](0002-plugins-may-ship-typed-config.md) — and that unknown keys
need their own validation layer
([ADR-0006](0006-config-unknown-keys-warn-by-default.md)).

### 2. The manager pattern orchestrates work

A base manager class and a set of domain-specific managers carry each
workflow step from configuration to result, keeping orchestration logic out
of the model adapters. Orchestration lives in one tier instead of being
duplicated across adapters, and a new model gets a working workflow as long
as it fits the existing manager contracts. The cost is that the base manager
is load-bearing for every model at once, so its contract changes demand test
coverage commensurate with that blast radius.

### 3. Mixins carry cross-cutting concerns

Logging, timing, validation, configuration access, path resolution, shapefile
handling, and file utilities are supplied through `core/mixins/` rather than
a deep inheritance tree. Each concern stays small and reusable, and the
framework avoids diamond-inheritance pathologies. The cost is a cognitive
surface of roughly three dozen mixin classes a contributor learns before
changing one safely; trimming that surface is future work, not a current
defect.

### 4. The component registry is the extensibility substrate

Models, optimizers, data handlers, and other components are looked up by name
through a single generic `Registry[T]` exposed via the `R` facade
(`core/registries.py`), and third-party code extends the framework by
registering against it through Python entry points
(`model_manifest()` being the model-facing form). The framework grows new
capabilities without `core/` changing, and the plugin contract is uniform
across component types. The contribution model is plugins-first
(GOVERNANCE.md §3): new model integrations default to independent packages,
including their typed configuration
([ADR-0002](0002-plugins-may-ship-typed-config.md)).

### 5. External model engines are compiled separately and run as subprocesses

SUMMA, FUSE, NGEN, MESH, and the other compiled engines are not Python; the
framework builds them through a binary install step and invokes them as
external processes. The canonical, peer-reviewed implementation of each
hydrological model is used directly, with no Python re-implementation that
would itself need re-validation against the original. The accepted costs are
that the compile step is build engineering in its own right (own CI
workflows, system-dependency registry, multi-platform packaging), and that
subprocess failures must surface clearly — a silent model crash is the
worst-case failure mode of this decision, which is why non-zero exits are
logged at WARNING/ERROR with the log path rather than swallowed.

### 6. Multiple installation methods are offered deliberately

The seven offered install commands collapse into four maintained families —
the pip family (pip, pipx, uv, uv tool), conda, npm prebuilt binaries, and a
source/bootstrap build — to meet users where they already are, on both HPC
systems and laptops. pip is the primary documented path. The cost is that
every family must keep working, which is why each is exercised by its own CI
workflow (`install-methods.yml`, `install-validate.yml`,
`install-validate-arm.yml`, `npm-multidistro.yml`) rather than validated by
hand.

## Consequences

- These six decisions are the baseline; later ADRs record refinements on top
  of them rather than re-deciding them.
- They are not immutable physics — any of them can be superseded by a future
  ADR — but a change to one of them is a major architectural event and should
  be treated as such.

## References

- `docs/source/architecture.rst` — the full architecture guide
- `ARCHITECTURE.md` — the at-a-glance summary at the repository root
- GOVERNANCE.md §3 (plugins-first), §4 (interface stewardship)
