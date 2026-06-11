# SYMFLUENCE Architecture — at a glance

SYMFLUENCE is a framework for reproducible multi-model hydrological science:
define a study area once, run many hydrological models against it, calibrate
them, and compare the results through one consistent workflow. This document
is the one-page orientation; the full architecture guide lives in the Sphinx
documentation ([`docs/source/architecture.rst`](docs/source/architecture.rst),
rendered on the docs site), and the decisions behind the design are recorded
as ADRs under [`docs/adr/`](docs/adr/).

## The workflow

Everything is organized around a single end-to-end modeling workflow, driven
by one typed YAML configuration file:

1. **Domain setup** — define the geographic area and discretize it into
   modeling units (GRUs/HRUs).
2. **Data preprocessing** — acquire and prepare meteorological forcing and
   physical attributes.
3. **Model instantiation** — configure and stage one or more hydrological
   models against the domain.
4. **Calibration and optimization** — tune parameters against observations
   (DDS, PSO, SCE-UA, DE, …).
5. **Evaluation** — score model output against observations and compare
   models.

The pipeline executes named steps tracked by stage markers, so completed steps
are skipped on re-run; the `symfluence workflow` CLI drives it.

## The layered structure

The source tree under `src/symfluence/` divides into three layers:

- **`core/`** is the foundation: the Pydantic configuration models, the
  component registry, the path resolver, shared mixins, the exception
  hierarchy, and profiling hooks. Everything depends on it; **it depends on
  nothing above it.** That rule is enforced, not aspirational — see
  `scripts/check_core_layering.py` (CI + pre-commit).
- **Capability packages** implement the workflow steps: `geospatial/` (domain
  setup), `data/` (acquisition and preprocessing), `models/` (one adapter
  package per hydrological model), `optimization/` (calibration algorithms),
  and `evaluation/` (scoring and benchmarking). `project/` orchestrates a full
  run across them.
- **Interface packages** are the ways a user drives the framework: `cli/`,
  `gui/` (Panel web UI), `tui/` (terminal UI), `agent/` (bundled AI
  assistant), `reporting/`, `coupling/`, and `fews/` (Delft-FEWS integration).

## The six decisions everything builds on

Recorded in full, with their costs, in
[ADR-0000](docs/adr/0000-initial-architecture-decisions.md):

1. **Typed configuration is the contract.** A run is valid iff its YAML
   validates against the Pydantic tree. Unknown keys are warned about at
   ingestion ([ADR-0006](docs/adr/0006-config-unknown-keys-warn-by-default.md)).
2. **The manager pattern orchestrates work** — orchestration lives in
   managers, not in model adapters.
3. **Mixins carry cross-cutting concerns** (logging, config access, path
   resolution) instead of a deep inheritance tree.
4. **The component registry is the extensibility substrate.** Components are
   looked up by name through the `R` facade; third-party packages register via
   entry points and `model_manifest()` — including their typed configuration
   ([ADR-0002](docs/adr/0002-plugins-may-ship-typed-config.md)). Contribution
   is plugins-first (GOVERNANCE.md §3).
5. **External model engines are compiled separately and run as
   subprocesses.** The canonical SUMMA/FUSE/NGEN/… binaries are used directly;
   no Python re-implementations.
6. **Multiple installation methods are deliberate** — four maintained
   families (pip, conda, npm prebuilt, source/bootstrap), each exercised by
   its own CI workflow; pip is the primary documented path.

## Extending the framework

New models, data handlers, and optimizers are independent Python packages that
register through entry points — no edits to `core/`. Start with
[GOVERNANCE.md](GOVERNANCE.md) §3 (the contribution model), the developer
guide in the Sphinx docs, and the JAX model packages (jHBV, jSACSMA, …) as
reference plugin implementations.

## Related documents

| Document | What it covers |
|----------|----------------|
| [`docs/source/architecture.rst`](docs/source/architecture.rst) | Full architecture guide (diagrams, data flow, patterns) |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records |
| [`docs/security.md`](docs/security.md) | Threat model and security posture |
| [`GOVERNANCE.md`](GOVERNANCE.md) | Contribution model, interface stewardship, decision-making |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Practical contributor guide |
| [`.github/SECURITY.md`](.github/SECURITY.md) | How to report a vulnerability |
