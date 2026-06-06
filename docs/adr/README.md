# Architecture Decision Records

This directory records significant architecture and policy decisions for
SYMFLUENCE as short, dated documents. An ADR captures the *context* a decision
was made in, the *decision* itself, and its *consequences*, so that a choice
which is easy to make once and easy to forget later remains legible to future
contributors.

ADRs are immutable once accepted. A decision is changed not by editing its ADR
but by adding a new ADR that supersedes it; the old record stays in place with
its status updated to `Superseded by ADR-XXXX`. This keeps the history of *why*
intact rather than overwriting it.

## When to write one

Write an ADR when a decision (a) affects a stable interface or public contract
(see [GOVERNANCE.md](../../GOVERNANCE.md) §4), (b) sets a project-wide policy a
contributor could otherwise reasonably guess wrong, or (c) resolves a question
that has surfaced more than once. Routine bug fixes and local implementation
choices do not need one.

## Format

Each ADR follows a light [MADR](https://adr.github.io/madr/)-style template:
a status line, **Context**, **Decision**, **Consequences**, and **References**.
Keep them short — an ADR is a record, not a design document.

## Index

| ADR | Title | Status | Resolves |
|-----|-------|--------|----------|
| [0001](0001-remove-legacy-registry-shims.md) | Legacy registry shim classes are removed before 1.0 | Accepted | Review Q1 |
| [0002](0002-plugins-may-ship-typed-config.md) | Plugins may ship their own typed configuration schema | Accepted | Review Q2 |
| [0003](0003-config-dict-override-is-supported.md) | `_config_dict_override` is a supported escape hatch | Accepted | Review Q4 |
| [0004](0004-bundled-agent-human-in-the-loop.md) | The bundled AI agent does not push without a human | Accepted | Review Q6 |
| [0005](0005-logging-level-policy.md) | Logging-level policy: ERROR is the operational ceiling | Accepted | Review Q9 |
| [0006](0006-config-unknown-keys-warn-by-default.md) | Unknown config keys: warn by default, strict opt-in | Accepted | Review Q3 |
| [0007](0007-gui-single-user-localhost.md) | The web GUI is a single-user localhost tool | Accepted | Review Q5 |
| [0008](0008-coverage-gate-raise-and-ratchet.md) | Coverage policy: raise the global gate and add a ratchet | Accepted (impl. follow-on) | Review Q7 |

The "Resolves" column references the open maintainer questions in the RTI
*Independent Architectural Review* (2026-05-29). With ADR-0001 through 0008, all
ten of that review's open maintainer questions have a recorded decision (Q8 and
Q10 were resolved directly in earlier work — pip-primary install tiering and the
tree-wide `from __future__ import annotations` rollout respectively).
