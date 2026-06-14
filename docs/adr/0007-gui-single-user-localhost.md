# ADR-0007: The web GUI is a single-user localhost tool

- **Status:** Accepted
- **Date:** 2026-06-05

## Context

SYMFLUENCE ships a Panel/Param web GUI (`gui/`). Its intended deployment
context was unstated: is it meant to run only on a single user's own machine
(localhost), or might it run on a shared HPC login node where other users are
present? The answer determines whether authentication is a 1.0 requirement or
a later enhancement.

The code already encodes a localhost-first stance: `gui/server.py` binds
`127.0.0.1` by default; binding to a non-loopback address (e.g. `0.0.0.0`) is an
explicit opt-in that emits a loud warning and restricts the websocket origin to
the bound host:port (`is_loopback_address` helper). There is no authentication
layer.

## Decision

The web GUI is a **single-user, localhost tool** at 1.0. This is the stated,
supported deployment context.

- The default and supported configuration is loopback (`127.0.0.1`).
- Binding to a non-loopback interface remains an **advanced, opt-in** action,
  unauthenticated and at the operator's own risk, surfaced by the existing
  warning. It is not a supported multi-tenant deployment.
- **Authentication is not a 1.0 requirement.** Multi-user / shared-node
  deployment with real auth is deferred past 1.0 (it would be net-new feature
  work, and the single-user model covers the framework's intended use).

The deferral is scoped, not permanent: if the GUI's role grows (e.g. a hosted
or shared deployment becomes a real use case), authentication becomes a
prerequisite for that work, recorded in a superseding ADR — not an optional
add-on.

## Consequences

- Documentation should state plainly that the GUI is for local single-user use
  and that exposing it on a network is unsupported and unauthenticated. Users
  who need shared access are responsible for fronting it with their own
  authenticating proxy.
- The loopback default plus opt-in warning is the enforced boundary; changes
  that weaken it (e.g. defaulting to `0.0.0.0`) are security regressions, not
  configuration choices.

## References

- `gui/server.py` — `127.0.0.1` default, opt-in non-loopback with warning
- `is_loopback_address` helper; `tests/unit/gui/test_server_bind.py`
- `docs/security.md` — network surface
