# ADR-0009: Per-family contract versioning for core extensibility surfaces

Date: 2026-07-24
Status: Accepted

## Context

The service decomposition extracts the model suite into its own package
(`symfluence-models`) and anticipates further extractions (a community
domain/geofabric service) and third-party contributions (optimization
algorithms, metrics, model adapters). The acquisition-backend contract
(`symfluence.data.backends.contract.PROTOCOL_VERSION` + `is_compatible`,
used by the CFS/CAS/CSFS/COS community services) has proven the release
mechanics: external packages release independently against uncapped pip
requirements, and version skew is declined cleanly at registration.

The surfaces external packages build against do not evolve in lockstep:
the calibration engine changes at a different cadence than the metric
facade or the geometry utilities. A single framework-wide contract version
would force every family to share breaking boundaries, and retrofitting
family granularity after packages depend on a monolithic version would
itself be a breaking change — so the granularity must be right from the
first stamped version.

## Decision

`symfluence.core.contracts` declares an independent semantic version per
contract family — `models`, `calibration`, `metrics`, `geospatial-utils` —
each starting at 0.1.0, with `contract_version()`, `is_compatible()` and
`assert_compatible()` helpers. Compatibility semantics are identical to the
acquisition contract: same major required; pre-1.0, minors are additive-only,
older-or-equal-minor targets are accepted, forward skew is declined.

The acquisition family keeps its existing constant and import path (external
services already depend on it); `contract_version("acquisition")` surfaces it
for a uniform view.

External packages declare the family versions they target and call
`assert_compatible()` in their `register()`. At extraction time,
`symfluence-models`' pip caps follow the same convention the community
services use (`>=x,<next-breaking-boundary`).

## Consequences

- Families gain capability (minor bumps) without forcing releases of
  packages built on other families.
- Each family's surface is enumerated in the module docstring; changing a
  surface without bumping its version is the failure mode to guard in review.
- The `models` family version becomes the anchor for the `symfluence-models`
  extraction (phase 2); `geospatial-utils` starts minimal
  (`core.geometry_utils`) and grows additively if model packages need more.
