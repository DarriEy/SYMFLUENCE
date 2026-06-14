# SYMFLUENCE security posture and threat model

This document describes what SYMFLUENCE trusts, where the trust boundaries
are, and what protects each one. It complements
[`.github/SECURITY.md`](../.github/SECURITY.md), which covers how to report a
vulnerability privately.

## Deployment context

SYMFLUENCE is a research framework run by a single user on their own machine
or HPC account, against data directories they own. It is not a network
service: nothing listens on a socket by default except the optional web GUI,
which binds loopback only
([ADR-0007](adr/0007-gui-single-user-localhost.md)). Multi-tenant or hosted
deployment is out of scope and unsupported.

## The most important boundary: configuration is trusted input

A SYMFLUENCE configuration file controls which executables the framework
builds and runs and where it reads and writes data. **A configuration from an
untrusted source must be treated as code**, exactly like a Makefile or a CI
workflow file: review it before running it. The framework validates configs
for *correctness* (typed Pydantic tree, unknown-key warnings — see
[ADR-0006](adr/0006-config-unknown-keys-warn-by-default.md)), not for
*malice*; that is by design, because the user running a config and the author
of that config are normally the same person.

## Trust boundaries and mitigations

### 1. Configuration files (YAML)

- Every YAML parse site uses `yaml.safe_load`; `yaml.load` (which can
  instantiate arbitrary Python objects) does not appear in the tree.
- The dynamic-execution builtins `eval`, `exec`, and `os.system` do not appear
  anywhere in the source tree.
- Configs are validated into a typed Pydantic tree; unrecognized keys are
  warned about at ingestion with a "did you mean?" suggestion, with an
  opt-in strict mode (`STRICT_CONFIG` / `SYMFLUENCE_STRICT_CONFIG`) that the
  project's own shipped configs are held to in CI.

### 2. Downloaded scientific data

Forcing, attribute, and observation data are fetched from remote services
(ERA5, SoilGrids, USGS, …) over HTTPS through configured `requests.Session`
objects with retry logic. Credentials are read from environment variables
only; there are no hardcoded secrets, and the logging layer masks sensitive
fields.

Downloaded archives are extracted through the hardened helpers in
`core/archive_extraction.py` (path-traversal-safe tar and zip extraction);
no raw `extractall` call sites remain. Residual risk: scientific file formats
(netCDF/HDF5/GeoTIFF) are parsed by compiled third-party libraries, so a
malicious data file could target a parser vulnerability — mitigated by
pinning those libraries via lockfiles (below) and by the deployment context
(users fetch from the canonical scientific providers).

### 3. Compiled model engines

Hydrological model binaries (SUMMA, FUSE, NGEN, …) are built from pinned
upstream sources by the framework's binary install step and invoked as
subprocesses with framework-constructed argument lists. `shell=True` is
confined to a small number of audited sites where shell semantics are
genuinely required (notably HPC `module load`, which is a shell function);
those sites quote tokens and are documented inline. Model engines read and
write inside the user's domain directory; they run with the user's own
privileges, as any locally built scientific code does.

### 4. Machine-learning checkpoints

All `torch.load` call sites pass `weights_only=True`, closing the
arbitrary-object-deserialization path (CVE-2025-32434 lineage) for model
checkpoints. Treat third-party checkpoint files with the same caution as any
downloaded binary regardless.

### 5. The bundled AI agent

The agent (`agent/`) processes repository content, which makes prompt
injection part of its threat model: text in a repo could try to instruct the
agent to take outward-facing actions. Mitigations: pushing and opening pull
requests are **off by default** and require explicit caller opt-in
([ADR-0004](adr/0004-bundled-agent-human-in-the-loop.md)), and file reads are
checked against a deny-list (`.git/`, `.env`, key material) with
directory-prefix and glob matching. Anyone wiring the agent into automation
should keep a human review between agent output and anything that publishes.

### 6. Network surface

- The web GUI binds `127.0.0.1` by default; a non-loopback bind is an
  explicit, warned, unauthenticated opt-in and is not a supported deployment
  ([ADR-0007](adr/0007-gui-single-user-localhost.md)).
- No other component listens on the network. The Delft-FEWS adapter is
  file-based (PI XML exchange).

### 7. Supply chain

- `uv.lock` pins the full PyPI dependency tree with SHA-256 hashes;
  `pixi.lock` does the same for the conda ecosystem. A substituted upstream
  package fails at install time rather than being silently pulled in.
- Every source file carries an SPDX license header; `bandit` runs in
  pre-commit and CI alongside ruff/mypy, and a broad-exception guard keeps
  error handling auditable.

## Residual risks and non-goals

- **Untrusted configs are not sandboxed** (see above) — running one is
  equivalent to running a script.
- **Parser vulnerabilities in scientific libraries** are mitigated by
  pinning, not eliminated.
- **No SBOM or artifact-provenance attestation is published yet**; the
  lockfiles are the current supply-chain record. Publishing provenance for
  the prebuilt binary distributions is future work.
- **The GUI has no authentication** because shared deployment is out of
  scope; that boundary is revisited (with a superseding ADR) if the scope
  changes.

## Reporting

Found something? Please follow the private reporting process in
[`.github/SECURITY.md`](../.github/SECURITY.md) — do not open a public issue.
