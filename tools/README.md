# Tools

This folder hosts development tooling that is not required for normal
development or usage.

- `quality/`: data files for the repository quality gates (e.g. the
  broad-exception allowlist consumed by `scripts/check_broad_exceptions.py`).

The npm package used to ship prebuilt binaries lives at the repo root in
`npm/`; it is published by the Release Binaries workflow
(`.github/workflows/release-binaries.yml`).
