# Release security policy

Release workflows are part of SYMFLUENCE's trusted computing base.

- GitHub environments used for PyPI, npm, and binary publication must require
  approval from a maintainer who did not author the release change.
- Release tags must be protected against deletion and force updates.
- Workflow and release-script changes require CODEOWNER review.
- Third-party actions are pinned to full commit hashes.
- PyPI uses trusted publishing with attestations; long-lived upload tokens are
  not permitted.
- External model sources used for a formal release must set
  `SYMFLUENCE_REQUIRE_IMMUTABLE_SOURCES=1` and provide full 40-character commit
  hashes. A moving branch or abbreviated hash is not a reproducible source.
- `toolchain.json`, release SBOMs, source commits, compiler versions, and
  artifact SHA-256 digests are retained with the release.

Repository administrators are responsible for configuring the protected-tag,
environment-approval, two-person-review, and immutable-release settings in
GitHub. The checked-in governance guard verifies the portions expressible in
repository files.
