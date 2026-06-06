# ADR-0002: Plugins may ship their own typed configuration schema

- **Status:** Accepted
- **Date:** 2026-06-05
- **Resolves:** Independent Architectural Review (2026-05-29), open question Q2
- **Related:** PR #139 (registry migration, Phase B), Tier 3 item 18

## Context

At the time of the review, runners and post-processors were pluggable but typed
configuration was not. The top-level configuration model named only the models
the framework already knew about, and `R.config_schemas` — where a plugin would
register a typed Pydantic schema — was not read by the core validation
pipeline. A third-party plugin could therefore register a runner but could not
supply a typed config schema that participated in the validated config tree; it
had to fall back on untyped pass-through keys.

The review asked: is that closed shape the intended 1.0 behavior, or is opening
it planned before release (Tier 3 item 18)? Either is workable, but if 1.0
ships with the closed list, the plugin documentation must say so plainly.

## Decision

Typed plugin configuration is **opened before 1.0**. A plugin may register its
own typed configuration schema and have it participate in the validated config
tree on equal footing with in-tree models.

This is already implemented in PR #139 (Phase B). A plugin declares its schema
through `model_manifest(config_schema=...)`; the schema is registered into
`R.config_schemas`, and the core resolution path
(`models/config_resolution.get_config_schema`) reads it during validation. The
top-level `ModelConfig` no longer carries a hardcoded `Optional[*Config]` field
per model — model-specific configuration is resolved generically from the
registry and re-flattened on serialization so that stage-marker hashing stays
byte-stable.

## Consequences

- The plugin contract at 1.0 is: **runners, post-processors, and typed
  configuration schemas are all pluggable.** The asymmetry the review noted is
  closed; plugin documentation should describe typed config as supported.
- Adding a new in-tree model no longer requires editing the top-level config
  model — it requires registering a schema, the same path a plugin uses. This
  removes a class of "registered but unvalidated" drift.
- `R.config_schemas` is now a consumed registry. A CI guard
  (`tests/unit/config/test_config_schema_consumed.py`) verifies registered
  schemas are actually read, so the path cannot silently regress to the closed
  shape.

## References

- PR #139 — registry migration Phase B (typed-config plumbing)
- `models/config_resolution.py` — `get_config_schema`
- Review item 18 (typed-config plugin path), item 26 (consumed-schema guard)
- GOVERNANCE.md §3 (Plugins First), §4.1 (configuration schema keys)
