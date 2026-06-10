# ADR-0006: Unknown configuration keys — warn by default, strict opt-in

- **Status:** Accepted
- **Date:** 2026-06-05 (status updated 2026-06-10)
- **Related:** [ADR-0003](0003-config-dict-override-is-supported.md) (the `extra='allow'` constraint this works within)

## Context

SYMFLUENCE configs are flat, uppercase YAML (`DOMAIN_NAME: ...`). The base
configuration models accept unrecognized fields (`extra='allow'`), so a typo
such as `HYDROLOGICAL_MDOEL` is accepted rather than flagged, which can let a
setting silently take no effect — the most common class of configuration
mistake a user encounters.

The obvious fix — flipping the Pydantic models to `extra='forbid'` — is the
wrong tool here for two reasons:

1. It acts at the **post flat→nested layer**, after key transformation, so it
   reports confusing internal names rather than the user's flat key.
2. It would reject **legitimate plugin keys** and the post-validation
   `_config_dict_override` path ([ADR-0003](0003-config-dict-override-is-supported.md)),
   both of which deliberately rely on extra keys flowing through.

## Decision

Keep the base config `extra='allow'`, and instead **validate raw flat keys at
ingestion** against an allowlist that unions every core alias, every registered
plugin schema (`R.config_schemas`), and every legacy alias. The *response* is
tiered, not the allowlist:

- **Warn by default** — each unknown key is logged with a "did you mean?"
  suggestion (`difflib`), so existing configs keep loading unchanged. This is
  the 1.0 default for user configs.
- **Strict mode** — raise `ConfigValidationError` instead, enabled per-config
  via a `STRICT_CONFIG` key or globally via `SYMFLUENCE_STRICT_CONFIG`.
- **Escape hatch** — genuinely freeform keys are listed under
  `ALLOW_UNKNOWN_KEYS` to suppress the warning/error without registering a
  schema.

This is implemented in `core/config/key_validation.py` and wired into the load
path at `core/config/factories.from_file_factory` (warns by default, escalates
under strict). The decision for 1.0 is to **keep warn-by-default for user
configs** rather than make strict the global default, because strict-by-default
could break configs and plugins in the wild that currently rely on silent
extras.

## Consequences

- The silent-typo concern is addressed: a misspelled key is surfaced with a
  suggestion at load time, at the correct (flat) layer, without breaking
  plugins or the `_config_dict_override` hook.
- `extra='forbid'` is **explicitly not adopted** for the base config; the
  ingestion validator is the mechanism.
- One known limitation: the validator runs on **flat** configs only — nested
  configs bypass it (`from_file_factory` validates in the flat branch). Nested
  configs get their safety from the typed tree itself; extending key validation
  to the nested path is possible future work.
- **The project dogfoods strict mode on its own shipped configs.** After an
  allowlist-completion pass (125 directly-read flat keys registered as
  `RECOGNIZED_FLAT_KEYS`; the widespread legacy spelling `TARGET_METRIC`
  normalized to `OPTIMIZATION_METRIC`; plugin-adapter keys bridged via the
  `CONFIG_PREFIX` fallback; verified-dead keys stripped from shipped templates
  and examples), strict validation is **enforced in CI and pre-commit** for the
  maintained shipped configs (`scripts/check_shipped_configs_strict.py`,
  `tests/unit/config/test_shipped_configs_strict.py`). Kitchen-sink
  documentation templates and the research-archive example configs are
  explicitly exempted.
- Whether `STRICT_CONFIG` becomes the **user-facing default** in a later
  release is an open follow-on; flipping it is a behavior change for existing
  user configs and warrants its own (superseding) ADR.

## References

- `core/config/key_validation.py`; caller `core/config/factories.py:from_file_factory`
- `core/config/legacy_aliases.py` — `RECOGNIZED_FLAT_KEYS`, `NORMALIZATION_ALIASES`
- `scripts/check_shipped_configs_strict.py`; `tests/unit/config/test_shipped_configs_strict.py`
- `tests/unit/config/test_key_validation.py`
