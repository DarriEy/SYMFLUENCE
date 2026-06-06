# ADR-0006: Unknown configuration keys — warn by default, strict opt-in

- **Status:** Accepted
- **Date:** 2026-06-05
- **Resolves:** Independent Architectural Review (2026-05-29), open question Q3 / Tier 3 item 21
- **Related:** [ADR-0003](0003-config-dict-override-is-supported.md) (the `extra='allow'` constraint this works within)

## Context

SYMFLUENCE configs are flat, uppercase YAML (`DOMAIN_NAME: ...`). The review
asked whether the frozen base config should reject unrecognized fields: today it
is `extra='allow'`, so a typo such as `HYDROLOGICAL_MDOEL` is accepted rather
than flagged, which can let a setting silently take no effect.

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
  the 1.0 default.
- **Strict mode** — raise `ConfigValidationError` instead, enabled per-config
  via a `STRICT_CONFIG` key or globally via `SYMFLUENCE_STRICT_CONFIG`.
- **Escape hatch** — genuinely freeform keys are listed under
  `ALLOW_UNKNOWN_KEYS` to suppress the warning/error without registering a
  schema.

This is already implemented in `core/config/key_validation.py` and wired into
the load path at `core/config/factories.from_file_factory` (warns by default,
escalates under strict). The 1.0 decision is to **keep warn-by-default** rather
than make strict the global default, because strict-by-default could break
configs and plugins in the wild that currently rely on silent extras.

## Consequences

- The review's concern (silent typos) is addressed: a misspelled key is now
  surfaced with a suggestion at load time, at the correct (flat) layer, without
  breaking plugins or the `_config_dict_override` hook.
- `extra='forbid'` is **explicitly not adopted** for the base config. Tier 3
  item 21 is resolved by the ingestion validator, not by tightening Pydantic.
- Strict mode is available for users who want unknown keys to be fatal; project
  example configs and CI should run with strict enabled so the framework's own
  configs stay clean. (Follow-on: turn strict on in CI / shipped examples.)

## References

- `core/config/key_validation.py`; caller `core/config/factories.py:from_file_factory`
- `tests/unit/config/test_key_validation.py`
- Review Q3 / Tier 3 item 21
