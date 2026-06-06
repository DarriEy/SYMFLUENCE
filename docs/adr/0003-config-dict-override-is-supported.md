# ADR-0003: `_config_dict_override` is a supported escape hatch

- **Status:** Accepted
- **Date:** 2026-06-05
- **Resolves:** Independent Architectural Review (2026-05-29), open question Q4
- **Related:** Review Q3 / Tier 3 item 21 (`extra='forbid'`, gated by this decision)

## Context

The `ConfigMixin` (`core/mixins/config.py`) provides a setter,
`_config_dict_override`, that lets code replace configuration values *after* the
configuration has been validated and frozen. The review asked whether this is
an intentional, supported hook or an accidental bypass of the "configuration is
immutable once validated" guarantee — and noted that if supported, it should be
documented as such; if not, it is worth closing.

On inspection the hook has real production callers that depend on it to inject
model-derived values that are only known after preprocessing:

- `models/fuse/subcatchment_processor.py`
- `models/summa/config_manager.py`

These are not tests reaching past the validation boundary; they are legitimate
model-specific steps that compute configuration from data (subcatchment
structure, SUMMA file manager paths) that cannot exist at initial validation
time.

## Decision

`_config_dict_override` is a **supported, intentional escape hatch**, not a bug
to be closed. It is the sanctioned mechanism for injecting values that are only
knowable after a model-specific preprocessing step has run.

Its contract:

- It is **internal API** (leading underscore): used by core and in-tree model
  code, not part of the plugin-facing surface.
- It applies *after* validation deliberately. Callers are responsible for the
  validity of values they inject; the immutability guarantee covers the
  *initial* validated tree, not post-preprocessing derived values.
- New uses should be rare and model-specific. Prefer expressing configuration
  in the YAML schema where the value is knowable up front.

## Consequences

- The "immutable once validated" property is scoped precisely: it describes the
  validated config tree as loaded, not a prohibition on derived-value injection
  during the pipeline.
- This decision **shapes the answer to Q3 (`extra='forbid'`)**. Because in-tree
  code and plugins rely on extra/derived keys flowing through the config object,
  the frozen base config cannot be tightened to reject unrecognized fields. Q3
  is therefore resolved *not* by `extra='forbid'` but by validating flat keys at
  ingestion (warn by default, strict opt-in) — see
  [ADR-0006](0006-config-unknown-keys-warn-by-default.md).

## References

- `core/mixins/config.py` — `_config_dict_override`
- Callers: `models/fuse/subcatchment_processor.py`, `models/summa/config_manager.py`
- Review Q3 / Tier 3 item 21 (gated by this decision)
