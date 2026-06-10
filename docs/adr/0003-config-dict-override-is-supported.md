# ADR-0003: `_config_dict_override` is a supported escape hatch

- **Status:** Accepted
- **Date:** 2026-06-05
- **Related:** [ADR-0006](0006-config-unknown-keys-warn-by-default.md) (works within the same `extra='allow'` constraint)

## Context

The `ConfigMixin` (`core/mixins/config.py`) provides a setter,
`_config_dict_override`, that lets code replace configuration values *after*
the configuration has been validated and frozen. On its face this looks like
an accidental bypass of the "configuration is immutable once validated"
guarantee, and the question arose whether it should be closed.

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

Its contract — and the limits of the commitment this ADR makes:

- It is **internal API** (leading underscore): used by core and in-tree model
  code, not part of the plugin-facing surface. It may be renamed, narrowed, or
  replaced by a structured mechanism in a future release without a deprecation
  cycle, provided the in-tree callers move with it.
- It applies *after* validation deliberately. Callers are responsible for the
  validity of values they inject; the immutability guarantee covers the
  *initial* validated tree, not post-preprocessing derived values.
- New uses should be rare and model-specific. Prefer expressing configuration
  in the YAML schema where the value is knowable up front. A growing caller
  count is the signal to revisit this ADR and design a first-class
  derived-values mechanism.

## Consequences

- The "immutable once validated" property is scoped precisely: it describes the
  validated config tree as loaded, not a prohibition on derived-value injection
  during the pipeline.
- Because in-tree code and plugins rely on extra/derived keys flowing through
  the config object, the frozen base config cannot be tightened to reject
  unrecognized fields wholesale. Unknown-key safety is therefore provided by
  ingestion-time validation instead — see
  [ADR-0006](0006-config-unknown-keys-warn-by-default.md).

## References

- `core/mixins/config.py` — `_config_dict_override`
- Callers: `models/fuse/subcatchment_processor.py`, `models/summa/config_manager.py`
