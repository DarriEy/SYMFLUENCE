# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tiered validation of unrecognized flat configuration keys.

SYMFLUENCE configs are flat, uppercase YAML (``DOMAIN_NAME: ...``). The base
config models keep ``extra='allow'`` so that plugins can inject their own keys,
which means a typo such as ``HYDROLOGICAL_MDOEL`` is otherwise accepted
silently (RTI architectural review, open question 3 / Tier 3 item 21).

Rather than flipping ``extra='forbid'`` on the Pydantic models — which would
also reject legitimate plugin keys and acts at the wrong (post flat->nested)
layer — this module validates raw flat keys at ingestion against an allowlist
that already unions:

* every core config alias,
* every registered plugin schema (``R.config_schemas``), and
* every legacy alias.

The *response* is tiered, not the allowlist:

* **warn by default** — log a warning naming each unknown key with a
  "did you mean?" suggestion, so existing configs keep loading unchanged;
* **strict mode** — raise :class:`ConfigValidationError` instead. Strict is
  enabled per-config via a ``STRICT_CONFIG`` key, or globally via the existing
  ``SYMFLUENCE_STRICT_CONFIG`` environment variable (shared with
  :mod:`symfluence.core.config.coercion`).

Escape hatch: list genuinely freeform keys under ``ALLOW_UNKNOWN_KEYS`` to
suppress the warning/error for those keys without registering a schema.
"""
from __future__ import annotations

import logging
import os
from difflib import get_close_matches
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Control keys consumed by this validator itself — never reported as unknown.
RESERVED_CONTROL_KEYS = frozenset({'STRICT_CONFIG', 'ALLOW_UNKNOWN_KEYS'})

# Truthy spellings shared with coercion's SYMFLUENCE_STRICT_CONFIG handling.
_TRUTHY = ('true', '1', 'yes')


def _env_strict() -> bool:
    """Return True if strict config mode is requested via the environment."""
    return os.environ.get('SYMFLUENCE_STRICT_CONFIG', '').strip().lower() in _TRUTHY


def is_strict_config_mode(flat_config: Optional[Dict[str, Any]] = None) -> bool:
    """Resolve whether unknown keys should be a hard error.

    Precedence: an explicit per-config ``STRICT_CONFIG`` key (truthy) wins over
    the ``SYMFLUENCE_STRICT_CONFIG`` environment variable, which in turn wins
    over the default (warn-only).
    """
    if flat_config:
        for key in ('STRICT_CONFIG', 'strict_config'):
            if key in flat_config:
                value = flat_config[key]
                if isinstance(value, str):
                    return value.strip().lower() in _TRUTHY
                return bool(value)
    return _env_strict()


def _allowed_unknown(flat_config: Dict[str, Any]) -> Set[str]:
    """Extract the user-declared ``ALLOW_UNKNOWN_KEYS`` allowlist (uppercased)."""
    raw = flat_config.get('ALLOW_UNKNOWN_KEYS') or flat_config.get('allow_unknown_keys')
    if not raw:
        return set()
    if isinstance(raw, str):
        items: Iterable[str] = (part.strip() for part in raw.split(','))
    elif isinstance(raw, (list, tuple, set)):
        items = (str(part).strip() for part in raw)
    else:
        return set()
    return {item.upper() for item in items if item}


def find_unknown_keys(
    flat_config: Dict[str, Any],
    known_keys: Iterable[str],
    *,
    allow_unknown: Iterable[str] = (),
) -> List[str]:
    """Return sorted flat keys not present in *known_keys*.

    Reserved control keys and any key listed in ``ALLOW_UNKNOWN_KEYS`` (merged
    with the explicit *allow_unknown* argument) are excluded. Comparison is
    case-insensitive on the canonical uppercase form.
    """
    known = {str(k).upper() for k in known_keys}
    allowed = {str(k).upper() for k in allow_unknown} | _allowed_unknown(flat_config)
    allowed |= RESERVED_CONTROL_KEYS

    unknown: Set[str] = set()
    for key in flat_config:
        if not isinstance(key, str):
            continue
        upper = key.upper()
        if upper in known or upper in allowed:
            continue
        unknown.add(upper)
    return sorted(unknown)


def _format_message(unknown: List[str], known_keys: Set[str], source: Optional[str]) -> str:
    """Build a user-facing message with 'did you mean?' suggestions."""
    where = f" in {source}" if source else ""
    lines = [f"Unrecognized configuration key(s){where}:"]
    for key in unknown:
        matches = get_close_matches(key, known_keys, n=1, cutoff=0.6)
        if matches:
            lines.append(f"  - {key}  (did you mean '{matches[0]}'?)")
        else:
            lines.append(f"  - {key}")
    lines.append(
        "Fix the spelling, or list intentional custom keys under "
        "ALLOW_UNKNOWN_KEYS to silence this."
    )
    return "\n".join(lines)


def validate_known_flat_keys(
    flat_config: Dict[str, Any],
    known_keys: Iterable[str],
    *,
    strict: Optional[bool] = None,
    source: Optional[str] = None,
    allow_unknown: Iterable[str] = (),
) -> List[str]:
    """Validate that every flat key is recognized; warn or raise on unknowns.

    Args:
        flat_config: Flat (uppercase) user configuration dictionary.
        known_keys: The allowlist (core ∪ plugins ∪ legacy ∪ model transformers).
        strict: Force strict (raise) or lenient (warn). When ``None`` (default),
            resolved from the ``STRICT_CONFIG`` key / ``SYMFLUENCE_STRICT_CONFIG``
            environment variable via :func:`is_strict_config_mode`.
        source: Optional origin (e.g. file path) included in the message.
        allow_unknown: Additional keys to treat as recognized.

    Returns:
        The sorted list of unknown keys found (empty when the config is clean).

    Raises:
        ConfigValidationError: In strict mode when unknown keys are present.
    """
    known = {str(k).upper() for k in known_keys}
    unknown = find_unknown_keys(flat_config, known, allow_unknown=allow_unknown)
    if not unknown:
        return []

    message = _format_message(unknown, known, source)

    if strict is None:
        strict = is_strict_config_mode(flat_config)

    if strict:
        from symfluence.core.exceptions import ConfigValidationError
        raise ConfigValidationError(message)

    logger.warning(message)
    return unknown
