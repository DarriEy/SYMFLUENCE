# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Registry-derived model choices for GUI selectors."""
from __future__ import annotations

# Classification only (not an enumeration): registered models matching these
# sets are offered as routing / fire / groundwater choices; every other
# registered model with a config schema is offered as a hydrological model.
# New (external) hydrological models therefore appear automatically.
_ROUTING_KEYS = frozenset({'MIZUROUTE', 'DROUTE', 'TROUTE'})
_NON_HYDRO_KEYS = _ROUTING_KEYS | frozenset({'WMFIRE', 'IGNACIO', 'MODFLOW'})

_FALLBACK_HYDRO = ['SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH', 'RHESSYS', 'NGEN', 'LSTM']
_FALLBACK_ROUTING = ['None', 'MIZUROUTE', 'DROUTE', 'TROUTE']


def hydro_model_choices() -> list[str]:
    """Registered hydrological models (canonical keys), registry-derived."""
    try:
        from symfluence.core.registries import R

        keys = sorted(set(R.config_schemas.keys()) - _NON_HYDRO_KEYS)
        return keys or list(_FALLBACK_HYDRO)
    except Exception:  # noqa: BLE001 — GUI must render even if registry is unavailable
        return list(_FALLBACK_HYDRO)


def routing_model_choices() -> list[str]:
    """Registered routing models, 'None' first, registry-derived."""
    try:
        from symfluence.core.registries import R

        keys = sorted(set(R.config_schemas.keys()) & _ROUTING_KEYS)
        return ['None'] + keys if keys else list(_FALLBACK_ROUTING)
    except Exception:  # noqa: BLE001 — GUI must render even if registry is unavailable
        return list(_FALLBACK_ROUTING)
