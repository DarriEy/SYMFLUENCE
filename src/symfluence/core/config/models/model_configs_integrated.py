# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Deprecated shim: per-model config schemas moved to their model packages
(``symfluence.models.<model>.config_schema``), registered in
``R.config_schemas`` by each package. Names here resolve through the registry
(PEP 562) so core never imports the models layer.
"""
from __future__ import annotations

_EXPORTS = {'MODFLOWConfig': 'MODFLOW', 'ParFlowConfig': 'PARFLOW', 'CLMParFlowConfig': 'CLMPARFLOW', 'PIHMConfig': 'PIHM'}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    key = _EXPORTS.get(name)
    if key is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from symfluence.core.registries import R

    schema = R.config_schemas.get(key)
    if schema is None:
        raise AttributeError(
            f"{name} is unavailable: no config schema registered for model "
            f"'{key}' (is the model package installed and registered?)"
        )
    globals()[name] = schema
    return schema
