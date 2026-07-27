# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NextGen (NGEN) calibration parameter bounds -- owned by this package.

NGEN is the pure tier-A case of the service-decomposition bounds split: it has
**zero solo parameters**. Every one of the 82 names it calibrates is a shared
formulation reused elsewhere -- CFE and NOAH-OWP soil physics, the Penman-
Monteith PET block, and the TOPMODEL / SAC-SMA / Snow-17 sets that the external
``jtopmodel``, ``jsacsma`` and ``jsnow17`` plugins also serve. Those definitions
stay in ``symfluence.core.calibration.parameters.parameter_bounds_registry``
(tier C, single source of truth) and are composed here by name only.

What this package owns is the *composition*: which BMI modules an NGEN
realization stacks, and therefore which catalogue names each bound set
contains. That is model identity, and changing it must not need a ``core``
release.

Registering NGEN_TOPMODEL from here rather than delegating to
``get_topmodel_bounds()`` also removes core's dependence on the standalone
TOPMODEL bound set: the two happen to be identical today, and each is now free
to move on its own.

:func:`register_bounds` is called from this package's ``register()``, i.e. at
plugin-discovery time, which runs on ``import symfluence`` -- before any
calibration code can read bounds.
"""
from __future__ import annotations

from typing import Dict, List

from symfluence.core.calibration.parameters.parameter_bounds_registry import (
    register_model_bounds,
)

#: CFE (Conceptual Functional Equivalent) module.
CFE_SET: List[str] = [
    'maxsmc', 'wltsmc', 'satdk', 'satpsi', 'bb', 'mult', 'slop',
    'smcmax', 'alpha_fc', 'expon', 'K_lf', 'K_nash', 'Klf', 'Kn',
    'Cgw', 'max_gw_storage', 'refkdt', 'soil_depth',
]

#: NOAH-OWP-Modular land-surface module.
NOAH_SET: List[str] = [
    'slope', 'dksat', 'psisat', 'bexp', 'smcmax', 'smcwlt', 'smcref',
    'noah_refdk', 'noah_refkdt', 'noah_czil', 'noah_z0',
    'noah_frzk', 'noah_salp', 'rain_snow_thresh', 'ZREF', 'refkdt',
]

#: PET module. Both the BMI config key names and the legacy aliases are
#: calibratable, which is why several parameters appear twice under two names.
PET_SET: List[str] = [
    'vegetation_height_m', 'zero_plane_displacement_height_m',
    'momentum_transfer_roughness_length', 'heat_transfer_roughness_length_m',
    'surface_shortwave_albedo', 'surface_longwave_emissivity',
    'wind_speed_measurement_height_m', 'humidity_measurement_height_m',
    'pet_albedo', 'pet_z0_mom', 'pet_z0_heat', 'pet_veg_h', 'pet_d0',
]

#: TOPMODEL module. Catalogue entries are ``topmodel_``-namespaced because the
#: served name ``m`` collides with RHESSys'; the prefix is stripped on the way
#: out so keys match TOPMODEL's own config conventions.
TOPMODEL_SET: List[str] = [
    'topmodel_m', 'topmodel_lnTe', 'topmodel_Srmax', 'topmodel_Sr0', 'topmodel_td',
    'topmodel_k_route',
    'topmodel_DDF', 'topmodel_T_melt', 'topmodel_T_snow',
    'topmodel_ti_std', 'topmodel_S0',
]

#: SAC-SMA module: soil-moisture accounting only (Snow-17 is its own module).
SACSMA_SET: List[str] = [
    'UZTWM', 'UZFWM', 'UZK', 'LZTWM', 'LZFPM', 'LZFSM', 'LZPK', 'LZSK',
    'ZPERC', 'REXP', 'PFREE', 'PCTIM', 'ADIMP', 'RIVA', 'SIDE', 'RSERV',
]

#: Snow-17 module.
SNOW17_SET: List[str] = [
    'SCF', 'PXTEMP', 'MFMAX', 'MFMIN', 'NMF', 'MBASE', 'TIPM', 'UADJ',
    'PLWHC', 'DAYGM',
]

#: The prefix stripped from each bound set. Only the TOPMODEL names are
#: namespaced, so one prefix covers the aggregate set too: no CFE / NOAH / PET
#: / SAC-SMA / Snow-17 name starts with ``topmodel_``.
_TOPMODEL_PREFIX = 'topmodel_'

#: Full NGEN bound set: the union of every module, in the historical merge
#: order. ``smcmax`` and ``refkdt`` appear in both CFE and NOAH (one shared
#: definition), so the union is 82 names, not 84.
BOUND_SET: List[str] = (
    CFE_SET + NOAH_SET + PET_SET + TOPMODEL_SET + SACSMA_SET + SNOW17_SET
)

#: Model key -> (composition, strip prefix).
BOUND_SETS: Dict[str, tuple] = {
    'NGEN': (BOUND_SET, _TOPMODEL_PREFIX),
    'NGEN_CFE': (CFE_SET, ''),
    'NGEN_NOAH': (NOAH_SET, ''),
    'NGEN_PET': (PET_SET, ''),
    'NGEN_TOPMODEL': (TOPMODEL_SET, _TOPMODEL_PREFIX),
    'NGEN_SACSMA': (SACSMA_SET, ''),
    'NGEN_SNOW17': (SNOW17_SET, ''),
}


def register_bounds() -> None:
    """Contribute NGEN's bound-set compositions to the central catalogue.

    No ``params=`` anywhere: NGEN contributes no parameter definition of its
    own, by design. If a future NGEN module needs a genuinely NGEN-only
    parameter, define it here; if it needs one another model already has, list
    the existing catalogue name instead of redefining it.
    """
    for model, (names, prefix) in BOUND_SETS.items():
        register_model_bounds(model, names=names, strip_prefix=prefix)
