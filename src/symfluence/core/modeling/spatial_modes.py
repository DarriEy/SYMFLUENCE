# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Spatial Mode Definitions for SYMFLUENCE Models.

Provides centralized spatial mode validation across all model runners.
This module defines the spatial-mode vocabulary, the *shape* of a model's
spatial capability declaration, and the validation logic that reads it.

Phase 3 Addition: Centralizes spatial mode validation that was previously scattered
across individual model runners.

Amended during service-decomposition prep (July 2026): the per-model VALUES are
no longer a table in this file. ``MODEL_SPATIAL_CAPABILITIES`` was a hardcoded
dict of 16 per-model declarations living in ``core``, which meant

* a model package could not change its own spatial capabilities without a core
  edit and a core release, and
* an out-of-tree plugin package could not declare them **at all** — it silently
  fell into the "unknown model" branch of :func:`validate_spatial_mode` and got
  no validation whatsoever.

:func:`register_model_spatial_capability` is the seam. It mirrors
``register_model_bounds()`` in
``core.calibration.parameters.parameter_bounds_registry``: core owns the record
type and the read side, the owning package contributes the values at
registration time, and the read side is unchanged for every consumer.

Why this shape rather than the alternatives considered:

* ``R.base_settings`` (a plain per-model string registry) carries untyped
  values and no notion of a structured declaration; a capability is a record
  with four fields and a merge policy, not a settings string.
* ``ModelConfigSchema`` / ``register_model_schema`` is the closest fit and is
  where ``spatial_mode_key`` already lives — but that schema is about *config
  keys* (installation/execution/input/output are all required fields), so a
  model would have to invent a full config contract just to say "I support
  lumped only". Several capability-declaring models (MHM, CRHM, PCRGLOBWB,
  WATFLOOD) have no such schema today. A dedicated registration function keeps
  the two declarations independent, exactly as ``register_model_bounds`` is
  independent of the schema.

Until the per-model migration lands, core seeds the historical values through
the same public function (see ``_BUILTIN_SPATIAL_CAPABILITIES`` below), so the
in-tree behaviour is byte-identical while the seam is already live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

_logger = logging.getLogger(__name__)


class SpatialMode(Enum):
    """
    Spatial modeling mode enumeration.

    Attributes:
        LUMPED: Single unit domain (point-scale, no spatial heterogeneity)
        SEMI_DISTRIBUTED: Multiple units with partial spatial representation
        DISTRIBUTED: Full spatial discretization (HRUs/grid cells with routing)
    """
    LUMPED = "lumped"
    SEMI_DISTRIBUTED = "semi_distributed"
    DISTRIBUTED = "distributed"

    @classmethod
    def from_string(cls, value: str) -> 'SpatialMode':
        """Normalize aliases and parse spatial mode from string.

        Handles common variations like 'point' -> LUMPED,
        'delineate' -> DISTRIBUTED, 'semidistributed' -> SEMI_DISTRIBUTED.
        """
        normalized = value.lower().replace('-', '_').replace(' ', '_')
        mapping = {
            'lumped': cls.LUMPED,
            'point': cls.LUMPED,
            'semi_distributed': cls.SEMI_DISTRIBUTED,
            'semidistributed': cls.SEMI_DISTRIBUTED,
            'distributed': cls.DISTRIBUTED,
            'delineate': cls.DISTRIBUTED,
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown spatial mode: {value}. Valid: {list(mapping.keys())}")
        return mapping[normalized]

    def __eq__(self, other):
        if isinstance(other, Enum) and hasattr(other, 'value'):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value


@dataclass
class ModelSpatialCapability:
    """
    Defines spatial mode capabilities for a specific model.

    Attributes:
        supported_modes: Set of SpatialMode values the model supports
        default_mode: The default spatial mode if none specified
        requires_routing: Dict mapping SpatialMode to whether routing is required
        warning_message: Optional warning message for suboptimal configurations
    """
    supported_modes: Set[SpatialMode]
    default_mode: SpatialMode
    requires_routing: Dict[SpatialMode, bool] = field(default_factory=dict)
    warning_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Capability registry (the extension seam)
# ---------------------------------------------------------------------------

class _CapabilityRegistry(Dict[str, "ModelSpatialCapability"]):
    """Registry storage that is also the legacy public mapping.

    ``MODEL_SPATIAL_CAPABILITIES`` is a PUBLIC name on the models contract
    surface: external model packages import it directly, and before the
    registration seam existed, assigning into it was the *only* way such a
    package could declare its spatial capabilities. The ``models`` contract is
    pre-1.0 and additive-only, so neither the name nor any behaviour it had may
    be withdrawn — which rules out both replacing it with a static snapshot
    (would go stale the moment a package registers) and wrapping it in a
    read-only proxy (a write that worked at 0.3.0 would start raising).

    Making the registry storage a ``dict`` subclass satisfies every constraint
    at once: the exported name IS the live registry (so old importers and the
    new seam can never disagree), it is a genuine ``dict`` (so
    ``isinstance(..., dict)`` in third-party code still holds), and legacy
    item assignment is forwarded to :func:`register_model_spatial_capability`
    so it picks up key normalisation and double-registration logging.
    """

    def __setitem__(self, key: str, value: "ModelSpatialCapability") -> None:
        # Route legacy `MODEL_SPATIAL_CAPABILITIES['X'] = cap` writes through
        # the seam. The seam writes back with dict.__setitem__ to avoid
        # recursing through this override.
        register_model_spatial_capability(key, value)


#: Runtime registry: model key (uppercase) -> declared capability. Populated by
#: :func:`register_model_spatial_capability`, seeded below with the values core
#: historically hardcoded, and exported as ``MODEL_SPATIAL_CAPABILITIES``.
_SPATIAL_CAPABILITIES: _CapabilityRegistry = _CapabilityRegistry()

#: Keys whose current value came from core's compatibility seed rather than
#: from the owning package. A package registration replaces one of these
#: silently and by design (that is the migration path); replacing a value a
#: *package* already contributed is a genuine double-registration and is logged.
_SEEDED_KEYS: Set[str] = set()


def register_model_spatial_capability(
    model: str,
    capability: ModelSpatialCapability,
) -> None:
    """Declare a model's spatial-mode capabilities.

    The extension seam for model packages (in-tree or external): call this from
    the package's ``register()`` so :func:`validate_spatial_mode` and
    :func:`get_model_capabilities` serve the model's own declaration without
    editing core.

    A model that never registers stays *unknown* to validation, which
    deliberately permits any spatial mode — declaring nothing must never be
    more restrictive than declaring something.

    Args:
        model: Model key, case-insensitive (e.g. ``'FUSE'``, ``'MYMODEL'``).
        capability: The :class:`ModelSpatialCapability` record to serve.
    """
    key = model.upper()
    if key in _SPATIAL_CAPABILITIES and key not in _SEEDED_KEYS:
        _logger.warning(
            "spatial capability for '%s' re-registered; the later declaration "
            "wins. Two packages should not own one model's capabilities.",
            key,
        )
    dict.__setitem__(_SPATIAL_CAPABILITIES, key, capability)
    _SEEDED_KEYS.discard(key)


def registered_spatial_capability_models() -> List[str]:
    """Model keys with a declared spatial capability."""
    return sorted(_SPATIAL_CAPABILITIES)


def spatial_capabilities() -> Dict[str, ModelSpatialCapability]:
    """Snapshot of every declared spatial capability.

    A detached copy: mutating it does not change the registry. Use
    :func:`register_model_spatial_capability` to contribute a declaration.
    """
    return dict(_SPATIAL_CAPABILITIES)


# ---------------------------------------------------------------------------
# Compatibility seed
# ---------------------------------------------------------------------------
# These are the values ``core`` hardcoded before the seam existed, kept here
# verbatim so in-tree behaviour is unchanged. FOLLOW-UP (tracked with the rest
# of service-decomposition item 2): move each entry into the ``register()`` of
# the model package that owns it — ``models/<name>/__init__.py`` calls
# ``register_model_spatial_capability('<NAME>', ...)`` and the entry is deleted
# from this dict. When the dict is empty this whole section goes away and core
# holds no per-model spatial knowledge at all.
_BUILTIN_SPATIAL_CAPABILITIES: Dict[str, ModelSpatialCapability] = {
    'SUMMA': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'FUSE': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'GR': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'LSTM': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # LSTM handles routing internally
            SpatialMode.SEMI_DISTRIBUTED: False,
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "LSTM works best in lumped mode for streamflow prediction. "
            "Consider using GNN for spatially-distributed graph-based modeling."
        )
    ),

    'GNN': ModelSpatialCapability(
        supported_modes={SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False  # GNN has internal graph-based routing
        },
        warning_message=(
            "GNN requires distributed domain with graph structure. "
            "Use LSTM for lumped modeling."
        )
    ),

    'HYPE': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.SEMI_DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # HYPE has internal routing
            SpatialMode.SEMI_DISTRIBUTED: False,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'MESH': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # MESH has internal routing (WATFLOOD/PDMROF)
            SpatialMode.SEMI_DISTRIBUTED: False,
            SpatialMode.LUMPED: False  # Uses noroute mode (RFF+DRAINSOL proxy)
        },
        warning_message=None  # Lumped mode now fully supported
    ),

    'NGEN': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,  # Uses t-route for routing
            SpatialMode.SEMI_DISTRIBUTED: True,
            SpatialMode.LUMPED: False
        },
        warning_message=None
    ),

    'RHESSYS': ModelSpatialCapability(
        # RHESSys is inherently hierarchical/distributed but can operate with a
        # single aggregate hillslope/patch for lumped experiments.
        supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # Internal hillslope routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "RHESSys performs best with distributed landscape hierarchy. "
            "Lumped mode is supported when world/flow files are pre-aggregated."
        )
    ),

    'VIC': ModelSpatialCapability(
        # VIC is designed for grid-based distributed modeling but can operate
        # with a single-cell domain for lumped experiments.
        supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: True,  # VIC outputs cell runoff, needs external routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "VIC is designed for distributed grid-based modeling. "
            "For lumped mode, a single-cell domain will be created."
        )
    ),

    'SWAT': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={SpatialMode.LUMPED: False},
        warning_message=(
            "SWAT is a semi-distributed model. Lumped mode uses "
            "a single-HRU/subbasin configuration."
        )
    ),

    'MHM': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={SpatialMode.LUMPED: False},
        warning_message=(
            "mHM is a mesoscale hydrological model. Lumped mode uses "
            "a single-cell domain with multiscale parameter regionalization."
        )
    ),

    'CRHM': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED},
        default_mode=SpatialMode.LUMPED,
        requires_routing={SpatialMode.LUMPED: False},
        warning_message=(
            "CRHM is a cold-region hydrological model. Lumped mode uses "
            "a single-HRU configuration with blowing snow and frozen soil."
        )
    ),

    'GSFLOW': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.SEMI_DISTRIBUTED},
        default_mode=SpatialMode.SEMI_DISTRIBUTED,
        requires_routing={
            SpatialMode.SEMI_DISTRIBUTED: False,  # Internal SFR routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "GSFLOW couples PRMS surface processes with MODFLOW-NWT groundwater. "
            "Internal SFR/UZF packages handle GW-SW exchange."
        )
    ),

    'WATFLOOD': ModelSpatialCapability(
        supported_modes={SpatialMode.LUMPED, SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # Internal channel routing
            SpatialMode.LUMPED: False
        },
        warning_message=(
            "WATFLOOD uses GRU-grid distributed structure with internal "
            "channel routing. Lumped mode uses a single-GRU configuration."
        )
    ),

    'PCRGLOBWB': ModelSpatialCapability(
        supported_modes={SpatialMode.DISTRIBUTED},
        default_mode=SpatialMode.DISTRIBUTED,
        requires_routing={
            SpatialMode.DISTRIBUTED: False,  # Internal accuTravelTime routing
        },
        warning_message=(
            "PCR-GLOBWB is inherently grid-based with internal "
            "accuTravelTime routing. Lumped mode uses a 3x3 grid."
        )
    ),
}


def _seed_builtin_capabilities() -> None:
    """Contribute the compatibility values through the public seam.

    Deliberately routed through :func:`register_model_spatial_capability`
    rather than assigned into the registry dict: the seed exercises the same
    code path a package uses, so the seam cannot rot while the built-ins still
    work. Keys seeded here are marked so a later package registration replaces
    them without a double-registration warning.
    """
    for name, capability in _BUILTIN_SPATIAL_CAPABILITIES.items():
        key = name.upper()
        if key in _SPATIAL_CAPABILITIES:
            # A package already declared it (import order put its register()
            # first) — the package owns the model, so leave its value alone.
            continue
        register_model_spatial_capability(key, capability)
        _SEEDED_KEYS.add(key)


_seed_builtin_capabilities()


def __getattr__(name: str):
    """Public module attribute ``MODEL_SPATIAL_CAPABILITIES`` (PEP 562).

    Part of the models contract surface — external model packages import this
    name directly, so it must keep resolving with its historical mapping
    semantics. It returns the LIVE registry (a real ``dict``; see
    :class:`_CapabilityRegistry`), not a snapshot, so old importers and the
    registration seam always see the same data.

    New code should prefer :func:`get_model_capabilities` /
    :func:`spatial_capabilities` for reads and
    :func:`register_model_spatial_capability` for writes.
    """
    if name == "MODEL_SPATIAL_CAPABILITIES":
        return _SPATIAL_CAPABILITIES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_spatial_mode_from_config(config_dict) -> SpatialMode:
    """
    Determine spatial mode from configuration dictionary or typed config.

    Uses DOMAIN_DEFINITION_METHOD and ROUTING_DELINEATION to infer the spatial mode.

    Args:
        config_dict: Configuration dictionary or typed config with domain settings

    Returns:
        Inferred SpatialMode
    """
    try:
        domain_method = config_dict.domain.definition_method or 'lumped'
    except (AttributeError, TypeError):
        domain_method = config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped') if isinstance(config_dict, dict) else 'lumped'
    try:
        routing_delineation = config_dict.model.mizuroute.routing_delineation or 'lumped'
    except (AttributeError, TypeError):
        routing_delineation = config_dict.get('ROUTING_DELINEATION', 'lumped') if isinstance(config_dict, dict) else 'lumped'

    # Map domain method to spatial mode
    if domain_method in ('point', 'lumped'):
        if routing_delineation == 'river_network':
            # Lumped domain but with network routing = semi-distributed behavior
            return SpatialMode.SEMI_DISTRIBUTED
        return SpatialMode.LUMPED

    elif domain_method in ('subset', 'semi_distributed'):
        return SpatialMode.SEMI_DISTRIBUTED

    elif domain_method in ('delineate', 'distributed'):
        return SpatialMode.DISTRIBUTED

    # Default to lumped if unknown
    return SpatialMode.LUMPED


def validate_spatial_mode(
    model_name: str,
    spatial_mode: SpatialMode,
    has_routing_configured: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Validate spatial mode for a specific model.

    Args:
        model_name: Name of the model (uppercase)
        spatial_mode: The spatial mode to validate
        has_routing_configured: Whether routing model is configured

    Returns:
        Tuple of (is_valid, warning_message)
    """
    model_name = model_name.upper()

    capability = _SPATIAL_CAPABILITIES.get(model_name)
    if capability is None:
        # Model declared no spatial capability - allow any mode. Declaring
        # nothing must never be more restrictive than declaring something,
        # otherwise adding the seam would break every model that has not
        # migrated (and every external plugin).
        return True, None

    # Check if mode is supported
    if spatial_mode not in capability.supported_modes:
        return False, (
            f"{model_name} does not support '{spatial_mode.value}' mode. "
            f"Supported modes: {[m.value for m in capability.supported_modes]}"
        )

    # Check routing requirements
    if capability.requires_routing.get(spatial_mode, False) and not has_routing_configured:
        warning = (
            f"{model_name} in {spatial_mode.value} mode typically requires a routing model "
            f"(e.g., mizuRoute). Consider adding ROUTING_MODEL to configuration."
        )
        return True, warning

    # Return any general warning for the model/mode combination
    if capability.warning_message and spatial_mode != capability.default_mode:
        return True, capability.warning_message

    return True, None


def get_model_capabilities(model_name: str) -> Optional[ModelSpatialCapability]:
    """
    Get the declared spatial capabilities for a model.

    Args:
        model_name: Name of the model

    Returns:
        ModelSpatialCapability if the model declared one, None otherwise.
        ``None`` means "no declaration", not "no capabilities" — see
        :func:`validate_spatial_mode`.
    """
    return _SPATIAL_CAPABILITIES.get(model_name.upper())
