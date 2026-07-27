# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Characterization of the model knowledge still hardcoded in ``core``.

Service-decomposition phase 0 moves per-model declarations out of core and
into the packages that own them (bounds -> ``register_model_bounds``; output
/ spatial-mode / routing metadata -> the registered ``ModelConfigSchema``).
Every one of those moves must be a **provable no-op**: the values a model
resolves after the move are the values it resolved before.

This module pins the "before". It deliberately asserts the CURRENT values,
including the ones known to be stale or incomplete — those are tracked as
separate issues and fixed in their own reviewed PRs, never as a side effect
of an extraction refactor (the campaign's repro runs depend on these values,
and an unattributable regression here is expensive to find).

Where a table is known to be wrong, the test says so in a comment rather than
asserting the *correct* value, so the intent is not mistaken for an oversight.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from symfluence.core.calibration.parameters import get_model_bounds
from symfluence.core.calibration.parameters import parameter_bounds_registry as pbr
from symfluence.core.constants import SupportedModels
from symfluence.core.modeling.mixins.spatial_mode_mixin import SpatialModeDetectionMixin
from symfluence.core.modeling.utilities.routing_decider import RoutingDecider
from symfluence.core.modeling.utilities.runoff_loader import get_model_config

pytestmark = [pytest.mark.unit]

_SNAPSHOT = Path(__file__).parent / "data" / "model_bounds_snapshot.json"


# ---------------------------------------------------------------------------
# Calibration bounds
# ---------------------------------------------------------------------------

def _load_snapshot() -> dict:
    with _SNAPSHOT.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_snapshot_covers_every_servable_model():
    """The snapshot must not silently stop covering a model."""
    snapshot = _load_snapshot()
    assert set(snapshot) == set(pbr._BUILTIN_MODEL_BOUNDS), (
        "model_bounds_snapshot.json is out of sync with _BUILTIN_MODEL_BOUNDS. "
        "A model may only leave the built-in table by registering equivalent "
        "bounds through register_model_bounds() — regenerate the snapshot only "
        "when the bound VALUES are intentionally changed."
    )


@pytest.mark.parametrize("model", sorted(_load_snapshot()))
def test_model_bounds_match_snapshot(model):
    """Every bound value survives the move to per-package registration.

    This is the guard that lets the bounds catalogue be split across model
    packages without changing a single calibration result.
    """
    expected = _load_snapshot()[model]
    actual = get_model_bounds(model)

    assert set(actual) == set(expected), (
        f"{model}: parameter set changed "
        f"(added: {sorted(set(actual) - set(expected))}, "
        f"removed: {sorted(set(expected) - set(actual))})"
    )
    for name, bounds in expected.items():
        assert actual[name]["min"] == pytest.approx(bounds["min"]), f"{model}.{name} min"
        assert actual[name]["max"] == pytest.approx(bounds["max"]), f"{model}.{name} max"


# Served bound names are model-local: a model's catalogue entry may be
# namespaced (``fuse_MBASE``) and stripped on the way out, so one served name
# can legitimately mean different physics in different model families. These
# are the eight that legitimately do today. A NEW entry here means a model
# registered a name that collides with another model's without namespacing —
# the exact defect #368 fixed, where SACSMA's definitions silently overrode
# FUSE's and FUSE calibrated against Snow-17 melt bounds.
_KNOWN_NAME_COLLISIONS = {
    "K", "MBASE", "MFMAX", "MFMIN", "PWR", "R2N", "m", "soil_depth",
}


def test_no_new_shared_parameter_collisions():
    """A name serving different bounds in different models must be deliberate."""
    snapshot = _load_snapshot()
    by_name: dict[str, dict[str, tuple]] = {}
    for model, params in snapshot.items():
        for name, bounds in params.items():
            by_name.setdefault(name, {})[model] = (bounds["min"], bounds["max"])

    collisions = {
        name: per_model
        for name, per_model in by_name.items()
        if len(set(per_model.values())) > 1
    }

    unexpected = set(collisions) - _KNOWN_NAME_COLLISIONS
    assert not unexpected, (
        "new parameter-name collisions across models: "
        f"{ {n: collisions[n] for n in sorted(unexpected)} } — namespace the "
        "catalogue entry (e.g. 'mymodel_PARAM' + strip_prefix) instead of "
        "letting two models' definitions fight over one name"
    )

    resolved = _KNOWN_NAME_COLLISIONS - set(collisions)
    assert not resolved, (
        f"collisions {sorted(resolved)} no longer occur — remove them from "
        "_KNOWN_NAME_COLLISIONS so the guard keeps its teeth"
    )


# ---------------------------------------------------------------------------
# Runoff-source metadata (runoff_loader.MODEL_CONFIGS)
# ---------------------------------------------------------------------------

# The exact table core serves today. Moving this into each model's registered
# ModelConfigSchema.output must reproduce it field for field.
_EXPECTED_RUNOFF = {
    "SUMMA": dict(
        output_dir_key="EXPERIMENT_OUTPUT_SUMMA", output_dir_name="SUMMA",
        default_var="averageRoutedRunoff", default_units="m/s", default_dt="3600",
        output_file_pattern="{experiment_id}_timestep.nc",
        hru_dim="hru", hru_var="hruId", comment_name="SUMMA",
    ),
    "FUSE": dict(
        output_dir_key="EXPERIMENT_OUTPUT_FUSE", output_dir_name="FUSE",
        default_var="q_routed", default_units="m/s", default_dt="86400",
        # NOTE: the registered FUSE schema spells this '{domain}_...'; the two
        # declarations have already drifted. Parity pins the runoff_loader
        # spelling, which is what routing actually consumes today.
        output_file_pattern="{domain_name}_{experiment_id}_runs_def.nc",
        hru_dim="gru", hru_var="gruId", comment_name="FUSE",
    ),
    "GR": dict(
        output_dir_key="EXPERIMENT_OUTPUT_GR", output_dir_name="GR",
        default_var="q_routed", default_units="m/s", default_dt="86400",
        output_file_pattern="{domain_name}_{experiment_id}_runs_def.nc",
        hru_dim="gru", hru_var="gruId", comment_name="GR4J",
    ),
    "HYPE": dict(
        output_dir_key="EXPERIMENT_OUTPUT_HYPE", output_dir_name="HYPE",
        default_var="cout", default_units="m3/s", default_dt="86400",
        output_file_pattern="{experiment_id}_timestep.nc",
        hru_dim="gru", hru_var="gruId", comment_name="HYPE",
    ),
    "NGEN": dict(
        output_dir_key="EXPERIMENT_OUTPUT_NGEN", output_dir_name="NGEN",
        default_var="runoff", default_units="m/s", default_dt="3600",
        output_file_pattern="{experiment_id}_runoff.nc",
        hru_dim="hru", hru_var="hruId", comment_name="NGEN",
    ),
}


@pytest.mark.parametrize("model", sorted(_EXPECTED_RUNOFF))
def test_runoff_config_fields(model):
    cfg = get_model_config(model)
    for field, expected in _EXPECTED_RUNOFF[model].items():
        assert getattr(cfg, field) == expected, f"{model}.{field}"


@pytest.mark.parametrize("variant", ["GR4J", "GR5J", "GR6J"])
def test_gr_variants_normalize_to_gr(variant):
    assert get_model_config(variant).output_dir_name == "GR"


def test_unknown_model_falls_back_to_summa():
    """An unregistered model silently gets SUMMA's runoff layout.

    Characterized, not endorsed: this fallback hides typos and unregistered
    models behind plausible-looking SUMMA paths. Preserved here so the move
    is a no-op; changing it is a separate decision.
    """
    assert get_model_config("NOT_A_MODEL").output_dir_name == "SUMMA"


def test_runoff_lookup_is_case_insensitive():
    assert get_model_config("summa").output_dir_name == "SUMMA"
    assert get_model_config("  FuSe  ").output_dir_name == "FUSE"


# ---------------------------------------------------------------------------
# Routing decisions (RoutingDecider)
# ---------------------------------------------------------------------------

def test_spatial_mode_keys_table():
    assert RoutingDecider.SPATIAL_MODE_KEYS == {
        "SUMMA": "DOMAIN_DEFINITION_METHOD",
        "FUSE": "FUSE_SPATIAL_MODE",
        "HYPE": "HYPE_SPATIAL_MODE",
        "GR": "GR_SPATIAL_MODE",
        "MESH": "MESH_SPATIAL_MODE",
        "NGEN": "NGEN_SPATIAL_MODE",
    }


def test_routing_integration_keys_table():
    """FUSE is the only model whose routing-integration key core consults.

    Characterized, not endorsed: GR, VIC, SWAT, MHM and CRHM all define a
    ``<MODEL>_ROUTING_INTEGRATION`` key that this table cannot see, while
    ``spatial_orchestrator`` derives the same key by convention. Preserved
    verbatim so the schema migration is a no-op; tracked separately.
    """
    assert RoutingDecider.ROUTING_INTEGRATION_KEYS == {
        "FUSE": "FUSE_ROUTING_INTEGRATION",
    }


# ---------------------------------------------------------------------------
# Spatial-mode config resolution (SpatialModeDetectionMixin)
# ---------------------------------------------------------------------------

class _StubModelConfig:
    """Stands in for ``config.model`` with one attribute per model section."""

    def __init__(self, **sections):
        for name, value in sections.items():
            setattr(self, name, value)


class _StubSection:
    def __init__(self, spatial_mode):
        self.spatial_mode = spatial_mode


class _StubHost(SpatialModeDetectionMixin):
    def __init__(self, model_sections):
        self.config = type("Cfg", (), {"model": _StubModelConfig(**model_sections)})()


# Every model core can currently resolve a configured spatial mode for.
_SPATIAL_MODE_MODELS = [
    "HBV", "GR", "FUSE", "CFUSE", "JFUSE", "LSTM",
    "GNN", "SUMMA", "HYPE", "MESH", "NGEN", "RHESSYS",
]


@pytest.mark.parametrize("model", _SPATIAL_MODE_MODELS)
def test_configured_spatial_mode_resolves(model):
    host = _StubHost({model.lower(): _StubSection("distributed")})
    assert host._get_configured_spatial_mode(model) == "distributed"


@pytest.mark.parametrize("model", ["SWAT", "VIC", "CLM", "MODFLOW", "PRMS"])
def test_unlisted_models_resolve_no_spatial_mode(model):
    """Models absent from the map get None even when configured.

    Characterized, not endorsed: the map is a hand-maintained allow-list whose
    every entry is just ``model_name.lower()``, so ~20 models silently lose
    their configured spatial mode. Preserved so the simplification to a plain
    lowercase lookup is a deliberate, separately reviewed behaviour change.
    """
    host = _StubHost({model.lower(): _StubSection("distributed")})
    assert host._get_configured_spatial_mode(model) is None


def test_missing_config_section_resolves_none():
    host = _StubHost({})
    assert host._get_configured_spatial_mode("FUSE") is None


# ---------------------------------------------------------------------------
# Registry aliases and model capabilities
# ---------------------------------------------------------------------------

_EXPECTED_ALIASES = {
    "HEC-HMS": "HECHMS",
    "SAC-SMA": "SACSMA",
    "CLM-PARFLOW": "CLMPARFLOW",
    "RHESS": "RHESSYS",
    "SUMMA-MODFLOW": "COUPLED_GW",
}


@pytest.mark.parametrize("alias,canonical", sorted(_EXPECTED_ALIASES.items()))
def test_model_aliases_resolve(alias, canonical):
    """Hyphenated / alternate spellings keep resolving after aliases move.

    These are declared in ``core._bootstrap``; the target state is a manifest
    field owned by the package that registers the canonical key.
    """
    from symfluence.core.registries import R

    for registry in (R.runners, R.preprocessors, R.postprocessors,
                     R.optimizers, R.workers):
        resolved = registry.get(alias)
        canonical_value = registry.get(canonical)
        if canonical_value is not None:
            assert resolved is canonical_value, f"{alias} -> {canonical}"


def test_self_training_models():
    """Models whose calibration is internal training, not a parameter search."""
    assert SupportedModels.SELF_TRAINING == frozenset({"LSTM", "GNN"})
