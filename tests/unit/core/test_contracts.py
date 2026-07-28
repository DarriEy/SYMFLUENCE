# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-family contract versioning (ADR-0009)."""
from __future__ import annotations

import pytest

from symfluence.core.contracts import (
    FAMILY_CONTRACTS,
    ContractCompatibilityError,
    assert_compatible,
    assert_plugin_compatible,
    contract_version,
    declared_plugin_contracts,
    is_compatible,
    plugin_contracts,
)


@pytest.mark.unit
def test_families_declared_with_valid_semver():
    assert set(FAMILY_CONTRACTS) == {"models", "calibration", "metrics", "geospatial-utils"}
    for family, version in FAMILY_CONTRACTS.items():
        major, minor, patch = (int(p) for p in version.split("."))
        assert (major, minor, patch) >= (0, 1, 0), family


@pytest.mark.unit
def test_acquisition_family_surfaces_existing_constant():
    from symfluence.data.backends.contract import PROTOCOL_VERSION

    assert contract_version("acquisition") == PROTOCOL_VERSION


@pytest.mark.unit
def test_compatibility_semantics_match_acquisition_contract():
    # pre-1.0: older-or-equal minor accepted, forward skew declined,
    # major mismatch declined — per family, independently.
    assert is_compatible("models", "0.1.0")  # older additive surface
    assert is_compatible("models", "0.2.0")  # older additive surface
    assert is_compatible("models", "0.3.0")  # older additive surface
    assert is_compatible("models", "0.4.0")  # current surface
    assert not is_compatible("models", "0.5.0")  # forward skew
    assert not is_compatible("models", "1.0.0")  # major mismatch
    assert not is_compatible("no-such-family", "0.1.0")
    assert not is_compatible("models", "not-a-version")


@pytest.mark.unit
def test_assert_compatible_message_names_versions():
    assert_compatible("metrics", "0.1.0")  # no raise
    with pytest.raises(RuntimeError, match=r"metrics contract 0\.9\.0.*provides 0\.1\.0"):
        assert_compatible("metrics", "0.9.0")


@pytest.mark.unit
def test_plugin_contract_declaration_is_copied_and_enforced():
    @plugin_contracts(models="0.1.0", metrics="0.1.0")
    def register():
        return None

    assert declared_plugin_contracts(register) == {
        "models": "0.1.0",
        "metrics": "0.1.0",
    }
    assert_plugin_compatible(register)


@pytest.mark.unit
def test_plugin_contract_declaration_can_live_on_parent_package():
    import symfluence.models
    from symfluence.models.summa import register

    targets = declared_plugin_contracts(register)
    assert targets == symfluence.models.__symfluence_contracts__
    assert targets["models"] == contract_version("models")
    assert_plugin_compatible(register)


@pytest.mark.unit
def test_incompatible_plugin_is_rejected_before_registration():
    @plugin_contracts(models="0.99.0")
    def register():
        raise AssertionError("must not be invoked")

    with pytest.raises(ContractCompatibilityError, match=r"models contract 0\.99\.0"):
        assert_plugin_compatible(register)


@pytest.mark.unit
def test_plugin_contract_decorator_rejects_bad_declarations():
    with pytest.raises(ValueError, match="unknown SYMFLUENCE contract"):
        plugin_contracts(unknown="0.1.0")
    with pytest.raises(ValueError, match="invalid models contract version"):
        plugin_contracts(models="latest")


@pytest.mark.unit
def test_mirrors_acquisition_is_compatible_semantics():
    """Same inputs -> same verdicts as the proven acquisition implementation."""
    from symfluence.data.backends.contract import (
        PROTOCOL_VERSION,
    )
    from symfluence.data.backends.contract import (
        is_compatible as acq_is_compatible,
    )

    for target in ["0.1.0", "0.3.0", PROTOCOL_VERSION, "0.99.0", "1.0.0", "bad"]:
        assert is_compatible("acquisition", target) == acq_is_compatible(target), target


@pytest.mark.unit
def test_register_algorithm_seam():
    """External algorithms register through the calibration-family seam."""
    from symfluence.core.calibration.optimizers.algorithms import (
        ALGORITHM_REGISTRY,
        get_algorithm,
        list_algorithms,
        register_algorithm,
    )
    from symfluence.core.calibration.optimizers.algorithms.base_algorithm import (
        OptimizationAlgorithm,
    )

    class _ExtAlgorithm(OptimizationAlgorithm):
        def __init__(self, config, logger):
            self.config, self.logger = config, logger

        @property
        def name(self) -> str:
            return "ext_search"

        def optimize(self, *a, **k):  # pragma: no cover - contract stub
            raise NotImplementedError

    try:
        register_algorithm("Ext-Search", _ExtAlgorithm, aliases=("es",))
        # normalized spellings reachable through the standard entry point
        import logging
        inst = get_algorithm("EXT_SEARCH", {}, logging.getLogger("t"))
        assert isinstance(inst, _ExtAlgorithm)
        assert isinstance(get_algorithm("es", {}, logging.getLogger("t")), _ExtAlgorithm)
        assert "ext_search" in list_algorithms()
        # collisions refused
        with pytest.raises(ValueError, match="already registered"):
            register_algorithm("dds", _ExtAlgorithm)
    finally:
        for key in ("ext_search", "es"):
            ALGORITHM_REGISTRY.pop(key, None)
        from symfluence.core.calibration.optimizers import algorithms as _alg
        if "ext_search" in _alg._EXTERNAL_PRIMARY_NAMES:
            _alg._EXTERNAL_PRIMARY_NAMES.remove("ext_search")
