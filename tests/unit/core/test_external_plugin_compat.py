# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""External model plugins keep working against the promoted contract tier.

The JAX model packages (jhbv, jsacsma, ...) are independent PyPI packages that
subclass the calibration and adapter bases. After the phase-0/step-0
promotions those bases live in core (``core.calibration``, ``core.modeling``)
with shims at the historical paths — these tests pin that an installed
external plugin still registers and that its components resolve to classes
subclassing the canonical (core) bases.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jhbv", reason="requires the jax extra (pip install 'symfluence[jax]')")

import symfluence  # noqa: E402,F401 — triggers plugin registration

from symfluence.core.calibration.workers.inmemory_worker import InMemoryModelWorker  # noqa: E402
from symfluence.core.registries import R  # noqa: E402


@pytest.mark.unit
def test_jax_plugin_registers_components():
    for namespace in (R.runners, R.workers, R.parameter_managers, R.config_schemas):
        assert namespace.get("HBV") is not None, f"HBV missing from {namespace}"


@pytest.mark.unit
def test_jax_worker_subclasses_canonical_core_base():
    worker_cls = R.workers.get("HBV")
    assert issubclass(worker_cls, InMemoryModelWorker), (
        f"{worker_cls} does not subclass the canonical core InMemoryModelWorker — "
        "an old-path shim is no longer aliasing the same class object"
    )


@pytest.mark.unit
def test_historical_base_import_paths_alias_core_classes():
    """The shim paths external packages import must be the SAME class objects."""
    from symfluence.core.calibration.optimizers.base_model_optimizer import (
        BaseModelOptimizer as core_bmo,
    )
    from symfluence.core.modeling.base import BaseModelRunner as core_bmr
    from symfluence.models.base import BaseModelRunner as legacy_bmr
    from symfluence.optimization.optimizers.base_model_optimizer import (
        BaseModelOptimizer as legacy_bmo,
    )

    assert legacy_bmr is core_bmr
    assert legacy_bmo is core_bmo
