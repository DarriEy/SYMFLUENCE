# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SYMFLUENCE-side smoke test for the external dRoute plugin contract.

dRoute model-specific code lives in the external ``droute`` package (JAX-model plugin pattern);
SYMFLUENCE only provides the boundary hooks. These tests assert that, when ``droute`` is installed,
the plugin registers its components through the entry-point/``model_manifest`` path -- and that the
dCoupler adapter no longer shadows the registered worker with an in-tree copy (the migration fixed a
bug where importing ``symfluence.models.droute.calibration.worker`` overwrote the canonical worker).
"""
from __future__ import annotations

import importlib.util

import pytest

# Skip cleanly when the optional droute plugin is not installed.
droute_installed = importlib.util.find_spec("droute") is not None
pytestmark = pytest.mark.skipif(not droute_installed, reason="droute plugin not installed")


def test_droute_worker_resolves_to_external_package():
    """The DROUTE worker/PM/schema resolve to the external droute package, not an in-tree copy."""
    import symfluence  # noqa: F401  -- triggers plugin discovery -> droute.register()
    from symfluence.core.registries import R

    worker = R.workers.get('DROUTE')
    pm = R.parameter_managers.get('DROUTE')
    schema = R.config_schemas.get('DROUTE')

    assert worker is not None and worker.__module__.startswith('droute.'), worker
    assert pm is not None and pm.__module__.startswith('droute.'), pm
    assert schema is not None and schema.__module__.startswith('droute.'), schema


def test_no_in_tree_droute_package():
    """The in-tree symfluence.models.droute package was removed during the migration."""
    assert importlib.util.find_spec("symfluence.models.droute") is None


def test_dcoupler_adapter_does_not_shadow_worker():
    """Regression: exercising the dCoupler adapter's worker import must not replace the registered
    DROUTE worker with an in-tree duplicate (the pre-migration shadowing bug)."""
    import symfluence  # noqa: F401
    from symfluence.core.registries import R

    before = R.workers.get('DROUTE')
    # This mirrors what droute_adapter.initialize() imports at call-time:
    from droute.calibration.worker import DRouteWorker  # noqa: F401
    after = R.workers.get('DROUTE')

    assert before is after, "DROUTE worker was shadowed after importing the adapter's worker"
    assert after.__module__.startswith('droute.')
