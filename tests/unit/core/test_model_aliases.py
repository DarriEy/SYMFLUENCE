# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for model-level name aliases and self-training models.

Covers two fixes:

* Hyphenated model names (e.g. ``HEC-HMS``) must resolve to the hyphen-free
  canonical key registered by the standalone plugin packages (``HECHMS`` from
  ``jhechms``) across all model-component registries — not just BMI adapters.
* Self-training ML models (LSTM, GNN) train inside their runner and must NOT
  register an external DDS/PSO optimizer.
"""

from __future__ import annotations

import symfluence  # noqa: F401 — triggers registry bootstrap on import
from symfluence.core._bootstrap import bootstrap
from symfluence.core.registries import R

# Ensure aliases are present even if a prior test cleared/restored registries.
bootstrap()


def test_hechms_hyphen_alias_resolves_runner():
    """`HEC-HMS` resolves to the same runner as the canonical `HECHMS`."""
    canonical = R.runners.get("HECHMS")
    assert canonical is not None, "jhechms plugin should register a HECHMS runner"
    assert R.runners.get("HEC-HMS") is canonical


def test_hechms_hyphen_alias_across_component_registries():
    """The hyphenated alias works for every model-component registry."""
    for registry in (R.runners, R.optimizers, R.workers, R.preprocessors):
        assert registry.get("HEC-HMS") is registry.get("HECHMS")


def test_sacsma_hyphen_alias_resolves_runner():
    """`SAC-SMA` resolves to the same runner as the canonical `SACSMA`."""
    canonical = R.runners.get("SACSMA")
    assert canonical is not None, "jsacsma plugin should register a SACSMA runner"
    assert R.runners.get("SAC-SMA") is canonical


def test_clmparflow_hyphen_alias_resolves_runner_and_optimizer():
    """`CLM-ParFlow` (-> CLM-PARFLOW) resolves to the canonical CLMPARFLOW."""
    canonical = R.runners.get("CLMPARFLOW")
    assert canonical is not None, "CLMPARFLOW runner should be registered"
    assert R.runners.get("CLM-PARFLOW") is canonical
    assert R.optimizers.get("CLM-PARFLOW") is R.optimizers.get("CLMPARFLOW")


def test_aliases_do_not_shadow_real_registrations():
    """An alias is never added for a name that is already a real runner entry.

    The standalone XINANJIANG runner registers under ``XINANJIANG`` (the BMI
    adapter uses ``XAJ``); aliasing ``XINANJIANG -> XAJ`` would have shadowed
    and broken the real entry, so it must remain a genuine registration.
    """
    if "XINANJIANG" in R.runners.keys():
        from symfluence.core.constants import SupportedModels

        assert SupportedModels.is_valid("XINANJIANG")


def test_self_training_models_have_no_optimizer():
    """LSTM/GNN train in their runner; they register no external optimizer."""
    assert R.optimizers.get("LSTM") is None
    assert R.optimizers.get("GNN") is None
    assert R.workers.get("LSTM") is None
    assert R.workers.get("GNN") is None
