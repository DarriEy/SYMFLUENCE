# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The JAX-plugin parameter managers are re-exported lazily (review item 15).

HBV / SAC-SMA / Xinanjiang parameter managers come from the optional ``jax`` extra.
``symfluence.optimization.parameter_managers`` must import without them, while still
exposing the names (PEP 562 ``__getattr__``) and giving a clear install hint when the
plugin is absent.
"""

from __future__ import annotations

import importlib

import pytest

from symfluence.optimization import parameter_managers


def test_jax_param_manager_resolves_when_installed():
    # The JAX plugins are present in the test environment (the `jax` extra).
    assert isinstance(parameter_managers.HBVParameterManager, type)
    assert isinstance(parameter_managers.XinanjiangParameterManager, type)


def test_missing_plugin_raises_clear_install_hint(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "jsacsma.calibration.parameter_manager":
            raise ImportError("No module named 'jsacsma'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"symfluence\[jax\]"):
        _ = parameter_managers.SacSmaParameterManager


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = parameter_managers.DefinitelyNotARealParameterManager
