# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""model_packages_with: disk discovery of per-model side-modules.

Replaces the hardcoded SupportedModels.WITH_FORCING_ADAPTER / WITH_PRESETS lists
(RTI review item 18). These submodules register themselves on import, so the set
of models that have them is read from the source tree rather than a registry they
populate — and is therefore always in sync.
"""

from __future__ import annotations

from pathlib import Path

import symfluence.models
from symfluence.models import model_packages_with


def test_forcing_adapter_discovery_includes_watflood():
    """watflood has a forcing_adapter.py but was missing from the old list (drift fix)."""
    pkgs = model_packages_with("forcing_adapter")
    assert {"summa", "fuse", "watflood"} <= set(pkgs)
    assert pkgs == tuple(sorted(pkgs))


def test_preset_discovery_matches_disk():
    """Preset discovery returns only models that actually ship init_preset.py."""
    pkgs = model_packages_with("init_preset")
    assert {"summa", "fuse"} <= set(pkgs)
    models_dir = Path(symfluence.models.__file__).resolve().parent
    for name in pkgs:
        assert (models_dir / name / "init_preset.py").is_file()


def test_unknown_submodule_returns_empty():
    assert model_packages_with("definitely_not_a_submodule") == ()
