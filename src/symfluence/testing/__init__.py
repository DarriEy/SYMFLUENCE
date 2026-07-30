# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Public test-support surface for repositories built on SYMFLUENCE.

This exists so the ``symfluence-models`` extraction has something supported to
test against. Without it an extracted repository would have to copy this project's
root ``conftest.py`` and reach into ``Registry._entries`` to isolate
registrations — private coupling that makes an extraction fail later rather than
at the seam.

Note what is NOT here: HDF5 locking, thread pinning, tqdm and the Windows torch
DLL order are all configured by ``symfluence/__init__.py`` during its own import,
via ``symfluence.core.hdf5_safety``. Importing anything from this package imports
``symfluence`` first, so that work is already done — and duplicating it here
would fork the authoritative definition.

What belongs here: framework-owned test primitives — process setup, registry
isolation, config construction, domain scaffolding. What does not: anything
naming a specific model. A helper in core that knew ``SETTINGS_SUMMA_*`` would
undo phase 0's de-modeling, so model-specific keys are passed in by the caller.

Importing this module has no side effects; ``configure_test_environment()`` acts
only when called. It is not a runtime dependency — nothing under
``src/symfluence`` outside this package imports it.

Typical use from a downstream root ``conftest.py``::

    from symfluence.testing import configure_test_environment

    configure_test_environment()          # before numpy/HDF5/GDAL land

    pytest_plugins = ["symfluence.testing.plugin"]
"""
from __future__ import annotations

from symfluence.testing.config import make_config, scaffold_domain
from symfluence.testing.environment import configure_test_environment
from symfluence.testing.registries import registry_snapshot

__all__ = [
    "configure_test_environment",
    "make_config",
    "registry_snapshot",
    "scaffold_domain",
]
