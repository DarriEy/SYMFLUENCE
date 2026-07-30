# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Test-session environment setup, on top of what importing SYMFLUENCE does.

Most of the dangerous, order-sensitive work is NOT here, and deliberately so.
``symfluence/__init__.py`` already performs it during its own import, before any
of its heavy imports run:

- ``configure_hdf5_safety()`` force-sets the HDF5/netCDF locking variables and
  pins the numerical thread pools, and disables tqdm's monitor thread;
- on Windows it imports torch ahead of conda's HDF5 DLLs and registers conda's
  ``Library\\bin`` via ``os.add_dll_directory()``.

That ordering cannot be delegated to a caller anyway: reaching this module means
importing the ``symfluence`` package, so the framework's setup has already
happened by the time anything here can be called. A conftest that re-set those
variables would be duplicating ``symfluence.core.hdf5_safety`` — the single
authoritative definition — and duplicated definitions in this repository have
drifted before.

What remains are the genuinely *test-only* concerns, which the framework has no
business imposing on a production process: GDAL's exception mode and a headless,
writable matplotlib.

Call it once from the root ``conftest.py``::

    from symfluence.testing import configure_test_environment

    configure_test_environment()

It is idempotent.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["configure_test_environment"]

_configured = False


def _configure_gdal() -> None:
    """Opt into GDAL's exception mode.

    GDAL 4.0 warns when this is left implicit, and the legacy behaviour of
    returning ``None`` instead of raising turns a failed read into a confusing
    ``TypeError`` further downstream. Tests want the exception.
    """
    try:
        from osgeo import gdal
    except ImportError:
        return  # GDAL is optional
    gdal.UseExceptions()


def _configure_matplotlib() -> None:
    """Force a headless backend and a writable config directory.

    Without ``MPLBACKEND=Agg`` a plotting test tries to open a window on a
    machine with no display. ``MPLCONFIGDIR`` keeps the font cache off a possibly
    read-only home directory — a real failure mode on HPC and in containers.

    ``setdefault`` throughout: a caller that has deliberately chosen a backend
    keeps it.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "symfluence_matplotlib")
    )


def configure_test_environment(*, force: bool = False) -> None:
    """Apply the test-only environment setup.

    Args:
        force: Re-run even if already applied in this process. Only useful when
            testing this function; the environment is process-global, so an
            ordinary second call is a no-op.

    Does NOT configure HDF5 locking, thread pools, tqdm or the Windows torch DLL
    order — importing ``symfluence`` has already done all of that. See the module
    docstring.
    """
    global _configured
    if _configured and not force:
        return

    _configure_gdal()
    _configure_matplotlib()

    _configured = True
