# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Locate binaries shipped via the npm package (``npm install -g symfluence``).

The npm bundle stages pre-built tools in a flat ``dist/bin`` (and shared
libraries in ``dist/lib``) under the global npm root. Several places need to
find that directory — the CLI diagnostics, the binary pass-through command, the
model runners' executable resolver, and the ngen preprocessor's BMI-lib lookup.
This module is the single source of truth for that location so the
``$(npm root -g)/symfluence/dist/...`` convention is not re-encoded per call
site.

Resolution honours the ``SYMFLUENCE_NPM_DIST_BIN`` environment override (an
explicit path to the bundle's ``bin`` dir), which is also how tests point at a
fixture bundle without a real npm install.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

_NPM_DIST_BIN_ENV = "SYMFLUENCE_NPM_DIST_BIN"


def _npm_global_dist() -> Optional[Path]:
    """Return ``$(npm root -g)/symfluence/dist`` if it exists, else ``None``."""
    try:
        result = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        return None

    if result.returncode != 0:
        return None
    try:
        dist = Path(result.stdout.strip()) / "symfluence" / "dist"
    except (ValueError, OSError):
        return None
    return dist if dist.is_dir() else None


def npm_bundle_bin() -> Optional[Path]:
    """Return the npm bundle's ``dist/bin`` directory, or ``None``.

    Honours ``SYMFLUENCE_NPM_DIST_BIN`` (explicit bin dir) before falling back to
    ``$(npm root -g)``. The ``npm root -g`` subprocess is only spawned when the
    override is unset.
    """
    override = os.getenv(_NPM_DIST_BIN_ENV)
    if override:
        override_path = Path(override)
        return override_path if override_path.is_dir() else None

    dist = _npm_global_dist()
    if dist is None:
        return None
    bin_dir = dist / "bin"
    return bin_dir if bin_dir.is_dir() else None


def npm_bundle_lib() -> Optional[Path]:
    """Return the npm bundle's ``dist/lib`` directory, or ``None``.

    When ``SYMFLUENCE_NPM_DIST_BIN`` is set, ``lib`` is taken as its sibling
    (``dist/bin`` and ``dist/lib`` always live side by side in the bundle).
    """
    override = os.getenv(_NPM_DIST_BIN_ENV)
    if override:
        lib_dir = Path(override).parent / "lib"
        return lib_dir if lib_dir.is_dir() else None

    dist = _npm_global_dist()
    if dist is None:
        return None
    lib_dir = dist / "lib"
    return lib_dir if lib_dir.is_dir() else None
