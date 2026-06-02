# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MPI launcher discovery for models that execute under mpirun/mpiexec.

Several process-based models (ParFlow, CLM-ParFlow, WRF-Hydro) launch their
executables through an MPI launcher with the ``<launcher> -np <n> <exe>``
convention. :func:`find_mpirun` locates a suitable launcher, preferring one
bundled alongside the model executable (as shipped by the npm binary
distribution) and falling back to the system ``PATH``.

Kept dependency-free (stdlib only) so it is safe to import from model runners
at module load time.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

# Launchers that accept the ``-np <n> <exe>`` calling convention the model
# runners use. ``srun`` is intentionally excluded — it needs different flags
# and PMI negotiation, so callers that want it build their command separately.
_LAUNCHERS = ("mpirun", "mpiexec")


def find_mpirun(exe: Union[str, os.PathLike, None] = None) -> Optional[str]:
    """Return the path to an MPI launcher (``mpirun``/``mpiexec``), or ``None``.

    Search order:

    1. A launcher bundled next to *exe* — the npm distribution ships one
       alongside the model binaries, so an offline/HPC install works without a
       system MPI.
    2. The system ``PATH`` (``shutil.which``).

    Parameters
    ----------
    exe:
        Path to the model executable about to be launched. Used only to look
        for a co-located launcher; may be ``None`` to search ``PATH`` only.

    Returns
    -------
    Optional[str]
        Absolute path (bundled case) or bare name resolved on ``PATH``, or
        ``None`` if no launcher is available.
    """
    if exe:
        exe_dir = Path(exe).resolve().parent
        for name in _LAUNCHERS:
            candidate = exe_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    for name in _LAUNCHERS:
        found = shutil.which(name)
        if found:
            return found

    return None
