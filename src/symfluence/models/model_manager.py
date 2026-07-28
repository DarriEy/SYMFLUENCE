# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.project.model_manager``.

Unlike the sibling shims in this package, this one resolves its target *only*
inside ``__getattr__``. Those point at ``symfluence.core``, which ``models`` may
depend on freely; this one points at ``symfluence.project``, an upper layer. A
module-level ``import symfluence.project.model_manager`` here would mean that
merely importing the models package pulls in the orchestration layer — the
wrong direction for a package being extracted into its own distribution, and
invisible to the guard until ``symfluence.project`` became a forbidden prefix
for ``models`` in ``scripts/check_core_layering.py``.
"""
from __future__ import annotations


def __getattr__(name: str):
    from symfluence.project import model_manager as _impl

    return getattr(_impl, name)


def __dir__() -> list[str]:
    from symfluence.project import model_manager as _impl

    return dir(_impl)
