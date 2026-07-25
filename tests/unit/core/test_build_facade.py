# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The core.build facade must be a real package, not a namespace package.

Regression guard: an unanchored ``build/`` in .gitignore once swallowed
``src/symfluence/core/build/__init__.py`` — the working tree had it, git did
not, and every fresh clone imported ``symfluence.core.build`` as a bare
namespace package whose facade names failed to import (masked at runtime by
the npm resolver's per-provider exception handling).
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_core_build_is_a_real_package_with_facade_exports():
    import symfluence.core.build as build_pkg

    assert build_pkg.__file__ is not None, (
        "symfluence.core.build resolved as a namespace package — its "
        "__init__.py is missing from the distribution"
    )

    from symfluence.core.build import (  # noqa: F401
        BuildSnippetCatalog,
        get_common_build_environment,
        get_netcdf_detection,
        get_safe_build_path,
    )
