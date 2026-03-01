# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Backward-compatible path resolution for the model-ready data store.

Provides a ``resolve_model_ready_path`` helper that returns the new
``data/model_ready/{data_type}/`` path when it exists and falls back to
legacy locations otherwise.  This follows the established "new first,
old fallback" pattern from ``core.path_resolver._get_catchment_file_path``.
"""

from pathlib import Path
from typing import Optional

# Maps data_type tokens to their legacy directory (relative to project_dir).
_LEGACY_PATHS = {
    'forcings':     'forcing/basin_averaged_data',
    'observations': 'observations',
    'attributes':   'shapefiles/catchment_intersection',
}


def resolve_model_ready_path(
    project_dir: Path,
    data_type: str,
    domain_name: Optional[str] = None,
    fallback: bool = True,
) -> Path:
    """Resolve a model-ready data path with optional legacy fallback.

    Args:
        project_dir: Root of the SYMFLUENCE domain directory.
        data_type: One of ``'forcings'``, ``'observations'``, or
            ``'attributes'``.
        domain_name: Domain name, used for file-level resolution (unused
            for directory-level resolution but kept for API symmetry).
        fallback: If *True* (default), return the legacy path when the
            new-style directory does not exist.

    Returns:
        Resolved ``Path``.  The new path is preferred; the legacy path is
        returned only when *fallback* is True and the new path is absent.

    Raises:
        ValueError: If *data_type* is not one of the known types.
    """
    if data_type not in _LEGACY_PATHS:
        raise ValueError(
            f"Unknown data_type '{data_type}'. "
            f"Must be one of {list(_LEGACY_PATHS)}"
        )

    new_path = project_dir / 'data' / 'model_ready' / data_type

    if new_path.exists():
        return new_path

    if fallback:
        legacy = project_dir / _LEGACY_PATHS[data_type]
        if legacy.exists():
            return legacy

    # Nothing exists yet — return the new canonical path so callers can
    # create it if needed.
    return new_path
