# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Pure filename-based selection for model-ready forcing artifacts."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)
_REMAP_TOKEN_RE = re.compile(r'_remapped_([a-z][a-z0-9-]*?)(?:_|\.nc$)')


def discretization_token(discretization: object) -> str:
    """Return a filename-safe token identifying a spatial discretization."""
    token = re.sub(r'[^a-z0-9]+', '-', str(discretization or '').strip().lower()).strip('-')
    return token or 'default'


def forcing_name_matches_discretization(name: str, token: str) -> bool:
    """Whether *name* is a remapped forcing belonging to *token*."""
    filename = Path(name).name
    return (f'_remapped_{token}_' in filename) or filename.endswith(f'_remapped_{token}.nc')


def discretization_key_from_name(name: str) -> Optional[str]:
    """Extract a discretization token from a remapped forcing filename."""
    match = _REMAP_TOKEN_RE.search(Path(name).name)
    return match.group(1) if match else None


def select_forcing_files(
    forcing_files: Union[str, Path, List[Path]],
    discretization: object = None,
) -> List[Path]:
    """Select forcing artifacts matching a run's spatial discretization."""
    files = [forcing_files] if isinstance(forcing_files, (str, Path)) else list(forcing_files)
    files = [Path(path) for path in files]
    if not discretization:
        return files
    token = discretization_token(discretization)
    matched = [path for path in files if forcing_name_matches_discretization(path.name, token)]
    if not matched:
        return files
    if len(matched) != len(files):
        logger.debug(
            "Selected %d/%d forcing file(s) for discretization '%s' (token '%s')",
            len(matched), len(files), discretization, token,
        )
    return matched


__all__ = [
    'discretization_key_from_name',
    'discretization_token',
    'forcing_name_matches_discretization',
    'select_forcing_files',
]
