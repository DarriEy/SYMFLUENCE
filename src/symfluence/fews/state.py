# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS state file exchange.

Copies warm-start NetCDF state files between the FEWS state directories
and the model's internal state directory.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from .exceptions import StateExchangeError

logger = logging.getLogger(__name__)


def import_states(
    fews_state_dir: Optional[Path],
    model_state_dir: Path,
    pattern: str = "*.nc",
) -> List[Path]:
    """Copy state files from FEWS ``stateInputDir`` into the model state directory.

    Args:
        fews_state_dir: FEWS state input directory (may be None / not exist)
        model_state_dir: Model's state directory
        pattern: Glob pattern for state files

    Returns:
        List of copied file paths in the model state directory

    Raises:
        StateExchangeError: If copy fails
    """
    if fews_state_dir is None:
        logger.debug("No FEWS state input directory specified, skipping import")
        return []

    fews_state_dir = Path(fews_state_dir)
    if not fews_state_dir.is_dir():
        logger.debug("FEWS state input directory does not exist: %s", fews_state_dir)
        return []

    state_files = sorted(fews_state_dir.glob(pattern))
    if not state_files:
        logger.debug("No state files matching '%s' in %s", pattern, fews_state_dir)
        return []

    try:
        model_state_dir = Path(model_state_dir)
        model_state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StateExchangeError(
            f"Cannot create model state directory {model_state_dir}: {exc}"
        ) from exc

    copied: List[Path] = []
    for src in state_files:
        dst = model_state_dir / src.name
        try:
            shutil.copy2(str(src), str(dst))
            copied.append(dst)
            logger.info("Imported state file: %s -> %s", src, dst)
        except OSError as exc:
            raise StateExchangeError(
                f"Failed to copy state file {src} -> {dst}: {exc}"
            ) from exc

    return copied


def export_states(
    model_state_dir: Path,
    fews_state_dir: Optional[Path],
    pattern: str = "*.nc",
) -> List[Path]:
    """Copy state files from the model state directory to FEWS ``stateOutputDir``.

    Args:
        model_state_dir: Model's state directory
        fews_state_dir: FEWS state output directory (may be None)
        pattern: Glob pattern for state files

    Returns:
        List of copied file paths in the FEWS state directory

    Raises:
        StateExchangeError: If copy fails
    """
    if fews_state_dir is None:
        logger.debug("No FEWS state output directory specified, skipping export")
        return []

    model_state_dir = Path(model_state_dir)
    if not model_state_dir.is_dir():
        logger.debug("Model state directory does not exist: %s", model_state_dir)
        return []

    state_files = sorted(model_state_dir.glob(pattern))
    if not state_files:
        logger.debug("No state files matching '%s' in %s", pattern, model_state_dir)
        return []

    fews_state_dir = Path(fews_state_dir)
    try:
        fews_state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StateExchangeError(
            f"Cannot create FEWS state output directory {fews_state_dir}: {exc}"
        ) from exc

    copied: List[Path] = []
    for src in state_files:
        dst = fews_state_dir / src.name
        try:
            shutil.copy2(str(src), str(dst))
            copied.append(dst)
            logger.info("Exported state file: %s -> %s", src, dst)
        except OSError as exc:
            raise StateExchangeError(
                f"Failed to copy state file {src} -> {dst}: {exc}"
            ) from exc

    return copied
