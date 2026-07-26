# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the promoted dataset-alignment helper."""
from __future__ import annotations

from symfluence.core.modeling.utilities.dataset_alignment_manager import (
    DatasetAlignmentManager,
    align_forcing_datasets,
)

__all__ = ["DatasetAlignmentManager", "align_forcing_datasets"]
