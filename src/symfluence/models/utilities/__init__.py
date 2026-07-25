# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: moved to ``symfluence.core.modeling.utilities``.

Also restores two historical re-exports that now live with the data layer
(they could not travel to core: core must not import data at module level).
"""
from __future__ import annotations

import symfluence.core.modeling.utilities as _impl
from symfluence.core.modeling.utilities import *  # noqa: F401,F403
from symfluence.data.preprocessing.dataset_alignment_manager import (  # noqa: F401
    DatasetAlignmentManager,
    align_forcing_datasets,
)
from symfluence.data.preprocessing.time_window_manager import TimeWindowManager  # noqa: F401


def __getattr__(name: str):
    return getattr(_impl, name)
