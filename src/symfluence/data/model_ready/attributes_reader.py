# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility shim for the promoted canonical attributes reader."""
from __future__ import annotations

from symfluence.core.modeling.model_ready.attributes_reader import AttributesReader, open_canonical_attributes

__all__ = ["AttributesReader", "open_canonical_attributes"]
