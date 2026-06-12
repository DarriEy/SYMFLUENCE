# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Acquisition-backend protocol (Phase A).

Layout:

* :mod:`~symfluence.data.backends.contract` — the versioned protocol types
  (stdlib-only; extraction-isolated, see the module docstring).
* :mod:`~symfluence.data.backends.errors` — the protocol error taxonomy
  (subclasses ``DataAcquisitionError`` in-tree).
"""
from __future__ import annotations

from symfluence.data.backends.contract import (
    MANIFEST_FILENAME,
    PROTOCOL_VERSION,
    AcquisitionBackend,
    AcquisitionRequest,
    AcquisitionResult,
    CredentialContext,
    DatasetCapability,
    GridClass,
    SchemaId,
)
from symfluence.data.backends.errors import (
    AcquisitionError,
    AuthRequired,
    DatasetUnsupported,
    IntegrityError,
    UpstreamOutage,
    WindowOutOfRange,
)

__all__ = [
    "PROTOCOL_VERSION",
    "MANIFEST_FILENAME",
    "GridClass",
    "SchemaId",
    "CredentialContext",
    "DatasetCapability",
    "AcquisitionRequest",
    "AcquisitionResult",
    "AcquisitionBackend",
    "AcquisitionError",
    "DatasetUnsupported",
    "AuthRequired",
    "WindowOutOfRange",
    "UpstreamOutage",
    "IntegrityError",
]
