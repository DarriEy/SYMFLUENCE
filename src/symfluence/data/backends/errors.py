# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
#
# NOTE: Apache-2.0, NOT GPL-3.0 — this error taxonomy ships alongside
# ``contract.py`` in the extraction-bound ``symfluence-data-interface``
# package and carries the same permissive license. See contract.py.

"""Protocol-owned, framework-actionable error taxonomy for acquisition backends.

Backends MUST map their internal failures onto these classes; the framework's
retry/fallback logic keys off the exception class, never message text. The
original internal exception is preserved as ``__cause__`` (``raise ... from``).

In-tree, :class:`AcquisitionError` subclasses the existing
``symfluence.core.exceptions.DataAcquisitionError`` so every current
``except DataAcquisitionError`` catch site keeps working unchanged.

EXTRACTION READINESS: like ``contract.py``, this module must remain loadable
without symfluence installed (it ships with the contract at extraction time).
The framework base class is therefore imported defensively — when symfluence
is unavailable the taxonomy degrades to plain ``Exception`` subclasses. This
is guard-tested by ``tests/conformance/test_extraction_readiness.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symfluence.core.exceptions import DataAcquisitionError as _FrameworkBase
else:
    try:
        from symfluence.core.exceptions import DataAcquisitionError as _FrameworkBase
    except ImportError:  # standalone / extracted usage: no symfluence available
        _FrameworkBase = Exception


class AcquisitionError(_FrameworkBase):
    """Base class for all acquisition-backend failures (protocol taxonomy root)."""


class DatasetUnsupported(AcquisitionError):
    """The backend cannot serve this dataset id."""

    def __init__(self, message: str, *, dataset_id: str | None = None, backend: str | None = None) -> None:
        super().__init__(message)
        self.dataset_id = dataset_id
        self.backend = backend


class AuthRequired(AcquisitionError):
    """Credentials are required (or invalid) for the upstream provider.

    Attributes:
        provider: Auth-provider id (matches ``DatasetCapability.auth`` /
            ``CredentialContext.providers`` keys, e.g. ``"earthdata"``).
        howto: Short, actionable hint on how to obtain/configure credentials.
    """

    def __init__(self, message: str, *, provider: str = "", howto: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.howto = howto


class WindowOutOfRange(AcquisitionError):
    """The requested time window falls outside the dataset's coverage.

    Attributes:
        coverage: The dataset's declared ``[start, end)`` coverage, if known.
    """

    def __init__(self, message: str, *, coverage: tuple[str, str] | None = None) -> None:
        super().__init__(message)
        self.coverage = coverage


class UpstreamOutage(AcquisitionError):
    """The upstream service is unavailable; the request is retryable.

    Attributes:
        upstream: Identifier of the upstream service that failed.
    """

    #: Framework retry logic keys off this flag (class-level, never message text).
    retryable = True

    def __init__(self, message: str, *, upstream: str = "") -> None:
        super().__init__(message)
        self.upstream = upstream


class IntegrityError(AcquisitionError):
    """Delivered data does not match what was requested/declared.

    The WSC-pagination class of bug: silent truncation, partial delivery,
    checksum mismatch. Honesty violations raise this, never a warning.
    """


__all__ = [
    "AcquisitionError",
    "DatasetUnsupported",
    "AuthRequired",
    "WindowOutOfRange",
    "UpstreamOutage",
    "IntegrityError",
]
