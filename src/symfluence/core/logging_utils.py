# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared logging utilities for the SYMFLUENCE logging protocol.

This module is the single home for the pieces of the logging protocol that
multiple subsystems need to agree on:

- ``FILE_FORMAT`` / ``FILE_FORMAT_DEBUG``: the canonical log-file record
  formats (used with :class:`ShortNameFilter`, which supplies ``%(sname)s``).
- :class:`ShortNameFilter`: injects a compact logger name (``symfluence.``
  prefix stripped) as ``record.sname``.
- :func:`log_once`: de-duplicated emission for messages that would otherwise
  repeat every iteration/file (first occurrence at the requested level,
  DEBUG thereafter).
- ``THIRD_PARTY_LOGGER_LEVELS`` / :func:`silence_third_party`: the one table
  of noisy third-party loggers and the helper that applies it.
- :func:`get_worker_logger`: logger bootstrap for spawned worker processes
  that have no configured ``symfluence`` root of their own.

Level policy: see ``docs/adr/0005-logging-level-policy.md`` (ERROR is the
operational ceiling; CRITICAL is reserved for whole-run aborts).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

__all__ = [
    'FILE_FORMAT',
    'FILE_FORMAT_DEBUG',
    'FILE_DATEFMT',
    'ShortNameFilter',
    'log_once',
    'reset_log_once',
    'THIRD_PARTY_LOGGER_LEVELS',
    'silence_third_party',
    'get_worker_logger',
]

#: Canonical log-file record format. Requires :class:`ShortNameFilter` on the
#: handler to supply ``%(sname)s``.
FILE_FORMAT = '%(asctime)s %(levelname)-7s [%(sname)s] %(message)s'

#: Log-file record format for debug mode: adds the emitting code location.
FILE_FORMAT_DEBUG = (
    '%(asctime)s %(levelname)-7s [%(sname)s] %(module)s.%(funcName)s: %(message)s'
)

#: Date format used with both file formats.
FILE_DATEFMT = '%Y-%m-%d %H:%M:%S'


class ShortNameFilter(logging.Filter):
    """
    Injects ``record.sname``: the logger name with a leading ``symfluence.``
    stripped (the ``symfluence`` root itself keeps its full name).

    Attach to handlers whose formatter uses ``%(sname)s`` (e.g.
    :data:`FILE_FORMAT`). Always returns True — this filter never drops
    records.
    """

    _PREFIX = 'symfluence.'

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name.startswith(self._PREFIX):
            record.sname = name[len(self._PREFIX):]
        else:
            # Covers the 'symfluence' root itself and any foreign loggers.
            record.sname = name
        return True


# --------------------------------------------------------------------------
# log_once: de-duplicated emission
# --------------------------------------------------------------------------

_log_once_seen: set = set()
_log_once_lock = threading.Lock()


def log_once(logger: logging.Logger, level: int, key: str, message: str) -> bool:
    """
    Emit ``message`` at ``level`` the first time ``key`` is seen in this
    process, and at DEBUG on every subsequent call with the same key.

    Thread-safe. Returns True if the message was emitted at ``level``
    (i.e. this was the first occurrence), False otherwise.

    Args:
        logger: Logger to emit through.
        level: Logging level (e.g. ``logging.WARNING``) for the first emission.
        key: Process-wide de-duplication key.
        message: The message to log.
    """
    # Lock-free fast path: hot loops hit already-seen keys on every
    # iteration; a set lookup under the GIL is safe without the lock.
    if key in _log_once_seen:
        logger.log(logging.DEBUG, message)
        return False
    with _log_once_lock:
        first = key not in _log_once_seen
        if first:
            _log_once_seen.add(key)
    logger.log(level if first else logging.DEBUG, message)
    return first


def reset_log_once() -> None:
    """Clear the :func:`log_once` seen-key registry (intended for tests)."""
    with _log_once_lock:
        _log_once_seen.clear()


# --------------------------------------------------------------------------
# Third-party logger suppression
# --------------------------------------------------------------------------

#: The single table of noisy third-party loggers and the level they are
#: capped at. Apply with :func:`silence_third_party`.
THIRD_PARTY_LOGGER_LEVELS = {
    name: logging.WARNING
    for name in (
        'matplotlib',
        'rasterio',
        'fiona',
        'urllib3',
        'requests',
        'botocore',
        'boto3',
        's3transfer',
        'easymore',
        'easymorepy',
        'rpy2',
        'numba',
        'distributed',
        'py4j',
    )
}


def silence_third_party() -> None:
    """Apply :data:`THIRD_PARTY_LOGGER_LEVELS` to the named loggers."""
    for name, level in THIRD_PARTY_LOGGER_LEVELS.items():
        logging.getLogger(name).setLevel(level)


# --------------------------------------------------------------------------
# Worker-process logger bootstrap
# --------------------------------------------------------------------------

def _has_real_handlers(logger: logging.Logger) -> bool:
    """True if ``logger`` has any handler other than a NullHandler."""
    return any(
        not isinstance(handler, logging.NullHandler)
        for handler in logger.handlers
    )


def get_worker_logger(
    worker_id: int,
    individual_id: Optional[int] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Return the logger for a (possibly spawned) worker process.

    Naming: ``symfluence.worker.P{worker_id:02d}`` with an
    ``.I{individual_id:03d}`` suffix when ``individual_id`` is given, so
    worker records participate in the ``symfluence`` hierarchy.

    In a true spawned subprocess — where neither this logger nor the
    ``symfluence`` root has any (non-Null) handler — a StreamHandler with a
    compact ``[P##[-I###]] LEVEL: message`` format is attached once,
    idempotently. In the parent process (root already configured) the logger
    simply propagates to the root's handlers.

    Args:
        worker_id: Numeric worker/process identifier.
        individual_id: Optional per-individual identifier (e.g. a population
            member in an evolutionary optimizer).
        level: Explicit level for this logger; defaults to the ``symfluence``
            root's effective level when that root is configured, else INFO
            (in a spawned subprocess the unconfigured root would otherwise
            fall through to the stdlib default of WARNING and silence
            workers' INFO output).
    """
    name = f'symfluence.worker.P{worker_id:02d}'
    prefix = f'P{worker_id:02d}'
    if individual_id is not None:
        name += f'.I{individual_id:03d}'
        prefix += f'-I{individual_id:03d}'

    logger = logging.getLogger(name)
    root = logging.getLogger('symfluence')

    if not _has_real_handlers(logger) and not _has_real_handlers(root):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f'[{prefix}] %(levelname)s: %(message)s')
        )
        logger.addHandler(handler)
        # The root has no handlers to duplicate through today, but if this
        # process later configures it we must not emit each record twice.
        logger.propagate = False

    if level is None:
        # root.level is NOTSET when no setup_logging ran in this process
        # (spawned subprocess): getEffectiveLevel() would fall through to the
        # stdlib root default (WARNING), silencing worker INFO output.
        level = root.getEffectiveLevel() if root.level != logging.NOTSET else logging.INFO
    logger.setLevel(level)
    return logger
