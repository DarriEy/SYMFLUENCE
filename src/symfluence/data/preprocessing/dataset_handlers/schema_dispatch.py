# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Declared-schema dataset-handler dispatch (AcquisitionBackend protocol, Phase A).

Protocol backends DECLARE what they produced: ``acquire()`` writes a sidecar
``acquisition_manifest.json`` next to the raw forcing files, carrying the
:class:`~symfluence.data.backends.contract.SchemaId` of the output. Forcing
preprocessing dispatches its dataset handler off that declared schema instead
of sniffing file contents:

* manifest present with a non-``native-raw`` schema (e.g. ``canonical-v1``)
  → the handler registered under the SCHEMA key serves (one handler per
  schema, registered once by the producing plugin — e.g. the cfs package
  registers ``R.dataset_handlers['canonical-v1']``);
* manifest absent (legacy native acquisitions never write one) or schema
  ``native-raw`` → the existing per-dataset lookup, byte-for-byte unchanged.

A declared schema with no registered handler is a hard error: silently
falling back to the per-dataset handler would mis-process canonical files
through native unit heuristics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from symfluence.data.preprocessing.dataset_handlers.dataset_registry import DatasetRegistry

#: Schema id marking the legacy per-dataset raw layout (dispatches per dataset).
_NATIVE_RAW = 'native-raw'


def read_declared_schema(project_dir: Path, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Return the schema declared by the acquisition manifest, if any.

    Looks for the sidecar manifest in the project's raw forcing directory
    (``data/forcing/raw_data`` with the legacy ``forcing/raw_data`` fallback).
    Returns ``None`` when no manifest exists — the legacy-native signal.

    Raises:
        ValueError: when a manifest exists but cannot be read/parsed (a
            corrupt manifest must not silently demote canonical files to the
            native heuristics path).
    """
    from symfluence.data.backends.contract import MANIFEST_FILENAME

    project_dir = Path(project_dir)
    for raw_dir in (
        project_dir / 'data' / 'forcing' / 'raw_data',
        project_dir / 'forcing' / 'raw_data',
    ):
        manifest_path = raw_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
            schema = payload.get('schema')
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(
                f"Unreadable acquisition manifest at {manifest_path}: {exc}. "
                f"Delete the raw forcing directory and re-acquire, or restore "
                f"a valid manifest — preprocessing refuses to guess the file "
                f"schema."
            ) from exc
        if logger:
            logger.debug(f"Acquisition manifest at {manifest_path} declares schema '{schema}'")
        return str(schema) if schema else None
    return None


def resolve_dataset_handler(
    forcing_dataset: str,
    config: Any,
    logger: Any,
    project_dir: Path,
    **kwargs: Any,
):
    """Resolve the preprocessing dataset handler via declared-schema dispatch.

    Args:
        forcing_dataset: Configured forcing dataset name (per-dataset fallback key).
        config: Framework config (dict or SymfluenceConfig).
        logger: Logger handed to the handler.
        project_dir: Project root (manifest lookup + handler argument).
        **kwargs: Extra handler arguments (e.g. ``forcing_timestep_seconds``).

    Returns:
        An instantiated dataset handler.

    Raises:
        ValueError: when the manifest declares a schema with no registered
            handler (the producing plugin is not installed), or when no
            handler exists for the dataset on the legacy path.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    schema = read_declared_schema(project_dir, logger=log)

    if schema and schema != _NATIVE_RAW:
        from symfluence.core.registries import R

        if R.dataset_handlers.get(schema) is None:
            raise ValueError(
                f"The raw forcing data declares schema '{schema}' (acquisition "
                f"manifest), but no dataset handler is registered under that "
                f"schema key. Install the plugin that produced the data "
                f"(e.g. community-forcing-service for 'canonical-v1'), or "
                f"re-acquire natively."
            )
        log.info(
            f"Dispatching forcing preprocessing on declared schema '{schema}' "
            f"(dataset {forcing_dataset})"
        )
        return DatasetRegistry.get_handler(schema, config, log, project_dir, **kwargs)

    return DatasetRegistry.get_handler(forcing_dataset, config, log, project_dir, **kwargs)


__all__ = ["read_declared_schema", "resolve_dataset_handler"]
