# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Discovery of externally-registered attribute processors.

Third-party packages can contribute attribute processors without modifying
SYMFLUENCE by advertising them under the ``symfluence.attribute_processors``
entry-point group, e.g. in their ``pyproject.toml``::

    [project.entry-points."symfluence.attribute_processors"]
    climate_classification = "climaclass.integrations.symfluence:ClimateClassificationProcessor"

Each entry point must resolve to a subclass of
:class:`~symfluence.data.preprocessing.attribute_processors.base.BaseAttributeProcessor`
(constructed with ``(config, logger)`` and exposing ``.process() -> dict``).

Discovered plugins run during model-agnostic preprocessing under
``DATA_ACCESS: community`` — the data manager's
``_run_community_attribute_pipeline`` invokes
:func:`symfluence.data.preprocessing.attribute_processor.attributeProcessor._process_plugin_attributes`,
whose results are written to ``data/attributes/community/`` and ingested by
the ``AttributesNetCDFBuilder`` as a ``community`` group. They also run inside
``attributeProcessor.process_attributes()`` for callers that use the full
in-memory attribute table. Control via ``ATTRIBUTE_PLUGINS_ENABLED`` /
``ATTRIBUTE_PLUGINS_EXCLUDE``.
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import List, Tuple, Type

from .base import BaseAttributeProcessor

ENTRY_POINT_GROUP = "symfluence.attribute_processors"


def discover_attribute_plugins(
    logger: logging.Logger | None = None,
) -> List[Tuple[str, Type[BaseAttributeProcessor]]]:
    """Return ``(name, processor_class)`` for every registered plugin.

    Failures to load an individual entry point are logged and skipped so a
    broken third-party package never breaks attribute processing.
    """
    plugins: List[Tuple[str, Type[BaseAttributeProcessor]]] = []

    try:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 selection API
        entry_points = metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    for ep in entry_points:
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break the run
            if logger:
                logger.warning(f"Could not load attribute plugin '{ep.name}': {exc}")
            continue

        if not (isinstance(cls, type) and issubclass(cls, BaseAttributeProcessor)):
            if logger:
                logger.warning(
                    f"Attribute plugin '{ep.name}' does not subclass BaseAttributeProcessor; skipping"
                )
            continue

        plugins.append((ep.name, cls))

    return plugins
