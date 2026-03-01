# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tree widget for nested YAML config display.
"""

from typing import Any, Dict

from textual.widgets import Tree


class ConfigTreeWidget(Tree):
    """Renders a nested dict (parsed YAML config) as an expandable tree."""

    def __init__(self, label: str = "Configuration", **kwargs):
        super().__init__(label, **kwargs)

    def load_config(self, config: Dict[str, Any]) -> None:
        """Build tree nodes from a config dictionary."""
        self.clear()
        self._add_nodes(self.root, config)
        self.root.expand()

    def _add_nodes(self, parent, data: Any, depth: int = 0) -> None:
        """Recursively add tree nodes."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    node = parent.add(str(key))
                    self._add_nodes(node, value, depth + 1)
                else:
                    parent.add_leaf(f"{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    node = parent.add(f"[{i}]")
                    self._add_nodes(node, item, depth + 1)
                else:
                    parent.add_leaf(f"[{i}]: {item}")
        else:
            parent.add_leaf(str(data))
