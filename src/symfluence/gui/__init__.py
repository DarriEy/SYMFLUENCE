# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE GUI Package.

Panel-based web application for interactive hydrological modeling workflows.
Provides map interaction, configuration editing, workflow execution, and results viewing.

Install dependencies: pip install "symfluence[gui]"
Launch: symfluence gui launch
"""

#: Bind addresses that keep the GUI reachable only from the local machine.
from __future__ import annotations

_LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback_address(address: str) -> bool:
    """Return True if *address* keeps the server local-only (loopback)."""
    return address in _LOOPBACK_ADDRESSES


def serve_app(config_path=None, port=5006, show=True, demo=None, address="127.0.0.1"):
    """
    Build and serve the SYMFLUENCE GUI as a Panel web application.

    Thin wrapper that imports the actual server module lazily so that
    ``from symfluence.gui import serve_app`` succeeds even when Panel
    is not installed (the ImportError is raised only when called).

    Args:
        config_path: Optional path to a YAML config file to preload.
        port: Server port (default 5006).
        show: Open a browser tab automatically.
        demo: Optional demo name (e.g. 'bow') to load a built-in config.
        address: Interface to bind (default 127.0.0.1, loopback only).
    """
    try:
        import panel  # noqa: F401
    except ImportError:
        raise ImportError(
            "Panel is required for the SYMFLUENCE GUI.\n"
            'Install with:  pip install "symfluence[gui]"'
        ) from None

    from .server import _serve_app
    _serve_app(config_path=config_path, port=port, show=show, demo=demo, address=address)


__all__ = ['serve_app', 'is_loopback_address']
