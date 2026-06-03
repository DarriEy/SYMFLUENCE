# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The Panel GUI binds loopback by default; non-loopback is opt-in (review item 12)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.gui import is_loopback_address

# --- panel-free: the bind decision (always runs) -------------------------


@pytest.mark.parametrize("addr", ["127.0.0.1", "localhost", "::1"])
def test_loopback_addresses(addr):
    assert is_loopback_address(addr) is True


@pytest.mark.parametrize("addr", ["0.0.0.0", "192.168.1.5", "10.0.0.1", ""])  # nosec B104
def test_non_loopback_addresses(addr):
    assert is_loopback_address(addr) is False


# --- integration: kwargs passed to pn.serve (needs Panel) ----------------


def _call_serve(**kwargs):
    """Run _serve_app with Panel fully mocked; return the pn.serve kwargs.

    Skips when Panel (the optional [gui] extra) is not installed.
    """
    pytest.importorskip("panel")
    from symfluence.gui import server

    with patch.object(server, "pn", MagicMock()) as mock_pn:
        server._serve_app(**kwargs)
    _, serve_kwargs = mock_pn.serve.call_args
    return serve_kwargs


def test_default_binds_loopback():
    serve_kwargs = _call_serve()
    assert serve_kwargs["address"] == "127.0.0.1"
    assert "websocket_origin" not in serve_kwargs


def test_nonloopback_sets_websocket_origin_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="symfluence.gui.server"):
        serve_kwargs = _call_serve(address="0.0.0.0", port=9999)  # nosec B104 - test input
    assert serve_kwargs["address"] == "0.0.0.0"  # nosec B104
    assert serve_kwargs["websocket_origin"] == "0.0.0.0:9999"  # nosec B104
    assert any("reachable from other machines" in r.getMessage() for r in caplog.records)
