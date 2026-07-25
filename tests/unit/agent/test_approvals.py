# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the file-based approval bridge (permission-prompt tool)."""
from __future__ import annotations

import threading
import time

import pytest

from symfluence.agent import approvals


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path / 'cache'))


def _answer_when_pending(approved: bool, message: str = ''):
    """Background thread playing the TUI: replies to the first pending request.

    Polls well past the requester's window: equal-magnitude timeouts made the
    denial reply land just after the deadline on loaded macOS CI runners.
    """
    def _run():
        deadline = time.time() + 60
        while time.time() < deadline:
            pending = approvals.list_pending()
            if pending:
                approvals.reply(pending[0]['id'], approved, message)
                return
            time.sleep(0.05)
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def test_approved_request_allows_with_original_input():
    _answer_when_pending(True)
    verdict = approvals.request_approval(
        'Edit', {'file_path': 'config.yaml'}, timeout_s=30)
    assert verdict == {'behavior': 'allow',
                       'updatedInput': {'file_path': 'config.yaml'}}
    assert approvals.list_pending() == []  # cleaned up


def test_denied_request_carries_message():
    _answer_when_pending(False, 'not on my watch')
    verdict = approvals.request_approval('Bash', {'command': 'rm -rf /'},
                                         timeout_s=30)
    assert verdict['behavior'] == 'deny'
    assert 'not on my watch' in verdict['message']


def test_timeout_denies():
    verdict = approvals.request_approval('Edit', {}, timeout_s=0.4)
    assert verdict['behavior'] == 'deny'
    assert 'No approval' in verdict['message']
    assert approvals.list_pending() == []


def test_list_pending_shows_request_payload():
    done = threading.Event()

    def _requester():
        approvals.request_approval('WebFetch', {'url': 'https://x'}, timeout_s=5)
        done.set()

    threading.Thread(target=_requester, daemon=True).start()
    deadline = time.time() + 5
    pending = []
    while time.time() < deadline and not pending:
        pending = approvals.list_pending()
        time.sleep(0.05)
    assert pending and pending[0]['tool_name'] == 'WebFetch'
    assert pending[0]['input'] == {'url': 'https://x'}
    approvals.reply(pending[0]['id'], False)
    assert done.wait(5)


def test_approve_action_tool_is_hidden_but_callable():
    from symfluence.agent import mcp_server

    listed = mcp_server.handle_message(
        {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list', 'params': {}})
    names = {t['name'] for t in listed['result']['tools']}
    assert 'approve_action' not in names       # hidden from the model
    assert 'get_plot_paths' in names
    assert 'compare_experiments' in names
    assert 'approve_action' in mcp_server.TOOLS  # but callable as the bridge
