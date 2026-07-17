# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the SYMFLUENCE MCP server (protocol layer; no heavy tool runs)."""
from __future__ import annotations

import io
import json

from symfluence.agent import mcp_server


def _request(method, msg_id=1, **params):
    return {'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params}


def test_initialize_reports_server_info():
    response = mcp_server.handle_message(_request('initialize'))
    result = response['result']
    assert result['serverInfo']['name'] == 'symfluence'
    assert result['protocolVersion'] == mcp_server.PROTOCOL_VERSION
    assert 'tools' in result['capabilities']


def test_notifications_get_no_reply():
    message = {'jsonrpc': '2.0', 'method': 'notifications/initialized'}
    assert mcp_server.handle_message(message) is None


def test_tools_list_exposes_all_tools():
    response = mcp_server.handle_message(_request('tools/list'))
    names = {tool['name'] for tool in response['result']['tools']}
    assert names == set(mcp_server.TOOLS)
    for tool in response['result']['tools']:
        assert tool['description']
        assert tool['inputSchema']['type'] == 'object'


def test_unknown_method_returns_jsonrpc_error():
    response = mcp_server.handle_message(_request('resources/list'))
    assert response['error']['code'] == -32601


def test_tool_error_is_reported_in_band():
    response = mcp_server.handle_message(
        _request('tools/call', name='list_capabilities',
                 arguments={'kind': 'nonsense'})
    )
    result = response['result']
    assert result['isError'] is True
    assert 'nonsense' in result['content'][0]['text']


def test_validate_config_tool_rejects_missing_file():
    response = mcp_server.handle_message(
        _request('tools/call', name='validate_config',
                 arguments={'config_path': '/does/not/exist.yaml'})
    )
    assert response['result']['isError'] is True


def test_serve_speaks_newline_delimited_jsonrpc():
    stdin = io.StringIO(
        json.dumps(_request('initialize')) + '\n'
        + json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n'
        + json.dumps(_request('ping', msg_id=2)) + '\n'
        + 'this is not json\n'
    )
    stdout = io.StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(responses) == 3  # notification produced no reply
    assert responses[0]['result']['serverInfo']['name'] == 'symfluence'
    assert responses[1]['result'] == {}
    assert responses[2]['error']['code'] == -32700


def test_non_object_json_gets_error_not_crash():
    """A JSON array/scalar on stdin must be answered in-band, never crash serve."""
    for payload in ([], [_request('ping')], 5, 'x'):
        response = mcp_server.handle_message(payload)
        assert response['error']['code'] == -32600

    stdin = io.StringIO('[]\n5\n' + json.dumps(_request('ping', msg_id=9)) + '\n')
    stdout = io.StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(responses) == 3  # server survived the junk and answered the ping
    assert responses[2]['result'] == {}


def test_profile_filter_mechanics():
    """A restricted tool set hides tools from list and call alike."""
    restricted = {'validate_config': mcp_server.TOOLS['validate_config']}

    listed = mcp_server.handle_message(_request('tools/list'), restricted)
    assert [t['name'] for t in listed['result']['tools']] == ['validate_config']

    called = mcp_server.handle_message(
        _request('tools/call', name='list_capabilities', arguments={}),
        restricted,
    )
    assert 'Unknown tool' in called['error']['message']


def test_tools_for_profile_resolves_mode_profiles():
    # Both current profiles expose every tool (mcp_tools=None); the mechanism
    # must still resolve them and reject unknown profile names.
    assert mcp_server.tools_for_profile(None) == mcp_server.TOOLS
    assert mcp_server.tools_for_profile('model') == mcp_server.TOOLS
    assert mcp_server.tools_for_profile('code') == mcp_server.TOOLS
    import pytest
    with pytest.raises(ValueError):
        mcp_server.tools_for_profile('bogus')


def test_run_workflow_step_output_is_bounded(tmp_path, monkeypatch):
    """Step output is spooled to disk and only the tail is returned."""
    config = tmp_path / 'c.yaml'
    config.write_text('DOMAIN_NAME: x\n', encoding='utf-8')

    import sys
    monkeypatch.setattr(
        mcp_server, '_symfluence_argv',
        lambda: [sys.executable, '-c',
                 "print('x' * 100000); print('TAIL-MARKER')"],
    )
    result = mcp_server._tool_run_workflow_step(
        {'config_path': str(config), 'step': 'noop'})
    assert result['ok'] is True
    assert len(result['output']) <= mcp_server._MAX_OUTPUT_CHARS + 1
    assert 'TAIL-MARKER' in result['output']
