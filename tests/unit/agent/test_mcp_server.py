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
