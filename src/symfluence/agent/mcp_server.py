# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The SYMFLUENCE MCP server: structured platform access for coding agents.

``symfluence agent mcp`` speaks the Model Context Protocol over stdio
(newline-delimited JSON-RPC 2.0), exposing the platform's registry, config
validation, and workflow engine as typed tools. ``symfluence agent launch``
wires it into the host CLI automatically (``--mcp-config`` for Claude Code), so
the agent can introspect and drive SYMFLUENCE without shelling out and parsing
human-oriented text.

Implemented by hand rather than on an SDK to keep SYMFLUENCE dependency-free:
the server only needs ``initialize``/``tools/list``/``tools/call``/``ping``.
"""
from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 — runs the symfluence CLI itself, argv-built
import sys
from pathlib import Path

PROTOCOL_VERSION = '2025-06-18'

# Cap the process output echoed back through a tool result.
_MAX_OUTPUT_CHARS = 20_000

_DEFAULT_STEP_TIMEOUT_S = 3600


def _server_info() -> dict:
    from symfluence import __version__
    return {'name': 'symfluence', 'version': __version__}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _tool_list_capabilities(arguments: dict) -> dict:
    """Live registry catalogs (models, forcings, optimizers, ...)."""
    from symfluence.cli.commands.list_commands import LIST_KINDS, _catalog

    catalog = _catalog()
    kind = arguments.get('kind')
    if kind is not None:
        if kind not in catalog:
            raise ValueError(f"Unknown kind {kind!r}. Choose from: {', '.join(LIST_KINDS)}")
        return {kind: catalog[kind]}
    return catalog


def _tool_validate_config(arguments: dict) -> dict:
    """Validate a config file with the typed SymfluenceConfig system."""
    from symfluence.core.config.models import SymfluenceConfig

    path = Path(arguments['config_path'])
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    try:
        config = SymfluenceConfig.from_file(path)
    except Exception as e:  # noqa: BLE001 — any load failure is the tool's answer
        return {'valid': False, 'error': str(e)}
    summary = {}
    for key in ('DOMAIN_NAME', 'EXPERIMENT_ID', 'HYDROLOGICAL_MODEL', 'FORCING_DATASET'):
        value = config.get(key)
        if value is not None:
            summary[key] = str(value)
    return {'valid': True, 'summary': summary}


def _tool_workflow_status(arguments: dict) -> dict:
    """Per-step workflow status for a config (done / pending / stale)."""
    from symfluence import SYMFLUENCE

    path = Path(arguments['config_path'])
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    instance = SYMFLUENCE(str(path))
    return {'status': instance.get_workflow_status()}


def _symfluence_argv() -> list[str]:
    """argv prefix that reaches the symfluence CLI from this interpreter."""
    binary = shutil.which('symfluence')
    if binary:
        return [binary]
    return [sys.executable, '-m', 'symfluence']


def _tool_run_workflow_step(arguments: dict) -> dict:
    """Run one workflow step as a subprocess and report its outcome."""
    path = Path(arguments['config_path'])
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    step = arguments['step']
    timeout = int(arguments.get('timeout_seconds', _DEFAULT_STEP_TIMEOUT_S))

    argv = [*_symfluence_argv(), 'workflow', 'step', step, '--config', str(path)]
    if arguments.get('force_rerun'):
        argv.append('--force-rerun')

    try:
        proc = subprocess.run(  # nosec B603 — argv built above, no shell
            argv, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            'ok': False,
            'error': f"Step {step!r} still running after {timeout}s; it was killed. "
                     f"Re-run with a larger timeout_seconds, or run it outside the "
                     f"agent: {' '.join(argv)}",
        }
    output = (proc.stdout + proc.stderr)[-_MAX_OUTPUT_CHARS:]
    return {'ok': proc.returncode == 0, 'exit_code': proc.returncode, 'output': output}


_CONFIG_PATH_PROP = {
    'type': 'string',
    'description': 'Path to the SYMFLUENCE config YAML file.',
}

TOOLS = {
    'list_capabilities': {
        'handler': _tool_list_capabilities,
        'description': (
            'List what this SYMFLUENCE install supports, read live from the '
            'registry: models, forcings, observations, optimizers, targets, '
            'metrics, presets, templates, steps, config-keys. Optionally filter '
            'to one catalog kind.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'kind': {
                    'type': 'string',
                    'description': "One catalog to return (e.g. 'models'); omit for all.",
                },
            },
        },
    },
    'validate_config': {
        'handler': _tool_validate_config,
        'description': (
            'Validate a SYMFLUENCE config file with the typed config system. '
            'Returns valid/invalid plus the key experiment fields. Run this '
            'before any workflow execution.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {'config_path': _CONFIG_PATH_PROP},
            'required': ['config_path'],
        },
    },
    'workflow_status': {
        'handler': _tool_workflow_status,
        'description': (
            'Report per-step workflow status for a config: which of the 16 '
            'pipeline steps are complete, pending, or stale.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {'config_path': _CONFIG_PATH_PROP},
            'required': ['config_path'],
        },
    },
    'run_workflow_step': {
        'handler': _tool_run_workflow_step,
        'description': (
            'Run a single SYMFLUENCE workflow step (see the steps catalog) for '
            'a config. Long-running: model runs and calibrations can take '
            'minutes to hours — set timeout_seconds accordingly.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'step': {
                    'type': 'string',
                    'description': "Step name or alias (e.g. 'model_run', 'calibrate').",
                },
                'force_rerun': {
                    'type': 'boolean',
                    'description': 'Re-run even if the stage marker says it is complete.',
                },
                'timeout_seconds': {
                    'type': 'integer',
                    'description': f'Kill the step after this long (default {_DEFAULT_STEP_TIMEOUT_S}).',
                },
            },
            'required': ['config_path', 'step'],
        },
    },
}


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------

def _tools_list_result() -> dict:
    return {
        'tools': [
            {
                'name': name,
                'description': spec['description'],
                'inputSchema': spec['inputSchema'],
            }
            for name, spec in TOOLS.items()
        ]
    }


def _tools_call_result(params: dict) -> dict:
    name = params.get('name')
    spec = TOOLS.get(name)
    if spec is None:
        raise ValueError(f"Unknown tool: {name!r}")
    try:
        payload = spec['handler'](params.get('arguments') or {})
    except Exception as e:  # noqa: BLE001 — tool errors go in-band per MCP spec
        return {
            'content': [{'type': 'text', 'text': f"{type(e).__name__}: {e}"}],
            'isError': True,
        }
    return {
        'content': [{'type': 'text', 'text': json.dumps(payload, indent=2, default=str)}],
        'isError': False,
    }


def handle_message(message: dict) -> dict | None:
    """Handle one JSON-RPC message; return the response, or None for notifications."""
    method = message.get('method')
    msg_id = message.get('id')
    params = message.get('params') or {}

    if msg_id is None:  # notification (e.g. notifications/initialized) — no reply
        return None

    try:
        if method == 'initialize':
            result = {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {'tools': {}},
                'serverInfo': _server_info(),
            }
        elif method == 'ping':
            result = {}
        elif method == 'tools/list':
            result = _tools_list_result()
        elif method == 'tools/call':
            result = _tools_call_result(params)
        else:
            return {
                'jsonrpc': '2.0', 'id': msg_id,
                'error': {'code': -32601, 'message': f"Method not found: {method}"},
            }
    except Exception as e:  # noqa: BLE001 — the server must answer, not crash
        return {
            'jsonrpc': '2.0', 'id': msg_id,
            'error': {'code': -32603, 'message': f"{type(e).__name__}: {e}"},
        }
    return {'jsonrpc': '2.0', 'id': msg_id, 'result': result}


def serve(stdin=None, stdout=None) -> int:
    """Serve MCP over stdio until EOF. Blocks; used by `symfluence agent mcp`."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = {
                'jsonrpc': '2.0', 'id': None,
                'error': {'code': -32700, 'message': 'Parse error'},
            }
        else:
            response = handle_message(message)
        if response is not None:
            stdout.write(json.dumps(response) + '\n')
            stdout.flush()
    return 0
