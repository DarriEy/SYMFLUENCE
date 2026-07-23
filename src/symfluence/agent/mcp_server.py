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
import os

# Security rationale: runs the symfluence CLI itself, argv-built
import subprocess  # nosec B404
import sys
import tempfile
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
    """argv prefix that reaches the symfluence CLI from this environment."""
    from .priming import symfluence_invocation
    return symfluence_invocation()


def _read_tail(buf) -> str:
    """Read the last ``_MAX_OUTPUT_CHARS`` worth of a binary temp file."""
    buf.seek(0, os.SEEK_END)
    size = buf.tell()
    buf.seek(max(0, size - _MAX_OUTPUT_CHARS))
    return buf.read().decode('utf-8', errors='replace')


def _tool_run_workflow_step(arguments: dict) -> dict:
    """Run one workflow step as a subprocess and report its outcome.

    Output is streamed to a temp file and only the tail is returned, so an
    hours-long chatty step never grows this long-lived server's memory.
    """
    path = Path(arguments['config_path'])
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    step = arguments['step']
    timeout = int(arguments.get('timeout_seconds', _DEFAULT_STEP_TIMEOUT_S))

    argv = [*_symfluence_argv(), 'workflow', 'step', step, '--config', str(path)]
    if arguments.get('force_rerun'):
        argv.append('--force-rerun')

    with tempfile.TemporaryFile() as buf:
        try:
            # Security rationale: argv built above, no shell
            proc = subprocess.run(  # nosec B603
                argv, stdout=buf, stderr=subprocess.STDOUT, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                'ok': False,
                'error': f"Step {step!r} still running after {timeout}s; it was killed. "
                         f"Re-run with a larger timeout_seconds, or run it outside the "
                         f"agent: {' '.join(argv)}",
                'output': _read_tail(buf),
            }
        output = _read_tail(buf)
    return {'ok': proc.returncode == 0, 'exit_code': proc.returncode, 'output': output}


def _workflow_job_argv(arguments: dict) -> tuple[list[str], str]:
    """Build the ``symfluence workflow ...`` argv for a background job."""
    path = Path(arguments['config_path'])
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    mode = arguments.get('mode', 'run')
    steps = arguments.get('steps') or []

    if mode == 'run':
        verb_argv, label = ['run'], 'workflow run'
    elif mode == 'resume':
        verb_argv, label = ['resume'], 'workflow resume'
    elif mode == 'step':
        if len(steps) != 1:
            raise ValueError("mode 'step' needs exactly one entry in 'steps'")
        verb_argv, label = ['step', steps[0]], f'step {steps[0]}'
    elif mode == 'steps':
        if not steps:
            raise ValueError("mode 'steps' needs at least one entry in 'steps'")
        verb_argv, label = ['steps', *steps], f"steps {' '.join(steps)}"
    else:
        raise ValueError(f"Unknown mode {mode!r}; use run|step|steps|resume")

    argv = [*_symfluence_argv(), 'workflow', *verb_argv, '--config', str(path)]
    if arguments.get('force_rerun'):
        argv.append('--force-rerun')
    return argv, label


def _tool_start_workflow_job(arguments: dict) -> dict:
    """Start a workflow run/step in the background; returns a pollable job."""
    from . import jobs

    argv, label = _workflow_job_argv(arguments)
    record = jobs.start_job(argv, description=label)
    return {
        'job_id': record['job_id'],
        'pid': record['pid'],
        'log_path': record['log_path'],
        'description': label,
        'hint': 'Poll with get_job_status; cancel with cancel_job.',
    }


def _tool_get_job_status(arguments: dict) -> dict:
    """State + log tail of a background job."""
    from . import jobs

    state = jobs.job_state(arguments['job_id'])
    state['log_tail'] = jobs.tail_log(
        arguments['job_id'], int(arguments.get('log_tail_lines', 50)))
    return state


def _tool_cancel_job(arguments: dict) -> dict:
    """Cancel a background job (TERM, then KILL)."""
    from . import jobs
    return jobs.cancel_job(arguments['job_id'])


def _tool_list_jobs(arguments: dict) -> dict:
    """All recorded background jobs, newest first."""
    from . import jobs
    return {'jobs': jobs.list_jobs()}


def _tool_read_run_log(arguments: dict) -> dict:
    """Tail (and optionally grep) the newest run log for a config's domain."""
    from .inspection import read_run_log

    return read_run_log(
        arguments['config_path'],
        log_type=arguments.get('log_type', 'general'),
        tail_lines=int(arguments.get('tail_lines', 100)),
        grep=arguments.get('grep'),
    )


def _tool_list_domains(arguments: dict) -> dict:
    """domain_* directories under a data root, with a shape summary."""
    from .inspection import list_domains

    return list_domains(
        root_path=arguments.get('root_path'),
        config_path=arguments.get('config_path'),
    )


def _tool_calibration_status(arguments: dict) -> dict:
    """Progress of the newest (or named) calibration run."""
    from .inspection import calibration_status

    return calibration_status(
        arguments['config_path'], experiment_id=arguments.get('experiment_id'))


def _tool_get_results_summary(arguments: dict) -> dict:
    """Headline metrics and artifact locations for an experiment."""
    from .inspection import get_results_summary

    return get_results_summary(
        arguments['config_path'], experiment_id=arguments.get('experiment_id'))


def _tool_update_config(arguments: dict) -> dict:
    """Guarded config edit: validate, back up, apply."""
    from .inspection import update_config

    return update_config(
        arguments['config_path'],
        arguments.get('changes') or {},
        dry_run=bool(arguments.get('dry_run', False)),
    )


def _tool_get_plot_paths(arguments: dict) -> dict:
    """Figures produced for an experiment run."""
    from .inspection import get_plot_paths

    return get_plot_paths(
        arguments['config_path'], experiment_id=arguments.get('experiment_id'))


def _tool_compare_experiments(arguments: dict) -> dict:
    """All optimization runs of a domain, best score first."""
    from .inspection import compare_experiments

    return compare_experiments(arguments['config_path'])


def _tool_approve_action(arguments: dict) -> dict:
    """Permission-prompt bridge: ask the human in the TUI, blocking."""
    from .approvals import request_approval

    return request_approval(
        str(arguments.get('tool_name', 'unknown tool')),
        arguments.get('input') or {},
    )


_CONFIG_PATH_PROP = {
    'type': 'string',
    'description': 'Path to the SYMFLUENCE config YAML file.',
}

_JOB_ID_PROP = {
    'type': 'string',
    'description': 'Job id returned by start_workflow_job.',
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
            'a config, BLOCKING until it finishes. Prefer start_workflow_job '
            'for model runs and calibrations — they can take minutes to hours '
            'and this call holds the connection the whole time.'
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
    'start_workflow_job': {
        'handler': _tool_start_workflow_job,
        'description': (
            'Start a workflow execution as a detached background job and '
            'return immediately with a job_id. The job survives this server; '
            'poll it with get_job_status, watch read_run_log or '
            'calibration_status for progress, cancel with cancel_job. Use this '
            '(not run_workflow_step) for model runs and calibrations.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'mode': {
                    'type': 'string',
                    'enum': ['run', 'step', 'steps', 'resume'],
                    'description': "What to execute: the full pipeline ('run'), "
                                   "one step ('step'), several steps ('steps'), "
                                   "or resume a partial run ('resume').",
                },
                'steps': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': "Step name(s) for mode 'step'/'steps'.",
                },
                'force_rerun': {
                    'type': 'boolean',
                    'description': 'Re-run even if stage markers say complete.',
                },
            },
            'required': ['config_path', 'mode'],
        },
    },
    'get_job_status': {
        'handler': _tool_get_job_status,
        'description': (
            'State of a background job (running / succeeded / failed / '
            'cancelled), its runtime, and the tail of its log.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'job_id': _JOB_ID_PROP,
                'log_tail_lines': {
                    'type': 'integer',
                    'description': 'How many log lines to include (default 50).',
                },
            },
            'required': ['job_id'],
        },
    },
    'cancel_job': {
        'handler': _tool_cancel_job,
        'description': 'Cancel a background job: SIGTERM its process group, '
                       'then SIGKILL if it lingers.',
        'inputSchema': {
            'type': 'object',
            'properties': {'job_id': _JOB_ID_PROP},
            'required': ['job_id'],
        },
    },
    'list_jobs': {
        'handler': _tool_list_jobs,
        'description': 'All recorded background jobs and their states, newest '
                       'first (useful after a restart when job ids were lost).',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    'read_run_log': {
        'handler': _tool_read_run_log,
        'description': (
            "Tail the newest SYMFLUENCE run log for a config's domain "
            "(_workLog_* directory), optionally filtered to lines containing "
            "a substring. Use while a job runs to watch progress or after a "
            "failure to find the error."
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'log_type': {
                    'type': 'string',
                    'description': "Log family (default 'general').",
                },
                'tail_lines': {
                    'type': 'integer',
                    'description': 'How many trailing lines to return (default 100).',
                },
                'grep': {
                    'type': 'string',
                    'description': 'Only lines containing this substring.',
                },
            },
            'required': ['config_path'],
        },
    },
    'list_domains': {
        'handler': _tool_list_domains,
        'description': (
            'List the domain_* directories under a SYMFLUENCE data root '
            '(from root_path, or the config\'s SYMFLUENCE_DATA_DIR), with '
            'their experiments and whether simulations/optimization exist.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'root_path': {
                    'type': 'string',
                    'description': 'Data root to scan (overrides config_path).',
                },
                'config_path': _CONFIG_PATH_PROP,
            },
        },
    },
    'calibration_status': {
        'handler': _tool_calibration_status,
        'description': (
            'Progress of the newest (or named) calibration run for a '
            "config's domain: algorithm, iterations so far, best score and "
            'iteration, whether it is still in progress.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'experiment_id': {
                    'type': 'string',
                    'description': 'Pin a specific experiment (default: newest run).',
                },
            },
            'required': ['config_path'],
        },
    },
    'get_results_summary': {
        'handler': _tool_get_results_summary,
        'description': (
            'Headline metrics (KGE/NSE/...) and artifact locations (result '
            'CSVs, plots) for the newest or named experiment of a config\'s '
            'domain.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'experiment_id': {
                    'type': 'string',
                    'description': 'Pin a specific experiment (default: newest run).',
                },
            },
            'required': ['config_path'],
        },
    },
    'update_config': {
        'handler': _tool_update_config,
        'description': (
            "Change keys in the user's experiment config YAML, safely: the "
            'edit is validated with the typed config system first, the '
            'original is backed up next to itself, and comments/ordering are '
            'preserved. State the exact key changes to the user before '
            'calling. Set dry_run to preview.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'changes': {
                    'type': 'object',
                    'description': 'Mapping of top-level config keys to new values.',
                },
                'dry_run': {
                    'type': 'boolean',
                    'description': 'Validate and report without writing.',
                },
            },
            'required': ['config_path', 'changes'],
        },
    },
    'get_plot_paths': {
        'handler': _tool_get_plot_paths,
        'description': (
            'List the figures (png/pdf/svg) produced for the newest or named '
            "experiment run of a config's domain."
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'config_path': _CONFIG_PATH_PROP,
                'experiment_id': {
                    'type': 'string',
                    'description': 'Pin a specific experiment (default: newest run).',
                },
            },
            'required': ['config_path'],
        },
    },
    'compare_experiments': {
        'handler': _tool_compare_experiments,
        'description': (
            "Compare every optimization run of a config's domain: algorithm, "
            'metric, best score, iterations, completion — best score first.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {'config_path': _CONFIG_PATH_PROP},
            'required': ['config_path'],
        },
    },
    'approve_action': {
        'handler': _tool_approve_action,
        # The permission-prompt bridge: called by Claude Code itself (via
        # --permission-prompt-tool), never meant for the model to pick from
        # the tool list — hence hidden from tools/list.
        'hidden': True,
        'description': (
            'Ask the human in the SYMFLUENCE TUI to approve one tool use '
            '(permission-prompt bridge; blocks until answered or timeout).'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'tool_name': {'type': 'string'},
                'input': {'type': 'object'},
            },
            'required': ['tool_name'],
        },
    },
}


def tools_for_profile(profile: str | None) -> dict:
    """The subset of :data:`TOOLS` an agent-mode profile exposes.

    ``profile`` is an :class:`~symfluence.agent.modes.AgentMode` value; None
    (or a profile that declares no filter) exposes every registered tool.
    """
    if profile is None:
        return TOOLS
    from .modes import get_profile
    allowed = get_profile(profile).mcp_tools
    if allowed is None:
        return TOOLS
    return {name: spec for name, spec in TOOLS.items() if name in allowed}


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------

def _tools_list_result(tools: dict) -> dict:
    return {
        'tools': [
            {
                'name': name,
                'description': spec['description'],
                'inputSchema': spec['inputSchema'],
            }
            for name, spec in tools.items()
            if not spec.get('hidden')
        ]
    }


def _tools_call_result(params: dict, tools: dict) -> dict:
    name = params.get('name')
    spec = tools.get(name)
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


def _invalid_request(detail: str) -> dict:
    return {
        'jsonrpc': '2.0', 'id': None,
        'error': {'code': -32600, 'message': f"Invalid Request: {detail}"},
    }


def handle_message(message, tools: dict | None = None) -> dict | None:
    """Handle one JSON-RPC message; return the response, or None for notifications.

    ``tools`` is the tool set this server instance serves (default: all).
    """
    if tools is None:
        tools = TOOLS
    if not isinstance(message, dict):
        # Valid JSON but not a request object (e.g. a batch array or scalar).
        # Answer in-band — the server must survive anything on stdin.
        return _invalid_request('expected a JSON-RPC request object')

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
            result = _tools_list_result(tools)
        elif method == 'tools/call':
            result = _tools_call_result(params, tools)
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


def serve(stdin=None, stdout=None, profile: str | None = None) -> int:
    """Serve MCP over stdio until EOF. Blocks; used by `symfluence agent mcp`.

    ``profile`` restricts the tool set to one agent mode's profile
    (``--profile`` on the CLI); None serves every registered tool.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    tools = tools_for_profile(profile)

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
            response = handle_message(message, tools)
        if response is not None:
            stdout.write(json.dumps(response) + '\n')
            stdout.flush()
    return 0
