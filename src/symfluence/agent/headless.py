# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Drive a headless Claude Code session over its stream-JSON protocol.

The native modelling chat runs the ``claude`` CLI non-interactively, one
subprocess per user turn::

    claude -p "<prompt>" --output-format stream-json --include-partial-messages
           --verbose [--resume <session_id>] <priming argv> <permission flags>

and parses the NDJSON events it emits into the typed events below. Session
state lives in Claude Code's own session store — each turn resumes the last
``session_id`` — so interrupting a turn is simply killing its subprocess, and
resuming after a TUI restart is free.

The parser is deliberately tolerant (unknown event types and fields are
ignored) because the stream-JSON schema evolves; anything unparseable becomes
a :class:`DriverError` event rather than an exception. Stdlib-only by design —
the official SDK could replace the transport behind the same event types.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from .modes import AgentMode, get_profile
from .registry import AgentLauncher

# ---------------------------------------------------------------- events

@dataclass(frozen=True)
class SessionInit:
    """First event of a turn: the session exists and tools are wired."""

    session_id: str
    tools: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextDelta:
    """A streamed fragment of assistant prose (partial messages on)."""

    text: str


@dataclass(frozen=True)
class AssistantText:
    """A complete assistant text block."""

    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    """The assistant invoked a tool."""

    tool_id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool returned (or failed)."""

    tool_id: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class TurnResult:
    """Terminal event of a turn."""

    session_id: str | None
    is_error: bool
    text: str
    duration_s: float | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class DriverError:
    """The driver itself failed (spawn error, undecodable stream, bad exit)."""

    message: str


# ----------------------------------------------------------------- parsing

def _content_blocks(message: dict):
    content = (message or {}).get('content')
    return content if isinstance(content, list) else []


def parse_stream_event(line: str):
    """Parse one NDJSON line into a list of events (empty for noise).

    Tolerant by contract: unknown types/fields yield nothing; only an
    undecodable line yields a :class:`DriverError`.
    """
    line = line.strip()
    if not line:
        return []
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return [DriverError(f"Undecodable stream line: {line[:120]!r}")]
    if not isinstance(data, dict):
        return []

    kind = data.get('type')

    if kind == 'system' and data.get('subtype') == 'init':
        return [SessionInit(
            session_id=str(data.get('session_id', '')),
            tools=tuple(data.get('tools') or ()),
            mcp_servers=tuple(
                server.get('name', '?') for server in data.get('mcp_servers') or ()
                if isinstance(server, dict)
            ),
        )]

    if kind == 'stream_event':
        event = data.get('event') or {}
        delta = event.get('delta') or {}
        if delta.get('type') == 'text_delta' and delta.get('text'):
            return [TextDelta(delta['text'])]
        return []

    if kind == 'assistant':
        events = []
        for block in _content_blocks(data.get('message')):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'text' and block.get('text'):
                events.append(AssistantText(block['text']))
            elif block.get('type') == 'tool_use':
                events.append(ToolCallStarted(
                    tool_id=str(block.get('id', '')),
                    name=str(block.get('name', '?')),
                    arguments=block.get('input') or {},
                ))
        return events

    if kind == 'user':
        events = []
        for block in _content_blocks(data.get('message')):
            if not isinstance(block, dict) or block.get('type') != 'tool_result':
                continue
            content = block.get('content')
            if isinstance(content, list):
                text = "\n".join(
                    part.get('text', '') for part in content
                    if isinstance(part, dict) and part.get('type') == 'text'
                )
            else:
                text = str(content or '')
            events.append(ToolCallFinished(
                tool_id=str(block.get('tool_use_id', '')),
                output=text,
                is_error=bool(block.get('is_error')),
            ))
        return events

    if kind == 'result':
        return [TurnResult(
            session_id=data.get('session_id'),
            is_error=bool(data.get('is_error')),
            text=str(data.get('result', '')),
            duration_s=(data.get('duration_ms') or 0) / 1000 or None,
            cost_usd=data.get('total_cost_usd'),
        )]

    return []


# ------------------------------------------------------------------ driver

def _sessions_path() -> Path:
    from symfluence.resources import agent_cache_root
    return agent_cache_root() / 'sessions.json'


def load_session_id(workdir: Path, mode: AgentMode) -> str | None:
    """The last session id recorded for (workdir, mode), if any."""
    try:
        data = json.loads(_sessions_path().read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return (data.get(str(workdir)) or {}).get(mode.value)


def save_session_id(workdir: Path, mode: AgentMode, session_id: str) -> None:
    """Persist the session id for (workdir, mode) across restarts."""
    path = _sessions_path()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault(str(workdir), {})[mode.value] = session_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


class HeadlessClaudeDriver:
    """Per-turn subprocess driver for a headless Claude Code session."""

    def __init__(
        self,
        launcher: AgentLauncher,
        workdir: Path,
        mode: AgentMode = AgentMode.MODELLING,
    ):
        if not launcher.supports_headless:
            raise ValueError(f"'{launcher.name}' cannot be driven headlessly")
        self.launcher = launcher
        self.workdir = workdir
        self.mode = mode
        self.session_id: str | None = load_session_id(workdir, mode)
        self._process: asyncio.subprocess.Process | None = None

    def build_turn_argv(self, prompt: str) -> list[str]:
        """The full argv for one turn (priming + streaming + permissions)."""
        from .launcher import build_launch_argv

        argv, _report = build_launch_argv(
            self.launcher, self.workdir, prompt=prompt, mode=self.mode,
        )
        argv += ['--output-format', 'stream-json',
                 '--include-partial-messages', '--verbose']
        if self.session_id:
            argv += ['--resume', self.session_id]

        profile = get_profile(self.mode)
        if profile.allowed_tools:
            argv += ['--allowedTools', ','.join(profile.allowed_tools)]
        if profile.disallowed_tools:
            argv += ['--disallowedTools', ','.join(profile.disallowed_tools)]
        return argv

    @property
    def turn_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def run_turn(self, prompt: str):
        """Run one turn; yield events as they stream. Ends with TurnResult
        (or DriverError)."""
        argv = self.build_turn_argv(prompt)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                limit=10 * 1024 * 1024,  # single NDJSON lines can be large
            )
        except OSError as e:
            yield DriverError(f"Failed to start '{argv[0]}': {e}")
            return

        saw_result = False
        try:
            assert self._process.stdout is not None
            async for raw in self._process.stdout:
                for event in parse_stream_event(raw.decode('utf-8', errors='replace')):
                    if isinstance(event, SessionInit) and event.session_id:
                        self.session_id = event.session_id
                        save_session_id(self.workdir, self.mode, event.session_id)
                    if isinstance(event, TurnResult):
                        saw_result = True
                        if event.session_id:
                            self.session_id = event.session_id
                            save_session_id(
                                self.workdir, self.mode, event.session_id)
                    yield event
            code = await self._process.wait()
            if code != 0 and not saw_result:
                stderr = b''
                if self._process.stderr is not None:
                    stderr = await self._process.stderr.read()
                detail = stderr.decode('utf-8', errors='replace').strip()
                yield DriverError(
                    f"'{self.launcher.binary}' exited with code {code}"
                    + (f": {detail[-500:]}" if detail else "")
                )
        finally:
            self._process = None

    def interrupt(self) -> bool:
        """Kill the running turn's subprocess (the user pressed stop)."""
        process = self._process
        if process is None or process.returncode is not None:
            return False
        process.terminate()
        return True
