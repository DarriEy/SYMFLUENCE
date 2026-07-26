"""Tests for safe optional dependency prompting and installation."""
from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from symfluence.cli.console import Console, ConsoleConfig
from symfluence.cli.exit_codes import ExitCode
from symfluence.cli.optional_dependencies import handle_optional_dependency
from symfluence.core.exceptions import OptionalDependencyError


def _console() -> Console:
    return Console(ConsoleConfig(use_colors=False, output_stream=StringIO(), error_stream=StringIO()))


def test_optional_dependency_error_is_actionable():
    error = OptionalDependencyError("CONUS404 acquisition", "conus404", dependency="intake-xarray")
    assert error.install_target == "symfluence[conus404]"
    assert 'pip install "symfluence[conus404]"' in str(error)
    assert isinstance(error, ImportError)


def test_noninteractive_mode_never_prompts(monkeypatch):
    monkeypatch.setattr("symfluence.cli.optional_dependencies.installation_prompt_allowed", lambda _console: False)
    prompted = False

    def confirm(_message):
        nonlocal prompted
        prompted = True
        return True

    result = handle_optional_dependency(OptionalDependencyError("The GUI", "gui"), _console(), confirm=confirm)
    assert result == ExitCode.DEPENDENCY_ERROR
    assert not prompted


def test_confirmed_install_uses_argv_without_shell(monkeypatch):
    calls = []
    monkeypatch.setattr("symfluence.cli.optional_dependencies.installation_prompt_allowed", lambda _console: True)
    monkeypatch.setattr("symfluence.cli.optional_dependencies._install_command", lambda target: ["installer", target])
    monkeypatch.setattr(
        "symfluence.cli.optional_dependencies.subprocess.run",
        lambda command, check: calls.append((command, check)) or SimpleNamespace(returncode=0),
    )

    result = handle_optional_dependency(
        OptionalDependencyError("The TUI", "tui"), _console(), confirm=lambda _message: True
    )
    assert result == ExitCode.DEPENDENCY_ERROR
    assert calls == [(["installer", "symfluence[tui]"], False)]
