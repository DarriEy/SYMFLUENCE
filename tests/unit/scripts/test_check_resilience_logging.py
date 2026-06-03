# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Self-test for scripts/check_resilience_logging.py (review item 10 regression guard)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_GUARD_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_resilience_logging.py"
_spec = importlib.util.spec_from_file_location("check_resilience_logging", _GUARD_PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

_TAG = "# noqa: BLE001 — model execution resilience"


def _scan(tmp_path, body: str):
    (tmp_path / "mod.py").write_text(body, encoding="utf-8")
    return guard.collect_matches(tmp_path)


def test_flags_resilience_handler_without_traceback(tmp_path):
    body = (
        "def f(logger):\n"
        "    try:\n        g()\n"
        f"    except Exception as e:  {_TAG}\n"
        "        logger.error(f'failed: {e}')\n"
    )
    matches = _scan(tmp_path, body)
    assert len(matches) == 1


def test_exc_info_true_is_accepted(tmp_path):
    body = (
        "def f(logger):\n"
        "    try:\n        g()\n"
        f"    except Exception as e:  {_TAG}\n"
        "        logger.error(f'failed: {e}', exc_info=True)\n"
    )
    assert _scan(tmp_path, body) == []


def test_logger_exception_is_accepted(tmp_path):
    body = (
        "def f(logger):\n"
        "    try:\n        g()\n"
        f"    except Exception as e:  {_TAG}\n"
        "        logger.exception('failed')\n"
    )
    assert _scan(tmp_path, body) == []


def test_silent_handler_not_flagged(tmp_path):
    body = (
        "def f():\n"
        "    try:\n        g()\n"
        f"    except Exception:  {_TAG}\n"
        "        return None\n"
    )
    assert _scan(tmp_path, body) == []


def test_untagged_handler_not_flagged(tmp_path):
    body = (
        "def f(logger):\n"
        "    try:\n        g()\n"
        "    except Exception as e:  # noqa: BLE001 — UI resilience\n"
        "        logger.error(f'failed: {e}')\n"
    )
    assert _scan(tmp_path, body) == []


def test_in_tree_resilience_handlers_are_clean():
    """Every tagged resilience handler in the tree captures a traceback (item 10)."""
    import symfluence

    src_root = Path(symfluence.__file__).resolve().parent
    assert guard.collect_matches(src_root) == []
