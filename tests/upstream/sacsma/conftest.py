from __future__ import annotations

try:
    import jsacsma  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
