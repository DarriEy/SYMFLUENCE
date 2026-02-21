"""Legacy CLI compatibility module.

This module is retained so imports such as ``from symfluence import cli``
continue to work in tests and external automation. The canonical command-line
entrypoint is ``symfluence.main_cli:main``.
"""

from symfluence.main_cli import main as _main


def main() -> int:
    """Compatibility wrapper that delegates to the canonical CLI entrypoint."""
    return _main()


if __name__ == "__main__":
    import sys

    sys.exit(main())
