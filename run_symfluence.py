#!/usr/bin/env python3
"""
SYMFLUENCE CLI Entry Point

This script exists for backward compatibility with the shell wrapper.
It delegates to the main CLI entry point.
"""
import sys


def main():
    """Run the SYMFLUENCE CLI."""
    try:
        from symfluence.main_cli import main as cli_main
        cli_main()
    except ImportError:
        # Package not installed yet - show helpful message
        print("SYMFLUENCE package not installed.", file=sys.stderr)
        print("Run: pip install -e .", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
