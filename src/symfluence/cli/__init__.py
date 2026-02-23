"""
SYMFLUENCE Command-Line Interface Package
"""
import sys

from .argument_parser import CLIParser


def main():
    """Main CLI entry point."""
    parser = CLIParser()
    args = parser.parse_args()

    if hasattr(args, 'func'):
        try:
            return args.func(args)
        except Exception as e:  # noqa: BLE001 — top-level fallback
            if hasattr(args, 'debug') and args.debug:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.parser.print_help()
        return 1
