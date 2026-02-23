"""
SYMFLUENCE Command-Line Interface entry point.

Provides the main() function that serves as the entry point for the
`symfluence` command. Handles argument parsing, command dispatch, and
error handling for all CLI operations.

The main() function is called by the console_scripts entry point defined
in pyproject.toml and is responsible for:
- Creating the CLI argument parser
- Parsing command-line arguments
- Dispatching to appropriate command handlers
- Handling interrupts and exceptions gracefully
"""


def main():
    """
    Main entry point for SYMFLUENCE CLI.

    Uses subcommand architecture for clean command organization.
    """
    import sys

    from symfluence.core.exceptions import SYMFLUENCEError

    # Windows consoles default to cp1252 which cannot encode Rich's Unicode
    # box-drawing and emoji characters.  Reconfigure to UTF-8 so output
    # renders correctly in modern terminals (Windows Terminal, VS Code, etc.).
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

    from symfluence.cli.argument_parser import CLIParser

    try:
        # Create parser and parse arguments
        parser = CLIParser()
        args = parser.parse_args()

        # Execute the command handler
        if hasattr(args, 'func'):
            return args.func(args)
        else:
            # No command specified - should not happen due to required=True on subparsers
            parser.parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        return 130
    except (SYMFLUENCEError, FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — top-level fallback
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
