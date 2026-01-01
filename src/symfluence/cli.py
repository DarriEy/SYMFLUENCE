def main():
    """
    Main entry point for SYMFLUENCE CLI.

    Uses subcommand architecture for clean command organization.
    """
    import sys
    from symfluence.utils.cli.argument_parser import CLIParser

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
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())