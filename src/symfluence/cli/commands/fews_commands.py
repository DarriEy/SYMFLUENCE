"""
FEWS CLI command handlers.

Provides ``symfluence fews pre|post|run|launch`` subcommands for
operating the Delft-FEWS General Adapter integration.
"""

import logging
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler

logger = logging.getLogger(__name__)


class FEWSCommands(BaseCommand):
    """Command handlers for the ``symfluence fews`` category."""

    @staticmethod
    @cli_exception_handler
    def pre(args: Namespace) -> int:
        """Run the FEWS pre-adapter."""
        from symfluence.fews.pi_diagnostics import DiagnosticsCollector
        from symfluence.fews.pre_adapter import FEWSPreAdapter

        run_info_path = Path(args.run_info)
        base_config = Path(args.config) if hasattr(args, 'config') and args.config else None
        data_format = getattr(args, 'format', 'netcdf-cf')
        id_map_path = getattr(args, 'id_map', None)

        # Determine diagnostics output location
        diag_dir = run_info_path.parent / "toFews"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag = DiagnosticsCollector(diag_dir / "diag.xml")

        try:
            adapter = FEWSPreAdapter(
                run_info_path=run_info_path,
                base_config_path=base_config,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            config, run_info = adapter.run(diag=diag)
            diag.info("Pre-adapter completed successfully")
            FEWSCommands._console.success("FEWS pre-adapter completed")
            return ExitCode.SUCCESS
        except Exception as exc:
            diag.fatal(str(exc))
            raise
        finally:
            diag.write()

    @staticmethod
    @cli_exception_handler
    def post(args: Namespace) -> int:
        """Run the FEWS post-adapter."""
        from symfluence.fews.pi_diagnostics import DiagnosticsCollector
        from symfluence.fews.post_adapter import FEWSPostAdapter

        run_info_path = Path(args.run_info)
        config_path = Path(args.config) if hasattr(args, 'config') and args.config else None
        data_format = getattr(args, 'format', 'netcdf-cf')
        id_map_path = getattr(args, 'id_map', None)

        diag_dir = run_info_path.parent / "toFews"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag = DiagnosticsCollector(diag_dir / "diag.xml")

        try:
            adapter = FEWSPostAdapter(
                run_info_path=run_info_path,
                config_path=config_path,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            output_dir = adapter.run(diag=diag)
            diag.info("Post-adapter completed successfully")
            FEWSCommands._console.success(f"FEWS post-adapter completed -> {output_dir}")
            return ExitCode.SUCCESS
        except Exception as exc:
            diag.fatal(str(exc))
            raise
        finally:
            diag.write()

    @staticmethod
    @cli_exception_handler
    def run_full(args: Namespace) -> int:
        """Run the full FEWS adapter cycle: pre -> model -> post."""
        from symfluence.fews.pi_diagnostics import DiagnosticsCollector
        from symfluence.fews.post_adapter import FEWSPostAdapter
        from symfluence.fews.pre_adapter import FEWSPreAdapter

        run_info_path = Path(args.run_info)
        base_config = Path(args.config) if hasattr(args, 'config') and args.config else None
        data_format = getattr(args, 'format', 'netcdf-cf')
        id_map_path = getattr(args, 'id_map', None)

        diag_dir = run_info_path.parent / "toFews"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag = DiagnosticsCollector(diag_dir / "diag.xml")

        try:
            # Pre-adapter
            diag.info("Starting pre-adapter")
            pre = FEWSPreAdapter(
                run_info_path=run_info_path,
                base_config_path=base_config,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            config, run_info = pre.run(diag=diag)
            FEWSCommands._console.info("Pre-adapter complete")

            # Model execution
            diag.info("Starting model execution")
            if base_config and Path(base_config).is_file():
                from symfluence.cli.commands.workflow_commands import WorkflowCommands
                model_args = Namespace(
                    config=str(base_config),
                    force_rerun=False,
                    continue_on_error=False,
                    debug=getattr(args, 'debug', False),
                )
                exit_code = WorkflowCommands.run(model_args)
                if exit_code != ExitCode.SUCCESS:
                    diag.error(f"Model execution returned exit code {exit_code}")
                    return exit_code
                diag.info("Model execution completed")
            else:
                diag.warning("No base config provided, skipping model execution")

            # Post-adapter
            diag.info("Starting post-adapter")
            post = FEWSPostAdapter(
                run_info_path=run_info_path,
                config_path=base_config,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            output_dir = post.run(diag=diag)
            diag.info("Full FEWS adapter cycle completed")
            FEWSCommands._console.success(f"FEWS run completed -> {output_dir}")
            return ExitCode.SUCCESS
        except Exception as exc:
            diag.fatal(str(exc))
            raise
        finally:
            diag.write()

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """Launch openFEWS with SYMFLUENCE adapter support.

        Starts the openFEWS application, auto-configuring the General Adapter
        module to point at the SYMFLUENCE CLI.
        """
        import shutil

        config_path = getattr(args, 'config', None)
        port = getattr(args, 'port', 8080)

        # Locate openFEWS installation
        openfews_exe = shutil.which("openfews")
        if openfews_exe is None:
            # Try SYMFLUENCE installs directory
            try:
                from symfluence.cli.services.build_registry import BuildInstructionsRegistry
                instructions = BuildInstructionsRegistry.get_instructions('openfews')
                if instructions:
                    # Try loading config for data dir
                    candidate = Path.cwd() / "data" / instructions.get('default_path_suffix', '')
                    if candidate.exists():
                        openfews_exe = str(candidate / instructions.get('default_exe', 'bin/fews.sh'))
            except (ImportError, AttributeError, OSError, TypeError, ValueError):
                pass

        if openfews_exe is None:
            FEWSCommands._console.error(
                "openFEWS not found. Install it with: symfluence binary install openfews"
            )
            return ExitCode.BINARY_ERROR

        # Build launch command
        cmd = [openfews_exe]
        if port:
            cmd.extend(["--port", str(port)])

        FEWSCommands._console.info(f"Launching openFEWS: {' '.join(cmd)}")

        # Set environment so openFEWS knows where SYMFLUENCE is
        import os
        env = os.environ.copy()
        env["SYMFLUENCE_CLI"] = sys.executable + " -m symfluence"
        if config_path:
            env["SYMFLUENCE_CONFIG"] = str(config_path)

        try:
            proc = subprocess.run(cmd, env=env)
            return proc.returncode
        except FileNotFoundError:
            FEWSCommands._console.error(f"Cannot execute: {openfews_exe}")
            return ExitCode.BINARY_ERROR
        except KeyboardInterrupt:
            FEWSCommands._console.info("openFEWS terminated by user")
            return ExitCode.SUCCESS
