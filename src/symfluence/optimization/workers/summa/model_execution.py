#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Execution for SUMMA Workers

This module contains functions for executing SUMMA and mizuRoute
models in worker processes.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict

from .netcdf_utilities import fix_summa_time_precision
from symfluence.core.profiling import get_system_profiler


def _cleanup_stale_output_files(output_dir: Path, logger) -> None:
    """Remove stale output files from previous iterations to prevent metric calculation errors.

    This is critical for calibration: if routing fails, we don't want metrics to be
    calculated from old output files from previous iterations.

    Args:
        output_dir: Directory containing model output files
        logger: Logger instance
    """
    if not output_dir.exists():
        return

    # Patterns to clean up (SUMMA and mizuRoute output files)
    cleanup_patterns = [
        "*.nc",           # All NetCDF files (SUMMA timestep, day, mizuRoute output)
        "*_restart_*.nc", # Restart files
        "runinfo.txt",    # SUMMA run info
    ]

    files_removed = 0
    for pattern in cleanup_patterns:
        for file_path in output_dir.glob(pattern):
            # Skip if it's a directory
            if file_path.is_dir():
                continue
            try:
                file_path.unlink()
                files_removed += 1
            except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
                logger.warning(f"Could not remove stale file {file_path}: {e}")

    if files_removed > 0:
        logger.debug(f"Cleaned up {files_removed} stale output files from {output_dir}")


def _deduplicate_output_control(output_control_path: Path, logger):
    """Ensure outputControl.txt doesn't have duplicate variables for the same frequency"""
    if not output_control_path.exists():
        return

    try:
        with open(output_control_path, 'r') as f:
            lines = f.readlines()

        seen_vars = {} # (var_name, frequency) -> True
        new_lines = []
        changed = False

        for line in lines:
            trimmed = line.strip()
            # Skip comments, empty lines, and lines without frequency separator
            if not trimmed or trimmed.startswith('!') or '|' not in trimmed:
                new_lines.append(line)
                continue

            parts = [p.strip() for p in trimmed.split('|')]
            var_name = parts[0]
            freq = parts[1] if len(parts) > 1 else '1'

            key = (var_name, freq)
            if key in seen_vars:
                logger.warning(f"Removing duplicate output request: {var_name} | {freq}")
                changed = True
                continue

            seen_vars[key] = True
            new_lines.append(line)

        if changed:
            with open(output_control_path, 'w') as f:
                f.writelines(new_lines)
            logger.debug(f"Deduplicated {output_control_path.name}")

    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
        logger.warning(f"Failed to deduplicate output control: {e}")


def _run_summa_worker(summa_exe: Path, file_manager: Path, summa_dir: Path, logger, debug_info: Dict, summa_settings_dir: Path = None) -> bool:
    """SUMMA execution with iteration-aware logging.

    Log files include iteration number to prevent overwriting during calibration.
    This enables post-hoc debugging of failed runs when PARAMS_KEEP_TRIALS is enabled.
    """
    try:
        # Create log directory
        log_dir = summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Clean up stale output files from previous iterations
        # This prevents metrics being calculated from old files if current run fails
        _cleanup_stale_output_files(summa_dir, logger)

        # Include iteration in log filename to prevent overwriting
        iteration = debug_info.get('iteration', 0)
        individual_id = debug_info.get('individual_id', 0)
        log_file = log_dir / f"summa_worker_{os.getpid()}_iter{iteration:05d}_ind{individual_id:03d}.log"

        # Set environment for single-threaded execution
        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'
        })

        # Convert paths to strings for subprocess
        summa_exe_str = str(summa_exe)
        file_manager_str = str(file_manager)

        # Verify executable permissions
        if not os.access(summa_exe, os.X_OK):
            error_msg = f"SUMMA executable is not executable: {summa_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        # Update file manager with correct output path and settings path
        try:
            with open(file_manager, 'r') as f:
                lines = f.readlines()

            # Remove stale runinfo to avoid SUMMA Fortran write errors on reruns
            runinfo_path = summa_dir / "runinfo.txt"
            if runinfo_path.exists():
                try:
                    runinfo_path.chmod(0o644)
                    runinfo_path.unlink()
                except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
                    logger.warning(f"Could not remove existing runinfo.txt: {e}")

            updated_lines = []
            output_path_updated = False
            settings_path_updated = False

            # Ensure paths end with slash for SUMMA
            output_path_str = str(summa_dir)
            if not output_path_str.endswith(os.sep):
                output_path_str += os.sep

            # Prepare settings path if provided
            settings_path_str = None
            if summa_settings_dir is not None:
                settings_path_str = str(summa_settings_dir)
                if not settings_path_str.endswith(os.sep):
                    settings_path_str += os.sep

            for line in lines:
                if 'outputPath' in line and not line.strip().startswith('!'):
                    # Preserve comment if present
                    parts = line.split('!')
                    comment = '!' + parts[1] if len(parts) > 1 else ''

                    # Update path - SUMMA requires quoted strings
                    updated_lines.append(f"outputPath '{output_path_str}' {comment}\n")
                    output_path_updated = True
                elif 'settingsPath' in line and not line.strip().startswith('!') and settings_path_str is not None:
                    # Preserve comment if present
                    parts = line.split('!')
                    comment = '!' + parts[1] if len(parts) > 1 else ''

                    # Update settings path - SUMMA requires quoted strings
                    updated_lines.append(f"settingsPath '{settings_path_str}' {comment}\n")
                    settings_path_updated = True
                else:
                    updated_lines.append(line)

            if not output_path_updated:
                # If outputPath not found, append it (though uncommon for valid file manager)
                updated_lines.append(f"outputPath '{output_path_str}'\n")

            if settings_path_str is not None and not settings_path_updated:
                # If settingsPath not found, append it
                updated_lines.append(f"settingsPath '{settings_path_str}'\n")

            with open(file_manager, 'w') as f:
                f.writelines(updated_lines)

            # Deduplicate outputControl.txt if it exists
            output_control_name = None
            for line in updated_lines:
                if 'outputControlFile' in line and not line.strip().startswith('!'):
                    output_control_name = line.split("'")[1] if "'" in line else line.split()[1]
                    break

            if output_control_name and summa_settings_dir:
                _deduplicate_output_control(summa_settings_dir / output_control_name, logger)

            logger.debug(f"Updated file manager output path to: {output_path_str}")
            if settings_path_str is not None:
                logger.debug(f"Updated file manager settings path to: {settings_path_str}")

        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            logger.warning(f"Failed to update file manager paths: {e}")
            # Continue anyway, hoping it works or fails later

        # Build command as list to avoid shell=True security concerns
        cmd = [summa_exe_str, "-m", file_manager_str]
        cmd_str = " ".join(cmd)

        logger.info(f"Executing SUMMA command: {cmd_str}")
        logger.debug(f"Working directory: {summa_dir}")

        debug_info['commands_run'].append(f"SUMMA: {cmd_str}")

        # Run SUMMA with system-level I/O profiling
        system_profiler = get_system_profiler()

        with open(log_file, 'w') as f:
            f.write("SUMMA Execution Log\n")
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Working Directory: {summa_dir}\n")
            f.write(f"Environment: OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', 'unset')}\n")
            f.write("=" * 50 + "\n")
            f.flush()

            # Profile SUMMA execution
            with system_profiler.profile_subprocess(
                command=cmd,
                component='summa',
                iteration=debug_info.get('iteration'),
                cwd=summa_dir,
                env=env,
                track_files=True,
                output_dir=summa_dir
            ) as proc:
                proc.run(
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )

        # Check if output files were created
        timestep_files = list(summa_dir.glob("*timestep.nc"))
        if not timestep_files:
            # Look for any .nc files
            nc_files = list(summa_dir.glob("*.nc"))
            if not nc_files:
                dir_contents = list(summa_dir.glob("*"))
                error_msg = f"No SUMMA output files found in {summa_dir}. Contents: {[f.name for f in dir_contents]}"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return False

        logger.info(f"SUMMA execution completed successfully. Output files: {len(timestep_files)} timestep files")
        debug_info['summa_output_files'] = [str(f) for f in timestep_files[:3]]  # First 3 files

        return True

    except subprocess.CalledProcessError as e:
        error_msg = f"SUMMA simulation failed with exit code {e.returncode}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

    except subprocess.TimeoutExpired:
        error_msg = "SUMMA simulation timed out (120 minutes)"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
        error_msg = f"Error running SUMMA: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _run_mizuroute_worker(task_data: Dict, mizuroute_dir: Path, logger, debug_info: Dict, summa_dir: Path = None) -> bool:
    """Updated mizuRoute worker with fixed time precision handling"""
    try:
        # Clean up stale mizuRoute output files from previous iterations
        _cleanup_stale_output_files(mizuroute_dir, logger)

        # Verify SUMMA output exists first
        if summa_dir is None:
            summa_dir = Path(task_data['summa_dir'])

        expected_files = list(summa_dir.glob("*timestep.nc"))

        if not expected_files:
            error_msg = f"No SUMMA timestep files found for mizuRoute input: {summa_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        # Fix SUMMA time precision with better error handling
        try:
            logger.info("Fixing SUMMA time precision for mizuRoute compatibility")
            fix_summa_time_precision(expected_files[0])
            logger.info("SUMMA time precision fixed successfully")
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            error_msg = f"Failed to fix SUMMA time precision: {str(e)}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        logger.info(f"Found {len(expected_files)} SUMMA output files for mizuRoute")
        config = task_data['config']

        # Get mizuRoute executable
        mizu_path = config.get('INSTALL_PATH_MIZUROUTE', 'default')
        if mizu_path == 'default':
            mizu_path = Path(config.get('SYMFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)

        mizu_exe = mizu_path / config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        control_file = Path(task_data['mizuroute_settings_dir']) / 'mizuroute.control'

        # Verify files exist
        if not mizu_exe.exists():
            error_msg = f"mizuRoute executable not found: {mizu_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        if not control_file.exists():
            error_msg = f"mizuRoute control file not found: {control_file}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        debug_info['files_checked'].extend([
            f"mizuRoute exe: {mizu_exe}",
            f"mizuRoute control: {control_file}"
        ])

        # Create log directory with iteration-aware naming
        log_dir = mizuroute_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        iteration = debug_info.get('iteration', 0)
        individual_id = debug_info.get('individual_id', 0)
        log_file = log_dir / f"mizuroute_worker_{os.getpid()}_iter{iteration:05d}_ind{individual_id:03d}.log"

        # Build command as list to avoid shell=True security concerns
        cmd = [str(mizu_exe), str(control_file)]
        cmd_str = " ".join(cmd)

        logger.info(f"Executing mizuRoute command: {cmd_str}")
        debug_info['commands_run'].append(f"mizuRoute: {cmd_str}")
        debug_info['mizuroute_log'] = str(log_file)

        # Run mizuRoute with system-level I/O profiling
        system_profiler = get_system_profiler()

        with open(log_file, 'w') as f:
            f.write("mizuRoute Execution Log\n")
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Working Directory: {control_file.parent}\n")
            f.write("=" * 50 + "\n")
            f.flush()

            # Profile mizuRoute execution
            with system_profiler.profile_subprocess(
                command=cmd,
                component='mizuroute',
                iteration=debug_info.get('iteration'),
                cwd=control_file.parent,
                track_files=True,
                output_dir=mizuroute_dir
            ) as proc:
                proc.run(
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )

        # Check for output files
        nc_files = list(mizuroute_dir.glob("*.nc"))
        if not nc_files:
            error_msg = f"No mizuRoute output files found in {mizuroute_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        logger.info(f"mizuRoute execution completed successfully. Output files: {len(nc_files)}")
        debug_info['mizuroute_output_files'] = [str(f) for f in nc_files[:3]]

        return True

    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
        error_msg = f"mizuRoute execution failed: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _needs_mizuroute_routing_worker(config: Dict) -> bool:
    """Check if mizuRoute routing is needed"""
    domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
    routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

    if domain_method not in ['point', 'lumped']:
        return True

    if domain_method == 'lumped' and routing_delineation == 'river_network':
        return True

    return False
