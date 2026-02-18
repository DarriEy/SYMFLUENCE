"""
FUSE Model Execution

Standalone functions for executing the FUSE model subprocess,
managing input file symlinks, and validating output.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_fuse_run_mode(config: Dict[str, Any], kwargs: Dict[str, Any],
                         log: Optional[logging.Logger] = None) -> str:
    """
    Determine FUSE run mode based on config and kwargs.

    Args:
        config: Configuration dictionary
        kwargs: Additional keyword arguments (may contain 'mode')
        log: Logger instance

    Returns:
        Run mode string ('run_def' or 'run_pre')
    """
    log = log or logger

    explicit_mode = config.get('FUSE_RUN_MODE')
    if explicit_mode:
        log.debug(f"FUSE using explicit run mode from config: {explicit_mode}")
        return explicit_mode

    regionalization_method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
    if config.get('USE_TRANSFER_FUNCTIONS', False):
        regionalization_method = 'transfer_function'

    if regionalization_method != 'lumped':
        mode = kwargs.get('mode', 'run_pre')
        log.info(f"FUSE auto-selected run_pre mode (regionalization={regionalization_method})")
    else:
        mode = kwargs.get('mode', 'run_def')
        log.debug("FUSE using run_def mode (lumped regionalization)")

    return mode


def resolve_fuse_paths(
    config: Dict[str, Any],
    settings_dir: Path,
    log: Optional[logging.Logger] = None
) -> Tuple[Path, Path, Path]:
    """
    Resolve FUSE executable, file manager, and execution directory paths.

    Args:
        config: Configuration dictionary
        settings_dir: FUSE settings directory
        log: Logger instance

    Returns:
        Tuple of (fuse_exe, filemanager_path, execution_cwd)
    """
    log = log or logger

    # Resolve executable
    fuse_install = config.get('FUSE_INSTALL_PATH', 'default')
    if fuse_install == 'default':
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        fuse_exe = data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
    else:
        fuse_exe = Path(fuse_install) / 'fuse.exe'

    # Resolve file manager and execution directory
    if settings_dir.name == 'FUSE':
        filemanager_path = settings_dir / 'fm_catch.txt'
        execution_cwd = settings_dir
    elif (settings_dir / 'FUSE').exists():
        filemanager_path = settings_dir / 'FUSE' / 'fm_catch.txt'
        execution_cwd = settings_dir / 'FUSE'
    else:
        filemanager_path = settings_dir / 'fm_catch.txt'
        execution_cwd = settings_dir

    return fuse_exe, filemanager_path, execution_cwd


def prepare_input_files(
    config: Dict[str, Any],
    execution_cwd: Path,
    log: Optional[logging.Logger] = None
) -> Tuple[str, str]:
    """
    Create symlinks and copies for FUSE input files.

    Args:
        config: Configuration dictionary
        execution_cwd: Directory where FUSE will execute
        log: Logger instance

    Returns:
        Tuple of (fuse_run_id, actual_decisions_file)
    """
    log = log or logger

    fuse_run_id = 'sim'
    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
    domain_name = config.get('DOMAIN_NAME')
    project_dir = data_dir / f"domain_{domain_name}"
    fuse_input_dir = project_dir / 'forcing' / 'FUSE_input'
    experiment_id = config.get('EXPERIMENT_ID', 'run_1')
    fuse_id = config.get('FUSE_FILE_ID', experiment_id)

    # Input files to symlink
    input_files = [
        (fuse_input_dir / f"{domain_name}_input.nc", f"{fuse_run_id}_input.nc"),
        (fuse_input_dir / f"{domain_name}_elev_bands.nc", f"{fuse_run_id}_elev_bands.nc")
    ]

    # COPY (not symlink) the parameter file to match the short alias
    _copy_parameter_file(execution_cwd, domain_name, fuse_id, fuse_run_id, log)

    # Ensure configuration files are present
    project_settings_dir = project_dir / 'settings' / 'FUSE'
    actual_decisions_file = _ensure_config_files(
        execution_cwd, project_settings_dir, experiment_id, input_files, log
    )

    # Create symlinks (but NOT para_def.nc which was copied above)
    _create_symlinks(input_files, execution_cwd, log)

    # Verify para_def copy exists
    param_file_dst = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
    if not param_file_dst.exists():
        log.error(f"FUSE para_def copy was not created. Expected: {param_file_dst}")

    return fuse_run_id, actual_decisions_file


def _copy_parameter_file(
    execution_cwd: Path,
    domain_name: str,
    fuse_id: str,
    fuse_run_id: str,
    log: logging.Logger
) -> None:
    """Copy para_def.nc to match the short alias."""
    param_file_src = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
    param_file_dst = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"

    if param_file_src.exists():
        if param_file_dst.exists():
            param_file_dst.unlink()
        shutil.copy2(param_file_src, param_file_dst)
        log.debug(f"FUSE para_def copied (not symlinked): {param_file_dst.name}")
    else:
        log.warning(f"FUSE para_def source not found: {param_file_src}")


def _ensure_config_files(
    execution_cwd: Path,
    project_settings_dir: Path,
    experiment_id: str,
    input_files: list,
    log: logging.Logger
) -> str:
    """Ensure FUSE configuration files are present, return actual decisions filename."""
    log.debug(f"Checking for config files in: {project_settings_dir}")

    config_files = ['input_info.txt', 'fuse_zNumerix.txt']

    # Add decisions file
    actual_decisions_file = f"fuse_zDecisions_{experiment_id}.txt"
    if (project_settings_dir / actual_decisions_file).exists():
        config_files.append(actual_decisions_file)
    else:
        log.warning(f"Decisions file {actual_decisions_file} not found in {project_settings_dir}")
        try:
            decisions = list(project_settings_dir.glob("fuse_zDecisions_*.txt"))
            if decisions:
                actual_decisions_file = decisions[0].name
                config_files.append(actual_decisions_file)
                log.warning(f"Using fallback decisions file: {actual_decisions_file}")
        except Exception as e:
            log.warning(f"Error searching for decisions files: {e}")

    for cfg_file in config_files:
        target_path = execution_cwd / cfg_file
        if not target_path.exists():
            src_path = project_settings_dir / cfg_file
            if src_path.exists():
                input_files.append((src_path, cfg_file))
                log.warning(f"Restoring missing config file: {cfg_file}")
            else:
                log.error(f"Source config file not found: {src_path}")

    return actual_decisions_file


def _create_symlinks(
    input_files: list,
    execution_cwd: Path,
    log: logging.Logger
) -> None:
    """Create symlinks for input files."""
    for src, link_name in input_files:
        if src.exists():
            link_path = execution_cwd / link_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                link_path.symlink_to(src)
                log.debug(f"Created symlink: {link_path} -> {src}")
            except OSError:
                shutil.copy2(src, link_path)
                log.debug(f"Copied (symlink unavailable): {src} -> {link_path}")
        else:
            log.warning(f"Symlink source not found: {src}")


def execute_fuse(
    fuse_exe: Path,
    filemanager_path: Path,
    execution_cwd: Path,
    fuse_run_id: str,
    mode: str,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Optional[subprocess.CompletedProcess]:
    """
    Execute the FUSE subprocess.

    Args:
        fuse_exe: Path to FUSE executable
        filemanager_path: Path to file manager
        execution_cwd: Working directory for execution
        fuse_run_id: Short run alias (e.g., 'sim')
        mode: Run mode ('run_def' or 'run_pre')
        config: Configuration dictionary
        log: Logger instance

    Returns:
        CompletedProcess result, or None if execution failed
    """
    log = log or logger
    fuse_id = config.get('FUSE_FILE_ID', config.get('EXPERIMENT_ID'))

    cmd = [str(fuse_exe), str(filemanager_path.name), fuse_run_id, mode]

    # For run_pre mode, append parameter file
    if mode == 'run_pre':
        param_file = _find_run_pre_param_file(execution_cwd, fuse_run_id, fuse_id,
                                               config.get('DOMAIN_NAME', ''), log)
        if param_file:
            cmd.append(str(param_file.name))
            cmd.append('1')
        else:
            return None

    log.debug(f"Executing FUSE: {' '.join(cmd)} in {execution_cwd}")

    # Verify key files
    expected_para_def = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
    if not expected_para_def.exists() and not expected_para_def.is_symlink():
        log.error(f"FUSE parameter file not found: {expected_para_def}")

    result = subprocess.run(
        cmd,
        cwd=str(execution_cwd),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=config.get('FUSE_TIMEOUT', 3600)
    )

    if result.returncode != 0:
        log.error(f"FUSE failed with return code {result.returncode}")
        log.error(f"STDOUT: {result.stdout}")
        log.error(f"STDERR: {result.stderr}")
        return None

    # Detect Fortran STOP messages (FUSE returns exit code 0 on macOS even on STOP)
    combined_output = (result.stdout or '') + (result.stderr or '')
    if 'STOP' in combined_output and 'failed to converge' in combined_output:
        log.error(
            f"FUSE hit convergence failure (Fortran STOP with exit code 0). "
            f"Output: {combined_output[-300:]}"
        )
        return None

    if result.stdout:
        log.debug(f"FUSE stdout (last 500 chars): {result.stdout[-500:]}")
    if result.stderr:
        log.debug(f"FUSE stderr: {result.stderr}")

    return result


def _find_run_pre_param_file(
    execution_cwd: Path,
    fuse_run_id: str,
    fuse_id: str,
    domain_name: str,
    log: logging.Logger
) -> Optional[Path]:
    """Find the parameter file for run_pre mode."""
    param_file = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
    if param_file.exists():
        return param_file

    param_file_alt = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
    if param_file_alt.exists():
        return param_file_alt

    log.error(f"Parameter file not found for run_pre: tried {param_file} and {param_file_alt}")
    return None


def handle_fuse_output(
    execution_cwd: Path,
    fuse_output_dir: Path,
    fuse_run_id: str,
    mode: str,
    config: Dict[str, Any],
    result: subprocess.CompletedProcess,
    log: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Move FUSE output to final destination and validate it.

    Args:
        execution_cwd: FUSE execution directory
        fuse_output_dir: Final output directory
        fuse_run_id: Short run alias
        mode: Run mode
        config: Configuration dictionary
        result: Subprocess result
        log: Logger instance

    Returns:
        Path to final output file, or None if validation failed
    """
    log = log or logger

    domain_name = config.get('DOMAIN_NAME')
    fuse_id = config.get('FUSE_FILE_ID', config.get('EXPERIMENT_ID'))

    run_suffix = 'runs_def' if mode == 'run_def' else 'runs_pre'
    local_output_filename = f"{fuse_run_id}_{fuse_id}_{run_suffix}.nc"
    local_output_path = execution_cwd / local_output_filename
    final_output_path = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

    if local_output_path.exists():
        try:
            if final_output_path.exists():
                final_output_path.unlink()
            shutil.move(str(local_output_path), str(final_output_path))
            log.debug(f"Moved output from {local_output_path} to {final_output_path}")
        except Exception as e:
            log.error(f"Failed to move output file: {e}")
            return None
    else:
        log.error(f"FUSE returned success but local output file not created: {local_output_path}")
        if result.stdout:
            stdout_lines = result.stdout.split('\n')
            if len(stdout_lines) > 20:
                log.error(f"FUSE stdout (first 10 lines): {chr(10).join(stdout_lines[:10])}")
                log.error(f"FUSE stdout (last 10 lines): {chr(10).join(stdout_lines[-10:])}")
            else:
                log.error(f"FUSE stdout: {result.stdout}")
        return None

    # Validate output has actual data
    if not _validate_fuse_output(final_output_path, result, log):
        return None

    log.debug(f"FUSE completed successfully, output: {final_output_path}")
    return final_output_path


def _validate_fuse_output(
    output_path: Path,
    result: subprocess.CompletedProcess,
    log: logging.Logger
) -> bool:
    """Validate that FUSE output file has actual data."""
    try:
        import xarray as xr
        with xr.open_dataset(output_path, decode_times=False) as ds_check:
            time_dim = 'time' if 'time' in ds_check.dims else None
            if time_dim and ds_check.sizes[time_dim] == 0:
                log.error(
                    f"FUSE output has 0 time steps — model likely crashed silently "
                    f"(Fortran STOP returns exit code 0). "
                    f"Stderr: {(result.stderr or '')[:500]}"
                )
                return False
            n_time = ds_check.sizes.get(time_dim, 0) if time_dim else -1
            log.debug(f"FUSE output validated: {n_time} time steps in {output_path.name}")
    except Exception as e:
        log.warning(f"Could not validate FUSE output time dimension: {e}")

    return True


def log_execution_directory(execution_cwd: Path, log: logging.Logger) -> None:
    """Log files in execution directory at DEBUG level."""
    log.debug(f"Files in execution CWD ({execution_cwd}):")
    try:
        for f in execution_cwd.iterdir():
            if f.is_symlink():
                log.debug(f"  {f.name} -> {f.resolve()}")
            else:
                log.debug(f"  {f.name}")
    except Exception as e:
        log.debug(f"Could not list directory: {e}")
