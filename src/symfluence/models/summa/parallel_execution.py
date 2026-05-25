# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Local GRU-parallel execution for SUMMA model runs and calibration.

Splits a SUMMA domain into GRU chunks using the ``-g startGRU numGRU`` flag,
runs those chunks as concurrent local subprocesses, then merges outputs.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_gru_count_from_attributes(settings_dir: Path) -> int:
    """Read total GRU count from SUMMA attributes.nc."""
    import xarray as xr

    attr_path = settings_dir / 'attributes.nc'
    if not attr_path.exists():
        raise FileNotFoundError(f"attributes.nc not found in {settings_dir}")

    with xr.open_dataset(attr_path) as ds:
        if 'gru' not in ds.sizes:
            raise KeyError("'gru' dimension not found in attributes.nc")
        return int(ds.sizes['gru'])


def compute_gru_splits(
    total_grus: int,
    num_processes: int,
) -> List[Tuple[int, int]]:
    """Compute (startGRU, numGRU) pairs for parallel execution.

    Divides GRUs as evenly as possible.  GRU indices are 1-based
    (SUMMA convention).
    """
    num_processes = max(1, min(num_processes, total_grus))
    base_size = total_grus // num_processes
    remainder = total_grus % num_processes

    splits: List[Tuple[int, int]] = []
    current = 1
    for i in range(num_processes):
        chunk = base_size + (1 if i < remainder else 0)
        splits.append((current, chunk))
        current += chunk
    return splits


def create_split_file_manager(
    source_file_manager: Path,
    split_output_dir: Path,
    split_id: int,
    logger: logging.Logger,
) -> Path:
    """Create a file-manager copy whose ``outputPath`` points at *split_output_dir*.

    ``settingsPath`` is left unchanged — all splits share the same
    trialParams.nc, attributes.nc, forcing, etc.
    """
    split_output_dir.mkdir(parents=True, exist_ok=True)
    split_fm = split_output_dir / 'fileManager.txt'

    output_path_str = str(split_output_dir).rstrip(os.sep) + os.sep

    with open(source_file_manager, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('!'):
            updated.append(line)
        elif 'outputPath' in line:
            updated.append(f"outputPath '{output_path_str}'\n")
        elif 'outFilePrefix' in line:
            orig = line.split("'")[1] if "'" in line else 'output'
            updated.append(f"outFilePrefix 'split_{split_id:02d}_{orig}'\n")
        else:
            updated.append(line)

    with open(split_fm, 'w', encoding='utf-8') as f:
        f.writelines(updated)

    logger.debug("Created split %d file manager: %s", split_id, split_fm)
    return split_fm


# Module-level function so it can be used with ThreadPoolExecutor
def _run_single_gru_split(split_args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single GRU split as a subprocess."""
    start = time.time()
    split_id = split_args['split_id']

    try:
        cmd = [
            split_args['summa_exe'],
            '-g', str(split_args['start_gru']), str(split_args['num_gru']),
            '-m', split_args['file_manager'],
        ]

        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
        })
        if split_args.get('env'):
            env.update(split_args['env'])

        with open(split_args['log_file'], 'w', encoding='utf-8') as log_f:
            log_f.write(
                f"Split {split_id}: GRUs {split_args['start_gru']}-"
                f"{split_args['start_gru'] + split_args['num_gru'] - 1}\n"
            )
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write('=' * 50 + '\n')
            log_f.flush()

            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=split_args['timeout'],
                env=env,
            )

        return {
            'split_id': split_id,
            'success': result.returncode == 0,
            'error': f'Exit code {result.returncode}' if result.returncode != 0 else None,
            'duration': time.time() - start,
        }

    except subprocess.TimeoutExpired:
        return {
            'split_id': split_id,
            'success': False,
            'error': 'Timeout',
            'duration': time.time() - start,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {
            'split_id': split_id,
            'success': False,
            'error': str(e),
            'duration': time.time() - start,
        }


def merge_split_outputs(
    split_dirs: List[Path],
    target_dir: Path,
    experiment_prefix: str,
    logger: logging.Logger,
) -> bool:
    """Merge per-split SUMMA NetCDF outputs along the ``hru`` dimension."""
    import xarray as xr

    for suffix in ('timestep', 'day'):
        split_files: list[Path] = []
        for split_dir in split_dirs:
            found = sorted(split_dir.glob(f'*_{suffix}.nc'))
            if found:
                split_files.append(found[0])

        if not split_files:
            continue

        try:
            datasets = [xr.open_dataset(f) for f in split_files]

            # SUMMA outputs some vars on (time, hru) and others on (time, gru).
            # Concat each dimension independently then merge.
            hru_vars = [v for v in datasets[0].data_vars if 'hru' in datasets[0][v].dims]
            gru_vars = [v for v in datasets[0].data_vars
                        if 'gru' in datasets[0][v].dims and v not in hru_vars]

            parts = []
            if hru_vars:
                parts.append(xr.concat(
                    [ds[hru_vars] for ds in datasets],
                    dim='hru', coords='minimal', compat='override',
                ))
            if gru_vars:
                parts.append(xr.concat(
                    [ds[gru_vars] for ds in datasets],
                    dim='gru', coords='minimal', compat='override',
                ))

            merged = xr.merge(parts, compat='override') if parts else datasets[0]

            encoding = {}
            for var in merged.data_vars:
                if merged[var].dtype.kind == 'i':
                    encoding[var] = {'_FillValue': None}

            output_file = target_dir / f'{experiment_prefix}_{suffix}.nc'
            merged.to_netcdf(output_file, format='NETCDF4', encoding=encoding)

            total_hrus = sum(ds.sizes.get('hru', 0) for ds in datasets)
            for ds in datasets:
                ds.close()
            merged.close()

            logger.info(
                "Merged %d splits -> %s (%d HRUs)",
                len(split_files),
                output_file.name,
                total_hrus,
            )
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            logger.error("Merge failed for %s files: %s", suffix, e)
            return False

    return True


def _read_out_file_prefix(file_manager: Path) -> str:
    """Extract outFilePrefix from a SUMMA file manager."""
    with open(file_manager, 'r', encoding='utf-8') as f:
        for line in f:
            if 'outFilePrefix' in line and not line.strip().startswith('!'):
                return line.split("'")[1] if "'" in line else 'output'
    return 'output'


def cleanup_split_dirs(summa_dir: Path) -> None:
    """Remove gru_split_* subdirectories from a previous iteration."""
    for d in summa_dir.glob('gru_split_*'):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def run_summa_gru_parallel(
    summa_exe: Path,
    file_manager: Path,
    summa_dir: Path,
    settings_dir: Path,
    num_parallel: int,
    logger: logging.Logger,
    debug_info: Dict[str, Any],
    timeout: int = 7200,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Run SUMMA locally with GRU-split execution.

    This launches multiple SUMMA subprocesses on the current machine. It does
    not submit SLURM jobs.

    1. Read GRU count from attributes.nc
    2. Compute GRU splits
    3. Create per-split output directories and file managers
    4. Launch all splits concurrently via ThreadPoolExecutor
    5. Wait for all with a total timeout
    6. Merge outputs into standard files
    7. Clean up split directories on success
    """
    # 0. Clean stale splits from previous iterations
    cleanup_split_dirs(summa_dir)

    # 1. GRU count
    try:
        total_grus = get_gru_count_from_attributes(settings_dir)
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Cannot read GRU count: %s", exc)
        return False

    # 2. Splits
    num_parallel = max(1, min(num_parallel, total_grus))
    splits = compute_gru_splits(total_grus, num_parallel)
    logger.info(
        "Parallel SUMMA: %d GRUs across %d processes (%d-%d GRUs each)",
        total_grus, len(splits), splits[-1][1], splits[0][1],
    )

    # 3. Prepare
    experiment_prefix = _read_out_file_prefix(file_manager)
    split_dirs: list[Path] = []
    split_args_list: list[dict] = []

    log_dir = summa_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    iteration = debug_info.get('iteration', 0)

    for split_id, (start_gru, num_gru) in enumerate(splits):
        split_dir = summa_dir / f'gru_split_{split_id:02d}'
        split_dirs.append(split_dir)

        split_fm = create_split_file_manager(
            file_manager, split_dir, split_id, logger,
        )

        split_args_list.append({
            'summa_exe': str(summa_exe),
            'file_manager': str(split_fm),
            'start_gru': start_gru,
            'num_gru': num_gru,
            'output_dir': str(split_dir),
            'log_file': str(
                log_dir / f'summa_split_{split_id:02d}_iter{iteration:05d}.log'
            ),
            'timeout': timeout,
            'env': env,
            'split_id': split_id,
        })

    # 4. Execute
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(splits)) as executor:
        futures = {
            executor.submit(_run_single_gru_split, args): args['split_id']
            for args in split_args_list
        }

        done, not_done = concurrent.futures.wait(
            futures, timeout=timeout,
            return_when=concurrent.futures.ALL_COMPLETED,
        )

        if not_done:
            logger.error("%d GRU splits timed out", len(not_done))
            for future in not_done:
                future.cancel()
            debug_info.setdefault('errors', []).append(
                f'GRU-parallel timeout: {len(not_done)} splits incomplete'
            )
            return False

        for future in done:
            result = future.result()
            if not result['success']:
                logger.error(
                    "Split %d failed: %s", result['split_id'], result.get('error'),
                )
                debug_info.setdefault('errors', []).append(
                    f"GRU split {result['split_id']} failed: {result.get('error')}"
                )
                return False

    wall_time = time.time() - start_time
    logger.info("All %d GRU splits completed in %.1fs", len(splits), wall_time)

    # 5. Merge
    if not merge_split_outputs(split_dirs, summa_dir, experiment_prefix, logger):
        debug_info.setdefault('errors', []).append('GRU split output merge failed')
        return False

    # 6. Cleanup on success
    cleanup_split_dirs(summa_dir)

    return True
