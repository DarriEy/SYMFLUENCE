#!/usr/bin/env python3
"""
Run the Section 4.11 Parallel Scaling Study.

Orchestrates parallel scaling experiments across six sub-experiments,
collecting wall-clock timing, resource utilization, and calibration
metrics for each configuration.

Usage:
    python run_scaling_study.py --part all                # All experiments
    python run_scaling_study.py --part 1                  # ProcessPool only
    python run_scaling_study.py --part 1 3 4              # Multiple parts
    python run_scaling_study.py --part all --platform hpc # HPC configs
    python run_scaling_study.py --part 1 --dry-run        # Preview
    python run_scaling_study.py --part 1 --repeats 3      # Multiple runs
"""

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Paths
STUDY_DIR = Path(__file__).parent.parent
CONFIG_DIR = STUDY_DIR / "configs"
RESULTS_DIR = STUDY_DIR / "results"
ANALYSIS_DIR = STUDY_DIR / "analysis"
LOG_DIR = STUDY_DIR / "logs"
SYMFLUENCE_CLI = "symfluence"


# =============================================================================
# Platform detection and reporting
# =============================================================================

def detect_platform() -> Dict[str, Any]:
    """Collect platform information for reproducibility reporting."""
    info = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count_physical": os.cpu_count(),
    }

    if HAS_PSUTIL:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)

    # Check for MPI
    try:
        result = subprocess.run(
            ["mpirun", "--version"], capture_output=True, text=True, timeout=5
        )
        info["mpi_version"] = result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["mpi_version"] = "not available"

    # Check for JAX
    try:
        import jax
        info["jax_version"] = jax.__version__
        info["jax_devices"] = str(jax.devices())
    except ImportError:
        info["jax_version"] = "not installed"

    # Check SYMFLUENCE version
    try:
        result = subprocess.run(
            [SYMFLUENCE_CLI, "--version"], capture_output=True, text=True, timeout=10
        )
        info["symfluence_version"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["symfluence_version"] = "unknown"

    return info


def save_platform_info(info: Dict[str, Any], filepath: Path) -> None:
    """Save platform information to JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(info, f, indent=2)


# =============================================================================
# Timing and execution
# =============================================================================

class TimingResult:
    """Container for a single timed experiment run."""

    def __init__(
        self,
        experiment: str,
        config_file: str,
        wall_clock_seconds: float,
        success: bool,
        num_processes: int,
        run_index: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
    ):
        self.experiment = experiment
        self.config_file = config_file
        self.wall_clock_seconds = wall_clock_seconds
        self.success = success
        self.num_processes = num_processes
        self.run_index = run_index
        self.metrics = metrics or {}
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment,
            "config_file": self.config_file,
            "wall_clock_seconds": self.wall_clock_seconds,
            "success": self.success,
            "num_processes": self.num_processes,
            "run_index": self.run_index,
            "timestamp": self.timestamp,
            "error": self.error,
            **self.metrics,
        }


class ScalingStudyRunner:
    """Orchestrates the parallel scaling study execution."""

    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = True,
        repeats: int = 1,
        platform_name: str = "laptop",
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.repeats = repeats
        self.platform_name = platform_name
        self.results: List[TimingResult] = []
        self.logger = self._setup_logging()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"scaling_study_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logger = logging.getLogger("scaling_study")
        logger.info(f"Logging to: {log_file}")
        return logger

    def run_command(
        self, cmd: List[str], description: str, timeout: int = 7200
    ) -> Tuple[bool, float, str]:
        """Execute a command and return (success, wall_clock_seconds, output)."""
        self.logger.info(f"{description}")
        self.logger.info(f"Command: {' '.join(cmd)}")

        if self.dry_run:
            self.logger.info("[DRY RUN] Would execute command")
            return True, 0.0, ""

        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            elapsed = time.perf_counter() - start
            if result.returncode != 0:
                self.logger.error(
                    f"Command failed (exit {result.returncode}): "
                    f"{result.stderr[:500]}"
                )
                return False, elapsed, result.stderr
            return True, elapsed, result.stdout
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            self.logger.error(f"Command timed out after {timeout}s")
            return False, elapsed, "TIMEOUT"
        except FileNotFoundError:
            self.logger.error(f"Command not found: {cmd[0]}")
            return False, 0.0, "COMMAND_NOT_FOUND"

    def run_calibration(
        self,
        config_file: str,
        experiment_name: str,
        use_mpi: bool = False,
        mpi_ranks: int = 1,
    ) -> TimingResult:
        """Run a single calibration experiment with timing."""
        config_path = CONFIG_DIR / config_file

        if not config_path.exists():
            self.logger.warning(f"Config not found: {config_file}")
            return TimingResult(
                experiment=experiment_name,
                config_file=config_file,
                wall_clock_seconds=0.0,
                success=False,
                num_processes=mpi_ranks if use_mpi else 1,
                error="Config file not found",
            )

        # Build the command
        if use_mpi and mpi_ranks > 1:
            cmd = [
                "mpirun", "-n", str(mpi_ranks),
                SYMFLUENCE_CLI, "workflow", "step", "calibrate_model",
                "--config", str(config_path),
            ]
        else:
            cmd = [
                SYMFLUENCE_CLI, "workflow", "step", "calibrate_model",
                "--config", str(config_path),
            ]

        description = f"[{experiment_name}] {config_file}"
        success, elapsed, output = self.run_command(cmd, description)

        # Extract calibration metrics from output if available
        metrics = self._parse_calibration_metrics(output)

        # Determine num_processes from config name
        np_val = mpi_ranks if use_mpi else self._extract_np(config_file)

        return TimingResult(
            experiment=experiment_name,
            config_file=config_file,
            wall_clock_seconds=elapsed,
            success=success,
            num_processes=np_val,
            metrics=metrics,
            error=output[:200] if not success else None,
        )

    def _extract_np(self, config_file: str) -> int:
        """Extract worker count from config filename."""
        import re
        match = re.search(r"np(\d+)", config_file)
        if match:
            return int(match.group(1))
        return 1

    def _parse_calibration_metrics(self, output: str) -> Dict[str, float]:
        """Attempt to parse KGE and other metrics from calibration output."""
        metrics = {}
        if not output:
            return metrics

        for line in output.split("\n"):
            line_lower = line.lower()
            if "best kge" in line_lower or "final kge" in line_lower:
                try:
                    val = float(line.split(":")[-1].strip())
                    metrics["best_kge"] = val
                except (ValueError, IndexError):
                    pass
        return metrics

    # =================================================================
    # Experiment runners
    # =================================================================

    def run_exp1_processpool(self) -> None:
        """4.11.1: Strong scaling with ProcessPool."""
        self.logger.info("\n" + "#" * 70)
        self.logger.info("# EXPERIMENT 1: Strong Scaling (ProcessPool)")
        self.logger.info("#" * 70)

        worker_counts = [1, 2, 4, 8]
        if self.platform_name == "hpc":
            worker_counts.extend([16, 32, 64])

        for np_val in worker_counts:
            config_file = f"strong_processpool_np{np_val}.yaml"
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file,
                    experiment_name="exp1_processpool",
                )
                result.run_index = run_idx
                self.results.append(result)

                if self.verbose and not self.dry_run:
                    self.logger.info(
                        f"  np={np_val}, run={run_idx}: "
                        f"{result.wall_clock_seconds:.1f}s "
                        f"({'OK' if result.success else 'FAIL'})"
                    )

    def run_exp2_mpi(self) -> None:
        """4.11.2: Strong scaling with MPI."""
        self.logger.info("\n" + "#" * 70)
        self.logger.info("# EXPERIMENT 2: Strong Scaling (MPI)")
        self.logger.info("#" * 70)

        rank_counts = [1, 2, 4, 8]
        if self.platform_name == "hpc":
            rank_counts.extend([16, 32, 64, 128])

        for np_val in rank_counts:
            config_file = f"strong_mpi_np{np_val}.yaml"
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file,
                    experiment_name="exp2_mpi",
                    use_mpi=True,
                    mpi_ranks=np_val,
                )
                result.run_index = run_idx
                self.results.append(result)

                if self.verbose and not self.dry_run:
                    self.logger.info(
                        f"  ranks={np_val}, run={run_idx}: "
                        f"{result.wall_clock_seconds:.1f}s "
                        f"({'OK' if result.success else 'FAIL'})"
                    )

    def run_exp3_async_dds(self) -> None:
        """4.11.3: Async vs synchronous DDS."""
        self.logger.info("\n" + "#" * 70)
        self.logger.info("# EXPERIMENT 3: Async vs Synchronous DDS")
        self.logger.info("#" * 70)

        # Synchronous baselines
        sync_configs = ["sync_dds_np4.yaml"]
        if self.platform_name == "hpc":
            sync_configs.extend(["sync_dds_np16.yaml", "sync_dds_np64.yaml"])

        for config_file in sync_configs:
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file, experiment_name="exp3_sync_dds"
                )
                result.run_index = run_idx
                self.results.append(result)

        # Async-DDS configurations
        async_configs = ["async_dds_pool5_batch4.yaml", "async_dds_pool10_batch8.yaml"]
        if self.platform_name == "hpc":
            async_configs.append("async_dds_pool20_batch16.yaml")

        for config_file in async_configs:
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file, experiment_name="exp3_async_dds"
                )
                result.run_index = run_idx
                self.results.append(result)

    def run_exp4_jax(self) -> None:
        """4.11.4: JAX acceleration benchmarks."""
        self.logger.info("\n" + "#" * 70)
        self.logger.info("# EXPERIMENT 4: JAX Acceleration")
        self.logger.info("#" * 70)

        # Backend comparison (single-process)
        backend_configs = [
            "jax_numpy.yaml",
            "jax_cpu_nojit.yaml",
            "jax_cpu_jit.yaml",
        ]
        if self.platform_name == "hpc":
            backend_configs.append("jax_gpu_jit.yaml")

        for config_file in backend_configs:
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file, experiment_name="exp4_jax_backend"
                )
                result.run_index = run_idx
                self.results.append(result)

        # Composability: JAX JIT + multiprocessing
        composability_configs = [
            "jax_jit_np1.yaml",
            "jax_jit_np4.yaml",
            "jax_jit_np8.yaml",
        ]
        for config_file in composability_configs:
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file, experiment_name="exp4_jax_composability"
                )
                result.run_index = run_idx
                self.results.append(result)

    def run_exp5_weak_scaling(self) -> None:
        """4.11.5: Weak scaling with increasing domain complexity."""
        self.logger.info("\n" + "#" * 70)
        self.logger.info("# EXPERIMENT 5: Weak Scaling")
        self.logger.info("#" * 70)

        domain_configs = [
            "weak_lumped.yaml",
            "weak_elevation.yaml",
        ]
        if self.platform_name == "hpc":
            domain_configs.extend([
                "weak_semidist.yaml",
                "weak_distributed.yaml",
            ])

        for config_file in domain_configs:
            for run_idx in range(self.repeats):
                result = self.run_calibration(
                    config_file, experiment_name="exp5_weak_scaling"
                )
                result.run_index = run_idx
                self.results.append(result)

    def run_exp6_ensemble(self) -> None:
        """4.11.6: Task-level ensemble parallelism."""
        self.logger.info("\n" + "#" * 70)
        self.logger.info("# EXPERIMENT 6: Task-Level Ensemble Parallelism")
        self.logger.info("#" * 70)

        models = ["hbv", "gr4j", "fuse", "jfuse"]
        model_configs = [f"ensemble_{m}.yaml" for m in models]

        # --- Mode A: Sequential execution (one model at a time, 4 workers) ---
        self.logger.info("\n--- Mode A: Sequential (4 workers per model) ---")
        seq_start = time.perf_counter()
        for config_file in model_configs:
            result = self.run_calibration(
                config_file, experiment_name="exp6_sequential"
            )
            self.results.append(result)
        seq_total = time.perf_counter() - seq_start

        # Record aggregate sequential timing
        self.results.append(TimingResult(
            experiment="exp6_sequential_total",
            config_file="all_models",
            wall_clock_seconds=seq_total if not self.dry_run else 0.0,
            success=True,
            num_processes=4,
        ))

        # --- Mode B: Parallel execution (all models concurrently, 1 worker each) ---
        self.logger.info("\n--- Mode B: Parallel (1 worker per model, 4 concurrent) ---")
        if self.dry_run:
            for config_file in model_configs:
                self.logger.info(f"  [DRY RUN] Would launch: {config_file}")
            self.results.append(TimingResult(
                experiment="exp6_parallel_total",
                config_file="all_models",
                wall_clock_seconds=0.0,
                success=True,
                num_processes=4,
            ))
        else:
            par_start = time.perf_counter()
            processes = []
            for config_file in model_configs:
                config_path = CONFIG_DIR / config_file
                cmd = [
                    SYMFLUENCE_CLI, "workflow", "step", "calibrate_model",
                    "--config", str(config_path),
                ]
                self.logger.info(f"  Launching: {config_file}")
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                processes.append((config_file, proc))

            # Wait for all to complete
            for config_file, proc in processes:
                stdout, stderr = proc.communicate(timeout=7200)
                elapsed = time.perf_counter() - par_start
                success = proc.returncode == 0
                self.results.append(TimingResult(
                    experiment="exp6_parallel",
                    config_file=config_file,
                    wall_clock_seconds=elapsed,
                    success=success,
                    num_processes=1,
                    error=stderr[:200] if not success else None,
                ))

            par_total = time.perf_counter() - par_start
            self.results.append(TimingResult(
                experiment="exp6_parallel_total",
                config_file="all_models",
                wall_clock_seconds=par_total,
                success=True,
                num_processes=4,
            ))

    # =================================================================
    # Result I/O
    # =================================================================

    def save_results(self) -> None:
        """Save all timing results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_DIR / f"timing_raw_{timestamp}.csv"

        if not self.results:
            self.logger.warning("No results to save")
            return

        fieldnames = sorted(
            set().union(*(r.to_dict().keys() for r in self.results))
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

        self.logger.info(f"Results saved to: {csv_path}")

        # Also save latest as a stable name for analysis scripts
        latest_path = RESULTS_DIR / "timing_raw.csv"
        with open(latest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

    def print_summary(self) -> None:
        """Print a summary table of results."""
        if not self.results:
            return

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info("SCALING STUDY SUMMARY")
        self.logger.info(f"{'=' * 70}")
        self.logger.info(
            f"{'Experiment':<25} {'Config':<35} {'NP':>4} "
            f"{'Time (s)':>10} {'Status':>8}"
        )
        self.logger.info("-" * 86)

        for r in self.results:
            status = "OK" if r.success else "FAIL"
            self.logger.info(
                f"{r.experiment:<25} {r.config_file:<35} {r.num_processes:>4} "
                f"{r.wall_clock_seconds:>10.1f} {status:>8}"
            )


# =============================================================================
# Study part definitions
# =============================================================================

STUDY_PARTS = {
    "1": ("4.11.1 Strong Scaling (ProcessPool)", "run_exp1_processpool"),
    "2": ("4.11.2 Strong Scaling (MPI)", "run_exp2_mpi"),
    "3": ("4.11.3 Async vs Sync DDS", "run_exp3_async_dds"),
    "4": ("4.11.4 JAX Acceleration", "run_exp4_jax"),
    "5": ("4.11.5 Weak Scaling", "run_exp5_weak_scaling"),
    "6": ("4.11.6 Ensemble Parallelism", "run_exp6_ensemble"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Section 4.11 Parallel Scaling Study"
    )
    parser.add_argument(
        "--part", nargs="+", default=["all"],
        help="Experiment parts to run: 1-6 or 'all' (default: all)"
    )
    parser.add_argument(
        "--platform", choices=["laptop", "hpc"], default="laptop",
        help="Target platform (controls config selection)"
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of repeated runs per configuration (default: 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview commands without executing"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Enable verbose output"
    )
    args = parser.parse_args()

    # Determine which parts to run
    if "all" in args.part:
        parts = list(STUDY_PARTS.keys())
    else:
        parts = args.part

    # Initialize runner
    runner = ScalingStudyRunner(
        dry_run=args.dry_run,
        verbose=args.verbose,
        repeats=args.repeats,
        platform_name=args.platform,
    )

    # Record platform info
    platform_info = detect_platform()
    save_platform_info(
        platform_info, RESULTS_DIR / "platform_info.json"
    )
    runner.logger.info(f"Platform: {json.dumps(platform_info, indent=2)}")

    # Run experiments
    for part_id in parts:
        if part_id not in STUDY_PARTS:
            runner.logger.warning(f"Unknown part '{part_id}', skipping")
            continue

        name, method_name = STUDY_PARTS[part_id]
        runner.logger.info(f"\n{'#' * 70}")
        runner.logger.info(f"# PART {part_id}: {name}")
        runner.logger.info(f"{'#' * 70}")

        method = getattr(runner, method_name)
        method()

    # Save and summarise
    runner.save_results()
    runner.print_summary()


if __name__ == "__main__":
    main()
