"""
System-Level I/O Profiler for External Tools

This module provides comprehensive I/O profiling for external executables
(SUMMA, mizuRoute, FUSE, etc.) that are invisible to Python-level profiling.

Tracks:
- Read/Write bytes from /proc/PID/io
- Read/Write operations (syscalls)
- File system sync operations
- Per-process CPU and memory usage
- Actual IOPS rates during execution

Usage:
    profiler = SystemIOProfiler()

    with profiler.profile_subprocess(
        command=['summa.exe', '-m', 'fileManager.txt'],
        component='summa_model',
        iteration=1
    ) as proc:
        result = proc.run()

    report = profiler.generate_report()
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime

from .platform_monitors import create_process_monitor, get_platform_capabilities


@dataclass
class ProcessIOStats:
    """I/O statistics for a single process execution."""
    command: str
    component: str
    iteration: Optional[int] = None
    pid: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    # /proc/PID/io metrics (Linux)
    read_bytes: int = 0
    write_bytes: int = 0
    read_syscalls: int = 0
    write_syscalls: int = 0

    # Calculated metrics
    read_iops: float = 0.0  # Read operations per second
    write_iops: float = 0.0  # Write operations per second
    total_iops: float = 0.0
    read_throughput_mbps: float = 0.0  # MB/s
    write_throughput_mbps: float = 0.0  # MB/s

    # CPU/Memory
    max_rss_mb: float = 0.0  # Maximum resident set size
    cpu_percent: float = 0.0

    # File operations detected
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    # Status
    success: bool = True
    return_code: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


# ProcessMonitor is now imported from platform_monitors module
# This provides platform-specific implementations


class SystemIOProfiler:
    """
    System-level I/O profiler for external process execution.

    Monitors external tools (SUMMA, mizuRoute, etc.) to capture:
    - Actual IOPS (I/O operations per second)
    - Read/Write bytes
    - CPU and memory usage
    - File creation/modification

    Thread-safe for use in multi-process optimization.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize system I/O profiler.

        Args:
            enabled: Whether profiling is active
        """
        self.enabled = enabled
        self._operations: List[ProcessIOStats] = []
        self._lock = threading.RLock()
        self._start_time = time.time()
        self.logger = logging.getLogger(__name__)

        # Check platform capabilities
        self.capabilities = get_platform_capabilities()
        self.platform = self.capabilities['platform']
        self.monitor_type = self.capabilities['monitor_type']

        if self.enabled:
            # Log platform info
            self.logger.info(
                f"System I/O profiler initialized: platform={self.platform}, "
                f"monitor={self.monitor_type}"
            )

            # Warn if limited capabilities
            if not self.capabilities['io_bytes']:
                self.logger.warning(
                    f"I/O byte tracking not available on {self.platform}. "
                    f"File creation tracking will still work. "
                    f"For full metrics on macOS, install psutil: pip install psutil"
                )

    @contextmanager
    def profile_subprocess(
        self,
        command: List[str],
        component: str,
        iteration: Optional[int] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        track_files: bool = True,
        output_dir: Optional[Path] = None
    ):
        """
        Context manager to profile a subprocess execution.

        Args:
            command: Command to execute
            component: Component name (e.g., 'summa', 'mizuroute')
            iteration: Calibration iteration number
            cwd: Working directory
            env: Environment variables
            track_files: Whether to track file creation/modification
            output_dir: Directory to monitor for new files

        Yields:
            SubprocessProfiler instance with run() method

        Example:
            with profiler.profile_subprocess(
                command=['summa.exe', '-m', 'fileManager.txt'],
                component='summa',
                iteration=1,
                output_dir=Path('/path/to/outputs')
            ) as proc:
                result = proc.run()
        """
        if not self.enabled:
            # Profiling disabled - return dummy profiler
            yield _DummySubprocessProfiler(command, cwd, env)
            return

        # Create subprocess profiler
        profiler = _SubprocessProfiler(
            command=command,
            component=component,
            iteration=iteration,
            cwd=cwd,
            env=env,
            track_files=track_files,
            output_dir=output_dir,
            logger=self.logger
        )

        try:
            yield profiler
        finally:
            # Record the stats
            if profiler.stats:
                self._record_operation(profiler.stats)

    def _record_operation(self, stats: ProcessIOStats):
        """Record process I/O statistics."""
        with self._lock:
            self._operations.append(stats)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive profiling statistics.

        Returns:
            Dictionary with aggregated statistics
        """
        with self._lock:
            if not self._operations:
                return {
                    'summary': {'total_operations': 0},
                    'by_component': {},
                    'by_iteration': {},
                }

            total_duration = time.time() - self._start_time

            # Aggregate by component
            by_component: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                'count': 0,
                'total_read_bytes': 0,
                'total_write_bytes': 0,
                'total_read_iops': 0.0,
                'total_write_iops': 0.0,
                'total_duration': 0.0,
                'files_created': 0,
            })

            # Aggregate by iteration
            by_iteration: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
                'read_bytes': 0,
                'write_bytes': 0,
                'read_iops': 0.0,
                'write_iops': 0.0,
                'duration': 0.0,
            })

            total_read_bytes = 0
            total_write_bytes = 0
            total_read_syscalls = 0
            total_write_syscalls = 0
            total_exec_time = 0

            for op in self._operations:
                # Component aggregation
                comp = by_component[op.component]
                comp['count'] += 1
                comp['total_read_bytes'] += op.read_bytes
                comp['total_write_bytes'] += op.write_bytes
                comp['total_read_iops'] += op.read_iops
                comp['total_write_iops'] += op.write_iops
                comp['total_duration'] += op.duration_seconds
                comp['files_created'] += len(op.files_created)

                # Iteration aggregation
                if op.iteration is not None:
                    iter_stats = by_iteration[op.iteration]
                    iter_stats['read_bytes'] += op.read_bytes
                    iter_stats['write_bytes'] += op.write_bytes
                    iter_stats['read_iops'] += op.read_iops
                    iter_stats['write_iops'] += op.write_iops
                    iter_stats['duration'] += op.duration_seconds

                # Totals
                total_read_bytes += op.read_bytes
                total_write_bytes += op.write_bytes
                total_read_syscalls += op.read_syscalls
                total_write_syscalls += op.write_syscalls
                total_exec_time += op.duration_seconds

            # Calculate overall IOPS
            avg_read_iops = total_read_syscalls / total_exec_time if total_exec_time > 0 else 0
            avg_write_iops = total_write_syscalls / total_exec_time if total_exec_time > 0 else 0

            return {
                'summary': {
                    'profiling_duration_seconds': total_duration,
                    'total_operations': len(self._operations),
                    'total_execution_time': total_exec_time,
                    'total_read_bytes': total_read_bytes,
                    'total_write_bytes': total_write_bytes,
                    'total_read_syscalls': total_read_syscalls,
                    'total_write_syscalls': total_write_syscalls,
                    'average_read_iops': avg_read_iops,
                    'average_write_iops': avg_write_iops,
                    'average_total_iops': avg_read_iops + avg_write_iops,
                    'peak_iops': max((op.total_iops for op in self._operations), default=0),
                },
                'by_component': {k: dict(v) for k, v in by_component.items()},
                'by_iteration': {k: dict(v) for k, v in by_iteration.items()},
            }

    def generate_report(
        self,
        output_path: Optional[str] = None,
        format: str = 'json'
    ) -> str:
        """
        Generate profiling report.

        Args:
            output_path: Path to save report
            format: 'json' or 'text'

        Returns:
            Report content as string
        """
        stats = self.get_statistics()

        if format == 'json':
            report = json.dumps(stats, indent=2, default=str)
        else:
            report = self._format_text_report(stats)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"System I/O profiling report written to: {output_path}")

        return report

    def _format_text_report(self, stats: Dict[str, Any]) -> str:
        """Format statistics as human-readable text."""
        lines = [
            "=" * 80,
            "SYSTEM I/O PROFILING REPORT (External Tools)",
            f"Generated: {datetime.now().isoformat()}",
            f"Platform: {self.platform} ({self.monitor_type})",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 40,
            f"Profiling duration:       {stats['summary']['profiling_duration_seconds']:.2f} seconds",
            f"Total subprocess calls:   {stats['summary']['total_operations']}",
            f"Total execution time:     {stats['summary']['total_execution_time']:.2f} seconds",
            f"Total bytes read:         {self._format_bytes(stats['summary']['total_read_bytes'])}",
            f"Total bytes written:      {self._format_bytes(stats['summary']['total_write_bytes'])}",
            f"Read syscalls:            {stats['summary']['total_read_syscalls']:,}",
            f"Write syscalls:           {stats['summary']['total_write_syscalls']:,}",
            f"Average Read IOPS:        {stats['summary']['average_read_iops']:.1f}",
            f"Average Write IOPS:       {stats['summary']['average_write_iops']:.1f}",
            f"Average Total IOPS:       {stats['summary']['average_total_iops']:.1f}",
            f"Peak IOPS:                {stats['summary']['peak_iops']:.1f}",
            "",
            "BY COMPONENT",
            "-" * 40,
        ]

        for component, comp_stats in sorted(stats['by_component'].items()):
            lines.extend([
                f"  {component}:",
                f"    Executions:       {comp_stats['count']}",
                f"    Read:             {self._format_bytes(comp_stats['total_read_bytes'])}",
                f"    Written:          {self._format_bytes(comp_stats['total_write_bytes'])}",
                f"    Avg Read IOPS:    {comp_stats['total_read_iops'] / max(comp_stats['count'], 1):.1f}",
                f"    Avg Write IOPS:   {comp_stats['total_write_iops'] / max(comp_stats['count'], 1):.1f}",
                f"    Files created:    {comp_stats['files_created']}",
                f"    Total time:       {comp_stats['total_duration']:.2f}s",
                ""
            ])

        if stats.get('by_iteration'):
            lines.extend([
                "BY ITERATION (first 10)",
                "-" * 40,
            ])
            for iteration in sorted(stats['by_iteration'].keys())[:10]:
                iter_stats = stats['by_iteration'][iteration]
                lines.append(
                    f"  Iter {iteration}: "
                    f"R:{self._format_bytes(iter_stats['read_bytes'])} "
                    f"W:{self._format_bytes(iter_stats['write_bytes'])} "
                    f"IOPS:{iter_stats['read_iops'] + iter_stats['write_iops']:.1f}"
                )

        lines.extend([
            "",
            "PLATFORM CAPABILITIES",
            "-" * 40,
        ])

        if self.capabilities['io_bytes']:
            lines.append("  ✓ I/O byte tracking available")
        else:
            lines.append("  ✗ I/O byte tracking not available")
            if self.platform == 'darwin':
                lines.append("    → Install psutil for full metrics: pip install psutil")

        if self.capabilities['io_syscalls']:
            lines.append("  ✓ I/O syscall counting available")
        else:
            lines.append("  ✗ I/O syscall counting not available")

        if self.capabilities['memory_tracking']:
            lines.append("  ✓ Memory usage tracking available")
        else:
            lines.append("  ✗ Memory usage tracking not available")

        lines.extend([
            "",
            "=" * 80,
            "END OF SYSTEM I/O REPORT",
            "=" * 80,
        ])

        return "\n".join(lines)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(num_bytes) < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} PB"

    def export_to_file(self, output_path: str) -> None:
        """Export raw operation data to JSON file."""
        with self._lock:
            data = {
                'operations': [op.to_dict() for op in self._operations],
                'start_time': self._start_time,
                'platform': sys.platform,
            }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class _SubprocessProfiler:
    """Internal class for profiling a single subprocess execution."""

    def __init__(
        self,
        command: List[str],
        component: str,
        iteration: Optional[int],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        track_files: bool,
        output_dir: Optional[Path],
        logger: logging.Logger
    ):
        self.command = command
        self.component = component
        self.iteration = iteration
        self.cwd = cwd
        self.env = env
        self.track_files = track_files
        self.output_dir = output_dir
        self.logger = logger
        self.stats: Optional[ProcessIOStats] = None
        self._files_before: set = set()

    def run(
        self,
        stdout=None,
        stderr=None,
        timeout: Optional[int] = None,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run the subprocess with profiling.

        Args:
            stdout: Stdout file handle or redirect
            stderr: Stderr file handle or redirect
            timeout: Timeout in seconds
            check: Raise exception on non-zero exit

        Returns:
            CompletedProcess instance
        """
        # Prepare stats object
        self.stats = ProcessIOStats(
            command=' '.join(str(c) for c in self.command),
            component=self.component,
            iteration=self.iteration,
        )

        # Track files before execution
        if self.track_files and self.output_dir:
            self._files_before = set(self.output_dir.glob('*')) if self.output_dir.exists() else set()

        # Merge environment
        run_env = os.environ.copy()
        if self.env:
            run_env.update(self.env)

        # Start the process
        start_time = time.time()
        self.stats.start_time = start_time

        try:
            # Launch process
            process = subprocess.Popen(
                self.command,
                stdout=stdout,
                stderr=stderr,
                cwd=str(self.cwd) if self.cwd else None,
                env=run_env,
            )

            self.stats.pid = process.pid

            # Start monitoring with platform-specific monitor
            monitor = create_process_monitor(process.pid, sample_interval=0.5)
            monitor.start()

            # Wait for completion
            try:
                return_code = process.wait(timeout=timeout)
            finally:
                # Stop monitoring
                monitor_stats = monitor.stop()

            # Record end time
            end_time = time.time()
            self.stats.end_time = end_time
            self.stats.duration_seconds = end_time - start_time
            self.stats.return_code = return_code
            self.stats.success = (return_code == 0)

            # Extract monitor stats
            if monitor_stats:
                self.stats.read_bytes = monitor_stats.get('read_bytes', 0)
                self.stats.write_bytes = monitor_stats.get('write_bytes', 0)
                self.stats.read_syscalls = monitor_stats.get('read_syscalls', 0)
                self.stats.write_syscalls = monitor_stats.get('write_syscalls', 0)
                self.stats.read_iops = monitor_stats.get('read_iops', 0.0)
                self.stats.write_iops = monitor_stats.get('write_iops', 0.0)
                self.stats.total_iops = monitor_stats.get('total_iops', 0.0)
                self.stats.read_throughput_mbps = monitor_stats.get('read_mbps', 0.0)
                self.stats.write_throughput_mbps = monitor_stats.get('write_mbps', 0.0)

                # Memory stats
                max_rss_pages = monitor_stats.get('max_rss_pages', 0)
                page_size = 4096  # Typical page size
                self.stats.max_rss_mb = (max_rss_pages * page_size) / (1024 * 1024)

                # Store platform info for reporting
                platform_name = monitor_stats.get('platform', 'unknown')
                if hasattr(self.stats, 'metadata'):
                    self.stats.metadata = {'platform': platform_name}
                else:
                    # Add metadata dict if needed for older ProcessIOStats
                    pass

            # Track files after execution
            if self.track_files and self.output_dir and self.output_dir.exists():
                files_after = set(self.output_dir.glob('*'))
                new_files = files_after - self._files_before
                self.stats.files_created = [str(f.name) for f in new_files]

            # Check if we should raise
            if check and return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    self.command,
                    None,
                    None
                )

            # Create CompletedProcess-like object
            return subprocess.CompletedProcess(
                args=self.command,
                returncode=return_code,
                stdout=None,
                stderr=None
            )

        except subprocess.TimeoutExpired:
            self.stats.success = False
            self.stats.error_message = f"Timeout after {timeout}s"
            raise
        except Exception as e:
            self.stats.success = False
            self.stats.error_message = str(e)
            raise


class _DummySubprocessProfiler:
    """Dummy profiler when profiling is disabled."""

    def __init__(self, command, cwd, env):
        self.command = command
        self.cwd = cwd
        self.env = env

    def run(self, stdout=None, stderr=None, timeout=None, check=True):
        """Run subprocess without profiling."""
        run_env = os.environ.copy()
        if self.env:
            run_env.update(self.env)

        return subprocess.run(
            self.command,
            stdout=stdout,
            stderr=stderr,
            cwd=str(self.cwd) if self.cwd else None,
            env=run_env,
            timeout=timeout,
            check=check
        )
