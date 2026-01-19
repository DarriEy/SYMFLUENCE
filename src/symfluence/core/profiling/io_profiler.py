"""
I/O Profiler for SYMFLUENCE.

Tracks file I/O operations, timing, and resource usage to help diagnose
IOPS bottlenecks on HPC shared filesystems during large calibration jobs.
"""

import os
import time
import json
import logging
import threading
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Iterator
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum


class IOOperationType(Enum):
    """Types of I/O operations tracked."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_OPEN = "file_open"
    FILE_CLOSE = "file_close"
    DIR_CREATE = "dir_create"
    DIR_LIST = "dir_list"
    FILE_COPY = "file_copy"
    FILE_DELETE = "file_delete"
    NETCDF_WRITE = "netcdf_write"
    NETCDF_READ = "netcdf_read"
    PICKLE_WRITE = "pickle_write"
    PICKLE_READ = "pickle_read"
    SYMLINK_CREATE = "symlink_create"


@dataclass
class IOOperation:
    """Record of a single I/O operation."""
    operation_type: IOOperationType
    path: str
    timestamp: float
    duration_seconds: float
    size_bytes: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    component: Optional[str] = None  # e.g., "parameter_manager", "model_executor"
    iteration: Optional[int] = None  # Calibration iteration number
    process_id: int = field(default_factory=os.getpid)
    thread_id: int = field(default_factory=lambda: threading.current_thread().ident)
    stack_trace: Optional[str] = None  # Optional call stack for debugging

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['operation_type'] = self.operation_type.value
        return result


@dataclass
class ComponentStats:
    """Aggregated statistics for a component."""
    component_name: str
    total_operations: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    total_time_seconds: float = 0.0
    operation_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'component_name': self.component_name,
            'total_operations': self.total_operations,
            'total_bytes_read': self.total_bytes_read,
            'total_bytes_written': self.total_bytes_written,
            'total_time_seconds': self.total_time_seconds,
            'operation_counts': dict(self.operation_counts),
            'errors': self.errors,
        }


class IOProfiler:
    """
    Comprehensive I/O profiler for SYMFLUENCE workflows.

    Tracks:
    - File read/write operations with timing and sizes
    - NetCDF-specific operations (high IOPS concern)
    - Pickle operations (MPI IPC)
    - Directory operations
    - Per-component and per-iteration statistics
    - IOPS rate calculations

    Thread-safe for use in multi-threaded contexts.
    """

    def __init__(self, enabled: bool = True, capture_stack_traces: bool = False):
        """
        Initialize the I/O profiler.

        Args:
            enabled: Whether profiling is active (can be toggled)
            capture_stack_traces: Whether to capture call stacks (expensive)
        """
        self.enabled = enabled
        self.capture_stack_traces = capture_stack_traces
        self._operations: List[IOOperation] = []
        self._component_stats: Dict[str, ComponentStats] = defaultdict(
            lambda: ComponentStats(component_name="unknown")
        )
        # Use RLock (reentrant lock) to allow nested locking from same thread
        # This is needed because get_statistics() calls get_current_iops() while holding the lock
        self._lock = threading.RLock()
        self._start_time = time.time()
        self._current_iteration: Optional[int] = None
        self._current_component: Optional[str] = None
        self._iteration_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {'operations': 0, 'bytes_written': 0, 'time_seconds': 0.0}
        )

        # High-frequency operation sampling (to reduce overhead)
        self._sample_rate = 1.0  # 1.0 = record all, 0.1 = record 10%
        self._operation_counter = 0

        # IOPS tracking
        self._iops_window_seconds = 1.0
        self._recent_ops: List[float] = []

        self.logger = logging.getLogger(__name__)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable profiling."""
        self.enabled = enabled

    def set_iteration(self, iteration: int) -> None:
        """Set the current calibration iteration for context."""
        self._current_iteration = iteration

    def set_component(self, component: str) -> None:
        """Set the current component for context."""
        self._current_component = component

    @contextmanager
    def component_context(self, component_name: str) -> Iterator[None]:
        """Context manager to set the current component."""
        previous = self._current_component
        self._current_component = component_name
        try:
            yield
        finally:
            self._current_component = previous

    @contextmanager
    def iteration_context(self, iteration: int) -> Iterator[None]:
        """Context manager to set the current iteration."""
        previous = self._current_iteration
        self._current_iteration = iteration
        try:
            yield
        finally:
            self._current_iteration = previous

    def _should_sample(self) -> bool:
        """Determine if this operation should be recorded (for sampling)."""
        if self._sample_rate >= 1.0:
            return True
        self._operation_counter += 1
        return (self._operation_counter % int(1 / self._sample_rate)) == 0

    def _record_operation(self, operation: IOOperation) -> None:
        """Record an I/O operation (thread-safe)."""
        if not self.enabled:
            return

        with self._lock:
            self._operations.append(operation)

            # Update component stats
            stats = self._component_stats[operation.component or "unknown"]
            stats.component_name = operation.component or "unknown"
            stats.total_operations += 1
            stats.total_time_seconds += operation.duration_seconds
            stats.operation_counts[operation.operation_type.value] += 1

            if not operation.success:
                stats.errors += 1

            if operation.size_bytes:
                if operation.operation_type in (
                    IOOperationType.FILE_READ, IOOperationType.NETCDF_READ,
                    IOOperationType.PICKLE_READ
                ):
                    stats.total_bytes_read += operation.size_bytes
                else:
                    stats.total_bytes_written += operation.size_bytes

            # Update iteration stats
            if operation.iteration is not None:
                iter_stats = self._iteration_stats[operation.iteration]
                iter_stats['operations'] += 1
                iter_stats['time_seconds'] += operation.duration_seconds
                if operation.size_bytes and operation.operation_type in (
                    IOOperationType.FILE_WRITE, IOOperationType.NETCDF_WRITE,
                    IOOperationType.PICKLE_WRITE
                ):
                    iter_stats['bytes_written'] += operation.size_bytes

            # Track IOPS
            self._recent_ops.append(operation.timestamp)

    @contextmanager
    def track_file_write(
        self,
        path: str,
        size_bytes: Optional[int] = None,
        component: Optional[str] = None
    ) -> Iterator[None]:
        """
        Context manager to track a file write operation.

        Args:
            path: File path being written
            size_bytes: Size of data being written (if known)
            component: Component name (uses current context if not provided)
        """
        if not self.enabled:
            yield
            return

        start = time.time()
        error_msg = None
        success = True
        stack = None

        if self.capture_stack_traces:
            stack = ''.join(traceback.format_stack()[:-1])

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start
            op = IOOperation(
                operation_type=IOOperationType.FILE_WRITE,
                path=str(path),
                timestamp=start,
                duration_seconds=duration,
                size_bytes=size_bytes,
                success=success,
                error_message=error_msg,
                component=component or self._current_component,
                iteration=self._current_iteration,
                stack_trace=stack,
            )
            self._record_operation(op)

    @contextmanager
    def track_file_read(
        self,
        path: str,
        size_bytes: Optional[int] = None,
        component: Optional[str] = None
    ) -> Iterator[None]:
        """Context manager to track a file read operation."""
        if not self.enabled:
            yield
            return

        start = time.time()
        error_msg = None
        success = True

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start
            op = IOOperation(
                operation_type=IOOperationType.FILE_READ,
                path=str(path),
                timestamp=start,
                duration_seconds=duration,
                size_bytes=size_bytes,
                success=success,
                error_message=error_msg,
                component=component or self._current_component,
                iteration=self._current_iteration,
            )
            self._record_operation(op)

    @contextmanager
    def track_netcdf_write(
        self,
        path: str,
        size_bytes: Optional[int] = None,
        component: Optional[str] = None
    ) -> Iterator[None]:
        """
        Context manager to track NetCDF write operations.

        NetCDF writes are a primary IOPS concern during calibration.
        """
        if not self.enabled:
            yield
            return

        start = time.time()
        error_msg = None
        success = True
        stack = None

        if self.capture_stack_traces:
            stack = ''.join(traceback.format_stack()[:-1])

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start
            # Try to get file size if not provided
            if size_bytes is None and success:
                try:
                    size_bytes = Path(path).stat().st_size
                except Exception:
                    pass

            op = IOOperation(
                operation_type=IOOperationType.NETCDF_WRITE,
                path=str(path),
                timestamp=start,
                duration_seconds=duration,
                size_bytes=size_bytes,
                success=success,
                error_message=error_msg,
                component=component or self._current_component,
                iteration=self._current_iteration,
                stack_trace=stack,
            )
            self._record_operation(op)

    @contextmanager
    def track_pickle_write(
        self,
        path: str,
        size_bytes: Optional[int] = None,
        component: Optional[str] = None
    ) -> Iterator[None]:
        """Context manager to track pickle write operations (MPI IPC)."""
        if not self.enabled:
            yield
            return

        start = time.time()
        error_msg = None
        success = True

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start
            if size_bytes is None and success:
                try:
                    size_bytes = Path(path).stat().st_size
                except Exception:
                    pass

            op = IOOperation(
                operation_type=IOOperationType.PICKLE_WRITE,
                path=str(path),
                timestamp=start,
                duration_seconds=duration,
                size_bytes=size_bytes,
                success=success,
                error_message=error_msg,
                component=component or self._current_component,
                iteration=self._current_iteration,
            )
            self._record_operation(op)

    @contextmanager
    def track_dir_create(
        self,
        path: str,
        component: Optional[str] = None
    ) -> Iterator[None]:
        """Context manager to track directory creation."""
        if not self.enabled:
            yield
            return

        start = time.time()
        error_msg = None
        success = True

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start
            op = IOOperation(
                operation_type=IOOperationType.DIR_CREATE,
                path=str(path),
                timestamp=start,
                duration_seconds=duration,
                success=success,
                error_message=error_msg,
                component=component or self._current_component,
                iteration=self._current_iteration,
            )
            self._record_operation(op)

    @contextmanager
    def track_file_copy(
        self,
        src_path: str,
        dst_path: str,
        size_bytes: Optional[int] = None,
        component: Optional[str] = None
    ) -> Iterator[None]:
        """Context manager to track file copy operations."""
        if not self.enabled:
            yield
            return

        start = time.time()
        error_msg = None
        success = True

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration = time.time() - start
            if size_bytes is None and success:
                try:
                    size_bytes = Path(dst_path).stat().st_size
                except Exception:
                    pass

            op = IOOperation(
                operation_type=IOOperationType.FILE_COPY,
                path=f"{src_path} -> {dst_path}",
                timestamp=start,
                duration_seconds=duration,
                size_bytes=size_bytes,
                success=success,
                error_message=error_msg,
                component=component or self._current_component,
                iteration=self._current_iteration,
            )
            self._record_operation(op)

    def record_operation(
        self,
        operation_type: IOOperationType,
        path: str,
        duration_seconds: float,
        size_bytes: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        component: Optional[str] = None
    ) -> None:
        """
        Directly record an I/O operation (for cases where context managers don't fit).

        Args:
            operation_type: Type of operation
            path: File/directory path
            duration_seconds: How long the operation took
            size_bytes: Size of data transferred (if applicable)
            success: Whether operation succeeded
            error_message: Error message if failed
            component: Component name
        """
        if not self.enabled:
            return

        op = IOOperation(
            operation_type=operation_type,
            path=str(path),
            timestamp=time.time(),
            duration_seconds=duration_seconds,
            size_bytes=size_bytes,
            success=success,
            error_message=error_message,
            component=component or self._current_component,
            iteration=self._current_iteration,
        )
        self._record_operation(op)

    def get_current_iops(self) -> float:
        """Calculate current IOPS rate (operations per second)."""
        current_time = time.time()
        cutoff = current_time - self._iops_window_seconds

        with self._lock:
            # Clean old entries
            self._recent_ops = [t for t in self._recent_ops if t >= cutoff]
            if not self._recent_ops:
                return 0.0

            return len(self._recent_ops) / self._iops_window_seconds

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive profiling statistics.

        Returns:
            Dictionary containing:
            - summary: Overall statistics
            - by_component: Stats broken down by component
            - by_operation_type: Stats broken down by operation type
            - by_iteration: Stats per calibration iteration
            - hotspots: Most I/O intensive paths
        """
        with self._lock:
            total_duration = time.time() - self._start_time
            total_ops = len(self._operations)

            # Calculate totals
            total_bytes_read = sum(
                op.size_bytes or 0 for op in self._operations
                if op.operation_type in (
                    IOOperationType.FILE_READ, IOOperationType.NETCDF_READ,
                    IOOperationType.PICKLE_READ
                )
            )
            total_bytes_written = sum(
                op.size_bytes or 0 for op in self._operations
                if op.operation_type in (
                    IOOperationType.FILE_WRITE, IOOperationType.NETCDF_WRITE,
                    IOOperationType.PICKLE_WRITE, IOOperationType.FILE_COPY
                )
            )
            total_io_time = sum(op.duration_seconds for op in self._operations)
            total_errors = sum(1 for op in self._operations if not op.success)

            # Operations by type
            ops_by_type: Dict[str, int] = defaultdict(int)
            for op in self._operations:
                ops_by_type[op.operation_type.value] += 1

            # Find hotspots (most frequently accessed paths)
            path_counts: Dict[str, int] = defaultdict(int)
            path_bytes: Dict[str, int] = defaultdict(int)
            for op in self._operations:
                # Normalize path for aggregation
                base_path = op.path.split(" -> ")[0] if " -> " in op.path else op.path
                path_counts[base_path] += 1
                if op.size_bytes:
                    path_bytes[base_path] += op.size_bytes

            # Sort hotspots by count
            hotspots = sorted(
                [
                    {
                        'path': path,
                        'operation_count': count,
                        'total_bytes': path_bytes.get(path, 0)
                    }
                    for path, count in path_counts.items()
                ],
                key=lambda x: x['operation_count'],  # type: ignore[return-value]
                reverse=True
            )[:20]  # Top 20 hotspots

            return {
                'summary': {
                    'profiling_duration_seconds': total_duration,
                    'total_operations': total_ops,
                    'total_bytes_read': total_bytes_read,
                    'total_bytes_written': total_bytes_written,
                    'total_io_time_seconds': total_io_time,
                    'average_iops': total_ops / total_duration if total_duration > 0 else 0,
                    'current_iops': self.get_current_iops(),
                    'total_errors': total_errors,
                    'io_time_percentage': (total_io_time / total_duration * 100) if total_duration > 0 else 0,
                },
                'by_component': {
                    name: stats.to_dict()
                    for name, stats in self._component_stats.items()
                },
                'by_operation_type': dict(ops_by_type),
                'by_iteration': dict(self._iteration_stats),
                'hotspots': hotspots,
            }

    def generate_report(
        self,
        output_path: Optional[str] = None,
        format: str = 'json'
    ) -> str:
        """
        Generate a profiling report.

        Args:
            output_path: Path to save report (if None, returns string)
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
            self.logger.info(f"Profiling report written to: {output_path}")

        return report

    def _format_text_report(self, stats: Dict[str, Any]) -> str:
        """Format statistics as human-readable text."""
        lines = [
            "=" * 80,
            "SYMFLUENCE I/O PROFILING REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 40,
            f"Profiling duration:     {stats['summary']['profiling_duration_seconds']:.2f} seconds",
            f"Total I/O operations:   {stats['summary']['total_operations']:,}",
            f"Total bytes read:       {self._format_bytes(stats['summary']['total_bytes_read'])}",
            f"Total bytes written:    {self._format_bytes(stats['summary']['total_bytes_written'])}",
            f"Total I/O time:         {stats['summary']['total_io_time_seconds']:.2f} seconds",
            f"Average IOPS:           {stats['summary']['average_iops']:.1f}",
            f"I/O time percentage:    {stats['summary']['io_time_percentage']:.1f}%",
            f"Total errors:           {stats['summary']['total_errors']}",
            "",
            "OPERATIONS BY TYPE",
            "-" * 40,
        ]

        for op_type, count in sorted(
            stats['by_operation_type'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {op_type:25s} {count:>10,}")

        lines.extend([
            "",
            "OPERATIONS BY COMPONENT",
            "-" * 40,
        ])

        for name, comp_stats in sorted(
            stats['by_component'].items(),
            key=lambda x: x[1]['total_operations'],
            reverse=True
        ):
            lines.append(f"  {name}")
            lines.append(f"    Operations:  {comp_stats['total_operations']:,}")
            lines.append(f"    Bytes read:  {self._format_bytes(comp_stats['total_bytes_read'])}")
            lines.append(f"    Bytes written: {self._format_bytes(comp_stats['total_bytes_written'])}")
            lines.append(f"    Time:        {comp_stats['total_time_seconds']:.2f}s")
            lines.append(f"    Errors:      {comp_stats['errors']}")

        if stats['hotspots']:
            lines.extend([
                "",
                "TOP I/O HOTSPOTS (by operation count)",
                "-" * 40,
            ])
            for i, hotspot in enumerate(stats['hotspots'][:10], 1):
                path = hotspot['path']
                if len(path) > 50:
                    path = "..." + path[-47:]
                lines.append(
                    f"  {i:2}. {path:50s} "
                    f"{hotspot['operation_count']:>8,} ops  "
                    f"{self._format_bytes(hotspot['total_bytes']):>10}"
                )

        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
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

    def clear(self) -> None:
        """Clear all recorded operations and reset statistics."""
        with self._lock:
            self._operations.clear()
            self._component_stats.clear()
            self._iteration_stats.clear()
            self._recent_ops.clear()
            self._start_time = time.time()
            self._current_iteration = None

    def export_to_file(self, output_path: str) -> None:
        """
        Export profiling data to a JSON file for cross-process aggregation.

        Used by worker processes to save their profiling data so the main
        process can aggregate it.

        Args:
            output_path: Path to write the profiling data
        """
        with self._lock:
            data = {
                'operations': [op.to_dict() for op in self._operations],
                'start_time': self._start_time,
                'process_id': os.getpid(),
            }

        with open(output_path, 'w') as f:
            json.dump(data, f)

    def import_from_file(self, input_path: str) -> None:
        """
        Import profiling data from a JSON file.

        Used by the main process to aggregate data from worker processes.

        Args:
            input_path: Path to read the profiling data from
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)

            with self._lock:
                for op_dict in data.get('operations', []):
                    # Reconstruct IOOperation from dict
                    op = IOOperation(
                        operation_type=IOOperationType(op_dict['operation_type']),
                        path=op_dict['path'],
                        timestamp=op_dict['timestamp'],
                        duration_seconds=op_dict['duration_seconds'],
                        size_bytes=op_dict.get('size_bytes'),
                        success=op_dict.get('success', True),
                        error_message=op_dict.get('error_message'),
                        component=op_dict.get('component'),
                        iteration=op_dict.get('iteration'),
                        process_id=op_dict.get('process_id', os.getpid()),
                        thread_id=op_dict.get('thread_id', 0),
                        stack_trace=op_dict.get('stack_trace'),
                    )
                    self._record_operation(op)
        except Exception as e:
            self.logger.warning(f"Failed to import profiling data from {input_path}: {e}")

    def aggregate_from_directory(self, profile_dir: str) -> int:
        """
        Aggregate profiling data from all worker profile files in a directory.

        Args:
            profile_dir: Directory containing worker profile files

        Returns:
            Number of files successfully imported
        """
        import glob as glob_module

        profile_dir = Path(profile_dir)
        if not profile_dir.exists():
            return 0

        files_imported = 0
        pattern = str(profile_dir / "worker_profile_*.json")

        for profile_file in glob_module.glob(pattern):
            try:
                self.import_from_file(profile_file)
                files_imported += 1
                # Clean up after importing
                try:
                    os.remove(profile_file)
                except Exception:
                    pass
            except Exception as e:
                self.logger.warning(f"Failed to import {profile_file}: {e}")

        return files_imported
