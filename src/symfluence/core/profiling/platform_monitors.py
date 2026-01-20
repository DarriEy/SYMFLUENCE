"""
Platform-specific I/O monitoring implementations.

Provides I/O monitoring for different operating systems:
- Linux: Uses /proc/PID/io for accurate syscall tracking
- macOS: Uses psutil for I/O counters (if available)
- Fallback: File-based tracking only

All monitors provide a consistent interface for the SystemIOProfiler.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class ProcessIOMonitor(ABC):
    """Abstract base class for platform-specific I/O monitoring."""

    def __init__(self, pid: int, sample_interval: float = 0.5):
        """
        Initialize process monitor.

        Args:
            pid: Process ID to monitor
            sample_interval: How often to sample I/O stats (seconds)
        """
        self.pid = pid
        self.sample_interval = sample_interval
        self.samples: List[Dict[str, Any]] = []
        self.running = False
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def _read_io_stats(self) -> Optional[Dict[str, Any]]:
        """Read I/O statistics from the OS. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_platform_name(self) -> str:
        """Return the platform name for logging."""
        pass

    def start(self):
        """Start monitoring in background thread."""
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated stats."""
        self.running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=2.0)
        return self._aggregate_samples()

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                sample = self._read_io_stats()
                if sample:
                    self.samples.append(sample)
            except Exception as e:
                self.logger.debug(f"Error sampling process {self.pid}: {e}")
            time.sleep(self.sample_interval)

    def _aggregate_samples(self) -> Dict[str, Any]:
        """Aggregate samples into final statistics."""
        if not self.samples:
            return {}

        first = self.samples[0]
        last = self.samples[-1]

        # Calculate deltas (last - first)
        result = {
            'read_bytes': last.get('read_bytes', 0) - first.get('read_bytes', 0),
            'write_bytes': last.get('write_bytes', 0) - first.get('write_bytes', 0),
            'read_syscalls': last.get('read_syscalls', 0) - first.get('read_syscalls', 0),
            'write_syscalls': last.get('write_syscalls', 0) - first.get('write_syscalls', 0),
            'duration': last['timestamp'] - first['timestamp'],
            'max_rss_pages': max(s.get('rss_pages', 0) for s in self.samples),
            'sample_count': len(self.samples),
            'platform': self.get_platform_name(),
        }

        # Calculate IOPS
        if result['duration'] > 0:
            result['read_iops'] = result['read_syscalls'] / result['duration']
            result['write_iops'] = result['write_syscalls'] / result['duration']
            result['total_iops'] = (result['read_syscalls'] + result['write_syscalls']) / result['duration']
            result['read_mbps'] = (result['read_bytes'] / 1024 / 1024) / result['duration']
            result['write_mbps'] = (result['write_bytes'] / 1024 / 1024) / result['duration']

        return result


class LinuxProcessIOMonitor(ProcessIOMonitor):
    """Linux-specific I/O monitoring using /proc/PID/io."""

    def get_platform_name(self) -> str:
        return "Linux"

    def _read_io_stats(self) -> Optional[Dict[str, Any]]:
        """Read I/O stats from /proc/PID/io."""
        io_file = Path(f"/proc/{self.pid}/io")

        if not io_file.exists():
            return None

        try:
            stats = {'timestamp': time.time()}
            with open(io_file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    stats[key] = int(value)

            # Map /proc/PID/io names to our standard names
            stats['read_bytes'] = stats.get('read_bytes', 0)
            stats['write_bytes'] = stats.get('write_bytes', 0)
            stats['read_syscalls'] = stats.get('syscr', 0)
            stats['write_syscalls'] = stats.get('syscw', 0)

            # Also get memory stats
            statm_file = Path(f"/proc/{self.pid}/statm")
            if statm_file.exists():
                with open(statm_file, 'r') as f:
                    fields = f.read().split()
                    # Second field is RSS in pages
                    stats['rss_pages'] = int(fields[1]) if len(fields) > 1 else 0

            return stats

        except Exception as e:
            self.logger.debug(f"Error reading /proc/{self.pid}/io: {e}")
            return None


class MacOSProcessIOMonitor(ProcessIOMonitor):
    """macOS-specific I/O monitoring using psutil."""

    def __init__(self, pid: int, sample_interval: float = 0.5):
        super().__init__(pid, sample_interval)

        # Try to import psutil
        self.psutil = None
        self.process = None
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process(pid)
            self.logger.info(f"macOS profiler initialized with psutil for PID {pid}")
        except ImportError:
            self.logger.warning(
                "psutil not available - macOS I/O monitoring will use file-based fallback only. "
                "Install psutil for better metrics: pip install psutil"
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize psutil for PID {pid}: {e}")

    def get_platform_name(self) -> str:
        return "macOS" if self.psutil else "macOS (limited)"

    def _read_io_stats(self) -> Optional[Dict[str, Any]]:
        """Read I/O stats using psutil."""
        if not self.process or not self.psutil:
            return None

        try:
            stats = {'timestamp': time.time()}

            # Get I/O counters
            # Note: On macOS, psutil.Process.io_counters() may not be available
            # depending on macOS version and permissions
            try:
                io_counters = self.process.io_counters()
                # psutil provides: read_count, write_count, read_bytes, write_bytes
                stats['read_bytes'] = io_counters.read_bytes
                stats['write_bytes'] = io_counters.write_bytes
                stats['read_syscalls'] = io_counters.read_count
                stats['write_syscalls'] = io_counters.write_count
            except (AttributeError, NotImplementedError):
                # io_counters not available on this macOS version
                stats['read_bytes'] = 0
                stats['write_bytes'] = 0
                stats['read_syscalls'] = 0
                stats['write_syscalls'] = 0

            # Get memory info
            try:
                mem_info = self.process.memory_info()
                # Convert RSS from bytes to pages (assuming 4KB pages)
                stats['rss_pages'] = mem_info.rss // 4096
            except Exception:
                stats['rss_pages'] = 0

            return stats

        except self.psutil.NoSuchProcess:
            self.logger.debug(f"Process {self.pid} no longer exists")
            return None
        except Exception as e:
            self.logger.debug(f"Error reading psutil stats for {self.pid}: {e}")
            return None


class FallbackProcessIOMonitor(ProcessIOMonitor):
    """Fallback monitor that tracks files only (no I/O metrics)."""

    def get_platform_name(self) -> str:
        return f"{sys.platform} (fallback)"

    def _read_io_stats(self) -> Optional[Dict[str, Any]]:
        """No I/O stats available - return minimal data."""
        try:
            # Check if process still exists
            os.kill(self.pid, 0)

            stats = {
                'timestamp': time.time(),
                'read_bytes': 0,
                'write_bytes': 0,
                'read_syscalls': 0,
                'write_syscalls': 0,
                'rss_pages': 0,
            }
            return stats
        except OSError:
            # Process doesn't exist
            return None


def create_process_monitor(pid: int, sample_interval: float = 0.5) -> ProcessIOMonitor:
    """
    Factory function to create the appropriate process monitor for the current platform.

    Args:
        pid: Process ID to monitor
        sample_interval: Sampling interval in seconds

    Returns:
        Platform-specific ProcessIOMonitor instance
    """
    logger = logging.getLogger(__name__)

    # Linux: Use /proc/PID/io
    if sys.platform.startswith('linux'):
        logger.debug(f"Creating Linux process monitor for PID {pid}")
        return LinuxProcessIOMonitor(pid, sample_interval)

    # macOS: Try psutil first, fall back if not available
    elif sys.platform == 'darwin':
        from importlib.util import find_spec
        if find_spec("psutil") is not None:
            logger.debug(f"Creating macOS process monitor (psutil) for PID {pid}")
            return MacOSProcessIOMonitor(pid, sample_interval)
        else:
            logger.info(
                "psutil not available on macOS - using fallback monitor. "
                "Install psutil for I/O metrics: pip install psutil"
            )
            return FallbackProcessIOMonitor(pid, sample_interval)

    # Other platforms: Fallback
    else:
        logger.info(f"Platform {sys.platform} - using fallback monitor (file tracking only)")
        return FallbackProcessIOMonitor(pid, sample_interval)


def get_platform_capabilities() -> Dict[str, Any]:
    """
    Check what I/O monitoring capabilities are available on this platform.

    Returns:
        Dictionary of capability flags
    """
    capabilities = {
        'io_bytes': False,
        'io_syscalls': False,
        'memory_tracking': False,
        'platform': sys.platform,
        'monitor_type': 'fallback',
    }

    # Linux capabilities
    if sys.platform.startswith('linux'):
        proc_io = Path('/proc/self/io')
        if proc_io.exists():
            capabilities['io_bytes'] = True
            capabilities['io_syscalls'] = True
            capabilities['memory_tracking'] = True
            capabilities['monitor_type'] = 'linux_proc'

    # macOS capabilities
    elif sys.platform == 'darwin':
        try:
            import psutil
            # Test if io_counters is available
            try:
                p = psutil.Process()
                p.io_counters()  # Test call - don't need the result
                capabilities['io_bytes'] = True
                capabilities['io_syscalls'] = True
                capabilities['memory_tracking'] = True
                capabilities['monitor_type'] = 'macos_psutil'
            except (AttributeError, NotImplementedError):
                # psutil installed but io_counters not available
                capabilities['memory_tracking'] = True
                capabilities['monitor_type'] = 'macos_psutil_limited'
        except ImportError:
            capabilities['monitor_type'] = 'macos_fallback'

    return capabilities


logger = logging.getLogger(__name__)


def log_platform_capabilities():
    """Log available platform capabilities for diagnostics."""
    caps = get_platform_capabilities()

    logger.info("System I/O Profiling Capabilities:")
    logger.info(f"  Platform: {caps['platform']}")
    logger.info(f"  Monitor Type: {caps['monitor_type']}")
    logger.info(f"  I/O Bytes Tracking: {'available' if caps['io_bytes'] else 'unavailable'}")
    logger.info(f"  I/O Syscalls Tracking: {'available' if caps['io_syscalls'] else 'unavailable'}")
    logger.info(f"  Memory Tracking: {'available' if caps['memory_tracking'] else 'unavailable'}")

    if not caps['io_bytes']:
        if sys.platform == 'darwin':
            logger.info("To enable I/O tracking on macOS: pip install psutil")
        elif sys.platform.startswith('win'):
            logger.info("Windows I/O tracking requires psutil: pip install psutil")
        else:
            logger.info("Full I/O tracking not available on this platform")


# Backward compatibility alias
print_platform_capabilities = log_platform_capabilities


if __name__ == '__main__':
    # Self-test with console output
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log_platform_capabilities()
