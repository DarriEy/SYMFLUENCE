# Cross-Platform System I/O Profiling

## Overview

SYMFLUENCE's system I/O profiler now supports **multiple platforms** with automatic detection and graceful degradation:

- ✅ **Linux**: Full I/O metrics via `/proc/PID/io`
- ✅ **macOS**: Execution tracking + memory monitoring via `psutil`
- ✅ **Other**: Fallback with file tracking only

**The profiler automatically detects your platform and uses the best available monitoring method.**

## Platform Support Matrix

| Feature | Linux | macOS (psutil) | macOS (fallback) | Other |
|---------|-------|----------------|------------------|-------|
| Execution tracking | ✅ | ✅ | ✅ | ✅ |
| Files created | ✅ | ✅ | ✅ | ✅ |
| Execution time | ✅ | ✅ | ✅ | ✅ |
| Memory usage | ✅ | ✅ | ❌ | ❌ |
| Read bytes | ✅ | ❌ | ❌ | ❌ |
| Write bytes | ✅ | ❌ | ❌ | ❌ |
| Read syscalls | ✅ | ❌ | ❌ | ❌ |
| Write syscalls | ✅ | ❌ | ❌ | ❌ |
| Real IOPS | ✅ | ❌ | ❌ | ❌ |

## How It Works

### Automatic Platform Detection

```python
from symfluence.core.profiling import get_system_profiler

profiler = get_system_profiler()
profiler.enabled = True

# Automatically uses:
# - Linux: LinuxProcessIOMonitor (/proc/PID/io)
# - macOS: MacOSProcessIOMonitor (psutil)
# - Other: FallbackProcessIOMonitor (file tracking only)
```

The profiler logs which monitor is being used:
```
System I/O profiler initialized: platform=darwin, monitor=macos_psutil_limited
```

### Platform-Specific Implementations

#### Linux (Best)
```
Monitor: LinuxProcessIOMonitor
Source: /proc/PID/io
Metrics:
  ✓ read_bytes, write_bytes
  ✓ syscr (read syscalls), syscw (write syscalls)
  ✓ Real IOPS calculations
  ✓ Memory usage (RSS)
```

#### macOS with psutil
```
Monitor: MacOSProcessIOMonitor
Source: psutil.Process()
Metrics:
  ✓ Execution time
  ✓ Files created
  ✓ Memory usage (RSS)
  ✗ I/O bytes (not available on macOS)
  ✗ Syscall counts (not available on macOS)
```

**Why no I/O metrics on macOS?**
- macOS's `psutil.Process.io_counters()` is not available on recent macOS versions
- Requires special permissions or kernel extensions
- This is a known limitation of macOS, not a bug in SYMFLUENCE

#### Fallback (No psutil)
```
Monitor: FallbackProcessIOMonitor
Source: File system analysis
Metrics:
  ✓ Execution time
  ✓ Files created
  ✗ All I/O metrics
  ✗ Memory usage
```

## Installation

### For Full macOS Support (Optional)

```bash
# Install psutil for better metrics on macOS
pip install psutil

# Or if using conda
conda install psutil
```

**Note**: Even with psutil, macOS won't provide I/O bytes/syscalls due to OS limitations.

### For Linux (Production)

No additional packages needed - uses built-in `/proc` filesystem.

## Usage

### Same Command, All Platforms

```bash
# Works on Linux, macOS, Windows, etc.
symfluence workflow step calibrate_model \
  --config your_config.yaml \
  --profile
```

### Checking Platform Capabilities

```python
from symfluence.core.profiling.platform_monitors import print_platform_capabilities

print_platform_capabilities()
```

Output on macOS:
```
System I/O Profiling Capabilities:
  Platform: darwin
  Monitor Type: macos_psutil_limited
  I/O Bytes Tracking: ✗
  I/O Syscalls Tracking: ✗
  Memory Tracking: ✓

To enable I/O tracking on macOS:
  Install psutil for full metrics: pip install psutil
```

Output on Linux:
```
System I/O Profiling Capabilities:
  Platform: linux
  Monitor Type: linux_proc
  I/O Bytes Tracking: ✓
  I/O Syscalls Tracking: ✓
  Memory Tracking: ✓
```

## Report Differences by Platform

### Linux Report
```
================================================================================
SYSTEM I/O PROFILING REPORT (External Tools)
Generated: 2026-01-17T14:00:00.000000
Platform: linux (linux_proc)
================================================================================

SUMMARY
----------------------------------------
Total subprocess calls:   5000
Total bytes read:         15.2 GB          ← Real data
Total bytes written:      3.5 GB           ← Real data
Read syscalls:            1,234,567        ← Real data
Write syscalls:           3,456,789        ← Real data
Average Read IOPS:        50.3             ← Real metric
Average Write IOPS:       303.6            ← Real metric
Total IOPS:              353.9             ← Real metric
Peak IOPS:                847.2            ← Real metric

PLATFORM CAPABILITIES
----------------------------------------
  ✓ I/O byte tracking available
  ✓ I/O syscall counting available
  ✓ Memory usage tracking available
```

### macOS Report
```
================================================================================
SYSTEM I/O PROFILING REPORT (External Tools)
Generated: 2026-01-17T14:00:00.000000
Platform: darwin (macos_psutil_limited)
================================================================================

SUMMARY
----------------------------------------
Total subprocess calls:   22
Total bytes read:         0.0 B            ← Not available on macOS
Total bytes written:      0.0 B            ← Not available on macOS
Read syscalls:            0                ← Not available on macOS
Write syscalls:           0                ← Not available on macOS
Average Read IOPS:        0.0              ← Not available on macOS
Average Write IOPS:       0.0              ← Not available on macOS
Total IOPS:              0.0               ← Not available on macOS

BY COMPONENT
----------------------------------------
  summa:
    Executions:       22                   ← Available!
    Files created:    22                   ← Available!
    Total time:       44.67s               ← Available!

PLATFORM CAPABILITIES
----------------------------------------
  ✗ I/O byte tracking not available
    → Install psutil for full metrics: pip install psutil
  ✗ I/O syscall counting not available
  ✓ Memory usage tracking available
```

## Development Workflow

### Local Development (macOS/Windows)
1. Develop and test your code locally
2. Use profiler to verify execution tracking works
3. File creation tracking confirms SUMMA is running

### Production Profiling (Linux HPC)
1. Deploy to FIR cluster (Linux)
2. Run with `--profile` flag
3. Get **real IOPS metrics** for optimization

## Best Practices

### ✅ Do This

```bash
# Test locally on macOS to verify integration
symfluence workflow step calibrate_model \
  --config test_small.yaml \
  --profile

# Deploy to Linux for real metrics
ssh fir_cluster
symfluence workflow step calibrate_model \
  --config production.yaml \
  --profile
```

### ❌ Don't Do This

```bash
# Don't rely on macOS metrics for IOPS optimization
# (They won't be available)

# Don't be surprised when macOS shows 0.0 B
# (This is expected and documented)
```

## Troubleshooting

### Q: Why does macOS show 0.0 IOPS?

**A:** macOS doesn't provide `io_counters()` in psutil. This is an OS limitation, not a bug. The profiler gracefully handles this and still tracks:
- Execution counts (22 SUMMA runs)
- Files created (22 files)
- Execution time (44.67s)
- Memory usage (if psutil available)

**Solution**: Run on Linux (FIR cluster) for real metrics.

### Q: Can I get I/O metrics on macOS?

**A:** Not through Python APIs. macOS restricts I/O monitoring to:
- System tools (Activity Monitor)
- Kernel extensions (requires admin)
- DTrace (requires permissions)

For calibration optimization, you need real IOPS data from Linux.

### Q: Do I need to change my code for different platforms?

**A:** No! The profiler automatically detects your platform:
```python
# Same code works everywhere
profiler = get_system_profiler()
with profiler.profile_subprocess(...) as proc:
    proc.run()
```

### Q: How do I know which monitor is being used?

**A:** Check the log output:
```
System I/O profiler initialized: platform=darwin, monitor=macos_psutil_limited
```

Or look at the report header:
```
Platform: darwin (macos_psutil_limited)
```

### Q: Should I install psutil on macOS?

**A:** It won't give you I/O metrics, but it will provide:
- Better memory tracking
- Cleaner error messages
- Future-proofing (if macOS adds io_counters support)

**Recommendation**: Yes, install it (`pip install psutil`).

## Technical Details

### Monitor Selection Logic

```python
def create_process_monitor(pid):
    if sys.platform.startswith('linux'):
        return LinuxProcessIOMonitor(pid)  # Uses /proc/PID/io

    elif sys.platform == 'darwin':
        try:
            import psutil
            return MacOSProcessIOMonitor(pid)  # Uses psutil
        except ImportError:
            return FallbackProcessIOMonitor(pid)  # File tracking only

    else:
        return FallbackProcessIOMonitor(pid)  # Other platforms
```

### Graceful Degradation

Each monitor implements the same interface:
```python
class ProcessIOMonitor(ABC):
    def start()  # Start monitoring
    def stop()   # Stop and return stats
    def _read_io_stats()  # Platform-specific implementation
```

If a monitor can't provide metrics, it returns 0 instead of failing.

### Why This Matters

**Local testing on macOS:**
- Verify your workflow works
- Confirm SUMMA executes correctly
- Check file creation
- Validate integration

**Production profiling on Linux:**
- Get real IOPS measurements
- Optimize SUMMA output configuration
- Stay within cluster limits
- Prevent cluster crashes

## Summary

| Platform | Use For | Limitations |
|----------|---------|-------------|
| **Linux** | Production profiling | None - full metrics available |
| **macOS** | Development & testing | No I/O bytes/syscalls (OS limitation) |
| **Other** | Basic testing | Limited metrics |

**Key Takeaway**: The profiler works everywhere, but Linux provides the real IOPS data you need for optimization.
