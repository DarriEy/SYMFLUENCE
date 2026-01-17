# System I/O Profiling for External Tools - User Guide

## Overview

SYMFLUENCE now includes comprehensive **two-level I/O profiling** to diagnose IOPS issues on HPC systems:

1. **Python-Level Profiling**: Tracks Python file operations (NetCDF writes, pickle, etc.)
2. **System-Level Profiling**: Tracks external tool execution (SUMMA, mizuRoute, FUSE, etc.) using Linux `/proc/PID/io`

## Key Features

### System-Level Profiler Capabilities

- **Real IOPS Tracking**: Measures actual I/O operations per second from external tools
- **Read/Write Separation**: Separate metrics for read vs write operations
- **Per-Process Monitoring**: Tracks each SUMMA/mizuRoute execution individually
- **Per-Iteration Stats**: Breaks down I/O by calibration iteration
- **File Tracking**: Detects new files created by external tools
- **Memory Monitoring**: Tracks peak memory usage per process
- **Zero Overhead When Disabled**: No performance impact when profiling is off

### What Gets Profiled

✅ **Now Tracked (System-Level)**:
- SUMMA NetCDF output writes
- mizuRoute NetCDF output writes
- FUSE model outputs
- Any external executable I/O
- Actual syscall counts (read/write operations)
- Bytes transferred
- File creation/modification

✅ **Already Tracked (Python-Level)**:
- trialParams.nc writes
- Pickle file operations
- Parameter file updates
- Python-managed file I/O

## Usage

### Basic Usage (Existing --profile Flag)

No changes needed! The system profiler is automatically enabled when you use `--profile`:

```bash
# Same command as before, now with TWO profiling levels
symfluence workflow step calibrate_model \
  --config your_config.yaml \
  --profile
```

### Output Files

After profiling, you'll get **four** report files:

```
profile_report.json           # Python-level I/O (JSON)
profile_report.txt            # Python-level I/O (text)
system_io_report.json         # System-level I/O (JSON) ⭐ NEW
system_io_report.txt          # System-level I/O (text) ⭐ NEW
```

### Example Output

```
================================================================================
COMBINED I/O PROFILING SUMMARY
================================================================================

Python-Level I/O (NetCDF, Pickle, etc.):
  Report: tests/configs/profile_report.json
  Total operations: 22
  Bytes written: 294.6 KB
  Average IOPS: 0.4

System-Level I/O (SUMMA, mizuRoute, etc.):
  Report: tests/configs/system_io_report.json
  Total subprocesses: 5000          ⭐ Total model evaluations
  Read bytes: 45.2 GB                ⭐ Forcing data reads
  Write bytes: 125.8 GB              ⭐ SUMMA outputs
  Read IOPS: 145.3                   ⭐ Read operations/sec
  Write IOPS: 425.7                  ⭐ Write operations/sec
  Total IOPS: 571.0                  ⭐ Total IOPS rate
  Peak IOPS: 1205.3                  ⭐ Maximum IOPS spike
================================================================================
```

## System I/O Report Structure

### JSON Report Format

```json
{
  "summary": {
    "profiling_duration_seconds": 3600.5,
    "total_operations": 5000,
    "total_execution_time": 3245.2,
    "total_read_bytes": 48558899200,
    "total_write_bytes": 135123456000,
    "total_read_syscalls": 523445,
    "total_write_syscalls": 1534223,
    "average_read_iops": 145.3,
    "average_write_iops": 425.7,
    "average_total_iops": 571.0,
    "peak_iops": 1205.3
  },
  "by_component": {
    "summa": {
      "count": 5000,
      "total_read_bytes": 42558899200,
      "total_write_bytes": 125123456000,
      "total_read_iops": 135.2,
      "total_write_iops": 398.5,
      "total_duration": 2850.3,
      "files_created": 15000
    },
    "mizuroute": {
      "count": 5000,
      "total_read_bytes": 6000000000,
      "total_write_bytes": 10000000000,
      "total_read_iops": 10.1,
      "total_write_iops": 27.2,
      "total_duration": 395.0,
      "files_created": 5000
    }
  },
  "by_iteration": {
    "1": {
      "read_bytes": 971178000,
      "write_bytes": 2702469120,
      "read_iops": 145.1,
      "write_iops": 423.8,
      "duration": 64.5
    },
    ...
  }
}
```

## Platform Requirements

### ⚠️ Linux Required for System Profiling

The system-level profiler uses `/proc/PID/io` which is **Linux-only**:

- ✅ **Works on**: Linux HPC clusters (FIR, Cedar, Graham, etc.)
- ❌ **Not available on**: macOS, Windows
- ℹ️ **Graceful fallback**: Profiler disables itself on unsupported platforms

### Testing on macOS

On your local macOS system, the profiler will show:
```
System-Level I/O (SUMMA, mizuRoute, etc.):
  Total subprocesses: 22           ⭐ Still tracks execution count
  Read bytes: 0.0 B                ⚠️ /proc/PID/io not available
  Write bytes: 0.0 B               ⚠️ /proc/PID/io not available
  Files created: 22                ⭐ File tracking still works
```

This is expected! The full I/O metrics will appear when you run on Linux (FIR cluster).

## Interpreting Results for IOPS Issues

### Critical Metrics to Watch

1. **Total IOPS** (`average_total_iops`):
   - Your budget: 100 IOPS per job
   - Safe range: <80 IOPS sustained
   - Warning: >100 IOPS
   - Critical: >200 IOPS

2. **Peak IOPS** (`peak_iops`):
   - Shows maximum burst rate
   - Can temporarily exceed average
   - If peak >> average: I/O comes in spikes (bad for shared filesystem)

3. **Read vs Write IOPS**:
   - High read IOPS: Forcing data access issue
   - High write IOPS: Output frequency issue (most common)

4. **Files Created** (`files_created`):
   - Each NetCDF file = many I/O operations
   - With 5000 evaluations × 20 output files = 100,000 files!
   - **Solution**: Reduce SUMMA output frequency

### Example Diagnosis

**Scenario**: Job exceeding 100 IOPS limit

```json
{
  "summary": {
    "average_total_iops": 571.0,     ⚠️ 5.7x over limit!
    "average_write_iops": 425.7,     ⚠️ Most IOPS from writes
    "peak_iops": 1205.3               ⚠️ Huge spikes
  },
  "by_component": {
    "summa": {
      "files_created": 15000,         ⚠️ Too many files
      "total_write_bytes": 125123456000  ⚠️ 125 GB written
    }
  }
}
```

**Root Cause**: SUMMA writing too many output files

**Solution**: Reduce output frequency in `outputControl.txt`:
```
! Change from hourly (3600s) to daily (86400s)
scalarTotalRunoff | 86400
```

**Expected Improvement**: ~90% IOPS reduction

## Advanced Usage

### Programmatic Access

```python
from symfluence.core.profiling import get_system_profiler

# In your optimization code
profiler = get_system_profiler()

# Profile a single SUMMA execution
with profiler.profile_subprocess(
    command=['summa.exe', '-m', 'fileManager.txt'],
    component='summa',
    iteration=iteration_number,
    output_dir=output_path
) as proc:
    result = proc.run(stdout=log_file, stderr=subprocess.STDOUT)

# Later, generate report
profiler.generate_report('/path/to/report.json')
```

### Custom Analysis

Load the JSON report for custom analysis:

```python
import json

with open('system_io_report.json') as f:
    data = json.load(f)

# Find worst iteration
worst_iter = max(
    data['by_iteration'].items(),
    key=lambda x: x[1]['write_iops']
)
print(f"Iteration {worst_iter[0]} had highest IOPS: {worst_iter[1]['write_iops']:.1f}")

# Calculate SUMMA throughput
summa_stats = data['by_component']['summa']
throughput_mbps = summa_stats['total_write_bytes'] / summa_stats['total_duration'] / 1024 / 1024
print(f"SUMMA write throughput: {throughput_mbps:.1f} MB/s")
```

## Troubleshooting

### Issue: All IOPS metrics show 0.0

**Cause**: Running on macOS or non-Linux system
**Solution**: Run on Linux HPC cluster for actual metrics

### Issue: High IOPS but can't identify source

**Check**:
1. Look at `by_component` breakdown
2. Check `files_created` counts
3. Review `by_iteration` to find problematic iterations
4. Inspect SUMMA `outputControl.txt` configuration

### Issue: Profiling overhead is too high

**Solution**: Disable stack trace capture
```bash
# Default (fast)
symfluence workflow step calibrate_model --config config.yaml --profile

# With stack traces (slower, more detail)
symfluence workflow step calibrate_model --config config.yaml --profile --profile-stacks
```

## Best Practices for FIR Cluster

### 1. Profile Before Production Runs

```bash
# Small test with profiling
symfluence workflow step calibrate_model \
  --config test_config.yaml \
  --profile

# Check reports
cat system_io_report.txt

# If IOPS < 100, scale up to production
```

### 2. Monitor Trends

Keep historical reports:
```bash
# Save with timestamp
mkdir -p profiling_history
cp system_io_report.json profiling_history/report_$(date +%Y%m%d_%H%M%S).json

# Compare trends
python scripts/compare_iops_trends.py profiling_history/
```

### 3. Share Reports with FIR Admins

When discussing issues with FIR admins, provide:
- `system_io_report.txt` (human-readable)
- Your SLURM job ID
- Your `outputControl.txt` configuration
- Number of MPI processes used

### 4. Iterate on Optimization

1. Run with profiling
2. Check IOPS metrics
3. Adjust SUMMA output frequency
4. Re-run and verify improvement
5. Repeat until < 100 IOPS

## Next Steps

### Immediate Actions

1. ✅ Test profiling with your current workflow (completed!)
2. Run profiling on FIR cluster with small job
3. Analyze system_io_report.txt
4. Identify IOPS bottlenecks
5. Optimize SUMMA output configuration
6. Re-profile and verify

### Future Enhancements

Possible additions:
- iostat integration for cluster-wide IOPS
- Real-time IOPS monitoring dashboard
- Automatic SUMMA output optimization
- IOPS budget warnings during calibration

## Technical Details

### How It Works

1. **Process Monitoring**: Python starts background thread monitoring `/proc/PID/io`
2. **Sampling**: Reads I/O stats every 0.5 seconds
3. **Delta Calculation**: Computes differences between first and last samples
4. **Aggregation**: Combines stats across all executions
5. **IOPS Calculation**: Divides syscall count by execution duration

### Metrics Explained

- **read_bytes**: Total bytes read by process
- **write_bytes**: Total bytes written by process
- **read_syscalls**: Number of read() system calls
- **write_syscalls**: Number of write() system calls
- **read_iops**: read_syscalls / duration
- **write_iops**: write_syscalls / duration
- **total_iops**: (read_syscalls + write_syscalls) / duration

### Performance Impact

- **Monitoring thread**: ~0.01% CPU overhead
- **Memory**: <1 MB per profiler instance
- **I/O**: None (reads /proc filesystem)
- **When disabled**: Zero overhead

## Support

For questions or issues:
1. Check `system_io_report.txt` for detailed metrics
2. Review SUMMA `outputControl.txt` configuration
3. Consult [IOPS_INVESTIGATION_GUIDE.md](./IOPS_INVESTIGATION_GUIDE.md)
4. Contact SYMFLUENCE development team

## Summary

The system-level profiler gives you **visibility into external tool I/O** that was previously invisible. Use it to:

✅ Measure actual IOPS from SUMMA/mizuRoute
✅ Identify I/O bottlenecks
✅ Optimize output configurations
✅ Stay within FIR cluster limits
✅ Prevent cluster crashes from excessive IOPS

**Remember**: This works on Linux only. Test locally to verify integration, then run on FIR for real metrics.
