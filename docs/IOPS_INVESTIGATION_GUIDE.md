# SYMFLUENCE IOPS Investigation Guide for HPC Systems

## Problem Statement
With 50 MPI processes and 5,000 model evaluations (100 generations × 50 members), the current profiling system only captures Python-level I/O (~22 operations). SUMMA and mizuRoute external executables generate massive I/O that's invisible to the profiler.

## Root Cause Analysis

### 1. External Tool I/O (Untracked)
Each SUMMA execution performs:
- **NetCDF writes**: Multiple timestep output files (potentially MB-GB each)
- **Restart files**: State variables for model continuation
- **History files**: Model diagnostic outputs
- **Log files**: Runtime information
- **Input reads**: Forcing data, parameter files

**At 50 concurrent processes**: This creates destructive I/O spikes

### 2. Current Profiling Limitations
The profiler tracks only:
```python
# These operations ARE tracked:
with profiler.track_netcdf_write("trialParams.nc"):
    write_params()  # ✓ Tracked

# These operations are NOT tracked:
subprocess.run(["summa.exe", "-m", "fileManager.txt"])  # ✗ Not tracked
# SUMMA writes *_timestep.nc, *_history.nc, restart files, etc.
```

## System-Level I/O Monitoring Solutions

### Option 1: SLURM Job Accounting (Recommended for FIR)
```bash
# Check I/O stats for your job
sacct -j <JOB_ID> --format=JobID,MaxDiskRead,MaxDiskWrite,AveCPU,AveRSS

# Detailed I/O profiling with sacct
sacct -j <JOB_ID> --format=JobID,MaxDiskRead,MaxDiskWrite,MaxDiskReadNode,MaxDiskWriteNode
```

### Option 2: `iostat` Monitoring (Real-time)
```bash
# Monitor I/O during job execution (run on compute node)
iostat -x 5  # Report every 5 seconds

# Focus on IOPS metrics:
# - r/s: Read operations per second
# - w/s: Write operations per second  
# - await: Average wait time
```

### Option 3: `iotop` Process-Level Monitoring
```bash
# Monitor per-process I/O (requires sudo on most systems)
iotop -o -P -a

# Or use pidstat if available
pidstat -d 5  # Disk I/O stats every 5 seconds
```

### Option 4: `strace` for Detailed Syscall Tracing
```bash
# Trace all system calls from SUMMA (WARNING: generates huge logs)
strace -f -e trace=file,write,read -o strace.log summa.exe -m fileManager.txt

# Count I/O operations
grep -E "write|read" strace.log | wc -l
```

### Option 5: Custom Wrapper Script
Create `/Users/darrieythorsson/compHydro/code/SYMFLUENCE/scripts/io_monitor_wrapper.sh`:
```bash
#!/bin/bash
# Wrapper to track SUMMA I/O operations

SUMMA_EXE=$1
FILE_MANAGER=$2
OUTPUT_DIR=$3

# Start monitoring in background
(
    PID=$$
    while kill -0 $PID 2>/dev/null; do
        # Record I/O stats
        echo "$(date +%s) $(cat /proc/$PID/io)" >> $OUTPUT_DIR/io_stats.log
        sleep 1
    done
) &
MONITOR_PID=$!

# Run SUMMA
$SUMMA_EXE -m $FILE_MANAGER

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
```

## SUMMA-Specific I/O Optimizations

### Critical: Minimize SUMMA Output Frequency

**Current issue**: SUMMA may be writing outputs at every timestep during calibration

**Solution**: Edit `outputControl.txt` in your settings:
```
! outputControl.txt - Minimal output for calibration
! Variable name           | Output frequency
! -------------------------+------------------
scalarTotalRunoff          | 86400     ! Daily instead of hourly
scalarSWE                  | 86400     ! Daily SWE
! Comment out all other variables during calibration
! poreSpaceHeatContent     | 1         ! DON'T write every timestep!
! mLayerTemp               | 1         ! Causes massive I/O!
```

**Expected reduction**: 80-95% fewer writes if you switch from hourly (3600s) to daily (86400s)

### File Manager Settings
Check your `fileManager.txt`:
```
! Ensure these are on fast storage during calibration:
settingsPath     '/scratch/your_job/settings/'
outputPath       '/scratch/your_job/outputs/'
decisionsFile    'decisions.txt'          
outputControlFile 'outputControl_minimal.txt'  ! Use minimal version
```

### SUMMA Decision Choices to Reduce I/O
Edit `decisions.txt`:
```
! Output control decisions
output_timestep          ! Use explicit timestep instead of every step
outputFrequency          daily     ! or monthly for calibration
writeRestart             no        ! Disable restart files during calibration
writeHistoryVars         no        ! Disable history outputs
```

## Recommended IOPS Reduction Strategy

### Phase 1: Immediate Actions (Target: <100 IOPS)

1. **Create minimal output control file**:
```bash
cat > outputControl_calibration.txt << 'EOC'
! Calibration-only outputs - absolute minimum
scalarTotalRunoff | 86400
EOC
```

2. **Update fileManager.txt** to use it:
```
outputControlFile 'outputControl_calibration.txt'
```

3. **Disable unnecessary outputs in decisions.txt**:
```
writeRestart           no
writeHistoryVars       no  
```

### Phase 2: Measure I/O (Use System Tools)

```bash
# In your SLURM job script, add:
#!/bin/bash
#SBATCH --job-name=symfluence_cal
#SBATCH --nodes=1
#SBATCH --ntasks=50

# Start I/O monitoring
iostat -x 5 > iostat_log_$SLURM_JOB_ID.txt &
IOSTAT_PID=$!

# Run calibration
symfluence workflow step calibrate_model --config config.yaml

# Stop monitoring
kill $IOSTAT_PID

# Check final IOPS
sacct -j $SLURM_JOB_ID --format=JobID,MaxDiskRead,MaxDiskWrite
```

### Phase 3: Calculate IOPS Budget

With 50 processes and 100 IOPS limit:
- **Budget per process**: 2 IOPS
- **Per model evaluation**: If evaluation takes 60s, max 120 I/O operations total
- **SUMMA output files**: Should write ≤1-2 files per evaluation

**Reality check**:
- Default SUMMA: Writes 20-50 files per run (hourly outputs for a year)
- Minimal SUMMA: Writes 1-2 files per run (daily aggregated outputs)

## Enhanced Profiling Integration

### Add System-Level Profiling to SYMFLUENCE

Create new profiler module:
`src/symfluence/core/profiling/system_io_profiler.py`:
```python
import os
import subprocess
from pathlib import Path

class SystemIOProfiler:
    """Profile external process I/O using system tools"""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.iostat_log = None
        self.process = None
    
    def start_monitoring(self, output_dir: Path):
        """Start iostat monitoring in background"""
        if not self.enabled:
            return
        
        self.iostat_log = output_dir / "system_io_stats.txt"
        # Start iostat in background
        self.process = subprocess.Popen(
            ['iostat', '-x', '5'],
            stdout=open(self.iostat_log, 'w'),
            stderr=subprocess.STDOUT
        )
    
    def stop_monitoring(self):
        """Stop iostat and parse results"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            return self._parse_iostat_log()
    
    def _parse_iostat_log(self):
        """Extract IOPS from iostat log"""
        # Parse log and extract r/s + w/s metrics
        # Return summary statistics
        pass
```

### Modify Calibration to Use It

In `src/symfluence/models/summa/calibration/optimizer.py`:
```python
from symfluence.core.profiling import get_profiler
from symfluence.core.profiling.system_io_profiler import SystemIOProfiler

def run_calibration(...):
    # Existing Python-level profiling
    python_profiler = get_profiler()
    
    # NEW: System-level I/O profiling
    system_profiler = SystemIOProfiler(enabled=profiling_enabled)
    system_profiler.start_monitoring(output_dir)
    
    try:
        # Run calibration
        results = optimizer.run()
    finally:
        # Stop and report
        sys_stats = system_profiler.stop_monitoring()
        python_profiler.generate_report()
```

## Quick Diagnostic Commands

### Check Current SUMMA Output Configuration
```bash
# Find your outputControl file
find /path/to/settings/SUMMA -name "outputControl.txt" -exec head -20 {} \;

# Count requested variables (excluding comments)
grep -v '^!' /path/to/settings/SUMMA/outputControl.txt | grep -v '^$' | wc -l

# If this returns >5, you're writing too many variables
```

### Estimate SUMMA I/O Load
```bash
# Check size of SUMMA outputs from a single run
du -sh /path/to/simulations/run_*/process_*/SUMMA/

# Count NetCDF files per process
find /path/to/simulations/run_*/process_* -name "*.nc" | wc -l

# With 50 processes, multiply by 50 to estimate total files
```

### Monitor Active IOPS During Calibration
```bash
# SSH to compute node during job
ssh <compute_node>

# Watch real-time IOPS
watch -n 1 'iostat -x 1 1 | grep -E "w/s|r/s"'

# Look for w/s (writes/second) spikes >100
```

## Expected Outcomes

### Before Optimization
- SUMMA writes: 20-50 NetCDF files per evaluation
- Total I/O ops: ~1,000-2,500 per second across 50 processes
- IOPS to shared filesystem: 500-1000 (exceeds limit)

### After Optimization (Minimal Outputs)
- SUMMA writes: 1-2 NetCDF files per evaluation
- Total I/O ops: 50-100 per second across 50 processes  
- IOPS to shared filesystem: <100 (within limit)

## Action Items for FIR Cluster

1. **Immediate**: Create minimal outputControl.txt with only scalarTotalRunoff
2. **Day 1**: Add iostat monitoring to your job submission script
3. **Day 2**: Check sacct reports to verify IOPS reduction
4. **Week 1**: Implement SystemIOProfiler for ongoing monitoring

## Contact FIR Admins

When discussing with FIR admins, provide:
1. SLURM job ID with sacct I/O statistics
2. outputControl.txt configuration
3. Confirmation that local scratch is being used
4. Number of concurrent processes and output frequency

Ask them:
- Which filesystem paths are counted toward IOPS limit?
- Is $SLURM_TMPDIR on local disk or shared filesystem?
- Can they provide per-job IOPS metrics from their monitoring?
