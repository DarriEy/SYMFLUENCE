#!/bin/bash
# =============================================================================
# Submit all Section 4.11 HPC experiments
# Usage: cd slurm && bash submit_all.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Create log directory
mkdir -p logs

echo "Submitting Section 4.11 Parallel Scaling HPC experiments..."
echo ""

# 4.11.1: ProcessPool strong scaling (7 array tasks)
JOB1=$(sbatch --parsable submit_strong_processpool.sh)
echo "4.11.1 ProcessPool: Job ${JOB1} (array 0-6)"

# 4.11.2: MPI strong scaling (8 array tasks)
JOB2=$(sbatch --parsable submit_strong_mpi.sh)
echo "4.11.2 MPI:         Job ${JOB2} (array 0-7)"

# 4.11.3: Async DDS (6 array tasks, depends on nothing)
JOB3=$(sbatch --parsable submit_async_dds.sh)
echo "4.11.3 Async DDS:   Job ${JOB3} (array 0-5)"

# 4.11.4: JAX GPU (single job with GPU)
JOB4=$(sbatch --parsable submit_jax_gpu.sh)
echo "4.11.4 JAX GPU:     Job ${JOB4}"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${JOB1},${JOB2},${JOB3},${JOB4}"
echo ""
echo "After completion, collect results with:"
echo "  cd ../scripts && python analyze_scaling.py"
