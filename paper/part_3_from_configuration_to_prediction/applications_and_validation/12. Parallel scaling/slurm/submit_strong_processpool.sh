#!/bin/bash
#SBATCH --job-name=sym_pp_scaling
#SBATCH --output=logs/processpool_%A_%a.out
#SBATCH --error=logs/processpool_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --account=def-yourpi

# =============================================================================
# 4.11.1 Strong Scaling (ProcessPool) -- HPC single-node
# Runs DDS calibration with varying NUM_PROCESSES on a single node.
# Submit with: sbatch submit_strong_processpool.sh
# =============================================================================

# Worker counts indexed by SLURM array task ID
WORKER_COUNTS=(1 2 4 8 16 32 64)
NP=${WORKER_COUNTS[$SLURM_ARRAY_TASK_ID]}

# Request matching number of CPUs
#SBATCH --cpus-per-task=${NP}

echo "============================================"
echo "ProcessPool strong scaling: NP=${NP}"
echo "Node: $(hostname)"
echo "CPUs available: $(nproc)"
echo "Date: $(date)"
echo "============================================"

# Environment
module load python/3.11
module load gcc/12
source ~/envs/symfluence/bin/activate

# Paths
STUDY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${STUDY_DIR}/configs"
CONFIG_FILE="${CONFIG_DIR}/strong_processpool_np${NP}.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config not found: ${CONFIG_FILE}"
    echo "Run generate_configs.py --hpc first"
    exit 1
fi

# Run calibration with timing
echo "Starting calibration with ${NP} ProcessPool workers..."
START_TIME=$(date +%s%N)

symfluence workflow step calibrate_model --config "${CONFIG_FILE}"
EXIT_CODE=$?

END_TIME=$(date +%s%N)
ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
echo "Wall-clock time: ${ELAPSED} ms (exit code: ${EXIT_CODE})"

# Save timing to shared results file
RESULTS_FILE="${STUDY_DIR}/results/timing_hpc_processpool.csv"
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "experiment,config_file,num_processes,wall_clock_ms,exit_code,hostname,timestamp" > "${RESULTS_FILE}"
fi
echo "exp1_processpool,strong_processpool_np${NP}.yaml,${NP},${ELAPSED},${EXIT_CODE},$(hostname),$(date -Iseconds)" >> "${RESULTS_FILE}"

echo "Done."
