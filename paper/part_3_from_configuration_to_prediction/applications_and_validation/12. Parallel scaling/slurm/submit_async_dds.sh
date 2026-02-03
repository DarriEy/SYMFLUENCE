#!/bin/bash
#SBATCH --job-name=sym_async_dds
#SBATCH --output=logs/async_dds_%A_%a.out
#SBATCH --error=logs/async_dds_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --array=0-5
#SBATCH --account=def-yourpi

# =============================================================================
# 4.11.3 Async vs Sync DDS -- HPC
# Compares synchronous and asynchronous DDS at HPC scale.
# Submit with: sbatch submit_async_dds.sh
# =============================================================================

CONFIGS=(
    "sync_dds_np4.yaml"
    "sync_dds_np16.yaml"
    "sync_dds_np64.yaml"
    "async_dds_pool5_batch4.yaml"
    "async_dds_pool10_batch8.yaml"
    "async_dds_pool20_batch16.yaml"
)

CONFIG_FILE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "Async/Sync DDS: ${CONFIG_FILE}"
echo "Node: $(hostname), CPUs: $(nproc)"
echo "Date: $(date)"
echo "============================================"

module load python/3.11
module load gcc/12
source ~/envs/symfluence/bin/activate

STUDY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${STUDY_DIR}/configs/${CONFIG_FILE}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

START_TIME=$(date +%s%N)
symfluence workflow step calibrate_model --config "${CONFIG_PATH}"
EXIT_CODE=$?
END_TIME=$(date +%s%N)
ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))

echo "Wall-clock time: ${ELAPSED} ms (exit code: ${EXIT_CODE})"

RESULTS_FILE="${STUDY_DIR}/results/timing_hpc_async_dds.csv"
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "experiment,config_file,wall_clock_ms,exit_code,hostname,timestamp" > "${RESULTS_FILE}"
fi
echo "exp3_async_dds,${CONFIG_FILE},${ELAPSED},${EXIT_CODE},$(hostname),$(date -Iseconds)" >> "${RESULTS_FILE}"
