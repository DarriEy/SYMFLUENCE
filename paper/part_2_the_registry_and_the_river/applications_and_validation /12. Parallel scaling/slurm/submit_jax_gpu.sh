#!/bin/bash
#SBATCH --job-name=sym_jax_gpu
#SBATCH --output=logs/jax_gpu_%j.out
#SBATCH --error=logs/jax_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=def-yourpi

# =============================================================================
# 4.11.4 JAX GPU Acceleration -- HPC with GPU
# Runs all JAX backend variants including GPU.
# Submit with: sbatch submit_jax_gpu.sh
# =============================================================================

echo "============================================"
echo "JAX GPU Acceleration Study"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "============================================"

module load python/3.11
module load gcc/12
module load cuda/12
source ~/envs/symfluence/bin/activate

STUDY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${STUDY_DIR}/configs"
RESULTS_FILE="${STUDY_DIR}/results/timing_hpc_jax.csv"

if [ ! -f "${RESULTS_FILE}" ]; then
    echo "experiment,config_file,wall_clock_ms,exit_code,hostname,gpu,timestamp" > "${RESULTS_FILE}"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "none")

# Run all JAX configurations sequentially
for CONFIG_FILE in jax_numpy.yaml jax_cpu_nojit.yaml jax_cpu_jit.yaml jax_gpu_jit.yaml jax_jit_np1.yaml jax_jit_np4.yaml jax_jit_np8.yaml; do
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "Skipping (not found): ${CONFIG_FILE}"
        continue
    fi

    echo ""
    echo "--- Running: ${CONFIG_FILE} ---"
    START_TIME=$(date +%s%N)
    symfluence workflow step calibrate_model --config "${CONFIG_PATH}"
    EXIT_CODE=$?
    END_TIME=$(date +%s%N)
    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))

    echo "  Wall-clock: ${ELAPSED} ms (exit: ${EXIT_CODE})"
    echo "exp4_jax,${CONFIG_FILE},${ELAPSED},${EXIT_CODE},$(hostname),${GPU_NAME},$(date -Iseconds)" >> "${RESULTS_FILE}"
done

echo ""
echo "All JAX experiments complete."
