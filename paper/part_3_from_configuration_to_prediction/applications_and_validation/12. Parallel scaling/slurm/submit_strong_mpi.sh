#!/bin/bash
#SBATCH --job-name=sym_mpi_scaling
#SBATCH --output=logs/mpi_%A_%a.out
#SBATCH --error=logs/mpi_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-7
#SBATCH --account=def-yourpi

# =============================================================================
# 4.11.2 Strong Scaling (MPI) -- HPC multi-node
# Runs DDS calibration with varying MPI ranks across nodes.
# Submit with: sbatch submit_strong_mpi.sh
# =============================================================================

# Rank counts and node allocations indexed by array task ID
RANK_COUNTS=(1 2 4 8 16 32 64 128)
NODE_COUNTS=(1 1 1 1 1 2  4   8)

NP=${RANK_COUNTS[$SLURM_ARRAY_TASK_ID]}
NNODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}

#SBATCH --ntasks=${NP}
#SBATCH --nodes=${NNODES}

echo "============================================"
echo "MPI strong scaling: NP=${NP}, Nodes=${NNODES}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# Environment
module load python/3.11
module load gcc/12
module load openmpi/4.1
source ~/envs/symfluence/bin/activate

# Paths
STUDY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${STUDY_DIR}/configs"
CONFIG_FILE="${CONFIG_DIR}/strong_mpi_np${NP}.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config not found: ${CONFIG_FILE}"
    echo "Run generate_configs.py --hpc first"
    exit 1
fi

# Run with MPI
echo "Starting calibration with ${NP} MPI ranks across ${NNODES} nodes..."
START_TIME=$(date +%s%N)

srun --ntasks=${NP} symfluence workflow step calibrate_model --config "${CONFIG_FILE}"
EXIT_CODE=$?

END_TIME=$(date +%s%N)
ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
echo "Wall-clock time: ${ELAPSED} ms (exit code: ${EXIT_CODE})"

# Save timing
RESULTS_FILE="${STUDY_DIR}/results/timing_hpc_mpi.csv"
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "experiment,config_file,num_processes,num_nodes,wall_clock_ms,exit_code,hostname,timestamp" > "${RESULTS_FILE}"
fi
echo "exp2_mpi,strong_mpi_np${NP}.yaml,${NP},${NNODES},${ELAPSED},${EXIT_CODE},$(hostname),$(date -Iseconds)" >> "${RESULTS_FILE}"

echo "Done."
