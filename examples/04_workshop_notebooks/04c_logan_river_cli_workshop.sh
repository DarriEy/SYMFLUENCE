#!/usr/bin/env bash
# =============================================================================
# SYMFLUENCE Workshop 04c — Logan River from the command line (SUMMA)
#
# Replicates the 04a Logan River workshop notebook without writing a single
# line of Python: configuration, watershed delineation, cloud data
# acquisition, SUMMA execution, and evaluation — all through the
# `symfluence` CLI.
#
# Usage:
#   ./04c_logan_river_cli_workshop.sh                 # setup through SUMMA run
#   RUN_CALIBRATION=1 ./04c_logan_river_cli_workshop.sh   # ... + DDS calibration
#   RUN_BENCHMARK=1  ./04c_logan_river_cli_workshop.sh    # ... + benchmarking
#
# See 04c_logan_river_cli_workshop.md for the guided walkthrough.
# =============================================================================
set -euo pipefail

DOMAIN=Logan_River_at_Logan
CONFIG=config_${DOMAIN}.yaml
RUN_CALIBRATION=${RUN_CALIBRATION:-0}
RUN_BENCHMARK=${RUN_BENCHMARK:-0}

step() { printf '\n\033[1;36m════ %s ════\033[0m\n' "$*"; }

# ─────────────────────────────────────────────────────────────────────────────
step "Step 0 — Environment check"
# ─────────────────────────────────────────────────────────────────────────────
# `doctor` reports missing tools without failing the run; install SUMMA with
#   symfluence binary install summa
# if it is not already available.
symfluence doctor || true

# ─────────────────────────────────────────────────────────────────────────────
step "Step 1 — Generate and localize the configuration"
# ─────────────────────────────────────────────────────────────────────────────
# `project init --minimal` writes a compact SUMMA config with sensible
# defaults; everything Logan-specific is appended below.
symfluence project init \
    --domain "$DOMAIN" \
    --model SUMMA \
    --forcing RDRS \
    --start-date 2018-01-01 \
    --end-date 2021-12-31 \
    --definition-method lumped \
    --discretization GRUs \
    --minimal

sed -i.bak \
    -e 's|^EXPERIMENT_ID:.*|EXPERIMENT_ID: cli_workshop_1|' \
    -e 's|^EXPERIMENT_TIME_START:.*|EXPERIMENT_TIME_START: 2018-01-01 01:00|' \
    "$CONFIG" && rm -f "$CONFIG.bak"

cat >> "$CONFIG" <<'EOF'

# ── Logan River at Logan, UT (USGS 10109000) ─────────────────────────────────
POUR_POINT_COORDS: 41.743098/-111.786432
BOUNDING_BOX_COORDS: 42.15/-111.90/41.70/-111.40
LUMPED_WATERSHED_METHOD: TauDEM
SPINUP_PERIOD: 2018-01-01, 2018-12-31
CALIBRATION_PERIOD: 2019-01-01, 2019-12-31
EVALUATION_PERIOD: 2020-01-01, 2020-12-31

# Data sources (all cloud-hosted — no manual downloads)
DATA_ACCESS: cloud
DEM_SOURCE: copdem90
FORCING_MEASUREMENT_HEIGHT: 2.0

# Streamflow observations
STATION_ID: '10109000'
STREAMFLOW_DATA_PROVIDER: USGS
DOWNLOAD_USGS_DATA: true

# Calibration (Step 5b — enable with RUN_CALIBRATION=1)
OPTIMIZATION_METHODS:
- iteration
OPTIMIZATION_TARGET: streamflow
ITERATIVE_OPTIMIZATION_ALGORITHM: DDS
OPTIMIZATION_METRIC: KGE
CALIBRATION_TIMESTEP: hourly
NUMBER_OF_ITERATIONS: 50
EOF

symfluence workflow validate --config "$CONFIG"

# ─────────────────────────────────────────────────────────────────────────────
step "Step 2 — Domain definition (delineate + discretize)"
# ─────────────────────────────────────────────────────────────────────────────
symfluence workflow steps \
    setup_project create_pour_point acquire_attributes \
    define_domain discretize_domain \
    --config "$CONFIG"

# ─────────────────────────────────────────────────────────────────────────────
step "Step 3 — Data acquisition and preprocessing"
# ─────────────────────────────────────────────────────────────────────────────
symfluence workflow steps \
    process_observed_data acquire_forcings model_agnostic_preprocessing \
    --config "$CONFIG"

# ─────────────────────────────────────────────────────────────────────────────
step "Step 4 — SUMMA setup, execution, and post-processing"
# ─────────────────────────────────────────────────────────────────────────────
symfluence workflow steps \
    model_specific_preprocessing run_model postprocess_results \
    --config "$CONFIG"

# ─────────────────────────────────────────────────────────────────────────────
step "Step 5 — Results"
# ─────────────────────────────────────────────────────────────────────────────
symfluence workflow status --config "$CONFIG" || true
DATA_DIR=$(awk -F': ' '/^SYMFLUENCE_DATA_DIR:/ {print $2; exit}' "$CONFIG")
echo
echo "Streamflow results: ${DATA_DIR}/domain_${DOMAIN}/results/cli_workshop_1_results.csv"

if [ "$RUN_CALIBRATION" = "1" ]; then
    step "Step 5b — Calibration (DDS, KGE, ${NUMBER_OF_ITERATIONS:-50} iterations)"
    symfluence workflow step calibrate_model --config "$CONFIG"
fi

if [ "$RUN_BENCHMARK" = "1" ]; then
    step "Step 6 — Benchmarking against reference predictors"
    symfluence workflow step run_benchmarking --config "$CONFIG"
fi

step "Done"
