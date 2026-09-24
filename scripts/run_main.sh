#!/bin/bash
set -euo pipefail

# Usage:
# bash scripts/run_training_sweep_algo.sh <Cleansed_version|no_clean> <Base_model> [Pretty_name_prefix]
#
# Example:
# bash scripts/run_training_sweep_algo.sh no_clean llama3_8b expA

CLEANSING=$1
BASE_MODEL=$2
PRETTY_PREFIX=${3:-""}

if [[ -z "${CLEANSING:-}" || -z "${BASE_MODEL:-}" ]]; then
  echo "Usage: bash scripts/run_training_sweep_algo.sh <Cleansed_version|no_clean> <Base_model> [Pretty_name_prefix]"
  exit 1
fi

# Algorithms to sweep
OPT_ALGOS=("DPO" "rDPO" "IPO")

echo "========================================"
echo "Calling run_training.sh with multiple OPT_ALGO"
echo " - Cleansing:   ${CLEANSING}"
echo " - Base model:  ${BASE_MODEL}"
echo " - Algorithms:  ${OPT_ALGOS[*]}"
echo "========================================"

for OPT_ALGO in "${OPT_ALGOS[@]}"; do
  if [[ -n "$PRETTY_PREFIX" ]]; then
    PRETTY_NAME="${PRETTY_PREFIX}_${OPT_ALGO}_${BASE_MODEL}_${CLEANSING}"
  else
    PRETTY_NAME="${OPT_ALGO}_${BASE_MODEL}_${CLEANSING}"
  fi

  echo "----------------------------------------"
  echo "Running OPT_ALGO=${OPT_ALGO}"
  echo "Pretty name: ${PRETTY_NAME}"
  echo "----------------------------------------"

  bash scripts/run_training.sh \
    "${CLEANSING}" \
    "${BASE_MODEL}" \
    "${OPT_ALGO}" \
    "${PRETTY_NAME}"
done

echo "========================================"
echo "All algorithm sweeps finished."
echo "========================================"
