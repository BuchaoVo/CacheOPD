#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd ${PROJECT_DIR}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_TORCHVISION=1
export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

BASE_MODEL=/home/zbc/data/models/ProLLaMA

FOLD_TEACHER=${PROJECT_DIR}/conditional_teachers_local/foldability
SOL_TEACHER=${PROJECT_DIR}/conditional_teachers_local/sol
THERMO_TEACHER=${PROJECT_DIR}/conditional_teachers_local/thermo

run_one () {
  local NAME="$1"
  local CSV="$2"
  local OUT="$3"

  echo "============================================================"
  echo "[RUN] ${NAME}"
  echo "CSV=${CSV}"
  echo "OUT=${OUT}"
  echo "START=$(date)"
  echo "============================================================"

  python -u tools/offline_opd/offline_cache_diagnostic.py \
    --rollouts_csv "${CSV}" \
    --base_model "${BASE_MODEL}" \
    --teacher foldability:${FOLD_TEACHER} \
    --teacher solubility:${SOL_TEACHER} \
    --teacher thermostability:${THERMO_TEACHER} \
    --out_dir "${OUT}" \
    --top_k 64 \
    --max_samples 128 \
    --max_length 512 \
    --dtype fp16 \
    --device cuda \
    --include_reference

  echo "[DONE] ${NAME} at $(date)"
}

run_one "Conditional-OPD-rollout" \
  analysis_outputs/p0_current_eval/cohorts/Conditional-OPD.csv \
  analysis_outputs/offline_proteinopd/cache_diag/conditional_opd_rollout

run_one "Released-Adapter-rollout" \
  analysis_outputs/p0_current_eval/cohorts/Released-Adapter.csv \
  analysis_outputs/offline_proteinopd/cache_diag/released_adapter_rollout

echo "[ALL DONE] $(date)"
