#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd "${PROJECT_DIR}"

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONUNBUFFERED=1
export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}

CACHE_NAME=${CACHE_NAME:-base_rollout}
KEEP_RATIO=${KEEP_RATIO:-0.5}
KEEP_TAG=${KEEP_TAG:-50}

BASE_MODEL=/home/zbc/data/models/ProLLaMA
CACHE_DIR="analysis_outputs/offline_proteinopd/cache_diag/${CACHE_NAME}"
SELECTION_DIR="analysis_outputs/offline_proteinopd/contribution2/${CACHE_NAME}_keep${KEEP_TAG}"
OUT_PREFIX="outputs/offline_proteinopd/contrib2_${CACHE_NAME}_k64_keep${KEEP_TAG}"

echo "Start Contribution 2 token-selection experiment at $(date)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "SELECTION_DIR=${SELECTION_DIR}"
echo "OUT_PREFIX=${OUT_PREFIX}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -u tools/offline_opd/score_cache_token_utility.py \
  --cache_dir "${CACHE_DIR}" \
  --out_dir "${SELECTION_DIR}" \
  --keep_ratio "${KEEP_RATIO}" \
  --seed 42 \
  --alpha_shift 1.0 \
  --beta_support 0.8 \
  --beta_target 0.4 \
  --lambda_conflict 0.8 \
  --mu_collapse 0.2

run_one_selection () {
  local NAME="$1"
  local MASK="${SELECTION_DIR}/mask_${NAME}.npy"
  local OUT_DIR="${OUT_PREFIX}_${NAME}_gamma01"

  echo "============================================================"
  echo "[TRAIN] selection=${NAME}"
  echo "MASK=${MASK}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "START=$(date)"
  echo "============================================================"

  python -u tools/offline_opd/train_offline_opd.py \
    --base_model "${BASE_MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --token_mask "${MASK}" \
    --output_dir "${OUT_DIR}" \
    --max_samples 128 \
    --max_length 512 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-5 \
    --warmup_ratio 0.03 \
    --dtype bf16 \
    --device cuda \
    --gamma_nll 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_steps 20 \
    --log_steps 2

  echo "[DONE] selection=${NAME} at $(date)"
}

run_one_selection high
run_one_selection random
run_one_selection low

echo "[ALL DONE] Contribution 2 token-selection experiment finished at $(date)"
