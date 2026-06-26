#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_MODEL=/home/zbc/data/models/ProLLaMA
GEN_ROOT=/home/zbc/data/opd/ProteinOPD/outputs/conditional/offline_opd_4superfamily
METHOD_DIR=final_utility_shift_entropy_q02_high50
ADAPTER=/home/zbc/data/opd/ProteinOPD/outputs/offline_proteinopd/final_utility_shift_entropy_q02_high50_gamma01
NUM_SEQUENCES=${NUM_SEQUENCES:-32}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
GPU_ID=${GPU_ID:-5}

run_one_superfamily () {
  OUT_DIR=$1
  SEED=$2
  SUPERFAMILY=$3
  SAFE_NAME=$4

  OUT_JSON=${OUT_DIR}/${SAFE_NAME}.json
  LOG_FILE=${OUT_DIR}/${SAFE_NAME}.log

  echo "[GPU ${GPU_ID}] ${SAFE_NAME}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6 \
  LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-} \
  python -u conditional/generate/generate.py \
    --load_mode lora \
    --model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER}" \
    --superfamily "${SUPERFAMILY}" \
    --num_sequences "${NUM_SEQUENCES}" \
    --generation_batch_size "${GEN_BATCH_SIZE}" \
    --output_json "${OUT_JSON}" \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 200 \
    --repetition_penalty 1.2 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    > "${LOG_FILE}" 2>&1
}

OUT_DIR=${GEN_ROOT}/${METHOD_DIR}
mkdir -p "${OUT_DIR}"

echo "[METHOD] FinalUtility-ShiftEntropy-q02-High50"
echo "[ADAPTER] ${ADAPTER}"
echo "[OUT_DIR] ${OUT_DIR}"

run_one_superfamily "${OUT_DIR}" 42 "Lysozyme-like domain superfamily" "${METHOD_DIR}_lysozyme_like_domain_superfamily"
run_one_superfamily "${OUT_DIR}" 43 "Immunoglobulin-like beta-sandwich superfamily" "${METHOD_DIR}_immunoglobulin_like_beta_sandwich_superfamily"
run_one_superfamily "${OUT_DIR}" 44 "TIM beta/alpha-barrel domain superfamily" "${METHOD_DIR}_tim_beta_alpha_barrel_domain_superfamily"
run_one_superfamily "${OUT_DIR}" 45 "Rossmann-like alpha/beta/alpha sandwich fold superfamily" "${METHOD_DIR}_rossmann_like_alpha_beta_alpha_sandwich_fold_superfamily"

echo "[OK] Final utility generation finished."
