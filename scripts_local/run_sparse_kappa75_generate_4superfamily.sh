#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_MODEL=/home/zbc/data/models/ProLLaMA
ADAPTER=/home/zbc/data/opd/ProteinOPD/outputs/offline_proteinopd/sensitivity_kappa75_high_gamma01
METHOD_DIR=cacheopd_sparse_kappa75_high
GEN_ROOT=/home/zbc/data/opd/ProteinOPD/outputs/conditional/offline_opd_4superfamily
OUT_DIR=${GEN_ROOT}/${METHOD_DIR}
NUM_SEQUENCES=${NUM_SEQUENCES:-32}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
GPU_ID=${GPU_ID:-5}

mkdir -p "${OUT_DIR}"

run_one_superfamily () {
  local seed=$1
  local superfamily=$2
  local safe_name=$3

  local out_json=${OUT_DIR}/${METHOD_DIR}_${safe_name}.json
  local log_file=${OUT_DIR}/${METHOD_DIR}_${safe_name}.log

  echo "[GPU ${GPU_ID}] ${METHOD_DIR}/${safe_name}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6 \
  LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-} \
  python -u conditional/generate/generate.py \
    --load_mode lora \
    --model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER}" \
    --superfamily "${superfamily}" \
    --num_sequences "${NUM_SEQUENCES}" \
    --generation_batch_size "${GEN_BATCH_SIZE}" \
    --output_json "${out_json}" \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 200 \
    --repetition_penalty 1.2 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${seed}" \
    > "${log_file}" 2>&1
}

echo "[METHOD] CacheOPD-Sparse-k75"
echo "[ADAPTER] ${ADAPTER}"
echo "[OUT_DIR] ${OUT_DIR}"

run_one_superfamily 42 "Lysozyme-like domain superfamily" "lysozyme_like_domain_superfamily"
run_one_superfamily 43 "Immunoglobulin-like beta-sandwich superfamily" "immunoglobulin_like_beta_sandwich_superfamily"
run_one_superfamily 44 "TIM beta/alpha-barrel domain superfamily" "tim_beta_alpha_barrel_domain_superfamily"
run_one_superfamily 45 "Rossmann-like alpha/beta/alpha sandwich fold superfamily" "rossmann_like_alpha_beta_alpha_sandwich_fold_superfamily"

echo "[OK] CacheOPD-Sparse-k75 generation finished."
