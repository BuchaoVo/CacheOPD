#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd "${PROJECT_DIR}"

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_TORCHVISION=1
export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}

MODEL=/home/zbc/data/models/ProLLaMA

declare -A ADAPTERS
ADAPTERS["Contrib2-High50"]="outputs/offline_proteinopd/contrib2_base_rollout_k64_keep50_high_gamma01"
ADAPTERS["Contrib2-Random50"]="outputs/offline_proteinopd/contrib2_base_rollout_k64_keep50_random_gamma01"
ADAPTERS["Contrib2-Low50"]="outputs/offline_proteinopd/contrib2_base_rollout_k64_keep50_low_gamma01"

SUPERFAMILIES=(
  "Lysozyme-like domain superfamily"
  "Immunoglobulin-like beta-sandwich superfamily"
  "TIM beta/alpha-barrel domain superfamily"
  "Rossmann-like alpha/beta/alpha sandwich fold superfamily"
)

safe_name () {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/_\+/_/g' | sed 's/^_//;s/_$//'
}

for METHOD in "${!ADAPTERS[@]}"; do
  ADAPTER_PATH="${ADAPTERS[$METHOD]}"

  if [ ! -f "${ADAPTER_PATH}/adapter_config.json" ]; then
    echo "[SKIP] missing adapter: ${ADAPTER_PATH}"
    continue
  fi

  METHOD_SAFE=$(safe_name "$METHOD")
  OUT_DIR="outputs/conditional/offline_opd_4superfamily/${METHOD_SAFE}"
  mkdir -p "${OUT_DIR}"

  echo "======================================================"
  echo "[RUN] METHOD=${METHOD}"
  echo "[ADAPTER] ${ADAPTER_PATH}"
  echo "======================================================"

  for SF in "${SUPERFAMILIES[@]}"; do
    SF_SAFE=$(safe_name "$SF")
    OUT_JSON="${OUT_DIR}/${METHOD_SAFE}_${SF_SAFE}.json"

    echo "[GENERATE] ${METHOD} | ${SF}"
    echo "[OUTPUT] ${OUT_JSON}"

    python -u conditional/generate/generate.py \
      --model "${MODEL}" \
      --adapter_path "${ADAPTER_PATH}" \
      --load_mode lora \
      --superfamily "${SF}" \
      --num_sequences 32 \
      --generation_batch_size 4 \
      --output_json "${OUT_JSON}" \
      --temperature 0.7 \
      --top_k 200 \
      --top_p 0.9 \
      --repetition_penalty 1.2 \
      --max_new_tokens 512 \
      --do_sample \
      --seed 42
  done
done

echo "[ALL DONE] $(date)"
