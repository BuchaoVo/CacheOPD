#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_MODEL=/home/zbc/data/models/ProLLaMA
GEN_ROOT=/home/zbc/data/opd/ProteinOPD/outputs/conditional/offline_opd_4superfamily
NUM_SEQUENCES=${NUM_SEQUENCES:-32}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
GPU_ID=${GPU_ID:-5}

declare -A METHOD_TO_ADAPTER=(
  [final_utility_shift_entropy_q02_random50]=/home/zbc/data/opd/ProteinOPD/outputs/offline_proteinopd/final_utility_shift_entropy_q02_random50_gamma01
  [final_utility_shift_entropy_q02_low50]=/home/zbc/data/opd/ProteinOPD/outputs/offline_proteinopd/final_utility_shift_entropy_q02_low50_gamma01
  [final_utility_shift_only_q02_high50]=/home/zbc/data/opd/ProteinOPD/outputs/offline_proteinopd/final_utility_shift_only_q02_high50_gamma01
)

declare -A METHOD_TO_LABEL=(
  [final_utility_shift_entropy_q02_random50]=FinalUtility-ShiftEntropy-q02-Random50
  [final_utility_shift_entropy_q02_low50]=FinalUtility-ShiftEntropy-q02-Low50
  [final_utility_shift_only_q02_high50]=FinalUtility-ShiftOnly-q02-High50
)

run_one_superfamily () {
  local adapter=$1
  local out_dir=$2
  local method_dir=$3
  local seed=$4
  local superfamily=$5
  local safe_name=$6

  local out_json=${out_dir}/${safe_name}.json
  local log_file=${out_dir}/${safe_name}.log

  echo "[GPU ${GPU_ID}] ${method_dir}/${safe_name}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6 \
  LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-} \
  python -u conditional/generate/generate.py \
    --load_mode lora \
    --model "${BASE_MODEL}" \
    --adapter_path "${adapter}" \
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

for method_dir in "${!METHOD_TO_ADAPTER[@]}"; do
  adapter=${METHOD_TO_ADAPTER[$method_dir]}
  out_dir=${GEN_ROOT}/${method_dir}
  mkdir -p "${out_dir}"

  echo "[METHOD] ${METHOD_TO_LABEL[$method_dir]}"
  echo "[ADAPTER] ${adapter}"
  echo "[OUT_DIR] ${out_dir}"

  run_one_superfamily "${adapter}" "${out_dir}" "${method_dir}" 42 "Lysozyme-like domain superfamily" "${method_dir}_lysozyme_like_domain_superfamily"
  run_one_superfamily "${adapter}" "${out_dir}" "${method_dir}" 43 "Immunoglobulin-like beta-sandwich superfamily" "${method_dir}_immunoglobulin_like_beta_sandwich_superfamily"
  run_one_superfamily "${adapter}" "${out_dir}" "${method_dir}" 44 "TIM beta/alpha-barrel domain superfamily" "${method_dir}_tim_beta_alpha_barrel_domain_superfamily"
  run_one_superfamily "${adapter}" "${out_dir}" "${method_dir}" 45 "Rossmann-like alpha/beta/alpha sandwich fold superfamily" "${method_dir}_rossmann_like_alpha_beta_alpha_sandwich_fold_superfamily"
done

echo "[OK] final validation generation finished."
