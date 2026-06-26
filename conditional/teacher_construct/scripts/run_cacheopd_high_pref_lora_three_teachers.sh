#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  conditional/teacher_construct/scripts/run_cacheopd_high_pref_lora_three_teachers.sh [epochs]

Fine-tune three ProteinOPD ProLLaMA LoRA teachers from CacheOPD high-preference
instruction data copied under:
  conditional/teacher_construct/scripts/instruction_tuning_dataset/cacheopd_high_pref/

Environment overrides:
  PROTEINOPD_ROOT, PROLLAMA_MODEL, CONDA_SH, CONDA_ENV
  SKIP_CONDA_ACTIVATE, CONDA_PREFIX_OVERRIDE
  GPU_FOLD, GPU_SOL, GPU_THERMO
  TRAIN_SPLIT, VALID_SPLIT, DATA_ROOT, OUTPUT_ROOT, LOG_ROOT
  FOLD_PEFT_PATH, SOL_PEFT_PATH, THERMO_PEFT_PATH
  LR, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, GRAD_ACCUM, MAX_SEQ_LENGTH
  LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LOAD_IN_KBITS
  LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, PRECISION
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EPOCHS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTEINOPD_ROOT="${PROTEINOPD_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PROLLAMA_MODEL="${PROLLAMA_MODEL:-/home/zbc/data/models/ProLLaMA}"
CONDA_SH="${CONDA_SH:-/home/zbc/data/software/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-proteinopd}"
CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE:-/home/zbc/data/software/miniconda/envs/${CONDA_ENV}}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/instruction_tuning_dataset/cacheopd_high_pref}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROTEINOPD_ROOT}/conditional_teachers_high_pref_local}"
LOG_ROOT="${LOG_ROOT:-${PROTEINOPD_ROOT}/logs_nohup/cacheopd_high_pref_lora_teachers}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train_5k}"
VALID_SPLIT="${VALID_SPLIT:-valid}"

GPU_FOLD="${GPU_FOLD:-5}"
GPU_SOL="${GPU_SOL:-6}"
GPU_THERMO="${GPU_THERMO:-7}"

FOLD_PEFT_PATH="${FOLD_PEFT_PATH:-${PROTEINOPD_ROOT}/conditional_teachers_local/foldability}"
SOL_PEFT_PATH="${SOL_PEFT_PATH:-${PROTEINOPD_ROOT}/conditional_teachers_local/sol}"
THERMO_PEFT_PATH="${THERMO_PEFT_PATH:-${PROTEINOPD_ROOT}/conditional_teachers_local/thermo}"

LR="${LR:-5e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-768}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LOAD_IN_KBITS="${LOAD_IN_KBITS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
PRECISION="${PRECISION:-bf16}"

ONE_SCRIPT="${SCRIPT_DIR}/run_cacheopd_high_pref_lora_one.sh"

require_file() {
  local path=$1
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file is missing: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path=$1
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory is missing: ${path}" >&2
    exit 1
  fi
}

fold_train="${DATA_ROOT}/foldability_high_pref/${TRAIN_SPLIT}.jsonl"
fold_valid="${DATA_ROOT}/foldability_high_pref/${VALID_SPLIT}.jsonl"
sol_train="${DATA_ROOT}/solubility_high_pref/${TRAIN_SPLIT}.jsonl"
sol_valid="${DATA_ROOT}/solubility_high_pref/${VALID_SPLIT}.jsonl"
thermo_train="${DATA_ROOT}/thermostability_high_pref/${TRAIN_SPLIT}.jsonl"
thermo_valid="${DATA_ROOT}/thermostability_high_pref/${VALID_SPLIT}.jsonl"

require_file "${fold_train}"
require_file "${fold_valid}"
require_file "${sol_train}"
require_file "${sol_valid}"
require_file "${thermo_train}"
require_file "${thermo_valid}"
require_dir "${FOLD_PEFT_PATH}"
require_dir "${SOL_PEFT_PATH}"
require_dir "${THERMO_PEFT_PATH}"
require_dir "${PROLLAMA_MODEL}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
timestamp="$(date +%Y%m%d_%H%M%S)"

echo "[INFO] ProteinOPD root=${PROTEINOPD_ROOT}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] epochs=${EPOCHS}, lr=${LR}, train_split=${TRAIN_SPLIT}, valid_split=${VALID_SPLIT}"

launch_one() {
  local attribute=$1
  local gpu=$2
  local train_file=$3
  local valid_file=$4
  local output_dir=$5
  local peft_path=$6
  local log_file=$7
  echo "[Launch] ${attribute} on GPU ${gpu}; log=${log_file}"
  env \
    PROTEINOPD_ROOT="${PROTEINOPD_ROOT}" \
    PROLLAMA_MODEL="${PROLLAMA_MODEL}" \
    CONDA_SH="${CONDA_SH}" \
    CONDA_ENV="${CONDA_ENV}" \
    CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE}" \
    SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE}" \
    LR="${LR}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
    LORA_RANK="${LORA_RANK}" \
    LORA_ALPHA="${LORA_ALPHA}" \
    LORA_DROPOUT="${LORA_DROPOUT}" \
    LOAD_IN_KBITS="${LOAD_IN_KBITS}" \
    LOGGING_STEPS="${LOGGING_STEPS}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
    PRECISION="${PRECISION}" \
    bash "${ONE_SCRIPT}" "${attribute}" "${gpu}" "${train_file}" "${valid_file}" "${output_dir}" "${peft_path}" "${EPOCHS}" \
    > "${log_file}" 2>&1 &
}

launch_one foldability "${GPU_FOLD}" "${fold_train}" "${fold_valid}" \
  "${OUTPUT_ROOT}/foldability_high_pref_${TRAIN_SPLIT}" "${FOLD_PEFT_PATH}" \
  "${LOG_ROOT}/foldability_high_pref_${TRAIN_SPLIT}_${timestamp}.log"
PID_FOLD=$!

launch_one solubility "${GPU_SOL}" "${sol_train}" "${sol_valid}" \
  "${OUTPUT_ROOT}/solubility_high_pref_${TRAIN_SPLIT}" "${SOL_PEFT_PATH}" \
  "${LOG_ROOT}/solubility_high_pref_${TRAIN_SPLIT}_${timestamp}.log"
PID_SOL=$!

launch_one thermostability "${GPU_THERMO}" "${thermo_train}" "${thermo_valid}" \
  "${OUTPUT_ROOT}/thermostability_high_pref_${TRAIN_SPLIT}" "${THERMO_PEFT_PATH}" \
  "${LOG_ROOT}/thermostability_high_pref_${TRAIN_SPLIT}_${timestamp}.log"
PID_THERMO=$!

echo "[INFO] foldability PID=${PID_FOLD}"
echo "[INFO] solubility PID=${PID_SOL}"
echo "[INFO] thermostability PID=${PID_THERMO}"

wait "${PID_FOLD}"
echo "[Done] foldability finished"
wait "${PID_SOL}"
echo "[Done] solubility finished"
wait "${PID_THERMO}"
echo "[Done] thermostability finished"

echo "[OK] All CacheOPD high-pref LoRA teachers finished."
find "${OUTPUT_ROOT}" -maxdepth 2 -name adapter_config.json -print
find "${OUTPUT_ROOT}" -maxdepth 2 \( -name "adapter_model*" -o -name "trainer_state.json" \) -print
