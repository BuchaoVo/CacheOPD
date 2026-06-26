#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  conditional/teacher_construct/scripts/run_cacheopd_high_pref_lora_one.sh \
    <attribute> <gpu_id> <train_file> <valid_file> <output_dir> <peft_path> [epochs]

Attributes:
  foldability | solubility | thermostability
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 6 ]]; then
  usage >&2
  exit 2
fi

ATTRIBUTE=$1
GPU_ID=$2
TRAIN_FILE=$3
VALID_FILE=$4
OUTPUT_DIR=$5
PEFT_PATH=$6
EPOCHS="${7:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTEINOPD_ROOT="${PROTEINOPD_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PROLLAMA_MODEL="${PROLLAMA_MODEL:-/home/zbc/data/models/ProLLaMA}"
CONDA_SH="${CONDA_SH:-/home/zbc/data/software/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-proteinopd}"
CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE:-/home/zbc/data/software/miniconda/envs/${CONDA_ENV}}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

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

case "${ATTRIBUTE}" in
  foldability|solubility|thermostability) ;;
  *)
    echo "[ERROR] attribute must be one of: foldability, solubility, thermostability" >&2
    exit 2
    ;;
esac

resolve_path() {
  local value=$1
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${PROTEINOPD_ROOT}/${value}"
  fi
}

TRAIN_FILE="$(resolve_path "${TRAIN_FILE}")"
VALID_FILE="$(resolve_path "${VALID_FILE}")"
OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR}")"
PEFT_PATH="$(resolve_path "${PEFT_PATH}")"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[ERROR] train_file not found: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VALID_FILE}" ]]; then
  echo "[ERROR] valid_file not found: ${VALID_FILE}" >&2
  exit 1
fi
if [[ ! -d "${PEFT_PATH}" ]]; then
  echo "[ERROR] peft_path not found: ${PEFT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${PROLLAMA_MODEL}" ]]; then
  echo "[ERROR] ProLLaMA model directory not found: ${PROLLAMA_MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${SKIP_CONDA_ACTIVATE}" == "1" ]]; then
  export CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE}"
  export PATH="${CONDA_PREFIX}/bin:/home/zbc/data/software/miniconda/bin:${PATH}"
else
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export LD_PRELOAD="${LD_PRELOAD:-/home/zbc/data/software/miniconda/lib/libstdc++.so.6}"
export LD_LIBRARY_PATH="/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${SCRIPT_DIR}"

echo "[INFO] attribute=${ATTRIBUTE}"
echo "[INFO] gpu=${GPU_ID}"
echo "[INFO] train=${TRAIN_FILE}"
echo "[INFO] valid=${VALID_FILE}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] peft_path=${PEFT_PATH}"
echo "[INFO] epochs=${EPOCHS}, lr=${LR}, max_seq_length=${MAX_SEQ_LENGTH}"
echo "[INFO] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[INFO] LD_PRELOAD=${LD_PRELOAD}"

if [[ "${PRECISION}" == "bf16" ]]; then
  PRECISION_FLAG="--bf16"
  TORCH_DTYPE="bfloat16"
elif [[ "${PRECISION}" == "fp16" ]]; then
  PRECISION_FLAG="--fp16"
  TORCH_DTYPE="float16"
else
  echo "[ERROR] PRECISION must be one of: bf16, fp16" >&2
  exit 2
fi

echo "[INFO] precision=${PRECISION_FLAG}, torch_dtype=${TORCH_DTYPE}"

python -u instruction_tune.py \
  --model_name_or_path "${PROLLAMA_MODEL}" \
  --tokenizer_name_or_path "${PROLLAMA_MODEL}" \
  --peft_path "${PEFT_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${VALID_FILE}" \
  --data_cache_dir "${OUTPUT_DIR}/dataset_cache" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  ${PRECISION_FLAG} \
  --torch_dtype "${TORCH_DTYPE}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --trainable q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --load_in_kbits "${LOAD_IN_KBITS}" \
  --save_safetensors False \
  --report_to none \
  --gradient_checkpointing
