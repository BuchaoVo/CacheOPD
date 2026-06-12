#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONUNBUFFERED=1

# 官方脚本是 CUDA_VISIBLE_DEVICES=6,7；这里按你的可用卡改成 2,3
export CUDA_VISIBLE_DEVICES=2,3

export WANDB_PROJECT="prollama_geometric_opd"
# 如果没有 wandb 登录，可以先用 offline，仍保留 report_to wandb
export WANDB_MODE=offline

NPROC_PER_NODE=2
MASTER_PORT=29519

OPD_CONFIG_PATH=/home/zbc/data/opd/ProteinOPD/configs_local/prollama_protein_opd_original_gpu23.yaml
OUTPUT_DIR=/home/zbc/data/opd/ProteinOPD/outputs/prollama_geometric_opd
DEEPSPEED_CONFIG_FILE=/home/zbc/data/opd/ProteinOPD/conditional/teacher_construct/scripts/ds_zero2_no_offload.json

PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-5
LOGGING_STEPS=5
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=20

if python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)"; then
  PRECISION_FLAG="--bf16"
else
  PRECISION_FLAG="--fp16"
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  /home/zbc/data/opd/ProteinOPD/conditional/proteinopd/prollama_opd_train.py \
  --deepspeed "${DEEPSPEED_CONFIG_FILE}" \
  --opd_config_path "${OPD_CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --do_train \
  --seed 42 \
  ${PRECISION_FLAG} \
  --num_train_epochs 1.0 \
  --lr_scheduler_type cosine \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio 0.1 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to wandb \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy no \
  --load_in_kbits 16 \
  --save_safetensors False \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing
