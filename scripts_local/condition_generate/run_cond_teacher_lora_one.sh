#!/usr/bin/env bash
set -euo pipefail

GPU_ID=$1
TRAIN_FILE=$2
VALID_FILE=$3
OUTPUT_DIR=$4
EPOCHS=${5:-1}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

mkdir -p "$(dirname "${OUTPUT_DIR}")"

export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_TORCHVISION=1

cd /home/zbc/data/opd/ProteinOPD/conditional/teacher_construct/scripts

python -u instruction_tune.py \
  --model_name_or_path /home/zbc/data/models/ProLLaMA \
  --tokenizer_name_or_path /home/zbc/data/models/ProLLaMA \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${VALID_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --bf16 \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs "${EPOCHS}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps 5 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 3 \
  --max_seq_length 512 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --trainable q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --load_in_kbits 16 \
  --save_safetensors False \
  --report_to none \
  --gradient_checkpointing
