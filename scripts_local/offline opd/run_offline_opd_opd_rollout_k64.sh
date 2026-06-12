#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd ${PROJECT_DIR}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONUNBUFFERED=1
export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Start Offline OPD OPDRollout K64 at $(date)"
echo "Teacher forward calls during training should be 0."

python -u tools/offline_opd/train_offline_opd.py \
  --base_model /home/zbc/data/models/ProLLaMA \
  --cache_dir analysis_outputs/offline_proteinopd/cache_diag/conditional_opd_rollout \
  --output_dir outputs/offline_proteinopd/opd_rollout_k64_gamma01 \
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

echo "Finished Offline OPD OPDRollout K64 at $(date)"
