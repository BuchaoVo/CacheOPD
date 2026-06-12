#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd ${PROJECT_DIR}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=2
export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_TORCHVISION=1

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "PWD=$(pwd)"
echo "Python=$(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Start HF ESMFold structure evaluation at $(date)"

python -u tools/p0_eval/score_esmfold_structure_hf.py \
  --cohort_dir analysis_outputs/p0_current_eval/cohorts \
  --out_dir analysis_outputs/p0_current_eval/structure_hf \
  --model_path /home/zbc/data/models/esmfold_v1_hf \
  --max_per_method 256 \
  --max_len 1024 \
  --chunk_size 64

echo "Finished HF ESMFold structure evaluation at $(date)"
