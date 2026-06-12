#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd ${PROJECT_DIR}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_TORCHVISION=1
export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

BASE_MODEL=/home/zbc/data/models/ProLLaMA

FOLD_TEACHER=/mnt/data/users/zbc/opd/ProteinOPD/conditional_teachers_local/foldability
SOL_TEACHER=/mnt/data/users/zbc/opd/ProteinOPD/conditional_teachers_local/sol
THERMO_TEACHER=/mnt/data/users/zbc/opd/ProteinOPD/conditional_teachers_local/thermo

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "FOLD_TEACHER=${FOLD_TEACHER}"
echo "SOL_TEACHER=${SOL_TEACHER}"
echo "THERMO_TEACHER=${THERMO_TEACHER}"

ls -lh ${FOLD_TEACHER}/adapter_config.json
ls -lh ${SOL_TEACHER}/adapter_config.json
ls -lh ${THERMO_TEACHER}/adapter_config.json

python -u tools/offline_opd/offline_cache_diagnostic.py \
  --rollouts_csv analysis_outputs/p0_current_eval/cohorts/Base-ProLLaMA.csv \
  --base_model ${BASE_MODEL} \
  --teacher foldability:${FOLD_TEACHER} \
  --teacher solubility:${SOL_TEACHER} \
  --teacher thermostability:${THERMO_TEACHER} \
  --out_dir analysis_outputs/offline_proteinopd/cache_diag/base_rollout \
  --top_k 64 \
  --max_samples 128 \
  --max_length 512 \
  --dtype fp16 \
  --device cuda \
  --include_reference
