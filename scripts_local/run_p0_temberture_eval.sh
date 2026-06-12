#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd ${PROJECT_DIR}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate temberture

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/zbc/data/models/TemBERTure:/home/zbc/data/models/TemBERTure/temBERTure:${PYTHONPATH:-}

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "Python=$(which python)"
echo "Start TemBERTure evaluation at $(date)"

python -u tools/p0_eval/score_temberture_cls.py \
  --cohort_dir analysis_outputs/p0_current_eval/cohorts \
  --out_dir analysis_outputs/p0_current_eval/property/temberture \
  --temberture_root /home/zbc/data/models/TemBERTure \
  --device cuda \
  --batch_size 1 \
  --max_per_method 256 \
  --max_len 1024

echo "Finished TemBERTure evaluation at $(date)"
