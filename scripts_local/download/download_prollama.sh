#!/usr/bin/env bash
set -euo pipefail

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

mkdir -p /home/zbc/data/models/ProLLaMA

# 如果 Hugging Face 官方源下载慢，可以取消下一行注释使用镜像
# export HF_ENDPOINT=https://hf-mirror.com

hf download GreatCaptainNemo/ProLLaMA \
  --local-dir /home/zbc/data/models/ProLLaMA \
  --max-workers 8

echo "ProLLaMA download finished."
ls -lh /home/zbc/data/models/ProLLaMA
